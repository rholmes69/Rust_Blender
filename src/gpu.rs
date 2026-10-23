//! GPU viewport renderer (wgpu): rasterizes the same primitive list as the software renderer,
//! with 4x MSAA, a real depth buffer, a 1x ID/depth pass (read back asynchronously for picking)
//! and Blender-style selection outlines computed in a composite shader.

use crate::render::RPrim;
use eframe::egui_wgpu::{self, wgpu};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const MSAA: u32 = 4;
const COLOR_FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const DEPTH_FMT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const FLAGS_W: u32 = 4096;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GpuVertex {
    pos: [f32; 3],
    col: [f32; 4],
    id: u32,
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain-old-data structs/arrays with repr(C) and no padding-sensitive invariants
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Everything needed to render one viewport frame on the GPU.
pub struct FrameData {
    pub version: u64,
    pub w: u32,
    pub h: u32,
    pub verts: Vec<GpuVertex>,
    pub n_opaque: u32,
    /// Selection flag per object index+1 (0 none, 1 selected, 2 active).
    pub flags: Vec<u32>,
    pub outline_width: u32,
    pub selected: [f32; 4],
    pub active: [f32; 4],
}

impl FrameData {
    pub fn new(
        version: u64,
        w: u32,
        h: u32,
        bg_top: [f32; 3],
        bg_bot: [f32; 3],
        prims: &[RPrim],
        sel: &[u8],
        outline_width: u32,
        selected: egui::Color32,
        active: egui::Color32,
    ) -> Self {
        let mut verts = Vec::with_capacity(prims.len() * 6 + 6);
        let (fw, fh) = (w as f32, h as f32);
        // background gradient at the far plane
        let t = [bg_top[0], bg_top[1], bg_top[2], 1.0];
        let b = [bg_bot[0], bg_bot[1], bg_bot[2], 1.0];
        let quad = [
            ([0.0, 0.0], t),
            ([fw, 0.0], t),
            ([fw, fh], b),
            ([0.0, 0.0], t),
            ([fw, fh], b),
            ([0.0, fh], b),
        ];
        for (p, c) in quad {
            verts.push(GpuVertex {
                pos: [p[0], p[1], 1.0],
                col: c,
                id: 0,
            });
        }
        let mut n_opaque = 6;
        for (pass_blend, count) in [(false, true), (true, false)] {
            for p in prims.iter().filter(|p| p.blend == pass_blend) {
                let idx: &[usize] = if p.n == 4 {
                    &[0, 1, 2, 0, 2, 3]
                } else {
                    &[0, 1, 2]
                };
                for &i in idx {
                    let z = if p.no_depth { 0.0 } else { p.p[i][2] };
                    verts.push(GpuVertex {
                        pos: [p.p[i][0], p.p[i][1], z],
                        col: p.c[i],
                        id: p.id,
                    });
                }
            }
            if count {
                n_opaque = verts.len() as u32;
            }
        }
        let c = |c: egui::Color32| {
            [
                c.r() as f32 / 255.0,
                c.g() as f32 / 255.0,
                c.b() as f32 / 255.0,
                1.0,
            ]
        };
        Self {
            version,
            w,
            h,
            verts,
            n_opaque,
            flags: sel.iter().map(|s| *s as u32).collect(),
            outline_width,
            selected: c(selected),
            active: c(active),
        }
    }
}

/// CPU copy of the GPU ID/depth buffers (one frame behind), for picking.
#[derive(Default)]
pub struct Readback {
    pub w: u32,
    pub h: u32,
    pub ids: Vec<u32>,
    pub depth: Vec<f32>,
    pub version: u64,
    pub fresh: bool,
}

#[derive(Clone)]
pub struct GpuHandle {
    pub device: wgpu::Device,
    pub readback: Arc<Mutex<Readback>>,
}

impl GpuHandle {
    /// Install the renderer into egui's callback resources. Returns None if wgpu isn't in use.
    pub fn install(rs: &egui_wgpu::RenderState) -> Option<Self> {
        let readback = Arc::new(Mutex::new(Readback::default()));
        // Catch validation errors (unsupported formats etc.) and fall back to the CPU renderer.
        let scope = rs.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let res = GpuResources::new(&rs.device, rs.target_format, readback.clone());
        if let Some(e) = pollster::block_on(scope.pop()) {
            eprintln!("GPU viewport renderer unavailable, using the software renderer: {e}");
            return None;
        }
        rs.renderer.write().callback_resources.insert(res);
        Some(Self {
            device: rs.device.clone(),
            readback,
        })
    }

    /// Drive pending buffer mappings (non-blocking).
    pub fn poll(&self) {
        let _ = self.device.poll(wgpu::PollType::Poll);
    }
}

struct Targets {
    w: u32,
    h: u32,
    msaa_color: wgpu::TextureView,
    msaa_depth: wgpu::TextureView,
    color: wgpu::TextureView,
    id_tex: wgpu::Texture,
    id: wgpu::TextureView,
    zcol_tex: wgpu::Texture,
    zcol: wgpu::TextureView,
    depth1: wgpu::TextureView,
    staging_id: wgpu::Buffer,
    staging_z: wgpu::Buffer,
    padded_bpr: u32,
    composite_bg: Option<wgpu::BindGroup>,
}

struct GpuResources {
    color_opaque: wgpu::RenderPipeline,
    color_blend: wgpu::RenderPipeline,
    id_opaque: wgpu::RenderPipeline,
    id_blend: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
    _scene_bgl: wgpu::BindGroupLayout,
    composite_bgl: wgpu::BindGroupLayout,
    scene_ubuf: wgpu::Buffer,
    comp_ubuf: wgpu::Buffer,
    scene_bg: wgpu::BindGroup,
    vbuf: Option<(wgpu::Buffer, u64)>,
    flags_tex: Option<(wgpu::Texture, wgpu::TextureView, u32)>,
    targets: Option<Targets>,
    last_version: u64,
    n_verts: u32,
    n_opaque: u32,
    map_pending: Arc<AtomicBool>,
    map_done: Arc<AtomicBool>,
    readback: Arc<Mutex<Readback>>,
    pending_version: u64,
    srgb_target: bool,
}

const SCENE_WGSL: &str = r#"
struct U { size: vec4<f32> };
@group(0) @binding(0) var<uniform> u: U;
struct VIn { @location(0) pos: vec3<f32>, @location(1) col: vec4<f32>, @location(2) id: u32 };
struct VOut { @builtin(position) p: vec4<f32>, @location(0) col: vec4<f32>, @location(1) @interpolate(flat) id: u32 };
@vertex fn vs(v: VIn) -> VOut {
    var o: VOut;
    o.p = vec4<f32>(v.pos.x / u.size.x * 2.0 - 1.0, 1.0 - v.pos.y / u.size.y * 2.0, v.pos.z, 1.0);
    o.col = v.col;
    o.id = v.id;
    return o;
}
@fragment fn fs_color(i: VOut) -> @location(0) vec4<f32> { return i.col; }
struct IdOut { @location(0) id: u32, @location(1) z: u32 };
@fragment fn fs_id_opaque(i: VOut) -> IdOut { var o: IdOut; o.id = i.id; o.z = bitcast<u32>(i.p.z); return o; }
@fragment fn fs_id_blend(i: VOut) -> IdOut {
    if (i.id == 0u || i.col.a < 0.3) { discard; }
    var o: IdOut; o.id = i.id; o.z = bitcast<u32>(i.p.z); return o;
}
"#;

const COMPOSITE_WGSL: &str = r#"
struct CU { size: vec4<f32>, sel_col: vec4<f32>, act_col: vec4<f32>, params: vec4<u32> };
@group(0) @binding(0) var tcol: texture_2d<f32>;
@group(0) @binding(1) var tid: texture_2d<u32>;
@group(0) @binding(2) var tflags: texture_2d<u32>;
@group(0) @binding(3) var<uniform> cu: CU;
struct VO { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i: u32) -> VO {
    var o: VO;
    let x = f32((i << 1u) & 2u);
    let y = f32(i & 2u);
    o.p = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    o.uv = vec2<f32>(x, y);
    return o;
}
fn flag(id: u32) -> u32 {
    if (id >= cu.params.x) { return 0u; }
    return textureLoad(tflags, vec2<i32>(i32(id % 4096u), i32(id / 4096u)), 0).r;
}
fn to_linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}
@fragment fn fs(i: VO) -> @location(0) vec4<f32> {
    let sz = vec2<i32>(cu.size.xy);
    let px = clamp(vec2<i32>(i.uv * cu.size.xy), vec2<i32>(0), sz - vec2<i32>(1));
    var col = textureLoad(tcol, px, 0);
    let me = textureLoad(tid, px, 0).r;
    let r = i32(cu.params.y);
    if (flag(me) == 0u && r > 0) {
        var best = 0u;
        for (var dy = -r; dy <= r; dy = dy + 1) {
            for (var dx = -r; dx <= r; dx = dx + 1) {
                if (dx * dx + dy * dy > r * r + 1) { continue; }
                let q = clamp(px + vec2<i32>(dx, dy), vec2<i32>(0), sz - vec2<i32>(1));
                let nid = textureLoad(tid, q, 0).r;
                if (nid != me) { best = max(best, flag(nid)); }
            }
        }
        if (best == 2u) { col = cu.act_col; } else if (best == 1u) { col = cu.sel_col; }
    }
    if (cu.params.z == 1u) { col = vec4<f32>(to_linear(col.rgb), col.a); }
    return vec4<f32>(col.rgb, 1.0);
}
"#;

impl GpuResources {
    fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        readback: Arc<Mutex<Readback>>,
    ) -> Self {
        let scene_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rb scene"),
            source: wgpu::ShaderSource::Wgsl(SCENE_WGSL.into()),
        });
        let comp_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rb composite"),
            source: wgpu::ShaderSource::Wgsl(COMPOSITE_WGSL.into()),
        });
        let scene_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rb scene bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let tex_entry =
            |binding: u32, sample_type: wgpu::TextureSampleType| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            };
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rb composite bgl"),
            entries: &[
                tex_entry(0, wgpu::TextureSampleType::Float { filterable: false }),
                tex_entry(1, wgpu::TextureSampleType::Uint),
                tex_entry(2, wgpu::TextureSampleType::Uint),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let scene_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rb scene layout"),
            bind_group_layouts: &[Some(&scene_bgl)],
            immediate_size: 0,
        });
        let comp_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rb comp layout"),
            bind_group_layouts: &[Some(&composite_bgl)],
            immediate_size: 0,
        });
        let vattrs = [
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 12,
                shader_location: 1,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32,
                offset: 28,
                shader_location: 2,
            },
        ];
        let vlayout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GpuVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &vattrs,
        };
        let alpha = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let make = |label: &str,
                    fs: &str,
                    targets: &[Option<wgpu::ColorTargetState>],
                    depth_write: bool,
                    samples: u32| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&scene_layout),
                vertex: wgpu::VertexState {
                    module: &scene_sm,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[Some(vlayout.clone())],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FMT,
                    depth_write_enabled: Some(depth_write),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: samples,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &scene_sm,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets,
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let color_opaque = make(
            "rb color opaque",
            "fs_color",
            &[Some(wgpu::ColorTargetState {
                format: COLOR_FMT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            true,
            MSAA,
        );
        let color_blend = make(
            "rb color blend",
            "fs_color",
            &[Some(wgpu::ColorTargetState {
                format: COLOR_FMT,
                blend: Some(alpha),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            false,
            MSAA,
        );
        let id_t = |zmask: wgpu::ColorWrites| {
            [
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Uint,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Uint,
                    blend: None,
                    write_mask: zmask,
                }),
            ]
        };
        let id_opaque = make(
            "rb id opaque",
            "fs_id_opaque",
            &id_t(wgpu::ColorWrites::ALL),
            true,
            1,
        );
        let id_blend = make(
            "rb id blend",
            "fs_id_blend",
            &id_t(wgpu::ColorWrites::empty()),
            false,
            1,
        );
        let composite = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rb composite"),
            layout: Some(&comp_layout),
            vertex: wgpu::VertexState {
                module: &comp_sm,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &comp_sm,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        let scene_ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rb scene ubuf"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let comp_ubuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rb comp ubuf"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let scene_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rb scene bg"),
            layout: &scene_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: scene_ubuf.as_entire_binding(),
            }],
        });
        Self {
            color_opaque,
            color_blend,
            id_opaque,
            id_blend,
            composite,
            _scene_bgl: scene_bgl,
            composite_bgl,
            scene_ubuf,
            comp_ubuf,
            scene_bg,
            vbuf: None,
            flags_tex: None,
            targets: None,
            last_version: u64::MAX,
            n_verts: 0,
            n_opaque: 0,
            map_pending: Arc::new(AtomicBool::new(false)),
            map_done: Arc::new(AtomicBool::new(false)),
            readback,
            pending_version: 0,
            srgb_target: target_format.is_srgb(),
        }
    }

    fn ensure_targets(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        if self.targets.as_ref().is_some_and(|t| t.w == w && t.h == h) {
            return;
        }
        let size = wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        };
        let tex =
            |label: &str, format: wgpu::TextureFormat, samples: u32, usage: wgpu::TextureUsages| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size,
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
            };
        let ra = wgpu::TextureUsages::RENDER_ATTACHMENT;
        let msaa_color = tex("rb msaa color", COLOR_FMT, MSAA, ra).create_view(&Default::default());
        let msaa_depth = tex("rb msaa depth", DEPTH_FMT, MSAA, ra).create_view(&Default::default());
        let color = tex(
            "rb color",
            COLOR_FMT,
            1,
            ra | wgpu::TextureUsages::TEXTURE_BINDING,
        )
        .create_view(&Default::default());
        let id_tex = tex(
            "rb id",
            wgpu::TextureFormat::R32Uint,
            1,
            ra | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        );
        let id = id_tex.create_view(&Default::default());
        // NDC depth stored as raw f32 bits (R32Float isn't renderable on every backend)
        let zcol_tex = tex(
            "rb zcol",
            wgpu::TextureFormat::R32Uint,
            1,
            ra | wgpu::TextureUsages::COPY_SRC,
        );
        let zcol = zcol_tex.create_view(&Default::default());
        let depth1 = tex("rb depth1", DEPTH_FMT, 1, ra).create_view(&Default::default());
        let padded_bpr = (w * 4).div_ceil(256) * 256;
        let sb = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (padded_bpr * h) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };
        let staging_id = sb("rb staging id");
        let staging_z = sb("rb staging z");
        // A size change while a mapping is in flight would orphan it; reset the state.
        self.map_pending.store(false, Ordering::SeqCst);
        self.map_done.store(false, Ordering::SeqCst);
        self.targets = Some(Targets {
            w,
            h,
            msaa_color,
            msaa_depth,
            color,
            id_tex,
            id,
            zcol_tex,
            zcol,
            depth1,
            staging_id,
            staging_z,
            padded_bpr,
            composite_bg: None,
        });
    }

    fn ensure_flags(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, flags: &[u32]) -> bool {
        let n = flags.len().max(1) as u32;
        let rows = n.div_ceil(FLAGS_W);
        let mut recreated = false;
        if self.flags_tex.as_ref().is_none_or(|(_, _, r)| *r < rows) {
            let t = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("rb flags"),
                size: wgpu::Extent3d {
                    width: FLAGS_W,
                    height: rows,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Uint,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let v = t.create_view(&Default::default());
            self.flags_tex = Some((t, v, rows));
            recreated = true;
        }
        let mut data = vec![0u32; (rows * FLAGS_W) as usize];
        data[..flags.len()].copy_from_slice(flags);
        let (t, _, _) = self.flags_tex.as_ref().unwrap();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: t,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            as_bytes(&data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(FLAGS_W * 4),
                rows_per_image: Some(rows),
            },
            wgpu::Extent3d {
                width: FLAGS_W,
                height: rows,
                depth_or_array_layers: 1,
            },
        );
        recreated
    }

    /// Copy a finished mapping into the shared readback.
    fn collect_readback(&mut self) {
        if !self.map_done.load(Ordering::SeqCst) {
            return;
        }
        let Some(t) = &self.targets else { return };
        {
            let (w, h, bpr) = (t.w as usize, t.h as usize, t.padded_bpr as usize);
            let (Ok(ids_m), Ok(z_m)) = (
                t.staging_id.slice(..).get_mapped_range(),
                t.staging_z.slice(..).get_mapped_range(),
            ) else {
                return;
            };
            let mut rb = self.readback.lock().unwrap();
            rb.w = t.w;
            rb.h = t.h;
            rb.ids.resize(w * h, 0);
            rb.depth.resize(w * h, 1.0);
            for y in 0..h {
                for x in 0..w {
                    let o = y * bpr + x * 4;
                    rb.ids[y * w + x] = u32::from_le_bytes(ids_m[o..o + 4].try_into().unwrap());
                    rb.depth[y * w + x] =
                        f32::from_bits(u32::from_le_bytes(z_m[o..o + 4].try_into().unwrap()));
                }
            }
            rb.version = self.pending_version;
            rb.fresh = true;
        }
        t.staging_id.unmap();
        t.staging_z.unmap();
        self.map_done.store(false, Ordering::SeqCst);
        self.map_pending.store(false, Ordering::SeqCst);
    }
}

pub struct ViewportCallback {
    pub frame: Arc<FrameData>,
}

impl egui_wgpu::CallbackTrait for ViewportCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _sd: &egui_wgpu::ScreenDescriptor,
        _enc: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(r) = resources.get_mut::<GpuResources>() else {
            return vec![];
        };
        r.collect_readback();
        let f = &self.frame;
        if f.w == 0 || f.h == 0 {
            return vec![];
        }
        let size_changed = r.targets.as_ref().is_none_or(|t| t.w != f.w || t.h != f.h);
        let need_render = size_changed || r.last_version != f.version;
        r.ensure_targets(device, f.w, f.h);
        let flags_recreated = r.ensure_flags(device, queue, &f.flags);
        // composite uniforms (cheap, every frame)
        let mut cu = [0u8; 64];
        let put_f = |b: &mut [u8], o: usize, v: f32| b[o..o + 4].copy_from_slice(&v.to_le_bytes());
        put_f(&mut cu, 0, f.w as f32);
        put_f(&mut cu, 4, f.h as f32);
        for i in 0..4 {
            put_f(&mut cu, 16 + i * 4, f.selected[i]);
            put_f(&mut cu, 32 + i * 4, f.active[i]);
        }
        cu[48..52].copy_from_slice(&(f.flags.len() as u32).to_le_bytes());
        cu[52..56].copy_from_slice(&f.outline_width.to_le_bytes());
        cu[56..60].copy_from_slice(&u32::from(r.srgb_target).to_le_bytes());
        queue.write_buffer(&r.comp_ubuf, 0, &cu);
        if size_changed
            || flags_recreated
            || r.targets.as_ref().is_some_and(|t| t.composite_bg.is_none())
        {
            let t = r.targets.as_mut().unwrap();
            let (_, fv, _) = r.flags_tex.as_ref().unwrap();
            t.composite_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rb composite bg"),
                layout: &r.composite_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&t.color),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&t.id),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(fv),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: r.comp_ubuf.as_entire_binding(),
                    },
                ],
            }));
        }
        if !need_render {
            return vec![];
        }
        // upload geometry
        let bytes = as_bytes(&f.verts);
        let cap = bytes.len().max(64) as u64;
        if r.vbuf.as_ref().is_none_or(|(_, c)| *c < cap) {
            let c = cap.next_power_of_two();
            r.vbuf = Some((
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("rb vbuf"),
                    size: c,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                c,
            ));
        }
        queue.write_buffer(&r.vbuf.as_ref().unwrap().0, 0, bytes);
        let mut su = [0u8; 16];
        put_f(&mut su, 0, f.w as f32);
        put_f(&mut su, 4, f.h as f32);
        queue.write_buffer(&r.scene_ubuf, 0, &su);
        r.n_verts = f.verts.len() as u32;
        r.n_opaque = f.n_opaque;
        r.last_version = f.version;

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rb viewport"),
        });
        let t = r.targets.as_ref().unwrap();
        let vb = &r.vbuf.as_ref().unwrap().0;
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rb color pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &t.msaa_color,
                    depth_slice: None,
                    resolve_target: Some(&t.color),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.msaa_depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &r.scene_bg, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_pipeline(&r.color_opaque);
            pass.draw(0..r.n_opaque, 0..1);
            if r.n_verts > r.n_opaque {
                pass.set_pipeline(&r.color_blend);
                pass.draw(r.n_opaque..r.n_verts, 0..1);
            }
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("rb id pass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &t.id,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &t.zcol,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: f32::to_bits(1.0) as f64,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &t.depth1,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &r.scene_bg, &[]);
            pass.set_vertex_buffer(0, vb.slice(..));
            pass.set_pipeline(&r.id_opaque);
            pass.draw(0..r.n_opaque, 0..1);
            if r.n_verts > r.n_opaque {
                pass.set_pipeline(&r.id_blend);
                pass.draw(r.n_opaque..r.n_verts, 0..1);
            }
        }
        let can_copy = !r.map_pending.load(Ordering::SeqCst);
        if can_copy {
            let layout = wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(t.padded_bpr),
                rows_per_image: Some(t.h),
            };
            let ext = wgpu::Extent3d {
                width: t.w,
                height: t.h,
                depth_or_array_layers: 1,
            };
            for (tex, buf) in [(&t.id_tex, &t.staging_id), (&t.zcol_tex, &t.staging_z)] {
                enc.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: buf,
                        layout,
                    },
                    ext,
                );
            }
        }
        // Submit ourselves so the staging buffers can be mapped right after.
        queue.submit([enc.finish()]);
        if can_copy {
            r.map_pending.store(true, Ordering::SeqCst);
            r.pending_version = f.version;
            let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            for buf in [&t.staging_id, &t.staging_z] {
                let (count, done) = (count.clone(), r.map_done.clone());
                buf.slice(..).map_async(wgpu::MapMode::Read, move |res| {
                    if res.is_ok() && count.fetch_add(1, Ordering::SeqCst) == 1 {
                        done.store(true, Ordering::SeqCst);
                    }
                });
            }
        }
        vec![]
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        rp: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(r) = resources.get::<GpuResources>() else {
            return;
        };
        let Some(t) = &r.targets else { return };
        let Some(bg) = &t.composite_bg else { return };
        rp.set_pipeline(&r.composite);
        rp.set_bind_group(0, bg, &[]);
        rp.draw(0..3, 0..1);
    }
}
