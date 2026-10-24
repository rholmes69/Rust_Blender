//! Binary glTF 2.0 (.glb) export: meshes (modifiers applied), PBR materials, lights
//! (KHR_lights_punctual), cameras and baked keyframe animation.
//! Converts Blender-style Z-up to glTF Y-up exactly like Blender's own exporter (+Y up, -Z forward).

use crate::scene::{LightType, MeshCache, ObjectKind, Scene};
use glam::{Mat3, Quat, Vec3};
use serde_json::{Value, json};

/// Z-up -> Y-up: (x, y, z) -> (x, z, -y)
fn cv(v: Vec3) -> Vec3 {
    Vec3::new(v.x, v.z, -v.y)
}
fn cq(q: Quat) -> Quat {
    Quat::from_xyzw(q.x, q.z, -q.y, q.w)
}
fn cs(s: Vec3) -> Vec3 {
    Vec3::new(s.x, s.z, s.y)
}
fn lin(c: f32) -> f32 {
    c.max(0.0).powf(2.2)
}

#[derive(Default)]
struct Bin {
    data: Vec<u8>,
    views: Vec<Value>,
    accessors: Vec<Value>,
}

impl Bin {
    fn align(&mut self) {
        while !self.data.len().is_multiple_of(4) {
            self.data.push(0);
        }
    }
    fn view(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        self.align();
        let off = self.data.len();
        self.data.extend_from_slice(bytes);
        let mut v = json!({"buffer": 0, "byteOffset": off, "byteLength": bytes.len()});
        if let Some(t) = target {
            v["target"] = json!(t);
        }
        self.views.push(v);
        self.views.len() - 1
    }
    fn vec3s(&mut self, v: &[Vec3], target: Option<u32>) -> usize {
        let mut b = Vec::with_capacity(v.len() * 12);
        let (mut lo, mut hi) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for p in v {
            lo = lo.min(*p);
            hi = hi.max(*p);
            for c in [p.x, p.y, p.z] {
                b.extend_from_slice(&c.to_le_bytes());
            }
        }
        let view = self.view(&b, target);
        self.accessors.push(json!({"bufferView": view, "componentType": 5126, "count": v.len(), "type": "VEC3", "min": [lo.x, lo.y, lo.z], "max": [hi.x, hi.y, hi.z]}));
        self.accessors.len() - 1
    }
    fn vec4s(&mut self, v: &[[f32; 4]]) -> usize {
        let mut b = Vec::with_capacity(v.len() * 16);
        for q in v {
            for c in q {
                b.extend_from_slice(&c.to_le_bytes());
            }
        }
        let view = self.view(&b, None);
        self.accessors.push(
            json!({"bufferView": view, "componentType": 5126, "count": v.len(), "type": "VEC4"}),
        );
        self.accessors.len() - 1
    }
    fn scalars(&mut self, v: &[f32]) -> usize {
        let mut b = Vec::with_capacity(v.len() * 4);
        for c in v {
            b.extend_from_slice(&c.to_le_bytes());
        }
        let view = self.view(&b, None);
        let lo = v.iter().copied().fold(f32::MAX, f32::min);
        let hi = v.iter().copied().fold(f32::MIN, f32::max);
        self.accessors.push(json!({"bufferView": view, "componentType": 5126, "count": v.len(), "type": "SCALAR", "min": [lo], "max": [hi]}));
        self.accessors.len() - 1
    }
    fn vec2s(&mut self, v: &[[f32; 2]]) -> usize {
        let mut b = Vec::with_capacity(v.len() * 8);
        for q in v {
            for c in q {
                b.extend_from_slice(&c.to_le_bytes());
            }
        }
        let view = self.view(&b, Some(34962));
        self.accessors.push(json!({"bufferView": view, "componentType": 5126, "count": v.len(), "type": "VEC2"}));
        self.accessors.len() - 1
    }
    fn indices(&mut self, v: &[u32]) -> usize {
        let mut b = Vec::with_capacity(v.len() * 4);
        for i in v {
            b.extend_from_slice(&i.to_le_bytes());
        }
        let view = self.view(&b, Some(34963));
        self.accessors.push(
            json!({"bufferView": view, "componentType": 5125, "count": v.len(), "type": "SCALAR"}),
        );
        self.accessors.len() - 1
    }
}

/// Image file bytes + MIME type for a texture (procedural ones are rendered to PNG).
fn texture_bytes(t: &crate::textures::Texture) -> Option<(Vec<u8>, &'static str)> {
    match &t.source {
        crate::textures::TexSource::Image { data } => {
            let mime = if data.starts_with(&[0x89, b'P', b'N', b'G']) { "image/png" } else { "image/jpeg" };
            Some((data.clone(), mime))
        }
        src @ crate::textures::TexSource::Procedural { .. } => {
            let img = crate::textures::decode(src).ok()?;
            let raw: Vec<u8> = img.rgba.iter().flatten().copied().collect();
            Some((crate::textures::encode_png(img.w, img.h, &raw).ok()?, "image/png"))
        }
    }
}

pub struct ExportStats {
    pub meshes: usize,
    pub lights: usize,
    pub cameras: usize,
    pub animations: usize,
    pub bytes: usize,
}

pub fn export_glb(scene: &Scene, cache: &mut MeshCache) -> (Vec<u8>, ExportStats) {
    let mut bin = Bin::default();
    let mut nodes = vec![];
    let mut meshes = vec![];
    let mut materials = vec![];
    let mut lights = vec![];
    let mut cameras = vec![];
    let mut channels = vec![];
    let mut samplers = vec![];
    let mut uses_emissive_strength = false;
    // textures: scene texture id -> glTF texture index
    let mut images = vec![];
    let mut gl_textures = vec![];
    let mut tex_map: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    // glTF lights/cameras look down -Z with +Y up; ours look down local -Z with +Y up in Z-up
    // space. After the axis conversion that axis becomes -Y, so add a -90° X rotation.
    let fix = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);

    for o in scene.objects.iter().filter(|o| o.visible) {
        let t = &o.transform;
        let mut node = json!({"name": o.name});
        let rot = |q: Quat, special: bool| {
            let r = cq(q);
            if special { r * fix } else { r }
        };
        let special = !o.is_mesh();
        let r = rot(t.quat(), special);
        let tr = cv(t.location);
        node["translation"] = json!([tr.x, tr.y, tr.z]);
        node["rotation"] = json!([r.x, r.y, r.z, r.w]);
        if !special {
            let s = cs(t.scale);
            node["scale"] = json!([s.x, s.y, s.z]);
        }
        match &o.kind {
            ObjectKind::Mesh => {
                let mesh = cache.for_object(o);
                let (mut pos, mut nrm, mut idx) = (vec![], vec![], vec![]);
                let mut uvs: Vec<[f32; 2]> = vec![];
                let tex = scene.texture(o.material.texture);
                for inst in o.instance_matrices() {
                    let nm = Mat3::from_mat4(inst).inverse().transpose();
                    let flip = inst.determinant() < 0.0;
                    let base = pos.len() as u32;
                    if tex.is_some() {
                        // per-corner vertices so UV seams survive
                        let s = o.material.tex_scale;
                        for (ti, tri) in mesh.tris.iter().enumerate() {
                            let order: [usize; 3] = if flip { [0, 2, 1] } else { [0, 1, 2] };
                            let [a, b, c] = tri.map(|i| inst.transform_point3(mesh.positions[i as usize]));
                            let fnrm = (b - a).cross(c - a).normalize_or_zero() * if flip { -1.0 } else { 1.0 };
                            let k = pos.len() as u32;
                            for &q in &order {
                                let vi = tri[q] as usize;
                                pos.push(cv(inst.transform_point3(mesh.positions[vi])));
                                let n = if mesh.smooth { (nm * mesh.normals[vi]).normalize_or_zero() } else { fnrm };
                                nrm.push(cv(n));
                                let uv = mesh.tri_uv.get(ti).map(|t| t[q]).unwrap_or([0.0; 2]);
                                uvs.push([uv[0] * s, uv[1] * s]);
                            }
                            idx.extend([k, k + 1, k + 2]);
                        }
                    } else if mesh.smooth {
                        for (p, n) in mesh.positions.iter().zip(&mesh.normals) {
                            pos.push(cv(inst.transform_point3(*p)));
                            nrm.push(cv((nm * *n).normalize_or_zero()));
                        }
                        for tri in &mesh.tris {
                            let tri = if flip { [tri[0], tri[2], tri[1]] } else { *tri };
                            idx.extend(tri.iter().map(|i| base + i));
                        }
                    } else {
                        for tri in &mesh.tris {
                            let tri = if flip { [tri[0], tri[2], tri[1]] } else { *tri };
                            let [a, b, c] =
                                tri.map(|i| inst.transform_point3(mesh.positions[i as usize]));
                            let n = cv((b - a).cross(c - a).normalize_or_zero());
                            let k = pos.len() as u32;
                            for p in [a, b, c] {
                                pos.push(cv(p));
                                nrm.push(n);
                            }
                            idx.extend([k, k + 1, k + 2]);
                        }
                    }
                }
                if idx.is_empty() {
                    continue;
                }
                let pa = bin.vec3s(&pos, Some(34962));
                let na = bin.vec3s(&nrm, Some(34962));
                let ua = (!uvs.is_empty()).then(|| bin.vec2s(&uvs));
                let ia = bin.indices(&idx);
                let m = o.material;
                let mut mat = json!({
                    "name": format!("{} Material", o.name),
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [lin(m.color[0]), lin(m.color[1]), lin(m.color[2]), m.alpha],
                        "metallicFactor": m.metallic,
                        "roughnessFactor": m.roughness
                    },
                    "doubleSided": false
                });
                if m.emission > 0.0 {
                    mat["emissiveFactor"] =
                        json!([lin(m.color[0]), lin(m.color[1]), lin(m.color[2])]);
                    if m.emission != 1.0 {
                        mat["extensions"] = json!({"KHR_materials_emissive_strength": {"emissiveStrength": m.emission}});
                        uses_emissive_strength = true;
                    }
                }
                if m.alpha < 0.999 {
                    mat["alphaMode"] = json!("BLEND");
                }
                if let (Some(t), Some(_)) = (tex, ua) {
                    let ti = match tex_map.get(&t.id) {
                        Some(i) => Some(*i),
                        None => texture_bytes(t).map(|(bytes, mime)| {
                            let view = bin.view(&bytes, None);
                            images.push(json!({"name": t.name, "bufferView": view, "mimeType": mime}));
                            gl_textures.push(json!({"sampler": 0, "source": images.len() - 1}));
                            tex_map.insert(t.id, gl_textures.len() - 1);
                            gl_textures.len() - 1
                        }),
                    };
                    if let Some(ti) = ti {
                        mat["pbrMetallicRoughness"]["baseColorTexture"] = json!({"index": ti});
                    }
                }
                materials.push(mat);
                let mut attrs = json!({"POSITION": pa, "NORMAL": na});
                if let Some(ua) = ua {
                    attrs["TEXCOORD_0"] = json!(ua);
                }
                meshes.push(json!({"name": o.name, "primitives": [{"attributes": attrs, "indices": ia, "material": materials.len() - 1, "mode": 4}]}));
                node["mesh"] = json!(meshes.len() - 1);
            }
            ObjectKind::Light(l) => {
                let c = [l.color[0], l.color[1], l.color[2]];
                let mut lj = match l.light_type {
                    LightType::Sun => json!({"type": "directional", "intensity": l.power}),
                    LightType::Point => {
                        json!({"type": "point", "intensity": l.power / (4.0 * std::f32::consts::PI)})
                    }
                    LightType::Spot => {
                        let outer = (l.spot_angle * 0.5)
                            .to_radians()
                            .min(std::f32::consts::FRAC_PI_2);
                        json!({"type": "spot", "intensity": l.power / (4.0 * std::f32::consts::PI), "spot": {"innerConeAngle": outer * (1.0 - l.spot_blend.clamp(0.0, 1.0)), "outerConeAngle": outer}})
                    }
                };
                lj["name"] = json!(o.name);
                lj["color"] = json!(c);
                lights.push(lj);
                node["extensions"] = json!({"KHR_lights_punctual": {"light": lights.len() - 1}});
            }
            ObjectKind::Camera(cd) => {
                cameras.push(json!({"name": o.name, "type": "perspective", "perspective": {"yfov": cd.fov_deg.to_radians(), "aspectRatio": 16.0 / 9.0, "znear": 0.1, "zfar": 1000.0}}));
                node["camera"] = json!(cameras.len() - 1);
            }
        }
        nodes.push(node);
        let ni = nodes.len() - 1;

        // baked animation (one sample per frame, preserving the eased interpolation)
        if o.keyframes.len() >= 2 {
            let f0 = o.keyframes.first().unwrap().frame;
            let f1 = o.keyframes.last().unwrap().frame;
            let fps = scene.fps.max(1) as f32;
            let mut times = vec![];
            let (mut tl, mut rl, mut sl) = (vec![], vec![], vec![]);
            for f in f0..=f1 {
                let tf = crate::scene::sample_keys(&o.keyframes, f).unwrap_or(*t);
                times.push((f - f0) as f32 / fps);
                tl.push(cv(tf.location));
                let q = rot(tf.quat(), special);
                rl.push([q.x, q.y, q.z, q.w]);
                sl.push(cs(tf.scale));
            }
            let input = bin.scalars(&times);
            let ta = bin.vec3s(&tl, None);
            let ra = bin.vec4s(&rl);
            let sa = bin.vec3s(&sl, None);
            for (path, out) in [("translation", ta), ("rotation", ra), ("scale", sa)] {
                if path == "scale" && special {
                    continue;
                }
                samplers.push(json!({"input": input, "output": out, "interpolation": "LINEAR"}));
                channels.push(
                    json!({"sampler": samplers.len() - 1, "target": {"node": ni, "path": path}}),
                );
            }
        }
    }

    let n_anim = usize::from(!channels.is_empty());
    let mut ext_used = vec![];
    if !lights.is_empty() {
        ext_used.push("KHR_lights_punctual");
    }
    if uses_emissive_strength {
        ext_used.push("KHR_materials_emissive_strength");
    }
    bin.align();
    let mut root = json!({
        "asset": {"version": "2.0", "generator": "Rust Blender"},
        "scene": 0,
        "scenes": [{"name": "Scene", "nodes": (0..nodes.len()).collect::<Vec<_>>()}],
        "nodes": nodes,
        "buffers": [{"byteLength": bin.data.len()}],
        "bufferViews": bin.views,
        "accessors": bin.accessors,
    });
    let (n_mesh, n_light, n_cam) = (meshes.len(), lights.len(), cameras.len());
    if !meshes.is_empty() {
        root["meshes"] = json!(meshes);
        root["materials"] = json!(materials);
    }
    if !images.is_empty() {
        root["images"] = json!(images);
        root["textures"] = json!(gl_textures);
        root["samplers"] = json!([{"magFilter": 9729, "minFilter": 9729, "wrapS": 10497, "wrapT": 10497}]);
    }
    if !cameras.is_empty() {
        root["cameras"] = json!(cameras);
    }
    if !lights.is_empty() {
        root["extensions"] = json!({"KHR_lights_punctual": {"lights": lights}});
    }
    if !ext_used.is_empty() {
        root["extensionsUsed"] = json!(ext_used);
    }
    if !channels.is_empty() {
        root["animations"] =
            json!([{"name": "Action", "channels": channels, "samplers": samplers}]);
    }
    let mut js = serde_json::to_vec(&root).unwrap();
    while !js.len().is_multiple_of(4) {
        js.push(b' ');
    }
    let total = 12 + 8 + js.len() + 8 + bin.data.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(&0x4654_6C67u32.to_le_bytes()); // "glTF"
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(js.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x4E4F_534Au32.to_le_bytes()); // "JSON"
    out.extend_from_slice(&js);
    out.extend_from_slice(&(bin.data.len() as u32).to_le_bytes());
    out.extend_from_slice(&0x004E_4942u32.to_le_bytes()); // "BIN\0"
    out.extend_from_slice(&bin.data);
    let bytes = out.len();
    (
        out,
        ExportStats {
            meshes: n_mesh,
            lights: n_light,
            cameras: n_cam,
            animations: n_anim,
            bytes,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn glb_structure() {
        let mut s = Scene::default();
        s.objects[0].insert_keyframe(1);
        s.objects[0].transform.location.z = 3.0;
        s.objects[0].insert_keyframe(24);
        let mut cache = MeshCache::default();
        let (glb, st) = export_glb(&s, &mut cache);
        assert_eq!(&glb[0..4], b"glTF");
        assert_eq!(
            u32::from_le_bytes(glb[8..12].try_into().unwrap()) as usize,
            glb.len()
        );
        assert_eq!(
            (st.meshes, st.lights, st.cameras, st.animations),
            (1, 1, 1, 1)
        );
        let jlen = u32::from_le_bytes(glb[12..16].try_into().unwrap()) as usize;
        let j: Value = serde_json::from_slice(&glb[20..20 + jlen]).unwrap();
        // Z-up -> Y-up: camera at (7.36,-6.93,4.96) -> (7.36,4.96,6.93)
        let cam = j["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n.get("camera").is_some())
            .unwrap();
        let t = &cam["translation"];
        assert!(
            (t[1].as_f64().unwrap() - 4.96).abs() < 1e-3
                && (t[2].as_f64().unwrap() - 6.93).abs() < 1e-3
        );
        // camera looks at the origin: rotate glTF -Z by its rotation
        let r = &cam["rotation"];
        let q = Quat::from_xyzw(
            r[0].as_f64().unwrap() as f32,
            r[1].as_f64().unwrap() as f32,
            r[2].as_f64().unwrap() as f32,
            r[3].as_f64().unwrap() as f32,
        );
        let dir = q * -Vec3::Z;
        let want = (-Vec3::new(7.36, 4.96, 6.93)).normalize();
        assert!((dir - want).length() < 1e-3, "{dir} {want}");
        assert!(j["animations"][0]["channels"].as_array().unwrap().len() == 3);
    }
}
