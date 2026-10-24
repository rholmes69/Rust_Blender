//! glTF 2.0 import (.glb and .gltf with embedded, data-URI or external buffers/images):
//! meshes (triangles, strips, fans) with UVs, PBR materials and base-colour textures,
//! KHR_lights_punctual lights, perspective cameras and the node hierarchy.
//! Converts glTF's Y-up to Blender-style Z-up (the inverse of `gltf.rs`).

use crate::mesh::PolyMesh;
use crate::scene::{CameraData, Light, LightType, ObjectKind, Scene};
use crate::textures::TexSource;
use base64::Engine;
use glam::{Mat4, Quat, Vec3};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Default)]
pub struct ImportStats {
    pub meshes: usize,
    pub lights: usize,
    pub cameras: usize,
    pub textures: usize,
    pub triangles: usize,
    /// Things that were skipped (animations, skins, unsupported primitives …).
    pub notes: Vec<String>,
}

impl ImportStats {
    pub fn summary(&self) -> String {
        let mut s = format!("{} mesh object(s), {} triangles", self.meshes, self.triangles);
        if self.textures > 0 {
            s += &format!(", {} texture(s)", self.textures);
        }
        if self.lights > 0 {
            s += &format!(", {} light(s)", self.lights);
        }
        if self.cameras > 0 {
            s += &format!(", {} camera(s)", self.cameras);
        }
        if !self.notes.is_empty() {
            s += &format!(" (skipped: {})", self.notes.join("; "));
        }
        s
    }
}

/// Y-up → Z-up: (x, y, z) → (x, -z, y)
fn icv(v: Vec3) -> Vec3 {
    Vec3::new(v.x, -v.z, v.y)
}
fn icq(q: Quat) -> Quat {
    Quat::from_xyzw(q.x, -q.z, q.y, q.w)
}
fn ics(s: Vec3) -> Vec3 {
    Vec3::new(s.x, s.z, s.y)
}
fn srgb(c: f32) -> f32 {
    c.max(0.0).powf(1.0 / 2.2)
}

struct Doc<'a> {
    json: Value,
    buffers: Vec<Vec<u8>>,
    base: Option<&'a Path>,
}

fn read_uri(uri: &str, base: Option<&Path>) -> Result<Vec<u8>, String> {
    if let Some(rest) = uri.strip_prefix("data:") {
        let (_, data) = rest.split_once(";base64,").ok_or("unsupported data URI")?;
        return base64::engine::general_purpose::STANDARD.decode(data).map_err(|e| format!("bad base64: {e}"));
    }
    let decoded = percent_decode(uri);
    let p = match base {
        Some(b) => b.join(&decoded),
        None => decoded.into(),
    };
    std::fs::read(&p).map_err(|e| format!("can't read {}: {e}", p.display()))
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

impl Doc<'_> {
    fn view(&self, i: usize) -> Result<(&[u8], usize), String> {
        let v = &self.json["bufferViews"][i];
        let buf = self.buffers.get(v["buffer"].as_u64().unwrap_or(0) as usize).ok_or("missing buffer")?;
        let off = v["byteOffset"].as_u64().unwrap_or(0) as usize;
        let len = v["byteLength"].as_u64().ok_or("bufferView without byteLength")? as usize;
        let data = buf.get(off..off + len).ok_or("bufferView out of range")?;
        Ok((data, v["byteStride"].as_u64().unwrap_or(0) as usize))
    }

    /// Read an accessor as f32 components (normalized integers are mapped to 0..1 / -1..1).
    fn floats(&self, i: usize) -> Result<(Vec<f32>, usize), String> {
        let a = &self.json["accessors"][i];
        let count = a["count"].as_u64().ok_or("accessor without count")? as usize;
        let ncomp = match a["type"].as_str().unwrap_or("") {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            "MAT4" => 16,
            t => return Err(format!("unsupported accessor type {t}")),
        };
        if a.get("sparse").is_some() {
            return Err("sparse accessors are not supported".into());
        }
        let ct = a["componentType"].as_u64().unwrap_or(5126);
        let csize = match ct {
            5120 | 5121 => 1,
            5122 | 5123 => 2,
            5125 | 5126 => 4,
            _ => return Err(format!("bad componentType {ct}")),
        };
        let norm = a["normalized"].as_bool().unwrap_or(false);
        let mut out = vec![0.0f32; count * ncomp];
        let Some(vi) = a["bufferView"].as_u64() else { return Ok((out, ncomp)) };
        let (data, stride) = self.view(vi as usize)?;
        let aoff = a["byteOffset"].as_u64().unwrap_or(0) as usize;
        let stride = if stride == 0 { csize * ncomp } else { stride };
        for e in 0..count {
            for c in 0..ncomp {
                let o = aoff + e * stride + c * csize;
                let b = data.get(o..o + csize).ok_or("accessor out of range")?;
                out[e * ncomp + c] = match ct {
                    5126 => f32::from_le_bytes(b.try_into().unwrap()),
                    5125 => u32::from_le_bytes(b.try_into().unwrap()) as f32,
                    5123 => {
                        let v = u16::from_le_bytes(b.try_into().unwrap()) as f32;
                        if norm { v / 65535.0 } else { v }
                    }
                    5122 => {
                        let v = i16::from_le_bytes(b.try_into().unwrap()) as f32;
                        if norm { (v / 32767.0).max(-1.0) } else { v }
                    }
                    5121 => {
                        let v = b[0] as f32;
                        if norm { v / 255.0 } else { v }
                    }
                    _ => {
                        let v = b[0] as i8 as f32;
                        if norm { (v / 127.0).max(-1.0) } else { v }
                    }
                };
            }
        }
        Ok((out, ncomp))
    }

    fn indices(&self, i: usize) -> Result<Vec<u32>, String> {
        let a = &self.json["accessors"][i];
        let count = a["count"].as_u64().unwrap_or(0) as usize;
        let ct = a["componentType"].as_u64().unwrap_or(5125);
        let Some(vi) = a["bufferView"].as_u64() else { return Ok(vec![0; count]) };
        let (data, stride) = self.view(vi as usize)?;
        let csize = match ct {
            5121 => 1,
            5123 => 2,
            _ => 4,
        };
        let stride = if stride == 0 { csize } else { stride };
        let aoff = a["byteOffset"].as_u64().unwrap_or(0) as usize;
        (0..count)
            .map(|e| {
                let o = aoff + e * stride;
                let b = data.get(o..o + csize).ok_or("index accessor out of range")?;
                Ok(match csize {
                    1 => b[0] as u32,
                    2 => u16::from_le_bytes(b.try_into().unwrap()) as u32,
                    _ => u32::from_le_bytes(b.try_into().unwrap()),
                })
            })
            .collect()
    }

    fn image_bytes(&self, img: usize) -> Result<Vec<u8>, String> {
        let im = &self.json["images"][img];
        if let Some(v) = im["bufferView"].as_u64() {
            return Ok(self.view(v as usize)?.0.to_vec());
        }
        let uri = im["uri"].as_str().ok_or("image without data")?;
        read_uri(uri, self.base)
    }
}

fn local_matrix(n: &Value) -> Mat4 {
    if let Some(m) = n["matrix"].as_array().filter(|m| m.len() == 16) {
        let v: Vec<f32> = m.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
        return Mat4::from_cols_slice(&v);
    }
    let v3 = |k: &str, d: Vec3| n[k].as_array().filter(|a| a.len() == 3).map(|a| Vec3::new(a[0].as_f64().unwrap_or(0.0) as f32, a[1].as_f64().unwrap_or(0.0) as f32, a[2].as_f64().unwrap_or(0.0) as f32)).unwrap_or(d);
    let r = n["rotation"].as_array().filter(|a| a.len() == 4).map(|a| Quat::from_xyzw(a[0].as_f64().unwrap_or(0.0) as f32, a[1].as_f64().unwrap_or(0.0) as f32, a[2].as_f64().unwrap_or(0.0) as f32, a[3].as_f64().unwrap_or(1.0) as f32).normalize()).unwrap_or(Quat::IDENTITY);
    Mat4::from_scale_rotation_translation(v3("scale", Vec3::ONE), r, v3("translation", Vec3::ZERO))
}

/// Import a .glb / .gltf file into `scene`. Returns stats and the ids of the new objects.
pub fn import_file(scene: &mut Scene, path: &Path) -> Result<(ImportStats, Vec<u64>), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("can't read {}: {e}", path.display()))?;
    let (st, ids) = import_bytes(scene, &bytes, path.parent())?;
    // put everything from the file into a collection named after it
    if let Some(stem) = path.file_stem().map(|s| s.to_string_lossy().to_string()) {
        let cid = scene.collection_by_name(&stem);
        for id in &ids {
            if let Some(o) = scene.get_mut(*id) {
                o.collection = cid;
            }
        }
    }
    Ok((st, ids))
}

pub fn import_bytes(scene: &mut Scene, bytes: &[u8], base: Option<&Path>) -> Result<(ImportStats, Vec<u64>), String> {
    // ---- container ----
    let (json, bin): (Value, Option<Vec<u8>>) = if bytes.starts_with(b"glTF") {
        if bytes.len() < 20 {
            return Err("truncated .glb".into());
        }
        let mut off = 12;
        let (mut json, mut bin) = (None, None);
        while off + 8 <= bytes.len() {
            let len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
            let kind = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
            let chunk = bytes.get(off + 8..off + 8 + len).ok_or("truncated .glb chunk")?;
            match kind {
                0x4E4F_534A => json = Some(serde_json::from_slice::<Value>(chunk).map_err(|e| format!("bad glTF JSON: {e}"))?),
                0x004E_4942 => bin = Some(chunk.to_vec()),
                _ => {}
            }
            off += 8 + len;
        }
        (json.ok_or("no JSON chunk")?, bin)
    } else {
        (serde_json::from_slice(bytes).map_err(|e| format!("not a glTF file: {e}"))?, None)
    };
    if !json["asset"]["version"].as_str().unwrap_or("2.0").starts_with('2') {
        return Err("only glTF 2.0 is supported".into());
    }
    let mut buffers = vec![];
    for (i, b) in json["buffers"].as_array().cloned().unwrap_or_default().iter().enumerate() {
        buffers.push(match b["uri"].as_str() {
            Some(uri) => read_uri(uri, base)?,
            None if i == 0 => bin.clone().ok_or("buffer 0 has no data")?,
            None => vec![],
        });
    }
    let doc = Doc { json, buffers, base };
    let j = &doc.json;
    let mut stats = ImportStats::default();
    let mut new_ids = vec![];

    // ---- textures (lazily, only the ones materials use) ----
    let mut tex_ids: HashMap<usize, u32> = HashMap::new();
    let mut texture = |scene: &mut Scene, stats: &mut ImportStats, ti: usize| -> Option<u32> {
        if let Some(id) = tex_ids.get(&ti) {
            return Some(*id);
        }
        let src = j["textures"][ti]["source"].as_u64()? as usize;
        let data = doc.image_bytes(src).ok()?;
        crate::textures::decode(&TexSource::Image { data: data.clone() }).ok()?;
        let name = j["images"][src]["name"].as_str().filter(|n| !n.is_empty()).map(String::from).unwrap_or_else(|| format!("Image {src}"));
        let id = scene.add_texture(&name, TexSource::Image { data });
        stats.textures += 1;
        tex_ids.insert(ti, id);
        Some(id)
    };

    // ---- materials ----
    let mut materials: Vec<crate::scene::Material> = vec![];
    for m in j["materials"].as_array().cloned().unwrap_or_default() {
        let pbr = &m["pbrMetallicRoughness"];
        let f = pbr["baseColorFactor"].as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(1.0) as f32).collect::<Vec<_>>()).unwrap_or(vec![1.0, 1.0, 1.0, 1.0]);
        let mut mat = crate::scene::Material {
            color: [srgb(f[0]), srgb(f[1]), srgb(f[2])],
            alpha: if m["alphaMode"].as_str() == Some("BLEND") { f.get(3).copied().unwrap_or(1.0) } else { 1.0 },
            metallic: pbr["metallicFactor"].as_f64().unwrap_or(1.0) as f32,
            roughness: pbr["roughnessFactor"].as_f64().unwrap_or(1.0) as f32,
            ..Default::default()
        };
        if let Some(e) = m["emissiveFactor"].as_array() {
            let e: Vec<f32> = e.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
            let strength = m["extensions"]["KHR_materials_emissive_strength"]["emissiveStrength"].as_f64().unwrap_or(1.0) as f32;
            let mx = e.iter().copied().fold(0.0, f32::max);
            if mx > 0.0 {
                mat.emission = mx * strength;
                if f[..3].iter().all(|c| *c > 0.99) {
                    mat.color = [srgb(e[0] / mx), srgb(e[1] / mx), srgb(e[2] / mx)];
                }
            }
        }
        if let Some(ti) = pbr["baseColorTexture"]["index"].as_u64() {
            if pbr["baseColorTexture"]["texCoord"].as_u64().unwrap_or(0) == 0 {
                if let Some(id) = texture(scene, &mut stats, ti as usize) {
                    mat.texture = id;
                    if let Some(s) = pbr["baseColorTexture"]["extensions"]["KHR_texture_transform"]["scale"].as_array() {
                        let s: Vec<f32> = s.iter().map(|x| x.as_f64().unwrap_or(1.0) as f32).collect();
                        mat.tex_scale = (s.iter().sum::<f32>() / s.len().max(1) as f32).max(0.01);
                    }
                }
            } else {
                stats.notes.push("textures on a second UV set".into());
            }
        }
        materials.push(mat);
    }

    // ---- node traversal ----
    let nodes = j["nodes"].as_array().cloned().unwrap_or_default();
    let roots: Vec<usize> = {
        let si = j["scene"].as_u64().unwrap_or(0) as usize;
        match j["scenes"][si]["nodes"].as_array() {
            Some(a) => a.iter().filter_map(|v| v.as_u64().map(|x| x as usize)).collect(),
            None => {
                let mut child = vec![false; nodes.len()];
                for n in &nodes {
                    for c in n["children"].as_array().into_iter().flatten() {
                        if let Some(c) = c.as_u64() {
                            child[c as usize] = true;
                        }
                    }
                }
                (0..nodes.len()).filter(|i| !child[*i]).collect()
            }
        }
    };
    // (node, parent world matrix, depth, nearest ancestor object)
    let mut stack: Vec<(usize, Mat4, usize, Option<u64>)> = roots.into_iter().map(|r| (r, Mat4::IDENTITY, 0, None)).collect();
    let fix_inv = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2).inverse();
    let mut seen = 0usize;
    while let Some((ni, parent, depth, parent_obj)) = stack.pop() {
        let mut node_obj: Option<u64> = None;
        let first_new = new_ids.len();
        seen += 1;
        if depth > 64 || seen > 100_000 {
            return Err("node hierarchy is cyclic or too deep".into());
        }
        let Some(n) = nodes.get(ni) else { continue };
        let world = parent * local_matrix(n);
        let (s, r, t) = world.to_scale_rotation_translation();
        let node_name = n["name"].as_str().filter(|s| !s.is_empty());
        // mesh
        if let Some(mi) = n["mesh"].as_u64() {
            let mesh = &j["meshes"][mi as usize];
            let base_name = node_name.or(mesh["name"].as_str()).unwrap_or("Mesh").to_string();
            for p in mesh["primitives"].as_array().cloned().unwrap_or_default() {
                let mode = p["mode"].as_u64().unwrap_or(4);
                if !matches!(mode, 4..=6) {
                    stats.notes.push("line/point primitives".into());
                    continue;
                }
                let Some(pa) = p["attributes"]["POSITION"].as_u64() else { continue };
                let (pos, _) = doc.floats(pa as usize)?;
                let nverts = pos.len() / 3;
                let uv = p["attributes"]["TEXCOORD_0"].as_u64().map(|a| doc.floats(a as usize)).transpose()?.map(|(v, _)| v);
                let normals = p["attributes"]["NORMAL"].as_u64().map(|a| doc.floats(a as usize)).transpose()?.map(|(v, _)| v);
                let idx: Vec<u32> = match p["indices"].as_u64() {
                    Some(a) => doc.indices(a as usize)?,
                    None => (0..nverts as u32).collect(),
                };
                let tris: Vec<[u32; 3]> = match mode {
                    5 => (2..idx.len()).map(|k| if k % 2 == 0 { [idx[k - 2], idx[k - 1], idx[k]] } else { [idx[k - 1], idx[k - 2], idx[k]] }).collect(),
                    6 => (2..idx.len()).map(|k| [idx[0], idx[k - 1], idx[k]]).collect(),
                    _ => idx.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
                };
                let tris: Vec<[u32; 3]> = tris.into_iter().filter(|t| t.iter().all(|&i| (i as usize) < nverts) && t[0] != t[1] && t[1] != t[2] && t[0] != t[2]).collect();
                if tris.is_empty() {
                    continue;
                }
                let verts: Vec<Vec3> = (0..nverts).map(|i| icv(Vec3::new(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]))).collect();
                let mut pm = PolyMesh {
                    verts,
                    faces: tris.iter().map(|t| t.to_vec()).collect(),
                    uvs: match &uv {
                        Some(uv) => tris.iter().map(|t| t.iter().map(|&i| [uv[i as usize * 2], uv[i as usize * 2 + 1]]).collect()).collect(),
                        None => vec![],
                    },
                };
                // smooth if the file's normals differ from the face normals
                let smooth = normals.as_ref().is_some_and(|nr| {
                    let mut dev = 0.0;
                    for t in tris.iter().take(2000) {
                        let [a, b, c] = t.map(|i| Vec3::new(pos[i as usize * 3], pos[i as usize * 3 + 1], pos[i as usize * 3 + 2]));
                        let fnrm = (b - a).cross(c - a).normalize_or_zero();
                        for &i in t {
                            let vn = Vec3::new(nr[i as usize * 3], nr[i as usize * 3 + 1], nr[i as usize * 3 + 2]).normalize_or_zero();
                            dev += 1.0 - vn.dot(fnrm);
                        }
                    }
                    dev / (tris.len().min(2000) * 3) as f32 > 0.002
                });
                let (lo, hi) = pm.verts.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(l, h), v| (l.min(*v), h.max(*v)));
                pm.weld(((hi - lo).max_element() * 1e-5).max(1e-7));
                merge_quads(&mut pm);
                stats.triangles += tris.len();
                let id = scene.add(crate::mesh::Primitive::Cube, Some(&base_name));
                let o = scene.get_mut(id).unwrap();
                o.mesh = Some(pm);
                o.smooth_override = Some(smooth);
                if smooth {
                    // keep the model's hard edges (glTF splits vertices there; welding merged them)
                    o.auto_smooth = 40.0;
                }
                o.material = p["material"].as_u64().and_then(|m| materials.get(m as usize).copied()).unwrap_or_default();
                o.transform.location = icv(t);
                o.transform.set_quat(icq(r));
                o.transform.scale = ics(s);
                stats.meshes += 1;
                new_ids.push(id);
            }
        }
        // camera
        if let Some(ci) = n["camera"].as_u64() {
            let cam = &j["cameras"][ci as usize];
            let id = scene.add_camera(Some(node_name.or(cam["name"].as_str()).unwrap_or("Camera")));
            let o = scene.get_mut(id).unwrap();
            if let Some(y) = cam["perspective"]["yfov"].as_f64() {
                o.kind = ObjectKind::Camera(CameraData { fov_deg: (y as f32).to_degrees().clamp(1.0, 170.0) });
            }
            o.transform.location = icv(t);
            o.transform.set_quat(icq(r * fix_inv));
            stats.cameras += 1;
            new_ids.push(id);
        }
        // light
        if let Some(li) = n["extensions"]["KHR_lights_punctual"]["light"].as_u64() {
            let l = &j["extensions"]["KHR_lights_punctual"]["lights"][li as usize];
            let ty = match l["type"].as_str().unwrap_or("point") {
                "directional" => LightType::Sun,
                "spot" => LightType::Spot,
                _ => LightType::Point,
            };
            let mut light = Light::new(ty);
            if let Some(c) = l["color"].as_array() {
                light.color = [0, 1, 2].map(|k| c.get(k).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32);
            }
            let inten = l["intensity"].as_f64().unwrap_or(1.0) as f32;
            light.power = if ty == LightType::Sun { inten } else { inten * 4.0 * std::f32::consts::PI };
            if ty == LightType::Spot {
                let outer = l["spot"]["outerConeAngle"].as_f64().unwrap_or(std::f64::consts::FRAC_PI_4) as f32;
                let inner = l["spot"]["innerConeAngle"].as_f64().unwrap_or(0.0) as f32;
                light.spot_angle = (outer * 2.0).to_degrees();
                light.spot_blend = (1.0 - inner / outer.max(1e-4)).clamp(0.0, 1.0);
            }
            let id = scene.add_light(ty, Some(node_name.or(l["name"].as_str()).unwrap_or("Light")));
            let o = scene.get_mut(id).unwrap();
            o.kind = ObjectKind::Light(light);
            o.transform.location = icv(t);
            o.transform.set_quat(icq(r * fix_inv));
            stats.lights += 1;
            new_ids.push(id);
        }
        // hierarchy: this node's objects become children of the nearest ancestor object; extra
        // primitives of the same mesh are parented to the first one
        for (k, id) in new_ids[first_new..].iter().enumerate() {
            let p = if k == 0 { parent_obj } else { node_obj };
            if k == 0 {
                node_obj = Some(*id);
            }
            if let Some(o) = scene.get_mut(*id) {
                o.parent = p;
            }
        }
        let carry = node_obj.or(parent_obj);
        for c in n["children"].as_array().into_iter().flatten().rev() {
            if let Some(c) = c.as_u64() {
                stack.push((c as usize, world, depth + 1, carry));
            }
        }
    }
    if j["animations"].as_array().is_some_and(|a| !a.is_empty()) {
        stats.notes.push("animations".into());
    }
    if j["skins"].as_array().is_some_and(|a| !a.is_empty()) {
        stats.notes.push("skinning (imported in rest pose)".into());
    }
    stats.notes.sort();
    stats.notes.dedup();
    if new_ids.is_empty() {
        return Err("the file contains no meshes, lights or cameras".into());
    }
    Ok((stats, new_ids))
}

/// Merge pairs of coplanar triangles that form a convex quad (nicer editing and wireframes).
pub fn merge_quads(pm: &mut PolyMesh) {
    let has_uv = pm.has_uvs();
    let n = pm.faces.len();
    if n > 400_000 {
        return;
    }
    let normal = |f: &[u32]| -> Vec3 {
        let [a, b, c] = [f[0], f[1], f[2]].map(|i| pm.verts[i as usize]);
        (b - a).cross(c - a).normalize_or_zero()
    };
    let mut edge: HashMap<(u32, u32), usize> = HashMap::new();
    for (fi, f) in pm.faces.iter().enumerate() {
        if f.len() == 3 {
            for k in 0..3 {
                edge.insert((f[k], f[(k + 1) % 3]), fi);
            }
        }
    }
    let mut used = vec![false; n];
    let mut out_faces = vec![];
    let mut out_uvs = vec![];
    for fi in 0..n {
        if used[fi] {
            continue;
        }
        let f = pm.faces[fi].clone();
        let mut merged = false;
        if f.len() == 3 {
            let nf = normal(&f);
            for k in 0..3 {
                let (a, b, c) = (f[k], f[(k + 1) % 3], f[(k + 2) % 3]);
                let Some(&gi) = edge.get(&(b, a)) else { continue };
                if gi == fi || used[gi] || pm.faces[gi].len() != 3 {
                    continue;
                }
                let g = &pm.faces[gi];
                if normal(g).dot(nf) < 0.9999 {
                    continue;
                }
                let gk = g.iter().position(|&x| x == b).unwrap();
                let d = g[(gk + 2) % 3];
                // quad b, c, a, d
                let quad = [b, c, a, d];
                let pts: Vec<Vec3> = quad.iter().map(|&i| pm.verts[i as usize]).collect();
                let convex = (0..4).all(|i| (pts[(i + 1) % 4] - pts[i]).cross(pts[(i + 2) % 4] - pts[(i + 1) % 4]).dot(nf) > 1e-9);
                if !convex {
                    continue;
                }
                if has_uv {
                    let uf = &pm.uvs[fi];
                    let ug = &pm.uvs[gi];
                    let uvf = |v: u32| uf[f.iter().position(|&x| x == v).unwrap()];
                    let uvg = |v: u32| ug[g.iter().position(|&x| x == v).unwrap()];
                    let same = |p: [f32; 2], q: [f32; 2]| (p[0] - q[0]).abs() < 1e-5 && (p[1] - q[1]).abs() < 1e-5;
                    if !same(uvf(a), uvg(a)) || !same(uvf(b), uvg(b)) {
                        continue;
                    }
                    out_uvs.push(vec![uvf(b), uvf(c), uvf(a), uvg(d)]);
                }
                out_faces.push(quad.to_vec());
                used[fi] = true;
                used[gi] = true;
                merged = true;
                break;
            }
        }
        if !merged {
            used[fi] = true;
            out_faces.push(f);
            if has_uv {
                out_uvs.push(pm.uvs[fi].clone());
            }
        }
    }
    pm.faces = out_faces;
    pm.uvs = if has_uv { out_uvs } else { vec![] };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::MeshCache;

    #[test]
    fn round_trip_through_export() {
        // build a scene with every feature, export it, import it into an empty scene
        let mut ed = crate::editor::Editor::default();
        let cmds: Vec<crate::editor::Command> = serde_json::from_value(serde_json::json!([
            {"action": "set_transform", "target": "Cube", "location": [1, 2, 3], "rotation": [10, 20, 30], "scale": [1, 2, 0.5]},
            {"action": "set_texture", "target": "Cube", "name": "bricks", "number": 2},
            {"action": "add", "primitive": "cube", "name": "Holder", "location": [-2, 0, 0], "rotation": [0, 0, 30], "scale": 0.5},
            {"action": "add", "primitive": "uv_sphere", "name": "Ball", "location": [-2, 0, 1], "color": "red", "parent": "Holder"},
            {"action": "add", "primitive": "spot_light", "name": "Spot", "location": [0, -4, 4], "look_at": [0, 0, 0], "power": 800}
        ]))
        .unwrap();
        for l in ed.execute(&cmds) {
            assert!(l.starts_with('✔'), "{l}");
        }
        let (glb, _) = crate::gltf::export_glb(&ed.scene, &mut MeshCache::default());
        let mut scene = Scene::default();
        scene.objects.clear();
        let (st, ids) = import_bytes(&mut scene, &glb, None).unwrap();
        assert_eq!((st.meshes, st.lights, st.cameras, st.textures), (3, 2, 1, 1), "{st:?}");
        assert_eq!(ids.len(), 6);
        let orig = ed.scene.objects.iter().find(|o| o.name == "Cube").unwrap();
        let cube = scene.objects.iter().find(|o| o.name == "Cube").unwrap();
        assert!((cube.transform.location - orig.transform.location).length() < 1e-4);
        assert!((cube.transform.scale - orig.transform.scale).length() < 1e-4);
        assert!(cube.transform.quat().dot(orig.transform.quat()).abs() > 0.9999);
        // the exporter bakes per-corner vertices; welding + quad merging restores the 6 quads
        let pm = cube.mesh.as_ref().unwrap();
        assert_eq!((pm.verts.len(), pm.faces.len()), (8, 6));
        assert!(pm.has_uvs());
        assert_ne!(cube.material.texture, 0);
        // world-space positions match
        let mut cache = MeshCache::default();
        let a = crate::agent::world_bounds(orig, &mut cache);
        let b = crate::agent::world_bounds(cube, &mut cache);
        assert!((a.lo - b.lo).length() < 1e-3 && (a.hi - b.hi).length() < 1e-3);
        let spot_o = ed.scene.objects.iter().find(|o| o.name == "Spot").unwrap();
        let spot_i = scene.objects.iter().find(|o| o.name == "Spot").unwrap();
        assert!(spot_o.forward().dot(spot_i.forward()) > 0.9999);
        assert!((spot_i.light().unwrap().power - 800.0).abs() < 0.1);
        let ball = scene.objects.iter().find(|o| o.name == "Ball").unwrap();
        let ball_o = ed.scene.objects.iter().find(|o| o.name == "Ball").unwrap();
        for k in 0..3 {
            assert!((ball.material.color[k] - ball_o.material.color[k]).abs() < 0.01, "{:?} vs {:?}", ball.material.color, ball_o.material.color);
        }
        assert!((ball.material.roughness - ball_o.material.roughness).abs() < 1e-4);
        assert_eq!(ball.smooth_override, Some(true));
        // Ball was parented to Holder before export: the hierarchy survives
        let holder = scene.objects.iter().find(|o| o.name == "Holder").unwrap();
        assert_eq!(ball.parent, Some(holder.id));
        let wb = crate::agent::world_bounds(ball_o, &mut cache);
        let wb2 = crate::agent::world_bounds(ball, &mut cache);
        assert!((wb.lo - wb2.lo).length() < 1e-3);
        // .rblend round trip keeps textures and UVs
        let js = serde_json::to_string(&scene).unwrap();
        let back: Scene = serde_json::from_str(&js).unwrap();
        assert_eq!(back.textures, scene.textures);
        assert_eq!(back.objects, scene.objects);
    }

    #[test]
    fn gltf_json_with_data_uri() {
        // a single triangle in a .gltf with an embedded buffer, no indices
        let pos: Vec<u8> = [0f32, 0., 0., 1., 0., 0., 0., 1., 0.].iter().flat_map(|f| f.to_le_bytes()).collect();
        let uri = format!("data:application/octet-stream;base64,{}", base64::engine::general_purpose::STANDARD.encode(&pos));
        let js = serde_json::json!({
            "asset": {"version": "2.0"},
            "buffers": [{"uri": uri, "byteLength": 36}],
            "bufferViews": [{"buffer": 0, "byteLength": 36}],
            "accessors": [{"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
            "nodes": [{"mesh": 0, "name": "Tri", "translation": [0, 5, 0]}]
        });
        let mut scene = Scene::default();
        let (st, ids) = import_bytes(&mut scene, js.to_string().as_bytes(), None).unwrap();
        assert_eq!(st.triangles, 1);
        let o = scene.get(ids[0]).unwrap();
        assert_eq!(o.name, "Tri");
        // glTF +Y up becomes +Z up
        assert!((o.transform.location - Vec3::new(0.0, 0.0, 5.0)).length() < 1e-6);
        assert!(import_bytes(&mut scene, b"hello", None).is_err());
    }
}
