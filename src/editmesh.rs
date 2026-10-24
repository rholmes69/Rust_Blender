//! Edit-mode mesh operators working on `PolyMesh` + a per-vertex selection.
//! Selection model (like Blender's "flush"): an edge is selected when both its vertices are,
//! a face when all of its vertices are.

use crate::mesh::PolyMesh;
use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectMode {
    Vertex,
    Edge,
    Face,
}

impl SelectMode {
    pub fn label(self) -> &'static str {
        match self {
            SelectMode::Vertex => "Vertex",
            SelectMode::Edge => "Edge",
            SelectMode::Face => "Face",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct EditState {
    pub obj: u64,
    pub mode: SelectMode,
    pub sel: Vec<bool>,
}

impl EditState {
    pub fn fit(&mut self, pm: &PolyMesh) {
        self.sel.resize(pm.verts.len(), false);
    }
    pub fn count(&self) -> usize {
        self.sel.iter().filter(|s| **s).count()
    }
}

pub fn face_selected(pm: &PolyMesh, sel: &[bool], f: usize) -> bool {
    pm.faces[f]
        .iter()
        .all(|&v| sel.get(v as usize).copied().unwrap_or(false))
}

pub fn selected_faces(pm: &PolyMesh, sel: &[bool]) -> Vec<usize> {
    (0..pm.faces.len())
        .filter(|&f| face_selected(pm, sel, f))
        .collect()
}

pub fn selected_edges(pm: &PolyMesh, sel: &[bool]) -> Vec<(u32, u32)> {
    pm.edges()
        .into_iter()
        .filter(|(a, b)| sel[*a as usize] && sel[*b as usize])
        .collect()
}

pub fn selection_center(pm: &PolyMesh, sel: &[bool]) -> Option<Vec3> {
    let pts: Vec<Vec3> = pm
        .verts
        .iter()
        .zip(sel)
        .filter(|(_, s)| **s)
        .map(|(v, _)| *v)
        .collect();
    (!pts.is_empty()).then(|| pts.iter().sum::<Vec3>() / pts.len() as f32)
}

/// Cleanup that keeps the selection aligned with surviving vertices.
fn cleanup_keep_sel(pm: &mut PolyMesh, sel: &mut Vec<bool>) {
    // tag selection on a parallel position list, then re-map by identity of index
    let n = pm.verts.len();
    sel.resize(n, false);
    let mut used = vec![false; n];
    for f in &mut pm.faces {
        f.dedup();
        while f.len() > 1 && f.first() == f.last() {
            f.pop();
        }
    }
    pm.faces.retain(|f| f.len() >= 3);
    for f in &pm.faces {
        for &i in f {
            used[i as usize] = true;
        }
    }
    let mut remap = vec![u32::MAX; n];
    let mut nv = vec![];
    let mut ns = vec![];
    for i in 0..n {
        if used[i] {
            remap[i] = nv.len() as u32;
            nv.push(pm.verts[i]);
            ns.push(sel[i]);
        }
    }
    pm.verts = nv;
    for f in &mut pm.faces {
        for i in f.iter_mut() {
            *i = remap[*i as usize];
        }
    }
    *sel = ns;
}

/// Extrude the selected faces as a region (or selected edges into new quads).
/// The new geometry becomes the selection. Returns the average normal (for "extrude along normal").
pub fn extrude(pm: &mut PolyMesh, sel: &mut Vec<bool>) -> Result<Vec3, String> {
    pm.uvs.clear(); // topology changes: fall back to generated UVs
    sel.resize(pm.verts.len(), false);
    let faces = selected_faces(pm, sel);
    if !faces.is_empty() {
        let fset: HashSet<usize> = faces.iter().copied().collect();
        let normal = faces
            .iter()
            .map(|&f| pm.face_normal(f))
            .sum::<Vec3>()
            .normalize_or_zero();
        // count edge usage within the region
        let mut count: HashMap<(u32, u32), usize> = HashMap::new();
        for &f in &faces {
            let fv = &pm.faces[f];
            for i in 0..fv.len() {
                let (a, b) = (fv[i], fv[(i + 1) % fv.len()]);
                *count.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        let mut dup: HashMap<u32, u32> = HashMap::new();
        for &f in &faces {
            for &v in &pm.faces[f] {
                dup.entry(v).or_insert_with(|| {
                    pm.verts.push(pm.verts[v as usize]);
                    (pm.verts.len() - 1) as u32
                });
            }
        }
        let mut sides = vec![];
        for &f in &faces {
            let fv = pm.faces[f].clone();
            for i in 0..fv.len() {
                let (a, b) = (fv[i], fv[(i + 1) % fv.len()]);
                if count[&(a.min(b), a.max(b))] == 1 {
                    sides.push(vec![a, b, dup[&b], dup[&a]]);
                }
            }
        }
        for (fi, f) in pm.faces.iter_mut().enumerate() {
            if fset.contains(&fi) {
                for v in f.iter_mut() {
                    *v = dup[v];
                }
            }
        }
        pm.faces.extend(sides);
        let mut ns = vec![false; pm.verts.len()];
        for nv in dup.values() {
            ns[*nv as usize] = true;
        }
        *sel = ns;
        cleanup_keep_sel(pm, sel);
        return Ok(normal);
    }
    let edges = selected_edges(pm, sel);
    if edges.is_empty() {
        return Err("select faces or edges to extrude".into());
    }
    // edge extrude: orient new quads consistently with the face that owns each edge
    let ef = pm.edge_faces();
    let mut dup: HashMap<u32, u32> = HashMap::new();
    let mut new_faces = vec![];
    let mut normal = Vec3::ZERO;
    for (a, b) in edges {
        let (mut x, mut y) = (a, b);
        if let Some(fs) = ef.get(&(a.min(b), a.max(b))) {
            if let Some(&f) = fs.first() {
                normal += pm.face_normal(f);
                let fv = &pm.faces[f];
                for i in 0..fv.len() {
                    if fv[i] == a && fv[(i + 1) % fv.len()] == b {
                        // face uses a->b, so the new face must use b->a
                        x = b;
                        y = a;
                    }
                }
            }
        }
        let dx = *dup.entry(x).or_insert_with(|| {
            pm.verts.push(pm.verts[x as usize]);
            (pm.verts.len() - 1) as u32
        });
        let dy = *dup.entry(y).or_insert_with(|| {
            pm.verts.push(pm.verts[y as usize]);
            (pm.verts.len() - 1) as u32
        });
        new_faces.push(vec![x, y, dy, dx]);
    }
    pm.faces.extend(new_faces);
    let mut ns = vec![false; pm.verts.len()];
    for nv in dup.values() {
        ns[*nv as usize] = true;
    }
    *sel = ns;
    Ok(normal.normalize_or_zero())
}

/// Inset each selected face individually by `t` (0..1, fraction towards the face centre).
pub fn inset(pm: &mut PolyMesh, sel: &mut Vec<bool>, t: f32) -> Result<usize, String> {
    pm.uvs.clear(); // topology changes: fall back to generated UVs
    sel.resize(pm.verts.len(), false);
    let faces = selected_faces(pm, sel);
    if faces.is_empty() {
        return Err("select faces to inset".into());
    }
    let t = t.clamp(0.0, 0.99);
    let mut ns_new = vec![];
    let mut ring = vec![];
    for &f in &faces {
        let fv = pm.faces[f].clone();
        let c = pm.face_center(f);
        let inner: Vec<u32> = fv
            .iter()
            .map(|&v| {
                let p = pm.verts[v as usize];
                pm.verts.push(p + (c - p) * t);
                (pm.verts.len() - 1) as u32
            })
            .collect();
        for i in 0..fv.len() {
            let j = (i + 1) % fv.len();
            ring.push(vec![fv[i], fv[j], inner[j], inner[i]]);
        }
        ns_new.extend(inner.iter().copied());
        pm.faces[f] = inner;
    }
    pm.faces.extend(ring);
    let mut ns = vec![false; pm.verts.len()];
    for v in ns_new {
        ns[v as usize] = true;
    }
    *sel = ns;
    Ok(faces.len())
}

/// Edge ring starting at `edge`: the quads it passes through and the ring edges (canonical keys).
pub fn edge_ring(pm: &PolyMesh, edge: (u32, u32)) -> (Vec<usize>, Vec<(u32, u32)>) {
    let ef = pm.edge_faces();
    let key = |a: u32, b: u32| (a.min(b), a.max(b));
    let start = key(edge.0, edge.1);
    if !ef.contains_key(&start) {
        return (vec![], vec![]);
    }
    let mut faces: Vec<usize> = vec![];
    let mut fseen: HashSet<usize> = HashSet::new();
    let mut edges: Vec<(u32, u32)> = vec![start];
    let mut eseen: HashSet<(u32, u32)> = HashSet::from([start]);
    for &first_face in ef[&start].iter() {
        let mut e = start;
        let mut f = first_face;
        loop {
            if fseen.contains(&f) || pm.faces[f].len() != 4 {
                break;
            }
            fseen.insert(f);
            faces.push(f);
            let q = &pm.faces[f];
            let k = (0..4).find(|&i| key(q[i], q[(i + 1) % 4]) == e).unwrap();
            let opp = key(q[(k + 2) % 4], q[(k + 3) % 4]);
            if !eseen.insert(opp) {
                break;
            }
            edges.push(opp);
            match ef[&opp].iter().find(|&&g| g != f) {
                Some(&g) => {
                    e = opp;
                    f = g;
                }
                None => break,
            }
        }
    }
    (faces, edges)
}

/// Insert `cuts` edge loops through the ring of `edge`. The new loop vertices become the selection.
pub fn loop_cut(
    pm: &mut PolyMesh,
    sel: &mut Vec<bool>,
    edge: (u32, u32),
    cuts: usize,
) -> Result<usize, String> {
    pm.uvs.clear(); // topology changes: fall back to generated UVs
    let cuts = cuts.clamp(1, 32);
    let (ring_faces, ring_edges) = edge_ring(pm, edge);
    if ring_faces.is_empty() {
        return Err("loop cut needs quads around the edge".into());
    }
    let key = |a: u32, b: u32| (a.min(b), a.max(b));
    // split points, stored from min -> max vertex
    let mut pts: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for &(a, b) in &ring_edges {
        let (pa, pb) = (pm.verts[a as usize], pm.verts[b as usize]);
        let ids = (1..=cuts)
            .map(|j| {
                pm.verts.push(pa.lerp(pb, j as f32 / (cuts + 1) as f32));
                (pm.verts.len() - 1) as u32
            })
            .collect();
        pts.insert((a, b), ids);
    }
    let from = |pts: &HashMap<(u32, u32), Vec<u32>>, x: u32, y: u32| -> Vec<u32> {
        let k = key(x, y);
        let v = &pts[&k];
        if x == k.0 {
            v.clone()
        } else {
            v.iter().rev().copied().collect()
        }
    };
    let rset: HashSet<usize> = ring_faces.iter().copied().collect();
    let mut new_faces = vec![];
    for (fi, f) in pm.faces.iter().enumerate() {
        if rset.contains(&fi) {
            let q = f.clone();
            let k = (0..4)
                .find(|&i| {
                    pts.contains_key(&key(q[i], q[(i + 1) % 4]))
                        && pts.contains_key(&key(q[(i + 2) % 4], q[(i + 3) % 4]))
                })
                .unwrap();
            let (a, b, c, d) = (q[k], q[(k + 1) % 4], q[(k + 2) % 4], q[(k + 3) % 4]);
            let p1 = from(&pts, a, b); // along a->b
            let p3 = from(&pts, d, c); // along d->c
            for j in 0..=cuts {
                let left = if j == 0 {
                    (a, d)
                } else {
                    (p1[j - 1], p3[j - 1])
                };
                let right = if j == cuts { (b, c) } else { (p1[j], p3[j]) };
                new_faces.push(vec![left.0, right.0, right.1, left.1]);
            }
        } else {
            // insert split points into neighbouring n-gons so no holes appear
            let mut out = vec![];
            for i in 0..f.len() {
                let (x, y) = (f[i], f[(i + 1) % f.len()]);
                out.push(x);
                if pts.contains_key(&key(x, y)) {
                    out.extend(from(&pts, x, y));
                }
            }
            new_faces.push(out);
        }
    }
    pm.faces = new_faces;
    let mut ns = vec![false; pm.verts.len()];
    for v in pts.values().flatten() {
        ns[*v as usize] = true;
    }
    *sel = ns;
    Ok(ring_faces.len())
}

/// Delete selected geometry. Vertex mode removes faces touching any selected vertex.
pub fn delete(pm: &mut PolyMesh, sel: &mut Vec<bool>, mode: SelectMode) -> usize {
    pm.uvs.clear(); // topology changes: fall back to generated UVs
    sel.resize(pm.verts.len(), false);
    let before = pm.faces.len();
    let s = sel.clone();
    pm.faces.retain(|f| match mode {
        SelectMode::Vertex => !f.iter().any(|&v| s[v as usize]),
        _ => !f.iter().all(|&v| s[v as usize]),
    });
    let removed = before - pm.faces.len();
    for x in sel.iter_mut() {
        *x = false;
    }
    cleanup_keep_sel(pm, sel);
    removed
}

/// Merge all selected vertices at their centre.
pub fn merge_center(pm: &mut PolyMesh, sel: &mut Vec<bool>) -> Result<usize, String> {
    pm.uvs.clear(); // topology changes: fall back to generated UVs
    sel.resize(pm.verts.len(), false);
    let ids: Vec<u32> = (0..pm.verts.len() as u32)
        .filter(|&i| sel[i as usize])
        .collect();
    if ids.len() < 2 {
        return Err("select at least two vertices to merge".into());
    }
    let c = ids.iter().map(|&i| pm.verts[i as usize]).sum::<Vec3>() / ids.len() as f32;
    let keep = ids[0];
    pm.verts[keep as usize] = c;
    let set: HashSet<u32> = ids.iter().copied().collect();
    for f in &mut pm.faces {
        for v in f.iter_mut() {
            if set.contains(v) {
                *v = keep;
            }
        }
    }
    cleanup_keep_sel(pm, sel);
    Ok(ids.len())
}

/// Subdivide selected faces: each n-gon becomes n quads (centre + edge midpoints).
pub fn subdivide(pm: &mut PolyMesh, sel: &mut Vec<bool>) -> Result<usize, String> {
    pm.uvs.clear(); // topology changes: fall back to generated UVs
    sel.resize(pm.verts.len(), false);
    let faces = selected_faces(pm, sel);
    if faces.is_empty() {
        return Err("select faces to subdivide".into());
    }
    let fset: HashSet<usize> = faces.iter().copied().collect();
    let key = |a: u32, b: u32| (a.min(b), a.max(b));
    let mut mids: HashMap<(u32, u32), u32> = HashMap::new();
    for &f in &faces {
        let fv = pm.faces[f].clone();
        for i in 0..fv.len() {
            let k = key(fv[i], fv[(i + 1) % fv.len()]);
            if let std::collections::hash_map::Entry::Vacant(e) = mids.entry(k) {
                pm.verts
                    .push((pm.verts[k.0 as usize] + pm.verts[k.1 as usize]) * 0.5);
                e.insert((pm.verts.len() - 1) as u32);
            }
        }
    }
    let mut out = vec![];
    let mut new_sel = vec![];
    for (fi, f) in pm.faces.clone().iter().enumerate() {
        if fset.contains(&fi) {
            pm.verts.push(pm.face_center(fi));
            let c = (pm.verts.len() - 1) as u32;
            new_sel.push(c);
            let n = f.len();
            for i in 0..n {
                let prev = mids[&key(f[(i + n - 1) % n], f[i])];
                let next = mids[&key(f[i], f[(i + 1) % n])];
                out.push(vec![f[i], next, c, prev]);
                new_sel.extend([f[i], next, prev]);
            }
        } else {
            let mut nf = vec![];
            for i in 0..f.len() {
                let (x, y) = (f[i], f[(i + 1) % f.len()]);
                nf.push(x);
                if let Some(m) = mids.get(&key(x, y)) {
                    nf.push(*m);
                }
            }
            out.push(nf);
        }
    }
    pm.faces = out;
    let mut ns = vec![false; pm.verts.len()];
    for v in new_sel {
        ns[v as usize] = true;
    }
    *sel = ns;
    Ok(faces.len())
}

/// Reverse the winding of selected faces (flip normals).
pub fn flip_normals(pm: &mut PolyMesh, sel: &[bool]) -> usize {
    pm.uvs.clear(); // topology changes: fall back to generated UVs
    let faces = selected_faces(pm, sel);
    for &f in &faces {
        pm.faces[f].reverse();
    }
    faces.len()
}

/// Select faces whose normal points along `dir` (dot > 0.7), e.g. [0,0,1] = "top faces".
pub fn select_by_normal(pm: &PolyMesh, dir: Vec3, extend: bool, sel: &mut Vec<bool>) -> usize {
    sel.resize(pm.verts.len(), false);
    if !extend {
        sel.iter_mut().for_each(|s| *s = false);
    }
    let d = dir.normalize_or_zero();
    let mut n = 0;
    for f in 0..pm.faces.len() {
        if pm.face_normal(f).dot(d) > 0.7 {
            n += 1;
            for &v in &pm.faces[f] {
                sel[v as usize] = true;
            }
        }
    }
    n
}

/// An edge whose direction best matches `axis` (for AI-driven loop cuts).
pub fn edge_along(pm: &PolyMesh, axis: Vec3) -> Option<(u32, u32)> {
    let a = axis.normalize_or_zero();
    pm.edges().into_iter().max_by(|x, y| {
        let dx = (pm.verts[x.1 as usize] - pm.verts[x.0 as usize])
            .normalize_or_zero()
            .dot(a)
            .abs();
        let dy = (pm.verts[y.1 as usize] - pm.verts[y.0 as usize])
            .normalize_or_zero()
            .dot(a)
            .abs();
        dx.partial_cmp(&dy).unwrap_or(std::cmp::Ordering::Equal)
    })
}

/// Bevel the selected edges (both vertices selected, shared by two faces).
/// `offset` is the width along each adjacent face; `segments` > 1 rounds the profile.
/// Returns the number of beveled edges; the new bevel faces end up selected.
pub fn bevel(pm: &mut PolyMesh, sel: &mut Vec<bool>, offset: f32, segments: usize) -> Result<usize, String> {
    pm.uvs.clear(); // topology changes: fall back to generated UVs
    sel.resize(pm.verts.len(), false);
    let key = |a: u32, b: u32| (a.min(b), a.max(b));
    let ef = pm.edge_faces();
    let bev: HashSet<(u32, u32)> = selected_edges(pm, sel)
        .into_iter()
        .map(|(a, b)| key(a, b))
        .filter(|k| ef.get(k).is_some_and(|f| f.len() == 2))
        .collect();
    if bev.is_empty() {
        return Err("select edges to bevel (edges shared by two faces)".into());
    }
    let offset = offset.max(1e-4);
    let segs = segments.clamp(1, 16);
    let bev_verts: HashSet<u32> = bev.iter().flat_map(|&(a, b)| [a, b]).collect();
    let orig = pm.verts.clone();
    let mut verts = pm.verts.clone();
    let new_vert = |verts: &mut Vec<Vec3>, p: Vec3| -> u32 {
        verts.push(p);
        (verts.len() - 1) as u32
    };
    // distance we may move along edge v→w without crossing the other end's bevel
    let reach = |v: u32, w: u32| -> f32 {
        let len = (orig[w as usize] - orig[v as usize]).length();
        let cap = if bev_verts.contains(&w) { 0.49 } else { 0.95 };
        offset.min(len * cap)
    };
    let dir = |v: u32, w: u32| (orig[w as usize] - orig[v as usize]).normalize_or_zero();

    // 1. new corner points for every face corner at a bevel vertex
    let mut on_edge: HashMap<(u32, u32), u32> = HashMap::new();
    let mut corner: HashMap<(usize, u32), Vec<u32>> = HashMap::new();
    let mut bisector: HashSet<u32> = HashSet::new();
    for (fi, f) in pm.faces.iter().enumerate() {
        let n = f.len();
        for i in 0..n {
            let v = f[i];
            if !bev_verts.contains(&v) {
                continue;
            }
            let (prev, next) = (f[(i + n - 1) % n], f[(i + 1) % n]);
            let ein = bev.contains(&key(prev, v));
            let eout = bev.contains(&key(v, next));
            let mut on = |w: u32, verts: &mut Vec<Vec3>| -> u32 {
                *on_edge.entry((v, w)).or_insert_with(|| {
                    let p = orig[v as usize] + dir(v, w) * reach(v, w);
                    new_vert(verts, p)
                })
            };
            let pts = match (ein, eout) {
                (false, false) => vec![on(prev, &mut verts), on(next, &mut verts)],
                (true, false) => vec![on(next, &mut verts)],
                (false, true) => vec![on(prev, &mut verts)],
                (true, true) => {
                    let p = orig[v as usize] + dir(v, prev) * reach(v, prev) + dir(v, next) * reach(v, next);
                    let id = new_vert(&mut verts, p);
                    bisector.insert(id);
                    vec![id]
                }
            };
            corner.insert((fi, v), pts);
        }
    }

    // 2. profile arcs at each end of each beveled edge (quadratic Bézier through the old corner)
    let mut arcs: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    let mut strips: Vec<Vec<u32>> = vec![];
    for &(a0, b0) in &bev {
        let fs = &ef[&(a0, b0)];
        // F1 = the face where the edge runs a→b
        let runs = |fi: usize, a: u32, b: u32| {
            let f = &pm.faces[fi];
            (0..f.len()).any(|i| f[i] == a && f[(i + 1) % f.len()] == b)
        };
        let (f1, f2) = if runs(fs[0], a0, b0) { (fs[0], fs[1]) } else { (fs[1], fs[0]) };
        let mut profile = |v: u32, w: u32, verts: &mut Vec<Vec3>| -> Vec<u32> {
            let p1 = corner[&(f1, v)][0];
            let p2 = corner[&(f2, v)][0];
            // the profile's corner: on the beveled edge, as far in as the corner points are
            let inward = (bisector.contains(&p1) as u8 + bisector.contains(&p2) as u8) as f32 * 0.5;
            let ctrl = orig[v as usize] + dir(v, w) * reach(v, w) * inward;
            let mut out = vec![p1];
            if segs > 1 {
                let interior: Vec<u32> = arcs.get(&(p1, p2)).cloned().unwrap_or_else(|| {
                    let (q1, c, q2) = (verts[p1 as usize], ctrl, verts[p2 as usize]);
                    // quarter-circle-like arc around the point opposite the profile corner
                    let k0 = q1 + q2 - c;
                    let pts: Vec<u32> = (1..segs)
                        .map(|k| {
                            let th = std::f32::consts::FRAC_PI_2 * k as f32 / segs as f32;
                            let p = k0 + (q1 - k0) * th.cos() + (q2 - k0) * th.sin();
                            new_vert(verts, p)
                        })
                        .collect();
                    arcs.insert((p1, p2), pts.clone());
                    arcs.insert((p2, p1), pts.iter().rev().copied().collect());
                    pts
                });
                out.extend(interior);
            }
            out.push(p2);
            out
        };
        let pa = profile(a0, b0, &mut verts);
        let pb = profile(b0, a0, &mut verts);
        for k in 0..segs {
            strips.push(vec![pb[k], pa[k], pa[k + 1], pb[k + 1]]);
        }
    }

    // insert arc interiors between consecutive points of a loop
    let expand = |lp: &[u32]| -> Vec<u32> {
        let mut out = vec![];
        for i in 0..lp.len() {
            let (p, q) = (lp[i], lp[(i + 1) % lp.len()]);
            out.push(p);
            if let Some(mid) = arcs.get(&(p, q)) {
                out.extend(mid);
            }
        }
        out
    };

    // 3. rebuild the original faces
    let mut faces: Vec<Vec<u32>> = vec![];
    for (fi, f) in pm.faces.iter().enumerate() {
        let mut lp = vec![];
        for &v in f {
            match corner.get(&(fi, v)) {
                Some(pts) => lp.extend(pts),
                None => lp.push(v),
            }
        }
        faces.push(expand(&lp));
    }

    // 4. patch the hole at vertices where several bevels meet
    let mut patches = vec![];
    for &v in &bev_verts {
        // corners around v: (face, prev, next)
        let around: Vec<(usize, u32, u32)> = pm
            .faces
            .iter()
            .enumerate()
            .filter_map(|(fi, f)| {
                let i = f.iter().position(|&x| x == v)?;
                let n = f.len();
                Some((fi, f[(i + n - 1) % n], f[(i + 1) % n]))
            })
            .collect();
        // fan order: the face after (prev, v, next) is the one whose prev is our next
        let mut order = vec![around[0]];
        while order.len() < around.len() {
            let last = order[order.len() - 1];
            match around.iter().find(|c| c.1 == last.2) {
                Some(c) if c.0 != order[0].0 => order.push(*c),
                _ => break,
            }
        }
        if order.len() != around.len() || order.last().map(|c| c.2) != Some(order[0].1) {
            continue; // open fan (mesh boundary): leave it
        }
        let mut lp: Vec<u32> = vec![];
        for c in &order {
            for &p in &corner[&(c.0, v)] {
                if lp.last() != Some(&p) {
                    lp.push(p);
                }
            }
        }
        while lp.len() > 1 && lp.first() == lp.last() {
            lp.pop();
        }
        if lp.len() < 3 {
            continue;
        }
        let mut patch = expand(&lp);
        // orient like the surrounding surface
        let pn = newell(&verts, &patch);
        let vn: Vec3 = order.iter().map(|c| pm.face_normal(c.0)).sum();
        if pn.dot(vn) < 0.0 {
            patch.reverse();
        }
        if patch.len() <= 4 {
            patches.push(patch);
        } else {
            // rounded corner: fan around a centre pulled towards the old corner (sphere-like)
            // sphere centre: in from the corner along every beveled edge
            let vp = orig[v as usize];
            let sc = vp + order.iter().filter(|c| bev.contains(&key(v, c.2))).map(|c| dir(v, c.2) * reach(v, c.2)).sum::<Vec3>();
            let r = patch.iter().map(|&i| verts[i as usize].distance(sc)).sum::<f32>() / patch.len() as f32;
            let c = new_vert(&mut verts, sc + (vp - sc).normalize_or_zero() * r);
            for i in 0..patch.len() {
                patches.push(vec![patch[i], patch[(i + 1) % patch.len()], c]);
            }
        }
    }

    let nb = bev.len();
    let strip_verts: HashSet<u32> = strips.iter().flatten().copied().collect();
    faces.extend(strips);
    faces.extend(patches);
    pm.verts = verts;
    pm.faces = faces;
    let mut ns = vec![false; pm.verts.len()];
    for v in strip_verts {
        ns[v as usize] = true;
    }
    // drop the old corner vertices (keeping the selection aligned)
    let mut used = vec![false; pm.verts.len()];
    for f in &pm.faces {
        for &i in f {
            used[i as usize] = true;
        }
    }
    let mut remap = vec![u32::MAX; pm.verts.len()];
    let (mut nv, mut nsel) = (vec![], vec![]);
    for i in 0..pm.verts.len() {
        if used[i] {
            remap[i] = nv.len() as u32;
            nv.push(pm.verts[i]);
            nsel.push(ns[i]);
        }
    }
    pm.verts = nv;
    for f in &mut pm.faces {
        for i in f.iter_mut() {
            *i = remap[*i as usize];
        }
    }
    *sel = nsel;
    Ok(nb)
}

fn newell(verts: &[Vec3], f: &[u32]) -> Vec3 {
    let mut n = Vec3::ZERO;
    for i in 0..f.len() {
        let (a, b) = (verts[f[i] as usize], verts[f[(i + 1) % f.len()] as usize]);
        n += Vec3::new((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y));
    }
    n
}

/// Smooth proportional-editing falloff: 1 at the selection, 0 at `radius`.
pub fn falloff(dist: f32, radius: f32) -> f32 {
    if radius <= 0.0 || dist >= radius {
        return 0.0;
    }
    let t = 1.0 - dist / radius;
    t * t * (3.0 - 2.0 * t)
}

/// One level of Catmull–Clark subdivision (all faces become quads; boundaries are kept
/// crease-like). Per-corner UVs are interpolated linearly.
pub fn catmull_clark(pm: &PolyMesh) -> PolyMesh {
    let key = |a: u32, b: u32| (a.min(b), a.max(b));
    let nv = pm.verts.len();
    let ef = pm.edge_faces();
    let face_pts: Vec<Vec3> = (0..pm.faces.len()).map(|f| pm.face_center(f)).collect();
    let mut verts: Vec<Vec3> = Vec::with_capacity(nv + ef.len() + pm.faces.len());
    verts.resize(nv, Vec3::ZERO);
    // edge points
    let mut edge_pt: HashMap<(u32, u32), u32> = HashMap::with_capacity(ef.len());
    let mut edges: Vec<(u32, u32)> = ef.keys().copied().collect();
    edges.sort_unstable();
    for &(a, b) in &edges {
        let fs = &ef[&(a, b)];
        let mid = (pm.verts[a as usize] + pm.verts[b as usize]) * 0.5;
        let p = if fs.len() == 2 { (mid * 2.0 + face_pts[fs[0]] + face_pts[fs[1]]) * 0.25 } else { mid };
        edge_pt.insert((a, b), verts.len() as u32);
        verts.push(p);
    }
    // face points
    let fbase = verts.len() as u32;
    verts.extend(face_pts.iter().copied());
    // moved original vertices
    let mut q = vec![Vec3::ZERO; nv]; // sum of adjacent face points
    let mut nf = vec![0u32; nv];
    let mut r = vec![Vec3::ZERO; nv]; // sum of incident edge midpoints
    let mut ne = vec![0u32; nv];
    let mut bsum = vec![Vec3::ZERO; nv]; // boundary neighbours
    let mut nb = vec![0u32; nv];
    for (fi, f) in pm.faces.iter().enumerate() {
        for &v in f {
            q[v as usize] += face_pts[fi];
            nf[v as usize] += 1;
        }
    }
    for &(a, b) in &edges {
        let mid = (pm.verts[a as usize] + pm.verts[b as usize]) * 0.5;
        for v in [a, b] {
            r[v as usize] += mid;
            ne[v as usize] += 1;
        }
        if ef[&(a, b)].len() != 2 {
            bsum[a as usize] += pm.verts[b as usize];
            bsum[b as usize] += pm.verts[a as usize];
            nb[a as usize] += 1;
            nb[b as usize] += 1;
        }
    }
    for v in 0..nv {
        let p = pm.verts[v];
        verts[v] = if nb[v] >= 2 {
            p * 0.75 + bsum[v] * (0.25 / nb[v] as f32)
        } else if nf[v] > 0 && ne[v] >= 3 {
            let n = ne[v] as f32;
            let qa = q[v] / nf[v] as f32;
            let ra = r[v] / ne[v] as f32;
            (qa + ra * 2.0 + p * (n - 3.0)) / n
        } else {
            p
        };
    }
    let has_uv = pm.has_uvs();
    let mut faces = Vec::with_capacity(pm.faces.len() * 4);
    let mut uvs = vec![];
    for (fi, f) in pm.faces.iter().enumerate() {
        let n = f.len();
        let fp = fbase + fi as u32;
        let fuv = if has_uv { Some(&pm.uvs[fi]) } else { None };
        let center_uv = fuv.map(|u| {
            let s = u.iter().fold([0.0, 0.0], |a, b| [a[0] + b[0], a[1] + b[1]]);
            [s[0] / n as f32, s[1] / n as f32]
        });
        for i in 0..n {
            let (prev, cur, next) = (f[(i + n - 1) % n], f[i], f[(i + 1) % n]);
            faces.push(vec![cur, edge_pt[&key(cur, next)], fp, edge_pt[&key(prev, cur)]]);
            if let (Some(u), Some(c)) = (fuv, center_uv) {
                let (up, uc, un) = (u[(i + n - 1) % n], u[i], u[(i + 1) % n]);
                let m = |a: [f32; 2], b: [f32; 2]| [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
                uvs.push(vec![uc, m(uc, un), c, m(up, uc)]);
            }
        }
    }
    PolyMesh { verts, faces, uvs }
}

/// `levels` rounds of Catmull–Clark, stopping early above ~1.5M faces.
pub fn subdivide_surface(pm: &PolyMesh, levels: u32) -> PolyMesh {
    let mut m = pm.clone();
    for _ in 0..levels.min(6) {
        let next_faces: usize = m.faces.iter().map(|f| f.len()).sum();
        if next_faces > 1_500_000 {
            break;
        }
        m = catmull_clark(&m);
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{cube, plane};

    fn closed(pm: &PolyMesh) -> bool {
        pm.edge_faces().values().all(|f| f.len() == 2)
    }
    fn outward(pm: &PolyMesh) -> bool {
        let c = pm.verts.iter().sum::<Vec3>() / pm.verts.len() as f32;
        (0..pm.faces.len()).all(|f| pm.face_normal(f).dot(pm.face_center(f) - c) > 0.0)
    }

    #[test]
    fn extrude_top_face_of_cube() {
        let mut pm = cube();
        let mut sel = vec![];
        assert_eq!(select_by_normal(&pm, Vec3::Z, false, &mut sel), 1);
        let n = extrude(&mut pm, &mut sel).unwrap();
        assert!((n - Vec3::Z).length() < 1e-4);
        assert_eq!(pm.verts.len(), 12);
        assert_eq!(pm.faces.len(), 10);
        assert!(closed(&pm));
        // move the new selection up and check normals
        for (v, s) in pm.verts.iter_mut().zip(&sel) {
            if *s {
                v.z += 1.0;
            }
        }
        assert!(outward(&pm));
    }

    #[test]
    fn extrude_plane_makes_sides() {
        let mut pm = plane();
        let mut sel = vec![true; 4];
        extrude(&mut pm, &mut sel).unwrap();
        assert_eq!(pm.faces.len(), 5);
        assert_eq!(pm.verts.len(), 8);
    }

    #[test]
    fn inset_and_subdivide() {
        let mut pm = cube();
        let mut sel = vec![];
        select_by_normal(&pm, Vec3::Z, false, &mut sel);
        inset(&mut pm, &mut sel, 0.3).unwrap();
        assert_eq!(pm.faces.len(), 10);
        assert!(closed(&pm));
        assert!(outward(&pm));
        let mut sel2 = vec![];
        select_by_normal(&pm, -Vec3::Z, false, &mut sel2);
        subdivide(&mut pm, &mut sel2).unwrap();
        assert!(closed(&pm), "subdivide must keep the mesh watertight");
        assert_eq!(pm.faces.len(), 13);
    }

    #[test]
    fn loop_cut_cube() {
        let mut pm = cube();
        let mut sel = vec![];
        // vertical edge -> horizontal loop around the cube
        let e = edge_along(&pm, Vec3::Z).unwrap();
        let ring = loop_cut(&mut pm, &mut sel, e, 1).unwrap();
        assert_eq!(ring, 4);
        assert_eq!(pm.faces.len(), 10);
        assert_eq!(pm.verts.len(), 12);
        assert!(closed(&pm));
        assert!(outward(&pm));
        assert_eq!(sel.iter().filter(|s| **s).count(), 4);
        let mut pm2 = cube();
        loop_cut(&mut pm2, &mut sel, e, 3).unwrap();
        assert_eq!(pm2.faces.len(), 6 + 4 * 3);
        assert!(closed(&pm2));
    }

    #[test]
    fn delete_and_merge() {
        let mut pm = cube();
        let mut sel = vec![];
        select_by_normal(&pm, Vec3::Z, false, &mut sel);
        assert_eq!(delete(&mut pm, &mut sel, SelectMode::Face), 1);
        assert_eq!(pm.faces.len(), 5);
        let mut pm = cube();
        let mut sel = vec![];
        select_by_normal(&pm, Vec3::Z, false, &mut sel);
        merge_center(&mut pm, &mut sel).unwrap();
        assert_eq!(pm.verts.len(), 5); // a pyramid
        assert_eq!(pm.faces.len(), 5);
    }

    #[test]
    fn bevel_edges() {
        // one edge of a cube
        let mut pm = cube();
        let e = edge_along(&pm, Vec3::Z).unwrap();
        let mut sel = vec![false; pm.verts.len()];
        sel[e.0 as usize] = true;
        sel[e.1 as usize] = true;
        assert_eq!(bevel(&mut pm, &mut sel, 0.2, 1).unwrap(), 1);
        assert_eq!(pm.faces.len(), 7);
        assert_eq!(pm.verts.len(), 10);
        assert!(closed(&pm) && outward(&pm));
        assert_eq!(sel.iter().filter(|s| **s).count(), 4);
        // rounded: 3 segments
        let mut pm = cube();
        let mut sel = vec![false; pm.verts.len()];
        sel[e.0 as usize] = true;
        sel[e.1 as usize] = true;
        bevel(&mut pm, &mut sel, 0.3, 3).unwrap();
        assert_eq!(pm.faces.len(), 6 + 3);
        assert!(closed(&pm) && outward(&pm));
        // every edge of a cube, with corner patches
        for segs in [1, 2, 4] {
            let mut pm = cube();
            let mut sel = vec![true; pm.verts.len()];
            assert_eq!(bevel(&mut pm, &mut sel, 0.15, segs).unwrap(), 12);
            let corner_faces = if segs == 1 { 8 } else { 8 * 3 * segs };
            assert_eq!(pm.faces.len(), 6 + 12 * segs + corner_faces, "segments {segs}");
            assert!(closed(&pm), "closed, segments {segs}");
            assert!(outward(&pm), "outward, segments {segs}");
            let lo = pm.verts.iter().fold(Vec3::splat(f32::MAX), |a, b| a.min(*b));
            let hi = pm.verts.iter().fold(Vec3::splat(f32::MIN), |a, b| a.max(*b));
            assert!((hi - lo - Vec3::splat(2.0)).abs().max_element() < 1e-4, "keeps its size");
        }
        // the top face's 4 edges
        let mut pm = cube();
        let mut sel = vec![];
        select_by_normal(&pm, Vec3::Z, false, &mut sel);
        assert_eq!(bevel(&mut pm, &mut sel, 0.2, 2).unwrap(), 4);
        assert!(closed(&pm) && outward(&pm));
        assert!(bevel(&mut cube(), &mut vec![false; 8], 0.2, 1).is_err());
    }

    #[test]
    fn falloff_curve() {
        assert_eq!(falloff(0.0, 1.0), 1.0);
        assert_eq!(falloff(1.0, 1.0), 0.0);
        assert!((falloff(0.5, 1.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn subdivision_surface() {
        let pm = cube();
        let s1 = catmull_clark(&pm);
        assert_eq!((s1.verts.len(), s1.faces.len()), (26, 24));
        assert!(closed(&s1) && outward(&s1));
        let s2 = subdivide_surface(&pm, 2);
        assert_eq!(s2.faces.len(), 96);
        // the cage shrinks towards a sphere-like shape inside the original cube
        let r: Vec<f32> = s2.verts.iter().map(|v| v.length()).collect();
        assert!(r.iter().all(|x| *x < 1.74 && *x > 0.8));
        // a plane (all boundary): edges stay on the outline, corners round off like Blender
        let p = subdivide_surface(&crate::mesh::plane(), 1);
        assert_eq!(p.faces.len(), 4);
        assert!(p.verts.iter().any(|v| (v.x - 1.0).abs() < 1e-5 && v.y.abs() < 1e-5));
        assert!(p.verts.iter().all(|v| v.x.abs() <= 1.0 + 1e-5 && v.y.abs() <= 1.0 + 1e-5 && v.z.abs() < 1e-5));
    }
}
