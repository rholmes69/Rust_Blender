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
}
