//! Mesh booleans (union / difference / intersect) with BSP trees, after Evan Wallace's csg.js.
//! Works on closed triangle soups in world space; the result is a set of convex polygons.

use glam::DVec3;

const EPS: f64 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoolOp {
    Union,
    Difference,
    Intersect,
}

impl BoolOp {
    pub fn parse(s: &str) -> Option<BoolOp> {
        let s = s.to_lowercase();
        if s.contains("union") || s.contains("join") || s.contains("add") || s.contains("merge") {
            Some(BoolOp::Union)
        } else if s.contains("diff") || s.contains("sub") || s.contains("cut") || s.contains("hole") || s.contains("minus") || s.contains("carve") {
            Some(BoolOp::Difference)
        } else if s.contains("inter") || s.contains("common") || s.contains("overlap") {
            Some(BoolOp::Intersect)
        } else {
            None
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            BoolOp::Union => "Union",
            BoolOp::Difference => "Difference",
            BoolOp::Intersect => "Intersect",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Plane {
    n: DVec3,
    w: f64,
}

impl Plane {
    fn from_points(a: DVec3, b: DVec3, c: DVec3) -> Option<Plane> {
        let n = (b - a).cross(c - a);
        let len = n.length();
        if len < 1e-14 {
            return None;
        }
        let n = n / len;
        Some(Plane { n, w: n.dot(a) })
    }
    fn flip(&mut self) {
        self.n = -self.n;
        self.w = -self.w;
    }
}

#[derive(Clone, Debug)]
pub struct Polygon {
    pub verts: Vec<DVec3>,
    plane: Plane,
}

impl Polygon {
    pub fn new(verts: Vec<DVec3>) -> Option<Polygon> {
        if verts.len() < 3 {
            return None;
        }
        let plane = Plane::from_points(verts[0], verts[1], verts[2])?;
        Some(Polygon { verts, plane })
    }
    fn flip(&mut self) {
        self.verts.reverse();
        self.plane.flip();
    }
}

const COPLANAR: u8 = 0;
const FRONT: u8 = 1;
const BACK: u8 = 2;
const SPANNING: u8 = 3;

/// Split `p` by `plane` into the four buckets.
fn split(plane: &Plane, p: &Polygon, cf: &mut Vec<Polygon>, cb: &mut Vec<Polygon>, f: &mut Vec<Polygon>, b: &mut Vec<Polygon>) {
    let mut ptype = 0u8;
    let types: Vec<u8> = p
        .verts
        .iter()
        .map(|v| {
            let t = plane.n.dot(*v) - plane.w;
            let ty = if t < -EPS { BACK } else if t > EPS { FRONT } else { COPLANAR };
            ptype |= ty;
            ty
        })
        .collect();
    match ptype {
        COPLANAR => {
            if plane.n.dot(p.plane.n) > 0.0 {
                cf.push(p.clone())
            } else {
                cb.push(p.clone())
            }
        }
        FRONT => f.push(p.clone()),
        BACK => b.push(p.clone()),
        _ => {
            let (mut fv, mut bv) = (vec![], vec![]);
            let n = p.verts.len();
            for i in 0..n {
                let j = (i + 1) % n;
                let (ti, tj) = (types[i], types[j]);
                let (vi, vj) = (p.verts[i], p.verts[j]);
                if ti != BACK {
                    fv.push(vi);
                }
                if ti != FRONT {
                    bv.push(vi);
                }
                if (ti | tj) == SPANNING {
                    let t = (plane.w - plane.n.dot(vi)) / plane.n.dot(vj - vi);
                    let v = vi + (vj - vi) * t;
                    fv.push(v);
                    bv.push(v);
                }
            }
            if fv.len() >= 3 {
                f.push(Polygon { verts: fv, plane: p.plane });
            }
            if bv.len() >= 3 {
                b.push(Polygon { verts: bv, plane: p.plane });
            }
        }
    }
}

#[derive(Default)]
struct Node {
    plane: Option<Plane>,
    front: Option<Box<Node>>,
    back: Option<Box<Node>>,
    polys: Vec<Polygon>,
}

impl Node {
    fn new(polys: Vec<Polygon>) -> Node {
        let mut n = Node::default();
        n.build(polys);
        n
    }

    fn invert(&mut self) {
        for p in &mut self.polys {
            p.flip();
        }
        if let Some(pl) = &mut self.plane {
            pl.flip();
        }
        if let Some(f) = &mut self.front {
            f.invert();
        }
        if let Some(b) = &mut self.back {
            b.invert();
        }
        std::mem::swap(&mut self.front, &mut self.back);
    }

    /// Remove the parts of `polys` that are inside this BSP tree.
    fn clip_polygons(&self, polys: Vec<Polygon>) -> Vec<Polygon> {
        let Some(plane) = &self.plane else { return polys };
        let (mut f, mut b) = (vec![], vec![]);
        for p in &polys {
            let (mut cf, mut cb) = (vec![], vec![]);
            split(plane, p, &mut cf, &mut cb, &mut f, &mut b);
            f.append(&mut cf);
            b.append(&mut cb);
        }
        let mut f = match &self.front {
            Some(n) => n.clip_polygons(f),
            None => f,
        };
        let b = match &self.back {
            Some(n) => n.clip_polygons(b),
            None => vec![],
        };
        f.extend(b);
        f
    }

    fn clip_to(&mut self, other: &Node) {
        self.polys = other.clip_polygons(std::mem::take(&mut self.polys));
        if let Some(f) = &mut self.front {
            f.clip_to(other);
        }
        if let Some(b) = &mut self.back {
            b.clip_to(other);
        }
    }

    fn all_polygons(&self) -> Vec<Polygon> {
        let mut out = self.polys.clone();
        if let Some(f) = &self.front {
            out.extend(f.all_polygons());
        }
        if let Some(b) = &self.back {
            out.extend(b.all_polygons());
        }
        out
    }

    fn build(&mut self, polys: Vec<Polygon>) {
        if polys.is_empty() {
            return;
        }
        // split by the plane of a polygon from the middle (better balanced than the first)
        let plane = *self.plane.get_or_insert(polys[polys.len() / 2].plane);
        let (mut f, mut b) = (vec![], vec![]);
        let mut cf = vec![];
        let mut cb = vec![];
        for p in &polys {
            split(&plane, p, &mut cf, &mut cb, &mut f, &mut b);
        }
        self.polys.append(&mut cf);
        self.polys.append(&mut cb);
        if !f.is_empty() {
            self.front.get_or_insert_with(Default::default).build(f);
        }
        if !b.is_empty() {
            self.back.get_or_insert_with(Default::default).build(b);
        }
    }
}

/// Boolean of two closed polygon soups. Runs on a thread with a large stack (BSP recursion).
pub fn boolean(a: Vec<Polygon>, b: Vec<Polygon>, op: BoolOp) -> Vec<Polygon> {
    let h = std::thread::Builder::new()
        .stack_size(512 << 20)
        .spawn(move || {
            let mut a = Node::new(a);
            let mut b = Node::new(b);
            match op {
                BoolOp::Union => {
                    a.clip_to(&b);
                    b.clip_to(&a);
                    b.invert();
                    b.clip_to(&a);
                    b.invert();
                    a.build(b.all_polygons());
                }
                BoolOp::Difference => {
                    a.invert();
                    a.clip_to(&b);
                    b.clip_to(&a);
                    b.invert();
                    b.clip_to(&a);
                    b.invert();
                    a.build(b.all_polygons());
                    a.invert();
                }
                BoolOp::Intersect => {
                    a.invert();
                    b.clip_to(&a);
                    b.invert();
                    a.clip_to(&b);
                    b.clip_to(&a);
                    a.build(b.all_polygons());
                    a.invert();
                }
            }
            a.all_polygons()
        })
        .expect("spawn csg");
    h.join().unwrap_or_default()
}

/// Convex polygons → welded n-gon mesh (in the space the polygons are in).
pub fn to_polymesh(polys: &[Polygon], weld: f64) -> crate::mesh::PolyMesh {
    use std::collections::HashMap;
    let q = |v: f64| (v / weld).round() as i64;
    let mut map: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let mut verts = vec![];
    let mut faces = vec![];
    for p in polys {
        let mut f: Vec<u32> = vec![];
        for v in &p.verts {
            let k = (q(v.x), q(v.y), q(v.z));
            let id = *map.entry(k).or_insert_with(|| {
                verts.push(glam::Vec3::new(v.x as f32, v.y as f32, v.z as f32));
                (verts.len() - 1) as u32
            });
            if f.last() != Some(&id) && f.first() != Some(&id) {
                f.push(id);
            }
        }
        if f.len() >= 3 {
            faces.push(f);
        }
    }
    let mut pm = crate::mesh::PolyMesh { verts, faces, uvs: vec![] };
    pm.cleanup();
    pm
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cube(c: DVec3, h: f64) -> Vec<Polygon> {
        let pm = crate::mesh::cube();
        pm.faces
            .iter()
            .filter_map(|f| Polygon::new(f.iter().map(|&i| c + pm.verts[i as usize].as_dvec3() * h).collect()))
            .collect()
    }
    fn volume(polys: &[Polygon]) -> f64 {
        // divergence theorem over fan triangles
        let mut v = 0.0;
        for p in polys {
            for k in 1..p.verts.len() - 1 {
                v += p.verts[0].dot(p.verts[k].cross(p.verts[k + 1])) / 6.0;
            }
        }
        v
    }
    #[test]
    fn cube_booleans() {
        // two 2x2x2 cubes overlapping by half along x
        let a = cube(DVec3::ZERO, 1.0);
        let b = cube(DVec3::new(1.0, 0.0, 0.0), 1.0);
        let u = boolean(a.clone(), b.clone(), BoolOp::Union);
        let d = boolean(a.clone(), b.clone(), BoolOp::Difference);
        let i = boolean(a, b, BoolOp::Intersect);
        assert!((volume(&u) - 12.0).abs() < 1e-6, "union {}", volume(&u));
        assert!((volume(&d) - 4.0).abs() < 1e-6, "difference {}", volume(&d));
        assert!((volume(&i) - 4.0).abs() < 1e-6, "intersect {}", volume(&i));
        let pm = to_polymesh(&d, 1e-6);
        let (lo, hi) = pm.verts.iter().fold((glam::Vec3::splat(9.0), glam::Vec3::splat(-9.0)), |(l, h), v| (l.min(*v), h.max(*v)));
        assert!((lo.x + 1.0).abs() < 1e-5 && hi.x.abs() < 1e-5, "{lo} {hi}");
        assert_eq!(BoolOp::parse("cut a hole"), Some(BoolOp::Difference));
    }
}
