//! Small triangle BVH used by the F12 render for shadow rays and ambient occlusion.

use glam::Vec3;

pub struct Bvh {
    tris: Vec<[Vec3; 3]>,
    nodes: Vec<Node>,
}

#[derive(Clone, Copy)]
struct Node {
    lo: Vec3,
    hi: Vec3,
    /// leaf: first triangle index; inner: index of the left child (right = left + 1)
    start: u32,
    /// number of triangles (0 = inner node)
    count: u32,
}

const LEAF: usize = 4;

impl Bvh {
    pub fn new(mut tris: Vec<[Vec3; 3]>) -> Self {
        let mut nodes = Vec::with_capacity(tris.len() / 2 + 1);
        if !tris.is_empty() {
            nodes.push(Node { lo: Vec3::ZERO, hi: Vec3::ZERO, start: 0, count: 0 });
            let n = tris.len();
            build(&mut tris, &mut nodes, 0, 0, n);
        }
        Self { tris, nodes }
    }

    /// Is anything hit between `o` and `o + d * tmax` (d normalised)?
    pub fn occluded(&self, o: Vec3, d: Vec3, tmax: f32) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let inv = Vec3::ONE / d;
        let mut stack = [0u32; 64];
        let mut sp = 1;
        while sp > 0 {
            sp -= 1;
            let n = self.nodes[stack[sp] as usize];
            if !slab(n.lo, n.hi, o, inv, tmax) {
                continue;
            }
            if n.count > 0 {
                for t in &self.tris[n.start as usize..(n.start + n.count) as usize] {
                    if hit(t, o, d, tmax) {
                        return true;
                    }
                }
            } else if sp + 2 <= stack.len() {
                stack[sp] = n.start;
                stack[sp + 1] = n.start + 1;
                sp += 2;
            }
        }
        false
    }
}

fn bounds(t: &[[Vec3; 3]]) -> (Vec3, Vec3) {
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    for tri in t {
        for p in tri {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
    }
    (lo, hi)
}

fn build(tris: &mut [[Vec3; 3]], nodes: &mut Vec<Node>, ni: usize, start: usize, end: usize) {
    let (lo, hi) = bounds(&tris[start..end]);
    // pad flat boxes so the slab test never divides 0 by 0
    let (lo, hi) = (lo - Vec3::splat(1e-4), hi + Vec3::splat(1e-4));
    if end - start <= LEAF {
        nodes[ni] = Node { lo, hi, start: start as u32, count: (end - start) as u32 };
        return;
    }
    let ext = hi - lo;
    let axis = if ext.x >= ext.y && ext.x >= ext.z { 0 } else if ext.y >= ext.z { 1 } else { 2 };
    let mid = (start + end) / 2;
    let key = |t: &[Vec3; 3]| t[0][axis] + t[1][axis] + t[2][axis];
    tris[start..end].select_nth_unstable_by(mid - start, |a, b| key(a).partial_cmp(&key(b)).unwrap_or(std::cmp::Ordering::Equal));
    let left = nodes.len();
    nodes.push(Node { lo, hi, start: 0, count: 0 });
    nodes.push(Node { lo, hi, start: 0, count: 0 });
    nodes[ni] = Node { lo, hi, start: left as u32, count: 0 };
    build(tris, nodes, left, start, mid);
    build(tris, nodes, left + 1, mid, end);
}

fn slab(lo: Vec3, hi: Vec3, o: Vec3, inv: Vec3, tmax: f32) -> bool {
    let t0 = (lo - o) * inv;
    let t1 = (hi - o) * inv;
    let tmin_v = t0.min(t1);
    let tmax_v = t0.max(t1);
    let tn = tmin_v.max_element().max(0.0);
    let tf = tmax_v.min_element().min(tmax);
    tn <= tf
}

/// Möller–Trumbore, both sides.
fn hit(t: &[Vec3; 3], o: Vec3, d: Vec3, tmax: f32) -> bool {
    let e1 = t[1] - t[0];
    let e2 = t[2] - t[0];
    let p = d.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-9 {
        return false;
    }
    let inv = 1.0 / det;
    let s = o - t[0];
    let u = s.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return false;
    }
    let q = s.cross(e1);
    let v = d.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
    let tt = e2.dot(q) * inv;
    tt > 1e-4 && tt < tmax
}

/// Fixed cosine-ish hemisphere directions around +Z, rotated per point for AO.
pub const AO_DIRS: [[f32; 3]; 8] = [
    [0.52, 0.0, 0.85],
    [-0.26, 0.45, 0.85],
    [-0.26, -0.45, 0.85],
    [0.90, 0.24, 0.36],
    [-0.24, 0.90, 0.36],
    [-0.66, -0.66, 0.36],
    [0.66, -0.66, 0.36],
    [0.0, 0.0, 1.0],
];

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn occlusion() {
        // a unit quad at z = 1 above the origin
        let q = [
            [Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), Vec3::new(1.0, 1.0, 1.0)],
            [Vec3::new(-1.0, -1.0, 1.0), Vec3::new(1.0, 1.0, 1.0), Vec3::new(-1.0, 1.0, 1.0)],
        ];
        let mut many = q.to_vec();
        for i in 0..200 {
            let x = 10.0 + i as f32;
            many.push([Vec3::new(x, 0.0, 0.0), Vec3::new(x + 0.5, 0.0, 0.0), Vec3::new(x, 0.5, 0.0)]);
        }
        let b = Bvh::new(many);
        assert!(b.occluded(Vec3::ZERO, Vec3::Z, 10.0));
        assert!(!b.occluded(Vec3::ZERO, Vec3::Z, 0.5), "blocker beyond tmax");
        assert!(!b.occluded(Vec3::ZERO, -Vec3::Z, 10.0));
        assert!(!b.occluded(Vec3::new(3.0, 0.0, 0.0), Vec3::Z, 10.0));
    }
}
