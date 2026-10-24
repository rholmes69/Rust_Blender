//! Shape keys (glTF morph targets): alternative vertex positions of a mesh blended by weights.

use crate::mesh::PolyMesh;
use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShapeKey {
    pub name: String,
    /// Absolute positions of every vertex in this shape.
    pub verts: Vec<Vec3>,
    pub value: f32,
    /// Animated value: (frame, value), sorted by frame.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keys: Vec<(i32, f32)>,
}

impl ShapeKey {
    pub fn sample(&self, frame: f32) -> Option<f32> {
        let first = self.keys.first()?;
        if frame <= first.0 as f32 {
            return Some(first.1);
        }
        let last = self.keys.last()?;
        if frame >= last.0 as f32 {
            return Some(last.1);
        }
        let i = self.keys.partition_point(|k| k.0 as f32 <= frame);
        let (a, b) = (self.keys[i - 1], self.keys[i]);
        let span = (b.0 - a.0).max(1) as f32;
        let mut t = (frame - a.0 as f32) / span;
        if span > 1.5 {
            t = t * t * (3.0 - 2.0 * t);
        }
        Some(a.1 + (b.1 - a.1) * t)
    }

    pub fn insert_key(&mut self, frame: i32) {
        self.keys.retain(|k| k.0 != frame);
        self.keys.push((frame, self.value));
        self.keys.sort_by_key(|k| k.0);
    }
}

/// The basis with every active, matching shape key blended in (None when nothing changes).
pub fn apply(basis: &PolyMesh, keys: &[ShapeKey]) -> Option<PolyMesh> {
    let active: Vec<&ShapeKey> = keys.iter().filter(|k| k.value.abs() > 1e-6 && k.verts.len() == basis.verts.len()).collect();
    if active.is_empty() {
        return None;
    }
    let mut out = basis.clone();
    for k in active {
        for (v, (t, b)) in out.verts.iter_mut().zip(k.verts.iter().zip(&basis.verts)) {
            *v += (*t - *b) * k.value;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn blend_and_animate() {
        let basis = crate::mesh::cube();
        let mut tall = ShapeKey { name: "Tall".into(), verts: basis.verts.iter().map(|v| *v * Vec3::new(1.0, 1.0, 2.0)).collect(), value: 0.5, keys: vec![] };
        let m = apply(&basis, std::slice::from_ref(&tall)).unwrap();
        let top = m.verts.iter().map(|v| v.z).fold(f32::MIN, f32::max);
        assert!((top - 1.5).abs() < 1e-5);
        tall.value = 0.0;
        assert!(apply(&basis, std::slice::from_ref(&tall)).is_none());
        tall.insert_key(1);
        tall.value = 1.0;
        tall.insert_key(11);
        assert!((tall.sample(6.0).unwrap() - 0.5).abs() < 1e-5);
    }
}
