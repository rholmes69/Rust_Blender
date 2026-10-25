//! Shader nodes: a small graph of texture / pattern / math nodes feeding a material output.
//! Evaluated per shading point by the path tracer and baked to textures for the raster
//! renderers and glTF export. Coordinates are the mesh UVs (× the material's repeats).

use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MixMode {
    Mix,
    Multiply,
    Add,
    Overlay,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MathOp {
    Add,
    Multiply,
    Subtract,
    Power,
    Greater,
    Smooth,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NodeKind {
    /// Material output: Color, Roughness, Metallic, Emission, Height (bump). Unconnected inputs
    /// fall back to the material's own values.
    Output { bump: f32 },
    /// UV coordinates scaled and rotated.
    Coords { scale: f32, rotate: f32 },
    Image { texture: u32 },
    Noise { scale: f32, detail: u32, distortion: f32 },
    Voronoi { scale: f32, randomness: f32 },
    Checker { scale: f32 },
    Bricks { scale: f32, mortar: f32 },
    Waves { scale: f32, distortion: f32 },
    /// Fac → colour through gradient stops (position, colour).
    Ramp { stops: Vec<(f32, [f32; 3])> },
    Mix { mode: MixMode, fac: f32, a: [f32; 3], b: [f32; 3] },
    Math { op: MathOp, a: f32, b: f32 },
    Invert,
    Rgb { color: [f32; 3] },
    Value { value: f32 },
}

impl NodeKind {
    pub fn label(&self) -> &'static str {
        match self {
            NodeKind::Output { .. } => "Material Output",
            NodeKind::Coords { .. } => "Texture Coordinate",
            NodeKind::Image { .. } => "Image Texture",
            NodeKind::Noise { .. } => "Noise Texture",
            NodeKind::Voronoi { .. } => "Voronoi Texture",
            NodeKind::Checker { .. } => "Checker Texture",
            NodeKind::Bricks { .. } => "Brick Texture",
            NodeKind::Waves { .. } => "Wave Texture",
            NodeKind::Ramp { .. } => "Color Ramp",
            NodeKind::Mix { .. } => "Mix Color",
            NodeKind::Math { .. } => "Math",
            NodeKind::Invert => "Invert",
            NodeKind::Rgb { .. } => "RGB",
            NodeKind::Value { .. } => "Value",
        }
    }
    pub fn inputs(&self) -> &'static [&'static str] {
        match self {
            NodeKind::Output { .. } => &["Color", "Roughness", "Metallic", "Emission", "Height"],
            NodeKind::Coords { .. } | NodeKind::Rgb { .. } | NodeKind::Value { .. } => &[],
            NodeKind::Image { .. } | NodeKind::Noise { .. } | NodeKind::Voronoi { .. } | NodeKind::Checker { .. } | NodeKind::Bricks { .. } | NodeKind::Waves { .. } => &["Vector"],
            NodeKind::Ramp { .. } => &["Fac"],
            NodeKind::Mix { .. } => &["Fac", "A", "B"],
            NodeKind::Math { .. } => &["A", "B"],
            NodeKind::Invert => &["Color"],
        }
    }
    pub fn outputs(&self) -> &'static [&'static str] {
        match self {
            NodeKind::Output { .. } => &[],
            NodeKind::Coords { .. } => &["UV"],
            NodeKind::Image { .. } => &["Color", "Alpha"],
            NodeKind::Noise { .. } => &["Fac", "Color"],
            NodeKind::Voronoi { .. } => &["Distance", "Color", "Edge"],
            NodeKind::Checker { .. } | NodeKind::Bricks { .. } | NodeKind::Waves { .. } => &["Fac"],
            NodeKind::Ramp { .. } | NodeKind::Mix { .. } | NodeKind::Invert | NodeKind::Rgb { .. } => &["Color"],
            NodeKind::Math { .. } | NodeKind::Value { .. } => &["Value"],
        }
    }
    /// The kinds offered in the Add menu (with default settings).
    pub fn catalog() -> Vec<NodeKind> {
        vec![
            NodeKind::Coords { scale: 1.0, rotate: 0.0 },
            NodeKind::Image { texture: 0 },
            NodeKind::Noise { scale: 5.0, detail: 4, distortion: 0.0 },
            NodeKind::Voronoi { scale: 5.0, randomness: 1.0 },
            NodeKind::Checker { scale: 8.0 },
            NodeKind::Bricks { scale: 4.0, mortar: 0.06 },
            NodeKind::Waves { scale: 6.0, distortion: 2.0 },
            NodeKind::Ramp { stops: vec![(0.0, [0.0; 3]), (1.0, [1.0; 3])] },
            NodeKind::Mix { mode: MixMode::Mix, fac: 0.5, a: [0.8; 3], b: [0.2; 3] },
            NodeKind::Math { op: MathOp::Multiply, a: 1.0, b: 1.0 },
            NodeKind::Invert,
            NodeKind::Rgb { color: [0.8, 0.3, 0.2] },
            NodeKind::Value { value: 0.5 },
        ]
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Node {
    pub id: u32,
    pub kind: NodeKind,
    /// Position in the node editor.
    pub pos: [f32; 2],
}

/// `from` (node, output) → `to` (node, input).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Link {
    pub from: (u32, usize),
    pub to: (u32, usize),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodeGraph {
    pub id: u32,
    pub name: String,
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
}

#[derive(Clone, Copy, Debug)]
pub enum Val {
    F(f32),
    C(Vec3),
}

impl Val {
    fn f(self) -> f32 {
        match self {
            Val::F(x) => x,
            Val::C(c) => c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722,
        }
    }
    fn c(self) -> Vec3 {
        match self {
            Val::F(x) => Vec3::splat(x),
            Val::C(c) => c,
        }
    }
}

/// What a graph produces at one point (None = use the material's value).
#[derive(Clone, Copy, Debug, Default)]
pub struct Shade {
    /// sRGB-ish colour, like `Material::color`
    pub color: Option<Vec3>,
    pub roughness: Option<f32>,
    pub metallic: Option<f32>,
    pub emission: Option<f32>,
    pub height: Option<f32>,
    pub bump: f32,
}

// ---- patterns ----

fn hash2(x: i32, y: i32, s: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ s.wrapping_mul(0xcb1a_b31f);
    h = (h ^ (h >> 13)).wrapping_mul(0x5bd1_e995);
    ((h ^ (h >> 15)) & 0x00ff_ffff) as f32 / 16_777_215.0
}

fn vnoise(p: Vec2, seed: u32) -> f32 {
    let (i, f) = (p.floor(), p - p.floor());
    let u = f * f * (Vec2::splat(3.0) - 2.0 * f);
    let (x, y) = (i.x as i32, i.y as i32);
    let a = hash2(x, y, seed);
    let b = hash2(x + 1, y, seed);
    let c = hash2(x, y + 1, seed);
    let d = hash2(x + 1, y + 1, seed);
    (a + (b - a) * u.x) + (c - a + (a - b + d - c) * u.x) * u.y
}

pub fn fbm(p: Vec2, octaves: u32, seed: u32) -> f32 {
    let (mut s, mut amp, mut norm, mut q) = (0.0, 0.5, 0.0, p);
    for o in 0..octaves.clamp(1, 8) {
        s += vnoise(q, seed + o) * amp;
        norm += amp;
        amp *= 0.5;
        q = q * 2.03 + Vec2::new(17.3, 9.1);
    }
    s / norm
}

/// (F1 distance, cell id hash, F2 − F1)
fn voronoi(p: Vec2, randomness: f32) -> (f32, f32, f32) {
    let (ix, iy) = (p.x.floor() as i32, p.y.floor() as i32);
    let (mut d1, mut d2, mut id) = (9.0f32, 9.0f32, 0.0);
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (cx, cy) = (ix + dx, iy + dy);
            let r = randomness.clamp(0.0, 1.0);
            let pt = Vec2::new(cx as f32 + 0.5 + (hash2(cx, cy, 11) - 0.5) * r, cy as f32 + 0.5 + (hash2(cx, cy, 23) - 0.5) * r);
            let d = pt.distance(p);
            if d < d1 {
                d2 = d1;
                d1 = d;
                id = hash2(cx, cy, 37);
            } else if d < d2 {
                d2 = d;
            }
        }
    }
    (d1, id, d2 - d1)
}

fn ramp(stops: &[(f32, [f32; 3])], t: f32) -> Vec3 {
    if stops.is_empty() {
        return Vec3::splat(t);
    }
    let mut s: Vec<&(f32, [f32; 3])> = stops.iter().collect();
    s.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    if t <= s[0].0 {
        return Vec3::from(s[0].1);
    }
    for w in s.windows(2) {
        if t <= w[1].0 {
            let k = ((t - w[0].0) / (w[1].0 - w[0].0).max(1e-6)).clamp(0.0, 1.0);
            return Vec3::from(w[0].1).lerp(Vec3::from(w[1].1), k);
        }
    }
    Vec3::from(s.last().unwrap().1)
}

const MAX_NODES: usize = 48;

impl NodeGraph {
    pub fn node(&self, id: u32) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }
    pub fn fresh_id(&self) -> u32 {
        self.nodes.iter().map(|n| n.id).max().unwrap_or(0) + 1
    }
    pub fn output(&self) -> Option<&Node> {
        self.nodes.iter().find(|n| matches!(n.kind, NodeKind::Output { .. }))
    }
    /// The link feeding input `i` of node `id`.
    pub fn input_link(&self, id: u32, i: usize) -> Option<&Link> {
        self.links.iter().find(|l| l.to == (id, i))
    }
    /// Connect (replacing whatever fed that input); refuses cycles.
    pub fn connect(&mut self, from: (u32, usize), to: (u32, usize)) -> bool {
        if from.0 == to.0 || self.depends_on(from.0, to.0) {
            return false;
        }
        self.links.retain(|l| l.to != to);
        self.links.push(Link { from, to });
        true
    }
    /// Does node `a` (transitively) take input from `b`?
    fn depends_on(&self, a: u32, b: u32) -> bool {
        let mut stack = vec![a];
        let mut seen = std::collections::HashSet::new();
        while let Some(n) = stack.pop() {
            if n == b {
                return true;
            }
            if !seen.insert(n) {
                continue;
            }
            stack.extend(self.links.iter().filter(|l| l.to.0 == n).map(|l| l.from.0));
        }
        false
    }
    pub fn remove_node(&mut self, id: u32) {
        if self.node(id).is_some_and(|n| matches!(n.kind, NodeKind::Output { .. })) {
            return;
        }
        self.nodes.retain(|n| n.id != id);
        self.links.retain(|l| l.from.0 != id && l.to.0 != id);
    }
    pub fn add(&mut self, kind: NodeKind, pos: [f32; 2]) -> u32 {
        let id = self.fresh_id();
        self.nodes.push(Node { id, kind, pos });
        id
    }

    /// Hash of everything that affects the result (for caches).
    pub fn signature(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        format!("{:?}{:?}", self.nodes.iter().map(|n| (&n.id, &n.kind)).collect::<Vec<_>>(), self.links).hash(&mut h);
        h.finish()
    }

    /// Does the output's Height input have something connected?
    pub fn has_bump(&self) -> bool {
        self.output().is_some_and(|o| self.input_link(o.id, 4).is_some())
    }
    pub fn drives(&self, input: usize) -> bool {
        self.output().is_some_and(|o| self.input_link(o.id, input).is_some())
    }

    /// Evaluate at `uv`. `tex` samples a scene texture (id, uv) → sRGB rgba.
    pub fn eval(&self, uv: Vec2, tex: &dyn Fn(u32, Vec2) -> Option<[f32; 4]>) -> Shade {
        let Some(out) = self.output() else { return Shade::default() };
        let mut memo: [Option<[Val; 3]>; MAX_NODES] = [None; MAX_NODES];
        let bump = match out.kind {
            NodeKind::Output { bump } => bump,
            _ => 1.0,
        };
        let mut get = |i: usize| -> Option<Val> {
            let l = self.input_link(out.id, i)?;
            Some(self.value(l.from.0, l.from.1, uv, tex, &mut memo, 0))
        };
        Shade {
            color: get(0).map(|v| v.c()),
            roughness: get(1).map(|v| v.f().clamp(0.0, 1.0)),
            metallic: get(2).map(|v| v.f().clamp(0.0, 1.0)),
            emission: get(3).map(|v| v.f().max(0.0)),
            height: get(4).map(|v| v.f()),
            bump,
        }
    }

    /// Only the height (for bump finite differences).
    pub fn height(&self, uv: Vec2, tex: &dyn Fn(u32, Vec2) -> Option<[f32; 4]>) -> f32 {
        let Some(out) = self.output() else { return 0.0 };
        let Some(l) = self.input_link(out.id, 4) else { return 0.0 };
        let mut memo: [Option<[Val; 3]>; MAX_NODES] = [None; MAX_NODES];
        self.value(l.from.0, l.from.1, uv, tex, &mut memo, 0).f()
    }

    fn value(&self, id: u32, out: usize, uv: Vec2, tex: &dyn Fn(u32, Vec2) -> Option<[f32; 4]>, memo: &mut [Option<[Val; 3]>; MAX_NODES], depth: usize) -> Val {
        let Some(slot) = self.nodes.iter().position(|n| n.id == id) else { return Val::F(0.0) };
        if depth > MAX_NODES {
            return Val::F(0.0);
        }
        if slot < MAX_NODES {
            if let Some(v) = memo[slot] {
                return v[out.min(2)];
            }
        }
        let n = &self.nodes[slot];
        let inp = |i: usize, memo: &mut [Option<[Val; 3]>; MAX_NODES]| -> Option<Val> {
            let l = self.input_link(id, i)?;
            Some(self.value(l.from.0, l.from.1, uv, tex, memo, depth + 1))
        };
        let vec_in = |v: Option<Val>| -> Vec2 {
            match v {
                Some(Val::C(c)) => Vec2::new(c.x, c.y),
                Some(Val::F(f)) => uv + Vec2::splat(f),
                None => uv,
            }
        };
        let z = Val::F(0.0);
        let r: [Val; 3] = match &n.kind {
            NodeKind::Output { .. } => [z; 3],
            NodeKind::Coords { scale, rotate } => {
                let a = rotate.to_radians();
                let (s, c) = a.sin_cos();
                let p = Vec2::new(uv.x * c - uv.y * s, uv.x * s + uv.y * c) * *scale;
                [Val::C(Vec3::new(p.x, p.y, 0.0)), z, z]
            }
            NodeKind::Image { texture } => {
                let p = vec_in(inp(0, memo));
                let c = tex(*texture, p).unwrap_or([1.0, 0.0, 1.0, 1.0]);
                [Val::C(Vec3::new(c[0], c[1], c[2])), Val::F(c[3]), z]
            }
            NodeKind::Noise { scale, detail, distortion } => {
                let mut p = vec_in(inp(0, memo)) * *scale;
                if *distortion != 0.0 {
                    p += Vec2::new(fbm(p + Vec2::splat(5.2), 2, 91) - 0.5, fbm(p + Vec2::splat(1.3), 2, 77) - 0.5) * *distortion;
                }
                let f = fbm(p, *detail, 3);
                let col = Vec3::new(f, fbm(p + Vec2::splat(31.7), *detail, 5), fbm(p + Vec2::splat(-12.9), *detail, 7));
                [Val::F(f), Val::C(col), z]
            }
            NodeKind::Voronoi { scale, randomness } => {
                let p = vec_in(inp(0, memo)) * *scale;
                let (d, idh, edge) = voronoi(p, *randomness);
                let col = Vec3::new(idh, (idh * 7.31).fract(), (idh * 13.17).fract());
                [Val::F(d), Val::C(col), Val::F(edge)]
            }
            NodeKind::Checker { scale } => {
                let p = vec_in(inp(0, memo)) * *scale;
                let k = (p.x.floor() as i64 + p.y.floor() as i64).rem_euclid(2) as f32;
                [Val::F(k), z, z]
            }
            NodeKind::Bricks { scale, mortar } => {
                let p = vec_in(inp(0, memo)) * *scale;
                let row = (p.y * 2.0).floor();
                let x = p.x + if row as i64 % 2 == 0 { 0.0 } else { 0.5 };
                let (fx, fy) = (x.fract().abs(), (p.y * 2.0).fract().abs());
                let m = *mortar;
                let inside = fx > m && fx < 1.0 - m && fy > m * 2.0 && fy < 1.0 - m * 2.0;
                // brick variation in the colour channel
                let var = hash2(x.floor() as i32, row as i32, 5);
                [Val::F(if inside { 0.6 + 0.4 * var } else { 0.0 }), z, z]
            }
            NodeKind::Waves { scale, distortion } => {
                let p = vec_in(inp(0, memo)) * *scale;
                let d = if *distortion != 0.0 { (fbm(p, 3, 13) - 0.5) * *distortion * 2.0 } else { 0.0 };
                let f = 0.5 + 0.5 * ((p.x + p.y) * std::f32::consts::TAU + d * std::f32::consts::TAU).sin();
                [Val::F(f), z, z]
            }
            NodeKind::Ramp { stops } => {
                let t = inp(0, memo).map(|v| v.f()).unwrap_or(0.5);
                [Val::C(ramp(stops, t)), z, z]
            }
            NodeKind::Mix { mode, fac, a, b } => {
                let f = inp(0, memo).map(|v| v.f()).unwrap_or(*fac).clamp(0.0, 1.0);
                let ca = inp(1, memo).map(|v| v.c()).unwrap_or(Vec3::from(*a));
                let cb = inp(2, memo).map(|v| v.c()).unwrap_or(Vec3::from(*b));
                let mixed = match mode {
                    MixMode::Mix => cb,
                    MixMode::Multiply => ca * cb,
                    MixMode::Add => ca + cb,
                    MixMode::Overlay => Vec3::new(
                        if ca.x < 0.5 { 2.0 * ca.x * cb.x } else { 1.0 - 2.0 * (1.0 - ca.x) * (1.0 - cb.x) },
                        if ca.y < 0.5 { 2.0 * ca.y * cb.y } else { 1.0 - 2.0 * (1.0 - ca.y) * (1.0 - cb.y) },
                        if ca.z < 0.5 { 2.0 * ca.z * cb.z } else { 1.0 - 2.0 * (1.0 - ca.z) * (1.0 - cb.z) },
                    ),
                };
                [Val::C(ca.lerp(mixed, f)), z, z]
            }
            NodeKind::Math { op, a, b } => {
                let x = inp(0, memo).map(|v| v.f()).unwrap_or(*a);
                let y = inp(1, memo).map(|v| v.f()).unwrap_or(*b);
                let v = match op {
                    MathOp::Add => x + y,
                    MathOp::Multiply => x * y,
                    MathOp::Subtract => x - y,
                    MathOp::Power => x.max(0.0).powf(y),
                    MathOp::Greater => (x > y) as u8 as f32,
                    MathOp::Smooth => {
                        let t = (x / y.max(1e-6)).clamp(0.0, 1.0);
                        t * t * (3.0 - 2.0 * t)
                    }
                };
                [Val::F(v), z, z]
            }
            NodeKind::Invert => {
                let c = inp(0, memo).map(|v| v.c()).unwrap_or(Vec3::ZERO);
                [Val::C(Vec3::ONE - c), z, z]
            }
            NodeKind::Rgb { color } => [Val::C(Vec3::from(*color)), z, z],
            NodeKind::Value { value } => [Val::F(*value), z, z],
        };
        if slot < MAX_NODES {
            memo[slot] = Some(r);
        }
        r[out.min(2)]
    }

    /// A new graph with just an output and one colour.
    pub fn basic(id: u32, name: &str, color: [f32; 3]) -> NodeGraph {
        let mut g = NodeGraph { id, name: name.into(), nodes: vec![], links: vec![] };
        let out = g.add(NodeKind::Output { bump: 1.0 }, [420.0, 120.0]);
        let rgb = g.add(NodeKind::Rgb { color }, [120.0, 100.0]);
        g.connect((rgb, 0), (out, 0));
        g.auto_layout();
        g
    }

    /// Ready-made materials.
    pub fn preset(id: u32, name: &str) -> Option<NodeGraph> {
        let low = name.to_lowercase();
        let mut g = NodeGraph { id, name: String::new(), nodes: vec![], links: vec![] };
        let out = g.add(NodeKind::Output { bump: 1.0 }, [640.0, 160.0]);
        let co = |g: &mut NodeGraph, s: f32| g.add(NodeKind::Coords { scale: s, rotate: 0.0 }, [0.0, 120.0]);
        let title;
        if low.contains("rust") {
            title = "Rusty Metal";
            let c = co(&mut g, 1.0);
            let n = g.add(NodeKind::Noise { scale: 6.0, detail: 6, distortion: 1.5 }, [160.0, 60.0]);
            let m = g.add(NodeKind::Math { op: MathOp::Smooth, a: 0.0, b: 0.62 }, [330.0, 60.0]);
            let r = g.add(NodeKind::Ramp { stops: vec![(0.35, [0.62, 0.64, 0.66]), (0.55, [0.45, 0.22, 0.09]), (0.8, [0.3, 0.12, 0.05])] }, [330.0, 180.0]);
            let inv = g.add(NodeKind::Math { op: MathOp::Subtract, a: 1.0, b: 0.0 }, [480.0, 260.0]);
            g.connect((c, 0), (n, 0));
            g.connect((n, 0), (m, 0));
            g.connect((n, 0), (r, 0));
            g.connect((r, 0), (out, 0));
            g.connect((m, 0), (out, 1)); // rough where rusty
            g.connect((m, 0), (inv, 1));
            g.connect((inv, 0), (out, 2)); // metal where clean
            g.connect((n, 0), (out, 4));
        } else if low.contains("wood") {
            title = "Wood";
            let c = co(&mut g, 1.0);
            let w = g.add(NodeKind::Waves { scale: 3.0, distortion: 3.0 }, [160.0, 100.0]);
            let r = g.add(NodeKind::Ramp { stops: vec![(0.0, [0.32, 0.17, 0.07]), (0.6, [0.55, 0.33, 0.15]), (1.0, [0.62, 0.4, 0.2])] }, [330.0, 100.0]);
            g.connect((c, 0), (w, 0));
            g.connect((w, 0), (r, 0));
            g.connect((r, 0), (out, 0));
            g.connect((w, 0), (out, 4));
            if let Some(o) = g.nodes.iter_mut().find(|n| n.id == out) {
                o.kind = NodeKind::Output { bump: 0.3 };
            }
        } else if low.contains("marble") {
            title = "Marble";
            let c = co(&mut g, 1.0);
            let w = g.add(NodeKind::Waves { scale: 2.0, distortion: 6.0 }, [160.0, 100.0]);
            let r = g.add(NodeKind::Ramp { stops: vec![(0.0, [0.25, 0.27, 0.3]), (0.12, [0.85, 0.85, 0.83]), (1.0, [0.95, 0.95, 0.94])] }, [330.0, 100.0]);
            let v = g.add(NodeKind::Value { value: 0.12 }, [330.0, 230.0]);
            g.connect((c, 0), (w, 0));
            g.connect((w, 0), (r, 0));
            g.connect((r, 0), (out, 0));
            g.connect((v, 0), (out, 1));
        } else if low.contains("stone") || low.contains("rock") || low.contains("cobble") {
            title = "Cobblestone";
            let c = co(&mut g, 1.0);
            let vo = g.add(NodeKind::Voronoi { scale: 6.0, randomness: 1.0 }, [160.0, 60.0]);
            let n = g.add(NodeKind::Noise { scale: 40.0, detail: 4, distortion: 0.0 }, [160.0, 240.0]);
            let edge = g.add(NodeKind::Math { op: MathOp::Smooth, a: 0.0, b: 0.12 }, [330.0, 40.0]);
            // each stone its own shade
            let shade = g.add(NodeKind::Ramp { stops: vec![(0.0, [0.38, 0.36, 0.33]), (0.5, [0.55, 0.52, 0.47]), (1.0, [0.46, 0.42, 0.38])] }, [330.0, 150.0]);
            let grain = g.add(NodeKind::Mix { mode: MixMode::Overlay, fac: 0.35, a: [0.5; 3], b: [0.5; 3] }, [480.0, 150.0]);
            let col = g.add(NodeKind::Mix { mode: MixMode::Mix, fac: 0.5, a: [0.12, 0.11, 0.1], b: [0.5; 3] }, [520.0, 40.0]);
            let rough = g.add(NodeKind::Value { value: 0.85 }, [480.0, 300.0]);
            g.connect((c, 0), (vo, 0));
            g.connect((c, 0), (n, 0));
            g.connect((vo, 2), (edge, 0));
            g.connect((vo, 1), (shade, 0));
            g.connect((shade, 0), (grain, 1));
            g.connect((n, 0), (grain, 2));
            g.connect((edge, 0), (col, 0));
            g.connect((grain, 0), (col, 2));
            g.connect((col, 0), (out, 0));
            g.connect((rough, 0), (out, 1));
            g.connect((edge, 0), (out, 4));
        } else if low.contains("lava") || low.contains("magma") {
            title = "Lava";
            let c = co(&mut g, 1.0);
            let vo = g.add(NodeKind::Voronoi { scale: 4.0, randomness: 1.0 }, [160.0, 60.0]);
            let glow = g.add(NodeKind::Math { op: MathOp::Smooth, a: 0.0, b: 0.15 }, [330.0, 60.0]);
            let inv = g.add(NodeKind::Math { op: MathOp::Subtract, a: 1.0, b: 0.0 }, [330.0, 180.0]);
            let col = g.add(NodeKind::Mix { mode: MixMode::Mix, fac: 0.5, a: [1.0, 0.45, 0.05], b: [0.08, 0.06, 0.05] }, [480.0, 60.0]);
            let em = g.add(NodeKind::Math { op: MathOp::Multiply, a: 0.0, b: 4.0 }, [480.0, 200.0]);
            g.connect((c, 0), (vo, 0));
            g.connect((vo, 2), (glow, 0));
            g.connect((glow, 0), (col, 0));
            g.connect((glow, 0), (inv, 1));
            g.connect((inv, 0), (em, 0));
            g.connect((col, 0), (out, 0));
            g.connect((em, 0), (out, 3));
            g.connect((glow, 0), (out, 4));
        } else if low.contains("tile") || low.contains("checker") {
            title = "Tiles";
            let c = co(&mut g, 1.0);
            let ch = g.add(NodeKind::Checker { scale: 8.0 }, [160.0, 100.0]);
            let mx = g.add(NodeKind::Mix { mode: MixMode::Mix, fac: 0.5, a: [0.9, 0.9, 0.88], b: [0.15, 0.15, 0.17] }, [330.0, 100.0]);
            g.connect((c, 0), (ch, 0));
            g.connect((ch, 0), (mx, 0));
            g.connect((mx, 0), (out, 0));
        } else if low.contains("brick") {
            title = "Bricks";
            let c = co(&mut g, 1.0);
            let b = g.add(NodeKind::Bricks { scale: 4.0, mortar: 0.05 }, [160.0, 100.0]);
            let r = g.add(NodeKind::Ramp { stops: vec![(0.0, [0.72, 0.7, 0.66]), (0.05, [0.4, 0.15, 0.1]), (1.0, [0.65, 0.28, 0.17])] }, [330.0, 100.0]);
            g.connect((c, 0), (b, 0));
            g.connect((b, 0), (r, 0));
            g.connect((r, 0), (out, 0));
            g.connect((b, 0), (out, 4));
        } else {
            return None;
        }
        g.name = title.into();
        g.auto_layout();
        Some(g)
    }

    /// Columns by distance from the output (inputs to the left), stacked without overlap.
    pub fn auto_layout(&mut self) {
        let Some(out) = self.output().map(|o| o.id) else { return };
        let mut depth: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
        depth.insert(out, 0);
        // longest path (graphs are small and acyclic)
        for _ in 0..self.nodes.len() {
            let mut changed = false;
            for l in &self.links {
                if let Some(&d) = depth.get(&l.to.0) {
                    let e = depth.entry(l.from.0).or_insert(0);
                    if *e < d + 1 {
                        *e = d + 1;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        let maxd = depth.values().copied().max().unwrap_or(0);
        let mut cols: Vec<Vec<usize>> = vec![vec![]; maxd + 2];
        for (i, n) in self.nodes.iter().enumerate() {
            let d = depth.get(&n.id).copied().unwrap_or(maxd + 1);
            cols[d].push(i);
        }
        for (d, col) in cols.iter_mut().enumerate() {
            col.sort_by(|a, b| self.nodes[*a].pos[1].partial_cmp(&self.nodes[*b].pos[1]).unwrap_or(std::cmp::Ordering::Equal));
            let mut y = 20.0;
            for &i in col.iter() {
                let x = 20.0 + (maxd + 1 - d.min(maxd + 1)) as f32 * 210.0;
                self.nodes[i].pos = [x, y];
                y += node_height_estimate(&self.nodes[i].kind) + 24.0;
            }
        }
    }
}

/// Height of a node box in the editor (kept in sync with node_editor.rs).
pub fn node_height_estimate(k: &NodeKind) -> f32 {
    let params = match k {
        NodeKind::Output { .. } | NodeKind::Checker { .. } | NodeKind::Image { .. } | NodeKind::Rgb { .. } | NodeKind::Value { .. } => 1,
        NodeKind::Coords { .. } | NodeKind::Voronoi { .. } | NodeKind::Bricks { .. } | NodeKind::Waves { .. } => 2,
        NodeKind::Noise { .. } | NodeKind::Math { .. } => 3,
        NodeKind::Mix { .. } => 4,
        NodeKind::Ramp { stops } => 2 + stops.len(),
        NodeKind::Invert => 0,
    };
    22.0 + k.outputs().len() as f32 * 18.0 + params as f32 * 22.0 + k.inputs().len() as f32 * 18.0 + 6.0
}

pub const PRESETS: &[&str] = &["Rusty Metal", "Wood", "Marble", "Cobblestone", "Lava", "Bricks", "Tiles"];

/// Bake the colour (with bump shading for the raster renderers when `shade_bumps`), the
/// roughness/metallic map and a normal map, each `size`² over UV 0..1.
pub struct Baked {
    pub color: crate::textures::TexImage,
    pub mr: Option<crate::textures::TexImage>,
    pub normal: Option<crate::textures::TexImage>,
}

pub fn bake(g: &NodeGraph, base: crate::scene::Material, size: usize, shade_bumps: bool, tex: &dyn Fn(u32, Vec2) -> Option<[f32; 4]>) -> Baked {
    let n = size.max(4);
    let q = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
    let mut heights = vec![0.0f32; n * n];
    let mut col = Vec::with_capacity(n * n);
    let mut mr = Vec::with_capacity(n * n);
    let (drives_mr, bump) = (g.drives(1) || g.drives(2), g.has_bump());
    for y in 0..n {
        for x in 0..n {
            let uv = Vec2::new((x as f32 + 0.5) / n as f32, (y as f32 + 0.5) / n as f32);
            let s = g.eval(uv, tex);
            let c = s.color.unwrap_or(Vec3::from(base.color));
            col.push(c);
            mr.push([255, q(s.roughness.unwrap_or(1.0)), q(s.metallic.unwrap_or(1.0)), 255]);
            heights[y * n + x] = s.height.unwrap_or(0.0);
        }
    }
    let strength = g.output().map(|o| if let NodeKind::Output { bump } = o.kind { bump } else { 1.0 }).unwrap_or(1.0);
    let normal = bump.then(|| {
        let hi = |x: i64, y: i64| heights[(y.rem_euclid(n as i64) as usize) * n + x.rem_euclid(n as i64) as usize];
        let k = strength * n as f32 / 64.0;
        let mut rgba = Vec::with_capacity(n * n);
        for y in 0..n as i64 {
            for x in 0..n as i64 {
                let dx = (hi(x + 1, y) - hi(x - 1, y)) * 0.5;
                let dy = (hi(x, y + 1) - hi(x, y - 1)) * 0.5;
                let nn = Vec3::new(-dx * k, dy * k, 1.0).normalize();
                let e = |c: f32| ((c * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
                rgba.push([e(nn.x), e(nn.y), e(nn.z), 255]);
            }
        }
        crate::textures::TexImage { w: n, h: n, rgba }
    });
    let mut color = crate::textures::TexImage { w: n, h: n, rgba: col.iter().map(|c| [q(c.x), q(c.y), q(c.z), 255]).collect() };
    if shade_bumps {
        if let Some(nm) = &normal {
            color = crate::textures::bake_bumps(Some(&color), nm, 1.0);
        }
    }
    Baked { color, mr: drives_mr.then(|| crate::textures::TexImage { w: n, h: n, rgba: mr }), normal }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn presets_evaluate_and_bake() {
        let none = |_: u32, _: Vec2| None;
        for p in PRESETS {
            let g = NodeGraph::preset(1, p).unwrap_or_else(|| panic!("{p}"));
            assert_eq!(g.name, *p);
            let a = g.eval(Vec2::new(0.1, 0.2), &none);
            let b = g.eval(Vec2::new(0.6, 0.7), &none);
            assert!(a.color.is_some(), "{p} drives the colour");
            assert!(a.color.unwrap().is_finite() && b.color.unwrap().is_finite());
            let bk = bake(&g, crate::scene::Material::default(), 32, true, &none);
            assert_eq!(bk.color.rgba.len(), 32 * 32);
            // the pattern varies
            let first = bk.color.rgba[0];
            assert!(bk.color.rgba.iter().any(|c| (c[0] as i32 - first[0] as i32).abs() > 8), "{p} varies");
        }
        let rust = NodeGraph::preset(1, "rust").unwrap();
        assert!(rust.drives(1) && rust.drives(2) && rust.has_bump());
        // cycles are refused, links replace
        let mut g = NodeGraph::basic(1, "x", [1.0, 0.0, 0.0]);
        let v = g.add(NodeKind::Value { value: 0.3 }, [0.0, 0.0]);
        let m = g.add(NodeKind::Math { op: MathOp::Add, a: 0.0, b: 0.0 }, [0.0, 0.0]);
        assert!(g.connect((v, 0), (m, 0)));
        assert!(!g.connect((m, 0), (m, 1)), "self link");
        let out = g.output().unwrap().id;
        assert!(g.connect((m, 0), (out, 1)));
        assert!((g.eval(Vec2::ZERO, &none).roughness.unwrap() - 0.3).abs() < 1e-6);
        g.remove_node(m);
        assert!(g.eval(Vec2::ZERO, &none).roughness.is_none());
    }
}
