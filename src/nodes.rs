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
pub enum GradMode {
    Linear,
    Radial,
    Spherical,
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
    /// UV coordinates (or the object's own 3D coordinates, which have no seams) scaled and rotated.
    Coords {
        scale: f32,
        rotate: f32,
        #[serde(default)]
        object: bool,
    },
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
    /// Fac from the vector: left→right, around, or from the centre.
    Gradient { mode: GradMode },
    HueSat { hue: f32, saturation: f32, value: f32 },
    /// A normal-map colour → tangent-space normal for the output's Normal input.
    NormalMap { strength: f32 },
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
            NodeKind::Gradient { .. } => "Gradient Texture",
            NodeKind::HueSat { .. } => "Hue/Saturation",
            NodeKind::NormalMap { .. } => "Normal Map",
            NodeKind::Rgb { .. } => "RGB",
            NodeKind::Value { .. } => "Value",
        }
    }
    pub fn inputs(&self) -> &'static [&'static str] {
        match self {
            NodeKind::Output { .. } => &["Color", "Roughness", "Metallic", "Emission", "Height", "Normal"],
            NodeKind::Gradient { .. } => &["Vector"],
            NodeKind::HueSat { .. } | NodeKind::NormalMap { .. } => &["Color"],
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
            NodeKind::Coords { .. } => &["Vector"],
            NodeKind::Image { .. } => &["Color", "Alpha"],
            NodeKind::Noise { .. } => &["Fac", "Color"],
            NodeKind::Voronoi { .. } => &["Distance", "Color", "Edge"],
            NodeKind::Checker { .. } | NodeKind::Bricks { .. } | NodeKind::Waves { .. } => &["Fac"],
            NodeKind::Ramp { .. } | NodeKind::Mix { .. } | NodeKind::Invert | NodeKind::Rgb { .. } | NodeKind::HueSat { .. } => &["Color"],
            NodeKind::Gradient { .. } => &["Fac"],
            NodeKind::NormalMap { .. } => &["Normal"],
            NodeKind::Math { .. } | NodeKind::Value { .. } => &["Value"],
        }
    }
    /// The kinds offered in the Add menu (with default settings).
    pub fn catalog() -> Vec<NodeKind> {
        vec![
            NodeKind::Coords { scale: 1.0, rotate: 0.0, object: false },
            NodeKind::Coords { scale: 1.0, rotate: 0.0, object: true },
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
            NodeKind::Gradient { mode: GradMode::Linear },
            NodeKind::HueSat { hue: 0.0, saturation: 1.0, value: 1.0 },
            NodeKind::NormalMap { strength: 1.0 },
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
    /// tangent-space normal (from a Normal Map node)
    pub normal: Option<Vec3>,
}

// ---- patterns ----

fn hash2(x: i32, y: i32, s: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ s.wrapping_mul(0xcb1a_b31f);
    h = (h ^ (h >> 13)).wrapping_mul(0x5bd1_e995);
    ((h ^ (h >> 15)) & 0x00ff_ffff) as f32 / 16_777_215.0
}


fn hash3(x: i32, y: i32, z: i32, s: u32) -> f32 {
    hash2(x ^ z.wrapping_mul(0x27d4_eb2d), y.wrapping_add(z.wrapping_mul(0x1656_67b1u32 as i32)), s)
}

fn vnoise3(p: Vec3, seed: u32) -> f32 {
    let i = p.floor();
    let f = p - i;
    let u = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let (x, y, z) = (i.x as i32, i.y as i32, i.z as i32);
    let c = |dx, dy, dz| hash3(x + dx, y + dy, z + dz, seed);
    let l = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = l(c(0, 0, 0), c(1, 0, 0), u.x);
    let x10 = l(c(0, 1, 0), c(1, 1, 0), u.x);
    let x01 = l(c(0, 0, 1), c(1, 0, 1), u.x);
    let x11 = l(c(0, 1, 1), c(1, 1, 1), u.x);
    l(l(x00, x10, u.y), l(x01, x11, u.y), u.z)
}

/// Fractal noise in 3D (2D inputs use z = 0).
pub fn fbm3(p: Vec3, octaves: u32, seed: u32) -> f32 {
    let (mut s, mut amp, mut norm, mut q) = (0.0, 0.5, 0.0, p);
    for o in 0..octaves.clamp(1, 8) {
        s += vnoise3(q, seed + o) * amp;
        norm += amp;
        amp *= 0.5;
        q = q * 2.03 + Vec3::new(17.3, 9.1, 4.7);
    }
    s / norm
}

fn voronoi3(p: Vec3, randomness: f32) -> (f32, f32, f32) {
    let (ix, iy, iz) = (p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32);
    let (mut d1, mut d2, mut id) = (9.0f32, 9.0f32, 0.0);
    let r = randomness.clamp(0.0, 1.0);
    // 2D inputs only need one z layer
    let zr = if p.z == 0.0 { 0..=0 } else { -1..=1 };
    for dz in zr {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (cx, cy, cz) = (ix + dx, iy + dy, iz + dz);
                let jz = if p.z == 0.0 { 0.0 } else { cz as f32 + 0.5 + (hash3(cx, cy, cz, 41) - 0.5) * r };
                let pt = Vec3::new(cx as f32 + 0.5 + (hash3(cx, cy, cz, 11) - 0.5) * r, cy as f32 + 0.5 + (hash3(cx, cy, cz, 23) - 0.5) * r, jz);
                let d = pt.distance(p);
                if d < d1 {
                    d2 = d1;
                    d1 = d;
                    id = hash3(cx, cy, cz, 37);
                } else if d < d2 {
                    d2 = d;
                }
            }
        }
    }
    (d1, id, d2 - d1)
}

fn hsv_adjust(c: Vec3, hue: f32, sat: f32, val: f32) -> Vec3 {
    let (r, g, b) = (c.x.max(0.0), c.y.max(0.0), c.z.max(0.0));
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    let d = mx - mn;
    let mut h = if d < 1e-6 {
        0.0
    } else if mx == r {
        ((g - b) / d).rem_euclid(6.0)
    } else if mx == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    } / 6.0;
    let s = if mx > 1e-6 { d / mx } else { 0.0 };
    h = (h + hue).rem_euclid(1.0);
    let s = (s * sat).clamp(0.0, 1.0);
    let v = mx * val;
    let k = |n: f32| {
        let k = (n + h * 6.0).rem_euclid(6.0);
        v - v * s * k.min(4.0 - k).clamp(0.0, 1.0)
    };
    Vec3::new(k(5.0), k(3.0), k(1.0))
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
    /// Set the colour of the graph's first RGB node.
    pub fn set_rgb(&mut self, c: [f32; 3]) {
        if let Some(n) = self.nodes.iter_mut().find(|n| matches!(n.kind, NodeKind::Rgb { .. })) {
            n.kind = NodeKind::Rgb { color: c };
        }
    }
    /// Uses the object's 3D coordinates somewhere (so a UV bake can't show it).
    pub fn uses_object_coords(&self) -> bool {
        self.nodes.iter().any(|n| matches!(n.kind, NodeKind::Coords { object: true, .. }))
    }
    pub fn drives(&self, input: usize) -> bool {
        self.output().is_some_and(|o| self.input_link(o.id, input).is_some())
    }

    /// Evaluate at `uv`. `tex` samples a scene texture (id, uv) → sRGB rgba.
    pub fn eval(&self, uv: Vec2, pos: Vec3, tex: &dyn Fn(u32, Vec2) -> Option<[f32; 4]>) -> Shade {
        let Some(out) = self.output() else { return Shade::default() };
        let mut memo: [Option<[Val; 3]>; MAX_NODES] = [None; MAX_NODES];
        let bump = match out.kind {
            NodeKind::Output { bump } => bump,
            _ => 1.0,
        };
        let mut get = |i: usize| -> Option<Val> {
            let l = self.input_link(out.id, i)?;
            Some(self.value(l.from.0, l.from.1, uv, pos, tex, &mut memo, 0))
        };
        Shade {
            color: get(0).map(|v| v.c()),
            roughness: get(1).map(|v| v.f().clamp(0.0, 1.0)),
            metallic: get(2).map(|v| v.f().clamp(0.0, 1.0)),
            emission: get(3).map(|v| v.f().max(0.0)),
            height: get(4).map(|v| v.f()),
            normal: get(5).map(|v| v.c().normalize_or(Vec3::Z)),
            bump,
        }
    }

    /// Only the height (for bump finite differences).
    pub fn height(&self, uv: Vec2, pos: Vec3, tex: &dyn Fn(u32, Vec2) -> Option<[f32; 4]>) -> f32 {
        let Some(out) = self.output() else { return 0.0 };
        let Some(l) = self.input_link(out.id, 4) else { return 0.0 };
        let mut memo: [Option<[Val; 3]>; MAX_NODES] = [None; MAX_NODES];
        self.value(l.from.0, l.from.1, uv, pos, tex, &mut memo, 0).f()
    }

    fn value(&self, id: u32, out: usize, uv: Vec2, pos: Vec3, tex: &dyn Fn(u32, Vec2) -> Option<[f32; 4]>, memo: &mut [Option<[Val; 3]>; MAX_NODES], depth: usize) -> Val {
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
            Some(self.value(l.from.0, l.from.1, uv, pos, tex, memo, depth + 1))
        };
        // pattern vectors are 3D (UVs have z = 0)
        let vec3_in = |v: Option<Val>| -> Vec3 {
            match v {
                Some(Val::C(c)) => c,
                Some(Val::F(f)) => Vec3::new(uv.x + f, uv.y + f, 0.0),
                None => Vec3::new(uv.x, uv.y, 0.0),
            }
        };
        let vec_in = |v: Option<Val>| -> Vec2 {
            let c = vec3_in(v);
            Vec2::new(c.x, c.y)
        };
        let z = Val::F(0.0);
        let r: [Val; 3] = match &n.kind {
            NodeKind::Output { .. } => [z; 3],
            NodeKind::Coords { scale, rotate, object } => {
                let a = rotate.to_radians();
                let (s, c) = a.sin_cos();
                let v = if *object { pos } else { Vec3::new(uv.x, uv.y, 0.0) };
                let p = Vec3::new(v.x * c - v.y * s, v.x * s + v.y * c, v.z) * *scale;
                [Val::C(p), z, z]
            }
            NodeKind::Image { texture } => {
                let p = vec_in(inp(0, memo));
                let c = tex(*texture, p).unwrap_or([1.0, 0.0, 1.0, 1.0]);
                [Val::C(Vec3::new(c[0], c[1], c[2])), Val::F(c[3]), z]
            }
            NodeKind::Noise { scale, detail, distortion } => {
                let mut p = vec3_in(inp(0, memo)) * *scale;
                let flat = p.z == 0.0;
                if *distortion != 0.0 {
                    let dz = if flat { 0.0 } else { fbm3(p + Vec3::splat(8.1), 2, 63) - 0.5 };
                    p += Vec3::new(fbm3(p + Vec3::splat(5.2), 2, 91) - 0.5, fbm3(p + Vec3::splat(1.3), 2, 77) - 0.5, dz) * *distortion;
                }
                let f = fbm3(p, *detail, 3);
                let col = Vec3::new(f, fbm3(p + Vec3::splat(31.7), *detail, 5), fbm3(p + Vec3::splat(-12.9), *detail, 7));
                [Val::F(f), Val::C(col), z]
            }
            NodeKind::Voronoi { scale, randomness } => {
                let p3 = vec3_in(inp(0, memo)) * *scale;
                let (d, idh, edge) = if p3.z == 0.0 { voronoi(Vec2::new(p3.x, p3.y), *randomness) } else { voronoi3(p3, *randomness) };
                let col = Vec3::new(idh, (idh * 7.31).fract(), (idh * 13.17).fract());
                [Val::F(d), Val::C(col), Val::F(edge)]
            }
            NodeKind::Checker { scale } => {
                let p = vec3_in(inp(0, memo)) * *scale;
                let k = (p.x.floor() as i64 + p.y.floor() as i64 + p.z.floor() as i64).rem_euclid(2) as f32;
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
                let p = vec3_in(inp(0, memo)) * *scale;
                let d = if *distortion != 0.0 { (fbm3(p, 3, 13) - 0.5) * *distortion * 2.0 } else { 0.0 };
                let f = 0.5 + 0.5 * ((p.x + p.y + p.z) * std::f32::consts::TAU + d * std::f32::consts::TAU).sin();
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
            NodeKind::Gradient { mode } => {
                let p = vec3_in(inp(0, memo));
                let f = match mode {
                    GradMode::Linear => p.x,
                    GradMode::Radial => p.y.atan2(p.x) / std::f32::consts::TAU + 0.5,
                    GradMode::Spherical => 1.0 - p.length(),
                };
                [Val::F(f.clamp(0.0, 1.0)), z, z]
            }
            NodeKind::HueSat { hue, saturation, value } => {
                let c = inp(0, memo).map(|v| v.c()).unwrap_or(Vec3::splat(0.5));
                [Val::C(hsv_adjust(c, *hue, *saturation, *value)), z, z]
            }
            NodeKind::NormalMap { strength } => {
                let c = inp(0, memo).map(|v| v.c()).unwrap_or(Vec3::new(0.5, 0.5, 1.0));
                let n = crate::textures::texel_normal([c.x, c.y, c.z, 1.0], *strength);
                [Val::C(n), z, z]
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
        let co = |g: &mut NodeGraph, s: f32| g.add(NodeKind::Coords { scale: s, rotate: 0.0, object: false }, [0.0, 120.0]);
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
        } else if low.contains("strand") {
            // hair: darker at the root, lighter at the tip (U runs along each strand)
            title = "Hair Strands";
            let c = co(&mut g, 1.0);
            let gr = g.add(NodeKind::Gradient { mode: GradMode::Linear }, [160.0, 100.0]);
            let r = g.add(NodeKind::Ramp { stops: vec![(0.0, [0.35, 0.35, 0.35]), (0.6, [0.85, 0.85, 0.85]), (1.0, [1.0, 1.0, 1.0])] }, [330.0, 100.0]);
            let rgb = g.add(NodeKind::Rgb { color: [0.4, 0.25, 0.12] }, [330.0, 240.0]);
            let mx = g.add(NodeKind::Mix { mode: MixMode::Multiply, fac: 1.0, a: [1.0; 3], b: [1.0; 3] }, [480.0, 100.0]);
            let rough = g.add(NodeKind::Value { value: 0.45 }, [480.0, 240.0]);
            g.connect((c, 0), (gr, 0));
            g.connect((gr, 0), (r, 0));
            g.connect((r, 0), (mx, 1));
            g.connect((rgb, 0), (mx, 2));
            g.connect((mx, 0), (out, 0));
            g.connect((rough, 0), (out, 1));
        } else if low.contains("granite") || low.contains("solid") {
            // solid 3D texture on Object coordinates: no UV seams, works on any mesh
            title = "Granite";
            let c = g.add(NodeKind::Coords { scale: 1.0, rotate: 0.0, object: true }, [0.0, 120.0]);
            let n = g.add(NodeKind::Noise { scale: 14.0, detail: 5, distortion: 0.4 }, [160.0, 60.0]);
            let r = g.add(NodeKind::Ramp { stops: vec![(0.3, [0.12, 0.11, 0.11]), (0.45, [0.62, 0.6, 0.58]), (0.62, [0.85, 0.72, 0.66]), (0.75, [0.95, 0.94, 0.92])] }, [330.0, 60.0]);
            let hs = g.add(NodeKind::HueSat { hue: 0.0, saturation: 0.8, value: 1.0 }, [480.0, 60.0]);
            let vo = g.add(NodeKind::Voronoi { scale: 30.0, randomness: 1.0 }, [160.0, 240.0]);
            let rough = g.add(NodeKind::Value { value: 0.35 }, [480.0, 200.0]);
            g.connect((c, 0), (n, 0));
            g.connect((c, 0), (vo, 0));
            g.connect((n, 0), (r, 0));
            g.connect((r, 0), (hs, 0));
            g.connect((hs, 0), (out, 0));
            g.connect((rough, 0), (out, 1));
            g.connect((vo, 0), (out, 4));
            if let Some(o) = g.nodes.iter_mut().find(|n| n.id == out) {
                o.kind = NodeKind::Output { bump: 0.15 };
            }
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
pub fn param_rows(k: &NodeKind) -> usize {
    match k {
        NodeKind::Output { .. } | NodeKind::Checker { .. } | NodeKind::Image { .. } | NodeKind::Rgb { .. } | NodeKind::Value { .. } => 1,
        NodeKind::Voronoi { .. } | NodeKind::Bricks { .. } | NodeKind::Waves { .. } => 2,
        NodeKind::Coords { .. } | NodeKind::Noise { .. } | NodeKind::Math { .. } => 3,
        NodeKind::Mix { .. } => 4,
        NodeKind::Ramp { stops } => 2 + stops.len(),
        NodeKind::Invert => 0,
        NodeKind::Gradient { .. } | NodeKind::NormalMap { .. } => 1,
        NodeKind::HueSat { .. } => 3,
    }
}

pub fn node_height_estimate(k: &NodeKind) -> f32 {
    let params = param_rows(k);
    22.0 + k.outputs().len() as f32 * 18.0 + params as f32 * 22.0 + k.inputs().len() as f32 * 18.0 + 6.0
}

pub const PRESETS: &[&str] = &["Rusty Metal", "Wood", "Marble", "Cobblestone", "Lava", "Bricks", "Tiles", "Granite"];

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
    let mut tnorm: Vec<Vec3> = vec![];
    let drives_n = g.drives(5);
    let mut col = Vec::with_capacity(n * n);
    let mut mr = Vec::with_capacity(n * n);
    let (drives_mr, bump) = (g.drives(1) || g.drives(2), g.has_bump());
    for y in 0..n {
        for x in 0..n {
            let uv = Vec2::new((x as f32 + 0.5) / n as f32, (y as f32 + 0.5) / n as f32);
            // Object coordinates have no UV mapping: bake a planar slice (z = 0) through the object
            let s = g.eval(uv, Vec3::new(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0), tex);
            let c = s.color.unwrap_or(Vec3::from(base.color));
            col.push(c);
            mr.push([255, q(s.roughness.unwrap_or(1.0)), q(s.metallic.unwrap_or(1.0)), 255]);
            heights[y * n + x] = s.height.unwrap_or(0.0);
            if drives_n {
                tnorm.push(s.normal.unwrap_or(Vec3::Z));
            }
        }
    }
    let strength = g.output().map(|o| if let NodeKind::Output { bump } = o.kind { bump } else { 1.0 }).unwrap_or(1.0);
    let normal = (bump || drives_n).then(|| {
        let hi = |x: i64, y: i64| heights[(y.rem_euclid(n as i64) as usize) * n + x.rem_euclid(n as i64) as usize];
        let k = strength * n as f32 / 64.0;
        let mut rgba = Vec::with_capacity(n * n);
        for y in 0..n as i64 {
            for x in 0..n as i64 {
                let dx = (hi(x + 1, y) - hi(x - 1, y)) * 0.5;
                let dy = (hi(x, y + 1) - hi(x, y - 1)) * 0.5;
                let mut nn = Vec3::new(-dx * k, dy * k, 1.0).normalize();
                if let Some(t) = tnorm.get(y as usize * n + x as usize) {
                    // whiteout blend of the bump and the Normal Map node
                    nn = Vec3::new(nn.x + t.x, nn.y + t.y, nn.z * t.z.max(0.0)).normalize_or(Vec3::Z);
                }
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

/// Bake an Object-coordinate graph's colour into a mesh's own UV layout: every texel evaluates
/// the graph at the 3D point it covers (like Blender's bake), so the raster previews match the
/// path tracer without UV seams. Edges are dilated a few texels so filtering doesn't bleed.
pub fn bake_object(g: &NodeGraph, base: crate::scene::Material, mesh: &crate::mesh::Mesh, size: usize, tex: &(dyn Fn(u32, Vec2) -> Option<[f32; 4]> + Sync)) -> crate::textures::TexImage {
    let n = size.max(4);
    // texel → (uv, local position), first triangle wins
    let mut at: Vec<Option<(Vec2, Vec3)>> = vec![None; n * n];
    for (ti, t) in mesh.tris.iter().enumerate() {
        let Some(uv) = mesh.tri_uv.get(ti) else { continue };
        let q = uv.map(|c| Vec2::new(c[0] * n as f32, c[1] * n as f32));
        let p = t.map(|i| mesh.positions[i as usize]);
        let (lo, hi) = (q[0].min(q[1]).min(q[2]), q[0].max(q[1]).max(q[2]));
        let area = (q[1] - q[0]).perp_dot(q[2] - q[0]);
        if area.abs() < 1e-9 || (hi - lo).max_element() > 4.0 * n as f32 {
            continue;
        }
        // conservative: include texels whose centre is within half a texel of the triangle
        let pad = 0.75 / area.abs().sqrt().max(1.0);
        for y in (lo.y.floor() as i64 - 1)..=(hi.y.ceil() as i64) {
            for x in (lo.x.floor() as i64 - 1)..=(hi.x.ceil() as i64) {
                let c = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                let w1 = (c - q[0]).perp_dot(q[2] - q[0]) / area;
                let w2 = (q[1] - q[0]).perp_dot(c - q[0]) / area;
                let w0 = 1.0 - w1 - w2;
                if w0 < -pad || w1 < -pad || w2 < -pad {
                    continue;
                }
                let idx = y.rem_euclid(n as i64) as usize * n + x.rem_euclid(n as i64) as usize;
                if at[idx].is_none() {
                    let (w0, w1, w2) = (w0.clamp(0.0, 1.0), w1.clamp(0.0, 1.0), w2.clamp(0.0, 1.0));
                    let ws = (w0 + w1 + w2).max(1e-6);
                    let pos = (p[0] * w0 + p[1] * w1 + p[2] * w2) / ws;
                    at[idx] = Some((c / n as f32, pos));
                }
            }
        }
    }
    let fallback = Vec3::from(base.color);
    let mut col: Vec<Option<Vec3>> = vec![None; n * n];
    let threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(4).clamp(1, 16);
    let chunk = (n * n).div_ceil(threads);
    std::thread::scope(|sc| {
        for (src, dst) in at.chunks(chunk).zip(col.chunks_mut(chunk)) {
            sc.spawn(move || {
                for (a, d) in src.iter().zip(dst.iter_mut()) {
                    if let Some((uv, pos)) = a {
                        *d = Some(g.eval(*uv, *pos, tex).color.unwrap_or(fallback));
                    }
                }
            });
        }
    });
    // dilate into the gutters
    for _ in 0..4 {
        let prev = col.clone();
        for y in 0..n {
            for x in 0..n {
                if prev[y * n + x].is_some() {
                    continue;
                }
                let (mut sum, mut k) = (Vec3::ZERO, 0);
                for (dx, dy) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                    let (xx, yy) = ((x as i64 + dx).rem_euclid(n as i64) as usize, (y as i64 + dy).rem_euclid(n as i64) as usize);
                    if let Some(c) = prev[yy * n + xx] {
                        sum += c;
                        k += 1;
                    }
                }
                if k > 0 {
                    col[y * n + x] = Some(sum / k as f32);
                }
            }
        }
    }
    let q = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
    crate::textures::TexImage { w: n, h: n, rgba: col.iter().map(|c| c.unwrap_or(fallback)).map(|c| [q(c.x), q(c.y), q(c.z), 255]).collect() }
}

/// A small lit sphere showing the graph's real result (colour, roughness, metallic, emission and
/// bumps) for the Material tab; RGBA8, transparent around the ball. Object coordinates are the
/// unit sphere, UVs a latitude/longitude mapping.
pub fn preview_ball(g: &NodeGraph, base: crate::scene::Material, size: usize, tex: &dyn Fn(u32, Vec2) -> Option<[f32; 4]>) -> Vec<u8> {
    let n = size.max(8);
    let mut out = vec![0u8; n * n * 4];
    let light = Vec3::new(-0.55, -0.7, 0.65).normalize();
    // towards the viewer (who looks along +Y)
    let view = Vec3::new(0.0, -1.0, 0.0);
    let lin = |c: Vec3| Vec3::new(c.x.max(0.0).powf(2.2), c.y.max(0.0).powf(2.2), c.z.max(0.0).powf(2.2));
    let uv_of = |p: Vec3| Vec2::new(p.y.atan2(p.x) / std::f32::consts::TAU + 0.5, p.z.clamp(-1.0, 1.0).acos() / std::f32::consts::PI);
    for y in 0..n {
        for x in 0..n {
            let sx = (x as f32 + 0.5) / n as f32 * 2.0 - 1.0;
            let sy = 1.0 - (y as f32 + 0.5) / n as f32 * 2.0;
            let r = (sx * sx + sy * sy).sqrt();
            let cover = ((1.0 - r) * n as f32 * 0.5 + 0.5).clamp(0.0, 1.0);
            if cover <= 0.0 {
                continue;
            }
            let rr = r.min(0.999);
            let (sx, sy) = if r > 0.999 { (sx / r * rr, sy / r * rr) } else { (sx, sy) };
            let mut nrm = Vec3::new(sx, -(1.0 - sx * sx - sy * sy).max(0.0).sqrt(), sy).normalize();
            let p = nrm;
            let uv = uv_of(p);
            let s = g.eval(uv, p, tex);
            if let Some(h0) = s.height {
                // bump over the sphere's surface frame
                let tu = Vec3::Z.cross(nrm).normalize_or(Vec3::X);
                let tv = nrm.cross(tu);
                let e = 0.004;
                let hu = g.height(uv_of((p + tu * e).normalize()), p + tu * e, tex);
                let hv = g.height(uv_of((p + tv * e).normalize()), p + tv * e, tex);
                let grad = tu * ((hu - h0) / e) + tv * ((hv - h0) / e);
                nrm = (nrm - grad * (s.bump * 0.05)).normalize_or(nrm);
            }
            let alb = lin(s.color.unwrap_or(Vec3::from(base.color)));
            let rough = s.roughness.unwrap_or(base.roughness).clamp(0.04, 1.0);
            let metal = s.metallic.unwrap_or(base.metallic).clamp(0.0, 1.0);
            let emit = s.emission.unwrap_or(base.emission);
            let ndl = nrm.dot(light).max(0.0);
            let h = (light + view).normalize();
            let shin = 2.0 / (rough * rough) + 2.0;
            let f0 = Vec3::splat(0.04).lerp(alb, metal);
            let nv = nrm.dot(view).max(0.0);
            let fres = f0 + (Vec3::ONE - f0) * (1.0 - nv).powf(5.0);
            let spec = fres * nrm.dot(h).max(0.0).powf(shin) * (shin + 8.0) / 25.0 * ndl;
            let env = fres * (1.0 - rough * 0.7) * (0.25 + 0.2 * nrm.z);
            let c = alb * (1.0 - metal) * (0.12 + 0.1 * (0.5 + 0.5 * nrm.z) + ndl * 1.1) + spec + env + alb * emit;
            let c = Vec3::new(c.x.max(0.0).powf(1.0 / 2.2), c.y.max(0.0).powf(1.0 / 2.2), c.z.max(0.0).powf(1.0 / 2.2));
            let i = (y * n + x) * 4;
            let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            out[i..i + 4].copy_from_slice(&[q(c.x), q(c.y), q(c.z), q(cover)]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn object_coords_gradient_huesat_normalmap() {
        let none = |_: u32, _: Vec2| None;
        // object coords: same UV, different 3D points give different results; same point, any UV, same
        let g = NodeGraph::preset(1, "granite").unwrap();
        assert!(g.uses_object_coords());
        let a = g.eval(Vec2::ZERO, Vec3::new(0.1, 0.2, 0.3), &none).color.unwrap();
        let b = g.eval(Vec2::ZERO, Vec3::new(0.7, -0.4, 0.9), &none).color.unwrap();
        let a2 = g.eval(Vec2::new(0.9, 0.9), Vec3::new(0.1, 0.2, 0.3), &none).color.unwrap();
        assert!((a - b).length() > 1e-3 && (a - a2).length() < 1e-6);
        // the z axis matters (true 3D noise)
        let c = g.eval(Vec2::ZERO, Vec3::new(0.1, 0.2, 0.8), &none).color.unwrap();
        assert!((a - c).length() > 1e-3);
        // gradient (linear over x) → output colour
        let mut g = NodeGraph::basic(2, "grad", [1.0; 3]);
        let out = g.output().unwrap().id;
        let co = g.add(NodeKind::Coords { scale: 1.0, rotate: 0.0, object: false }, [0.0, 0.0]);
        let gr = g.add(NodeKind::Gradient { mode: GradMode::Linear }, [0.0, 0.0]);
        g.connect((co, 0), (gr, 0));
        g.connect((gr, 0), (out, 0));
        let l = g.eval(Vec2::new(0.2, 0.5), Vec3::ZERO, &none).color.unwrap();
        let r = g.eval(Vec2::new(0.8, 0.5), Vec3::ZERO, &none).color.unwrap();
        assert!((l.x - 0.2).abs() < 0.02 && (r.x - 0.8).abs() < 0.02, "{l} {r}");
        // hue/sat: saturation 0 → grey
        let hs = hsv_adjust(Vec3::new(1.0, 0.2, 0.1), 0.0, 0.0, 1.0);
        assert!((hs.x - hs.y).abs() < 1e-4 && (hs.y - hs.z).abs() < 1e-4);
        let rot = hsv_adjust(Vec3::new(1.0, 0.0, 0.0), 1.0 / 3.0, 1.0, 1.0);
        assert!(rot.y > 0.9 && rot.x < 0.1, "red + 120° = green, got {rot}");
        // normal map node → Shade.normal and the baked normal texture
        let mut g = NodeGraph::basic(3, "nm", [1.0; 3]);
        let out = g.output().unwrap().id;
        let rgb = g.add(NodeKind::Rgb { color: [0.8, 0.5, 0.9] }, [0.0, 0.0]);
        let nm = g.add(NodeKind::NormalMap { strength: 1.0 }, [0.0, 0.0]);
        g.connect((rgb, 0), (nm, 0));
        assert!(g.connect((nm, 0), (out, 5)));
        let s = g.eval(Vec2::ZERO, Vec3::ZERO, &none);
        let n = s.normal.unwrap();
        assert!(n.x > 0.3 && n.z > 0.5 && (n.length() - 1.0).abs() < 1e-3, "{n}");
        let bk = bake(&g, crate::scene::Material::default(), 8, false, &none);
        let px = bk.normal.expect("normal baked").rgba[0];
        assert!(px[0] > 160, "tilted toward +x: {px:?}");
    }

    #[test]
    fn presets_evaluate_and_bake() {
        let none = |_: u32, _: Vec2| None;
        for p in PRESETS {
            let g = NodeGraph::preset(1, p).unwrap_or_else(|| panic!("{p}"));
            assert_eq!(g.name, *p);
            let a = g.eval(Vec2::new(0.1, 0.2), Vec3::ZERO, &none);
            let b = g.eval(Vec2::new(0.6, 0.7), Vec3::ZERO, &none);
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
        assert!((g.eval(Vec2::ZERO, Vec3::ZERO, &none).roughness.unwrap() - 0.3).abs() < 1e-6);
        g.remove_node(m);
        assert!(g.eval(Vec2::ZERO, Vec3::ZERO, &none).roughness.is_none());
    }
}
