//! Agent loop, "check" stage: a deterministic scene inspector that looks at what an AI build
//! changed and reports likely mistakes (floating or sunken objects, hidden or duplicated
//! objects, failed commands). A reviewer model then decides which issues are real and
//! sends fix commands.

use crate::editor::Command;
use crate::scene::{MeshCache, Object, Scene};
use glam::Vec3;
use serde_json::json;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub lo: Vec3,
    pub hi: Vec3,
}

impl Aabb {
    pub fn size(&self) -> Vec3 {
        self.hi - self.lo
    }
    pub fn volume(&self) -> f32 {
        let s = self.size().max(Vec3::splat(1e-4));
        s.x * s.y * s.z
    }
    fn xy_overlap(&self, o: &Aabb, shrink: f32) -> bool {
        self.lo.x + shrink < o.hi.x && o.lo.x + shrink < self.hi.x && self.lo.y + shrink < o.hi.y && o.lo.y + shrink < self.hi.y
    }
    fn overlap_volume(&self, o: &Aabb) -> f32 {
        let lo = self.lo.max(o.lo);
        let hi = self.hi.min(o.hi);
        let d = (hi - lo).max(Vec3::ZERO);
        d.x * d.y * d.z
    }
}

/// World-space bounding box of an object (modifiers/instances included).
pub fn world_bounds(o: &Object, cache: &mut MeshCache) -> Aabb {
    if !o.is_mesh() {
        let p = o.transform.location;
        return Aabb { lo: p - Vec3::splat(0.1), hi: p + Vec3::splat(0.1) };
    }
    let mesh = cache.for_object(o);
    let mut lo = Vec3::splat(f32::MAX);
    let mut hi = Vec3::splat(f32::MIN);
    let base = o.transform.matrix();
    for inst in o.instance_matrices() {
        let m = base * inst;
        for p in &mesh.positions {
            let w = m.transform_point3(*p);
            lo = lo.min(w);
            hi = hi.max(w);
        }
    }
    if mesh.positions.is_empty() {
        lo = o.transform.location;
        hi = lo;
    }
    Aabb { lo, hi }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Issue {
    pub object: String,
    pub text: String,
    /// A safe automatic fix, used if no reviewer model is available.
    pub auto_fix: Option<Command>,
}

/// Is this object a ground/floor slab (wide and thin)?
fn is_ground(o: &Object, b: &Aabb) -> bool {
    let s = b.size();
    let n = o.name.to_lowercase();
    (n.contains("ground") || n.contains("floor") || n.contains("terrain")) || (s.z < 0.15 && s.x > 3.0 && s.y > 3.0)
}

/// Objects that were added or changed between `before` and `after`.
pub fn changed_objects(before: &Scene, after: &Scene) -> Vec<u64> {
    after
        .objects
        .iter()
        .filter(|o| match before.get(o.id) {
            None => true,
            Some(b) => b.transform != o.transform || b.mesh != o.mesh || b.primitive != o.primitive || b.modifiers != o.modifiers || b.visible != o.visible,
        })
        .map(|o| o.id)
        .collect()
}

/// Inspect the objects in `subjects` (usually what the last AI batch touched).
pub fn inspect(scene: &Scene, cache: &mut MeshCache, subjects: &[u64], failed: &[String]) -> Vec<Issue> {
    let mut issues: Vec<Issue> = failed.iter().map(|f| Issue { object: String::new(), text: format!("command failed: {}", f.trim_start_matches("✖ ")), auto_fix: None }).collect();
    let objs: Vec<(&Object, Aabb)> = scene.objects.iter().filter(|o| o.visible && o.is_mesh()).map(|o| (o, world_bounds(o, cache))).collect();
    let ground_top = objs.iter().filter(|(o, b)| is_ground(o, b)).map(|(_, b)| b.hi.z).fold(f32::MIN, f32::max);
    let ground = if ground_top > f32::MIN { ground_top } else { 0.0 };
    let has_ground_obj = ground_top > f32::MIN;

    for (o, b) in objs.iter().filter(|(o, _)| subjects.contains(&o.id)) {
        if is_ground(o, b) {
            continue;
        }
        let name = o.name.clone();
        // sunk into the ground
        if b.lo.z < ground - 0.05 && b.hi.z > ground - 0.05 - 10.0 {
            let d = ground - b.lo.z;
            let mut fix = Command::new("move");
            fix.target = Some(name.clone());
            fix.vector = Some(json!([0.0, 0.0, d]));
            issues.push(Issue { object: name.clone(), text: format!("\"{name}\" is {d:.2} units below the ground (bottom z={:.2}, ground z={ground:.2})", b.lo.z), auto_fix: Some(fix) });
            continue;
        }
        // floating: nothing underneath within the footprint
        let support = objs
            .iter()
            .filter(|(p, pb)| p.id != o.id && pb.xy_overlap(b, 0.02) && pb.hi.z <= b.lo.z + 0.25 && pb.hi.z > b.lo.z - (b.size().z * 0.9).max(0.1))
            .map(|(_, pb)| pb.hi.z.min(b.lo.z))
            .fold(if has_ground_obj || ground == 0.0 { ground } else { f32::MIN }, f32::max);
        let resting_on_other = objs.iter().any(|(p, pb)| p.id != o.id && pb.xy_overlap(b, 0.02) && pb.lo.z < b.lo.z + 0.05 && pb.hi.z > b.lo.z - 0.05);
        let gap = b.lo.z - support;
        if gap > 0.1 && !resting_on_other {
            let mut fix = Command::new("move");
            fix.target = Some(name.clone());
            fix.vector = Some(json!([0.0, 0.0, -gap]));
            issues.push(Issue { object: name.clone(), text: format!("\"{name}\" floats {gap:.2} units above whatever is below it (bottom z={:.2}, support z={support:.2})", b.lo.z), auto_fix: Some(fix) });
        }
        // hidden inside another object / exact duplicates
        for (p, pb) in &objs {
            if p.id == o.id || is_ground(p, pb) {
                continue;
            }
            let ov = b.overlap_volume(pb);
            if (o.transform.location - p.transform.location).length() < 0.01 && o.primitive == p.primitive && (o.transform.scale - p.transform.scale).length() < 0.01 && o.mesh.is_none() && p.mesh.is_none() {
                let mut fix = Command::new("delete");
                fix.target = Some(name.clone());
                issues.push(Issue { object: name.clone(), text: format!("\"{name}\" is an exact duplicate of \"{}\" in the same place", p.name), auto_fix: Some(fix) });
                break;
            }
            if ov > 0.92 * b.volume() && b.volume() < pb.volume() * 0.9 && mostly_inside(o, p, cache) {
                issues.push(Issue { object: name.clone(), text: format!("\"{name}\" is completely inside \"{}\" and can't be seen", p.name), auto_fix: None });
                break;
            }
        }
    }
    issues.truncate(12);
    issues
}

/// True when (nearly) every vertex of `inner` lies inside the closed mesh of `outer`.
/// (Bounding boxes alone can't tell a scarf around a neck from a nose buried in a head.)
fn mostly_inside(inner: &Object, outer: &Object, cache: &mut MeshCache) -> bool {
    let om = cache.for_object(outer);
    let ob = outer.transform.matrix();
    let mut tris: Vec<[Vec3; 3]> = vec![];
    for inst in outer.instance_matrices() {
        let m = ob * inst;
        tris.extend(om.tris.iter().map(|t| t.map(|i| m.transform_point3(om.positions[i as usize]))));
    }
    let im = cache.for_object(inner);
    let ib = inner.transform.matrix();
    let step = (im.positions.len() / 64).max(1);
    let pts: Vec<Vec3> = im.positions.iter().step_by(step).map(|p| ib.transform_point3(*p)).collect();
    if pts.is_empty() || tris.is_empty() {
        return false;
    }
    let inside = pts.iter().filter(|p| point_in_mesh(**p, &tris)).count();
    inside as f32 >= pts.len() as f32 * 0.95
}

/// Ray-parity point-in-mesh test (ray along a slightly skewed +X to avoid edge hits).
fn point_in_mesh(p: Vec3, tris: &[[Vec3; 3]]) -> bool {
    let d = Vec3::new(1.0, 0.000_123, 0.000_457).normalize();
    let mut hits = 0;
    for [a, b, c] in tris {
        let e1 = *b - *a;
        let e2 = *c - *a;
        let h = d.cross(e2);
        let det = e1.dot(h);
        if det.abs() < 1e-9 {
            continue;
        }
        let f = 1.0 / det;
        let s = p - *a;
        let u = f * s.dot(h);
        if !(0.0..=1.0).contains(&u) {
            continue;
        }
        let q = s.cross(e1);
        let v = f * d.dot(q);
        if v < 0.0 || u + v > 1.0 {
            continue;
        }
        if f * e2.dot(q) > 1e-6 {
            hits += 1;
        }
    }
    hits % 2 == 1
}

/// Scene description with world bounds, for the reviewer model.
pub fn bounds_report(scene: &Scene, cache: &mut MeshCache) -> String {
    let mut s = String::new();
    for o in scene.objects.iter().filter(|o| o.visible) {
        let b = world_bounds(o, cache);
        s += &format!(
            "- \"{}\" ({}): location=[{:.2},{:.2},{:.2}] scale=[{:.2},{:.2},{:.2}] bounds min=[{:.2},{:.2},{:.2}] max=[{:.2},{:.2},{:.2}]\n",
            o.name,
            o.type_label(),
            o.transform.location.x,
            o.transform.location.y,
            o.transform.location.z,
            o.transform.scale.x,
            o.transform.scale.y,
            o.transform.scale.z,
            b.lo.x,
            b.lo.y,
            b.lo.z,
            b.hi.x,
            b.hi.y,
            b.hi.z
        );
    }
    s
}

pub fn review_prompt(request: &str, issues: &[Issue], bounds: &str) -> String {
    let list: Vec<String> = issues.iter().map(|i| format!("- {}", i.text)).collect();
    format!(
        r#"You are the QUALITY CHECKER of "Rust Blender" (Z is up). The user asked:
"{request}"
The builder model has finished. Automatic checks flagged these possible problems:
{issues}

Decide which are REAL mistakes. Leave things alone that the user plausibly intended (e.g. objects
that should hover: rings above something, planets, birds, lamps, clouds, anything "above"/"floating").
Fix real mistakes with commands, preferring set_transform with an absolute location, or move with a vector.
If nothing needs fixing, return an empty command list.
Respond ONLY with JSON: {{"reply": "<one short sentence>", "commands": [ ... ]}}

SCENE WITH WORLD BOUNDS:
{bounds}"#,
        issues = list.join("\n")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::editor::Editor;

    fn cmd(s: &str) -> Command {
        serde_json::from_str(s).unwrap()
    }

    #[test]
    fn finds_floating_sunken_and_duplicates() {
        let mut e = Editor::default();
        let before = e.scene.clone();
        e.execute(&[
            cmd(r#"{"action":"add","primitive":"plane","name":"Ground","location":[0,0,-1],"scale":[10,10,1]}"#),
            cmd(r#"{"action":"add","primitive":"cube","name":"Floaty","location":[5,5,3]}"#),
            cmd(r#"{"action":"add","primitive":"uv_sphere","name":"Sunk","location":[-5,-5,-1.5]}"#),
            cmd(r#"{"action":"add","primitive":"cube","name":"Twin","location":[0,0,0]}"#),
            cmd(r#"{"action":"add","primitive":"cone","name":"OnCube","location":[0,0,2]}"#),
        ]);
        let changed = changed_objects(&before, &e.scene);
        assert_eq!(changed.len(), 5);
        let mut cache = MeshCache::default();
        let issues = inspect(&e.scene, &mut cache, &changed, &["✖ move: no target".into()]);
        let txt: Vec<&str> = issues.iter().map(|i| i.text.as_str()).collect();
        assert!(txt.iter().any(|t| t.contains("\"Floaty\" floats 3.00")), "{txt:?}");
        assert!(txt.iter().any(|t| t.contains("\"Sunk\" is 1.50 units below")), "{txt:?}");
        assert!(txt.iter().any(|t| t.contains("\"Twin\" is an exact duplicate of \"Cube\"")), "{txt:?}");
        assert!(txt.iter().any(|t| t.contains("command failed")));
        // the cone resting on the cube and the cube on the ground are fine
        assert!(!txt.iter().any(|t| t.contains("OnCube")), "{txt:?}");
        assert!(!txt.iter().any(|t| t.contains("\"Ground\"")), "{txt:?}");
        // applying the automatic fixes resolves the geometric issues
        let fixes: Vec<Command> = issues.iter().filter_map(|i| i.auto_fix.clone()).collect();
        e.execute(&fixes);
        let after = inspect(&e.scene, &mut cache, &changed, &[]);
        assert!(after.is_empty(), "{after:?}");
    }

    #[test]
    fn stacked_snowman_is_clean() {
        let mut e = Editor::default();
        e.execute(&[cmd(r#"{"action":"delete","target":"Cube"}"#)]);
        let before = e.scene.clone();
        e.execute(&[
            cmd(r#"{"action":"add","primitive":"uv_sphere","name":"Base","location":[0,0,1.2],"scale":[1.2,1.2,1.2]}"#),
            cmd(r#"{"action":"add","primitive":"uv_sphere","name":"Mid","location":[0,0,2.9],"scale":[0.85,0.85,0.85]}"#),
            cmd(r#"{"action":"add","primitive":"uv_sphere","name":"Head","location":[0,0,4.2],"scale":[0.6,0.6,0.6]}"#),
        ]);
        let changed = changed_objects(&before, &e.scene);
        let issues = inspect(&e.scene, &mut MeshCache::default(), &changed, &[]);
        assert!(issues.is_empty(), "{issues:?}");
        assert!(review_prompt("snowman", &issues, "x").contains("QUALITY CHECKER"));
        // a scarf around the middle is visible; a nose buried in the head is not
        let before = e.scene.clone();
        e.execute(&[
            cmd(r#"{"action":"add","primitive":"torus","name":"Scarf","location":[0,0,3.55],"scale":[0.55,0.55,0.9]}"#),
            cmd(r#"{"action":"add","primitive":"cone","name":"Nose","location":[0,0,4.2],"scale":[0.1,0.1,0.3]}"#),
        ]);
        let changed = changed_objects(&before, &e.scene);
        let issues = inspect(&e.scene, &mut MeshCache::default(), &changed, &[]);
        let txt: Vec<&str> = issues.iter().map(|i| i.text.as_str()).collect();
        assert!(!txt.iter().any(|t| t.contains("Scarf")), "{txt:?}");
        assert!(txt.iter().any(|t| t.contains("\"Nose\" is completely inside \"Head\"")), "{txt:?}");
    }
}
