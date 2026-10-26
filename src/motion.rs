//! NLA-lite: a character's performance as a sequence of action strips on its armature —
//! "Idle, then walk to the table, wave, run along the path" — with cross-fades between them and
//! root motion: walk/run strips carry the rig along a path at the speed its feet actually move
//! (measured from the action), turning to face the way it goes. Gaps and the end fall back to
//! the Idle action (or hold the last pose).

use crate::armature::{Armature, BonePose};
use crate::scene::Transform;
use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Strip {
    /// Action name (looked up in the armature's actions).
    pub action: String,
    /// Scene frames [start, end).
    pub start: i32,
    pub end: i32,
    /// Cross-fade in from what came before (frames).
    #[serde(default = "default_blend")]
    pub blend: f32,
    /// Ground path (world) the rig travels along during the strip; empty = stay in place.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub path: Vec<Vec3>,
    /// Ground speed (m/frame) measured from the action when the strip was made (0 = measure).
    #[serde(default)]
    pub speed: f32,
}

fn default_blend() -> f32 {
    6.0
}

/// Arc length along a polyline.
pub fn path_len(p: &[Vec3]) -> f32 {
    p.windows(2).map(|w| w[0].distance(w[1])).sum()
}

/// Point and direction at distance `s` along a polyline (clamped).
pub fn along(p: &[Vec3], s: f32) -> (Vec3, Vec3) {
    if p.len() < 2 {
        return (p.first().copied().unwrap_or(Vec3::ZERO), Vec3::NEG_Y);
    }
    let mut left = s.max(0.0);
    for w in p.windows(2) {
        let l = w[0].distance(w[1]);
        if left <= l || std::ptr::eq(w.as_ptr(), p[p.len() - 2..].as_ptr()) {
            let t = if l > 1e-6 { (left / l).min(1.0) } else { 1.0 };
            return (w[0].lerp(w[1], t), (w[1] - w[0]).normalize_or(Vec3::NEG_Y));
        }
        left -= l;
    }
    let n = p.len();
    (p[n - 1], (p[n - 1] - p[n - 2]).normalize_or(Vec3::NEG_Y))
}

/// Yaw (degrees about Z) that turns the rig's front (−Y) towards `d`.
pub fn yaw_for(d: Vec3) -> f32 {
    d.x.atan2(-d.y).to_degrees()
}

fn sample_action(arm: &Armature, ai: usize, t: f32) -> Vec<BonePose> {
    let a = &arm.actions[ai];
    // the active action's keys live (possibly edited) in the bones
    let live = ai == arm.active_action;
    arm.bones.iter().enumerate().map(|(i, b)| if live { crate::armature::sample_pub(&b.keys, t) } else { a.keys.get(i).and_then(|k| crate::armature::sample_pub(k, t)) }.unwrap_or(b.pose)).collect()
}

/// Local action time for a strip at scene frame `f` (actions loop).
fn local_time(arm: &Armature, ai: usize, start: i32, f: f32) -> f32 {
    let Some((lo, hi)) = arm.actions[ai].range() else { return 0.0 };
    let len = (hi - lo).max(1) as f32;
    lo as f32 + (f - start as f32).rem_euclid(len)
}

/// Metres per frame the feet push the body along in a locomotion action (0 for actions that
/// don't travel): from how far the lower foot slides back over the cycle.
pub fn ground_speed(arm: &Armature, ai: usize) -> f32 {
    let Some((lo, hi)) = arm.actions.get(ai).and_then(|a| a.range()) else { return 0.0 };
    let feet: Vec<usize> = arm.bones.iter().enumerate().filter(|(_, b)| b.name.starts_with("foot")).map(|(i, _)| i).collect();
    if feet.is_empty() || hi - lo < 4 {
        return 0.0;
    }
    let mut a = arm.clone();
    a.blend = None;
    let mut total = 0.0;
    let mut n = 0.0;
    let mut prev: Option<Vec<Vec3>> = None;
    for f in lo..=hi {
        for (b, p) in a.bones.iter_mut().zip(sample_action(arm, ai, f as f32)) {
            b.pose = p;
        }
        let g = a.pose_globals();
        let pts: Vec<Vec3> = feet.iter().map(|&i| g[i].transform_point3(Vec3::ZERO)).collect();
        if let Some(pp) = &prev {
            // the planted foot is the lower one; it moves backwards (+Y) as the body goes forward
            let k = (0..pts.len()).min_by(|x, y| pts[*x].z.partial_cmp(&pts[*y].z).unwrap()).unwrap();
            let dy = pts[k].y - pp[k].y;
            if dy > 0.0 {
                total += dy;
            }
            n += 1.0;
        }
        prev = Some(pts);
    }
    if n == 0.0 { 0.0 } else { total / n }
}

/// Evaluate a performance at `frame`: sets the bone poses and returns the rig's transform.
pub fn evaluate(arm: &mut Armature, home: &Transform, frame: f32) -> Option<Transform> {
    if arm.strips.is_empty() {
        return None;
    }
    let (poses, tr) = compute(arm, home, frame);
    if let Some(p) = poses {
        for (b, q) in arm.bones.iter_mut().zip(p) {
            b.pose = q;
        }
    }
    Some(tr)
}

fn compute(arm: &Armature, home: &Transform, frame: f32) -> (Option<Vec<BonePose>>, Transform) {
    let find = |name: &str| arm.find_action(name);
    let idle = find("idle");
    // the strip playing now (gaps and the end play Idle)
    let mut strips: Vec<&Strip> = arm.strips.iter().collect();
    strips.sort_by_key(|s| s.start);
    let cur = strips.iter().rposition(|s| s.start as f32 <= frame && frame < s.end as f32);
    let pose_of = |s: Option<&Strip>, f: f32| -> Option<Vec<BonePose>> {
        match s {
            Some(s) => {
                let ai = find(&s.action)?;
                Some(sample_action(arm, ai, local_time(arm, ai, s.start, f)))
            }
            None => idle.map(|ai| sample_action(arm, ai, local_time(arm, ai, 0, f))),
        }
    };
    // what's playing, when it began, and what played before it (for the cross-fade)
    let (now, since, before): (Option<&Strip>, f32, Option<Option<&Strip>>) = match cur {
        Some(i) => {
            let s = strips[i];
            let prev = if i > 0 && strips[i - 1].end >= s.start { Some(strips[i - 1]) } else { None };
            (Some(s), s.start as f32, Some(prev))
        }
        None => {
            let last = strips.iter().rposition(|s| (s.end as f32) <= frame);
            match last {
                Some(i) => (None, strips[i].end as f32, Some(Some(strips[i]))),
                None => (None, f32::MIN, None),
            }
        }
    };
    let blend = now.map(|s| s.blend).unwrap_or(8.0).max(0.0);
    let mut poses = pose_of(now, frame);
    if let (Some(prev), Some(p)) = (before, poses.as_mut()) {
        let t = if blend > 0.0 { (frame - since) / blend } else { 1.0 };
        if (0.0..1.0).contains(&t) {
            // the previous strip frozen where it stopped
            if let Some(from) = pose_of(prev, since.max(prev.map(|s| s.start as f32).unwrap_or(since)) - 0.001) {
                let t = t * t * (3.0 - 2.0 * t);
                for (a, b) in p.iter_mut().zip(from) {
                    *a = b.lerp(a, t);
                }
            }
        }
    }
    // root motion: replay every travelling strip up to now
    let mut tr = *home;
    for s in strips.iter().filter(|s| s.path.len() >= 2 && s.start as f32 <= frame) {
        let speed = if s.speed > 0.0 { s.speed } else { find(&s.action).map(|ai| ground_speed(arm, ai)).unwrap_or(0.0) };
        let len = path_len(&s.path);
        let f = frame.min(s.end as f32) - s.start as f32;
        // ease in/out over the blend so starts and stops aren't instant
        let dur = (s.end - s.start).max(1) as f32;
        let travel = if speed > 0.0 { (speed * f).min(len) } else { len * (f / dur) };
        let (p, d) = along(&s.path, travel);
        // look a little ahead so corners turn smoothly
        let (_, d2) = along(&s.path, travel + 0.35);
        tr.location = Vec3::new(p.x, p.y, home.location.z);
        tr.rotation.z = yaw_for((d + d2).normalize_or(d));
    }
    (poses, tr)
}

/// Frames a travelling strip needs to cover its path at the action's own speed.
pub fn frames_for(arm: &Armature, action: &str, path: &[Vec3]) -> i32 {
    let speed = arm.find_action(action).map(|ai| ground_speed(arm, ai)).unwrap_or(0.0);
    let len = path_len(path);
    if speed <= 1e-5 {
        return (len / 0.05).ceil().max(12.0) as i32;
    }
    (len / speed).ceil().max(4.0) as i32
}

/// Where the rig is (and faces) at the end of all strips.
pub fn end_state(arm: &Armature, home: &Transform) -> (Vec3, f32, i32) {
    let mut pos = home.location;
    let mut yaw = home.rotation.z;
    let mut end = 1;
    let mut strips: Vec<&Strip> = arm.strips.iter().collect();
    strips.sort_by_key(|s| s.start);
    for s in strips {
        end = end.max(s.end);
        if s.path.len() >= 2 {
            let n = s.path.len();
            pos = Vec3::new(s.path[n - 1].x, s.path[n - 1].y, pos.z);
            yaw = yaw_for(s.path[n - 1] - s.path[n - 2]);
        }
    }
    (pos, yaw, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_sequence_blend_and_travel() {
        let (_, mut arm, _, _) = crate::character::build();
        let walk = arm.find_action("walk").unwrap();
        let sp = ground_speed(&arm, walk);
        // the walk moves about 1 m/s at 24 fps; Wave doesn't travel
        assert!(sp * 24.0 > 0.6 && sp * 24.0 < 1.8, "walk speed {} m/s", sp * 24.0);
        let run = arm.find_action("run").unwrap();
        assert!(ground_speed(&arm, run) > sp * 1.4, "running is faster");
        assert!(ground_speed(&arm, arm.find_action("wave").unwrap()) * 24.0 < 0.2);
        let home = Transform::default();
        let path = vec![Vec3::ZERO, Vec3::new(0.0, -3.0, 0.0), Vec3::new(3.0, -3.0, 0.0)];
        let n = frames_for(&arm, "Walk", &path);
        arm.strips = vec![
            Strip { action: "Walk".into(), start: 10, end: 10 + n, blend: 6.0, path: path.clone(), speed: sp },
            Strip { action: "Wave".into(), start: 10 + n, end: 10 + n + 48, blend: 6.0, path: vec![], speed: 0.0 },
        ];
        // before the walk: at home, idling
        let t0 = evaluate(&mut arm, &home, 5.0).unwrap();
        assert!(t0.location.length() < 1e-4);
        // halfway along the path, facing along it (−Y first: yaw ~0)
        let t1 = evaluate(&mut arm, &home, 10.0 + n as f32 * 0.25).unwrap();
        assert!(t1.location.y < -0.5 && t1.location.x.abs() < 0.2, "{}", t1.location);
        assert!(t1.rotation.z.abs() < 20.0, "faces −Y: {}", t1.rotation.z);
        // on the second leg it turns to +X (yaw +90)
        let t2 = evaluate(&mut arm, &home, 10.0 + n as f32 * 0.85).unwrap();
        assert!(t2.location.x > 1.0 && (t2.rotation.z - 90.0).abs() < 25.0, "{} {}", t2.location, t2.rotation.z);
        // after: stays at the end of the path, waving
        let t3 = evaluate(&mut arm, &home, (10 + n + 20) as f32).unwrap();
        assert!((t3.location - Vec3::new(3.0, -3.0, 0.0)).length() < 1e-3);
        let (pos, _, end) = end_state(&arm, &home);
        assert!((pos - Vec3::new(3.0, -3.0, 0.0)).length() < 1e-4 && end == 10 + n + 48);
        // the cross-fade: just after the wave starts the pose is between walk and wave
        let at = |arm: &mut Armature, f: f32| {
            evaluate(arm, &home, f);
            arm.bones[crate::character::BONES.iter().position(|b| b.0 == "upper_arm.R").unwrap()].pose.rot
        };
        let a0 = at(&mut arm, (10 + n) as f32 - 0.5);
        let a1 = at(&mut arm, (10 + n) as f32 + 3.0);
        let a2 = at(&mut arm, (10 + n) as f32 + 12.0);
        assert!(a0.angle_between(a1) > 1e-3 && a1.angle_between(a2) > 1e-3, "blends");
        assert!(along(&path, 100.0).0.distance(Vec3::new(3.0, -3.0, 0.0)) < 1e-5);
    }
}
