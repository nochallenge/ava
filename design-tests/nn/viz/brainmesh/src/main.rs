// TESTING / DESIGN ONLY
//! Turns the BodyParts3D brain meshes (one OBJ per part, in mm) into one file the chat page can load from disk:
//! `window.BRAIN = {...}` with base64 buffers. The page cannot fetch files over file://, but it can load a script.
//!
//! Usage: brainmesh <folder with FJxxxx.obj + elements.tsv> <out.js>
//!
//! Each vertex keeps: its position (Int16, the brain scaled into [-1, 1]), its smooth normal (Int8), its part (Uint8),
//! how far it lies below the brain's outer envelope (Uint8, how deep in a fold) and its local bend (Int8, valley vs
//! crown). The last two are what make the folds read in a see-through hologram.
//! The output frame is fixed by the anatomy, found from the data rather than assumed: x = toward the back,
//! y = up, z = toward her left. Looking from +z (her left side) shows the front on the left of the screen.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

type V3 = [f64; 3];

fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: V3) -> V3 {
    let l = dot(a, a).sqrt().max(1e-12);
    [a[0] / l, a[1] / l, a[2] / l]
}

/// The lighting groups. The page gives each a colour and drives its glow; 0 is plain cortex.
const GROUPS: [&str; 19] = [
    "cortex", "understand", "understand_r", "reading", "meaning", "speak", "speak_r", "workmem", "plan",
    "attention", "memory", "effort", "mood", "vision", "motor", "cerebellum", "brainstem", "inner", "ventricles",
];

fn group_of(name: &str) -> usize {
    let left = name.starts_with("left ");
    let has = |s: &str| name.contains(s);
    let g = if has("angular") || has("supramarginal") || has("middle temporal") {
        if left { "understand" } else { "understand_r" }
    } else if has("fusiform") {
        "reading"
    } else if has("inferior temporal") {
        "meaning"
    } else if has("inferior frontal") {
        if left { "speak" } else { "speak_r" }
    } else if has("middle frontal") {
        "workmem"
    } else if has("superior frontal") {
        "plan"
    } else if has("superior parietal") {
        "attention"
    } else if has("hippocamp") {
        "memory"
    } else if has("cingulate") {
        "effort"
    } else if has("hypothalam") || has("insula") {
        "mood"
    } else if has("occipital") {
        "vision"
    } else if has("precentral") || has("postcentral") {
        "motor"
    } else if has("cerebellum") {
        "cerebellum"
    } else if has("pons") || has("medulla") || has("midbrain") || has("colliculus") {
        "brainstem"
    } else if has("ventricle") || has("aqueduct") {
        "ventricles"
    } else if has("subcortex") || has("internal capsule") || has("habenula") || has("pineal") {
        "inner"
    } else {
        "cortex"
    };
    GROUPS.iter().position(|x| *x == g).unwrap()
}

struct Part {
    name: String,
    pos: Vec<V3>,
    tri: Vec<[u32; 3]>,
}

fn read_obj(path: &Path) -> Result<(Vec<V3>, Vec<[u32; 3]>), String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (mut pos, mut tri) = (Vec::new(), Vec::new());
    for line in text.lines() {
        let mut it = line.split_whitespace();
        match it.next() {
            Some("v") => {
                let v: Vec<f64> = it.take(3).map(|s| s.parse().unwrap_or(0.0)).collect();
                pos.push([v[0], v[1], v[2]]);
            }
            Some("f") => {
                let ix: Vec<u32> = it.map(|s| s.split('/').next().unwrap().parse::<u32>().unwrap() - 1).collect();
                for k in 1..ix.len().saturating_sub(1) {
                    tri.push([ix[0], ix[k], ix[k + 1]]);
                }
            }
            _ => {}
        }
    }
    Ok((pos, tri))
}

fn centroid(parts: &[Part], pick: impl Fn(&str) -> bool) -> V3 {
    let (mut s, mut n) = ([0.0; 3], 0.0);
    for p in parts.iter().filter(|p| pick(&p.name)) {
        for v in &p.pos {
            for k in 0..3 {
                s[k] += v[k];
            }
            n += 1.0;
        }
    }
    [s[0] / n, s[1] / n, s[2] / n]
}

/// The data axis (signed unit vector) nearest to `d`.
fn snap(d: V3) -> V3 {
    let k = (0..3).max_by(|&a, &b| d[a].abs().total_cmp(&d[b].abs())).unwrap();
    let mut a = [0.0; 3];
    a[k] = d[k].signum();
    a
}

fn b64(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(bytes.len() * 4 / 3 + 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        s.push(T[(n >> 18) as usize & 63] as char);
        s.push(T[(n >> 12) as usize & 63] as char);
        s.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        s.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    s
}

fn main() -> Result<(), String> {
    let t0 = Instant::now();
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: brainmesh <folder with FJxxxx.obj + elements.tsv> <out.js>".into());
    }
    let dir = Path::new(&args[1]);
    let list = fs::read_to_string(dir.join("elements.tsv")).map_err(|e| format!("elements.tsv: {e}"))?;
    let mut parts = Vec::new();
    for line in list.trim_start_matches('\u{feff}').lines().filter(|l| !l.trim().is_empty()) {
        let (fj, name) = line.split_once('\t').ok_or(format!("bad line: {line}"))?;
        let (pos, tri) = read_obj(&dir.join(format!("{}.obj", fj.trim())))?;
        parts.push(Part { name: name.trim().to_string(), pos, tri });
    }

    // The anatomy's axes, from the data.
    let left = snap(sub(centroid(&parts, |n| n.starts_with("left ")), centroid(&parts, |n| n.starts_with("right "))));
    let front = snap(sub(centroid(&parts, |n| n.contains("frontal gyrus")), centroid(&parts, |n| n.contains("occipital"))));
    let up = snap(sub(centroid(&parts, |n| n.contains("superior frontal") || n.contains("precentral")), centroid(&parts, |n| n.contains("medulla"))));
    let back = [-front[0], -front[1], -front[2]];
    let hand = dot(cross(back, up), left);
    println!("axes in the data: left {left:?}, front {front:?}, up {up:?}; frame (back, up, left) handedness {hand:+}");
    if hand < 0.0 {
        println!("WARNING: the frame is mirrored; the data's left/right labels and its axes disagree");
    }

    // Into the frame (back, up, left), centred on the bounding box, scaled into [-1, 1].
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in &mut parts {
        for v in &mut p.pos {
            *v = [dot(*v, back), dot(*v, up), dot(*v, left)];
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
    }
    let c = [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (lo[2] + hi[2]) / 2.0];
    let half = (0..3).map(|k| (hi[k] - lo[k]) / 2.0).fold(0.0, f64::max);
    println!("size (mm): back-front {:.0}, up {:.0}, left-right {:.0}", hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);

    // One buffer for all parts; per-vertex normal, bend and part.
    let (mut pos, mut nrm, mut bend, mut pid, mut idx) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (pi, p) in parts.iter().enumerate() {
        let base = pos.len() as u32;
        let mut n = vec![[0.0; 3]; p.pos.len()];
        let mut nb: Vec<HashMap<u32, ()>> = vec![HashMap::new(); p.pos.len()];
        for t in &p.tri {
            let [a, b, cc] = t.map(|i| p.pos[i as usize]);
            let f = cross(sub(b, a), sub(cc, a)); // area-weighted
            for (k, &i) in t.iter().enumerate() {
                for j in 0..3 {
                    n[i as usize][j] += f[j];
                }
                nb[i as usize].insert(t[(k + 1) % 3], ());
                nb[i as usize].insert(t[(k + 2) % 3], ());
            }
            idx.extend(t.iter().map(|&i| base + i));
        }
        for (i, v) in p.pos.iter().enumerate() {
            let nn = norm(n[i]);
            // bend: how far the neighbours sit along the normal, per unit edge (+ valley, - crown)
            let (mut m, mut e) = ([0.0; 3], 0.0);
            for &j in nb[i].keys() {
                let d = sub(p.pos[j as usize], *v);
                for k in 0..3 {
                    m[k] += d[k];
                }
                e += dot(d, d).sqrt();
            }
            let cnt = nb[i].len().max(1) as f64;
            bend.push(dot([m[0] / cnt, m[1] / cnt, m[2] / cnt], nn) / (e / cnt).max(1e-9));
            pos.push(*v);
            nrm.push(nn);
            pid.push(pi as u8);
        }
    }

    // depth: how far each vertex lies below the outer envelope of its own side (sulci are deep, crowns are 0).
    // The envelope is the largest radius seen in each direction from the side's centre (a 96 x 48 grid of
    // directions, widened by one cell so small gaps between meshes do not count as deep).
    let side_of = |name: &str| if name.starts_with("left ") { 0 } else if name.starts_with("right ") { 1 } else { 2 };
    let centres: Vec<V3> = (0..3)
        .map(|s| {
            let (mut acc, mut k) = ([0.0; 3], 0.0);
            for (v, &q) in pos.iter().zip(&pid) {
                if side_of(&parts[q as usize].name) == s {
                    for j in 0..3 {
                        acc[j] += v[j];
                    }
                    k += 1.0;
                }
            }
            [acc[0] / k, acc[1] / k, acc[2] / k]
        })
        .collect();
    const NA: usize = 96;
    const NE: usize = 48;
    let cell = |d: V3| -> (usize, usize) {
        let az = d[2].atan2(d[0]) + std::f64::consts::PI;
        let el = d[1].clamp(-1.0, 1.0).acos();
        (((az / (2.0 * std::f64::consts::PI)) * NA as f64) as usize % NA, ((el / std::f64::consts::PI) * NE as f64).min(NE as f64 - 1.0) as usize)
    };
    let mut env = vec![vec![0.0f64; NA * NE]; 3];
    for (v, &q) in pos.iter().zip(&pid) {
        let s = side_of(&parts[q as usize].name);
        let d = sub(*v, centres[s]);
        let r = dot(d, d).sqrt();
        let (a, e) = cell(norm(d));
        env[s][e * NA + a] = env[s][e * NA + a].max(r);
    }
    let wide: Vec<Vec<f64>> = env
        .iter()
        .map(|g| {
            let mut w = vec![0.0; NA * NE];
            for e in 0..NE {
                for a in 0..NA {
                    let mut m: f64 = 0.0;
                    for de in -1i32..=1 {
                        let ee = (e as i32 + de).clamp(0, NE as i32 - 1) as usize;
                        for da in -1i32..=1 {
                            m = m.max(g[ee * NA + (a as i32 + da).rem_euclid(NA as i32) as usize]);
                        }
                    }
                    w[e * NA + a] = m;
                }
            }
            w
        })
        .collect();
    let depth: Vec<u8> = pos
        .iter()
        .zip(&pid)
        .map(|(v, &q)| {
            let s = side_of(&parts[q as usize].name);
            let d = sub(*v, centres[s]);
            let r = dot(d, d).sqrt();
            let (a, e) = cell(norm(d));
            // 0 at the envelope, 255 at 20 mm or more below it
            (((wide[s][e * NA + a] - r) / 20.0).clamp(0.0, 1.0) * 255.0) as u8
        })
        .collect();

    // bend scaled by its 95th percentile so the page sees -1..1
    let mut ab: Vec<f64> = bend.iter().map(|b| b.abs()).collect();
    ab.sort_by(|a, b| a.total_cmp(b));
    let p95 = ab[ab.len() * 95 / 100].max(1e-9);

    let mut b_pos = Vec::with_capacity(pos.len() * 6);
    for v in &pos {
        for k in 0..3 {
            b_pos.extend_from_slice(&((((v[k] - c[k]) / half).clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
        }
    }
    let b_nrm: Vec<u8> = nrm.iter().flat_map(|n| n.map(|x| (x * 127.0).round() as i8 as u8)).collect();
    let b_bend: Vec<u8> = bend.iter().map(|b| ((b / p95).clamp(-1.0, 1.0) * 127.0).round() as i8 as u8).collect();
    let b_idx: Vec<u8> = idx.iter().flat_map(|i| i.to_le_bytes()).collect();

    let names: Vec<String> = parts.iter().map(|p| format!("{:?}", p.name)).collect();
    let pg: Vec<String> = parts.iter().map(|p| group_of(&p.name).to_string()).collect();
    let groups: Vec<String> = GROUPS.iter().map(|g| format!("{g:?}")).collect();
    let js = format!(
        "// Brain model: BodyParts3D, (c) The Database Center for Life Science (DBCLS), CC BY 4.0. Converted by brainmesh.\n\
         window.BRAIN={{nv:{},ni:{},half_mm:{:.2},groups:[{}],parts:[{}],partGroup:[{}],\n\
         pos:\"{}\",\nnrm:\"{}\",\npart:\"{}\",\ndepth:\"{}\",\nbend:\"{}\",\nidx:\"{}\"}};\n",
        pos.len(),
        idx.len(),
        half,
        groups.join(","),
        names.join(","),
        pg.join(","),
        b64(&b_pos),
        b64(&b_nrm),
        b64(&pid),
        b64(&depth),
        b64(&b_bend),
        b64(&b_idx)
    );
    fs::write(&args[2], &js).map_err(|e| format!("{}: {e}", args[2]))?;
    let mut per: Vec<usize> = vec![0; GROUPS.len()];
    for p in &parts {
        per[group_of(&p.name)] += 1;
    }
    let shown: Vec<String> = GROUPS.iter().zip(&per).filter(|(_, &n)| n > 0).map(|(g, n)| format!("{g} {n}")).collect();
    println!("parts per group: {}", shown.join(", "));
    println!("{} parts, {} vertices, {} triangles -> {} ({:.1} MB) in {:.2} s", parts.len(), pos.len(), idx.len() / 3, args[2], js.len() as f64 / 1e6, t0.elapsed().as_secs_f64());
    Ok(())
}
