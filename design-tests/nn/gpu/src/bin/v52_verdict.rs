// TESTING / DESIGN ONLY — v52 Stage 1 verdict (runs/v52/prereg_v52.md, "Stage 1"): which parts help the trained brain.
//
// Input (written by runs/v52/run_v52s1.sh):
//   points.tsv  arm \t seed \t step \t val_bpb   (v41-mix val, 1 M targets; steps 1000, 2000, 4000; NaN allowed)
//   cost.tsv    arm \t seed \t train_seconds \t steps
//   --broke "Pr S"  arms that broke in training (Amendment S1-A); may be empty
//   --parts "P DA Y TH Hn"  which parts are judged (Amendment S1-E, Stage 1c); default R M N Pr S CB (Stage 1b)
// Rules (pre-registered): s_p^2 = mean over arms of the two seeds' variance at 4000 (B and every part that did not
// break); g(t) = mean B(t) - mean X(t); bar = max(2 s_p, 0.002); WIN g > bar, LOSE g < -bar, else TIE; fading
// g(4000) < g(2000) - s_p; keep = WIN, not fading, train time per step <= 1.25 x B's. A BROKE arm is never kept; its
// gaps are shown from the seeds it has (B's same seed minus its own).

use std::collections::BTreeMap;

const SEEDS: [u32; 2] = [41, 52];
const STEPS: [u32; 3] = [1000, 2000, 4000];
const PARTS: [&str; 6] = ["R", "M", "N", "Pr", "S", "CB"];

type Points = BTreeMap<(String, u32, u32), f64>;

struct Verdict {
    arm: String,
    g: [Option<f64>; 3],
    call: &'static str,
    fading: Option<bool>,
    cost: f64,
    keep: bool,
}

fn read_tsv(path: &str) -> Result<Vec<Vec<String>>, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    Ok(s.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#')).map(|l| l.trim_end_matches('\r').split('\t').map(str::to_string).collect()).collect())
}

/// (arm, seed, step) -> val_bpb
fn points(rows: &[Vec<String>]) -> Result<Points, String> {
    let mut m = BTreeMap::new();
    for r in rows {
        if r.len() < 4 {
            return Err(format!("points row {r:?}: want arm, seed, step, val_bpb"));
        }
        let k = (r[0].clone(), r[1].parse().map_err(|e| format!("seed {:?}: {e}", r[1]))?, r[2].parse().map_err(|e| format!("step {:?}: {e}", r[2]))?);
        let v: f64 = r[3].parse().map_err(|e| format!("val_bpb {:?}: {e}", r[3]))?;
        if m.insert(k.clone(), v).is_some() {
            return Err(format!("points: {k:?} twice"));
        }
    }
    Ok(m)
}

/// The mean over both seeds; every point must be there and finite.
fn mean_at(p: &Points, arm: &str, step: u32) -> Result<f64, String> {
    let mut s = 0.0;
    for seed in SEEDS {
        let v = *p.get(&(arm.to_string(), seed, step)).ok_or(format!("missing {arm} seed {seed} step {step}"))?;
        if !v.is_finite() {
            return Err(format!("{arm} seed {seed} step {step} is {v}: list the arm in --broke"));
        }
        s += v;
    }
    Ok(s / SEEDS.len() as f64)
}

/// A broke arm's gap at one step: mean over the seeds it has of B(seed) - X(seed); None if it has none (or NaN).
fn broke_gap(p: &Points, arm: &str, step: u32) -> Result<Option<f64>, String> {
    let mut v = vec![];
    for seed in SEEDS {
        if let Some(x) = p.get(&(arm.to_string(), seed, step)) {
            let b = *p.get(&("B".to_string(), seed, step)).ok_or(format!("missing B seed {seed} step {step}"))?;
            v.push(b - x);
        }
    }
    Ok(if v.is_empty() { None } else { Some(v.iter().sum::<f64>() / v.len() as f64) }.filter(|g| g.is_finite()))
}

/// seconds per step, mean over the rows there are (`want` of them unless the arm broke)
fn cost_of(rows: &[Vec<String>], arm: &str, want: Option<usize>) -> Result<f64, String> {
    let mut v = vec![];
    for r in rows.iter().filter(|r| r.first().map(|a| a == arm).unwrap_or(false)) {
        let secs: f64 = r.get(2).ok_or("cost row: no train_seconds")?.parse().map_err(|e| format!("train_seconds: {e}"))?;
        let steps: f64 = r.get(3).ok_or("cost row: no steps")?.parse().map_err(|e| format!("steps: {e}"))?;
        v.push(secs / steps.max(1.0));
    }
    if want.map(|w| v.len() != w).unwrap_or(v.is_empty()) {
        return Err(format!("cost: {} rows for {arm}, want {}", v.len(), want.unwrap_or(1)));
    }
    Ok(v.iter().sum::<f64>() / v.len() as f64)
}

/// The pooled within-arm SD at 4000 over B and every part that did not break.
fn pooled_sd(p: &Points, broke: &[String], parts: &[&str]) -> Result<f64, String> {
    let mut s = 0.0;
    let arms: Vec<&str> = std::iter::once("B").chain(parts.iter().copied()).filter(|a| !broke.iter().any(|b| b == a)).collect();
    for a in &arms {
        let x: Vec<f64> = SEEDS.iter().map(|&sd| p.get(&(a.to_string(), sd, 4000)).copied().ok_or(format!("missing {a} seed {sd} step 4000"))).collect::<Result<_, _>>()?;
        let m = x.iter().sum::<f64>() / x.len() as f64;
        s += x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (x.len() - 1) as f64;
    }
    let sp = (s / arms.len() as f64).sqrt();
    if !sp.is_finite() {
        return Err("s_p is not finite: an arm that did not break has a NaN at 4000".into());
    }
    Ok(sp)
}

#[cfg(test)]
fn judge(p: &Points, cost: &[Vec<String>], broke: &[String]) -> Result<(f64, f64, f64, Vec<Verdict>), String> {
    judge_parts(p, cost, broke, &PARTS)
}

/// The verdict over `parts` (Stage 1b: PARTS; Stage 1c, Amendment S1-E: `--parts "P DA Y TH Hn"`), each against B.
fn judge_parts(p: &Points, cost: &[Vec<String>], broke: &[String], parts: &[&str]) -> Result<(f64, f64, f64, Vec<Verdict>), String> {
    if let Some(b) = broke.iter().find(|b| !parts.contains(&b.as_str())) {
        return Err(format!("--broke {b}: not a part (if B broke, Stage 1 stops)"));
    }
    let sp = pooled_sd(p, broke, parts)?;
    let bar = (2.0 * sp).max(0.002);
    let b_cost = cost_of(cost, "B", Some(SEEDS.len()))?;
    let mut out = vec![];
    for &part in parts {
        let mut g = [None; 3];
        if broke.iter().any(|b| b == part) {
            for (i, st) in STEPS.iter().enumerate() {
                g[i] = broke_gap(p, part, *st)?;
            }
            let c = cost_of(cost, part, None)? / b_cost;
            out.push(Verdict { arm: part.to_string(), g, call: "BROKE", fading: None, cost: c, keep: false });
            continue;
        }
        for (i, st) in STEPS.iter().enumerate() {
            g[i] = Some(mean_at(p, "B", *st)? - mean_at(p, part, *st)?);
        }
        let (g2, g4) = (g[1].unwrap(), g[2].unwrap());
        let call = if g4 > bar { "WIN" } else if g4 < -bar { "LOSE" } else { "TIE" };
        let fading = g4 < g2 - sp;
        let c = cost_of(cost, part, Some(SEEDS.len()))? / b_cost;
        let keep = call == "WIN" && !fading && c <= 1.25;
        out.push(Verdict { arm: part.to_string(), g, call, fading: Some(fading), cost: c, keep });
    }
    Ok((sp, bar, b_cost, out))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let get = |k: &str| opt(k).ok_or(format!("need {k}"));
    let (pp, cp, op) = (get("--points")?, get("--cost")?, get("--out")?);
    let broke: Vec<String> = opt("--broke").unwrap_or_default().split(|c: char| c == ',' || c.is_whitespace()).filter(|s| !s.is_empty()).map(str::to_string).collect();
    let p = points(&read_tsv(&pp)?)?;
    let cost = read_tsv(&cp)?;
    // Amendment S1-E: which parts are judged (default: Stage 1b's six)
    let given = opt("--parts").unwrap_or_default();
    let mut parts: Vec<&str> = given.split(|c: char| c == ',' || c.is_whitespace()).filter(|s| !s.is_empty()).collect();
    if parts.is_empty() {
        parts = PARTS.to_vec();
    }
    if parts.iter().any(|&a| a == "B") {
        return Err("--parts: B is the base, not a part".into());
    }
    let (sp, bar, b_cost, vs) = judge_parts(&p, &cost, &broke, &parts)?;
    let f = |x: Option<f64>| x.map(|v| format!("{v:+.6}")).unwrap_or("-".into());
    let mut s = String::new();
    s += "# v52 Stage 1 verdict (prereg_v52.md): g = B - X in bits/byte (positive: the part helps), v41-mix val, mean of seeds 41 and 52\n";
    if !broke.is_empty() {
        s += &format!("# BROKE (Amendment S1-A: broke in training, never kept, left out of s_p; gaps from the seeds they have): {}\n", broke.join(" "));
    }
    s += &format!("s_p {sp:.6}  bar {bar:.6}  B: val@4000 {:.6}, {b_cost:.4} s/step\n", mean_at(&p, "B", 4000)?);
    s += "part\tg@1000\tg@2000\tg@4000\tverdict\tfading\tcost_vs_B\tkeep\n";
    for v in &vs {
        let fd = match v.fading { Some(true) => "yes", Some(false) => "no", None => "-" };
        s += &format!("{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\n", v.arm, f(v.g[0]), f(v.g[1]), f(v.g[2]), v.call, fd, v.cost, if v.keep { "KEEP" } else { "-" });
    }
    let kept: Vec<&str> = vs.iter().filter(|v| v.keep).map(|v| v.arm.as_str()).collect();
    s += &format!("kept: {}\n", if kept.is_empty() { "none".to_string() } else { kept.join(" ") });
    std::fs::write(&op, &s).map_err(|e| format!("{op}: {e}"))?;
    print!("{s}");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("v52_verdict: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(v: &[(&str, u32, u32, f64)]) -> Points {
        v.iter().map(|&(a, s, t, x)| ((a.to_string(), s, t), x)).collect()
    }

    fn base(gap: &dyn Fn(&str, u32) -> f64) -> Points {
        let mut v = vec![];
        for a in std::iter::once("B").chain(PARTS) {
            for (j, s) in SEEDS.iter().enumerate() {
                for t in STEPS {
                    // seeds differ by 0.002 everywhere: s_p = sqrt(0.002^2 / 2) = 0.0014142
                    v.push((a, *s, t, 1.6 - gap(a, t) + if j == 0 { 0.001 } else { -0.001 }));
                }
            }
        }
        rows(&v)
    }

    fn cost(slow: &str, f: f64) -> Vec<Vec<String>> {
        std::iter::once("B").chain(PARTS).flat_map(|a| SEEDS.map(|s| vec![a.to_string(), s.to_string(), format!("{}", if a == slow { 600.0 * f } else { 600.0 }), "4000".into()])).collect()
    }

    #[test]
    fn a_steady_gain_over_the_bar_is_kept_and_the_rest_are_judged_by_the_rules() {
        let p = base(&|a, t| match (a, t) {
            ("M", _) => 0.005,                // steady WIN
            ("S", 2000) => 0.010,             // WIN at 4000 but fading (0.004 < 0.010 - 0.0014)
            ("S", 4000) => 0.004,
            ("CB", _) => 0.006,               // WIN but 30% slower
            ("N", _) => -0.004,               // LOSE
            ("R", _) => 0.002,                // TIE (bar = max(2 x 0.0014142, 0.002) = 0.0028284)
            _ => 0.0,
        });
        let (sp, bar, _, vs) = judge(&p, &cost("CB", 1.3), &[]).unwrap();
        assert!((sp - 0.0014142).abs() < 1e-6, "{sp}");
        assert!((bar - 0.0028284).abs() < 1e-6, "{bar}");
        let get = |a: &str| vs.iter().find(|v| v.arm == a).unwrap();
        assert_eq!((get("M").call, get("M").keep), ("WIN", true));
        assert_eq!((get("S").call, get("S").fading, get("S").keep), ("WIN", Some(true), false));
        assert_eq!((get("CB").call, get("CB").keep), ("WIN", false));
        assert_eq!(get("N").call, "LOSE");
        assert_eq!((get("R").call, get("R").keep), ("TIE", false));
        assert_eq!((get("Pr").call, get("Pr").keep), ("TIE", false));
    }

    #[test]
    fn the_bar_never_drops_under_0_002_and_a_missing_point_is_an_error() {
        let mut p = base(&|_, _| 0.0);
        for v in p.values_mut() {
            *v = 1.6; // no seed noise at all
        }
        let (sp, bar, _, _) = judge(&p, &cost("", 1.0), &[]).unwrap();
        assert_eq!((sp, bar), (0.0, 0.002));
        p.remove(&("R".to_string(), 52, 2000));
        assert!(judge(&p, &cost("", 1.0), &[]).is_err());
    }

    #[test]
    fn a_broke_arm_is_never_kept_and_stays_out_of_the_noise() {
        // M wins steadily; Pr and S broke on seed 41 (no seed 52, S has a NaN); without them s_p is unchanged
        let mut p = base(&|a, _| if a == "M" { 0.005 } else { 0.0 });
        for k in [("Pr", 52), ("S", 52)] {
            for t in STEPS {
                p.remove(&(k.0.to_string(), k.1, t));
            }
        }
        p.insert(("Pr".into(), 41, 4000), 1.601 + 0.125); // B(41, 4000) = 1.601
        p.insert(("S".into(), 41, 2000), f64::NAN);
        p.insert(("S".into(), 41, 4000), f64::NAN);
        let mut c = cost("S", 1.35);
        c.retain(|r| !((r[0] == "Pr" || r[0] == "S") && r[1] == "52"));
        let broke = vec!["Pr".to_string(), "S".to_string()];
        let (sp, _, _, vs) = judge(&p, &c, &broke).unwrap();
        assert!((sp - 0.0014142).abs() < 1e-6, "{sp}");
        let get = |a: &str| vs.iter().find(|v| v.arm == a).unwrap();
        assert_eq!((get("M").call, get("M").keep), ("WIN", true));
        let pr = get("Pr");
        assert_eq!((pr.call, pr.keep, pr.fading), ("BROKE", false, None));
        assert!((pr.g[2].unwrap() + 0.125).abs() < 1e-9, "{:?}", pr.g);
        let s = get("S");
        assert_eq!((s.call, s.keep, s.g[1], s.g[2]), ("BROKE", false, None, None));
        assert!((s.cost - 1.35).abs() < 1e-9);
        // without --broke the same points are an error, and B can never be listed as broke
        assert!(judge(&p, &c, &[]).is_err());
        assert!(judge(&p, &c, &["B".to_string()]).is_err());
    }

    #[test]
    fn other_parts_are_judged_against_b_with_their_own_noise() {
        // Stage 1c (Amendment S1-E): only B and the given parts; Stage 1b's arms are absent and not needed
        let parts = ["P", "DA", "Y", "TH", "Hn"];
        let mut v = vec![];
        for a in std::iter::once("B").chain(parts) {
            for (j, s) in SEEDS.iter().enumerate() {
                for t in STEPS {
                    let g = match a { "DA" => 0.004, "Hn" => -0.003, _ => 0.0 };
                    v.push((a, *s, t, 1.6 - g + if j == 0 { 0.001 } else { -0.001 }));
                }
            }
        }
        let p = rows(&v);
        let c: Vec<Vec<String>> = std::iter::once("B").chain(parts).flat_map(|a| SEEDS.map(|s| vec![a.to_string(), s.to_string(), "600".into(), "4000".into()])).collect();
        let (sp, bar, _, vs) = judge_parts(&p, &c, &[], &parts).unwrap();
        assert!((sp - 0.0014142).abs() < 1e-6, "{sp}");
        assert!((bar - 0.0028284).abs() < 1e-6, "{bar}");
        assert_eq!(vs.iter().map(|v| v.arm.as_str()).collect::<Vec<_>>(), parts);
        let get = |a: &str| vs.iter().find(|v| v.arm == a).unwrap();
        assert_eq!(get("DA").call, "WIN");
        assert_eq!(get("Hn").call, "LOSE");
        assert_eq!(get("P").call, "TIE");
        // a broke name outside the given parts is an error; Stage 1b's default parts would miss points here
        assert!(judge_parts(&p, &c, &["R".to_string()], &parts).is_err());
        assert!(judge(&p, &c, &[]).is_err());
    }
}
