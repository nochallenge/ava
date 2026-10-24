// TESTING / DESIGN ONLY. (v29 self-modelling) Knowing when she knows.
//
//   cargo run --release --example know_what -- <explore|screen|confirm> <answers-dir> [out.txt]
//
// After answering a science question (WP4, `read_facts`), Ava has more than her answer: she
// felt how her recall went. People judge their own answers from the same kind of evidence --
// how clearly one memory beat the rest, how strongly anything came back at all (metacognition,
// anterior prefrontal cortex) -- and they learn, from being told right or wrong, how far that
// feeling can be trusted. Here:
//   feeling     margin (best vs runner-up), lead (best vs the mean of the rest), fluency (how
//               strong the best recall was), and how many choices there were
//   self-model  P(right | feeling), learned only from her own Train answers with right/wrong
//               feedback (a logistic readout; graded, no thresholds on the feelings)
//   decision    under right - wrong scoring an answer is worth p - (1 - p); she answers when
//               that is positive, otherwise she says she doesn't know
// She is never told which set a question is from (Easy or Challenge).
// Input: the frozen WP4 answer files (read_dev7.tsv for Train/Dev, read_test.tsv for Test).
// Nothing is re-answered: only the decision to speak is new.
use std::collections::HashMap;

#[derive(Clone)]
struct Q { set: String, right: bool, wp4: bool, scores: Vec<f64> }

fn load(path: &str) -> Vec<Q> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    // read_test.tsv has a header; read_dev7.tsv has none (set, id, key, choice, margin, scores, question, choices)
    let first: Vec<&str> = text.lines().next().expect("a line").split('\t').collect();
    let headed = first[0] == "set";
    let head: Vec<&str> = if headed { first } else { vec!["set", "id", "key", "choice", "margin", "scores"] };
    let col = |n: &str| head.iter().position(|h| *h == n).unwrap_or_else(|| panic!("no column {n} in {path}"));
    let (cs, ck, cc, cm, csc) = (col("set"), col("key"), col("choice"), col("margin"), col("scores"));
    // WP4's own decision to answer: exact in read_test.tsv; from the 3-decimal margin in read_dev7.tsv
    let ca = head.iter().position(|h| *h == "answered");
    text.lines().skip(headed as usize).filter(|l| !l.is_empty()).map(|l| {
        let p: Vec<&str> = l.split('\t').collect();
        let margin: f64 = p[cm].parse().unwrap();
        Q { set: p[cs].to_string(), right: p[ck] == p[cc], wp4: ca.map_or(margin >= 0.01, |c| p[c] == "1"),
            scores: p[csc].split(',').map(|x| x.parse().unwrap()).collect() }
    }).collect()
}

/// her feeling about one answer: [margin, lead, fluency, chance]
fn feel(q: &Q) -> [f64; 4] {
    let mut s = q.scores.clone();
    s.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let best = s[0].max(1e-9);
    // margin from the scores themselves (both files keep scores; their margins are rounded differently)
    let margin = (s[0] - s[1]) / best;
    let rest = s[1..].iter().sum::<f64>() / (s.len() - 1).max(1) as f64;
    [margin, (best - rest) / best, best.ln(), 1.0 / s.len() as f64]
}

/// which feelings the self-model reads
#[derive(Clone, Copy, PartialEq)]
enum Use { Margin, All }

fn xs(q: &Q, u: Use) -> Vec<f64> {
    let f = feel(q);
    match u { Use::Margin => vec![f[0]], Use::All => f.to_vec() }
}

struct Model { mean: Vec<f64>, sd: Vec<f64>, w: Vec<f64> }

impl Model {
    /// logistic readout, full-batch gradient descent, deterministic
    fn fit(qs: &[Q], u: Use) -> Model {
        let x: Vec<Vec<f64>> = qs.iter().map(|q| xs(q, u)).collect();
        let d = x[0].len();
        let n = x.len() as f64;
        let mean: Vec<f64> = (0..d).map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n).collect();
        let sd: Vec<f64> = (0..d).map(|j| (x.iter().map(|r| (r[j] - mean[j]).powi(2)).sum::<f64>() / n).sqrt().max(1e-9)).collect();
        let z: Vec<Vec<f64>> = x.iter().map(|r| (0..d).map(|j| (r[j] - mean[j]) / sd[j]).collect()).collect();
        let y: Vec<f64> = qs.iter().map(|q| q.right as u8 as f64).collect();
        let mut w = vec![0.0; d + 1];
        for _ in 0..3000 {
            let mut g = vec![0.0; d + 1];
            for (r, &t) in z.iter().zip(&y) {
                let e = sig(w[0] + (0..d).map(|j| w[j + 1] * r[j]).sum::<f64>()) - t;
                g[0] += e;
                for j in 0..d { g[j + 1] += e * r[j]; }
            }
            for j in 0..=d { w[j] -= 0.5 * (g[j] / n + if j > 0 { 1e-3 * w[j] } else { 0.0 }); }
        }
        Model { mean, sd, w }
    }
    fn p(&self, q: &Q, u: Use) -> f64 {
        let x = xs(q, u);
        sig(self.w[0] + x.iter().enumerate().map(|(j, v)| self.w[j + 1] * (v - self.mean[j]) / self.sd[j]).sum::<f64>())
    }
}

fn sig(v: f64) -> f64 { 1.0 / (1.0 + (-v).exp()) }

/// (answered, right, wrong) for a decision
fn tally(qs: &[Q], speak: impl Fn(&Q) -> bool) -> (usize, usize, usize) {
    let (mut a, mut r) = (0, 0);
    for q in qs { if speak(q) { a += 1; if q.right { r += 1; } } }
    (a, r, a - r)
}

fn line(name: &str, t: (usize, usize, usize), n: usize) -> String {
    format!("   {name:<34} answered {:>5}/{n:<5} right {:>5} wrong {:>5}  right-wrong {:>+6}  share right {:5.1}%",
            t.0, t.1, t.2, t.1 as i64 - t.2 as i64, 100.0 * t.1 as f64 / t.0.max(1) as f64)
}

fn splitmix(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// paired bootstrap of the per-question difference in (right - wrong): splitmix64 seed 1, 20,000 resamples
fn boot(d: &[i64]) -> (f64, f64, f64) {
    let n = d.len();
    let mut s = 1u64;
    let mut v: Vec<f64> = (0..20_000).map(|_| (0..n).map(|_| d[(splitmix(&mut s) % n as u64) as usize]).sum::<i64>() as f64).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (d.iter().sum::<i64>() as f64, v[499], v[19_499])
}

fn score(q: &Q, speak: bool) -> i64 { if !speak { 0 } else if q.right { 1 } else { -1 } }

fn report(out: &mut String, title: &str, qs: &[Q], m: &Model, u: Use) {
    let sets: Vec<String> = { let mut s: Vec<String> = qs.iter().map(|q| q.set.clone()).collect(); s.sort(); s.dedup(); s };
    out.push_str(&format!("{title}\n"));
    for set in sets.iter().map(Some).chain(std::iter::once(None)) {
        let sub: Vec<Q> = qs.iter().filter(|q| set.map_or(true, |s| &q.set == s)).cloned().collect();
        let name = set.cloned().unwrap_or_else(|| "pooled".into());
        out.push_str(&format!("  {name} ({} questions)\n", sub.len()));
        out.push_str(&line("answer every question", tally(&sub, |_| true), sub.len()));
        out.push('\n');
        out.push_str(&line("WP4 margin >= 0.01 (frozen)", tally(&sub, |q| q.wp4), sub.len()));
        out.push('\n');
        out.push_str(&line("self-model: answer when p > 1/2", tally(&sub, |q| m.p(q, u) > 0.5), sub.len()));
        out.push('\n');
    }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mode = a.get(1).map(String::as_str).unwrap_or("explore");
    let dir = a.get(2).expect("answers dir (holding read_dev7.tsv and read_test.tsv)");
    let dev7 = load(&format!("{dir}/read_dev7.tsv"));
    let train: Vec<Q> = dev7.iter().filter(|q| q.set.ends_with("Train")).cloned().collect();
    let mut out = String::new();
    out.push_str(&format!("self-model learned from {} Train answers (Easy + Challenge, set not shown to her)\n", train.len()));
    match mode {
        "explore" => {
            // Train only: 5-fold cross-validation inside Train, to choose which feelings she reads
            for (u, un) in [(Use::Margin, "margin only"), (Use::All, "margin + lead + fluency + chance")] {
                let mut d_self: Vec<i64> = Vec::new();
                let (mut tot_self, mut tot_wp4, mut tot_all) = (0i64, 0i64, 0i64);
                for f in 0..5 {
                    let fit: Vec<Q> = train.iter().enumerate().filter(|(i, _)| i % 5 != f).map(|(_, q)| q.clone()).collect();
                    let held: Vec<&Q> = train.iter().enumerate().filter(|(i, _)| i % 5 == f).map(|(_, q)| q).collect();
                    let m = Model::fit(&fit, u);
                    for q in held {
                        let s = score(q, m.p(q, u) > 0.5);
                        tot_self += s; tot_wp4 += score(q, q.wp4); tot_all += score(q, true);
                        d_self.push(s - score(q, q.wp4));
                    }
                }
                out.push_str(&format!("  5-fold on Train, {un}: right-wrong self {tot_self:+}  WP4 {tot_wp4:+}  answer-all {tot_all:+}  (self - WP4 {:+})\n", tot_self - tot_wp4));
            }
            let m = Model::fit(&train, Use::All);
            out.push_str(&format!("  weights on standardized feelings (bias, margin, lead, fluency, chance): {:?}\n",
                                  m.w.iter().map(|w| format!("{w:.3}")).collect::<Vec<_>>()));
            report(&mut out, "in-sample on Train (optimistic):", &train, &m, Use::All);
        }
        "screen" | "confirm" => {
            let u = match std::env::var("AVA_FEEL").as_deref() { Ok("margin") => Use::Margin, _ => Use::All };
            let m = Model::fit(&train, u);
            let qs: Vec<Q> = if mode == "screen" {
                dev7.iter().filter(|q| q.set.ends_with("Dev")).cloned().collect()
            } else {
                load(&format!("{dir}/read_test.tsv"))
            };
            report(&mut out, &format!("{mode}: {} questions", qs.len()), &qs, &m, u);
            let d: Vec<i64> = qs.iter().map(|q| score(q, m.p(q, u) > 0.5) - score(q, q.wp4)).collect();
            let (pt, lo, hi) = boot(&d);
            out.push_str(&format!("  pooled right-wrong, self-model minus WP4: {pt:+}  95% CI {lo:+} .. {hi:+}  (paired bootstrap, splitmix64 seed 1, 20,000 resamples)\n"));
            let mut per: HashMap<String, i64> = HashMap::new();
            for (q, x) in qs.iter().zip(&d) { *per.entry(q.set.clone()).or_default() += x; }
            let mut pv: Vec<_> = per.into_iter().collect();
            pv.sort();
            out.push_str(&format!("  by set (self-model minus WP4): {}\n", pv.iter().map(|(k, v)| format!("{k} {v:+}")).collect::<Vec<_>>().join(", ")));
            let pass = lo > 0.0;
            out.push_str(&format!("VERDICT ({mode}; pooled right-wrong gain over WP4 with 95% CI above 0): {}\n", if pass { "PASS" } else { "FAIL" }));
        }
        _ => panic!("mode: explore | screen | confirm"),
    }
    print!("{out}");
    if let Some(p) = a.get(3) { std::fs::write(p, &out).expect("write output"); }
}
