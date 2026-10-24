// TESTING / DESIGN ONLY.
// v29 ADDRESSING BENCH. Offline, seconds, one process, no brain loaded.
//
//   cargo build --release --example address_bench
//   address_bench [frames-file] [exploratory frames-file]
//       (default frames file: scratchpad/v29/frames.bin, from frames_build)
//
// The question asked: for each day-2 question of the six USED blocks, which of that block's
// 20 taught lessons does it address — or does she abstain? A taught question's right lesson
// is its own; for an untaught question any named lesson is wrong. Scored her way: right - wrong.
//
// She answers only when her estimated P(right) > 1/2, the break-even under right - wrong. That
// P is LEARNED, not set: a logistic map from the match evidence (top score a1, lead a1 - a2)
// fitted on five blocks and applied to the sixth (leave-one-block-out), so no block is scored
// with a calibration fitted on itself.
//
// Encodings (pre-registered in scratchpad/v29/address_prereg.md before any candidate ran):
//   B0   compose.rs activate() as it runs by default: spec-weighted word overlap
//   B1   B0 + the same-lexeme rule (AVA_LEMMA, UniMorph)
//   C1   HDC: words as random D=8192-bit vectors, spec-weighted majority bundle, cleanup
//   C1o  C1 + bound bigrams (rho(a) XOR b): word ORDER in the code
//   C1+lemma, C1o+lemma   the same over lexemes
//   C2   B0 + frames learned from reading (frames_build.rs, DIRT-style slot fillers)
//   C2+lemma
// Nothing here reads the fact files or the fresh confirmation blocks.
#[path = "common/addr.rs"]
mod addr;
use addr::*;
use std::collections::HashMap;

const D: usize = 8192;
const W: usize = D / 64;
const KINDS: [&str; 3] = ["as-taught", "paraphrase", "untaught"];

#[derive(Clone, Copy, PartialEq, Debug)]
enum E { B0, B1, C1, C1o, C1L, C1oL, C2, C2L, C2x, C2xL, X1L }
fn ename(e: E) -> &'static str {
    match e { E::B0 => "B0", E::B1 => "B1 lemma", E::C1 => "C1 hdc", E::C1o => "C1o hdc+order", E::C1L => "C1+lemma",
        E::C1oL => "C1o+lemma", E::C2 => "C2 frames", E::C2L => "C2+lemma", E::C2x => "C2x (EXPL)", E::C2xL => "C2x+lemma (EXPL)", E::X1L => "X1 cos+lemma (EXPL)" }
}

// ---------------------------------------------------------------- HDC
fn rand_hv(seed: u64) -> Vec<u64> { let mut s = seed; (0..W).map(|_| splitmix(&mut s)).collect() }
fn majority(items: &[(Vec<u64>, f32)], tie: &[u64]) -> Vec<u64> {
    let mut acc = vec![0f32; D];
    for (v, w) in items {
        for k in 0..W {
            let x = v[k];
            let a = &mut acc[k * 64..k * 64 + 64];
            for b in 0..64 { a[b] += if (x >> b) & 1 == 1 { *w } else { -*w }; }
        }
    }
    (0..W).map(|k| {
        let mut o = 0u64;
        for b in 0..64 {
            let a = acc[k * 64 + b];
            if a > 0.0 || (a == 0.0 && (tie[k] >> b) & 1 == 1) { o |= 1 << b; }
        }
        o
    }).collect()
}
fn rot1(v: &[u64]) -> Vec<u64> { (0..W).map(|k| (v[k] << 1) | (v[(k + W - 1) % W] >> 63)).collect() }
fn xor(a: &[u64], b: &[u64]) -> Vec<u64> { a.iter().zip(b).map(|(x, y)| x ^ y).collect() }
fn hsim(a: &[u64], b: &[u64]) -> f32 {
    let h: u32 = a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum();
    1.0 - 2.0 * h as f32 / D as f32
}
struct Hdc<'a> { lex: &'a Lex, lemma: bool, order: bool, tie: Vec<u64>, cache: HashMap<String, Vec<u64>> }
impl<'a> Hdc<'a> {
    fn new(lex: &'a Lex, lemma: bool, order: bool) -> Self {
        Hdc { lex, lemma, order, tie: rand_hv(fnv("tie-breaker v29")), cache: HashMap::new() }
    }
    fn word(&mut self, w: &str) -> Vec<u64> {
        if let Some(v) = self.cache.get(w) { return v.clone(); }
        let v = match (self.lemma, self.lex.of.get(w)) {
            (true, Some(ids)) if ids.len() == 1 => rand_hv(fnv(&format!("lexeme:{}", self.lex.name[ids[0] as usize]))),
            (true, Some(ids)) => {
                let items: Vec<(Vec<u64>, f32)> = ids.iter().map(|&i| (rand_hv(fnv(&format!("lexeme:{}", self.lex.name[i as usize]))), 1.0)).collect();
                majority(&items, &self.tie)
            }
            _ => rand_hv(fnv(&format!("word:{w}"))),
        };
        self.cache.insert(w.to_string(), v.clone());
        v
    }
    fn encode(&mut self, toks: &[String], spec: &Spec) -> Vec<u64> {
        let mut set: Vec<&str> = toks.iter().map(|s| s.as_str()).collect();
        set.sort_unstable(); set.dedup();
        let mut items: Vec<(Vec<u64>, f32)> = set.iter().map(|w| (self.word(w), spec.of(w))).collect();
        if self.order {
            let mut bg: Vec<(&str, &str)> = toks.windows(2).map(|p| (p[0].as_str(), p[1].as_str())).collect();
            bg.sort_unstable(); bg.dedup();
            for (a, b) in bg {
                let v = xor(&rot1(&self.word(a)), &self.word(b));
                items.push((v, (spec.of(a) * spec.of(b)).sqrt()));
            }
        }
        majority(&items, &self.tie)
    }
}

// ---------------------------------------------------------------- word overlap (compose activate)
fn set_of(ws: &[String]) -> Vec<&str> { let mut v: Vec<&str> = ws.iter().map(|w| w.as_str()).collect(); v.sort_unstable(); v.dedup(); v }
/// compose.rs activate() for one engram, AVA_CUE=0 (no stem, no answer words): a = shared / all.
/// `near(w, v)` is the graded credit for a word that is not present (0 in v26, same lexeme
/// under AVA_LEMMA, frame similarity under C2).
fn overlap(qs: &[&str], es: &[&str], spec: &Spec, near: &mut dyn FnMut(&str, &str) -> f32) -> f32 {
    let mut hit = |w: &str, ws: &[&str]| -> f32 {
        if ws.binary_search(&w).is_ok() { 1.0 } else { ws.iter().map(|v| near(w, v)).fold(0.0, f32::max) }
    };
    let (mut shared, mut all) = (0f32, 0f32);
    for w in qs { let x = spec.of(w); all += x; shared += x * hit(w, es); }
    for w in es { all += spec.of(w) * (1.0 - hit(w, qs)); }
    if all > 0.0 { shared / all } else { 0.0 }
}
/// the most active engram, a1, a2 — compose's order (first max wins; nothing at or below
/// `floor` is ever "best")
fn top2(s: &[f32], floor: f32) -> (Option<usize>, f32, f32) {
    let (mut best, mut second): (Option<(usize, f32)>, f32) = (None, floor);
    for (i, &a) in s.iter().enumerate() {
        if a > best.map_or(floor, |b| b.1) { second = best.map_or(floor, |b| b.1); best = Some((i, a)); }
        else if a > second { second = a; }
    }
    match best { Some((i, a)) => (Some(i), a, second), None => (None, 0.0, 0.0) }
}

// ---------------------------------------------------------------- calibration (logistic, IRLS)
fn sigm(z: f64) -> f64 { 1.0 / (1.0 + (-z).exp()) }
fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> [f64; 3] {
    for c in 0..3 {
        let p = (c..3).max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs())).unwrap();
        a.swap(c, p); b.swap(c, p);
        if a[c][c].abs() < 1e-300 { continue; }
        for r in 0..3 {
            if r == c { continue; }
            let f = a[r][c] / a[c][c];
            for k in 0..3 { a[r][k] -= f * a[c][k]; }
            b[r] -= f * b[c];
        }
    }
    [0, 1, 2].map(|i| if a[i][i].abs() < 1e-300 { 0.0 } else { b[i] / a[i][i] })
}
fn fit(x: &[[f64; 3]], y: &[f64]) -> [f64; 3] {
    let lam = 1e-4;
    let mut beta = [0f64; 3];
    for _ in 0..50 {
        let (mut g, mut h) = ([0f64; 3], [[0f64; 3]; 3]);
        for (xi, &yi) in x.iter().zip(y) {
            let p = sigm(beta[0] * xi[0] + beta[1] * xi[1] + beta[2] * xi[2]);
            let w = (p * (1.0 - p)).max(1e-12);
            for a in 0..3 { g[a] += xi[a] * (yi - p); for b in 0..3 { h[a][b] += w * xi[a] * xi[b]; } }
        }
        for a in 1..3 { g[a] -= lam * beta[a]; h[a][a] += lam; }
        let step = solve3(h, g);
        let mut m = 0f64;
        for a in 0..3 { beta[a] += step[a]; m = m.max(step[a].abs()); }
        if m < 1e-10 { break; }
    }
    beta
}
fn feat(a1: f32, a2: f32) -> [f64; 3] { [1.0, a1 as f64, (a1 - a2) as f64] }

// ---------------------------------------------------------------- results
#[derive(Clone)]
struct Q { blk: usize, qi: usize, kind: u8, dom: usize, own: Option<usize> }
#[derive(Clone, Default)]
struct Res { top: Vec<Option<usize>>, a1: Vec<f32>, a2: Vec<f32>, p: Vec<f64>, all: Vec<Vec<f32>> }
impl Res {
    fn call(&self, i: usize, gated: bool) -> Option<usize> { if gated && self.p[i] <= 0.5 { None } else { self.top[i] } }
}
/// +1 right, -1 wrong, 0 abstain. equiv: a lesson with the same answer text counts as the own one
fn score(blocks: &[Block], q: &Q, call: Option<usize>, equiv: bool) -> i32 {
    let Some(c) = call else { return 0 };
    match q.own {
        None => -1,
        Some(o) if o == c => 1,
        Some(o) if equiv && norm(&blocks[q.blk].lessons[o].a_raw) == norm(&blocks[q.blk].lessons[c].a_raw) => 1,
        _ => -1,
    }
}
#[derive(Default, Clone, Copy)]
struct T { r: i32, w: i32, a: i32 }
impl T {
    fn add(&mut self, s: i32) { match s { 1 => self.r += 1, -1 => self.w += 1, _ => self.a += 1 } }
    fn net(&self) -> i32 { self.r - self.w }
    fn share(&self) -> f64 { self.r as f64 / (self.r + self.w).max(1) as f64 * 100.0 }
    fn cell(&self) -> String { format!("{:>3} {:>3} {:>3} {:>+4}", self.r, self.w, self.a, self.net()) }
}
fn tally(blocks: &[Block], qs: &[Q], res: &Res, gated: bool, equiv: bool) -> [[T; 4]; 3] {
    let mut t = [[T::default(); 4]; 3];
    for (i, q) in qs.iter().enumerate() {
        let s = score(blocks, q, res.call(i, gated), equiv);
        t[q.kind as usize][q.dom].add(s);
        t[q.kind as usize][3].add(s);
    }
    t
}
fn boot_ci(d: &[i64]) -> (i64, i64, i64, f64) {
    let sum: i64 = d.iter().sum();
    let mut rng = 1u64;
    let mut boot: Vec<i64> = (0..20_000).map(|_| (0..d.len()).map(|_| d[(splitmix(&mut rng) % d.len() as u64) as usize]).sum()).collect();
    boot.sort_unstable();
    (sum, boot[500], boot[19_499], boot.iter().filter(|&&x| x <= 0).count() as f64 / boot.len() as f64)
}

// ---------------------------------------------------------------- the saved real runs
struct Said { reply: String, rem: Option<(usize, f32, f32)> }
fn parse_run(path: &str, n: usize) -> Option<Vec<Said>> {
    let t = std::fs::read_to_string(path).ok()?.replace("\r\n", "\n");
    let replies: Vec<String> = t.lines().filter_map(|l| l.find("ava> ").map(|i| norm(&l[i + 5..]))).collect();
    let reads: Vec<Option<(usize, f32, f32)>> = t.lines().filter(|l| l.contains("[reading:")).map(|l| {
        let i = l.find("reminds her of lesson ")?;
        let r = &l[i + 22..];
        let n: usize = r.split(' ').next()?.parse().ok()?;
        let p = r.find('(')? + 1;
        let a1: f32 = r[p..].split('%').next()?.trim().parse().ok()?;
        let q = r.find("runner-up ")? + 10;
        let a2: f32 = r[q..].split('%').next()?.trim().parse().ok()?;
        Some((n - 1, a1, a2))
    }).collect();
    if replies.len() < n || reads.len() < n { return None; }
    Some((0..n).map(|i| Said { reply: replies[i].clone(), rem: reads[i] }).collect())
}
/// teach_test's grading of what she actually said: +1 right, -1 a taught answer that is wrong here, 0 other
fn harness(blocks: &[Block], q: &Q, reply: &str) -> i32 {
    let b = &blocks[q.blk];
    let want: Vec<String> = b.lessons.iter().map(|l| norm(&l.a_raw)).collect();
    match q.own {
        Some(o) if reply == want[o] => 1,
        _ if want.iter().any(|w| w == reply) => -1,
        _ => 0,
    }
}

fn main() {
    let t0 = std::time::Instant::now();
    ram_guard("start");
    let args: Vec<String> = std::env::args().collect();
    let fpath = args.get(1).cloned().unwrap_or_else(|| format!("{SCRATCH}/v29/frames.bin"));
    let xpath = args.get(2).cloned();
    println!("// TESTING / DESIGN ONLY.");
    println!("== v29 ADDRESSING BENCH: which taught lesson does each day-2 question address? ==");
    println!("   pre-registration: scratchpad/v29/address_prereg.md (written before any candidate ran)\n");

    // ---------------------------------------------------------------- data
    let blocks: Vec<Block> = BLOCKS.iter().map(|(r, t, d)| load_block(r, t, d)).collect();
    let mut qs: Vec<Q> = Vec::new();
    for (bi, b) in blocks.iter().enumerate() {
        assert_eq!(b.lessons.len(), 20, "{}: lessons", b.tag);
        assert_eq!(b.qs.len(), 80, "{}: questions", b.tag);
        for (qi, q) in b.qs.iter().enumerate() { qs.push(Q { blk: bi, qi, kind: q.kind, dom: b.dom, own: q.fact }); }
    }
    // the lessons as she stored them: compose prints each one as it notes it on day 1
    let mut same = 0;
    for b in &blocks {
        let t = std::fs::read_to_string(format!("{}/day1_out.txt", b.dir)).unwrap_or_default();
        let noted: Vec<(String, String)> = t.lines().filter_map(|l| {
            let i = l.find("[noted for tonight: \"")? + 21;
            let r = &l[i..];
            let j = r.find("\" -> \"")?;
            let a = &r[j + 6..];
            Some((r[..j].to_string(), a[..a.rfind("\"]")?].to_string()))
        }).collect();
        for (k, l) in b.lessons.iter().enumerate() {
            if noted.get(k).map_or(false, |n| n.0 == l.q.join(" ") && n.1 == l.a.join(" ")) { same += 1; }
        }
    }
    println!("   6 used blocks: {} lessons, {} questions ({} as-taught, {} paraphrase, {} untaught)",
        blocks.len() * 20, qs.len(), qs.iter().filter(|q| q.kind == 0).count(), qs.iter().filter(|q| q.kind == 1).count(),
        qs.iter().filter(|q| q.kind == 2).count());
    println!("   lesson tokens identical to what compose noted on day 1: {same}/120");
    let (life_n, life_c, n_train) = life_writes(15_000, 1_200);
    println!("   her life (spec weights): {n_train} training exchanges of triples.tsv (compose says 19844), {life_n} cortex writes");
    let specs: Vec<Spec> = blocks.iter().map(|b| Spec::new(life_n, &life_c, b)).collect();
    println!("   sleep replay words per block: {}", blocks.iter().zip(&specs).map(|(b, s)| format!("{} {}", b.tag, s.sleep_words)).collect::<Vec<_>>().join(", "));
    let lex = Lex::load(&format!("{AVA}/data/unimorph-eng.tsv"));
    println!("   UniMorph: {} forms, {} lexemes", lex.of.len(), lex.name.len());
    let frames = Frames::load(&fpath).ok();
    match &frames { Some(f) => println!("   frames: {} profiled words from {fpath}", f.words.len()), None => println!("   frames: none at {fpath} (C2 skipped)") }
    let xframes = xpath.as_ref().and_then(|p| Frames::load(p).ok());
    if let (Some(p), Some(f)) = (&xpath, &xframes) { println!("   EXPLORATORY frames: {} profiled words from {p}", f.words.len()); }
    let t_data = t0.elapsed().as_secs_f32();
    ram_guard("after data");

    // ---------------------------------------------------------------- scores per encoding
    let mut encs: Vec<E> = vec![E::B0, E::B1, E::C1, E::C1o, E::C1L, E::C1oL];
    // BENCH_BASELINES_ONLY=1: B0 and B1 alone (the validity check, run before any candidate)
    let base_only = std::env::var("BENCH_BASELINES_ONLY").map_or(false, |v| v == "1");
    if base_only { encs.truncate(2); }
    if frames.is_some() && !base_only { encs.extend([E::C2, E::C2L]); }
    if xframes.is_some() && !base_only { encs.extend([E::C2x, E::C2xL]); }
    // EXPLORATORY (added after the pre-registered results): is C1+lemma's gain the HDC code, or just its geometry?
    // X1 = the same lexeme matching, scored as a deterministic cosine of spec weights (what a weighted
    // majority bundle approximates), no random vectors
    if !base_only { encs.push(E::X1L); }
    let mut simc: HashMap<(usize, usize), f32> = HashMap::new();
    let mut simx: HashMap<(usize, usize), f32> = HashMap::new();
    let mut results: Vec<(E, Res, f32)> = Vec::new();
    for &e in &encs {
        let te = std::time::Instant::now();
        let mut res = Res::default();
        let floor = if matches!(e, E::C1 | E::C1o | E::C1L | E::C1oL) { f32::NEG_INFINITY } else { 0.0 };
        let mut hdc = match e {
            E::C1 => Some(Hdc::new(&lex, false, false)), E::C1o => Some(Hdc::new(&lex, false, true)),
            E::C1L => Some(Hdc::new(&lex, true, false)), E::C1oL => Some(Hdc::new(&lex, true, true)), _ => None,
        };
        for (bi, b) in blocks.iter().enumerate() {
            let sp = &specs[bi];
            let lv: Vec<Vec<u64>> = match hdc.as_mut() { Some(h) => b.lessons.iter().map(|l| h.encode(&l.q, sp)).collect(), None => Vec::new() };
            for q in &b.qs {
                let s: Vec<f32> = match hdc.as_mut() {
                    Some(h) => { let v = h.encode(&q.toks, sp); lv.iter().map(|x| hsim(&v, x)).collect() }
                    None => {
                        let qset = set_of(&q.toks);
                        b.lessons.iter().map(|l| {
                            let es = set_of(&l.q);
                            let mut near = |w: &str, v: &str| -> f32 {
                                let lx: f32 = if matches!(e, E::B1 | E::C2L | E::C2xL | E::X1L) && lex.same(w, v) { 1.0 } else { 0.0 };
                                let fr = match e {
                                    E::C2 | E::C2L => frames.as_ref().map_or(0.0, |f| match (f.ix.get(w), f.ix.get(v)) {
                                        (Some(&i), Some(&j)) => *simc.entry((i.min(j), i.max(j))).or_insert_with(|| f.sim_ix(i, j)),
                                        _ => 0.0 }),
                                    E::C2x | E::C2xL => xframes.as_ref().map_or(0.0, |f| match (f.ix.get(w), f.ix.get(v)) {
                                        (Some(&i), Some(&j)) => *simx.entry((i.min(j), i.max(j))).or_insert_with(|| f.sim_ix(i, j)),
                                        _ => 0.0 }),
                                    _ => 0.0,
                                };
                                lx.max(fr)
                            };
                            if e == E::X1L { cosine(&qset, &es, sp, &mut near) } else { overlap(&qset, &es, sp, &mut near) }
                        }).collect()
                    }
                };
                let (top, a1, a2) = top2(&s, floor);
                res.top.push(top); res.a1.push(a1); res.a2.push(a2); res.all.push(s);
            }
        }
        // leave-one-block-out calibration
        res.p = vec![0.0; qs.len()];
        for hold in 0..blocks.len() {
            let (mut x, mut y) = (Vec::new(), Vec::new());
            for (i, q) in qs.iter().enumerate() {
                if q.blk == hold { continue; }
                x.push(feat(res.a1[i], res.a2[i]));
                y.push(if res.top[i].is_some() && res.top[i] == q.own { 1.0 } else { 0.0 });
            }
            let beta = fit(&x, &y);
            for (i, q) in qs.iter().enumerate() {
                if q.blk != hold { continue; }
                let f = feat(res.a1[i], res.a2[i]);
                res.p[i] = if res.top[i].is_none() { 0.0 } else { sigm(beta[0] * f[0] + beta[1] * f[1] + beta[2] * f[2]) };
            }
        }
        results.push((e, res, te.elapsed().as_secs_f32()));
        ram_guard("encoding");
    }
    let get = |e: E| &results.iter().find(|r| r.0 == e).unwrap().1;

    // ---------------------------------------------------------------- 1. validity
    println!("\n== 1. VALIDITY: does the bench reproduce the real harness? ==");
    println!("   B0 vs the saved v26 L0 day-2 runs; B1 vs the saved v28 lemma runs (same day-2 brains).");
    println!("   top = same \"reminds her of lesson N\";  a1/a2 = within the printed rounding (|diff| <= 0.5 points)");
    let mut val_rows: Vec<(E, Vec<Option<Vec<Said>>>)> = Vec::new();
    for (e, pat) in [(E::B0, "L0"), (E::B1, "lemma")] {
        let runs: Vec<Option<Vec<Said>>> = blocks.iter().map(|b| {
            let p = if pat == "L0" { format!("{}/{}L0_day2.txt", b.dir, b.tag) } else { format!("{SCRATCH}/v28/lemma/{}_day2.txt", b.tag) };
            parse_run(&p, 80)
        }).collect();
        let res = get(e);
        let (mut n, mut top_ok, mut a1_ok, mut a2_ok, mut none_both) = (0, 0, 0, 0, 0);
        let mut per = Vec::new();
        for (bi, run) in runs.iter().enumerate() {
            let Some(run) = run else { per.push(format!("{} missing", blocks[bi].tag)); continue };
            let mut bt = 0;
            for (i, q) in qs.iter().enumerate() {
                if q.blk != bi { continue; }
                n += 1;
                let s = &run[q.qi];
                match (s.rem, res.top[i]) {
                    (Some((l, x1, x2)), Some(t)) => {
                        if l == t { top_ok += 1; bt += 1; }
                        if (res.a1[i] * 100.0 - x1).abs() <= 0.5 + 1e-3 { a1_ok += 1; }
                        if (res.a2[i] * 100.0 - x2).abs() <= 0.5 + 1e-3 { a2_ok += 1; }
                    }
                    (None, None) => { top_ok += 1; a1_ok += 1; a2_ok += 1; none_both += 1; bt += 1; }
                    _ => {}
                }
            }
            per.push(format!("{} {bt}/80", blocks[bi].tag));
        }
        println!("   {:<9} vs {:<6}: top {top_ok}/{n}  a1 {a1_ok}/{n}  a2 {a2_ok}/{n}  (no engram in both: {none_both})   per block top: {}",
            ename(e), pat, per.join(", "));
        val_rows.push((e, runs));
    }
    // what she actually said vs the bench's call
    println!("\n   her actual answers (graded as teach_test: right / taught-but-wrong / other) vs the bench's call");
    println!("   (bench scored harness-equivalent: a lesson with the same answer text counts as right)");
    let mut hnet = [0i32; 2];
    for (k, (e, runs)) in val_rows.iter().enumerate() {
        let res = get(*e);
        for gated in [false, true] {
            let mut m = [[0i32; 3]; 3];
            let (mut bnet, mut hn, mut agree, mut nn) = (0, 0, 0, 0);
            let mut ans_match = (0, 0);
            for (i, q) in qs.iter().enumerate() {
                let Some(run) = &runs[q.blk] else { continue };
                let h = harness(&blocks, q, &run[q.qi].reply);
                let c = score(&blocks, q, res.call(i, gated), true);
                if q.kind != 0 { m[(1 - h) as usize][(1 - c) as usize] += 1; bnet += c; hn += h; nn += 1; if h == c { agree += 1; } }
                // when she gave some taught answer, was it the bench's top lesson's?
                let want: Vec<String> = blocks[q.blk].lessons.iter().map(|l| norm(&l.a_raw)).collect();
                if let Some(j) = want.iter().position(|w| *w == run[q.qi].reply) {
                    ans_match.1 += 1;
                    if res.top[i].map_or(false, |t| want[t] == want[j]) { ans_match.0 += 1; }
                }
            }
            if !gated { hnet[k] = hn; }
            println!("   {:<9} {}: on the {nn} paraphrase+untaught questions, same outcome {agree}/{nn} ({:.1}%)   harness net {hn:+}   bench net {bnet:+}",
                ename(*e), if gated { "gated  " } else { "ungated" }, agree as f64 / nn.max(1) as f64 * 100.0);
            println!("             rows = harness right / other / taught-but-wrong, cols = bench right / abstain / wrong: {:?}", m);
            if !gated { println!("             when she gave a taught answer, it was the bench's top lesson's answer: {}/{}", ans_match.0, ans_match.1); }
        }
    }
    let bnet = |e: E, gated: bool| -> i32 { qs.iter().enumerate().filter(|(_, q)| q.kind != 0).map(|(i, q)| score(&blocks, q, get(e).call(i, gated), true)).sum() };
    println!("   net change B1 - B0: real harness {:+}   bench ungated {:+}   bench gated {:+}",
        hnet[1] - hnet[0], bnet(E::B1, false) - bnet(E::B0, false), bnet(E::B1, true) - bnet(E::B0, true));

    // ---------------------------------------------------------------- 2. results
    println!("\n== 2. RESULTS (gated: answer iff learned P(right) > 1/2; strict: only the own lesson is right) ==");
    println!("   cells: right wrong abstain net;  ALL adds share right = right / (right + wrong)");
    println!("   net p+u = the bench's net over paraphrase + untaught (NOT the real harness's net, which has no gate)");
    println!("   {:<17} {:<11} {:<17} {:<17} {:<17} {}", "encoding", "kind", "rustdoc", "refnomicon", "ai", "ALL");
    for (e, res, _) in &results {
        let t = tally(&blocks, &qs, res, true, false);
        for k in 0..3 {
            println!("   {:<17} {:<11} {:<17} {:<17} {:<17} {} {:>5.1}%", if k == 0 { ename(*e) } else { "" }, KINDS[k],
                t[k][0].cell(), t[k][1].cell(), t[k][2].cell(), t[k][3].cell(), t[k][3].share());
        }
        let hn: Vec<i32> = (0..4).map(|d| t[1][d].net() + t[2][d].net()).collect();
        // (reviewer: this is the BENCH's gated net over paraphrase + untaught, not the real harness's)
        println!("   {:<17} {:<11} {:>17} {:>17} {:>17} {:>16}", "", "net p+u", format!("{:+}", hn[0]), format!("{:+}", hn[1]), format!("{:+}", hn[2]), format!("{:+}", hn[3]));
    }

    // ---------------------------------------------------------------- 3. against B1, the pre-registered bar
    println!("\n== 3. AGAINST B1 (lemma), the pre-registered bar ==");
    println!("   winner iff paraphrase net - B1 >= +6, paired bootstrap 95% CI low > 0 (240 paraphrases, 20,000");
    println!("   resamples, seed 1, splitmix64), as-taught right >= B1's, untaught wrong <= B1's + 2");
    let b1 = get(E::B1);
    let tb1 = tally(&blocks, &qs, b1, true, false);
    println!("   {:<17} {:>5} {:>4} {:>4} {:>4} {:>5} {:>6} {:>5} {:>6} {:>18} {:>7}  {}", "encoding", "q0 R", "pR", "pW", "pA", "pNet", "share", "uW",
        "Δnet", "95% CI", "P(<=0)", "verdict");
    for (e, res, _) in &results {
        let t = tally(&blocks, &qs, res, true, false);
        let d: Vec<i64> = qs.iter().enumerate().filter(|(_, q)| q.kind == 1)
            .map(|(i, q)| (score(&blocks, q, res.call(i, true), false) - score(&blocks, q, b1.call(i, true), false)) as i64).collect();
        let (s, lo, hi, p0) = boot_ci(&d);
        let expl = matches!(e, E::C2x | E::C2xL | E::X1L);
        let verdict = if matches!(e, E::B0 | E::B1) { "baseline".to_string() } else {
            let ok = [s >= 6, lo > 0, t[0][3].r >= tb1[0][3].r, t[2][3].w <= tb1[2][3].w + 2];
            let tag = if ok.iter().all(|&x| x) { if expl { "passes (EXPLORATORY, cannot win)" } else { "SCREEN WINNER" } } else { "not a winner" };
            format!("{tag} [{}]", ["Δ>=6", "CI>0", "q0", "untaught"].iter().zip(ok).map(|(n, o)| format!("{n}:{}", if o { "ok" } else { "no" })).collect::<Vec<_>>().join(" "))
        };
        println!("   {:<17} {:>5} {:>4} {:>4} {:>4} {:>+5} {:>5.1}% {:>5} {:>+6} {:>18} {:>7.3}  {}", ename(*e), t[0][3].r, t[1][3].r, t[1][3].w, t[1][3].a,
            t[1][3].net(), t[1][3].share(), t[2][3].w, s, format!("{lo:+} .. {hi:+}"), p0, verdict);
    }

    // ---------------------------------------------------------------- 4. secondary
    println!("\n== 4. SECONDARY (not part of the bar) ==");
    println!("   ungated (always name the top lesson), strict:");
    println!("   {:<17} {:>8} {:>16} {:>9} {:>12}", "encoding", "q0 R/W", "para R/W net", "unt W", "net p+u");
    for (e, res, _) in &results {
        let t = tally(&blocks, &qs, res, false, false);
        println!("   {:<17} {:>8} {:>16} {:>9} {:>+12}", ename(*e), format!("{}/{}", t[0][3].r, t[0][3].w),
            format!("{}/{} {:+}", t[1][3].r, t[1][3].w, t[1][3].net()), t[2][3].w, t[1][3].net() + t[2][3].net());
    }
    println!("   gated, harness-equivalent (same answer text counts right):");
    for (e, res, _) in &results {
        let t = tally(&blocks, &qs, res, true, true);
        println!("   {:<17} para {} share {:>5.1}%   net p+u {:+}", ename(*e), t[1][3].cell(), t[1][3].share(), t[1][3].net() + t[2][3].net());
    }

    // ---------------------------------------------------------------- 5. calibration
    println!("\n== 5. CALIBRATION (held-out P(right) vs observed, questions with a top lesson, pooled) ==");
    for (e, res, _) in &results {
        let mut bins = [(0usize, 0f64, 0usize); 10];
        for (i, q) in qs.iter().enumerate() {
            if res.top[i].is_none() { continue; }
            let b = ((res.p[i] * 10.0) as usize).min(9);
            bins[b].0 += 1; bins[b].1 += res.p[i];
            if res.top[i] == q.own { bins[b].2 += 1; }
        }
        let (mut ece, mut nall) = (0f64, 0usize);
        let cells: Vec<String> = bins.iter().map(|&(n, ps, r)| {
            if n == 0 { return "   -".into(); }
            ece += (ps - r as f64).abs(); nall += n;
            format!("{:.2}/{:.2}({n})", ps / n as f64, r as f64 / n as f64)
        }).collect();
        println!("   {:<17} ECE {:.3}  bins predicted/observed(n): {}", ename(*e), ece / nall.max(1) as f64, cells.join(" "));
    }

    // ---------------------------------------------------------------- 6. what the frames learned
    for (fr, sc, en, tag) in [(&frames, &simc, E::C2L, "C2 (pre-registered)"), (&xframes, &simx, E::C2xL, "C2x (EXPLORATORY)")] {
        let Some(f) = fr else { continue };
        if !results.iter().any(|r| r.0 == en) { continue; }
        let _ = sc;
        println!("\n== 6. WHAT THE FRAMES LEARNED: {tag} ==");
        // the bridges recall actually used: a question word absent from the top lesson, credited
        // through its best frame-similar lesson word; right = the top lesson was the own lesson
        let res = get(en);
        let mut br: HashMap<(String, String), (f32, i32, i32, f32)> = HashMap::new();
        for (i, q) in qs.iter().enumerate() {
            if q.kind == 0 { continue; }
            let Some(t) = res.top[i] else { continue };
            let b = &blocks[q.blk];
            let es = set_of(&b.lessons[t].q);
            let ok = q.own == Some(t);
            for w in set_of(&b.qs[q.qi].toks) {
                if es.binary_search(&w).is_ok() || es.iter().any(|v| lex.same(w, v)) { continue; }
                let best = es.iter().map(|v| (f.sim(w, v), *v)).fold((0f32, ""), |a, x| if x.0 > a.0 { x } else { a });
                if best.0 <= 0.0 { continue; }
                let cr = specs[q.blk].of(w) * best.0;
                let x = br.entry((w.to_string(), best.1.to_string())).or_insert((best.0, 0, 0, 0.0));
                if ok { x.1 += 1 } else { x.2 += 1 }
                x.3 += cr;
            }
        }
        let mut v: Vec<_> = br.into_iter().collect();
        v.sort_by(|a, b| b.1 .3.total_cmp(&a.1 .3));
        println!("   the 15 bridges carrying the most credit in {} (question word -> lesson word, sim, used in a right / wrong top lesson):", ename(en));
        for ((w, u), (s, r, x, c)) in v.iter().take(15) {
            println!("     {w:>14} -> {u:<14} sim {s:.3}  right {r}  wrong {x}  credit {c:.2}  {}", if r > x { "good" } else if x > r { "BAD" } else { "mixed" });
        }
        let (good, bad): (Vec<_>, Vec<_>) = v.iter().partition(|x| x.1 .1 >= x.1 .2);
        println!("   all bridges: {} pairs; used mostly in right top lessons {}, mostly in wrong ones {}", v.len(), good.len(), bad.len());
        let pairs = [("begin", "start"), ("big", "large"), ("big", "small"), ("designed", "created"), ("input", "output"),
            ("mutable", "changeable"), ("mutable", "immutable"), ("define", "declare"), ("coined", "introduced"), ("keyword", "word")];
        println!("   probe pairs: {}", pairs.iter().map(|(x, y)| format!("{x}~{y} {:.3}", f.sim(x, y))).collect::<Vec<_>>().join(", "));
        for w in ["mutable", "declares", "coined", "created", "default", "keyword", "output"] {
            let Some(&i) = f.ix.get(w) else { continue };
            let mut nb: Vec<(f32, &str)> = (0..f.words.len()).filter(|&j| j != i).map(|j| (f.sim_ix(i, j), f.words[j].as_str())).collect();
            nb.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut fr: Vec<(u64, f32)> = f.prof[i][0].iter().chain(f.prof[i][1].iter()).copied().collect();
            fr.sort_by(|a, b| b.1.total_cmp(&a.1));
            println!("   {w:<9} nearest: {}", nb.iter().take(6).map(|(s, x)| format!("{x} {s:.3}")).collect::<Vec<_>>().join(", "));
            println!("   {:<9} its top frames: {}", "", fr.iter().take(4).map(|(k, x)| format!("[{}] {x:.1}", f.show(*k))).collect::<Vec<_>>().join("  "));
        }
    }

    // ---------------------------------------------------------------- 7. examples where a candidate differs from B1
    for (ce, cname) in [(E::C2L, "C2+lemma"), (E::C1L, "C1+lemma")] {
        let Some((_, cr, _)) = results.iter().find(|r| r.0 == ce) else { continue };
        println!("\n== 7. QUESTIONS WHERE {cname} AND B1 DIFFER (gated, strict; paraphrase + untaught, all, in block order) ==");
        let (mut up, mut down) = (0, 0);
        for (i, q) in qs.iter().enumerate() {
            if q.kind == 0 { continue; }
            let (sb, sc) = (score(&blocks, q, b1.call(i, true), false), score(&blocks, q, cr.call(i, true), false));
            if sb == sc { continue; }
            if sc > sb { up += 1 } else { down += 1 }
            let b = &blocks[q.blk];
            let name = |c: Option<usize>| c.map_or("abstain".to_string(), |t| format!("L{} \"{}\"", t + 1, b.lessons[t].a_raw));
            println!("   {:<3} {:<10} {:+} -> {:+}  \"{}\"\n         B1: {}   {cname}: {}   (own {})", b.tag, KINDS[q.kind as usize], sb, sc,
                b.qs[q.qi].raw, name(b1.call(i, true)), name(cr.call(i, true)), q.own.map_or("none".into(), |o| format!("L{}", o + 1)));
        }
        println!("   {cname} better than B1 on {up} questions, worse on {down}");
    }

    // ---------------------------------------------------------------- 8. predicted real harness (added after the results)
    // in the real harness there is no P > 1/2 gate: the engram's activation a1 is its weight in her
    // vote (final = (1 - a1) cortex + a1 engram), so whether she SAYS the recalled answer depends on
    // a1's scale. learn P(she says the top lesson's answer | a1) from the saved v26 run (B0's a1),
    // check it by predicting the saved lemma run from B1's a1, then apply it to the candidates whose
    // score would be her activation (word-overlap scale; not the HDC rows, which have no such role).
    println!("\n== 8. PREDICTED REAL-HARNESS NET (EXPLORATORY: a speak model learned from the saved v26 runs) ==");
    if let Some((_, runs0)) = val_rows.iter().find(|r| r.0 == E::B0) {
        let b0 = get(E::B0);
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for (i, q) in qs.iter().enumerate() {
            let (Some(run), Some(t)) = (&runs0[q.blk], b0.top[i]) else { continue };
            x.push([1.0, b0.a1[i] as f64, 0.0]);
            y.push(if run[q.qi].reply == norm(&blocks[q.blk].lessons[t].a_raw) { 1.0 } else { 0.0 });
        }
        let beta = fit(&x, &y);
        println!("   P(says the top lesson's answer) = logistic({:.2} {:+.2} a1), fitted on {} v26 questions", beta[0], beta[1], x.len());
        let actual = |e: E| -> Option<i32> {
            let runs = &val_rows.iter().find(|r| r.0 == e)?.1;
            Some(qs.iter().filter(|q| q.kind != 0).map(|q| runs[q.blk].as_ref().map_or(0, |r| harness(&blocks, q, &r[q.qi].reply))).sum())
        };
        for (e, res, _) in &results {
            if matches!(e, E::C1 | E::C1o | E::C1L | E::C1oL) { continue; }
            let (mut pr, mut pw, mut pu) = (0f64, 0f64, 0f64);
            for (i, q) in qs.iter().enumerate() {
                if q.kind == 0 { continue; }
                let Some(_) = res.top[i] else { continue };
                let p = sigm(beta[0] + beta[1] * res.a1[i] as f64);
                match (q.kind, score(&blocks, q, res.top[i], true)) { (1, 1) => pr += p, (1, _) => pw += p, _ => pu += p }
            }
            println!("   {:<19} expected paraphrase right {:>5.1} wrong {:>5.1}, untaught wrong {:>5.1}  -> net {:>+6.1}{}", ename(*e), pr, pw, pu, pr - pw - pu,
                actual(*e).map_or(String::new(), |a| format!("   (real harness: {a:+})")));
        }
        // the hybrid a switch could be: CHOOSE the engram by the cosine (X1), but let it weigh in her
        // vote with its usual activation (B1's shared / all for that engram), so the scale her
        // speaking was measured on is kept
        if let Some((_, x1, _)) = results.iter().find(|r| r.0 == E::X1L) {
            let b1r = get(E::B1);
            let (mut pr, mut pw, mut pu) = (0f64, 0f64, 0f64);
            for (i, q) in qs.iter().enumerate() {
                if q.kind == 0 { continue; }
                let Some(t) = x1.top[i] else { continue };
                let p = sigm(beta[0] + beta[1] * b1r.all[i][t] as f64);
                match (q.kind, score(&blocks, q, Some(t), true)) { (1, 1) => pr += p, (1, _) => pw += p, _ => pu += p }
            }
            println!("   {:<19} expected paraphrase right {:>5.1} wrong {:>5.1}, untaught wrong {:>5.1}  -> net {:>+6.1}", "X1 rank, B1 weight",
                pr, pw, pu, pr - pw - pu);
        }
    }

    println!("\n== COST ==");
    println!("   data + spec + lexemes + frames loaded in {t_data:.2}s");
    for (e, _, s) in &results { println!("   {:<17} {:.2}s", ename(*e), s); }
    println!("   total {:.2}s, peak working set {:.2} GB", t0.elapsed().as_secs_f32(), peak_gb());
}

/// EXPLORATORY X1: cosine of the spec-weighted word sets (what bundling approximates):
/// sum over question words of s(w)^2 * hit(w, lesson) / sqrt(sum_q s^2 * sum_l s^2)
fn cosine(qs: &[&str], es: &[&str], spec: &Spec, near: &mut dyn FnMut(&str, &str) -> f32) -> f32 {
    let mut hit = |w: &str, ws: &[&str]| -> f32 {
        if ws.binary_search(&w).is_ok() { 1.0 } else { ws.iter().map(|v| near(w, v)).fold(0.0, f32::max) }
    };
    let (mut num, mut nq, mut ne) = (0f32, 0f32, 0f32);
    for w in qs { let x = spec.of(w); nq += x * x; num += x * x * hit(w, es); }
    for w in es { let x = spec.of(w); ne += x * x; }
    if nq > 0.0 && ne > 0.0 { num / (nq * ne).sqrt() } else { 0.0 }
}
