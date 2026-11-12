// TESTING / DESIGN ONLY — v51_answers: v51 Part D's SM (self-model) and Part B's G (the choosing part, basal ganglia),
// runs/v51/prereg_v51.md. CPU only.
//
// v51_answers collect --model FILE --questions TSV --out FILE [--library FILE] [--k 2] [--tokens 40] [--cands 3]
//                     [--no-int8] [--threads 8] [--limit N]
//     Each question asked fresh, exactly as answer_eval asks it ([EOS] + "Context: <her library's best --k passages>
//     \nQuestion: q\nAnswer:", int8 + the fused step, the chat app's way), read with Model::step_trace (the same
//     logits and state as `step`, bit for bit, plus her activity). Candidates: 0 = her greedy answer (answer_eval's
//     answer, the same rules: until EOS, a cut, or --tokens); 1.. = the next most likely first words (EOS left out),
//     each continued greedily from a copy of her state after the question. Per candidate one row: id, cat, cand,
//     grade (lookup::grade against the accepted answers), the state features below, the answer.
//     Features (what SM reads of her own state): lp_mean, lp_min, lp_first (log p of the words she wrote, her own
//     guesses' probabilities), ent_mean (entropy of her guesses), margin0 (log p top1 - top2 at the first word),
//     surp_prompt and surp_tail (her surprise at the prompt: mean -log p of its tokens, all and the last 24), spk_lo
//     and spk_hi (share of LIF units firing in the lower and upper half of her blocks), mem_read and mem_write (size
//     of her fast-memory reads and writes), eng_peak and eng_read (engram: sharpest attention, size of its read),
//     len (words written / 10), rank (the candidate's first-word rank), ctx (1 if her library cued passages); all
//     averaged over the steps that wrote the answer (the question's last token and each written word).
// v51_answers train   --selfq FILE --out FIT [--epochs 30] [--alpha 0.02] [--tau 0.3] [--lambda 0.01]
//     From the practice questions only. SM: logistic regression of right (1) vs wrong (0) on the standardised
//     features (abstains left out), every candidate a row, L2 lambda, full-batch gradient descent (2000 steps, rate
//     0.5, from 0). Cross-fitted: the questions in 2 folds (even, odd position); each fold's SM scores come from the SM
//     fitted on the other fold (out of fold). G: a value per option, V(c) = w . phi(c) for each candidate (phi: SM's
//     out-of-fold logit, lp_mean, lp_first, len, abstain wording, rank one-hot; the first four standardised) and
//     V(IDK) = w_idk; trained by reward (+1 right, -1 wrong, 0 abstain or IDK) as a bandit: per question it picks an
//     option from softmax(V / tau), gets that option's reward and moves w by alpha (r - V) phi (the reward prediction
//     error); --epochs passes, questions shuffled each pass (SplitMix64 seed 4848). Development numbers (practice
//     questions, out of fold): SM vs lp_mean AUROC; G trained on one fold, scored on the other, vs her greedy answers.
//     FIT holds SM fitted on every practice question and G trained on the out-of-fold SM logits.
// v51_answers score   --dev FILE --fit FIT [--items FILE] [--check FILE]
//     Once, on the dev collection: AUROC (her right vs wrong greedy answers) of SM vs her plain top-guess probability
//     (exp lp_mean), paired bootstrap of the difference over questions (the v49 bootstrap's generator, seed 4848, 10000
//     resamples, 95%); G's pick per question vs her greedy answer alone: right - wrong, precision, and the v49 bootstrap's
//     paired bootstrap of right - wrong (the same generator and quantiles). --items writes "id, G's grade, greedy
//     grade" (the v49 bootstrap's input); --check compares her greedy answers with an answer_eval items file.

use ava_data::bpe::{Bpe, EOS};
use ava_data::numbers::strip_marks;
use ava_data::rng::SplitMix64;
use ava_gpu::diary;
use ava_gpu::lookup::{grade, Grade, Library};
use ava_gpu::model::{Kind, Model, StepTrace};
use candle_core::Tensor;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

const FEATURES: [&str; 16] =
    ["lp_mean", "lp_min", "lp_first", "ent_mean", "margin0", "surp_prompt", "surp_tail", "spk_lo", "spk_hi", "mem_read", "mem_write", "eng_peak", "eng_read", "len", "rank", "ctx"];
const NF: usize = FEATURES.len();
const USAGE: &str = "usage: v51_answers collect|train|score [options] (see the top of src/bin/v51_answers.rs)";

fn main() {
    if let Err(e) = run() {
        eprintln!("v51_answers: {e}");
        std::process::exit(1);
    }
}

struct Args(Vec<String>);

impl Args {
    fn get(&self, k: &str) -> Option<String> {
        self.0.iter().position(|a| a == k).and_then(|i| self.0.get(i + 1)).cloned()
    }
    fn num(&self, k: &str, d: usize) -> Result<usize, String> {
        self.get(k).map(|v| v.parse().map_err(|_| format!("bad {k} {v:?}"))).transpose().map(|v| v.unwrap_or(d))
    }
    fn fl(&self, k: &str, d: f64) -> Result<f64, String> {
        self.get(k).map(|v| v.parse().map_err(|_| format!("bad {k} {v:?}"))).transpose().map(|v| v.unwrap_or(d))
    }
    fn path(&self, k: &str) -> Result<PathBuf, String> {
        self.get(k).map(PathBuf::from).ok_or_else(|| format!("{k} is required\n{USAGE}"))
    }
    fn has(&self, k: &str) -> bool {
        self.0.iter().any(|a| a == k)
    }
}

fn refuse_sealed(p: &std::path::Path) -> Result<(), String> {
    if p.to_string_lossy().to_lowercase().contains("sealed") {
        return Err(format!("{}: a sealed path is never read", p.display()));
    }
    Ok(())
}

fn run() -> Result<(), String> {
    let a = Args(std::env::args().skip(1).collect());
    let mode = a.0.first().cloned().ok_or(USAGE)?;
    diary::set_threads(a.num("--threads", 8)?.clamp(1, 8));
    match mode.as_str() {
        "collect" => collect(&a),
        "train" => train(&a),
        "score" => score(&a),
        "check" => check(&a),
        _ => Err(USAGE.into()),
    }
}

// ------------------------------------------------------------------------------------------------ collect

/// log-softmax summary of one logits row: (log p of every token as needed, entropy, top two).
struct Dist {
    row: Vec<f32>,
    lse: f64,
}

impl Dist {
    fn new(l: &Tensor) -> Result<Dist, String> {
        let row: Vec<f32> = l.to_dtype(candle_core::DType::F32).and_then(|x| x.flatten_all()).and_then(|x| x.to_vec1()).map_err(|e| e.to_string())?;
        let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let lse = mx + row.iter().map(|&x| (x as f64 - mx).exp()).sum::<f64>().ln();
        Ok(Dist { row, lse })
    }
    fn logp(&self, t: u32) -> f64 {
        self.row[t as usize] as f64 - self.lse
    }
    fn entropy(&self) -> f64 {
        self.row.iter().map(|&x| {
            let lp = x as f64 - self.lse;
            -lp.exp() * lp
        }).sum()
    }
    /// token ids by logit, highest first
    fn order(&self) -> Vec<u32> {
        let mut o: Vec<u32> = (0..self.row.len() as u32).collect();
        o.sort_by(|&a, &b| self.row[b as usize].total_cmp(&self.row[a as usize]));
        o
    }
}

/// One step's activity: per-layer firing share, and the memory's and engram's sizes (None where absent).
#[derive(Default, Clone)]
struct Act {
    spk: Vec<f64>,
    mem_read: Vec<f64>,
    mem_write: Vec<f64>,
    eng_peak: Vec<f64>,
    eng_read: Vec<f64>,
}

fn v1(t: &Tensor) -> Result<Vec<f32>, String> {
    t.flatten_all().and_then(|x| x.to_vec1::<f32>()).map_err(|e| e.to_string())
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn act(tr: &StepTrace) -> Result<Act, String> {
    let mut a = Act::default();
    for ly in &tr.layers {
        let s = v1(&ly.spikes)?;
        a.spk.push(s.iter().filter(|&&x| x != 0.0).count() as f64 / s.len().max(1) as f64);
        if let Some(t) = &ly.mem_read {
            a.mem_read.push(mean(&v1(t)?.iter().map(|&x| x as f64).collect::<Vec<_>>()));
        }
        if let Some(t) = &ly.mem_write {
            a.mem_write.push(mean(&v1(t)?.iter().map(|&x| x as f64).collect::<Vec<_>>()));
        }
        if let Some(t) = &ly.engram_attn {
            let (h, n) = t.dims2().map_err(|e| e.to_string())?;
            let w = v1(t)?;
            a.eng_peak.push(mean(&(0..h).map(|i| w[i * n..(i + 1) * n].iter().copied().fold(0f32, f32::max) as f64).collect::<Vec<_>>()));
        }
        if let Some(t) = &ly.engram_read {
            let w = v1(t)?;
            a.eng_read.push((w.iter().map(|&x| (x * x) as f64).sum::<f64>() / w.len().max(1) as f64).sqrt());
        }
    }
    Ok(a)
}

struct Cand {
    cand: usize,
    grade: Grade,
    f: [f64; NF],
    answer: String,
}

#[allow(clippy::too_many_arguments)]
fn ask(m: &Model, bpe: &Bpe, glance: bool, ctx: &str, q: &str, accept: &str, cands: usize, max_tokens: usize) -> Result<(Vec<Cand>, usize), String> {
    let e = |e: candle_core::Error| e.to_string();
    let mut ids = vec![EOS];
    ids.extend(diary::encode(bpe, glance, &ava_gpu::qa::prompt_text(ctx, &diary::one_line(q))));
    let mut st = m.stream().map_err(e)?;
    let mut nll = Vec::with_capacity(ids.len());
    let mut last: Option<(Tensor, StepTrace)> = None;
    for &t in &ids {
        if let Some((l, _)) = &last {
            nll.push(-Dist::new(l)?.logp(t));
        }
        last = Some(m.step_trace(&mut st, t).map_err(e)?);
    }
    let (l0, tr0) = last.ok_or("empty prompt")?;
    let d0 = Dist::new(&l0)?;
    let order = d0.order();
    let greedy = l0.argmax(0).and_then(|x| x.to_scalar::<u32>()).map_err(e)?; // diary::pick at temperature 0
    let mut firsts = vec![greedy];
    firsts.extend(order.iter().copied().filter(|&t| t != greedy && t != EOS).take(cands.saturating_sub(1)));
    let margin0 = d0.logp(order[0]) - d0.logp(order[1]);
    let surp_prompt = mean(&nll);
    let surp_tail = mean(&nll[nll.len().saturating_sub(24)..]);
    let a0 = act(&tr0)?;
    let base = Model::stream_state(&st).cloned().ok_or("a State-backed stream is needed (a model with block options)")?;
    let mut out = Vec::with_capacity(firsts.len());
    let mut steps = ids.len();
    let mut st0 = Some(st);
    for (c, &first) in firsts.iter().enumerate() {
        let mut s = match st0.take() {
            Some(s) if c == 0 => s,
            _ => m.stream_from(base.clone()).map_err(e)?,
        };
        let mut next = Dist::new(&l0)?;
        let mut acts = vec![a0.clone()];
        let (mut toks, mut lps, mut ents) = (Vec::new(), Vec::new(), Vec::new());
        let lp_first = next.logp(first);
        for n in 0..max_tokens {
            let tok = if n == 0 { first } else { next.order_first() };
            if tok == EOS {
                break;
            }
            lps.push(next.logp(tok));
            ents.push(next.entropy());
            toks.push(tok);
            if diary::cut_reply(&bpe.decode(&strip_marks(&toks))).1 {
                break;
            }
            if n + 1 < max_tokens {
                let (l, tr) = m.step_trace(&mut s, tok).map_err(e)?;
                steps += 1;
                next = Dist::new(&l)?;
                acts.push(act(&tr)?);
            }
        }
        let answer = diary::cut_reply(&bpe.decode(&strip_marks(&toks))).0;
        let half = acts[0].spk.len() / 2;
        let f = [
            if lps.is_empty() { lp_first } else { mean(&lps) },
            if lps.is_empty() { lp_first } else { lps.iter().copied().fold(f64::INFINITY, f64::min) },
            lp_first,
            if ents.is_empty() { d0.entropy() } else { mean(&ents) },
            margin0,
            surp_prompt,
            surp_tail,
            mean(&acts.iter().map(|a| mean(&a.spk[..half])).collect::<Vec<_>>()),
            mean(&acts.iter().map(|a| mean(&a.spk[half..])).collect::<Vec<_>>()),
            mean(&acts.iter().map(|a| mean(&a.mem_read)).collect::<Vec<_>>()),
            mean(&acts.iter().map(|a| mean(&a.mem_write)).collect::<Vec<_>>()),
            mean(&acts.iter().map(|a| mean(&a.eng_peak)).collect::<Vec<_>>()),
            mean(&acts.iter().map(|a| mean(&a.eng_read)).collect::<Vec<_>>()),
            toks.len() as f64 / 10.0,
            c as f64,
            if ctx.trim().is_empty() { 0.0 } else { 1.0 },
        ];
        out.push(Cand { cand: c, grade: grade(&answer, accept), f, answer });
    }
    Ok((out, steps))
}

impl Dist {
    /// her greedy pick from this row (the first highest logit, as argmax)
    fn order_first(&self) -> u32 {
        let mut best = 0usize;
        for (j, &x) in self.row.iter().enumerate() {
            if x > self.row[best] {
                best = j;
            }
        }
        best as u32
    }
}

fn collect(a: &Args) -> Result<(), String> {
    let model_path = a.path("--model")?;
    let qpath = a.path("--questions")?;
    refuse_sealed(&qpath)?;
    let out = a.path("--out")?;
    let (k, tokens, cands, limit) = (a.num("--k", 2)?, a.num("--tokens", 40)?, a.num("--cands", 3)?.max(1), a.num("--limit", usize::MAX)?);
    let dev = ava_gpu::device(true).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&PathBuf::from(a.get("--bpe").unwrap_or_else(|| ava_gpu::DEFAULT_BPE.into()))).map_err(|e| e.to_string())?;
    let (cfg, _vm, model) = ava_gpu::ckpt::load_auto(&model_path, Kind::D, 4096, &dev)?;
    model.set_track_activity(false);
    model.set_inference(true);
    let int8 = !a.has("--no-int8");
    if int8 {
        model.set_int8(true).map_err(|e| format!("int8: {e}"))?;
        model.set_fused(true);
    }
    let lib = match a.get("--library") {
        Some(p) => Some(Library::load(&PathBuf::from(&p))?),
        None => None,
    };
    let text = std::fs::read_to_string(&qpath).map_err(|e| format!("{}: {e}", qpath.display()))?;
    let rows: Vec<Vec<&str>> = text.lines().skip(1).filter(|l| !l.trim().is_empty()).map(|l| l.split('\t').collect()).take(limit).collect();
    let mut w = std::io::BufWriter::new(std::fs::File::create(&out).map_err(|e| format!("{}: {e}", out.display()))?);
    writeln!(w, "id\tcat\tcand\tgrade\t{}\tanswer", FEATURES.join("\t")).map_err(|e| e.to_string())?;
    let t0 = Instant::now();
    let (mut steps, mut g0) = (0usize, [0usize; 3]);
    for r in &rows {
        if r.len() < 4 {
            return Err(format!("{}: a row without 4 columns: {r:?}", qpath.display()));
        }
        let ctx = lib.as_ref().map(|l| l.context(r[2], k)).unwrap_or_default();
        let (cs, n) = ask(&model, &bpe, cfg.glance, &ctx, r[2], r[3], cands, tokens)?;
        steps += n;
        for c in &cs {
            if c.cand == 0 {
                g0[match c.grade {
                    Grade::Right => 0,
                    Grade::Abstain => 1,
                    Grade::Wrong => 2,
                }] += 1;
            }
            let f: Vec<String> = c.f.iter().map(|x| format!("{x:.6}")).collect();
            writeln!(w, "{}\t{}\t{}\t{:?}\t{}\t{}", r[0], r[1], c.cand, c.grade, f.join("\t"), c.answer.replace(['\t', '\n'], " ")).map_err(|e| e.to_string())?;
        }
    }
    w.flush().map_err(|e| e.to_string())?;
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "collect: {} questions x {cands} candidates in {secs:.1} s ({:.0} steps/s, {steps} steps; {}); greedy answers: right {} abstain {} wrong {} (right - wrong {:+})",
        rows.len(),
        steps as f64 / secs.max(1e-9),
        if int8 { "int8 + fused step" } else { "f32" },
        g0[0],
        g0[1],
        g0[2],
        g0[0] as i64 - g0[2] as i64
    );
    Ok(())
}

// ------------------------------------------------------------------------------------------------ reading a collection

struct Row {
    id: String,
    cand: usize,
    grade: Grade,
    f: [f64; NF],
    abst: bool,
    answer: String,
}

fn read_rows(p: &PathBuf) -> Result<Vec<Row>, String> {
    refuse_sealed(p)?;
    let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut out = Vec::new();
    for line in text.lines().skip(1).filter(|l| !l.is_empty()) {
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() < 4 + NF {
            return Err(format!("{}: short row {line:?}", p.display()));
        }
        let gd = match c[3] {
            "Right" => Grade::Right,
            "Abstain" => Grade::Abstain,
            "Wrong" => Grade::Wrong,
            g => return Err(format!("bad grade {g:?}")),
        };
        let mut f = [0f64; NF];
        for (j, x) in f.iter_mut().enumerate() {
            *x = c[4 + j].parse().map_err(|_| format!("bad feature {:?}", c[4 + j]))?;
        }
        let answer = c.get(4 + NF).copied().unwrap_or("");
        // her own wording says she does not know (the grader's abstain rule with no accepted answer: no key needed)
        let abst = grade(answer, "") == Grade::Abstain;
        out.push(Row { id: c[0].to_string(), cand: c[2].parse().map_err(|_| "bad cand".to_string())?, grade: gd, f, abst, answer: answer.to_string() });
    }
    Ok(out)
}

/// questions in order of appearance: the row indices of each
fn questions(rows: &[Row]) -> Vec<Vec<usize>> {
    let mut order: Vec<String> = Vec::new();
    let mut by: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, r) in rows.iter().enumerate() {
        if !by.contains_key(&r.id) {
            order.push(r.id.clone());
        }
        by.entry(r.id.clone()).or_default().push(i);
    }
    order.into_iter().map(|id| by.remove(&id).unwrap()).collect()
}

fn reward(g: Grade) -> f64 {
    match g {
        Grade::Right => 1.0,
        Grade::Wrong => -1.0,
        Grade::Abstain => 0.0,
    }
}

// ------------------------------------------------------------------------------------------------ SM

#[derive(Clone)]
struct Sm {
    mu: [f64; NF],
    sd: [f64; NF],
    w: [f64; NF],
    b: f64,
}

fn standardise(rows: &[&Row]) -> ([f64; NF], [f64; NF]) {
    let n = rows.len().max(1) as f64;
    let (mut mu, mut sd) = ([0f64; NF], [1f64; NF]);
    for j in 0..NF {
        mu[j] = rows.iter().map(|r| r.f[j]).sum::<f64>() / n;
        let v = rows.iter().map(|r| (r.f[j] - mu[j]).powi(2)).sum::<f64>() / n;
        sd[j] = if v > 1e-12 { v.sqrt() } else { 1.0 };
    }
    (mu, sd)
}

impl Sm {
    fn fit(rows: &[&Row], lambda: f64) -> Sm {
        let used: Vec<&Row> = rows.iter().copied().filter(|r| r.grade != Grade::Abstain).collect();
        let (mu, sd) = standardise(&used);
        let x: Vec<[f64; NF]> = used.iter().map(|r| std::array::from_fn(|j| (r.f[j] - mu[j]) / sd[j])).collect();
        let y: Vec<f64> = used.iter().map(|r| if r.grade == Grade::Right { 1.0 } else { 0.0 }).collect();
        let (mut w, mut b) = ([0f64; NF], 0f64);
        let n = x.len().max(1) as f64;
        for _ in 0..2000 {
            let (mut gw, mut gb) = ([0f64; NF], 0f64);
            for (xi, &yi) in x.iter().zip(&y) {
                let z = b + (0..NF).map(|j| w[j] * xi[j]).sum::<f64>();
                let p = 1.0 / (1.0 + (-z).exp());
                let d = p - yi;
                gb += d;
                for j in 0..NF {
                    gw[j] += d * xi[j];
                }
            }
            b -= 0.5 * gb / n;
            for j in 0..NF {
                w[j] -= 0.5 * (gw[j] / n + lambda * w[j]);
            }
        }
        Sm { mu, sd, w, b }
    }
    fn logit(&self, r: &Row) -> f64 {
        self.b + (0..NF).map(|j| self.w[j] * (r.f[j] - self.mu[j]) / self.sd[j]).sum::<f64>()
    }
}

/// AUROC of `s` for telling 1 (right) from 0 (wrong): P(s_right > s_wrong) + P(tie) / 2.
fn auroc(s: &[f64], y: &[bool]) -> f64 {
    let (mut num, mut pairs) = (0f64, 0f64);
    for (i, &a) in s.iter().enumerate() {
        if !y[i] {
            continue;
        }
        for (j, &b) in s.iter().enumerate() {
            if y[j] {
                continue;
            }
            pairs += 1.0;
            num += if a > b { 1.0 } else if a == b { 0.5 } else { 0.0 };
        }
    }
    if pairs == 0.0 {
        f64::NAN
    } else {
        num / pairs
    }
}

/// the v49 bootstrap's generator (Park-Miller, seed 4848): the resampled index stream.
struct Lcg(f64);

impl Lcg {
    fn new() -> Lcg {
        Lcg(4848.0)
    }
    fn index(&mut self, n: usize) -> usize {
        self.0 = (self.0 * 16807.0) % 2147483647.0;
        (self.0 / 2147483647.0 * n as f64) as usize
    }
}

/// the v49 bootstrap exactly: 10000 resamples of sum(v), the 250th and 9751st of them sorted.
fn boot_v49(v: &[i64]) -> (i64, i64) {
    let mut g = Lcg::new();
    let mut b: Vec<i64> = (0..10000).map(|_| (0..v.len()).map(|_| v[g.index(v.len())]).sum()).collect();
    b.sort();
    (b[249], b[9750])
}

/// Paired bootstrap of AUROC(a) - AUROC(b) over items (the v49 bootstrap's generator and quantiles); resamples with one class
/// missing are skipped (counted).
fn boot_auroc(a: &[f64], b: &[f64], y: &[bool]) -> (f64, f64, usize) {
    let mut g = Lcg::new();
    let n = y.len();
    let mut d = Vec::with_capacity(10000);
    let mut skipped = 0;
    for _ in 0..10000 {
        let idx: Vec<usize> = (0..n).map(|_| g.index(n)).collect();
        let yy: Vec<bool> = idx.iter().map(|&i| y[i]).collect();
        if yy.iter().all(|&x| x) || yy.iter().all(|&x| !x) {
            skipped += 1;
            continue;
        }
        let aa: Vec<f64> = idx.iter().map(|&i| a[i]).collect();
        let bb: Vec<f64> = idx.iter().map(|&i| b[i]).collect();
        d.push(auroc(&aa, &yy) - auroc(&bb, &yy));
    }
    d.sort_by(|x, y| x.partial_cmp(y).expect("finite"));
    let q = |p: f64| d[((p * d.len() as f64) as usize).min(d.len() - 1)];
    (q(0.025), q(0.975), skipped)
}

// ------------------------------------------------------------------------------------------------ G

#[derive(Clone)]
struct G {
    mu: [f64; 4],
    sd: [f64; 4],
    w: [f64; 8],
    w_idk: f64,
}

/// phi(c): SM's logit, lp_mean, lp_first, len (standardised), her abstain wording, rank one-hot (0, 1, 2+).
fn phi(z: f64, r: &Row, mu: &[f64; 4], sd: &[f64; 4]) -> [f64; 8] {
    let raw = [z, r.f[0], r.f[2], r.f[13]];
    let s: [f64; 4] = std::array::from_fn(|j| (raw[j] - mu[j]) / sd[j]);
    [s[0], s[1], s[2], s[3], if r.abst { 1.0 } else { 0.0 }, (r.cand == 0) as u8 as f64, (r.cand == 1) as u8 as f64, (r.cand >= 2) as u8 as f64]
}

impl G {
    fn value(&self, p: &[f64; 8]) -> f64 {
        (0..8).map(|j| self.w[j] * p[j]).sum()
    }

    /// option per question: Some(row index) for a candidate, None for IDK (argmax of V; ties to the earlier option)
    fn choose(&self, qs: &[Vec<usize>], rows: &[Row], z: &[f64]) -> Vec<Option<usize>> {
        qs.iter()
            .map(|q| {
                let mut best: (Option<usize>, f64) = (None, f64::NEG_INFINITY);
                for &i in q {
                    let v = self.value(&phi(z[i], &rows[i], &self.mu, &self.sd));
                    if v > best.1 {
                        best = (Some(i), v);
                    }
                }
                if self.w_idk > best.1 {
                    (None, self.w_idk)
                } else {
                    best
                }
                .0
            })
            .collect()
    }

    fn train(qs: &[Vec<usize>], rows: &[Row], z: &[f64], epochs: usize, alpha: f64, tau: f64) -> G {
        let all: Vec<usize> = qs.iter().flatten().copied().collect();
        let raw: Vec<[f64; 4]> = all.iter().map(|&i| [z[i], rows[i].f[0], rows[i].f[2], rows[i].f[13]]).collect();
        let n = raw.len().max(1) as f64;
        let mut mu = [0f64; 4];
        let mut sd = [1f64; 4];
        for j in 0..4 {
            mu[j] = raw.iter().map(|r| r[j]).sum::<f64>() / n;
            let v = raw.iter().map(|r| (r[j] - mu[j]).powi(2)).sum::<f64>() / n;
            sd[j] = if v > 1e-12 { v.sqrt() } else { 1.0 };
        }
        let mut g = G { mu, sd, w: [0.0; 8], w_idk: 0.0 };
        let mut rng = SplitMix64::new(4848);
        let mut order: Vec<usize> = (0..qs.len()).collect();
        for _ in 0..epochs {
            rng.shuffle(&mut order);
            for &qi in &order {
                let q = &qs[qi];
                let ph: Vec<[f64; 8]> = q.iter().map(|&i| phi(z[i], &rows[i], &g.mu, &g.sd)).collect();
                let mut v: Vec<f64> = ph.iter().map(|p| g.value(p)).collect();
                v.push(g.w_idk);
                let mx = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let ex: Vec<f64> = v.iter().map(|x| ((x - mx) / tau).exp()).collect();
                let tot: f64 = ex.iter().sum();
                let mut u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * tot;
                let mut pick = ex.len() - 1;
                for (j, x) in ex.iter().enumerate() {
                    u -= x;
                    if u <= 0.0 {
                        pick = j;
                        break;
                    }
                }
                if pick == q.len() {
                    g.w_idk += alpha * (0.0 - g.w_idk);
                } else {
                    let delta = reward(rows[q[pick]].grade) - v[pick];
                    for j in 0..8 {
                        g.w[j] += alpha * delta * ph[pick][j];
                    }
                }
            }
        }
        g
    }
}

/// right, abstain, wrong of the chosen options
fn tally(choice: &[Option<usize>], rows: &[Row]) -> (i64, i64, i64) {
    let mut t = (0, 0, 0);
    for c in choice {
        match c.map(|i| rows[i].grade).unwrap_or(Grade::Abstain) {
            Grade::Right => t.0 += 1,
            Grade::Abstain => t.1 += 1,
            Grade::Wrong => t.2 += 1,
        }
    }
    t
}

fn greedy(qs: &[Vec<usize>], rows: &[Row]) -> Vec<Option<usize>> {
    qs.iter().map(|q| q.iter().copied().find(|&i| rows[i].cand == 0)).collect()
}

fn fmt_t(t: (i64, i64, i64)) -> String {
    let prec = if t.0 + t.2 > 0 { t.0 as f64 / (t.0 + t.2) as f64 } else { f64::NAN };
    format!("right {} abstain {} wrong {}: right - wrong {:+}, precision {prec:.3}", t.0, t.1, t.2, t.0 - t.2)
}

// ------------------------------------------------------------------------------------------------ train and score

fn train(a: &Args) -> Result<(), String> {
    let rows = read_rows(&a.path("--selfq")?)?;
    let out = a.path("--out")?;
    let (epochs, alpha, tau, lambda) = (a.num("--epochs", 30)?, a.fl("--alpha", 0.02)?, a.fl("--tau", 0.3)?, a.fl("--lambda", 0.01)?);
    let qs = questions(&rows);
    let fold_of: HashMap<&str, usize> = qs.iter().enumerate().map(|(k, q)| (rows[q[0]].id.as_str(), k % 2)).collect();
    // SM per fold, out-of-fold logits
    let mut z = vec![0f64; rows.len()];
    let mut sms = Vec::new();
    for f in 0..2 {
        let train: Vec<&Row> = rows.iter().filter(|r| fold_of[r.id.as_str()] != f).collect();
        let sm = Sm::fit(&train, lambda);
        for (i, r) in rows.iter().enumerate() {
            if fold_of[r.id.as_str()] == f {
                z[i] = sm.logit(r);
            }
        }
        sms.push(sm);
    }
    // development numbers (practice questions only, out of fold)
    let g0: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].cand == 0 && rows[i].grade != Grade::Abstain).collect();
    let y: Vec<bool> = g0.iter().map(|&i| rows[i].grade == Grade::Right).collect();
    let sz: Vec<f64> = g0.iter().map(|&i| z[i]).collect();
    let base: Vec<f64> = g0.iter().map(|&i| rows[i].f[0]).collect();
    println!(
        "practice questions: {} ({} rows); greedy answers right {} wrong {}; out-of-fold AUROC: SM {:.4}, plain top-guess probability {:.4}",
        qs.len(),
        rows.len(),
        y.iter().filter(|&&x| x).count(),
        y.iter().filter(|&&x| !x).count(),
        auroc(&sz, &y),
        auroc(&base, &y)
    );
    let mut dev_choice = vec![None; qs.len()];
    for f in 0..2 {
        let tr: Vec<Vec<usize>> = qs.iter().enumerate().filter(|(k, _)| k % 2 != f).map(|(_, q)| q.clone()).collect();
        let g = G::train(&tr, &rows, &z, epochs, alpha, tau);
        let te_idx: Vec<usize> = (0..qs.len()).filter(|k| k % 2 == f).collect();
        let te: Vec<Vec<usize>> = te_idx.iter().map(|&k| qs[k].clone()).collect();
        for (k, c) in te_idx.iter().zip(g.choose(&te, &rows, &z)) {
            dev_choice[*k] = c;
        }
    }
    println!("practice, out of fold: G {}", fmt_t(tally(&dev_choice, &rows)));
    println!("practice, out of fold: greedy {}", fmt_t(tally(&greedy(&qs, &rows), &rows)));
    // the fit: SM on every practice question; G on the out-of-fold logits
    let all: Vec<&Row> = rows.iter().collect();
    let sm = Sm::fit(&all, lambda);
    let g = G::train(&qs, &rows, &z, epochs, alpha, tau);
    let j = |v: &[f64]| v.iter().map(|x| format!("{x:.9e}")).collect::<Vec<_>>().join("\t");
    let txt = format!(
        "# v51 SM + G fit from {} practice questions; epochs {epochs} alpha {alpha} tau {tau} lambda {lambda}\nsm_mu\t{}\nsm_sd\t{}\nsm_w\t{}\nsm_b\t{:.9e}\ng_mu\t{}\ng_sd\t{}\ng_w\t{}\ng_idk\t{:.9e}\n",
        qs.len(),
        j(&sm.mu),
        j(&sm.sd),
        j(&sm.w),
        sm.b,
        j(&g.mu),
        j(&g.sd),
        j(&g.w),
        g.w_idk
    );
    std::fs::write(&out, &txt).map_err(|e| format!("{}: {e}", out.display()))?;
    println!("SM weights (standardised features): {}", FEATURES.iter().zip(sm.w.iter()).map(|(n, w)| format!("{n} {w:+.3}")).collect::<Vec<_>>().join(", "));
    println!("G weights (sm, lp_mean, lp_first, len, abstain, rank0, rank1, rank2+): {:?}; IDK value {:.4}", g.w.map(|x| (x * 1e4).round() / 1e4), g.w_idk);
    Ok(())
}

fn read_fit(p: &PathBuf) -> Result<(Sm, G), String> {
    let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut kv: HashMap<&str, Vec<f64>> = HashMap::new();
    for l in text.lines().filter(|l| !l.starts_with('#') && !l.is_empty()) {
        let mut it = l.split('\t');
        let k = it.next().unwrap_or("");
        kv.insert(k, it.map(|x| x.parse::<f64>().map_err(|_| format!("bad number {x:?}"))).collect::<Result<_, _>>()?);
    }
    let arr = |k: &str| -> Result<Vec<f64>, String> { kv.get(k).cloned().ok_or_else(|| format!("{}: no {k}", p.display())) };
    let fix = |v: Vec<f64>, n: usize| -> Result<Vec<f64>, String> { if v.len() == n { Ok(v) } else { Err(format!("{} values, {n} expected", v.len())) } };
    let to = |v: Vec<f64>| -> [f64; NF] { std::array::from_fn(|j| v[j]) };
    let sm = Sm { mu: to(fix(arr("sm_mu")?, NF)?), sd: to(fix(arr("sm_sd")?, NF)?), w: to(fix(arr("sm_w")?, NF)?), b: arr("sm_b")?[0] };
    let gm = fix(arr("g_mu")?, 4)?;
    let gs = fix(arr("g_sd")?, 4)?;
    let gw = fix(arr("g_w")?, 8)?;
    let g = G { mu: std::array::from_fn(|j| gm[j]), sd: std::array::from_fn(|j| gs[j]), w: std::array::from_fn(|j| gw[j]), w_idk: arr("g_idk")?[0] };
    Ok((sm, g))
}

fn score(a: &Args) -> Result<(), String> {
    let rows = read_rows(&a.path("--dev")?)?;
    let (sm, g) = read_fit(&a.path("--fit")?)?;
    let qs = questions(&rows);
    let z: Vec<f64> = rows.iter().map(|r| sm.logit(r)).collect();
    // SM: her greedy answers that were right or wrong
    let g0: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].cand == 0 && rows[i].grade != Grade::Abstain).collect();
    let y: Vec<bool> = g0.iter().map(|&i| rows[i].grade == Grade::Right).collect();
    let sz: Vec<f64> = g0.iter().map(|&i| z[i]).collect();
    let base: Vec<f64> = g0.iter().map(|&i| rows[i].f[0]).collect();
    let (lo, hi, skipped) = boot_auroc(&sz, &base, &y);
    let (au_s, au_b) = (auroc(&sz, &y), auroc(&base, &y));
    println!(
        "SM: AUROC {au_s:.4} vs plain top-guess probability {au_b:.4} on {} answered questions ({} right, {} wrong); difference {:+.4} [95% {lo:+.4}, {hi:+.4}] (the v49 bootstrap's generator, seed 4848, 10000 resamples, {skipped} skipped); bar (lo > 0): {}",
        y.len(),
        y.iter().filter(|&&x| x).count(),
        y.iter().filter(|&&x| !x).count(),
        au_s - au_b,
        if lo > 0.0 { "PASS" } else { "FAIL" }
    );
    // G vs her greedy answers
    let gc = g.choose(&qs, &rows, &z);
    let base_c = greedy(&qs, &rows);
    let (tg, tb) = (tally(&gc, &rows), tally(&base_c, &rows));
    let gr = |c: &Option<usize>| c.map(|i| rows[i].grade).unwrap_or(Grade::Abstain);
    let v: Vec<i64> = gc.iter().zip(&base_c).map(|(x, b)| reward(gr(x)) as i64 - reward(gr(b)) as i64).collect();
    let (blo, bhi) = boot_v49(&v);
    let picks = [gc.iter().filter(|c| c.is_none()).count(), gc.iter().filter(|c| c.is_some_and(|i| rows[i].cand == 0)).count(), gc.iter().filter(|c| c.is_some_and(|i| rows[i].cand > 0)).count()];
    println!("G: {}; picks: IDK {}, her greedy answer {}, another candidate {}", fmt_t(tg), picks[0], picks[1], picks[2]);
    println!("Q4b alone: {}", fmt_t(tb));
    println!(
        "G - Q4b alone: right - wrong {:+}, paired bootstrap (the v49 bootstrap, seed 4848) {blo} to {bhi}; bar (lo > 0): {}",
        (tg.0 - tg.2) - (tb.0 - tb.2),
        if blo > 0 { "PASS" } else { "FAIL" }
    );
    if let Some(p) = a.get("--items") {
        let mut s = String::from("id\tgrade_G\tgrade_Q4b\n");
        for ((q, x), b) in qs.iter().zip(&gc).zip(&base_c) {
            s += &format!("{}\t{:?}\t{:?}\n", rows[q[0]].id, gr(x), gr(b));
        }
        std::fs::write(&p, s).map_err(|e| format!("{p}: {e}"))?;
    }
    if let Some(p) = a.get("--check") {
        check_items(&rows, &p)?;
    }
    Ok(())
}

/// Her greedy answers (candidate 0) vs an answer_eval items file of the same model and questions (id, cat, grade,
/// answer, context): how many answers and grades are equal.
fn check_items(rows: &[Row], p: &str) -> Result<(), String> {
    let text = std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?;
    let theirs: HashMap<String, (String, String)> = text
        .lines()
        .skip(1)
        .filter_map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            (c.len() >= 4).then(|| (c[0].to_string(), (c[2].to_string(), c[3].to_string())))
        })
        .collect();
    let (mut same, mut grade_only, mut diff, mut missing) = (0, 0, 0, 0);
    for r in rows.iter().filter(|r| r.cand == 0) {
        match theirs.get(&r.id) {
            Some((g, ans)) if *g == format!("{:?}", r.grade) && *ans == r.answer => same += 1,
            Some((g, _)) if *g == format!("{:?}", r.grade) => grade_only += 1,
            Some(_) => diff += 1,
            None => missing += 1,
        }
    }
    println!("check vs {p}: answer and grade equal on {same}, grade only on {grade_only}, different on {diff}; {missing} not in the file");
    Ok(())
}

fn check(a: &Args) -> Result<(), String> {
    let rows = read_rows(&a.path("--collect")?)?;
    check_items(&rows, &a.get("--items").ok_or("--items FILE is required")?)
}
