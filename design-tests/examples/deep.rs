// TESTING / DESIGN ONLY.
//
// DEPTH. The one gap [[v10]] named that is not decoration.
//
//   cargo run --release --example deep -- <triples> <stage2 0|1>
//
// Everything Ava does is one stage: features -> units -> one weighted vote. There is
// no layer that reads the OUTPUT of another layer. Dendritic depth turned out to be
// redundant with the specificity term (v10), but that was depth INSIDE a unit. This
// is depth between stages, which she genuinely does not have.
//
//   STAGE 1  the current model. features -> a guess.
//   STAGE 2  the same machinery, plus stage 1's GUESS as an input feature, trained on
//            the true answer. it can only learn one thing: when stage 1 is wrong in a
//            systematic way, and what to say instead.
//
// Stage 1's guess enters as an IDENTITY key, because identity keys are free — the
// competition returns a fixed number of winners however many keys you supply (v9 §3e).
// A general feature would tax every position to pay for corrections at a few.
use design_tests::reply;
use design_tests::seq::{self, BOS, EOS};
use design_tests::v4::{self, Params};
use design_tests::v6::*;
use std::env;
use std::fs;
use std::time::Instant;

const CORPUS: &str = "../data/triples.tsv";
const MIN_FREE_GB: f64 = 3.0;
const ORDER: usize = 3;
/// A FIXED held-out set, taken from the FRONT of the corpus and never trained on.
///
/// It used to be the last 2% of whatever was loaded, which silently made every
/// corpus size a different exam. Measured: 60k exchanges scored 23.32% against 30k's
/// 36.47%, and doubling the neuron pool moved it by 0.09 — because the drop was never
/// about the model, it was two different test sets. triples.tsv is ordered by film, so
/// the tail of a 60k window is a different genre from the tail of a 30k one.
///
/// Comparisons across training size are meaningless unless the exam is held constant.
const TEST: usize = 1200;
const K_ID: usize = 6;

fn mem_gb() -> (f64, f64) {
    use std::process::Command;
    let pid = std::process::id();
    let script = format!(
        "$p=(Get-Process -Id {pid}).WorkingSet64; \
         $f=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory*1KB; \
         Write-Output \"$p $f\""
    );
    let out = Command::new("powershell").args(["-NoProfile", "-Command", &script])
        .output().ok().and_then(|o| String::from_utf8(o.stdout).ok());
    let Some(t) = out else { return (0.0, f64::INFINITY) };
    let mut it = t.split_whitespace();
    (it.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0) / 1e9,
     it.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(f64::INFINITY) / 1e9)
}

type Nbrs = std::collections::HashMap<String, Vec<String>>;
struct Stage { fac: Factored, ro: OpReadout, nbrs: Option<Nbrs>, mode: u8, rungs: u32, per: usize,
              cls: Option<std::collections::HashMap<String, u16>>, bits: u32 }

impl Stage {
    fn new(n_id: u32, n_gram: u32) -> Stage {
        Stage { fac: Factored::new(n_id, n_gram, K_ID, 2), ro: OpReadout::default(),
                nbrs: None, mode: 0, rungs: 0, per: 0, cls: None, bits: 0 }
    }
    /// `extra` is stage 1's guess, empty for stage 1 itself.
    ///
    /// `read` distinguishes the two directions, which matters only in neighbour mode 2:
    /// backoff is a READ-time fallback. Writes stay literal so a memory is always filed
    /// under its own spelling, and the approximate addresses are tried only when the
    /// literal one turns out not to exist.
    fn feats(&self, q: &[String], ctx: &[&str], pos: usize, extra: Option<&str>, read: bool)
        -> (Vec<String>, Vec<String>)
    {
        let (mut id, g) = match (&self.nbrs, self.mode, read) {
            (Some(n), 1, _) => reply::feats_near(q, ctx, pos, n),          // v12: flat
            (Some(n), 2, true) => reply::feats_backoff(q, ctx, pos, n, &self.fac.id.feat_id),
            // mode 3: ordered ladder. writes lay down every rung, reads take the
            // finest rung that exists and only fall through when there is none.
            (n, 3, true) => reply::feats_ladder(
                q, ctx, pos, self.rungs, n.as_ref(), &self.fac.id.feat_id),
            _ => match &self.cls {
                Some(c) => reply::feats_class(q, ctx, pos, self.rungs, self.bits, c),
                None => reply::feats_rungs(q, ctx, pos, self.rungs),
            },
        };
        if let Some(w) = extra {
            id.push(format!("S1:{w}|R:{}", ctx.join(" ")));
            id.push(format!("S1:{w}|P1:{}", ctx.last().copied().unwrap_or(BOS)));
        }
        (id, g)
    }
    fn settle(&mut self, q: &[String], ctx: &[&str], pos: usize, extra: Option<&str>) {
        let (id, g) = self.feats(q, ctx, pos, extra, false);
        if self.per > 0 { self.fac.learn_feats_each(&id, &g, self.per); }
        else { self.fac.learn_feats(&id, &g); }
    }
    fn code(&self, id: &[String], g: &[String]) -> Vec<NeuronId> {
        let (idc, gc) = if self.per > 0 { self.fac.query_feats_each(id, g, self.per) }
                        else { self.fac.query_feats(id, g, K_ID) };
        seq::code_of(idc, &gc)
    }
    fn write(&mut self, brain: &mut v4::Brain, q: &[String], ctx: &[&str], pos: usize,
             extra: Option<&str>, target: &str) {
        let (id, g) = self.feats(q, ctx, pos, extra, false);
        let code = self.code(&id, &g);
        brain.learn_bulk(&code, 1);
        self.ro.learn(&code, target);
    }
    fn ask(&self, q: &[String], ctx: &[&str], pos: usize, extra: Option<&str>, top: usize)
        -> Vec<(String, f32)>
    {
        let (id, g) = self.feats(q, ctx, pos, extra, true);
        self.ro.decode(&self.code(&id, &g), top)
    }
}

fn main() -> std::io::Result<()> {
    let a: Vec<String> = env::args().collect();
    let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(30_000);
    let two: bool = a.get(2).and_then(|s| s.parse::<u32>().ok()).unwrap_or(0) != 0;
    // CREDIT ASSIGNMENT: how many error-correcting epochs after the Hebbian write.
    let epochs: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    // NEIGHBOUR KEYS: address each memory by distributionally similar words too.
    //   0 off   1 flat (v12: neighbours compete with the exact key at write AND read)
    //   2 ordered backoff (v13: read-time only, and only when no exact address exists)
    let near: u8 = a.get(4).and_then(|s| s.parse::<u8>().ok()).unwrap_or(0);
    // REPLY-SIDE LADDER RUNGS: bitmask, see reply::feats_rungs.
    let rungs: u32 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);
    // PER-KEY UNITS: 0 = the old shared budget of K_ID winners across all identity
    // keys; >0 = each key competes alone for this many units and the code is the union.
    let per: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(0);
    // IDENTITY POOL, in millions. 5M was hard-coded and it is a CAPACITY, not a
    // constant: doubling the corpus to 60k exchanges against the same 5M neurons took
    // top-1 from 36.47% to 23.32% and accuracy on fully-addressed positions from
    // 87.37% to 16.92%. An addressed position scoring 17% is not a model that has
    // learned the wrong thing, it is a model whose addresses have collided — the
    // reply-side rungs alone intern roughly 5 keys per position, so 58,800 exchanges
    // ask for ~4.7M unit-slots out of 5M neurons and every neuron is claimed.
    // v7's reachability law said this in one line: reachable = facts x 16.
    // CLASS-CROSSED KEYS: bitmask, see reply::push_class_keys. 0 = off.
    let bits: u32 = a.get(8).and_then(|s| s.parse().ok()).unwrap_or(0);
    let nclass: usize = a.get(9).and_then(|s| s.parse().ok()).unwrap_or(128);
    let pool: u32 = a.get(7).and_then(|s| s.parse::<u32>().ok())
        .map(|m| m * 1_000_000)
        .unwrap_or_else(|| 5_000_000 * ((n as u32 + 29_999) / 30_000).max(1));
    let (base, _) = mem_gb();

    let raw = fs::read_to_string(CORPUS)?;
    let want = n + TEST;
    let mut pairs: Vec<(Vec<String>, Vec<String>)> = Vec::with_capacity(want);
    for line in raw.lines() {
        let mut p = line.split('\t');
        let (Some(_), Some(q), Some(r)) = (p.next(), p.next(), p.next()) else { continue };
        let (qt, rt) = (seq::tokenize(q), seq::tokenize(r));
        if !qt.is_empty() && !rt.is_empty() && rt.len() <= 16 && qt.len() <= 20 {
            pairs.push((qt, rt));
            if pairs.len() >= want { break; }
        }
    }
    drop(raw);
    let (test, train) = pairs.split_at(TEST.min(pairs.len() / 4));
    println!("== depth {} : {} train / {} held out ==",
        if two { 2 } else { 1 }, train.len(), test.len());

    let t0 = Instant::now();
    println!("   identity pool {:.1}M neurons for {} exchanges",
        pool as f64 / 1e6, train.len());
    let mut s1 = Stage::new(pool, 500_000);
    s1.rungs = rungs;
    s1.per = per;
    s1.bits = bits;
    if bits != 0 {
        let mut docs: Vec<&[String]> = Vec::with_capacity(train.len() * 2);
        for (q, r) in train { docs.push(q.as_slice()); docs.push(r.as_slice()); }
        let (prof, freq, _) = design_tests::wordclass::profiles(docs.into_iter(), 2, 20);
        let (cls, ex) = design_tests::wordclass::exemplar_classes(&prof, &freq, nclass);
        println!("   {} words -> {nclass} classes at {:?}", cls.len(), t0.elapsed());
        // what did the classes actually carve up? print a few, largest first.
        let mut members: std::collections::HashMap<u16, Vec<&String>> = Default::default();
        for (w, c) in &cls { members.entry(*c).or_default().push(w); }
        let mut ks: Vec<u16> = members.keys().copied().collect();
        ks.sort_unstable_by_key(|k| std::cmp::Reverse(members[k].len()));
        for k in ks.iter().take(5) {
            let mut m: Vec<&str> = members[k].iter().map(|s| s.as_str()).collect();
            m.sort_unstable();
            println!("     [{}] {:>4} words: {}", ex[*k as usize], m.len(),
                m.iter().take(12).cloned().collect::<Vec<_>>().join(" "));
        }
        println!("   class bits 0b{bits:04b}");
        s1.cls = Some(cls);
    }
    if per > 0 { println!("   per-key identity units: {per} (union, not a shared budget)"); }
    if rungs != 0 { println!("   reply-side rungs: 0b{rungs:04b}"); }
    if near == 3 { s1.mode = 3; }
    if near == 1 || near == 2 || near == 3 {
        let mut docs: Vec<&[String]> = Vec::with_capacity(train.len() * 2);
        for (q, r) in train { docs.push(q.as_slice()); docs.push(r.as_slice()); }
        let (prof, _, _) = design_tests::wordclass::profiles(docs.into_iter(), 2, 20);
        let nb = design_tests::wordclass::neighbours(&prof, 3, 0.08);
        println!("   learned neighbours for {} words at {:?}", nb.len(), t0.elapsed());
        for probe in ["monday", "eat", "happy"] {
            if let Some(n) = nb.get(probe) { println!("     {probe} ~ {}", n.join(" ")); }
        }
        s1.nbrs = Some(nb);
        if near != 3 { s1.mode = near; }
        println!("   neighbour mode {} ({})", near,
            if near == 1 { "flat, competes with the exact key" }
            else { "ordered backoff, read-time only" });
    }
    let mut b1 = v4::Brain::new(s1.fac.n_total, Params::default());
    let mut buf = Vec::new();

    for pass in 0..3 {
        for (i, (q, r)) in train.iter().enumerate() {
            if i % 5_000 == 0 && i > 0 && mem_gb().1 < MIN_FREE_GB {
                println!("   !! stopping, low memory"); break;
            }
            reply::turns(r, ORDER, &mut buf);
            for t in &buf {
                if pass < 2 { s1.settle(q, &t.ctx, t.pos, None); }
                else { s1.write(&mut b1, q, &t.ctx, t.pos, None, t.next); }
            }
        }
    }
    println!("   stage 1 trained at {:?}, {:.2} GB", t0.elapsed(), mem_gb().0 - base);

    // ---- ERROR-CORRECTING EPOCHS. The Hebbian pass above counted co-occurrence and
    // never asked whether the answer was right. These passes ask: predict, and only
    // when wrong, strengthen the true answer and weaken the one that won.
    for ep in 0..epochs {
        let (mut seen, mut ok) = (0usize, 0usize);
        for (i, (q, r)) in train.iter().enumerate() {
            if i % 5_000 == 0 && i > 0 && mem_gb().1 < MIN_FREE_GB {
                println!("   !! stopping epoch, low memory"); break;
            }
            reply::turns(r, ORDER, &mut buf);
            for t in &buf {
                let (id, g) = s1.feats(q, &t.ctx, t.pos, None, false);
                let code = s1.code(&id, &g);
                if s1.ro.learn_delta(&code, t.next, 1) { ok += 1; }
                seen += 1;
            }
        }
        println!("   epoch {}: {:.2}% of training positions already correct before update",
            ep + 1, ok as f64 / seen as f64 * 100.0);
    }

    // ---- STAGE 2: same machinery, plus stage 1's guess as an input.
    let mut s2 = Stage::new(pool, 500_000);
    let mut b2 = v4::Brain::new(s2.fac.n_total, Params::default());
    if two {
        for pass in 0..3 {
            for (i, (q, r)) in train.iter().enumerate() {
                if i % 5_000 == 0 && i > 0 && mem_gb().1 < MIN_FREE_GB {
                    println!("   !! stopping stage 2, low memory"); break;
                }
                reply::turns(r, ORDER, &mut buf);
                for t in &buf {
                    let guess = s1.ask(q, &t.ctx, t.pos, None, 1)
                        .into_iter().next().map(|(w, _)| w);
                    let g = guess.as_deref();
                    if pass < 2 { s2.settle(q, &t.ctx, t.pos, g); }
                    else { s2.write(&mut b2, q, &t.ctx, t.pos, g, t.next); }
                }
            }
        }
        println!("   stage 2 trained at {:?}, {:.2} GB", t0.elapsed(), mem_gb().0 - base);
    }

    let answer = |q: &[String], ctx: &[&str], pos: usize, top: usize| -> Vec<(String, f32)> {
        let first = s1.ask(q, ctx, pos, None, 1).into_iter().next().map(|(w, _)| w);
        if !two { return s1.ask(q, ctx, pos, None, top); }
        let out = s2.ask(q, ctx, pos, first.as_deref(), top);
        if out.is_empty() { s1.ask(q, ctx, pos, None, top) } else { out }
    };

    // ---- how often does stage 2 CHANGE stage 1, and is it right to?
    let (mut n_all, mut t1, mut t5) = (0usize, 0usize, 0usize);
    let (mut changed, mut fixed, mut broke) = (0usize, 0usize, 0usize);
    let mut b3 = Vec::new();
    // IS REACHABILITY STILL THE BOTTLENECK? Every fix that ever worked in this project
    // made stored knowledge easier to REACH. So before adding another addressing
    // scheme, ask what fraction of held-out positions have no literal address at all,
    // and whether having one is what separates the right answers from the wrong ones.
    // rung[r] = positions where exactly r of {Q:, QS:, QW:} are interned; hit[r] = right.
    let (mut rung, mut hit) = ([0usize; 4], [0usize; 4]);
    for (q, r) in test {
        reply::turns(r, ORDER, &mut b3);
        for t in &b3 {
            let one = s1.ask(q, &t.ctx, t.pos, None, 1).into_iter().next().map(|(w, _)| w);
            let got = answer(q, &t.ctx, t.pos, 5);
            let top = got.first().map(|(w, _)| w.clone());
            if top.as_deref() == Some(t.next) { t1 += 1; }
            if got.iter().any(|(w, _)| w == t.next) { t5 += 1; }
            {
                let (id, _) = s1.feats(q, &t.ctx, t.pos, None, true);
                // count only the LITERAL rungs. class keys (`C...`) are interned for
                // nearly every position, so counting them would redefine the buckets
                // and make this incomparable to the runs it exists to be compared with.
                let r = id.iter().filter(|k| !k.starts_with('C'))
                    .filter(|k| s1.fac.id.feat_id.contains_key(k.as_str())).count();
                let r = r.min(3);
                rung[r] += 1;
                if top.as_deref() == Some(t.next) { hit[r] += 1; }
            }
            if two && one != top {
                changed += 1;
                let was = one.as_deref() == Some(t.next);
                let now = top.as_deref() == Some(t.next);
                if !was && now { fixed += 1 }
                if was && !now { broke += 1 }
            }
            n_all += 1;
        }
    }
    println!("\n   next word : top-1 {:.2}%   top-5 {:.2}%   ({n_all} positions)",
        t1 as f64 / n_all as f64 * 100.0, t5 as f64 / n_all as f64 * 100.0);
    println!("   by how many identity keys are interned:");
    for r in 0..4 {
        if rung[r] == 0 { continue }
        let lab = if r == 3 { "3+" } else { "  " };
        println!("     {r}{lab} interned : {:>5} positions ({:>5.1}%)   top-1 {:.2}%",
            rung[r], rung[r] as f64 / n_all as f64 * 100.0,
            hit[r] as f64 / rung[r] as f64 * 100.0);
    }
    if two {
        println!("   stage 2 overruled stage 1 {changed} times: {fixed} fixed, {broke} broke, \
{} neutral", changed - fixed - broke);
        println!("   -> net {:+} correct answers from having a second layer",
            fixed as i64 - broke as i64);
    }

    let gen = |text: &str| -> String {
        let q = seq::tokenize(text);
        let mut out: Vec<String> = Vec::new();
        for _ in 0..12 {
            let have = out.len().min(ORDER);
            let mut c: Vec<&str> = vec![BOS; ORDER];
            for k in 0..have { c[ORDER - have + k] = out[out.len() - have + k].as_str(); }
            let Some((w, _)) = answer(&q, &c, out.len(), 1).into_iter().next() else { break };
            if w == EOS { break; }
            out.push(w);
        }
        if out.is_empty() { "(silence)".into() } else { out.join(" ") }
    };
    println!();
    for p in ["hello", "thank you very much", "i love you", "what time is it ?",
              "i am hungry", "where are you going ?"] {
        println!("   you> {:<22} ava> {}", p, gen(p));
    }
    println!("\nTSV\tdepth{}\t{}\t{:.2}\t{:.2}", if two { 2 } else { 1 }, train.len(),
        t1 as f64 / n_all as f64 * 100.0, t5 as f64 / n_all as f64 * 100.0);
    Ok(())
}
