// Talk to ava.
//
//   cargo run --release --example converse -- 70000 3 8000000 800000 20   (once, to train)
//   cargo run --release --example chat                                    (then, to talk)
//
// Loads the conversational brain once and keeps it in memory, so each reply is a
// few milliseconds instead of a two-minute retrain.
//
// SELF-MODEL (v19). At start-up she counts her own hits and misses on the exchanges
// converse.rs held out (the last 1% of the slice it trained on, which she never learned)
// and builds P(right | rung, margin decile). Every reply then says how sure she is, and
// below 50% she says so before guessing. If you trained with a different exchange count,
// pass it:  cargo run --release --example chat -- 70000
use design_tests::reply;
use design_tests::seq::{self, BOS, EOS};
use design_tests::store;
use design_tests::v4::Params;
use design_tests::v6::*;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::time::Instant;

const BRAIN: &str = "../brain/ava-chat.brain";
const CORPUS: &str = "../data/replies.tsv";
const K_ID: usize = 6;
const ORDER: usize = 3;

/// One step of her reply: her word, how many identity keys she reached, and the margin
/// s1/(s1+s2) of the vote (v19: AUC 0.86 on its own, and informative inside every rung).
fn state(fac: &Factored, ro: &OpReadout, q: &[String], ctx: &[&str], pos: usize)
    -> (Option<String>, usize, f32)
{
    let (idf, gf) = reply::feats(q, ctx, pos);
    let rung = idf.iter().filter(|k| fac.id.feat_id.contains_key(k.as_str())).count().min(3);
    let (idc, gc) = fac.query_feats(&idf, &gf, K_ID);
    let top = ro.decode(&seq::code_of(idc, &gc), 2);
    let m = match (top.first(), top.get(1)) {
        (Some(a), Some(b)) if a.1 + b.1 > 0.0 => a.1 / (a.1 + b.1),
        (Some(_), _) => 1.0,
        _ => 0.0,
    };
    (top.into_iter().next().map(|(w, _)| w), rung, m)
}

/// Her whole reply, word by word, and how sure she is of it: the mean of her per-word
/// chance of matching what a person would say.
fn speak(fac: &Factored, ro: &OpReadout, conf: Option<&Conf>, q: &[String]) -> (Vec<String>, Option<f64>) {
    let mut out: Vec<String> = Vec::new();
    let mut ps: Vec<f64> = Vec::new();
    for _ in 0..20 {
        let have = out.len().min(ORDER);
        let mut ctx: Vec<&str> = vec![BOS; ORDER];
        for i in 0..have {
            ctx[ORDER - have + i] = out[out.len() - have + i].as_str();
        }
        let (w, r, m) = state(fac, ro, q, &ctx, out.len());
        let Some(w) = w else { break };
        if let Some(c) = conf { ps.push(c.p(r, m)); }
        if w == EOS { break; }
        out.push(w);
    }
    let sure = if ps.is_empty() { None } else { Some(ps.iter().sum::<f64>() / ps.len() as f64) };
    (out, sure)
}

/// P(her word is the one a person said | rung, margin decile), counted on held-out data,
/// and `line`: the confidence below which a reply is in her own bottom quarter.
struct Conf { edges: Vec<f32>, p: [[f64; 10]; 4], line: f64 }
impl Conf {
    fn decile(&self, m: f32) -> usize { self.edges.iter().filter(|e| m >= **e).count() }
    fn p(&self, rung: usize, m: f32) -> f64 { self.p[rung][self.decile(m)] }
    /// returns the table, how many positions it was counted on, and her hit rate there.
    fn fit(fac: &Factored, ro: &OpReadout, n_pairs: usize) -> Option<(Conf, usize, f64)> {
        let raw = std::fs::read_to_string(CORPUS).ok()?;
        // exactly converse.rs's loader, so the tail is the slice she never learned.
        let mut pairs: Vec<(Vec<String>, Vec<String>)> = Vec::new();
        for line in raw.lines() {
            let Some((q, r)) = line.split_once('\t') else { continue };
            let (qt, rt) = (seq::tokenize(q), seq::tokenize(r));
            if !qt.is_empty() && rt.len() >= 1 && rt.len() <= 16 && qt.len() <= 20 {
                pairs.push((qt, rt));
                if pairs.len() >= n_pairs { break; }
            }
        }
        let cut = pairs.len() * 99 / 100;
        let mut obs: Vec<(usize, f32, bool)> = Vec::new();
        let mut buf = Vec::new();
        for (q, r) in &pairs[cut..] {
            reply::turns(r, ORDER, &mut buf);
            for t in &buf {
                let (w, rung, m) = state(fac, ro, q, &t.ctx, t.pos);
                obs.push((rung, m, w.as_deref() == Some(t.next)));
            }
        }
        if obs.len() < 200 { return None; }
        let mut ms: Vec<f32> = obs.iter().map(|o| o.1).collect();
        ms.sort_by(|a, b| a.total_cmp(b));
        let mut c = Conf { edges: (1..10).map(|k| ms[ms.len() * k / 10]).collect(), p: [[0.0; 10]; 4], line: 0.0 };
        let mut n = [[[0usize; 2]; 10]; 4];
        for o in &obs {
            let d = c.decile(o.1);
            n[o.0][d][1] += 1;
            if o.2 { n[o.0][d][0] += 1; }
        }
        for r in 0..4 { for d in 0..10 {
            c.p[r][d] = (n[r][d][0] as f64 + 1.0) / (n[r][d][1] as f64 + 2.0);   // Laplace
        } }
        // what counts as "unsure" for HER: answer every held-out prompt and draw the line at
        // the bottom quarter of her own reply confidence. matching a person word for word is
        // a harsh target in open dialogue (her replies average ~20%), so a flat 50% line
        // fired on every reply and told you nothing.
        let mut rc: Vec<f64> = pairs[cut..].iter().filter_map(|(q, _)| speak(fac, ro, Some(&c), q).1).collect();
        rc.sort_by(|a, b| a.total_cmp(b));
        c.line = rc.get(rc.len() / 4).copied().unwrap_or(0.0);
        let right = obs.iter().filter(|o| o.2).count() as f64 / obs.len() as f64;
        Some((c, obs.len(), right))
    }
}

fn main() -> rustyline::Result<()> {
    println!("waking ava...");
    let t = Instant::now();
    let (brain, fac, ro) = match store::load(BRAIN, Params::default()) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("could not load {BRAIN}: {e}");
            eprintln!("teach her to talk first:");
            eprintln!("  cargo run --release --example converse -- 70000 3 8000000 800000 20");
            return Ok(());
        }
    };
    let syn: u64 = brain.units.iter().map(|u| u.out.len() as u64).sum();
    println!(
        "awake. {} neurons, {} synapses, {} things she can say, loaded in {:?}\n",
        brain.units.len(), syn, ro.ops.len(), t.elapsed()
    );
    let n_pairs: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(70_000);
    let conf = match Conf::fit(&fac, &ro, n_pairs) {
        Some((c, n, right)) => {
            println!("self-model: counted on {n} positions she never learned (right on {:.1}% of them)",
                right * 100.0);
            println!("            below ~{:.0}% sure she says so (her own bottom quarter)", c.line * 100.0);
            Some(c)
        }
        None => { println!("self-model: could not read {CORPUS}, replies will not say how sure she is"); None }
    };
    println!("just talk to her. :q to leave.\n");

    let mut rl = DefaultEditor::new()?;
    loop {
        match rl.readline("you> ") {
            Ok(line) => {
                let line = line.trim().to_string();
                if line.is_empty() { continue; }
                if line == ":q" || line == ":quit" { break; }
                rl.add_history_entry(line.as_str())?;

                let q = seq::tokenize(&line);
                let t = Instant::now();
                let (out, sure) = speak(&fac, &ro, conf.as_ref(), &q);
                let dt = t.elapsed();
                // Which route answered? `any key interned` was useless here: the
                // BACKED-OFF key `QW:{last content word}` is interned for nearly every
                // prompt — that is precisely why the backoff fix worked — so it
                // reported "remembered" for everything, improvisation included.
                //
                // The honest question is which RUNG of the ladder she reached.
                let (idf, _) = reply::feats(&q, &vec![BOS; ORDER], 0);
                let rung = if idf.first().map_or(false, |f| fac.id.feat_id.contains_key(f)) {
                    "remembered — she has heard this exact sentence"
                } else if idf.get(1).map_or(false, |f| fac.id.feat_id.contains_key(f)) {
                    "recognised — same words, new sentence"
                } else if idf.get(2).map_or(false, |f| fac.id.feat_id.contains_key(f)) {
                    "improvised from one familiar word"
                } else {
                    "improvised — nothing familiar at all"
                };

                // how sure: the mean of her per-word chance of matching what a person would
                // say. in her bottom quarter she says so first — v19: confidence separates
                // her right answers from her wrong ones, and until now she sounded the same
                // either way.
                let tag = sure.map_or(String::new(), |s| format!(", ~{:.0}% sure", s * 100.0));
                let unsure = match (sure, &conf) { (Some(s), Some(c)) => s < c.line, _ => false };
                if out.is_empty() {
                    println!("ava> ...   [nothing to say{tag}, {dt:?}]");
                } else if unsure {
                    println!("ava> i'm not sure. my best guess: {}\n     [{rung}{tag}, {dt:?}]", out.join(" "));
                } else {
                    println!("ava> {}\n     [{rung}{tag}, {dt:?}]", out.join(" "));
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
            Err(e) => { eprintln!("{e}"); break; }
        }
    }
    println!("bye");
    Ok(())
}
