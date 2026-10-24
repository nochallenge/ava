// TESTING / DESIGN ONLY.
//
// Rung 2: can she say what comes NEXT?  Rung 3: can she hold a sentence together?
//
//   cargo run --release --example speak -- <sentences> <order> <N_ID> <N_GRAM> <state>
//   cargo run --release --example speak -- 200000 3 8000000 800000 1
//
// <state> 0 = sliding window only (what an n-gram model sees)
// <state> 1 = window PLUS working memory: how far into the sentence she is, and which
//             words she has already used. That is the mechanism a window cannot have.
//
// Same neurons, same competitive allocation, same specificity-weighted vote as the
// morphology brain. Only the encoder and the readout changed.
use design_tests::seq::{self, EOS, USED_MAX};
use design_tests::v4::{self, Params};
use design_tests::v6::*;
use std::env;
use std::fs;
use std::time::Instant;

const CORPUS: &str = "../data/eng_sentences.tsv";

fn rss_gb() -> f64 {
    use std::process::Command;
    let pid = std::process::id();
    Command::new("powershell")
        .args(["-NoProfile", "-Command", &format!("(Get-Process -Id {pid}).WorkingSet64")])
        .output().ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|b| b as f64 / 1e9).unwrap_or(0.0)
}

fn main() -> std::io::Result<()> {
    let a: Vec<String> = env::args().collect();
    let n_sent: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let order: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let n_id: u32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(4_000_000);
    let n_gram: u32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(400_000);
    let state: bool = a.get(5).and_then(|s| s.parse::<u32>().ok()).unwrap_or(0) != 0;
    // HARD CEILING. An order-5 run silently climbed to 7.3 GB and took the machine
    // down to 2.1 GB free before it was killed by hand. A run that has to be watched
    // is a run that will eventually be left unwatched, so the budget goes in the code.
    let max_gb: f64 = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(6.0);
    const K_ID: usize = 6;
    const PER_TAG: usize = 2;

    let base = rss_gb();
    println!("== sequence: {n_sent} sentences, order {order}, {} neurons, working memory {} ==",
        n_id as u64 + n_gram as u64, if state { "ON" } else { "off" });

    let raw = fs::read_to_string(CORPUS)?;
    let mut sents: Vec<Vec<String>> = Vec::with_capacity(n_sent);
    for line in raw.lines() {
        let Some(text) = line.split('\t').nth(2) else { continue };
        let t = seq::tokenize(text);
        if t.len() >= 3 && t.len() <= 18 {
            sents.push(t);
            if sents.len() >= n_sent { break; }
        }
    }
    drop(raw);
    let cut = sents.len() * 98 / 100;
    let (train, test) = sents.split_at(cut);
    let positions: usize = train.iter().map(|s| s.len() + 1).sum();
    println!("   {} train / {} held-out sentences, {} positions to learn",
        train.len(), test.len(), positions);

    // 7th arg: identity backoff (the v9 fix), so this is a clean A/B in one binary.
    // Defaults ON: measured at 200k sentences, it costs 0.3 points of top-1 (within
    // noise) and saves 21% of the synapses — 139,294,952 -> 110,520,776.
    let backoff: bool = a.get(7).and_then(|s| s.parse::<u32>().ok()).unwrap_or(1) != 0;
    println!("   identity route: {}", if backoff { "BACKOFF LADDER" } else { "single exact key" });
    let enc = |ctx: &[&str], pos: usize, used: &[&str]| -> (Vec<String>, Vec<String>) {
        if state { seq::feats_state(ctx, pos, used) } else { seq::feats_opt(ctx, backoff) }
    };

    let mut fac = Factored::new(n_id, n_gram, K_ID, PER_TAG);
    let mut ro = OpReadout::default();
    let mut brain = v4::Brain::new(fac.n_total, Params::default());

    // passes 1-2 settle the competition; pass 3 writes with the allocator FROZEN.
    let t0 = Instant::now();
    let mut buf = Vec::new();
    let mut aborted = false;
    for pass in 0..3 {
        for (si, s) in train.iter().enumerate() {
            if si % 20_000 == 0 && si > 0 {
                let g = rss_gb() - base;
                if g > max_gb {
                    println!("   !! ABORTING at {g:.2} GB (ceiling {max_gb:.1} GB), {si}/{} sentences into pass {}", train.len(), pass + 1);
                    aborted = true;
                    break;
                }
            }
            seq::steps(s, order, &mut buf);
            for st in &buf {
                let (idf, gf) = enc(&st.ctx, st.pos, &st.used);
                if pass < 2 {
                    fac.learn_feats(&idf, &gf);
                } else {
                    let (idc, gc) = fac.query_feats(&idf, &gf, K_ID);
                    let code = seq::code_of(idc, &gc);
                    brain.learn_bulk(&code, 1);
                    ro.learn(&code, st.next);
                }
            }
        }
        if aborted { break; }
        println!("   pass {} done at {:?}", pass + 1, t0.elapsed());
    }
    if aborted {
        println!("   run stopped early to protect the machine; numbers below are partial.");
    }
    brain.compact();
    let train_t = t0.elapsed();
    let peak = rss_gb();
    println!("   trained in {train_t:?}, {:.2} GB, {} synapses, vocabulary {}",
        peak - base, brain.total_synapses(), ro.ops.len());

    let ask = |ctx: &[&str], pos: usize, used: &[&str], top: usize| -> Vec<(String, f32)> {
        let (idf, gf) = enc(ctx, pos, used);
        let (idc, gc) = fac.query_feats(&idf, &gf, K_ID);
        ro.decode(&seq::code_of(idc, &gc), top)
    };

    // ---- 1. next-word prediction on sentences she has never seen
    let (mut n, mut t1, mut t5) = (0usize, 0usize, 0usize);
    let mut buf2 = Vec::new();
    for s in test {
        seq::steps(s, order, &mut buf2);
        for st in &buf2 {
            let got = ask(&st.ctx, st.pos, &st.used, 5);
            if got.first().map(|(w, _)| w.as_str()) == Some(st.next) { t1 += 1; }
            if got.iter().any(|(w, _)| w == st.next) { t5 += 1; }
            n += 1;
        }
    }
    println!("\n   --- next word, on {} held-out sentences ---", test.len());
    println!("   top-1 : {:.1}%   ({t1}/{n})", t1 as f64 / n as f64 * 100.0);
    println!("   top-5 : {:.1}%   ({t5}/{n})", t5 as f64 / n as f64 * 100.0);
    println!("   (a uniform guess over {} words would be {:.4}%)",
        ro.ops.len(), 100.0 / ro.ops.len() as f64);

    // generate from a list of seed words, using her own output as the next context
    let gen = |seed: &[&str], limit: usize| -> String {
        let mut words: Vec<String> = seed.iter().map(|s| s.to_string()).collect();
        for _ in 0..limit {
            let have = words.len().min(order);
            let mut ctx: Vec<&str> = vec![seq::BOS; order];
            for i in 0..have {
                ctx[order - have + i] = words[words.len() - have + i].as_str();
            }
            let scrolled = words.len().saturating_sub(order);
            let used: Vec<&str> = words[scrolled.saturating_sub(USED_MAX)..scrolled]
                .iter().map(|s| s.as_str()).collect();
            let Some((w, _)) = ask(&ctx, words.len(), &used, 1).into_iter().next() else { break };
            if w == EOS { break; }
            words.push(w);
        }
        words.join(" ")
    };

    println!("\n   --- finish the sentence ---");
    for seed in [
        vec!["i", "want", "to"],
        vec!["how", "are", "you"],
        vec!["the", "weather", "is"],
        vec!["she", "said", "that"],
        vec!["i", "don't", "know"],
        vec!["he", "gave", "her", "a"],
        vec!["there", "is", "a", "big"],
    ] {
        println!("   {:<24} -> {}", seed.join(" "), gen(&seed, 16));
    }

    println!("\n   --- start from silence ---");
    for s in [vec!["the"], vec!["i"], vec!["we"], vec!["yesterday"]] {
        println!("   {}", gen(&s, 18));
    }

    // ---- 3. does she finish? a window model runs on; state should learn to stop.
    let mut lens: Vec<usize> = Vec::new();
    let mut ran_on = 0;
    for s in ["the", "i", "we", "he", "she", "they", "it", "you", "there", "my"] {
        let g = gen(&[s], 30);
        let l = g.split_whitespace().count();
        if l >= 30 { ran_on += 1; }
        lens.push(l);
    }
    println!("\n   sentence length when left to run: avg {:.1} words, {ran_on}/10 never stopped",
        lens.iter().sum::<usize>() as f64 / lens.len() as f64);

    let t = Instant::now();
    for i in 0..2000 {
        let s = &test[i % test.len()];
        let c: Vec<&str> = s.iter().take(order).map(|x| x.as_str()).collect();
        let _ = ask(&c, order, &[], 1);
    }
    println!("   speed: {:.0} predictions/sec", 2000.0 / t.elapsed().as_secs_f64());

    println!("\nTSV\t{n_sent}\t{order}\t{}\t{positions}\t{}\t{:.2}\t{:.1}\t{:.1}\t{ran_on}",
        if state { "state" } else { "window" }, brain.total_synapses(), peak - base,
        t1 as f64 / n as f64 * 100.0, t5 as f64 / n as f64 * 100.0);
    Ok(())
}
