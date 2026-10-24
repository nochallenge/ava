// TESTING / DESIGN ONLY.
//
// Does remembering the previous turn make her a better conversationalist?
//
//   cargo run --release --example remember -- <triples> <hist 0|1>
//
// Same three-turn corpus both ways; the only difference is whether the turn BEFORE
// the prompt is visible. Two things get measured, because they are not the same:
//
//   1. next word of the reply, on unseen conversations  — the usual metric
//   2. PRONOUN positions specifically                   — where reference actually
//      matters. if history helps anywhere it must help here, and if it helps only
//      here that is still a real result.
use design_tests::reply;
use design_tests::seq::{self, BOS, EOS};
use design_tests::v4::{self, Params};
use design_tests::v6::*;
use std::env;
use std::fs;
use std::time::Instant;

const CORPUS: &str = "../data/triples.tsv";
const MIN_FREE_GB: f64 = 3.0;

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

/// words whose meaning depends on something said earlier
fn is_referring(w: &str) -> bool {
    matches!(w, "it" | "its" | "they" | "them" | "their" | "he" | "she" | "him"
        | "her" | "his" | "that" | "this" | "those" | "these" | "there" | "one")
}

fn main() -> std::io::Result<()> {
    let a: Vec<String> = env::args().collect();
    let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(70_000);
    let use_hist: bool = a.get(2).and_then(|s| s.parse::<u32>().ok()).unwrap_or(0) != 0;
    let mask: u32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(0xF);
    // DENDRITES: 0 = flat linear sum (a point neuron, depth 1);
    // >1.0 = supralinear summation WITHIN each feature branch, linear across them.
    let dend: f32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let gate: f32 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0.0);
    println!("   readout: {}", if dend > 0.0 {
        format!("DENDRITIC, power {dend}, branch gate {gate}") } else { "flat linear sum".into() });
    let names = ["B suffixes", "P positions", "Q prompt words", "QL last word"];
    let on: Vec<&str> = (0..4).filter(|i| mask & (1 << i) != 0).map(|i| names[i]).collect();
    println!("   general features ON: {}", on.join(", "));
    const ORDER: usize = 3;
    const K_ID: usize = 6;

    let (base, _) = mem_gb();
    let raw = fs::read_to_string(CORPUS)?;
    let mut tri: Vec<(Vec<String>, Vec<String>, Vec<String>)> = Vec::with_capacity(n);
    for line in raw.lines() {
        let mut p = line.split('\t');
        let (Some(h), Some(q), Some(r)) = (p.next(), p.next(), p.next()) else { continue };
        let (ht, qt, rt) = (seq::tokenize(h), seq::tokenize(q), seq::tokenize(r));
        if !qt.is_empty() && !rt.is_empty() && rt.len() <= 16 && qt.len() <= 20 && ht.len() <= 20 {
            tri.push((ht, qt, rt));
            if tri.len() >= n { break; }
        }
    }
    drop(raw);
    let cut = tri.len() * 99 / 100;
    let (train, test) = tri.split_at(cut);
    println!("== memory of the previous turn: {} ==", if use_hist { "ON" } else { "off" });
    println!("   {} train / {} held-out conversations", train.len(), test.len());

    let enc = |h: &[String], q: &[String], ctx: &[&str], pos: usize| {
        if use_hist { reply::feats_mask(q, h, ctx, pos, mask) }
        else { reply::feats_mask(q, &[], ctx, pos, mask) }
    };

    let mut fac = Factored::new(10_000_000, 1_000_000, K_ID, 2);
    let mut ro = OpReadout::default();
    let mut brain = v4::Brain::new(fac.n_total, Params::default());

    let t0 = Instant::now();
    let mut buf = Vec::new();
    let mut aborted = false;
    for pass in 0..3 {
        for (i, (h, q, r)) in train.iter().enumerate() {
            if i % 5_000 == 0 && i > 0 {
                let (ws, free) = mem_gb();
                if free < MIN_FREE_GB {
                    println!("   !! STOPPING at {ws:.2} GB used / {free:.2} GB free, \
{i}/{} into pass {}", train.len(), pass + 1);
                    aborted = true;
                    break;
                }
            }
            reply::turns(r, ORDER, &mut buf);
            for t in &buf {
                let (idf, gf) = enc(h, q, &t.ctx, t.pos);
                if pass < 2 {
                    fac.learn_feats(&idf, &gf);
                } else {
                    let (idc, gc) = fac.query_feats(&idf, &gf, K_ID);
                    let code = seq::code_of(idc, &gc);
                    brain.learn_bulk(&code, 1);
                    ro.learn(&code, t.next);
                }
            }
        }
        if aborted { break; }
        println!("   pass {} at {:?}", pass + 1, t0.elapsed());
    }
    // The guard covered the training passes but NOT this. compact() calls
    // shrink_to_fit per unit, which allocates the smaller buffer before releasing the
    // old one, so memory keeps climbing after the last guarded check — measured: the
    // history arm ended pass 3 at 8.34 GB and reached 9.2 GB / 2.1 GB free here,
    // and had to be killed by hand. Prune first when memory is tight: it makes the
    // vectors smaller BEFORE the copy rather than after.
    let (_, free_now) = mem_gb();
    if free_now < MIN_FREE_GB + 2.0 {
        println!("   only {free_now:.1} GB free — pruning to degree 512 before compacting");
        let cut = brain.compact_pruned(512);
        println!("   pruned {cut} synapses");
    } else {
        brain.compact();
    }
    let (peak, _) = mem_gb();
    println!("   trained in {:?}, {:.2} GB, {} synapses",
        t0.elapsed(), peak - base, brain.total_synapses());

    let next = |h: &[String], q: &[String], ctx: &[&str], pos: usize, top: usize| {
        let (idf, gf) = enc(h, q, ctx, pos);
        let (idc, gc) = fac.query_feats(&idf, &gf, K_ID);
        if dend <= 0.0 {
            return ro.decode(&seq::code_of(idc, &gc), top);
        }
        // Three dendritic branches, matching the factoring the code already has:
        // identity (few units, near-private), reply-context, and prompt. Splitting
        // the grammar units by which family produced them keeps a strongly-agreeing
        // branch from being averaged away by two diffuse ones.
        let win: Vec<NeuronId> = fac.gram.code(
            &gf.iter().filter(|f| f.starts_with('B') || f.starts_with('P'))
                .cloned().collect::<Vec<_>>());
        let pro: Vec<NeuronId> = fac.gram.code(
            &gf.iter().filter(|f| !(f.starts_with('B') || f.starts_with('P')))
                .cloned().collect::<Vec<_>>());
        ro.decode_dendritic(&[&idc, &win, &pro], dend, gate, top)
    };

    let (mut n_all, mut t1, mut t5) = (0usize, 0usize, 0usize);
    let (mut n_ref, mut r1) = (0usize, 0usize);
    let mut b2 = Vec::new();
    for (h, q, r) in test {
        reply::turns(r, ORDER, &mut b2);
        for t in &b2 {
            let got = next(h, q, &t.ctx, t.pos, 5);
            let hit1 = got.first().map(|(w, _)| w.as_str()) == Some(t.next);
            if hit1 { t1 += 1; }
            if got.iter().any(|(w, _)| w == t.next) { t5 += 1; }
            n_all += 1;
            // the positions where reference actually matters
            if is_referring(t.next) { n_ref += 1; if hit1 { r1 += 1; } }
        }
    }
    // ---- RESPONSIVENESS, take two.
    //
    // Take one scored the true reply under the true prompt against a stranger's
    // prompt. It was confounded two ways, so its numbers (49.2 / 47.2 / 17.2) are
    // discarded rather than reported:
    //
    //   1. TIES. Scoring only the top 5 meant that when the true word was in neither
    //      list both sides scored 0.0, and a strict `>` counted every tie as a loss.
    //      With top-1 near 24% ties are the majority, so the ceiling sat below 50%.
    //   2. CONFOUNDED AVAILABILITY. The ablation mask gates only GENERAL features;
    //      identity keys are always emitted. A stranger's `QW:` key can be interned
    //      when the true one is not, handing the stranger the win for reasons that
    //      have nothing to do with responsiveness.
    //
    // This version cannot be confounded that way. Generate a reply to each of many
    // DIFFERENT prompts and count the distinct replies. A model that ignores the
    // prompt produces one reply for everything, however good its perplexity.
    let probe: Vec<&(Vec<String>, Vec<String>, Vec<String>)> = test.iter().take(200).collect();
    let gen_reply = |h: &[String], q: &[String]| -> String {
        let mut out: Vec<String> = Vec::new();
        for _ in 0..12 {
            let have = out.len().min(ORDER);
            let mut c: Vec<&str> = vec![BOS; ORDER];
            for k in 0..have { c[ORDER - have + k] = out[out.len() - have + k].as_str(); }
            let Some((w, _)) = next(h, q, &c, out.len(), 1).into_iter().next() else { break };
            if w == EOS { break; }
            out.push(w);
        }
        out.join(" ")
    };
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut echo = 0usize;
    for (h, q, r) in &probe {
        let g = gen_reply(h, q);
        let want: std::collections::HashSet<&str> =
            r.iter().map(|x| x.as_str()).filter(|w| reply::is_content(w)).collect();
        if g.split_whitespace().any(|w| want.contains(w)) { echo += 1; }
        seen.insert(g);
    }
    let distinct = seen.len();
    let ontopic = echo as f64 / probe.len() as f64 * 100.0;
    println!("
   RESPONSIVENESS");
    println!("     distinct replies : {distinct} of {}   (1 = same thing to everyone)", probe.len());
    println!("     on-topic         : {ontopic:.1}%   (shares a content word with the real reply)");

    println!("\n   next word overall : top-1 {:.2}%   top-5 {:.2}%   ({n_all} positions)",
        t1 as f64 / n_all as f64 * 100.0, t5 as f64 / n_all as f64 * 100.0);
    println!("   referring words   : top-1 {:.2}%              ({n_ref} positions)",
        r1 as f64 / n_ref.max(1) as f64 * 100.0);

    // a conversation where the answer depends on the turn before
    println!("\n   --- does she follow a thread? ---");
    let say = |h: &str, q: &str| -> String {
        let (ht, qt) = (seq::tokenize(h), seq::tokenize(q));
        let mut out: Vec<String> = Vec::new();
        for _ in 0..14 {
            let have = out.len().min(ORDER);
            let mut ctx: Vec<&str> = vec![BOS; ORDER];
            for i in 0..have { ctx[ORDER - have + i] = out[out.len() - have + i].as_str(); }
            let Some((w, _)) = next(&ht, &qt, &ctx, out.len(), 1).into_iter().next() else { break };
            if w == EOS { break; }
            out.push(w);
        }
        if out.is_empty() { "(silence)".into() } else { out.join(" ") }
    };
    for (h, q) in [
        ("i have a dog", "what is its name ?"),
        ("i am going to paris", "when do you leave ?"),
        ("my car broke down", "can you fix it ?"),
        ("do you like coffee ?", "why not ?"),
        ("", "what is its name ?"),
    ] {
        println!("   [before: {:<24}] you> {:<24} ava> {}",
            if h.is_empty() { "(nothing)" } else { h }, q, say(h, q));
    }
    println!("\nTSV\t{}\t{}\t{:.2}\t{:.2}\t{:.2}",
        if use_hist { "hist" } else { "none" }, train.len(),
        t1 as f64 / n_all as f64 * 100.0, t5 as f64 / n_all as f64 * 100.0,
        r1 as f64 / n_ref.max(1) as f64 * 100.0);
    Ok(())
}
