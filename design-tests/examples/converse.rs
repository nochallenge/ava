// TESTING / DESIGN ONLY.
//
// Rung 4: can she answer a person?
//
//   cargo run --release --example converse -- <pairs> <order> <N_ID> <N_GRAM> <max_gb>
//   cargo run --release --example converse -- 200000 3 8000000 800000 6
//
// Same neurons again. Third domain, third encoder, zero changes below the seam.
use design_tests::reply;
use design_tests::store;
use design_tests::seq::{self, BOS, EOS};
use design_tests::v4::{self, Params};
use design_tests::v6::*;
use std::env;
use std::fs;
use std::time::Instant;

const CORPUS: &str = "../data/replies.tsv";

/// (our working set, the WHOLE MACHINE's free physical memory), both in GB.
///
/// Capping our own size was the wrong guard. A 7.5 GB ceiling sounded safe when the
/// machine had 9.5 GB free, and by the time we reached 6.7 GB the system was down to
/// 2.6 GB and about to swap — because everything else on the machine had grown too.
/// The quantity that matters is what is LEFT, not what we hold. One call returns
/// both, which also halves the overhead of asking.
fn mem_gb() -> (f64, f64) {
    use std::process::Command;
    let pid = std::process::id();
    let script = format!(
        "$p=(Get-Process -Id {pid}).WorkingSet64;          $f=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory*1KB;          Write-Output \"$p $f\""
    );
    let out = Command::new("powershell")
        .args(["-NoProfile", "-Command", &script])
        .output().ok()
        .and_then(|o| String::from_utf8(o.stdout).ok());
    let Some(t) = out else { return (0.0, f64::INFINITY) };
    let mut it = t.split_whitespace();
    let ws = it.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0);
    let free = it.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(f64::INFINITY);
    (ws / 1e9, free / 1e9)
}
fn rss_gb() -> f64 { mem_gb().0 }

/// Never let the machine drop below this much free physical memory.
const MIN_FREE_GB: f64 = 3.0;

fn main() -> std::io::Result<()> {
    let a: Vec<String> = env::args().collect();
    let n_pairs: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let order: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);
    let n_id: u32 = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(8_000_000);
    let n_gram: u32 = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(800_000);
    let max_gb: f64 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(6.0);
    const K_ID: usize = 6;
    const PER_TAG: usize = 2;

    let base = rss_gb();
    let raw = fs::read_to_string(CORPUS)?;
    let mut pairs: Vec<(Vec<String>, Vec<String>)> = Vec::with_capacity(n_pairs);
    for line in raw.lines() {
        let Some((q, r)) = line.split_once('\t') else { continue };
        let (qt, rt) = (seq::tokenize(q), seq::tokenize(r));
        if !qt.is_empty() && rt.len() >= 1 && rt.len() <= 16 && qt.len() <= 20 {
            pairs.push((qt, rt));
            if pairs.len() >= n_pairs { break; }
        }
    }
    drop(raw);
    let cut = pairs.len() * 99 / 100;
    let (train, test) = pairs.split_at(cut);
    let positions: usize = train.iter().map(|(_, r)| r.len() + 1).sum();
    println!("== reply: {} exchanges, {} positions, order {order}, {} neurons ==",
        train.len(), positions, n_id as u64 + n_gram as u64);

    let mut fac = Factored::new(n_id, n_gram, K_ID, PER_TAG);
    let mut ro = OpReadout::default();
    let mut brain = v4::Brain::new(fac.n_total, Params::default());

    let t0 = Instant::now();
    let mut buf = Vec::new();
    let mut aborted = false;
    for pass in 0..3 {
        for (i, (q, r)) in train.iter().enumerate() {
            if i % 2_000 == 0 && i > 0 {
                let (ws, free) = mem_gb();
                if ws - base > max_gb || free < MIN_FREE_GB {
                    println!("   !! STOPPING at {:.2} GB used / {:.2} GB free on the machine, {i}/{} into pass {}", ws - base, free, train.len(), pass + 1);
                    aborted = true;
                    break;
                }
            }
            reply::turns(r, order, &mut buf);
            for t in &buf {
                let (idf, gf) = reply::feats(q, &t.ctx, t.pos);
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
    brain.compact();
    let peak = rss_gb();
    println!("   trained in {:?}, {:.2} GB, {} synapses, vocabulary {}",
        t0.elapsed(), peak - base, brain.total_synapses(), ro.ops.len());

    let next = |q: &[String], ctx: &[&str], pos: usize, top: usize| -> Vec<(String, f32)> {
        let (idf, gf) = reply::feats(q, ctx, pos);
        let (idc, gc) = fac.query_feats(&idf, &gf, K_ID);
        ro.decode(&seq::code_of(idc, &gc), top)
    };

    let answer = |text: &str| -> String {
        let q = seq::tokenize(text);
        let mut out: Vec<String> = Vec::new();
        for _ in 0..16 {
            let have = out.len().min(order);
            let mut ctx: Vec<&str> = vec![BOS; order];
            for i in 0..have {
                ctx[order - have + i] = out[out.len() - have + i].as_str();
            }
            let Some((w, _)) = next(&q, &ctx, out.len(), 1).into_iter().next() else { break };
            if w == EOS { break; }
            out.push(w);
        }
        if out.is_empty() { "(silence)".into() } else { out.join(" ") }
    };

    // ---- how often is her next word the one a real person actually said?
    let (mut n, mut t1, mut t5) = (0usize, 0usize, 0usize);
    let mut b2 = Vec::new();
    for (q, r) in test {
        reply::turns(r, order, &mut b2);
        for t in &b2 {
            let got = next(q, &t.ctx, t.pos, 5);
            if got.first().map(|(w, _)| w.as_str()) == Some(t.next) { t1 += 1; }
            if got.iter().any(|(w, _)| w == t.next) { t5 += 1; }
            n += 1;
        }
    }
    println!("\n   --- next word of the reply, on {} unseen exchanges ---", test.len());
    println!("   top-1 : {:.1}%   top-5 : {:.1}%   (chance {:.4}%)",
        t1 as f64 / n as f64 * 100.0, t5 as f64 / n as f64 * 100.0,
        100.0 / ro.ops.len() as f64);

    println!("\n   --- talking to her ---");
    for q in [
        "hello", "how are you ?", "what is your name ?", "thank you very much",
        "i am hungry", "do you like music ?", "what time is it ?",
        "i had a terrible day", "where are you going ?", "goodbye",
        "can you help me ?", "i love you",
    ] {
        println!("   you> {q}\n   ava> {}\n", answer(q));
    }

    let t = Instant::now();
    for i in 0..300 { let _ = answer(if i % 2 == 0 { "hello" } else { "how are you ?" }); }
    println!("   speed: {:.0} replies/sec", 300.0 / t.elapsed().as_secs_f64());

    // Persist, so she can be talked to without retraining for two minutes first.
    // store.rs is generic over Factored/OpReadout — the conversational brain saves
    // through exactly the same code as the morphology one, no changes needed.
    std::fs::create_dir_all("../brain").ok();
    let t = Instant::now();
    match store::save("../brain/ava-chat.brain", &brain, &fac, &ro) {
        Ok(b) => println!("   saved {} MB to ../brain/ava-chat.brain in {:?}",
            b / 1_000_000, t.elapsed()),
        Err(e) => println!("   could not save: {e}"),
    }

    // ---- WHY does she degenerate on some prompts?
    //
    // Four guesses in a row about this system have been wrong, so: measure instead.
    // The cue is three groups of units with different jobs. Decode each group ALONE
    // and the vote that is actually winning becomes visible.
    //
    //   IDENTITY : the exact (prompt, reply-context) pair. memorises.
    //   WINDOW   : backoff over my reply so far. knows English word order.
    //   PROMPT   : their content words, crossed with reply position. knows the TOPIC.
    //
    // If PROMPT is being drowned out, the reply will be fluent and irrelevant, which
    // is exactly what `i . . .` looks like.
    println!("
   --- who is actually voting? ---");
    for probe in ["i love you", "do you like music ?", "thank you very much"] {
        let q = seq::tokenize(probe);
        println!("
   prompt: \"{probe}\"");
        let mut out: Vec<String> = Vec::new();
        for step in 0..4 {
            let have = out.len().min(order);
            let mut ctx: Vec<&str> = vec![BOS; order];
            for i in 0..have {
                ctx[order - have + i] = out[out.len() - have + i].as_str();
            }
            let (idf, gf) = reply::feats(&q, &ctx, out.len());

            // split the general features into window-derived and prompt-derived
            let (win, pro): (Vec<String>, Vec<String>) = gf.into_iter()
                .partition(|f| f.starts_with('B') || f.starts_with('P'));

            let id_units = fac.id.allocate(&idf, K_ID);
            let win_units = fac.gram.code(&win);
            let pro_units = fac.gram.code(&pro);

            let show = |label: &str, units: &[NeuronId]| {
                let top = ro.decode(units, 2);
                let a = top.first().map(|(w, c)| format!("{w} {c:.1}")).unwrap_or("-".into());
                let b = top.get(1).map(|(w, c)| format!("{w} {c:.1}")).unwrap_or("-".into());
                format!("{label}[{:>2}u] {:<18} {:<18}", units.len(), a, b)
            };
            let mut all = id_units.clone();
            all.extend_from_slice(&win_units);
            all.extend_from_slice(&pro_units);
            let chosen = ro.decode(&all, 1).into_iter().next()
                .map(|(w, c)| format!("{w} ({c:.1})")).unwrap_or("(silence)".into());

            println!("     step {step}: ctx=[{}]  -> {}", ctx.join(" "), chosen);
            println!("       {}", show("id  ", &id_units));
            println!("       {}", show("win ", &win_units));
            println!("       {}", show("prom", &pro_units));

            let Some((w, _)) = ro.decode(&all, 1).into_iter().next() else { break };
            if w == EOS { break; }
            out.push(w);
        }
    }
    Ok(())
}
