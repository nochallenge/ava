// ava — teach her, save her, then ask her things in a later run.
//
//   cargo run --release --example ava -- teach 80000
//   cargo run --release --example ava -- ask eat V PST
//   cargo run --release --example ava -- ask wug N PL
//   cargo run --release --example ava -- stats
use design_tests::living::Fact;
use design_tests::store;
use design_tests::v4::{self, Params};
use design_tests::v6::*;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::time::Instant;

const CORPUS: &str = "../data/unimorph-eng.tsv";
const BRAIN: &str = "../brain/ava.brain";

const N_ID: u32 = 10_000_000;
const N_GRAM: u32 = 200_000;
const K_ID: usize = 10;
const PER_TAG: usize = 3;

fn ask(fac: &Factored, readout: &OpReadout, lemma: &str, tags: &[String], top: usize)
    -> Vec<(String, f32)>
{
    let f = Fields { lemma, form: None, tags };
    let (idc, gc) = fac.query_code(&f, K_ID);
    let mut cue = idc;
    cue.extend_from_slice(&gc);
    readout
        .decode(&cue, top)
        .into_iter()
        .filter_map(|(op, conf)| apply_op(lemma, &op).map(|w| (w, conf)))
        .collect()
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = env::args().collect();
    let cmd = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    match cmd {
        "teach" => {
            let limit: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(652_477);
            let raw = fs::read_to_string(CORPUS)?;
            let facts: Vec<Fact> = raw.lines().filter_map(Fact::parse).take(limit).collect();
            println!("teaching ava {} facts...", facts.len());

            let mut fac = Factored::new(N_ID, N_GRAM, K_ID, PER_TAG);
            let mut readout = OpReadout::default();
            let mut brain = v4::Brain::new(fac.n_total, Params::default());

            let t = Instant::now();
            // passes 1-2: let the competition settle
            for _ in 0..2 {
                for f in &facts {
                    fac.learn_code(&Fields { lemma: &f.lemma, form: Some(&f.form), tags: &f.tags });
                }
            }
            // pass 3: allocator FROZEN, encode exactly as a query will, then write.
            // competitive learning is non-stationary; writing before freezing stores
            // codes that later queries can no longer address.
            for f in &facts {
                let (idc, gc) = fac.query_code(
                    &Fields { lemma: &f.lemma, form: None, tags: &f.tags }, K_ID);
                let mut full = idc;
                full.extend_from_slice(&gc);
                brain.learn_bulk(&full, 1);
                readout.learn(&full, &edit_op(&f.lemma, &f.form));
            }
            brain.compact();
            println!("learned in {:?}", t.elapsed());

            // answers BEFORE saving, so the round trip can be checked
            let probe: Vec<&Fact> = facts.iter().step_by(facts.len() / 200).take(200).collect();
            let before: Vec<Option<String>> = probe
                .iter()
                .map(|f| ask(&fac, &readout, &f.lemma, &f.tags, 1).first().map(|(w, _)| w.clone()))
                .collect();

            fs::create_dir_all("../brain")?;
            let t = Instant::now();
            let bytes = store::save(BRAIN, &brain, &fac, &readout)?;
            println!("saved {} MB to {BRAIN} in {:?}", bytes / 1_000_000, t.elapsed());

            let t = Instant::now();
            let (_b2, fac2, read2) = store::load(BRAIN, Params::default())?;
            println!("reloaded in {:?}", t.elapsed());

            let after: Vec<Option<String>> = probe
                .iter()
                .map(|f| ask(&fac2, &read2, &f.lemma, &f.tags, 1).first().map(|(w, _)| w.clone()))
                .collect();
            let same = before.iter().zip(&after).filter(|(a, b)| a == b).count();
            println!("round trip: {same}/{} answers identical after save+load", probe.len());
            if same != probe.len() {
                println!("  ^^ MISMATCH — the reloaded brain is not the same brain");
            }
            let correct = probe.iter().zip(&after)
                .filter(|(f, a)| a.as_deref() == Some(f.form.as_str())).count();
            println!("accuracy after reload: {correct}/{} ({:.0}%)",
                probe.len(), correct as f64 / probe.len() as f64 * 100.0);
        }

        "ask" => {
            if args.len() < 4 {
                println!("usage: ask <lemma> <TAG> [TAG...]     e.g. ask eat V PST");
                return Ok(());
            }
            let lemma = args[2].to_lowercase();
            let tags: Vec<String> = args[3..].iter().map(|s| s.to_uppercase()).collect();
            let t = Instant::now();
            let (_brain, fac, readout) = store::load(BRAIN, Params::default())?;
            let load_t = t.elapsed();

            let known = fac.id.feat_id.contains_key(&format!(
                "L:{}|A:{}", lemma,
                { let mut v: Vec<&str> = tags.iter().map(|s| s.as_str()).collect(); v.sort(); v.join(",") }
            ));
            println!("(brain loaded in {load_t:?}; this exact lemma+tag combination is {})",
                if known { "one she was taught" } else { "NEW to her - she has to use the rule" });

            let ans = ask(&fac, &readout, &lemma, &tags, 3);
            println!("\n  {} + {}", lemma, tags.join(";"));
            for (i, (w, c)) in ans.iter().enumerate() {
                println!("    {}. {:<20} confidence {:.2}", i + 1, w, c);
            }
            if ans.is_empty() { println!("    (silence — no unit fired)"); }
        }

        "stats" => {
            let (brain, fac, readout) = store::load(BRAIN, Params::default())?;
            let syn: u64 = brain.units.iter().map(|u| u.out.len() as u64).sum();
            let used = brain.units.iter().filter(|u| !u.out.is_empty()).count();
            println!("neurons        : {}", brain.units.len());
            println!("  with synapses: {used} ({:.1}%)",
                used as f64 / brain.units.len() as f64 * 100.0);
            println!("synapses       : {syn}");
            println!("features known : {}", fac.id.feat_id.len());
            println!("edit operations: {}", readout.ops.len());
            let mut ops: Vec<(&String, i32)> = readout.ops.iter().enumerate()
                .map(|(i, o)| (o, readout.votes.values()
                    .filter_map(|m| m.get(&(i as u32))).sum::<i32>()))
                .collect();
            ops.sort_unstable_by_key(|&(_, c)| std::cmp::Reverse(c));
            println!("most used operations:");
            for (o, c) in ops.iter().take(8) {
                let (r, add) = o.split_once('|').unwrap_or(("?", "?"));
                println!("    delete {r}, add \"{add}\"   used {c} times");
            }
            let busiest = brain.units.iter().enumerate()
                .max_by_key(|(_, u)| u.out.len()).map(|(i, u)| (i, u.out.len()));
            if let Some((i, d)) = busiest {
                println!("busiest neuron : #{i} with {d} connections");
            }
        }

        _ => {
            println!("ava — a spiking associative memory that learns english morphology\n");
            println!("  teach <n>              learn n facts and save the brain to disk");
            println!("  ask <lemma> <TAGS...>  ask her to inflect a word");
            println!("  stats                  what is inside her head\n");
            println!("examples:");
            println!("  cargo run --release --example ava -- teach 80000");
            println!("  cargo run --release --example ava -- ask eat V PST");
            println!("  cargo run --release --example ava -- ask wug N PL");
        }
    }
    let _: HashSet<u8> = HashSet::new();
    Ok(())
}
