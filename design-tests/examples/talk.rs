// talk to ava. loads the brain ONCE, then keeps it in memory so every question is
// a ~0.1 ms recall instead of a 190 ms file load.
//
//   cargo run --release --example talk
//
// then just type:
//   eat past
//   wug plural
//   child plural
//   :stats
use design_tests::store;
use design_tests::v4::{self, Params};
use design_tests::v6::*;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::time::Instant;

const BRAIN: &str = "../brain/ava.brain";
const K_ID: usize = 10;

/// plain english -> unimorph tags, so you can say "past" instead of "V;PST"
fn parse_tags(words: &[&str]) -> Option<Vec<String>> {
    let mut tags: Vec<String> = Vec::new();
    for w in words {
        let t: &[&str] = match w.to_lowercase().as_str() {
            "past" | "ed" => &["V", "PST"],
            "participle" | "pastparticiple" => &["V", "V.PTCP", "PST"],
            "ing" | "gerund" | "present-participle" => &["V", "V.PTCP", "PRS"],
            "plural" | "plurals" => &["N", "PL"],
            "singular" => &["N", "SG"],
            "he" | "she" | "it" | "3rd" | "third" => &["V", "PRS", "3", "SG"],
            other => {
                // allow raw unimorph tags too: eat V PST
                if other.chars().all(|c| c.is_ascii_alphabetic() || c == '.') {
                    tags.push(other.to_uppercase());
                    continue;
                }
                return None;
            }
        };
        for x in t { tags.push(x.to_string()); }
    }
    tags.sort();
    tags.dedup();
    if tags.is_empty() { None } else { Some(tags) }
}

fn answer(fac: &Factored, ro: &OpReadout, lemma: &str, tags: &[String], top: usize)
    -> Vec<(String, f32)>
{
    let f = Fields { lemma, form: None, tags };
    let (idc, gc) = fac.query_code(&f, K_ID);
    let mut cue = idc;
    cue.extend_from_slice(&gc);
    ro.decode(&cue, top)
        .into_iter()
        .filter_map(|(op, c)| apply_op(lemma, &op).map(|w| (w, c)))
        .collect()
}

fn main() -> rustyline::Result<()> {
    println!("loading ava...");
    let t = Instant::now();
    let (brain, fac, ro) = match store::load(BRAIN, Params::default()) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("could not load {BRAIN}: {e}");
            eprintln!("teach her first:  cargo run --release --example ava -- teach 80000");
            return Ok(());
        }
    };
    let syn: u64 = brain.units.iter().map(|u| u.out.len() as u64).sum();
    println!("ava is awake. {} neurons, {} synapses, {} features, loaded in {:?}\n",
        brain.units.len(), syn, fac.id.feat_id.len(), t.elapsed());
    println!("ask her things like:   eat past     child plural     wug plural");
    println!("or use raw tags:       eat V PST");
    println!(":stats for what is in her head, :q to leave\n");

    let mut rl = DefaultEditor::new()?;
    loop {
        match rl.readline("you> ") {
            Ok(line) => {
                let line = line.trim().to_string();
                if line.is_empty() { continue; }
                rl.add_history_entry(line.as_str())?;
                if line == ":q" || line == ":quit" { break; }
                if line == ":stats" {
                    let used = brain.units.iter().filter(|u| !u.out.is_empty()).count();
                    println!("  {} neurons, {used} of them wired, {syn} synapses",
                        brain.units.len());
                    println!("  {} features known, {} edit operations learned",
                        fac.id.feat_id.len(), ro.ops.len());
                    continue;
                }

                let words: Vec<&str> = line.split_whitespace().collect();
                if words.len() < 2 {
                    println!("ava> give me a word and what you want: `eat past`");
                    continue;
                }
                let lemma = words[0].to_lowercase();
                let Some(tags) = parse_tags(&words[1..]) else {
                    println!("ava> i do not know that form. try: past, plural, ing, he");
                    continue;
                };

                let key = {
                    let mut v: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
                    v.sort();
                    format!("L:{}|A:{}", lemma, v.join(","))
                };
                let taught = fac.id.feat_id.contains_key(&key);

                let t = Instant::now();
                let ans = answer(&fac, &ro, &lemma, &tags, 3);
                let dt = t.elapsed();

                if ans.is_empty() {
                    println!("ava> (silence)");
                    continue;
                }
                println!("ava> {}   [{}, {:.2} confidence, {:?}]",
                    ans[0].0,
                    if taught { "remembered" } else { "worked out from the rule" },
                    ans[0].1, dt);
                if ans.len() > 1 {
                    let alts: Vec<String> = ans[1..].iter()
                        .map(|(w, c)| format!("{w} ({c:.1})")).collect();
                    println!("      also considered: {}", alts.join(", "));
                }
            }
            Err(ReadlineError::Interrupted) | Err(ReadlineError::Eof) => break,
            Err(e) => { eprintln!("{e}"); break; }
        }
    }
    println!("bye");
    Ok(())
}
