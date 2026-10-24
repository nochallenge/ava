// Does factoring the code fix the tension?
//   (a) do same-tag facts now SHARE units, without unrelated facts sharing any?
//   (b) can she inflect a word whose past tense she has never seen?
//   (c) the real Berko wug test: a word that is not in the corpus at all.
use design_tests::living::Fact;
use design_tests::v6::*;
use std::collections::HashSet;
use std::fs;

const PATH: &str = "../data/unimorph-eng.tsv";
const N_ID: u32 = 200_000;
const N_GRAM: u32 = 2_000;
const K_ID: usize = 10;
const PER_TAG: usize = 3;

fn fields<'a>(f: &'a Fact, with_form: bool) -> Fields<'a> {
    Fields { lemma: &f.lemma, form: if with_form { Some(&f.form) } else { None }, tags: &f.tags }
}

fn main() {
    let raw = fs::read_to_string(PATH).expect("run from design-tests/");
    let all: Vec<Fact> = raw.lines().filter_map(Fact::parse).take(80_000).collect();

    let lemmas: HashSet<&str> = all.iter().map(|f| f.lemma.as_str()).collect();
    let mut held: HashSet<String> = HashSet::new();
    for (i, l) in lemmas.iter().enumerate() { if i % 10 == 0 { held.insert((*l).to_string()); } }
    let is_held = |f: &Fact| held.contains(&f.lemma) && f.tags.iter().any(|t| t == "PST");

    let train: Vec<&Fact> = all.iter().filter(|f| !is_held(f)).collect();
    let test: Vec<&Fact> = all.iter().filter(|f| is_held(f)).collect();
    println!("train {} facts, held out {} V;PST facts", train.len(), test.len());
    println!("held-out lemmas appear in OTHER tenses; only their past tense is hidden.\n");

    let mut fac = Factored::new(N_ID, N_GRAM, K_ID, PER_TAG);
    let mut readout = OpReadout::default();
    let mut codes: Vec<(Vec<NeuronId>, Vec<NeuronId>)> = Vec::with_capacity(train.len());

    // pass 1: let the competition settle
    for f in &train { fac.learn_code(&fields(f, true)); }
    // pass 2: the codes we keep, and teach the readout the edit operation
    for f in &train {
        let (idc, gc) = fac.learn_code(&fields(f, true));
        let op = edit_op(&f.lemma, &f.form);
        let mut full = idc.clone();
        full.extend_from_slice(&gc);
        readout.learn(&full, &op);
        codes.push((idc, gc));
    }
    println!("learned. {} distinct edit operations, {} identity features",
        readout.ops.len(), fac.id.feat_id.len());

    // ---- (a) does the code now carry grammatical category? ----
    println!("\n=== overlap structure (identity part | grammar part) ===");
    let pick = |pred: &dyn Fn(&Fact) -> bool| -> Vec<usize> {
        train.iter().enumerate().filter(|(_, f)| pred(f)).map(|(i, _)| i).take(400).collect()
    };
    let pst = pick(&|f: &Fact| f.tags.iter().any(|t| t == "PST"));
    let pl = pick(&|f: &Fact| f.tags.iter().any(|t| t == "PL"));
    let mixed: Vec<usize> = (0..400).map(|i| i * 173 % train.len()).collect();

    for (label, idx) in [("both PST", &pst), ("both PL", &pl), ("unrelated", &mixed)] {
        let (mut sid, mut sg, mut n) = (0u64, 0u64, 0u64);
        for a in 0..idx.len() {
            for b in a + 1..idx.len().min(a + 25) {
                let (i, j) = (idx[a], idx[b]);
                if train[i].lemma == train[j].lemma { continue; }
                sid += overlap(&codes[i].0, &codes[j].0) as u64;
                sg += overlap(&codes[i].1, &codes[j].1) as u64;
                n += 1;
            }
        }
        println!("  {:<10} identity overlap {:.3}   grammar overlap {:.3}",
            label, sid as f64 / n as f64, sg as f64 / n as f64);
    }
    println!("  (different lemmas only. identity should be ~0 everywhere;");
    println!("   grammar should be HIGH for same-tag and ~0 for unrelated.)");

    // ---- (b) held-out past tenses ----
    let answer = |f: &Fact| -> Option<String> {
        let fl = fields(f, false);
        let (idc, gc) = fac.query_code(&fl, K_ID);
        let mut cue = idc;
        cue.extend_from_slice(&gc);
        let ops = readout.decode(&cue, 1);
        ops.first().and_then(|(op, _)| apply_op(&f.lemma, op))
    };

    let mut seen_ok = 0;
    let sample: Vec<usize> = (0..1500).map(|i| i * 53 % train.len()).collect();
    for &i in &sample {
        if answer(train[i]).as_deref() == Some(train[i].form.as_str()) { seen_ok += 1; }
    }
    let mut held_ok = 0;
    for f in &test {
        if answer(f).as_deref() == Some(f.form.as_str()) { held_ok += 1; }
    }
    println!("\n=== accuracy ===");
    println!("  facts it WAS taught       : {:.1}%", seen_ok as f64 / sample.len() as f64 * 100.0);
    println!("  facts it was NEVER taught : {:.1}%   <-- v5 scored 0.0% here",
        held_ok as f64 / test.len() as f64 * 100.0);

    println!("\n  examples from the held-out set:");
    for f in test.iter().take(8) {
        let got = answer(f).unwrap_or("(silence)".into());
        println!("    {:>16} + PST -> {:<18} {}", f.lemma, got,
            if got == f.form { "CORRECT".to_string() } else { format!("wanted {}", f.form) });
    }

    // ---- (c) the actual wug test: words that are not in the corpus at all ----
    println!("\n=== the wug test: nonce words, never seen in any form ===");
    let known: HashSet<&str> = all.iter().map(|f| f.lemma.as_str()).collect();
    let nonce = ["wug", "blicket", "dax", "tove", "florp", "zib", "gorp", "spad"];
    for w in nonce {
        if known.contains(w) { println!("    {w}: actually in the corpus, skipping"); continue; }
        for (tags, label) in [(vec!["N".to_string(), "PL".to_string()], "plural"),
                              (vec!["V".to_string(), "PST".to_string()], "past")] {
            let fl = Fields { lemma: w, form: None, tags: &tags };
            let (idc, gc) = fac.query_code(&fl, K_ID);
            let mut cue = idc;
            cue.extend_from_slice(&gc);
            let ops = readout.decode(&cue, 2);
            let best = ops.first().map(|(o, s)| (apply_op(w, o).unwrap_or("?".into()), *s));
            match best {
                Some((word, conf)) => println!("    {:>9} {:<7} -> {:<14} (confidence {:.2})", w, label, word, conf),
                None => println!("    {:>9} {:<7} -> (silence)", w, label),
            }
        }
    }
}
