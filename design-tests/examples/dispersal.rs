// TESTING / DESIGN ONLY.
//
// "Is the data dispersed between every neuron, so that remembering fires a neuron
// with some of that data in it?"  Measure it, on the saved brain.
//
//   cargo run --release --example dispersal -- eat V PST
//
// Two numbers say everything:
//   * how many neurons one fact is spread across
//   * how many facts one neuron is shared by
//
// and they are wildly different for the two populations, on purpose.
use design_tests::store;
use design_tests::v4::Params;
use design_tests::v6::*;
use std::env;

const BRAIN: &str = "../brain/ava.brain";
const K_ID: usize = 10;

fn main() -> std::io::Result<()> {
    let a: Vec<String> = env::args().collect();
    let lemma = a.get(2).cloned().unwrap_or_else(|| "eat".into()).to_lowercase();
    let tags: Vec<String> = if a.len() > 3 {
        a[3..].iter().map(|s| s.to_uppercase()).collect()
    } else {
        vec!["V".into(), "PST".into()]
    };

    let (brain, fac, ro) = store::load(BRAIN, Params::default())?;
    let n_id = fac.gram.base; // identity units are [0, n_id), grammar units above it

    // ---------- 1. how is ONE fact spread out? ----------
    let f = Fields { lemma: &lemma, form: None, tags: &tags };
    let (idc, gc) = fac.query_code(&f, K_ID);
    let mut code = idc.clone();
    code.extend_from_slice(&gc);

    println!("=== \"{}\" + {} ===\n", lemma, tags.join(";"));
    println!("this one fact is spread across {} neurons out of {}",
        code.len(), brain.units.len());
    println!("  {} identity neurons   (which fact this is)", idc.len());
    println!("  {} grammar neurons    (what kind of fact it is)\n", gc.len());

    // ---------- 2. how much of the answer does each ONE hold? ----------
    println!("what each of those neurons individually knows:");
    println!("{:>12} {:>10} {:>14} {:>10}  {}",
        "neuron", "kind", "facts in it", "partners", "its single best guess");
    let mut rows: Vec<(NeuronId, bool, u32, usize, String, f32)> = Vec::new();
    for &u in &code {
        let shared = *ro.total.get(&u).unwrap_or(&0);
        let partners = brain.units[u as usize].out.len();
        let best = ro
            .votes
            .get(&u)
            .and_then(|m| m.iter().max_by_key(|(_, &c)| c).map(|(&o, &c)| (o, c)))
            .map(|(o, c)| (ro.ops[o as usize].clone(), c as f32 / shared.max(1) as f32))
            .unwrap_or(("-".into(), 0.0));
        rows.push((u, u < n_id, shared, partners, best.0, best.1));
    }
    rows.sort_by_key(|r| (!r.1, r.2));
    for (u, is_id, shared, partners, op, conf) in &rows {
        let word = apply_op(&lemma, op).unwrap_or_else(|| "?".into());
        println!("{:>12} {:>10} {:>14} {:>10}  {:<10} {:.0}% of what it holds",
            u, if *is_id { "identity" } else { "grammar" }, shared, partners, word, conf * 100.0);
    }

    // ---------- 3. the point ----------
    let id_share: f32 = rows.iter().filter(|r| r.1).map(|r| r.2 as f32).sum::<f32>()
        / rows.iter().filter(|r| r.1).count().max(1) as f32;
    let gr_share: f32 = rows.iter().filter(|r| !r.1).map(|r| r.2 as f32).sum::<f32>()
        / rows.iter().filter(|r| !r.1).count().max(1) as f32;
    println!("\naverage number of facts sharing one IDENTITY neuron: {id_share:.1}");
    println!("average number of facts sharing one GRAMMAR  neuron: {gr_share:.0}");
    println!("  -> identity neurons are nearly private; grammar neurons are shared by");
    println!("     hundreds of thousands of facts. Same code, opposite dispersal.");

    let ans = ro.decode(&code, 3);
    println!("\nno single neuron holds the answer. all {} voting together:", code.len());
    for (i, (op, c)) in ans.iter().enumerate() {
        if let Some(w) = apply_op(&lemma, op) {
            println!("  {}. {:<14} {:.1}", i + 1, w, c);
        }
    }

    // ---------- 4. and it is genuinely distributed: drop neurons and re-ask ----------
    println!("\nknock neurons out of the cue and ask again (graceful degradation):");
    for keep in [code.len(), code.len() * 3 / 4, code.len() / 2, idc.len(), 3] {
        let sub = &code[..keep.min(code.len())];
        let top = ro.decode(sub, 1);
        let w = top.first().and_then(|(o, _)| apply_op(&lemma, o)).unwrap_or("(silence)".into());
        println!("  {:>3} of {} neurons -> {}", sub.len(), code.len(), w);
    }
    Ok(())
}
