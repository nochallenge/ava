// TESTING / DESIGN ONLY.
//
// Are the neurons doing anything, or is the encoder doing all the work?
//
//   cargo run --release --example spiking
//
// A grep found that every path from v6 onward — ava, exam, talk, dispersal, speak —
// goes  cue -> readout.decode(cue)  and never calls brain.recall(). The last code that
// actually spiked was v5. So the 1,988 MB brain file holds 292 million Hebbian
// synapses that no question reads, and the "spiking neural network" has been a stored
// artifact rather than a working part for three versions.
//
// That is either a serious problem or a dead weight worth deleting, and which one it
// is depends on a number nobody has measured: does letting the cue COMPLETE through
// the synapses before voting change the answer?
//
//   A. decode(cue)                     — what she does today
//   B. decode(cue + recall(cue))       — let the pattern complete first, then vote
//
// If B is better, the synapses earn their place and recall goes back in the pipeline.
// If B is the same, the honest thing is to say so and stop calling it a spiking network.
use design_tests::living::Fact;
use design_tests::store;
use design_tests::v4::{self, Params};
use design_tests::v6::*;
use std::fs;
use std::time::Instant;

const CORPUS: &str = "../data/unimorph-eng.tsv";
const BRAIN: &str = "../brain/ava.brain";
const K_ID: usize = 10;

fn main() -> std::io::Result<()> {
    // A SPARSE VOLLEY. The default max_spikes is 4096, so one recall can fire 4096
    // neurons each emitting up to `max_degree` events — ~16.7M events per question.
    // Pruning the degree fixed only half of that. A cortical volley is sparse: a few
    // dozen cells, not thousands. The code is 28 units, so a volley that recovers it
    // needs tens of spikes, and anything past that is the avalanche v4 already warned
    // about. Both caps together are what make completion affordable.
    let mut p = Params::default();
    p.max_spikes = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(128);
    // 4th arg: 1 = count distinct presynaptic contributors instead of summing raw
    // strength, putting the threshold and the drive in the same currency.
    p.count_drive = std::env::args().nth(4).and_then(|s| s.parse::<u32>().ok()).unwrap_or(0) != 0;
    println!("drive = {}", if p.count_drive { "CONTRIBUTOR COUNT" } else { "summed strength" });
    let t = Instant::now();
    let (mut brain, fac, ro) = store::load(BRAIN, p)?;
    println!("loaded {} neurons, {} synapses in {:?}",
        brain.units.len(), brain.total_synapses(), t.elapsed());

    // PRUNE. Bound every unit's out-degree, keeping the strongest synapses, and see
    // whether recall becomes usable — and whether the answers change.
    let max_deg: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    if max_deg > 0 {
        let biggest = brain.units.iter().map(|u| u.out.len()).max().unwrap_or(0);
        let t = Instant::now();
        let cut = brain.compact_pruned(max_deg);
        println!("pruned to max degree {max_deg} in {:?}: dropped {cut} synapses ({:.1}%), busiest neuron {} -> {}",
            t.elapsed(), cut as f64 / (cut + brain.total_synapses()) as f64 * 100.0,
            biggest, brain.units.iter().map(|u| u.out.len()).max().unwrap_or(0));
        println!("synapses now {}", brain.total_synapses());
    }

    let raw = fs::read_to_string(CORPUS)?;
    let all: Vec<Fact> = raw.lines().filter_map(Fact::parse).collect();
    // 2000 was far too many. A grammar hub has up to 1,225,703 outgoing synapses, and
    // recall() seeds by emitting EVERY outgoing spike of every cue unit, so a single
    // query pushes millions of events into the heap. That cost is itself the finding:
    // measure a handful and report the per-query price honestly.
    let n_probe: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let step = (all.len() / n_probe).max(1);
    let probe: Vec<&Fact> = all.iter().step_by(step).take(n_probe).collect();
    println!("asking {} facts both ways\n", probe.len());

    let cue_of = |f: &Fact| -> Vec<NeuronId> {
        let (idc, gc) = fac.query_code(
            &Fields { lemma: &f.lemma, form: None, tags: &f.tags }, K_ID);
        let mut c = idc;
        c.extend_from_slice(&gc);
        c
    };

    // ---------- A: decode straight from the cue (today) ----------
    let t = Instant::now();
    let mut ok_a = 0usize;
    for f in &probe {
        let cue = cue_of(f);
        if ro.decode(&cue, 1).first()
            .and_then(|(op, _)| apply_op(&f.lemma, op))
            .as_deref() == Some(f.form.as_str()) { ok_a += 1; }
    }
    let ta = t.elapsed();

    // ---------- B: let it spike first, then decode the completed pattern ----------
    let t = Instant::now();
    let (mut ok_b, mut spikes, mut recovered, mut extra) = (0usize, 0u64, 0u64, 0u64);
    for f in &probe {
        let cue = cue_of(f);
        let fired: Vec<NeuronId> = brain.recall(&cue, 1).into_iter().map(|(_, u)| u).collect();
        spikes += fired.len() as u64;
        // how much of what fired was ALREADY in the cue vs genuinely completed?
        for u in &fired {
            if cue.contains(u) { recovered += 1 } else { extra += 1 }
        }
        let mut full = cue.clone();
        for u in fired { if !full.contains(&u) { full.push(u); } }
        if ro.decode(&full, 1).first()
            .and_then(|(op, _)| apply_op(&f.lemma, op))
            .as_deref() == Some(f.form.as_str()) { ok_b += 1; }
    }
    let tb = t.elapsed();

    let pa = ok_a as f64 / probe.len() as f64 * 100.0;
    let pb = ok_b as f64 / probe.len() as f64 * 100.0;
    println!("{:<44} {:>8} {:>14}", "", "correct", "time/answer");
    println!("{:<44} {:>7.1}% {:>14?}", "A  decode(cue)            — today", pa, ta / probe.len() as u32);
    println!("{:<44} {:>7.1}% {:>14?}", "B  decode(cue + recall)   — spiking", pb, tb / probe.len() as u32);
    println!("\naverage spikes per recall : {:.1}", spikes as f64 / probe.len() as f64);
    println!("  of those, already in the cue : {:.1}", recovered as f64 / probe.len() as f64);
    println!("  genuinely completed (new)    : {:.1}", extra as f64 / probe.len() as f64);

    println!("\nverdict:");
    if pb > pa + 0.5 {
        println!("  the synapses earn their place — completion adds {:.1} points.", pb - pa);
    } else if pb < pa - 0.5 {
        println!("  completion HURTS by {:.1} points. the extra units it recruits are noise.", pa - pb);
    } else {
        println!("  no difference. 292M synapses and {:.0}x the time buy nothing at this",
            tb.as_secs_f64() / ta.as_secs_f64());
        println!("  cue size, because the cue ALREADY contains the whole pattern — there is");
        println!("  nothing left to complete. pattern completion needs a PARTIAL cue.");
    }

    // ---------- C: so give it a partial cue, which is the case completion is FOR ----------
    println!("\n--- and with a partial cue, which is what completion is actually for ---");
    println!("{:>10} {:>12} {:>12} {:>10}", "cue kept", "A decode", "B spiking", "spikes");
    for frac in [0.75f32, 0.5, 0.35] {
        let (mut a, mut b, mut sp) = (0usize, 0usize, 0u64);
        for f in &probe {
            let cue = cue_of(f);
            let keep = ((cue.len() as f32 * frac) as usize).max(2);
            let part = &cue[..keep];
            if ro.decode(part, 1).first()
                .and_then(|(op, _)| apply_op(&f.lemma, op))
                .as_deref() == Some(f.form.as_str()) { a += 1; }
            let fired: Vec<NeuronId> = brain.recall(part, 1).into_iter().map(|(_, u)| u).collect();
            sp += fired.len() as u64;
            let mut full = part.to_vec();
            for u in fired { if !full.contains(&u) { full.push(u); } }
            if ro.decode(&full, 1).first()
                .and_then(|(op, _)| apply_op(&f.lemma, op))
                .as_deref() == Some(f.form.as_str()) { b += 1; }
        }
        println!("{:>9.0}% {:>11.1}% {:>11.1}% {:>10.1}",
            frac * 100.0,
            a as f64 / probe.len() as f64 * 100.0,
            b as f64 / probe.len() as f64 * 100.0,
            sp as f64 / probe.len() as f64);
    }
    Ok(())
}
