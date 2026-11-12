// TESTING / DESIGN ONLY — grow_exact: A15, grow a trained model deeper and its MLPs wider so that it computes the same
// function (src/grow.rs has the maths, what is refused and why). CPU only.
//
// grow_exact --init DIR_OR_FILE --out DIR [--mlp-times C [--mlp-split first|even]] [--deeper N [--at L]]
//            [--seed 1] [--kind D] [--check-rows 2] [--check-tokens 96]
//
// --init     a model file, or a run folder holding model.safetensors (its block options from NAME.arch.txt or the
//            folder's arch.txt, as every tool loads it)
// --out      an empty folder: model.safetensors, model.arch.txt (the grown block options, when any) and grow.txt
//            (what was done, the measured logit gap and the seconds)
// --mlp-times C   every MLP hidden unit copied C times (recorded as --mlp-mult)
// --mlp-split     first (default): copy 0 keeps each unit's outgoing row, the other copies start at 0 (bit for bit on
//                 the CPU up to 512 old units); even: Net2Net's row / C (regroups the sum, ~1e-7 per MLP, which a
//                 spiking model can amplify past 1e-5 when a spike flips; the trained full-all-a16 stays at 1e-6)
// --deeper N      N new blocks after block --at L (default: the last), each the identity at first
// --seed S        the new blocks' fresh init
// The grown model's logits on --check-rows x --check-tokens fixed tokens must equal the source's within 1e-5 of the
// largest logit, in memory (else nothing is written) and after the written file is loaded back.
// The optimizer's state is not carried: train the grown model as a new run (e.g. grow_gpu --phase grow --init DIR/model.safetensors).

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match ava_gpu::grow::run(&args) {
        Ok(r) => {
            let c = &r.cfg;
            for n in &r.notes {
                println!("{n}");
            }
            println!("grown: {} d={} layers={} hidden={} kwta k={}{}", c.kind.name(), c.d, c.layers, c.hidden(), c.kwta_k(), c.arch.signature());
            println!("weights {} -> {}", r.params.0, r.params.1);
            println!(
                "logit gap {:.3e} (relative {:.3e}) in memory, {:.3e} (relative {:.3e}) loaded back from {}",
                r.gap_mem.0,
                r.gap_mem.1,
                r.gap_file.0,
                r.gap_file.1,
                r.out.join("model.safetensors").display()
            );
            let s = r.secs;
            println!("seconds: load {:.2}, grow {:.2}, check {:.2}, save {:.2}, load back + check {:.2} (total {:.2})", s[0], s[1], s[2], s[3], s[4], s.iter().sum::<f64>());
            let f = c.arch.flags();
            if !f.is_empty() {
                println!("block flags of the grown model (in {}): {f}", r.out.join("model.arch.txt").display());
            }
            println!("the optimizer's state is not carried: start a new run (fresh AdamW, its own warm-up)");
        }
        Err(e) => {
            eprintln!("grow_exact: {e}");
            std::process::exit(1);
        }
    }
}
