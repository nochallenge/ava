// TESTING / DESIGN ONLY — probe_states (A13): linear probes on every block's state of a saved Ava 0.4 model.
//
// probe_states --model FILE --val DIR [--tokens 100000] [--ctx 256] [--threads 8] [--by-source DIR] [--sources a,b]
//              [--per-source 16384] [--epochs 20] [--out DIR] [--bpe FILE] [--seed 1]
//
// CPU only. Logistic regression trained on the first half of the windows and scored on the second, for labels made
// from the text alone (next token a digit, inside a number, end of sentence, inside quotes) and, with --by-source, the
// source of the text. Prints and writes report.md and probes.tsv. See src/sae/run.rs.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = ava_gpu::sae::run::probe_states(&args) {
        eprintln!("probe_states: {e}");
        std::process::exit(1);
    }
}
