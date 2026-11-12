// TESTING / DESIGN ONLY
//! v42_maths: C11's worked word problems into a docs file, each one re-checked by our own arithmetic
//! (see the `maths` module; nothing downloaded is run).
//!
//!   v42_maths [--in data/downloads/gsm8k/train.jsonl] [--out data/nn/docs/gsm8k_train.docs]
//!
//! Paths are relative to the repo root. Files whose path names a test set are refused.

use ava_data::docs::DocsWriter;
use ava_data::maths::{gsm8k_doc, Reject};
use std::io;
use std::path::PathBuf;
use std::time::Instant;

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned()
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let inp = PathBuf::from(arg(&args, "--in").unwrap_or("data/downloads/gsm8k/train.jsonl".into()));
    let out = PathBuf::from(arg(&args, "--out").unwrap_or("data/nn/docs/gsm8k_train.docs".into()));
    let low = inp.to_string_lossy().to_lowercase();
    if ["sealed", "test", "spent", "heldout"].iter().any(|w| low.contains(w)) {
        return Err(io::Error::other(format!("refused: {} looks like a test set", inp.display())));
    }
    let t = Instant::now();
    let text = std::fs::read_to_string(&inp)?;
    let mut w = DocsWriter::create(&out)?;
    let (mut kept, mut notes) = (0usize, 0usize);
    let mut rejected = [0usize; 4];
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        match gsm8k_doc(line) {
            Ok((d, n)) => {
                w.write_doc(&d)?;
                kept += 1;
                notes += n;
            }
            Err(r) => rejected[r as usize] += 1,
        }
    }
    let (_, bytes) = w.finish()?;
    let names = [Reject::NotJson, Reject::NoFinal, Reject::BadNote, Reject::WrongNote];
    let rej: Vec<String> = names.iter().zip(rejected).map(|(n, c)| format!("{n:?} {c}")).collect();
    println!(
        "v42_maths: {} -> {}: {kept} problems kept ({notes} calculator notes re-computed, all matching), left out: {}; {bytes} bytes in {:.2}s",
        inp.display(),
        out.display(),
        rej.join(", "),
        t.elapsed().as_secs_f64()
    );
    Ok(())
}
