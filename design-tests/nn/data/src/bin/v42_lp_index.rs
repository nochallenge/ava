// TESTING / DESIGN ONLY
//! v42_lp_index: what B6's learning-progress data choice would see in a shard set (CPU only, writes nothing).
//!
//!   v42_lp_index --train DIR [--ctx 256] [--probe 8]
//!
//! Prints each class (a source holding more than half of a window, or "mixed"), its windows, share and
//! purity, whether it is probed, and the time the index took to build.

use ava_data::lp_mix::{read_tags, LpCfg, LpMix, SourceIndex};
use ava_data::shards::ShardSet;
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("v42_lp_index: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (mut train, mut ctx, mut probe) = (None::<PathBuf>, 256usize, 8usize);
    let mut i = 0;
    while i < a.len() {
        let v = a.get(i + 1).ok_or_else(|| format!("{} needs a value", a[i]))?;
        match a[i].as_str() {
            "--train" => train = Some(v.into()),
            "--ctx" => ctx = v.parse().map_err(|_| format!("bad --ctx {v:?}"))?,
            "--probe" => probe = v.parse().map_err(|_| format!("bad --probe {v:?}"))?,
            k => return Err(format!("unknown option {k}")),
        }
        i += 2;
    }
    let train = train.ok_or("--train DIR is required")?;
    let t0 = Instant::now();
    let set = ShardSet::open(&train).map_err(|e| e.to_string())?;
    let (names, docs) = read_tags(&train.join("tags.tsv")).map_err(|e| e.to_string())?;
    let t1 = Instant::now();
    let ix = SourceIndex::build(&set, &names, &docs, ctx);
    let t2 = Instant::now();
    let m = LpMix::new(&ix, LpCfg::parse(&format!("0,250,4,{probe}"))?, 1);
    println!(
        "{}: {} docs, {} windows of ctx {ctx}; tags read {:.2} s, index {:.2} s, mix {:.2} s",
        train.display(),
        docs.len(),
        ix.total(),
        (t1 - t0).as_secs_f64(),
        (t2 - t1).as_secs_f64(),
        t2.elapsed().as_secs_f64()
    );
    println!("{:16} {:>9} {:>8} {:>7} probed", "class", "windows", "share", "purity");
    for (c, n) in ix.names.iter().enumerate() {
        println!("{n:16} {:>9} {:>7.3}% {:>7.3} {}", ix.windows[c].len(), 100.0 * m.base[c], ix.purity[c], if m.probes[c].is_empty() { "no" } else { "yes" });
    }
    Ok(())
}
