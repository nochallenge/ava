//! TESTING / DESIGN ONLY — num_items: write the number test (dev, v39's sealed test, or v40's sealed test2) as TSV.
//!
//!   num_items --set dev|test|test2|test3 [--n-per-kind 400] --out FILE
//!
//! One line per item, no header, ids from 0:
//!   id \t kind \t bin \t right \t choice0 \t choice1 [\t choice2 \t choice3]
//! Then prints the counts per kind and bin and the file's sha256 (never an item, so the sealed
//! sets stay unseen). Creates FILE's parent directory if needed. `test2` (v40) is built clear of
//! Dev, Test and every key of v39's practice text (see `ava_data::num_items`). `test3` (v41) is built clear of
//! Dev, Test, Test2 and every key of v39's and v40's practice texts (SEALED as well).

use ava_data::input::parse_count;
use ava_data::num_items::{items, tsv_line, Set, KINDS};
use ava_data::sha256::sha256_file;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: num_items --set dev|test|test2|test3 [--n-per-kind 400] --out FILE";

fn parse_args() -> Result<(Set, usize, PathBuf), String> {
    let (mut set, mut out) = (None, None);
    let mut n = 400usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |flag: &str| args.next().ok_or_else(|| format!("{flag} needs a value"));
        match a.as_str() {
            "--set" => {
                set = Some(match val("--set")?.as_str() {
                    "dev" => Set::Dev,
                    "test" => Set::Test,
                    "test2" => Set::Test2,
                    "test3" => Set::Test3,
                    s => return Err(format!("--set must be dev, test, test2 or test3, not {s:?}")),
                })
            }
            "--n-per-kind" => n = parse_count(&val("--n-per-kind")?)? as usize,
            "--out" => out = Some(PathBuf::from(val("--out")?)),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            s => return Err(format!("unknown argument {s}")),
        }
    }
    Ok((set.ok_or("--set is required")?, n, out.ok_or("--out is required")?))
}

fn main() {
    let (set, n, out) = match parse_args() {
        Ok(x) => x,
        Err(e) => {
            eprintln!("error: {e}\n{USAGE}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(set, n, &out) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(set: Set, n: usize, out: &PathBuf) -> std::io::Result<()> {
    let t0 = Instant::now();
    let all = items(set, n);
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let mut w = BufWriter::new(std::fs::File::create(out)?);
    for it in &all {
        writeln!(w, "{}", tsv_line(it))?;
    }
    w.flush()?;
    drop(w);
    println!("{set:?} set, n_per_kind {n}: {} items -> {}", all.len(), out.display());
    for kind in KINDS {
        let of: Vec<&str> = all.iter().filter(|x| x.kind == kind).map(|x| x.bin).collect();
        let mut bins: Vec<(&str, usize)> = Vec::new();
        for b in &of {
            match bins.iter_mut().find(|x| x.0 == *b) {
                Some(x) => x.1 += 1,
                None => bins.push((b, 1)),
            }
        }
        bins.sort();
        let s: Vec<String> = bins.iter().map(|(b, c)| format!("{b} {c}")).collect();
        println!("  {kind:<14} {:>5}   {}", of.len(), s.join("  "));
    }
    println!("sha256 {}  {} bytes", sha256_file(out)?, std::fs::metadata(out)?.len());
    println!("time: {:.3}s", t0.elapsed().as_secs_f64());
    Ok(())
}
