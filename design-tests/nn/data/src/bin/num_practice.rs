//! TESTING / DESIGN ONLY — num_practice: write child-style number practice text (v39, v40's version 2 or v41's version 3), one doc per line.
//!
//!   num_practice [--v2 | --v3] [--seed 3 (v2: 4, v3: 5)] --docs N --out FILE
//!
//! Without --v2/--v3: v39's generator, its exclusion the keys of items(Dev, 400) ∪ items(Test, 400)
//! (unchanged; seed 3 / 300,000 docs is data/num/practice.txt byte for byte). With --v2: the
//! version-2 generator (adds comparisons that never say "is more/less than"), its exclusion
//! items(Dev, 400) ∪ items(Test, 400) ∪ items(Test2, 400). With --v3 (v41): version 3 (adds
//! comparisons in both directions, the test's wording included, and ordered lists), its exclusion
//! items(s, 400) for s in Dev, Test, Test2 and Test3. All exclusions are generated in memory; no
//! sealed file is read. Prints docs, bytes, sha256 and seconds.

use ava_data::input::parse_count;
use ava_data::num_practice::{
    exclusion_keys, exclusion_keys_v40, exclusion_keys_v41, practice_docs, practice_docs_v2, practice_docs_v3, DEFAULT_SEED,
    DEFAULT_SEED_V2, DEFAULT_SEED_V3, EXCLUDE_N_PER_KIND,
};
use ava_data::sha256::sha256_file;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: num_practice [--v2 | --v3] [--seed 3 (v2: 4, v3: 5)] --docs N --out FILE";

fn parse_args() -> Result<(u8, u64, usize, PathBuf), String> {
    let (mut docs, mut out, mut seed) = (None, None, None);
    let mut v = 1u8;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |flag: &str| args.next().ok_or_else(|| format!("{flag} needs a value"));
        match a.as_str() {
            "--v2" => v = 2,
            "--v3" => v = 3,
            "--seed" => seed = Some(parse_count(&val("--seed")?)?),
            "--docs" => docs = Some(parse_count(&val("--docs")?)? as usize),
            "--out" => out = Some(PathBuf::from(val("--out")?)),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            s => return Err(format!("unknown argument {s}")),
        }
    }
    let seed = seed.unwrap_or([DEFAULT_SEED, DEFAULT_SEED_V2, DEFAULT_SEED_V3][v as usize - 1]);
    Ok((v, seed, docs.ok_or("--docs is required")?, out.ok_or("--out is required")?))
}

fn main() {
    let (v, seed, n, out) = match parse_args() {
        Ok(x) => x,
        Err(e) => {
            eprintln!("error: {e}\n{USAGE}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(v, seed, n, &out) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(v: u8, seed: u64, n: usize, out: &PathBuf) -> std::io::Result<()> {
    let t0 = Instant::now();
    let ex = match v {
        1 => exclusion_keys(EXCLUDE_N_PER_KIND),
        2 => exclusion_keys_v40(EXCLUDE_N_PER_KIND),
        _ => exclusion_keys_v41(EXCLUDE_N_PER_KIND),
    };
    let t_ex = t0.elapsed().as_secs_f64();
    let docs = match v {
        1 => practice_docs(seed, n, &ex),
        2 => practice_docs_v2(seed, n, &ex),
        _ => practice_docs_v3(seed, n, &ex),
    };
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let mut w = BufWriter::new(std::fs::File::create(out)?);
    let mut bytes = 0u64;
    for d in &docs {
        writeln!(w, "{d}")?;
        bytes += d.len() as u64 + 1;
    }
    w.flush()?;
    drop(w);
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "{} seed {seed}: {} docs, {bytes} bytes -> {} ({} test keys excluded, built in {t_ex:.2}s); {secs:.3}s",
        ["version 1 (v39)", "version 2 (v40)", "version 3 (v41)"][v as usize - 1],
        docs.len(),
        out.display(),
        ex.len()
    );
    println!("sha256 {}", sha256_file(out)?);
    Ok(())
}
