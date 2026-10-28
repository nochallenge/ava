// TESTING / DESIGN ONLY — sleep_v41: write v41's sealed sleep test (150 fresh made-up people and questions about them).
//
//   sleep_v41 [--n 150] [--q-seed 3] [--out-dir data/v41/sealed]
//
// The people: sleep_facts' generator, seed 41 (`sleep_facts::V41_SEED`), with every name word and town of v40's
// population (seed 40: 150 dev + 150 sealed people, regenerated in memory, no file read) counted as used, so the
// new people share no name word and no town with v40's dev or sealed sets. Recall practice (ava_data::recall) can
// produce no word sleep's grammar can, so it shares none either. Formats exactly as v40's:
//   OUT/sleep_test_facts.txt   one diary doc per line: "{Name} was born in {year} in {Town}. {Name} worked as a/an {job}."
//   OUT/sleep_test_q.tsv       3 questions per person (year, town, job); id \t kind \t kind \t right \t prompt \t c0..c3
// SEALED: prints counts, the overlap checks and sha256 only. (v40's data/v40/sleep_dev_* stay the dev set.)

use ava_data::input::parse_count;
use ava_data::recall::sleep_producible;
use ava_data::sha256::sha256_file;
use ava_data::sleep_facts::{diary, people_excluding, q_line, questions, split_people, words_of, DEFAULT_SEED, KINDS, V41_SEED};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "usage: sleep_v41 [--n 150] [--q-seed 3] [--out-dir data/v41/sealed]";

fn parse_args() -> Result<(usize, u64, PathBuf), String> {
    let (mut n, mut q_seed, mut out) = (150usize, 3u64, PathBuf::from("data/v41/sealed"));
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |flag: &str| args.next().ok_or_else(|| format!("{flag} needs a value"));
        match a.as_str() {
            "--n" => n = parse_count(&val("--n")?)? as usize,
            "--q-seed" => q_seed = parse_count(&val("--q-seed")?)?,
            "--out-dir" => out = PathBuf::from(val("--out-dir")?),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            s => return Err(format!("unknown argument {s}")),
        }
    }
    Ok((n, q_seed, out))
}

fn main() {
    let (n, q_seed, out) = match parse_args() {
        Ok(x) => x,
        Err(e) => {
            eprintln!("error: {e}\n{USAGE}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(n, q_seed, &out) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn write_lines(path: &Path, lines: impl Iterator<Item = String>) -> std::io::Result<usize> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let mut w = BufWriter::new(std::fs::File::create(path)?);
    let mut k = 0;
    for l in lines {
        writeln!(w, "{l}")?;
        k += 1;
    }
    w.flush()?;
    Ok(k)
}

fn run(n: usize, q_seed: u64, out: &Path) -> std::io::Result<()> {
    let t0 = Instant::now();
    let (dev, sealed40) = split_people(DEFAULT_SEED, 150, 150);
    let old = words_of(&[dev, sealed40].concat());
    let people = people_excluding(V41_SEED, n, &old);
    let new = words_of(&people);
    let shared = new.intersection(&old).count();
    let recall_style = new.iter().filter(|w| !sleep_producible(w)).count();
    if shared != 0 || new.len() != 3 * n || recall_style != 0 {
        return Err(std::io::Error::other(format!("overlap check failed: {shared} shared, {} distinct, {recall_style} not sleep-style", new.len())));
    }
    let facts = out.join("sleep_test_facts.txt");
    let qfile = out.join("sleep_test_q.tsv");
    let docs = write_lines(&facts, people.iter().map(diary))?;
    let qs = questions(&people, q_seed);
    let nq = write_lines(&qfile, qs.iter().map(q_line))?;
    let per: Vec<String> = KINDS.iter().map(|k| format!("{k} {}", qs.iter().filter(|q| q.kind == *k).count())).collect();
    println!(
        "v41 sealed sleep test: {} people (seed {V41_SEED}), {docs} diary docs, {nq} questions ({}), question seed {q_seed}",
        people.len(),
        per.join(", ")
    );
    println!(
        "  {} distinct name words + towns; 0 shared with v40's 300 people ({} words); all of sleep's style (recall cannot make them)",
        new.len(),
        old.len()
    );
    for p in [&facts, &qfile] {
        println!("  sha256 {}  {} bytes  {}", sha256_file(p)?, std::fs::metadata(p)?.len(), p.display());
    }
    println!("time: {:.3}s", t0.elapsed().as_secs_f64());
    Ok(())
}
