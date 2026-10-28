// TESTING / DESIGN ONLY — sleep_facts: write v40's made-up people (the diary) and the questions about them.
//
//   sleep_facts [--seed 40] [--n 150] [--out-dir data/v40]
//
// One population of 2n people (seed): the first n are dev, the next n sealed; no name word or town is shared.
//   OUT/sleep_dev_facts.txt            one diary doc per line: "{Name} was born in {year} in {Town}. {Name} worked as a/an {job}."
//   OUT/sleep_dev_q.tsv                3 questions per person (year, town, job), question seed 1
//   OUT/sealed/sleep_test_facts.txt    the same for the sealed people (SEALED)
//   OUT/sealed/sleep_test_q.tsv        question seed 2 (SEALED)
// TSV: id \t kind \t bin(=kind) \t right \t prompt \t choice0..3 (escaped as in fact_items). Prints counts and
// sha256 only.

use ava_data::input::parse_count;
use ava_data::sha256::sha256_file;
use ava_data::sleep_facts::{diary, q_line, questions, split_people, Person, DEFAULT_SEED, KINDS};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "usage: sleep_facts [--seed 40] [--n 150] [--out-dir data/v40]";

fn parse_args() -> Result<(u64, usize, PathBuf), String> {
    let (mut seed, mut n, mut out) = (DEFAULT_SEED, 150usize, PathBuf::from("data/v40"));
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |flag: &str| args.next().ok_or_else(|| format!("{flag} needs a value"));
        match a.as_str() {
            "--seed" => seed = parse_count(&val("--seed")?)?,
            "--n" => n = parse_count(&val("--n")?)? as usize,
            "--out-dir" => out = PathBuf::from(val("--out-dir")?),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            s => return Err(format!("unknown argument {s}")),
        }
    }
    Ok((seed, n, out))
}

fn main() {
    let (seed, n, out) = match parse_args() {
        Ok(x) => x,
        Err(e) => {
            eprintln!("error: {e}\n{USAGE}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(seed, n, &out) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn write_lines(path: &Path, lines: impl Iterator<Item = String>) -> std::io::Result<usize> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let mut w = BufWriter::new(std::fs::File::create(path)?);
    let mut n = 0;
    for l in lines {
        writeln!(w, "{l}")?;
        n += 1;
    }
    w.flush()?;
    Ok(n)
}

fn run(seed: u64, n: usize, out: &Path) -> std::io::Result<()> {
    let t0 = Instant::now();
    let (dev, test) = split_people(seed, n, n);
    let parts: [(&str, &Vec<Person>, u64, PathBuf, PathBuf); 2] = [
        ("dev", &dev, 1, out.join("sleep_dev_facts.txt"), out.join("sleep_dev_q.tsv")),
        ("sealed test", &test, 2, out.join("sealed").join("sleep_test_facts.txt"), out.join("sealed").join("sleep_test_q.tsv")),
    ];
    for (name, people, qseed, facts, qfile) in parts {
        let docs = write_lines(&facts, people.iter().map(diary))?;
        let qs = questions(people, qseed);
        let nq = write_lines(&qfile, qs.iter().map(q_line))?;
        let per: Vec<String> = KINDS.iter().map(|k| format!("{k} {}", qs.iter().filter(|q| q.kind == *k).count())).collect();
        println!("{name}: {} people, {docs} diary docs, {nq} questions ({}), question seed {qseed}", people.len(), per.join(", "));
        for p in [&facts, &qfile] {
            println!("  sha256 {}  {} bytes  {}", sha256_file(p)?, std::fs::metadata(p)?.len(), p.display());
        }
    }
    println!("seed {seed}; time: {:.3}s", t0.elapsed().as_secs_f64());
    Ok(())
}
