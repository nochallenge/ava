// TESTING / DESIGN ONLY
//! v42_bios: synthetic biographies for the capacity measurement (see `bios`), and the bit count.
//!
//!   v42_bios [--out DIR] [--people N] [--exposures E] [--seed S] [--bpe FILE]
//!   v42_bios bits --people N (--acc a1,..,a6 | --loss l1,..,l6) [--params P]
//!
//! The first form writes into --out (default data/nn/v42/bios):
//!   bios_n{N}_e{E}.docs   E rounds, each one exposure of every person (one biography per document)
//!   people_n{N}.tsv       the truth: name and the six values
//!   probe_n{N}.tsv        person, attribute, number of choices, prompt, answer (6 rows per person)
//!   choices.tsv           attribute and every choice (with its leading space, as it follows a prompt)
//!   report_n{N}_e{E}.txt  sizes, tokens (encode + length marks + EOS), speed
//! Defaults: --people 10000 --exposures 100 --seed 33. A finished docs file is skipped.
//! The `bits` form prints the bits stored for the given per-attribute accuracies or losses (bits),
//! in the order birth_day, birth_month, birth_year, birth_city, job, employer.

use ava_data::bios::{bits_from_accuracy, bits_from_loss, choices, entropy_bits, people, probes, round_docs, ATTRS, DEFAULT_SEED};
use ava_data::bpe::Bpe;
use ava_data::docs::DocsWriter;
use ava_data::input::parse_count;
use ava_data::numbers::insert_marks;
use std::io;
use std::path::PathBuf;
use std::time::Instant;

const DATA: &str = ava_data::DATA_DIR;

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
}

fn floats(s: &str) -> io::Result<Vec<f64>> {
    s.split(',').map(|x| x.trim().parse::<f64>().map_err(io::Error::other)).collect()
}

fn run() -> io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let n = arg(&args, "--people").map(|v| parse_count(&v).map_err(io::Error::other)).transpose()?.unwrap_or(10_000);
    if args.first().map(|s| s.as_str()) == Some("bits") {
        let (vals, from_acc) = match (arg(&args, "--acc"), arg(&args, "--loss")) {
            (Some(a), None) => (floats(&a)?, true),
            (None, Some(l)) => (floats(&l)?, false),
            _ => return Err(io::Error::other("give exactly one of --acc or --loss (6 comma-separated values)")),
        };
        if vals.len() != ATTRS.len() {
            return Err(io::Error::other("need 6 values"));
        }
        let per: Vec<(usize, f64)> = ATTRS.iter().zip(&vals).map(|(&(_, s), &v)| (s, v)).collect();
        let bits = if from_acc { bits_from_accuracy(n, &per) } else { bits_from_loss(n, &per) };
        let max = n as f64 * entropy_bits();
        println!("people {n}: {bits:.0} bits stored of {max:.0} possible ({:.1}%)", 100.0 * bits / max);
        for (k, &(name, s)) in ATTRS.iter().enumerate() {
            let one = if from_acc { bits_from_accuracy(n, &per[k..k + 1]) } else { bits_from_loss(n, &per[k..k + 1]) };
            println!("  {name:<12} {s:>3} choices  {} {:.4}  -> {one:.0} bits", if from_acc { "acc" } else { "loss" }, vals[k]);
        }
        if let Some(p) = arg(&args, "--params") {
            let p = parse_count(&p).map_err(io::Error::other)? as f64;
            println!("bits per parameter: {:.4}", bits / p);
        }
        return Ok(());
    }

    let e = arg(&args, "--exposures").map(|v| parse_count(&v).map_err(io::Error::other)).transpose()?.unwrap_or(100);
    let seed = arg(&args, "--seed").map(|v| parse_count(&v).map_err(io::Error::other)).transpose()?.unwrap_or(DEFAULT_SEED);
    let out = arg(&args, "--out").map(PathBuf::from).unwrap_or(PathBuf::from(format!("{DATA}/nn/v42/bios")));
    let bpe = Bpe::load(arg(&args, "--bpe").unwrap_or(format!("{DATA}/nn/bpe-4096.txt")))?;
    std::fs::create_dir_all(&out)?;
    let docs_path = out.join(format!("bios_n{n}_e{e}.docs"));
    if docs_path.exists() {
        println!("{} exists (finished); skipping", docs_path.display());
        return Ok(());
    }
    let t0 = Instant::now();
    let ps = people(n as usize, seed);
    let mut truth = String::from("name\tbirth_day\tbirth_month\tbirth_year\tbirth_city\tjob\temployer\n");
    let mut probe = String::from("person\tattribute\tchoices\tprompt\tanswer\n");
    for (i, p) in ps.iter().enumerate() {
        truth.push_str(&p.name);
        for k in 0..6 {
            truth.push('\t');
            truth.push_str(&p.value(k));
        }
        truth.push('\n');
        for (k, prompt, ans) in probes(p) {
            probe.push_str(&format!("{i}\t{}\t{}\t{prompt}\t{ans}\n", ATTRS[k].0, ATTRS[k].1));
        }
    }
    std::fs::write(out.join(format!("people_n{n}.tsv")), truth)?;
    std::fs::write(out.join(format!("probe_n{n}.tsv")), probe)?;
    let mut ch = String::from("attribute\tchoice\n");
    for (k, &(name, _)) in ATTRS.iter().enumerate() {
        for c in choices(k) {
            ch.push_str(&format!("{name}\t{c}\n"));
        }
    }
    std::fs::write(out.join("choices.tsv"), ch)?;

    let mut w = DocsWriter::create(&docs_path)?;
    let mut tokens = 0u64;
    for r in 0..e {
        for d in round_docs(&ps, seed, r) {
            tokens += insert_marks(&bpe.encode_doc(&d)).len() as u64;
            w.write_doc(&d)?;
        }
    }
    let (docs, bytes) = w.finish()?;
    let secs = t0.elapsed().as_secs_f64();
    let first = round_docs(&ps[..1], seed, 0).remove(0);
    let report = format!(
        "v42_bios: people {n}  exposures {e}  seed {seed}\n\
         bits per person {:.2} (birth_day 28, birth_month 12, birth_year 100, birth_city 64, job 64, employer 32); total {:.0} bits\n\
         docs {docs} ({:.1} MB), tokens {tokens} (EOS + marks), {:.1} tokens per biography\n\
         time {secs:.1}s ({:.0}k docs/s, {:.2}M tokens/s)\n\
         example: {first}\n",
        entropy_bits(),
        n as f64 * entropy_bits(),
        bytes as f64 / 1e6,
        tokens as f64 / docs.max(1) as f64,
        docs as f64 / 1e3 / secs,
        tokens as f64 / 1e6 / secs
    );
    print!("{report}");
    std::fs::write(out.join(format!("report_n{n}_e{e}.txt")), report)?;
    Ok(())
}
