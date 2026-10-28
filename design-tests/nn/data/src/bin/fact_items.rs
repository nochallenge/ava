// TESTING / DESIGN ONLY — fact_items: write v40's find-the-fact items (a made-up code in a book, a gap, a cue) as TSV.
//
//   fact_items --set dev|test [--books DIR] [--bpe FILE] [--n-per-bin N] [--seed S] [--out FILE] [--threads 8]
//
//   dev  : --books data/pg19/validation  --n-per-bin 50   --seed 1  --out data/v40/fact_dev.tsv
//   test : --books data/pg19/test        --n-per-bin 100  --seed 2  --out data/v40/sealed/fact_test.tsv  (SEALED)
//
// Line format (see ava_data::fact_items):
//   id \t fact \t bin \t right \t prompt \t choice0 \t choice1 \t choice2 \t choice3
// with the prompt (and choices) escaped: `\\` = backslash, `\t` = tab, `\n` = newline, `\r` = CR.
// Bins g256, g1024, g4096, g12288 (tokens between the fact and the cue), equal counts. Prints counts and
// token statistics per bin and the file's sha256, never an item.

use ava_data::books::{encode_books, list_books};
use ava_data::bpe::Bpe;
use ava_data::fact_items::{fact_items, item_line, BINS};
use ava_data::input::parse_count;
use ava_data::sha256::sha256_file;
use std::collections::HashSet;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: fact_items --set dev|test [--books DIR] [--bpe FILE] [--n-per-bin N] [--seed S] [--out FILE] [--threads 8]";

struct Opts {
    books: PathBuf,
    bpe: PathBuf,
    n: usize,
    seed: u64,
    out: PathBuf,
    threads: usize,
    set: String,
}

fn parse_args() -> Result<Opts, String> {
    let mut set: Option<String> = None;
    let (mut books, mut n, mut seed, mut out) = (None, None, None, None);
    let mut bpe = PathBuf::from("data/nn/bpe-4096.txt");
    let mut threads = 8usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |flag: &str| args.next().ok_or_else(|| format!("{flag} needs a value"));
        match a.as_str() {
            "--set" => set = Some(val("--set")?),
            "--books" => books = Some(PathBuf::from(val("--books")?)),
            "--bpe" => bpe = PathBuf::from(val("--bpe")?),
            "--n-per-bin" => n = Some(parse_count(&val("--n-per-bin")?)? as usize),
            "--seed" => seed = Some(parse_count(&val("--seed")?)?),
            "--out" => out = Some(PathBuf::from(val("--out")?)),
            "--threads" => threads = parse_count(&val("--threads")?)?.max(1) as usize,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            s => return Err(format!("unknown argument {s}")),
        }
    }
    let set = set.ok_or("--set is required")?;
    let (db, dn, ds, dout) = match set.as_str() {
        "dev" => ("data/pg19/validation", 50, 1, "data/v40/fact_dev.tsv"),
        "test" => ("data/pg19/test", 100, 2, "data/v40/sealed/fact_test.tsv"),
        s => return Err(format!("--set must be dev or test, not {s:?}")),
    };
    Ok(Opts {
        books: books.unwrap_or_else(|| db.into()),
        bpe,
        n: n.unwrap_or(dn),
        seed: seed.unwrap_or(ds),
        out: out.unwrap_or_else(|| dout.into()),
        threads,
        set,
    })
}

fn main() {
    let o = match parse_args() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: {e}\n{USAGE}");
            std::process::exit(2);
        }
    };
    if let Err(e) = run(&o) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(o: &Opts) -> Result<(), String> {
    let t0 = Instant::now();
    let bpe = Bpe::load(&o.bpe).map_err(|e| format!("{}: {e}", o.bpe.display()))?;
    let paths = list_books(&o.books).map_err(|e| format!("{}: {e}", o.books.display()))?;
    let books = encode_books(&bpe, &paths, o.threads).map_err(|e| e.to_string())?;
    let toks: Vec<Vec<u32>> = books.into_iter().map(|b| b.tokens).collect();
    let t_enc = t0.elapsed().as_secs_f64();
    let t = Instant::now();
    let (items, stats) = fact_items(&bpe, &toks, o.n, o.seed)?;
    let t_gen = t.elapsed().as_secs_f64();
    if let Some(dir) = o.out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let mut w = BufWriter::new(std::fs::File::create(&o.out).map_err(|e| e.to_string())?);
    for it in &items {
        writeln!(w, "{}", item_line(it)).map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;
    drop(w);
    println!(
        "{} set: {} books ({} tokens) from {}, seed {}, {} items -> {}",
        o.set,
        toks.len(),
        toks.iter().map(|b| b.len()).sum::<usize>(),
        o.books.display(),
        o.seed,
        items.len(),
        o.out.display()
    );
    println!("  bin      items  books  prompt tokens mean [min, max]   gap text tokens mean [min, max]   right pos   short-book draws");
    for (bi, bin) in BINS.iter().enumerate() {
        let of: Vec<_> = items.iter().filter(|x| x.bin == *bin).collect();
        if of.is_empty() {
            continue;
        }
        let used: HashSet<usize> = of.iter().map(|x| x.book).collect();
        let st = |f: &dyn Fn(&&ava_data::fact_items::FactItem) -> usize| {
            let v: Vec<usize> = of.iter().map(f).collect();
            (v.iter().sum::<usize>() as f64 / v.len() as f64, *v.iter().min().unwrap(), *v.iter().max().unwrap())
        };
        let (pm, pmin, pmax) = st(&|x| x.prompt_tokens);
        let (gm, gmin, gmax) = st(&|x| x.gap_tokens);
        let mut pos = [0usize; 4];
        for x in &of {
            pos[x.right] += 1;
        }
        println!(
            "  {bin:<8} {:>5}  {:>5}  {pm:>9.1} [{pmin}, {pmax}]   {gm:>12.1} [{gmin}, {gmax}]   {pos:?}   {}",
            of.len(),
            used.len(),
            stats.short_books[bi]
        );
    }
    let bytes = std::fs::metadata(&o.out).map_err(|e| e.to_string())?.len();
    println!("sha256 {}  {bytes} bytes", sha256_file(&o.out).map_err(|e| e.to_string())?);
    println!("time: encode books {t_enc:.2}s, items {t_gen:.2}s, total {:.2}s", t0.elapsed().as_secs_f64());
    Ok(())
}
