// TESTING / DESIGN ONLY — v40_mix: build v40's reading data: the train mix, a copy of v39's val shards, and the PG-19 dev/test book shards.
//
//   v40_mix [--pg19 data/pg19] [--usual data/nn/shards-4112-marks-practice]
//           [--practice data/num/practice_v40.txt] [--v39-practice data/num/practice.txt]
//           [--bpe data/nn/bpe-4096.txt] [--out data/nn/shards-v40-mix]
//           [--books-dev-out data/nn/shards-v40-books-dev]
//           [--books-test-out data/nn/sealed/shards-v40-books-test]
//           [--seed 40] [--threads 8] [--shard-tokens 50000000] [--max-book-tokens N]
//
// Writes (vocab 4112, AVATOK01 shards; every book is ONE document: encode_doc + length marks,
// line endings \r\n -> \n):
//   OUT/train/       the mix (ava_data::v40_mix): every book of PG19/train, usual text read in
//                    order from USUAL/train, practice v2 docs; shares 50 / 47 / 3; shuffled by
//                    document (seed). With --v39-practice, the manifest also counts how much of
//                    v39's practice text the usual text carries (by document hash).
//   OUT/val/         USUAL/val copied byte for byte (validation BPB stays comparable with v39)
//   OUT/bpe.txt, OUT/manifest.txt, OUT/SHA256SUMS
//   BOOKS_DEV_OUT/   PG19/validation, one doc per book in name order; books.tsv (name, tokens,
//                    bytes), manifest.txt, SHA256SUMS
//   BOOKS_TEST_OUT/  PG19/test the same way. SEALED: only counts, tokens and sha256 are printed.
// --max-book-tokens N keeps only the first train books in name order whose tokens fit in N (default: every
// book; the shares hold either way, so the total is about twice the book tokens).
// A part whose manifest.txt already exists is skipped, so a re-run resumes.

use ava_data::books::{encode_books, list_books, Book};
use ava_data::bpe::Bpe;
use ava_data::input::parse_count;
use ava_data::numbers::insert_marks;
use ava_data::sha256::sha256_file;
use ava_data::shards::{shard_name, ShardSet, DEFAULT_SHARD_TOKENS};
use ava_data::v40_mix::{build_mix, copy_shards, token_hash, write_docs, DEFAULT_SEED, SHARES, SOURCES};
use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "usage: v40_mix [--pg19 DIR] [--usual DIR] [--practice FILE] [--v39-practice FILE] [--bpe FILE] [--out DIR] \
[--books-dev-out DIR] [--books-test-out DIR] [--seed 40] [--threads 8] [--shard-tokens 50000000] [--max-book-tokens N]";

struct Opts {
    pg19: PathBuf,
    usual: PathBuf,
    practice: PathBuf,
    v39_practice: Option<PathBuf>,
    bpe: PathBuf,
    out: PathBuf,
    dev_out: PathBuf,
    test_out: PathBuf,
    seed: u64,
    threads: usize,
    shard_tokens: usize,
    max_book_tokens: Option<u64>,
}

fn parse_args() -> Result<Opts, String> {
    let mut o = Opts {
        pg19: "data/pg19".into(),
        usual: "data/nn/shards-4112-marks-practice".into(),
        practice: "data/num/practice_v40.txt".into(),
        v39_practice: Some("data/num/practice.txt".into()),
        bpe: "data/nn/bpe-4096.txt".into(),
        out: "data/nn/shards-v40-mix".into(),
        dev_out: "data/nn/shards-v40-books-dev".into(),
        test_out: "data/nn/sealed/shards-v40-books-test".into(),
        seed: DEFAULT_SEED,
        threads: 8,
        shard_tokens: DEFAULT_SHARD_TOKENS,
        max_book_tokens: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |flag: &str| args.next().ok_or_else(|| format!("{flag} needs a value"));
        match a.as_str() {
            "--pg19" => o.pg19 = val("--pg19")?.into(),
            "--usual" => o.usual = val("--usual")?.into(),
            "--practice" => o.practice = val("--practice")?.into(),
            "--v39-practice" => {
                let v = val("--v39-practice")?;
                o.v39_practice = if v == "none" { None } else { Some(v.into()) };
            }
            "--bpe" => o.bpe = val("--bpe")?.into(),
            "--out" => o.out = val("--out")?.into(),
            "--books-dev-out" => o.dev_out = val("--books-dev-out")?.into(),
            "--books-test-out" => o.test_out = val("--books-test-out")?.into(),
            "--seed" => o.seed = parse_count(&val("--seed")?)?,
            "--threads" => o.threads = parse_count(&val("--threads")?)?.max(1) as usize,
            "--shard-tokens" => o.shard_tokens = parse_count(&val("--shard-tokens")?)?.max(1) as usize,
            "--max-book-tokens" => o.max_book_tokens = Some(parse_count(&val("--max-book-tokens")?)?),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            s => return Err(format!("unknown argument {s}")),
        }
    }
    Ok(o)
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

fn err(msg: String) -> std::io::Error {
    std::io::Error::other(msg)
}

/// sha256 of every regular file under `root` (recursively), as `hash  relative/path` lines,
/// sorted by path; written to `root/SHA256SUMS` (which is itself left out) and returned.
fn sha_sums(root: &Path) -> std::io::Result<String> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
        for e in std::fs::read_dir(dir)? {
            let p = e?.path();
            if p.is_dir() {
                walk(&p, out)?;
            } else {
                out.push(p);
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(root, &mut files)?;
    let mut rel: Vec<(String, PathBuf)> = files
        .into_iter()
        .filter(|p| p.file_name().is_some_and(|n| n != "SHA256SUMS"))
        .map(|p| (p.strip_prefix(root).unwrap().to_string_lossy().replace('\\', "/"), p))
        .collect();
    rel.sort();
    let mut s = String::new();
    for (r, p) in rel {
        s.push_str(&format!("{}  {r}\n", sha256_file(&p)?));
    }
    std::fs::write(root.join("SHA256SUMS"), &s)?;
    Ok(s)
}

fn encode_dir(bpe: &Bpe, dir: &Path, threads: usize, what: &str) -> std::io::Result<(Vec<Book>, String)> {
    let t = Instant::now();
    let paths = list_books(dir)?;
    if paths.is_empty() {
        return Err(err(format!("{}: no *.txt books", dir.display())));
    }
    let books = encode_books(bpe, &paths, threads)?;
    let secs = t.elapsed().as_secs_f64();
    let toks: u64 = books.iter().map(|b| b.tokens.len() as u64).sum();
    let bytes: u64 = books.iter().map(|b| b.bytes).sum();
    let lossy = books.iter().filter(|b| b.lossy).count();
    let (min, max) = books.iter().fold((u64::MAX, 0u64), |(a, b), x| (a.min(x.tokens.len() as u64), b.max(x.tokens.len() as u64)));
    let line = format!(
        "{what}: {} books, {bytes} bytes -> {toks} tokens (marks and EOS included; {:.3} bytes/token; book tokens min {min} max {max}); \
{lossy} not valid UTF-8; encoded on {threads} threads in {secs:.1}s = {:.2}M tok/s, {:.1} MB/s\n",
        books.len(),
        bytes as f64 / toks.max(1) as f64,
        toks as f64 / 1e6 / secs.max(1e-9),
        bytes as f64 / 1e6 / secs.max(1e-9)
    );
    Ok((books, line))
}

fn run(o: &Opts) -> std::io::Result<()> {
    let t0 = Instant::now();
    let bpe = Bpe::load(&o.bpe)?;
    if bpe.vocab_size() != 4096 {
        return Err(err(format!("{}: vocab {} (want 4096)", o.bpe.display(), bpe.vocab_size())));
    }

    // ---- 1. the train mix and the val copy
    if o.out.join("manifest.txt").exists() {
        println!("{}: manifest.txt exists, mix skipped", o.out.display());
    } else {
        let mut rep_txt = format!(
            "v40_mix: pg19 {}  usual {}  practice {}  v39-practice {}  bpe {}  seed {}  shard_tokens {}\n",
            o.pg19.display(),
            o.usual.display(),
            o.practice.display(),
            o.v39_practice.as_ref().map_or("none".into(), |p| p.display().to_string()),
            o.bpe.display(),
            o.seed,
            o.shard_tokens
        );
        let (mut books, line) = encode_dir(&bpe, &o.pg19.join("train"), o.threads, "train books")?;
        print!("{line}");
        rep_txt.push_str(&line);
        if let Some(cap) = o.max_book_tokens {
            let (mut sum, mut keep) = (0u64, 0usize);
            while keep < books.len() && sum + books[keep].tokens.len() as u64 <= cap {
                sum += books[keep].tokens.len() as u64;
                keep += 1;
            }
            let line = format!("--max-book-tokens {cap}: the first {keep} of {} books in name order ({sum} tokens) are used\n", books.len());
            print!("{line}");
            rep_txt.push_str(&line);
            books.truncate(keep);
        }
        let known = match &o.v39_practice {
            Some(p) => {
                let t = Instant::now();
                let mut k = HashSet::new();
                for l in BufReader::with_capacity(1 << 20, std::fs::File::open(p)?).lines() {
                    let l = l?;
                    let l = l.trim_end_matches('\r');
                    if !l.is_empty() {
                        k.insert(token_hash(insert_marks(&bpe.encode_doc(l))));
                    }
                }
                let line = format!("v39 practice: {} distinct docs hashed in {:.1}s\n", k.len(), t.elapsed().as_secs_f64());
                print!("{line}");
                rep_txt.push_str(&line);
                Some(k)
            }
            None => None,
        };
        let practice_reader = BufReader::with_capacity(1 << 20, std::fs::File::open(&o.practice)?);
        let bpe2 = bpe.clone();
        let practice = practice_reader
            .lines()
            .map_while(Result::ok)
            .filter(|l| !l.trim_end_matches('\r').is_empty())
            .map(move |l| insert_marks(&bpe2.encode_doc(l.trim_end_matches('\r'))));
        let t = Instant::now();
        let book_tokens: Vec<Vec<u32>> = books.into_iter().map(|b| b.tokens).collect();
        let rep = build_mix(book_tokens, &o.usual.join("train"), practice, &o.out.join("train"), o.seed, o.shard_tokens, known.as_ref())?;
        let secs = t.elapsed().as_secs_f64();
        let mut line = format!("train mix: {} tokens in {} shards; built in {secs:.1}s = {:.1}M tok/s\n", rep.total, rep.shards, rep.total as f64 / 1e6 / secs.max(1e-9));
        for s in 0..3 {
            line.push_str(&format!(
                "  {:<12} {:>9} docs {:>11} tokens  target {:>11}  share {:.4}% (want {:.0}%)\n",
                SOURCES[s],
                rep.docs[s],
                rep.tokens[s],
                rep.targets[s],
                100.0 * rep.share(s),
                100.0 * SHARES[s]
            ));
        }
        if known.is_some() {
            line.push_str(&format!(
                "  inside usual: {} docs, {} tokens of v39's practice text = {:.4}% of the mix ({:.3}% of usual)\n",
                rep.known_docs,
                rep.known_tokens,
                100.0 * rep.known_tokens as f64 / rep.total as f64,
                100.0 * rep.known_tokens as f64 / rep.tokens[1].max(1) as f64
            ));
        }
        print!("{line}");
        rep_txt.push_str(&line);

        let n = copy_shards(&o.usual.join("val"), &o.out.join("val"))?;
        let src = ShardSet::open(o.usual.join("val"))?;
        for i in 0..n {
            let (a, b) = (src.dir.join(shard_name(i)), o.out.join("val").join(shard_name(i)));
            if sha256_file(&a)? != sha256_file(&b)? {
                return Err(err(format!("{} differs from {}", b.display(), a.display())));
            }
        }
        let line = format!("val: {n} shards copied byte for byte from {} ({} tokens)\n", o.usual.join("val").display(), src.total_tokens);
        print!("{line}");
        rep_txt.push_str(&line);
        bpe.save(o.out.join("bpe.txt"))?;
        std::fs::File::create(o.out.join("manifest.txt"))?.write_all(rep_txt.as_bytes())?;
        print!("{}", sha_sums(&o.out)?);
    }

    // ---- 2. and 3. the PG-19 validation (dev) and test (SEALED) books
    for (split, out, sealed) in [("validation", &o.dev_out, false), ("test", &o.test_out, true)] {
        if out.join("manifest.txt").exists() {
            println!("{}: manifest.txt exists, skipped", out.display());
            continue;
        }
        let (books, line) = encode_dir(&bpe, &o.pg19.join(split), o.threads, &format!("{split} books"))?;
        print!("{line}");
        let docs: Vec<&[u32]> = books.iter().map(|b| b.tokens.as_slice()).collect();
        let st = write_docs(out, &docs, o.shard_tokens)?;
        let mut tsv = String::from("name\ttokens\tbytes\n");
        for b in &books {
            tsv.push_str(&format!("{}\t{}\t{}\n", b.name, b.tokens.len(), b.bytes));
        }
        std::fs::write(out.join("books.tsv"), tsv)?;
        let man = format!(
            "v40 PG-19 {split} books{}: {} books, one document each (encode_doc + length marks, \\r\\n -> \\n), in name order; \
{} tokens in {} shards (vocab 4112); bpe {}\n{line}",
            if sealed { " (SEALED)" } else { "" },
            books.len(),
            st.tokens,
            st.shards,
            o.bpe.display()
        );
        std::fs::write(out.join("manifest.txt"), &man)?;
        println!("{}: {} books, {} tokens, {} shards", out.display(), books.len(), st.tokens, st.shards);
        print!("{}", sha_sums(out)?);
    }
    println!("time: {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}
