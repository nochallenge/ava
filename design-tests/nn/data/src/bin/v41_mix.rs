// TESTING / DESIGN ONLY — v41_mix: build v41's reading data: 100 held-out PG-19 train books (sealed long-text and find-the-fact tests) and the train mix.
//
//   v41_mix [--pg19 data/pg19] [--usual data/nn/shards-4112-marks-practice] [--practice data/num/practice_v41.txt]
//           [--v39-practice data/num/practice.txt] [--bpe data/nn/bpe-4096.txt] [--out data/nn/shards-v41-mix]
//           [--books-test-out data/nn/sealed/shards-v41-books-test] [--fact-test-out data/v41/sealed/fact_test.tsv]
//           [--fact-seed 41] [--seed 41] [--threads 6] [--shard-tokens 50000000] [--book-target 45000000]
//
// Every book is ONE document (encode_doc + length marks, \r\n -> \n), vocab 4112, as in v40.
//   BOOKS_TEST_OUT/  the 100 held-out books (ava_data::v41_mix::heldout: a seeded draw among PG19/train books
//                    with 16,385+ tokens), one doc per book in name order; books.tsv, manifest.txt, SHA256SUMS.
//                    SEALED: only counts, tokens and sha256 are printed.
//   FACT_TEST_OUT    ava_data::fact_items from the held-out books, 100 items per bin (g256, g1024, g4096,
//                    g12288), seed --fact-seed. SEALED: counts and sha256 only.
//   OUT/train/       the mix: the other 500 books in a seeded order, whole, until --book-target tokens (25%);
//                    usual text in order from USUAL/train (70%); practice v3 docs in order (5%); doc-shuffled.
//   OUT/val/         USUAL/val copied byte for byte (the English rule compares on it, as in v40)
//   OUT/bpe.txt, OUT/books.tsv (the training books: name, tokens), OUT/manifest.txt, OUT/SHA256SUMS
// A part whose manifest.txt (or, for the fact test, whose file) already exists is skipped, so a re-run resumes.

use ava_data::books::{encode_books, list_books, Book};
use ava_data::bpe::Bpe;
use ava_data::fact_items::{fact_items, item_line, BINS};
use ava_data::input::parse_count;
use ava_data::numbers::insert_marks;
use ava_data::sha256::sha256_file;
use ava_data::shards::{shard_name, ShardSet, DEFAULT_SHARD_TOKENS};
use ava_data::v40_mix::{build_mix_with, copy_shards, token_hash, write_docs, SOURCES};
use ava_data::v41_mix::{heldout, train_books, BOOK_TARGET, DEFAULT_SEED, HELDOUT_BOOKS, HELDOUT_MIN_TOKENS, SHARES};
use std::collections::HashSet;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "usage: v41_mix [--pg19 DIR] [--usual DIR] [--practice FILE] [--v39-practice FILE] [--bpe FILE] [--out DIR] \
[--books-test-out DIR] [--fact-test-out FILE] [--fact-seed 41] [--seed 41] [--threads 6] [--shard-tokens 50000000] [--book-target 45000000]";

struct Opts {
    pg19: PathBuf,
    usual: PathBuf,
    practice: PathBuf,
    v39_practice: Option<PathBuf>,
    bpe: PathBuf,
    out: PathBuf,
    test_out: PathBuf,
    fact_out: PathBuf,
    fact_seed: u64,
    seed: u64,
    threads: usize,
    shard_tokens: usize,
    book_target: u64,
}

fn parse_args() -> Result<Opts, String> {
    let mut o = Opts {
        pg19: "data/pg19".into(),
        usual: "data/nn/shards-4112-marks-practice".into(),
        practice: "data/num/practice_v41.txt".into(),
        v39_practice: Some("data/num/practice.txt".into()),
        bpe: "data/nn/bpe-4096.txt".into(),
        out: "data/nn/shards-v41-mix".into(),
        test_out: "data/nn/sealed/shards-v41-books-test".into(),
        fact_out: "data/v41/sealed/fact_test.tsv".into(),
        fact_seed: 41,
        seed: DEFAULT_SEED,
        threads: 6,
        shard_tokens: DEFAULT_SHARD_TOKENS,
        book_target: BOOK_TARGET,
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
            "--books-test-out" => o.test_out = val("--books-test-out")?.into(),
            "--fact-test-out" => o.fact_out = val("--fact-test-out")?.into(),
            "--fact-seed" => o.fact_seed = parse_count(&val("--fact-seed")?)?,
            "--seed" => o.seed = parse_count(&val("--seed")?)?,
            "--threads" => o.threads = parse_count(&val("--threads")?)?.max(1) as usize,
            "--shard-tokens" => o.shard_tokens = parse_count(&val("--shard-tokens")?)?.max(1) as usize,
            "--book-target" => o.book_target = parse_count(&val("--book-target")?)?,
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

/// sha256 of every regular file under `root` (recursively), as `hash  relative/path` lines, sorted by path;
/// written to `root/SHA256SUMS` (which is itself left out) and returned.
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

fn run(o: &Opts) -> std::io::Result<()> {
    let t0 = Instant::now();
    let bpe = Bpe::load(&o.bpe)?;
    if bpe.vocab_size() != 4096 {
        return Err(err(format!("{}: vocab {} (want 4096)", o.bpe.display(), bpe.vocab_size())));
    }
    let t = Instant::now();
    let paths = list_books(&o.pg19.join("train"))?;
    let books: Vec<Book> = encode_books(&bpe, &paths, o.threads)?;
    let secs = t.elapsed().as_secs_f64();
    let toks: Vec<usize> = books.iter().map(|b| b.tokens.len()).collect();
    let all: u64 = toks.iter().map(|&x| x as u64).sum();
    let enc_line = format!(
        "train books: {} books -> {all} tokens (marks and EOS included); {} not valid UTF-8; encoded on {} threads in {secs:.1}s = {:.1}M tok/s\n",
        books.len(),
        books.iter().filter(|b| b.lossy).count(),
        o.threads,
        all as f64 / 1e6 / secs.max(1e-9)
    );
    print!("{enc_line}");
    let held = heldout(&toks, HELDOUT_BOOKS, HELDOUT_MIN_TOKENS, o.seed).map_err(err)?;
    let held_tokens: u64 = held.iter().map(|&i| toks[i] as u64).sum();
    let eligible = toks.iter().filter(|&&x| x >= HELDOUT_MIN_TOKENS).count();
    let held_line = format!(
        "held out: {} of {eligible} books with {HELDOUT_MIN_TOKENS}+ tokens (seed {}), {held_tokens} tokens; the other {} books are training books\n",
        held.len(),
        o.seed,
        books.len() - held.len()
    );
    print!("{held_line}");

    // ---- 1. the held-out books as a SEALED shard set
    if o.test_out.join("manifest.txt").exists() {
        println!("{}: manifest.txt exists, skipped", o.test_out.display());
    } else {
        let docs: Vec<&[u32]> = held.iter().map(|&i| books[i].tokens.as_slice()).collect();
        let st = write_docs(&o.test_out, &docs, o.shard_tokens)?;
        let mut tsv = String::from("name\ttokens\tbytes\n");
        for &i in &held {
            tsv.push_str(&format!("{}\t{}\t{}\n", books[i].name, books[i].tokens.len(), books[i].bytes));
        }
        std::fs::write(o.test_out.join("books.tsv"), tsv)?;
        let man = format!(
            "v41 held-out PG-19 train books (SEALED): {} books, one document each (encode_doc + length marks, \\r\\n -> \\n), \
in name order; {} tokens in {} shards (vocab 4112); bpe {}\n{enc_line}{held_line}",
            held.len(),
            st.tokens,
            st.shards,
            o.bpe.display()
        );
        std::fs::write(o.test_out.join("manifest.txt"), &man)?;
        println!("{}: {} books, {} tokens, {} shards (SEALED)", o.test_out.display(), held.len(), st.tokens, st.shards);
        print!("{}", sha_sums(&o.test_out)?);
    }

    // ---- 2. the SEALED find-the-fact test from the held-out books
    if o.fact_out.exists() {
        println!("{}: exists, skipped", o.fact_out.display());
    } else {
        let t = Instant::now();
        let held_toks: Vec<Vec<u32>> = held.iter().map(|&i| books[i].tokens.clone()).collect();
        let (items, stats) = fact_items(&bpe, &held_toks, 100, o.fact_seed).map_err(err)?;
        if let Some(dir) = o.fact_out.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        let mut w = BufWriter::new(std::fs::File::create(&o.fact_out)?);
        for it in &items {
            writeln!(w, "{}", item_line(it))?;
        }
        w.flush()?;
        drop(w);
        println!("{} (SEALED): {} items from {} held-out books, seed {}", o.fact_out.display(), items.len(), held_toks.len(), o.fact_seed);
        for (bi, bin) in BINS.iter().enumerate() {
            let of: Vec<_> = items.iter().filter(|x| x.bin == *bin).collect();
            let used: HashSet<usize> = of.iter().map(|x| x.book).collect();
            let mean = |f: &dyn Fn(&&ava_data::fact_items::FactItem) -> usize| of.iter().map(f).sum::<usize>() as f64 / of.len().max(1) as f64;
            let maxp = of.iter().map(|x| x.prompt_tokens).max().unwrap_or(0);
            let mut pos = [0usize; 4];
            for x in &of {
                pos[x.right] += 1;
            }
            println!(
                "  {bin:<8} {:>4} items  {:>3} books  prompt tokens mean {:.1} max {maxp}  gap tokens mean {:.1}  right pos {pos:?}  short-book draws {}",
                of.len(),
                used.len(),
                mean(&|x| x.prompt_tokens),
                mean(&|x| x.gap_tokens),
                stats.short_books[bi]
            );
        }
        println!("  sha256 {}  {} bytes  ({:.2}s)", sha256_file(&o.fact_out)?, std::fs::metadata(&o.fact_out)?.len(), t.elapsed().as_secs_f64());
    }

    // ---- 3. the train mix and the val copy
    if o.out.join("manifest.txt").exists() {
        println!("{}: manifest.txt exists, mix skipped", o.out.display());
    } else {
        let mut rep_txt = format!(
            "v41_mix: pg19 {}  usual {}  practice {}  v39-practice {}  bpe {}  seed {}  shard_tokens {}  book_target {}  shares books/usual/practice {:?}\n{enc_line}{held_line}",
            o.pg19.display(),
            o.usual.display(),
            o.practice.display(),
            o.v39_practice.as_ref().map_or("none".into(), |p| p.display().to_string()),
            o.bpe.display(),
            o.seed,
            o.shard_tokens,
            o.book_target,
            SHARES
        );
        let train = train_books(&toks, &held, o.book_target, o.seed);
        let mut btsv = String::from("name\ttokens\tbytes\n");
        for &i in &train {
            btsv.push_str(&format!("{}\t{}\t{}\n", books[i].name, books[i].tokens.len(), books[i].bytes));
        }
        let line = format!(
            "training books: {} of the {} not held out, seeded order, whole books, {} tokens (target {})\n",
            train.len(),
            books.len() - held.len(),
            train.iter().map(|&i| toks[i] as u64).sum::<u64>(),
            o.book_target
        );
        print!("{line}");
        rep_txt.push_str(&line);
        let known = match &o.v39_practice {
            Some(p) => {
                let mut k = HashSet::new();
                for l in BufReader::with_capacity(1 << 20, std::fs::File::open(p)?).lines() {
                    let l = l?;
                    let l = l.trim_end_matches('\r');
                    if !l.is_empty() {
                        k.insert(token_hash(insert_marks(&bpe.encode_doc(l))));
                    }
                }
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
        let mut slots: Vec<Option<Book>> = books.into_iter().map(Some).collect();
        let book_tokens: Vec<Vec<u32>> = train.iter().map(|&i| slots[i].take().unwrap().tokens).collect();
        drop(slots);
        let t = Instant::now();
        let rep = build_mix_with(book_tokens, &o.usual.join("train"), practice, &o.out.join("train"), o.seed, o.shard_tokens, known.as_ref(), SHARES)?;
        let secs = t.elapsed().as_secs_f64();
        let mut line = format!(
            "train mix: {} tokens in {} shards; built in {secs:.1}s = {:.1}M tok/s\n",
            rep.total,
            rep.shards,
            rep.total as f64 / 1e6 / secs.max(1e-9)
        );
        let names = [SOURCES[0], SOURCES[1], "practice_v3"];
        for s in 0..3 {
            line.push_str(&format!(
                "  {:<12} {:>9} docs {:>11} tokens  target {:>11}  share {:.4}% (want {:.0}%)\n",
                names[s],
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
        std::fs::write(o.out.join("books.tsv"), btsv)?;
        std::fs::File::create(o.out.join("manifest.txt"))?.write_all(rep_txt.as_bytes())?;
        print!("{}", sha_sums(&o.out)?);
    }
    println!("time: {:.1}s", t0.elapsed().as_secs_f64());
    Ok(())
}
