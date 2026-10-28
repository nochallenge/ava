// TESTING / DESIGN ONLY — recall_dev: write v41's recall dev items (made-up notes in PG-19 validation books, cued later) as TSV.
//
//   recall_dev [--books data/pg19/validation] [--bpe data/nn/bpe-4096.txt] [--n-per-bin 150] [--seed 41]
//              [--out data/v41/recall_dev.tsv] [--threads 6] [--peek N [--peek-len 512]]
//
// Line format (mc.rs read_items; fields escaped as in fact_items):
//   id \t recall \t bin \t right \t prompt \t choice0 \t choice1 \t choice2 \t choice3
// Bins g256, g1024, g1900 (book tokens between the note and the cue), equal counts; every scored sequence
// [EOS] prompt choice [EOS] is at most 2048 tokens. A DEV set (the books are PG-19 validation): items may be read.
// --peek N prints N practice rows (`recall::recall_row`, length --peek-len) cut from the same books, decoded, with
// where the notes, cues and re-read landed, and writes nothing.

use ava_data::books::{encode_books, list_books};
use ava_data::bpe::Bpe;
use ava_data::input::parse_count;
use ava_data::numbers::strip_marks;
use ava_data::recall::{dev_items, dev_line, recall_row_plan, DEV_BINS, FRAMES};
use ava_data::rng::SplitMix64;
use ava_data::sha256::sha256_file;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: recall_dev [--books DIR] [--bpe FILE] [--n-per-bin 150] [--seed 41] [--out FILE] [--threads 6] [--peek N [--peek-len 512]]";

struct Opts {
    books: PathBuf,
    bpe: PathBuf,
    n: usize,
    seed: u64,
    out: PathBuf,
    threads: usize,
    peek: usize,
    peek_len: usize,
}

fn parse_args() -> Result<Opts, String> {
    let mut o = Opts {
        books: "data/pg19/validation".into(),
        bpe: "data/nn/bpe-4096.txt".into(),
        n: 150,
        seed: 41,
        out: "data/v41/recall_dev.tsv".into(),
        threads: 6,
        peek: 0,
        peek_len: 512,
    };
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut val = |flag: &str| args.next().ok_or_else(|| format!("{flag} needs a value"));
        match a.as_str() {
            "--books" => o.books = val("--books")?.into(),
            "--bpe" => o.bpe = val("--bpe")?.into(),
            "--n-per-bin" => o.n = parse_count(&val("--n-per-bin")?)? as usize,
            "--seed" => o.seed = parse_count(&val("--seed")?)?,
            "--out" => o.out = val("--out")?.into(),
            "--threads" => o.threads = parse_count(&val("--threads")?)?.max(1) as usize,
            "--peek" => o.peek = parse_count(&val("--peek")?)? as usize,
            "--peek-len" => o.peek_len = parse_count(&val("--peek-len")?)?.max(1) as usize,
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

fn run(o: &Opts) -> Result<(), String> {
    let t0 = Instant::now();
    let bpe = Bpe::load(&o.bpe).map_err(|e| format!("{}: {e}", o.bpe.display()))?;
    let paths = list_books(&o.books).map_err(|e| format!("{}: {e}", o.books.display()))?;
    let toks: Vec<Vec<u32>> = encode_books(&bpe, &paths, o.threads).map_err(|e| e.to_string())?.into_iter().map(|b| b.tokens).collect();
    let t_enc = t0.elapsed().as_secs_f64();

    if o.peek > 0 {
        let mut r = SplitMix64::new(o.seed);
        let t = Instant::now();
        for i in 0..o.peek {
            let book = &toks[r.below(toks.len() as u64) as usize];
            let s = r.below((book.len() - o.peek_len - 1) as u64) as usize;
            let (row, plan) = recall_row_plan(&mut r, &bpe, &book[s..s + o.peek_len + 1], o.peek_len);
            println!("==== row {i}: {} tokens", row.len());
            for p in &plan.notes {
                println!(
                    "  note [{}] at {} ({} tok), cue at {} ({} + {} tok), gap {}: {:?} / {:?}",
                    p.fact.frame,
                    p.note_at,
                    p.note_len,
                    p.cue_at,
                    p.cue_len,
                    p.answer_len,
                    p.gap(),
                    p.fact.note,
                    p.fact.cue
                );
            }
            if let Some(rr) = &plan.reread {
                println!("  re-read: {} tokens at {} copied to {}", rr.len, rr.src_at, rr.copy_at);
            }
            println!("{}", bpe.decode(&strip_marks(&row)));
        }
        println!("peek: {} rows in {:.3}s", o.peek, t.elapsed().as_secs_f64());
        return Ok(());
    }

    let t = Instant::now();
    let (items, st) = dev_items(&bpe, &toks, o.n, o.seed)?;
    let t_gen = t.elapsed().as_secs_f64();
    if let Some(dir) = o.out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let mut w = BufWriter::new(std::fs::File::create(&o.out).map_err(|e| e.to_string())?);
    for it in &items {
        writeln!(w, "{}", dev_line(it)).map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())?;
    drop(w);
    println!(
        "recall dev: {} books ({} tokens) from {}, seed {}, {} items -> {}",
        toks.len(),
        toks.iter().map(|b| b.len()).sum::<usize>(),
        o.books.display(),
        o.seed,
        items.len(),
        o.out.display()
    );
    println!("  bin      items  prompt tokens mean [min, max]  gap tokens mean [min, max]  longest seq  right pos  redraws (short/no-sentence/too-long/value)");
    for (bi, bin) in DEV_BINS.iter().enumerate() {
        let of: Vec<_> = items.iter().filter(|x| x.bin == *bin).collect();
        if of.is_empty() {
            continue;
        }
        let stat = |v: Vec<usize>| (v.iter().sum::<usize>() as f64 / v.len() as f64, *v.iter().min().unwrap(), *v.iter().max().unwrap());
        let (pm, pmin, pmax) = stat(of.iter().map(|x| x.prompt_tokens).collect());
        let (gm, gmin, gmax) = stat(of.iter().map(|x| x.gap_tokens).collect());
        let longest = of.iter().map(|x| x.max_seq).max().unwrap();
        let mut pos = [0usize; 4];
        for x in &of {
            pos[x.right] += 1;
        }
        println!(
            "  {bin:<8} {:>5}  {pm:>8.1} [{pmin}, {pmax}]  {gm:>8.1} [{gmin}, {gmax}]  {longest:>5}  {pos:?}  {}/{}/{}/{}",
            of.len(),
            st.short_books[bi],
            st.no_sentence_start[bi],
            st.too_long[bi],
            st.value_in_text[bi]
        );
    }
    let per: Vec<String> = FRAMES.iter().map(|f| format!("{f} {}", items.iter().filter(|x| x.frame == *f).count())).collect();
    println!("  frames: {}", per.join(", "));
    let bytes = std::fs::metadata(&o.out).map_err(|e| e.to_string())?.len();
    println!("sha256 {}  {bytes} bytes", sha256_file(&o.out).map_err(|e| e.to_string())?);
    println!("time: encode books {t_enc:.2}s, items {t_gen:.2}s, total {:.2}s", t0.elapsed().as_secs_f64());
    Ok(())
}
