// TESTING / DESIGN ONLY
//! v42_shard_check: reads a finished v42 shard set back and proves what went in. Every document of
//! train/ and val/ (found through the tag sidecar) is decoded (length marks removed) and checked:
//!   - test wordings: the wording guard (test templates filled with every fact name) must find nothing;
//!   - control facts: the fact ledger over the CONTROL facts counts sentences naming both sides of one.
//!     The restated-facts source must hold 0; other sources are counted as natural exposure (allowed).
//!
//!   v42_shard_check --dir DIR [--threads 2] [--keys-dir DIR] [--wikidata DIR] [--templates FILE]
//!                   [--facts-source NAME]   (default `facts`: the source that must hold no control fact)
//!
//! From --keys-dir it opens only restate_fact_keys.txt, control_fact_keys.txt and the templates file
//! (the same three files v42_facts and v42_shards open). It prints counts only, never a matched text.
//! Exit status 2 when a test wording or a control fact in the facts source is found.

use ava_data::bpe::Bpe;
use ava_data::fact_ledger::{all_facts, guard_names, plan, KeyLists, Ledger, DEFAULT_DEV_PERCENT, DEFAULT_SEED, KEYS_DIR, TEMPLATES_FILE, WIKIDATA_DIR};
use ava_data::numbers::strip_marks;
use ava_data::shards::{ShardSet, ShardTokens};
use ava_data::wikidata::Wikidata;
use ava_data::wording_guard::WordingGuard;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    match run() {
        Ok(true) => {}
        Ok(false) => std::process::exit(2),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn arg(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1).cloned())
}

/// One tag row: (start, len, source row).
type Row = (u64, u64, usize);

/// The tag sidecar: source names (from the `# sources:` header line) and the rows.
fn read_tags(path: &Path) -> io::Result<(Vec<String>, Vec<Row>)> {
    let text = std::fs::read_to_string(path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    let mut names = Vec::new();
    let mut rows = Vec::new();
    for line in text.lines() {
        if let Some(h) = line.strip_prefix("# sources:") {
            for item in h.split_whitespace() {
                let (k, rest) = item.split_once('=').ok_or_else(|| io::Error::other(format!("bad sources header item {item:?}")))?;
                let k: usize = k.parse().map_err(|_| io::Error::other(format!("bad source row {k:?}")))?;
                let name = rest.split('(').next().unwrap_or(rest).to_string();
                if names.len() <= k {
                    names.resize(k + 1, String::new());
                }
                names[k] = name;
            }
            continue;
        }
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let mut it = line.split('\t');
        let p = |s: Option<&str>| -> io::Result<u64> { s.and_then(|x| x.parse().ok()).ok_or_else(|| io::Error::other(format!("bad tag row {line:?}"))) };
        let (s, l, src) = (p(it.next())?, p(it.next())?, p(it.next())? as usize);
        rows.push((s, l, src));
    }
    Ok((names, rows))
}

#[derive(Clone, Default)]
struct Counts {
    docs: u64,
    tokens: u64,
    bytes: u64,
    guard_docs: u64,
    guard_hits: u64,
    guard_full: u64,
}

/// Reads documents of one split by global token position, loading shards as needed (at most two held).
struct DocReader<'a> {
    set: &'a ShardSet,
    /// Global start of every shard.
    starts: Vec<u64>,
    cur: Option<(usize, ShardTokens)>,
}

impl<'a> DocReader<'a> {
    fn new(set: &'a ShardSet) -> DocReader<'a> {
        let mut starts = Vec::new();
        let mut s = 0u64;
        for sh in &set.shards {
            starts.push(s);
            s += sh.header.n_tokens;
        }
        DocReader { set, starts, cur: None }
    }

    fn doc(&mut self, start: u64, len: u64, out: &mut Vec<u32>) -> io::Result<()> {
        out.clear();
        let (mut pos, end) = (start, start + len);
        while pos < end {
            let k = self.starts.partition_point(|&s| s <= pos) - 1;
            if self.cur.as_ref().is_none_or(|(i, _)| *i != k) {
                self.cur = Some((k, self.set.load_shard(k)?));
            }
            let (_, t) = self.cur.as_ref().unwrap();
            let off = (pos - self.starts[k]) as usize;
            let take = ((end - pos) as usize).min(t.len() - off);
            if take == 0 {
                return Err(io::Error::other(format!("document at {start} (+{len}) runs past the split's end")));
            }
            for i in off..off + take {
                out.push(t.get(i));
            }
            pos += take as u64;
        }
        Ok(())
    }
}

fn run() -> io::Result<bool> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = PathBuf::from(arg(&args, "--dir").ok_or_else(|| io::Error::other("--dir is required"))?);
    let threads: usize = arg(&args, "--threads").map(|v| v.parse().unwrap_or(2)).unwrap_or(2).max(1);
    let keys_dir = arg(&args, "--keys-dir").map(PathBuf::from).unwrap_or(PathBuf::from(KEYS_DIR));
    let wd_dir = arg(&args, "--wikidata").map(PathBuf::from).unwrap_or(PathBuf::from(WIKIDATA_DIR));
    let templates = arg(&args, "--templates").map(PathBuf::from).unwrap_or(keys_dir.join(TEMPLATES_FILE));
    let facts_source = arg(&args, "--facts-source").unwrap_or("facts".into());

    let t0 = Instant::now();
    let facts = all_facts();
    let wd = Wikidata::load(&wd_dir)?;
    let keys = KeyLists::load(&keys_dir)?;
    let p = plan(facts.clone(), &keys, &wd, DEFAULT_SEED, DEFAULT_DEV_PERCENT, true);
    let guard = WordingGuard::load(&templates, guard_names(&facts, &wd))?;
    let bpe = Bpe::load(dir.join("bpe.txt"))?;
    println!(
        "v42_shard_check: {}  guard {} templates x {} names  control facts {}  threads {threads}",
        dir.display(),
        guard.templates.len(),
        guard.n_names(),
        p.control.len()
    );

    let mut clean = true;
    let (mut all_tok, mut all_docs) = (0u64, 0u64);
    for split in ["train", "val"] {
        let set = ShardSet::open(dir.join(split))?;
        let (names, rows) = read_tags(&dir.join(split).join("tags.tsv"))?;
        let tag_tokens: u64 = rows.iter().map(|r| r.1).sum();
        if tag_tokens != set.total_tokens {
            return Err(io::Error::other(format!("{split}: tags cover {tag_tokens} tokens, shards hold {}", set.total_tokens)));
        }
        let ts = Instant::now();
        // contiguous slices of rows, one per worker, cut at equal token counts
        let mut cuts = vec![0usize];
        let mut acc = 0u64;
        for (i, r) in rows.iter().enumerate() {
            acc += r.1;
            if acc * threads as u64 >= set.total_tokens * cuts.len() as u64 && cuts.len() < threads {
                cuts.push(i + 1);
            }
        }
        cuts.push(rows.len());
        let ns = names.len();
        let results: Vec<io::Result<(Vec<Counts>, Vec<Ledger>)>> = std::thread::scope(|s| {
            let hs: Vec<_> = cuts
                .windows(2)
                .map(|w| {
                    let (a, b) = (w[0], w[1]);
                    let (rows, set, bpe, guard, control) = (&rows, &set, &bpe, &guard, &p.control);
                    s.spawn(move || -> io::Result<(Vec<Counts>, Vec<Ledger>)> {
                        let mut c = vec![Counts::default(); ns];
                        let mut l: Vec<Ledger> = (0..ns).map(|_| Ledger::new(control.clone())).collect();
                        let mut rd = DocReader::new(set);
                        let mut ids = Vec::new();
                        for &(start, len, src) in &rows[a..b] {
                            rd.doc(start, len, &mut ids)?;
                            let text = bpe.decode(&strip_marks(&ids));
                            let (h, f) = guard.count(&text);
                            let cs = &mut c[src];
                            cs.docs += 1;
                            cs.tokens += len;
                            cs.bytes += text.len() as u64;
                            cs.guard_docs += (h > 0) as u64;
                            cs.guard_hits += h as u64;
                            cs.guard_full += f as u64;
                            l[src].add_text(&text);
                        }
                        Ok((c, l))
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut c = vec![Counts::default(); ns];
        let mut l: Vec<Ledger> = (0..ns).map(|_| Ledger::new(p.control.clone())).collect();
        for r in results {
            let (rc, rl) = r?;
            for i in 0..ns {
                let (x, y) = (&mut c[i], &rc[i]);
                (x.docs, x.tokens, x.bytes) = (x.docs + y.docs, x.tokens + y.tokens, x.bytes + y.bytes);
                (x.guard_docs, x.guard_hits, x.guard_full) = (x.guard_docs + y.guard_docs, x.guard_hits + y.guard_hits, x.guard_full + y.guard_full);
                l[i].merge(&rl[i]);
            }
        }
        let secs = ts.elapsed().as_secs_f64();
        println!("{split}: {} docs, {} tokens, {:.1}s ({:.2}M tokens/s)", rows.len(), set.total_tokens, secs, set.total_tokens as f64 / 1e6 / secs.max(1e-9));
        println!("  {:<15} {:>9} {:>11} {:>9} {:>11} {:>11} {:>14} {:>13}", "source", "docs", "tokens", "MB", "guard_docs", "guard_hits", "ctrl_mentions", "ctrl_stated");
        for i in 0..ns {
            let (m, st): (u64, u64) = (l[i].mentions.iter().sum(), l[i].stated.iter().sum());
            let x = &c[i];
            println!(
                "  {:<15} {:>9} {:>11} {:>9.1} {:>11} {:>11} {:>14} {:>13}",
                names[i],
                x.docs,
                x.tokens,
                x.bytes as f64 / 1e6,
                x.guard_docs,
                x.guard_hits,
                m,
                st
            );
            if x.guard_hits > 0 {
                clean = false;
            }
            if names[i] == facts_source && m > 0 {
                clean = false;
                let bad: Vec<String> = l[i].facts().iter().enumerate().filter(|(k, _)| l[i].mentions[*k] > 0).map(|(k, f)| format!("{} x{}", f.id(), l[i].mentions[k])).collect();
                println!("    CONTROL FACTS IN {facts_source}: {}", bad.join(", "));
            }
        }
        if !names.iter().any(|n| *n == facts_source) {
            println!("  (no source named {facts_source} in {split})");
        }
        all_tok += set.total_tokens;
        all_docs += rows.len() as u64;
    }
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "total: {all_docs} docs, {all_tok} tokens in {secs:.1}s ({:.2}M tokens/s on {threads} threads)  verdict: {}",
        all_tok as f64 / 1e6 / secs.max(1e-9),
        if clean { "CLEAN (0 test wordings; 0 control facts in the facts source)" } else { "NOT CLEAN" }
    );
    Ok(clean)
}
