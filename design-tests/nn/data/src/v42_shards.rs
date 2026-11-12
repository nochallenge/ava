// TESTING / DESIGN ONLY
//! v42_shards: v42's shard builder. Sources (whole, or a token budget of each) into train/val shards, with
//! a val slice per source (V2), a tag sidecar holding each document's origin and soft toxicity score
//! (V6 + C2), a mixture given as weights per source, and a guard that keeps the test wordings out.
//!
//! SOURCES. `NAME:ORIGIN:SPEC`, e.g. `cornell:dialogue:C:/.../dialogue_cornell.docs`, optionally followed by
//! `|val=VALSPEC` (a separate val file, e.g. BabyLM's dev files). NAME is lowercase letters, digits, `_`
//! or `-`; ORIGIN one of `tags::Origin`; SPEC any `docs::SourceReader` spec (a `.docs` file, a line spec
//! such as `path@tab:2`, or a book list `DIR@list:FILE`).
//!
//! MIXTURE. A source may have a train token budget (None = all of it). The bin turns `--mix
//! name=weight,...` and `--train-tokens T` into budgets T * weight / sum of weights. A source holding less
//! than its budget gives all it has; the realised shares are in the report and in sources.tsv.
//!
//! ORDER. Sources are interleaved proportionally, as `make_shards` does: next comes the source with the
//! smallest fraction done, the fraction being max(bytes read / total bytes, docs / max docs, train tokens /
//! budget). So every shard holds roughly the global mixture and the sources end together. Ties go to the
//! first source.
//!
//! TOKENS. `encode_doc` (EOS last) then `numbers::insert_marks` (vocab 4112), as v39-v41's shards.
//!
//! SPLIT. A source with a val file puts every document of SPEC into train and takes its val slice from
//! VALSPEC. Any other source splits by the doc-hash rule: its k-th document (0-based, counting every
//! document read from that source, in its own order) is a val document when `shards::is_val_doc(seed, k,
//! val_per_mille)` (the rule of every earlier shard set, counted per source so a source's slice does not
//! move when other sources are added or removed). A val document is NEVER trained on. With a val cap
//! (`val_tokens`) the val documents past the cap are skipped (neither val nor train), and after a source's
//! train budget is met its reading goes on, without tokenizing train documents, until its val cap is met:
//! so a source's val slice is the same whatever its budget. With no cap, the slice is the val documents
//! among the documents read. A val file much larger than the cap is thinned evenly (document k kept when
//! `mix64(seed ^ k) % 1000 < keep`, keep = 1000 * 3 * cap / file bytes, at least 1) and read until the
//! cap. `no_val` writes no val at all (val documents are still never trained on): for proxy runs that are
//! scored on a full build's val slices. Val documents are written twice: to `val/` (all sources) and to
//! `val_by_source/NAME/` (that source only), so English can be scored per source.
//!
//! GUARD. With a `wording_guard::WordingGuard`, every document about to be written is checked. One that
//! holds a test template filled with a name (the test's own question, with or without its "Q:" frame,
//! e.g. a Tatoeba sentence "What is the capital of X?") is dropped and counted, so no training or val text
//! holds a filled test wording. `guard_keep_questions` drops only the framed prompt and keeps the bare
//! question (counted). `max_bit_run` drops a document with a longer run of single 0/1 words than that
//! (`wording_guard::longest_bit_run`; number tables and bit vectors), which could share a 13-word gram
//! with the v42 parity items by chance. The doc index k still counts a dropped document, so no other
//! split moves.
//!
//! TAG SIDECAR. `train/tags.tsv` and `val/tags.tsv`: one row per document in written order,
//! `start \t len \t source \t origin \t level \t per_mille`, where start is the document's first token in
//! that split's token stream (shard = start / shard_tokens, offset = start % shard_tokens), len its
//! tokens (EOS included), source the row number of the source table in the header, origin the origin
//! name, and level / per_mille the toxicity score (`tags::ToxScore`). Tags are metadata only: no
//! document is ever dropped for them.
//!
//! OUTPUT. train/, val/, val_by_source/NAME/, the two tag files, sources.tsv (one row per source: weight,
//! budget, tokens and shares, val, guard counts, estimated total tokens), and from the bin bpe.txt,
//! SHA256SUMS (the sha256 of every other file) and manifest.txt (the report).
//!
//! RESUME. The build writes `.building` in the output first and `manifest.txt` last. An output with a
//! manifest is finished and is skipped; one with `.building` but no manifest is an interrupted run of
//! this tool and is removed and rebuilt; any other non-empty directory is refused.

use crate::bpe::{Bpe, EOS};
use crate::docs::SourceReader;
use crate::numbers::insert_marks_into;
use crate::rng::mix64;
use crate::sha256::sha256_file;
use crate::shards::{is_val_doc, ShardWriter};
use crate::tags::{Origin, TagStats, ToxLexicon};
use crate::wording_guard::{longest_bit_run, WordingGuard};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceDef {
    pub name: String,
    pub origin: Origin,
    pub spec: String,
    /// A separate val file (then SPEC is all train).
    pub val_spec: Option<String>,
}

impl SourceDef {
    pub fn parse(s: &str) -> Result<SourceDef, String> {
        let (main, val) = match s.split_once("|val=") {
            Some((m, v)) => (m, Some(v)),
            None => (s, None),
        };
        let mut it = main.splitn(3, ':');
        let (Some(name), Some(origin), Some(spec)) = (it.next(), it.next(), it.next()) else {
            return Err(format!("bad source {s:?}: want NAME:ORIGIN:SPEC[|val=SPEC]"));
        };
        if name.is_empty() || !name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-') {
            return Err(format!("bad source name {name:?}: lowercase letters, digits, _ and - only"));
        }
        if spec.is_empty() || val == Some("") {
            return Err(format!("bad source {s:?}: empty spec"));
        }
        Ok(SourceDef { name: name.to_string(), origin: Origin::parse(origin)?, spec: spec.to_string(), val_spec: val.map(str::to_string) })
    }

    /// Back to the `NAME:ORIGIN:SPEC[|val=SPEC]` form.
    pub fn to_arg(&self) -> String {
        let mut s = format!("{}:{}:{}", self.name, self.origin.name(), self.spec);
        if let Some(v) = &self.val_spec {
            s.push_str("|val=");
            s.push_str(v);
        }
        s
    }
}

pub struct BuildParams {
    pub out: PathBuf,
    pub sources: Vec<SourceDef>,
    pub max_docs: Option<u64>,
    pub shard_tokens: usize,
    pub val_per_mille: u32,
    pub seed: u64,
    /// Only score tags; write no shards (budgets are then ignored).
    pub tags_only: bool,
    /// Train token budget per source, in source order (empty = no budgets).
    pub budgets: Vec<Option<u64>>,
    /// The mixture weights the budgets came from, for the report (empty = none).
    pub weights: Vec<f64>,
    /// Most val tokens per source.
    pub val_tokens: Option<u64>,
    /// Write no val at all.
    pub no_val: bool,
    pub guard: Option<WordingGuard>,
    /// Keep documents whose filled test question has no "Q:" frame (drop only the test's own prompt).
    pub guard_keep_questions: bool,
    /// Drop a document with a run of more than this many single-digit 0/1 words (`longest_bit_run`).
    pub max_bit_run: Option<usize>,
}

impl BuildParams {
    /// The plain build: whole sources, no cap, no guard (the defaults of every earlier v42 build).
    pub fn new(out: PathBuf, sources: Vec<SourceDef>, shard_tokens: usize, val_per_mille: u32, seed: u64) -> BuildParams {
        BuildParams {
            out,
            sources,
            max_docs: None,
            shard_tokens,
            val_per_mille,
            seed,
            tags_only: false,
            budgets: Vec::new(),
            weights: Vec::new(),
            val_tokens: None,
            no_val: false,
            guard: None,
            guard_keep_questions: false,
            max_bit_run: None,
        }
    }

    fn budget(&self, i: usize) -> Option<u64> {
        self.budgets.get(i).copied().flatten()
    }
}

#[derive(Clone, Debug, Default)]
pub struct SourceStats {
    /// Documents read from SPEC (k counts these).
    pub docs: u64,
    /// Tokens written, train + val.
    pub tokens: u64,
    /// Text bytes of the documents written.
    pub bytes: u64,
    pub train_docs: u64,
    pub train_tokens: u64,
    pub val_docs: u64,
    pub val_tokens: u64,
    /// Val documents not written (past the cap, or no val).
    pub val_skipped: u64,
    /// Documents dropped by the guard.
    pub guard_dropped: u64,
    /// Documents that held a filled test question (with or without "Q:"; dropped or not).
    pub guard_core: u64,
    /// Documents dropped for a long run of 0/1 words (`max_bit_run`).
    pub bit_dropped: u64,
    /// SPEC was read to its end.
    pub exhausted: bool,
    /// Size of SPEC's file(s).
    pub total_bytes: u64,
    pub tags: TagStats,
}

impl SourceStats {
    /// Tokens SPEC would give in all (tokens per text byte written, times its size).
    pub fn est_total_tokens(&self) -> u64 {
        if self.bytes == 0 {
            return 0;
        }
        (self.tokens as f64 / self.bytes as f64 * self.total_bytes as f64).round() as u64
    }
}

fn tags_header(sources: &[SourceDef]) -> String {
    let mut h = String::from("# v42 tags, one row per document in written order: start\tlen\tsource\torigin\tlevel\tper_mille\n# sources:");
    for (i, s) in sources.iter().enumerate() {
        h.push_str(&format!(" {i}={}({})", s.name, s.origin.name()));
    }
    h.push('\n');
    h
}

struct Writers {
    train: ShardWriter,
    val: ShardWriter,
    by_source: Vec<ShardWriter>,
    train_tags: BufWriter<File>,
    val_tags: BufWriter<File>,
    train_pos: u64,
    val_pos: u64,
}

struct Run<'a> {
    p: &'a BuildParams,
    bpe: &'a Bpe,
    lex: ToxLexicon,
    w: Option<Writers>,
    stats: Vec<SourceStats>,
    ids: Vec<u32>,
    marked: Vec<u32>,
}

impl Run<'_> {
    fn val_full(&self, i: usize) -> bool {
        self.p.val_tokens.is_some_and(|c| self.stats[i].val_tokens >= c)
    }

    /// Guard, tag, tokenize and write one document of source `i` (to val when `val`).
    fn put(&mut self, i: usize, doc: &str, val: bool) -> io::Result<()> {
        if let Some(g) = &self.p.guard {
            let (core, full) = g.count(doc);
            self.stats[i].guard_core += (core > 0) as u64;
            if full > 0 || (core > 0 && !self.p.guard_keep_questions) {
                self.stats[i].guard_dropped += 1;
                return Ok(());
            }
        }
        if self.p.max_bit_run.is_some_and(|m| longest_bit_run(doc) > m) {
            self.stats[i].bit_dropped += 1;
            return Ok(());
        }
        let score = self.lex.score(doc);
        let st = &mut self.stats[i];
        st.tags.add(&score);
        st.bytes += doc.len() as u64;
        let Some(w) = self.w.as_mut() else {
            if val {
                st.val_docs += 1;
            } else {
                st.train_docs += 1;
            }
            return Ok(());
        };
        self.ids.clear();
        self.bpe.encode_into(doc, &mut self.ids);
        self.ids.push(EOS);
        self.marked.clear();
        insert_marks_into(&self.ids, &mut self.marked);
        let len = self.marked.len() as u64;
        st.tokens += len;
        let origin = self.p.sources[i].origin.name();
        let row = |start: u64| format!("{start}\t{len}\t{i}\t{origin}\t{}\t{}\n", score.level, score.per_mille());
        if val {
            st.val_docs += 1;
            st.val_tokens += len;
            w.val.push_doc(&self.marked)?;
            w.by_source[i].push_doc(&self.marked)?;
            w.val_tags.write_all(row(w.val_pos).as_bytes())?;
            w.val_pos += len;
        } else {
            st.train_docs += 1;
            st.train_tokens += len;
            w.train.push_doc(&self.marked)?;
            w.train_tags.write_all(row(w.train_pos).as_bytes())?;
            w.train_pos += len;
        }
        Ok(())
    }
}

/// Build (or with `tags_only`, just score) all sources. Returns per-source stats and the report text.
/// Writes sources.tsv too (not with `tags_only`).
pub fn build(p: &BuildParams, bpe: &Bpe) -> io::Result<(Vec<SourceStats>, String)> {
    let t0 = Instant::now();
    let n = p.sources.len();
    let mut readers = Vec::new();
    let mut totals = Vec::new();
    for s in &p.sources {
        let r = SourceReader::open(&s.spec).map_err(io::Error::other)?;
        totals.push(r.total_bytes(&s.spec).max(1));
        readers.push(r);
    }
    let vocab = crate::numbers::VOCAB_GLANCE;
    let w = if p.tags_only {
        None
    } else {
        let mut by_source = Vec::new();
        for s in &p.sources {
            by_source.push(ShardWriter::new(p.out.join("val_by_source").join(&s.name), vocab, p.shard_tokens)?);
        }
        let mut train_tags = BufWriter::with_capacity(1 << 20, File::create(p.out.join("train_tags.tsv.tmp"))?);
        let mut val_tags = BufWriter::with_capacity(1 << 20, File::create(p.out.join("val_tags.tsv.tmp"))?);
        train_tags.write_all(tags_header(&p.sources).as_bytes())?;
        val_tags.write_all(tags_header(&p.sources).as_bytes())?;
        Some(Writers {
            train: ShardWriter::new(p.out.join("train"), vocab, p.shard_tokens)?,
            val: ShardWriter::new(p.out.join("val"), vocab, p.shard_tokens)?,
            by_source,
            train_tags,
            val_tags,
            train_pos: 0,
            val_pos: 0,
        })
    };
    let mut run = Run { p, bpe, lex: ToxLexicon::builtin(), w, stats: vec![SourceStats::default(); n], ids: Vec::new(), marked: Vec::new() };
    for (st, &t) in run.stats.iter_mut().zip(&totals) {
        st.total_bytes = t;
    }
    let budgets_on = !p.tags_only;
    let mut done = vec![false; n];

    // 1. the interleaved train stream (and the doc-hash val slices met on the way)
    loop {
        let mut pick: Option<(f64, usize)> = None;
        for i in 0..n {
            if done[i] {
                continue;
            }
            let mut f = readers[i].raw_bytes() as f64 / totals[i] as f64;
            if let Some(m) = p.max_docs {
                f = f.max(readers[i].docs() as f64 / m as f64);
            }
            if let (true, Some(b)) = (budgets_on, p.budget(i)) {
                f = f.max(run.stats[i].train_tokens as f64 / b.max(1) as f64);
            }
            if pick.is_none_or(|(b, _)| f < b) {
                pick = Some((f, i));
            }
        }
        let Some((_, i)) = pick else { break };
        if p.max_docs.is_some_and(|m| readers[i].docs() >= m) || (budgets_on && p.budget(i).is_some_and(|b| run.stats[i].train_tokens >= b)) {
            done[i] = true;
            continue;
        }
        let Some(doc) = readers[i].next_doc()? else {
            done[i] = true;
            run.stats[i].exhausted = true;
            continue;
        };
        let k = run.stats[i].docs;
        run.stats[i].docs += 1;
        let val = p.sources[i].val_spec.is_none() && is_val_doc(p.seed, k, p.val_per_mille);
        if val && (p.no_val || run.val_full(i)) {
            run.stats[i].val_skipped += 1;
            continue;
        }
        run.put(i, &doc, val)?;
    }

    // 2. a source stopped by its budget reads on for its val slice (train documents are not tokenized)
    if !p.no_val && p.val_tokens.is_some() {
        for i in 0..n {
            if p.sources[i].val_spec.is_some() || run.stats[i].exhausted {
                continue;
            }
            while !run.val_full(i) && !p.max_docs.is_some_and(|m| readers[i].docs() >= m) {
                let Some(doc) = readers[i].next_doc()? else {
                    run.stats[i].exhausted = true;
                    break;
                };
                let k = run.stats[i].docs;
                run.stats[i].docs += 1;
                if is_val_doc(p.seed, k, p.val_per_mille) {
                    run.put(i, &doc, true)?;
                }
            }
        }
    }

    // 3. sources with their own val file
    let mut val_file_note = String::new();
    if !p.no_val {
        for i in 0..n {
            let Some(vs) = &p.sources[i].val_spec else { continue };
            let mut r = SourceReader::open(vs).map_err(io::Error::other)?;
            let bytes = r.total_bytes(vs).max(1);
            let keep = match p.val_tokens {
                Some(c) => ((1000.0 * 3.0 * c as f64 / bytes as f64).ceil() as u64).clamp(1, 1000),
                None => 1000,
            };
            let mut k = 0u64;
            while !run.val_full(i) {
                let Some(doc) = r.next_doc()? else { break };
                let take = mix64(p.seed ^ k) % 1000 < keep;
                k += 1;
                if take {
                    run.put(i, &doc, true)?;
                }
            }
            val_file_note.push_str(&format!("  {}: val from {} (keep {keep} per mille, read {k} docs)\n", p.sources[i].name, vs));
        }
    }

    let stats = run.stats;
    let secs = t0.elapsed().as_secs_f64();
    let mut report = String::new();
    report.push_str(&format!(
        "v42_shards: out {}  tags_only {}  shard_tokens {}  val_per_mille {}  seed {}  max_docs {:?}  val_tokens {:?}  no_val {}  guard {}  max_bit_run {:?}\n",
        p.out.display(),
        p.tags_only,
        p.shard_tokens,
        p.val_per_mille,
        p.seed,
        p.max_docs,
        p.val_tokens,
        p.no_val,
        p.guard.as_ref().map(|g| format!("{} templates x {} names", g.templates.len(), g.n_names())).unwrap_or_else(|| "off".into()),
        p.max_bit_run
    ));
    let train_total: u64 = stats.iter().map(|s| s.train_tokens).sum();
    let wsum: f64 = p.weights.iter().sum();
    report.push_str("sources (name, origin, docs read, train docs/tokens, share of train [target], budget, val docs/tokens [skipped], guard dropped/question docs, bit-run dropped, est. total tokens):\n");
    let (mut td, mut tt, mut tb, mut vd, mut vt) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let mut tsv = String::from(
        "name\torigin\tweight\ttarget_share\tbudget\ttrain_docs\ttrain_tokens\ttrain_share\tval_docs\tval_tokens\tval_skipped\tguard_dropped\tguard_question_docs\tbit_dropped\tdocs_read\texhausted\test_total_tokens\tspec\tval_spec\n",
    );
    for (i, (s, st)) in p.sources.iter().zip(&stats).enumerate() {
        let share = st.train_tokens as f64 / train_total.max(1) as f64;
        let target = if wsum > 0.0 { p.weights.get(i).map(|w| w / wsum) } else { None };
        report.push_str(&format!(
            "  {:<15} {:<8} {:>9} rd {:>9} tr {:>11} tok {:>6.2}% [{}] budget {:>11}  val {:>6} / {:>9} [{}]  guard {}/{}  bits {}  est {}{}\n    tags: {}\n    spec {}{}\n",
            s.name,
            s.origin.name(),
            st.docs,
            st.train_docs,
            st.train_tokens,
            100.0 * share,
            target.map(|t| format!("{:.2}%", 100.0 * t)).unwrap_or_else(|| "-".into()),
            p.budget(i).map(|b| b.to_string()).unwrap_or_else(|| "all".into()),
            st.val_docs,
            st.val_tokens,
            st.val_skipped,
            st.guard_dropped,
            st.guard_core,
            st.bit_dropped,
            st.est_total_tokens(),
            if st.exhausted { " (read to the end)" } else { "" },
            st.tags.line(),
            s.spec,
            s.val_spec.as_ref().map(|v| format!("\n    val  {v}")).unwrap_or_default()
        ));
        tsv.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            s.name,
            s.origin.name(),
            p.weights.get(i).map(|w| w.to_string()).unwrap_or_default(),
            target.map(|t| format!("{t:.6}")).unwrap_or_default(),
            p.budget(i).map(|b| b.to_string()).unwrap_or_default(),
            st.train_docs,
            st.train_tokens,
            share,
            st.val_docs,
            st.val_tokens,
            st.val_skipped,
            st.guard_dropped,
            st.guard_core,
            st.bit_dropped,
            st.docs,
            st.exhausted,
            st.est_total_tokens(),
            s.spec,
            s.val_spec.as_deref().unwrap_or("")
        ));
        td += st.docs;
        tt += st.tokens;
        tb += st.bytes;
        vd += st.val_docs;
        vt += st.val_tokens;
    }
    report.push_str(&val_file_note);
    if let Some(w) = run.w {
        let Writers { train, val, by_source, mut train_tags, mut val_tags, .. } = w;
        let tr = train.finish()?;
        let va = val.finish()?;
        for b in by_source {
            b.finish()?;
        }
        train_tags.flush()?;
        val_tags.flush()?;
        drop((train_tags, val_tags));
        fs::create_dir_all(p.out.join("train"))?;
        fs::create_dir_all(p.out.join("val"))?;
        fs::rename(p.out.join("train_tags.tsv.tmp"), p.out.join("train").join("tags.tsv"))?;
        fs::rename(p.out.join("val_tags.tsv.tmp"), p.out.join("val").join("tags.tsv"))?;
        fs::write(p.out.join("sources.tsv"), tsv)?;
        report.push_str(&format!(
            "total: {td} docs read, {tt} tokens written (train {} in {} shards, val {} in {} shards; val by source {vd} docs, {vt} tokens)\n",
            tr.tokens, tr.shards, va.tokens, va.shards
        ));
    } else {
        report.push_str(&format!("total: {td} docs (tags only, no shards written)\n"));
    }
    report.push_str(&format!(
        "time: {secs:.1}s  ({:.0}k docs/s, {:.1} MB/s of text written, {:.2}M tokens/s)\n",
        td as f64 / 1e3 / secs,
        tb as f64 / 1e6 / secs,
        tt as f64 / 1e6 / secs
    ));
    Ok((stats, report))
}

/// Parse `name=weight,name=weight` (weights >= 0).
pub fn parse_mix(s: &str) -> Result<Vec<(String, f64)>, String> {
    let mut v: Vec<(String, f64)> = Vec::new();
    for part in s.split(',').map(str::trim).filter(|x| !x.is_empty()) {
        let (n, w) = part.split_once('=').ok_or_else(|| format!("bad mix entry {part:?}: want name=weight"))?;
        let w: f64 = w.trim().parse().map_err(|_| format!("bad weight in {part:?}"))?;
        if !(w >= 0.0 && w.is_finite()) {
            return Err(format!("weight must be >= 0 in {part:?}"));
        }
        if v.iter().any(|(m, _)| m == n.trim()) {
            return Err(format!("{n:?} given twice in the mix"));
        }
        v.push((n.trim().to_string(), w));
    }
    if v.is_empty() || v.iter().all(|x| x.1 == 0.0) {
        return Err("the mix has no positive weight".into());
    }
    Ok(v)
}

/// Apply a mix to the sources: every source needs a weight; weight-0 sources are removed. Returns the kept
/// sources, their weights and their budgets (train_tokens * w / sum of weights).
pub fn apply_mix(sources: Vec<SourceDef>, mix: &[(String, f64)], train_tokens: u64) -> Result<(Vec<SourceDef>, Vec<f64>, Vec<Option<u64>>), String> {
    for (n, _) in mix {
        if !sources.iter().any(|s| &s.name == n) {
            return Err(format!("mix names {n:?}, which is not a source"));
        }
    }
    let missing: Vec<&str> = sources.iter().filter(|s| !mix.iter().any(|(n, _)| n == &s.name)).map(|s| s.name.as_str()).collect();
    if !missing.is_empty() {
        return Err(format!("no mix weight for: {}", missing.join(", ")));
    }
    let wsum: f64 = mix.iter().map(|x| x.1).sum();
    let (mut kept, mut ws, mut bs) = (Vec::new(), Vec::new(), Vec::new());
    for s in sources {
        let w = mix.iter().find(|(n, _)| n == &s.name).unwrap().1;
        if w > 0.0 {
            bs.push(Some((train_tokens as f64 * w / wsum).round() as u64));
            ws.push(w);
            kept.push(s);
        }
    }
    Ok((kept, ws, bs))
}

/// Write `out/SHA256SUMS`: the sha256 of every file under `out` but itself, manifest.txt and the
/// building marker (sha256sum format, `/` paths, sorted), hashed on `threads` threads. Returns (files, bytes).
pub fn write_sha256sums(out: &Path, threads: usize) -> io::Result<(usize, u64)> {
    fn walk(dir: &Path, base: &Path, v: &mut Vec<(String, PathBuf, u64)>) -> io::Result<()> {
        for e in fs::read_dir(dir)? {
            let e = e?;
            let p = e.path();
            if e.file_type()?.is_dir() {
                walk(&p, base, v)?;
            } else {
                let rel = p.strip_prefix(base).unwrap().to_string_lossy().replace('\\', "/");
                if !matches!(rel.as_str(), "SHA256SUMS" | "manifest.txt" | ".building") {
                    v.push((rel, p.clone(), e.metadata()?.len()));
                }
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(out, out, &mut files)?;
    files.sort();
    let next = std::sync::atomic::AtomicUsize::new(0);
    let sums: Vec<std::sync::Mutex<Option<io::Result<String>>>> = files.iter().map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= files.len() {
                    break;
                }
                *sums[i].lock().unwrap() = Some(sha256_file(&files[i].1));
            });
        }
    });
    let mut text = String::new();
    let mut bytes = 0u64;
    for ((rel, _, len), s) in files.iter().zip(sums) {
        text.push_str(&format!("{}  {rel}\n", s.into_inner().unwrap().unwrap()?));
        bytes += len;
    }
    fs::write(out.join("SHA256SUMS"), text)?;
    Ok((files.len(), bytes))
}


/// Prepare the output directory for a build (see RESUME). Returns false when it is already finished.
pub fn prepare_out(out: &Path) -> io::Result<bool> {
    if out.join("manifest.txt").exists() {
        return Ok(false);
    }
    if out.exists() {
        let marker = out.join(".building");
        let empty = fs::read_dir(out)?.next().is_none();
        if marker.exists() {
            fs::remove_dir_all(out)?;
        } else if !empty {
            return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("{} is not empty and is not an unfinished v42 build", out.display())));
        }
    }
    fs::create_dir_all(out)?;
    File::create(out.join(".building"))?;
    Ok(true)
}

/// Write the manifest and remove the building marker.
pub fn finish_out(out: &Path, report: &str) -> io::Result<()> {
    fs::write(out.join("manifest.txt"), report)?;
    fs::remove_file(out.join(".building"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shards::ShardSet;
    use crate::wording_guard::parse_templates;

    fn tmp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ava_v42sh_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn ntok(s: &ShardSet) -> u64 {
        (0..s.len()).map(|i| s.load_shard(i).unwrap().len() as u64).sum::<u64>()
    }

    /// Decoded documents of a shard set.
    fn texts(s: &ShardSet, bpe: &Bpe) -> Vec<String> {
        let mut all = Vec::new();
        for i in 0..s.len() {
            all.extend(s.load_shard(i).unwrap().to_vec());
        }
        all.split(|&t| t == EOS).filter(|d| !d.is_empty()).map(|d| bpe.decode(&crate::numbers::strip_marks(d))).collect()
    }

    #[test]
    fn source_defs_parse() {
        let s = SourceDef::parse("cornell:dialogue:C:/data/x.docs").unwrap();
        assert_eq!((s.name.as_str(), s.origin, s.spec.as_str(), s.val_spec.as_deref()), ("cornell", Origin::Dialogue, "C:/data/x.docs", None));
        let s = SourceDef::parse("childes:speech:C:/b/childes_train.docs|val=C:/b/childes_dev.docs").unwrap();
        assert_eq!((s.spec.as_str(), s.val_spec.as_deref()), ("C:/b/childes_train.docs", Some("C:/b/childes_dev.docs")));
        assert_eq!(SourceDef::parse(&s.to_arg()).unwrap(), s);
        assert!(SourceDef::parse("Bad:dialogue:x").is_err());
        assert!(SourceDef::parse("a:radio:x").is_err());
        assert!(SourceDef::parse("a:book").is_err());
        assert!(SourceDef::parse("a:book:x|val=").is_err());
    }

    #[test]
    fn mix_to_budgets() {
        let src = |n: &str| SourceDef::parse(&format!("{n}:web:x.docs")).unwrap();
        let mix = parse_mix("a=3, b=1,c=0").unwrap();
        let (kept, ws, bs) = apply_mix(vec![src("a"), src("b"), src("c")], &mix, 1000).unwrap();
        assert_eq!(kept.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), vec!["a", "b"]);
        assert_eq!((ws, bs), (vec![3.0, 1.0], vec![Some(750), Some(250)]));
        assert!(apply_mix(vec![src("a"), src("b")], &parse_mix("a=1").unwrap(), 10).is_err(), "b has no weight");
        assert!(apply_mix(vec![src("a")], &parse_mix("a=1,z=2").unwrap(), 10).is_err(), "z is not a source");
        assert!(parse_mix("a=-1").is_err() && parse_mix("a=0").is_err() && parse_mix("a=1,a=2").is_err());
    }

    #[test]
    fn builds_per_source_val_and_tags() {
        let dir = tmp("plain");
        let lines = dir.join("lines.txt");
        let body: String = (0..400).map(|i| format!("line number {i} about rocks\n")).collect();
        fs::write(&lines, body).unwrap();
        let docs = dir.join("conv.docs");
        let mut dw = crate::docs::DocsWriter::create(&docs).unwrap();
        for i in 0..300 {
            dw.write_doc(&format!("A: hello {i}\nB: hi there")).unwrap();
        }
        dw.finish().unwrap();
        let out = dir.join("out");
        assert!(prepare_out(&out).unwrap());
        let p = BuildParams::new(
            out.clone(),
            vec![
                SourceDef::parse(&format!("lines:sentence:{}", lines.display())).unwrap(),
                SourceDef::parse(&format!("conv:dialogue:{}", docs.display())).unwrap(),
            ],
            2000,
            100,
            7,
        );
        let bpe = Bpe::from_merges(Vec::new()).unwrap();
        let (st, rep) = build(&p, &bpe).unwrap();
        let (nfiles, _) = write_sha256sums(&out, 2).unwrap();
        finish_out(&out, &rep).unwrap();
        assert!(!prepare_out(&out).unwrap(), "finished output is skipped");
        assert_eq!(st[0].docs, 400);
        assert_eq!(st[1].docs, 300);
        // the per-source split follows the doc-hash rule on the source's own doc index
        let want0 = (0..400u64).filter(|&k| is_val_doc(7, k, 100)).count() as u64;
        let want1 = (0..300u64).filter(|&k| is_val_doc(7, k, 100)).count() as u64;
        assert_eq!((st[0].val_docs, st[1].val_docs), (want0, want1));
        let v0 = ShardSet::open(out.join("val_by_source/lines")).unwrap();
        let v1 = ShardSet::open(out.join("val_by_source/conv")).unwrap();
        let all = ShardSet::open(out.join("val")).unwrap();
        assert_eq!(ntok(&v0), st[0].val_tokens);
        assert_eq!(ntok(&v1), st[1].val_tokens);
        assert_eq!(ntok(&all), st[0].val_tokens + st[1].val_tokens);
        // tag rows: one per document, starts are running sums of lengths
        let tags = fs::read_to_string(out.join("train/tags.tsv")).unwrap();
        let rows: Vec<Vec<&str>> = tags.lines().filter(|l| !l.starts_with('#')).map(|l| l.split('\t').collect()).collect();
        assert_eq!(rows.len() as u64, 700 - want0 - want1);
        let mut pos = 0u64;
        for r in &rows {
            assert_eq!(r[0].parse::<u64>().unwrap(), pos);
            pos += r[1].parse::<u64>().unwrap();
            assert!(r[3] == "sentence" || r[3] == "dialogue");
        }
        let train = ShardSet::open(out.join("train")).unwrap();
        assert_eq!(ntok(&train), pos);
        // the decoded first val doc of a source is one of its documents
        let t = texts(&v1, &bpe);
        assert!(t[0].starts_with("A: hello ") && t[0].ends_with("\nB: hi there"), "{:?}", t[0]);
        // the sums cover every file but themselves and the manifest, and match
        let sums = fs::read_to_string(out.join("SHA256SUMS")).unwrap();
        assert_eq!(sums.lines().count(), nfiles);
        assert!(sums.contains("  sources.tsv\n") && sums.contains("  train/tags.tsv\n") && !sums.contains("manifest.txt"));
        let line = sums.lines().find(|l| l.ends_with("  sources.tsv")).unwrap();
        assert_eq!(&line[..64], sha256_file(out.join("sources.tsv")).unwrap());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn budgets_val_caps_val_files_and_the_guard() {
        let dir = tmp("mix");
        let bpe = Bpe::from_merges(Vec::new()).unwrap();
        // a: 2000 docs of ~20 tokens; b: 300 docs; c: a train file and its own val file
        let mk = |name: &str, n: usize, f: &dyn Fn(usize) -> String| {
            let path = dir.join(name);
            let mut w = crate::docs::DocsWriter::create(&path).unwrap();
            for i in 0..n {
                w.write_doc(&f(i)).unwrap();
            }
            w.finish().unwrap();
            path
        };
        let a = mk("a.docs", 2000, &|i| format!("alpha document {i:05}"));
        let b = mk("b.docs", 300, &|i| if i == 5 { "Q: What is the capital of France?\nA: Paris".into() } else if i == 6 { "Quiz: What is the capital of France? Paris.".into() } else if i == 8 { "bits 1 0 1 1 0 0 1 0 1 1 0 1 0 1 end".into() } else if i == 9 { "short 1 0 1 1 0 0 1 0 1 1 0 1 end".into() } else { format!("beta {i:04}") });
        let c = mk("c.docs", 200, &|i| format!("gamma train {i:04}"));
        let cv = mk("cv.docs", 500, &|i| format!("gamma dev {i:04}"));
        let sources = || {
            vec![
                SourceDef::parse(&format!("a:web:{}", a.display())).unwrap(),
                SourceDef::parse(&format!("b:wiki:{}", b.display())).unwrap(),
                SourceDef::parse(&format!("c:speech:{}|val={}", c.display(), cv.display())).unwrap(),
            ]
        };
        let guard = || Some(WordingGuard::new(parse_templates("Q: What is the capital of {}?\\nA:").unwrap(), ["France".to_string()]));
        let go_k = |tag: &str, train_tokens: u64, val_tokens: Option<u64>, no_val: bool, keep_q: bool| {
            let out = dir.join(tag);
            assert!(prepare_out(&out).unwrap());
            let (srcs, ws, bs) = apply_mix(sources(), &parse_mix("a=1,b=1,c=1").unwrap(), train_tokens).unwrap();
            let mut p = BuildParams::new(out.clone(), srcs, 100_000, 50, 3);
            (p.weights, p.budgets, p.val_tokens, p.no_val, p.guard, p.guard_keep_questions) = (ws, bs, val_tokens, no_val, guard(), keep_q);
            p.max_bit_run = Some(12);
            let (st, rep) = build(&p, &bpe).unwrap();
            finish_out(&out, &rep).unwrap();
            (st, out)
        };
        let go = |tag: &str, t: u64, v: Option<u64>, nv: bool| go_k(tag, t, v, nv, false);
        // budgets: a is cut near its budget, b and c run out and give all they have
        let (st, out) = go("big", 3 * 8000, Some(400), false);
        assert!(st[0].train_tokens >= 8000 && st[0].train_tokens < 8000 + 40, "{}", st[0].train_tokens);
        assert!(st[1].exhausted && st[2].exhausted && !st[0].exhausted);
        // the guard drops the test prompt and the bare test question, and keeps the doc index running
        assert_eq!((st[1].guard_dropped, st[1].guard_core), (2, 2));
        // a run of 14 single 0/1 words is dropped, a run of 12 is kept
        assert_eq!(st[1].bit_dropped, 1);
        let tb = texts(&ShardSet::open(out.join("train")).unwrap(), &bpe);
        let vb = texts(&ShardSet::open(out.join("val_by_source/b")).unwrap(), &bpe);
        assert!(!tb.iter().chain(&vb).any(|t| t.contains("What is the capital")));
        assert!(tb.iter().chain(&vb).any(|t| t == "beta 0007"), "the next doc is still there");
        assert!(!tb.iter().chain(&vb).any(|t| t.starts_with("bits ")) && tb.iter().chain(&vb).any(|t| t.starts_with("short ")));
        // keeping bare questions: only the framed prompt is dropped
        let (stk, outk) = go_k("keepq", 3 * 8000, Some(400), false, true);
        assert_eq!((stk[1].guard_dropped, stk[1].guard_core), (1, 2));
        let tk = texts(&ShardSet::open(outk.join("train")).unwrap(), &bpe);
        let vk = texts(&ShardSet::open(outk.join("val_by_source/b")).unwrap(), &bpe);
        assert!(!tk.iter().chain(&vk).any(|t| t.contains("Q: What is the capital")));
        assert!(tk.iter().chain(&vk).any(|t| t.starts_with("Quiz:")));
        // a val doc is never trained on; val stays within the cap (+ one doc)
        let va = texts(&ShardSet::open(out.join("val_by_source/a")).unwrap(), &bpe);
        assert!(!va.is_empty() && va.iter().all(|t| !tb.contains(t)));
        assert!(st[0].val_tokens >= 400 && st[0].val_tokens < 400 + 40);
        // c's val comes from its own file, thinned evenly, and c's train has all of c.docs
        let vc = texts(&ShardSet::open(out.join("val_by_source/c")).unwrap(), &bpe);
        assert!(vc.iter().all(|t| t.starts_with("gamma dev")) && st[2].val_tokens >= 400);
        assert!(vc.iter().any(|t| t.as_str() > "gamma dev 0100"), "thinned, not just the first docs");
        assert_eq!(st[2].train_docs, 200);
        // a small budget: same val slices (the val read goes on past the budget), a smaller train
        let (st2, out2) = go("small", 3 * 2000, Some(400), false);
        assert!(st2[0].train_tokens < st[0].train_tokens);
        let va2 = texts(&ShardSet::open(out2.join("val_by_source/a")).unwrap(), &bpe);
        assert_eq!(va2, va);
        // its train is the start of the big build's train for that source, in order
        let ta: Vec<String> = tb.iter().filter(|t| t.starts_with("alpha")).cloned().collect();
        let ta2: Vec<String> = texts(&ShardSet::open(out2.join("train")).unwrap(), &bpe).into_iter().filter(|t| t.starts_with("alpha")).collect();
        assert_eq!(&ta[..ta2.len()], &ta2[..]);
        // no val: nothing in val, and still no val doc in train
        let (st3, out3) = go("noval", 3 * 8000, Some(400), true);
        assert_eq!(st3.iter().map(|s| s.val_docs).sum::<u64>(), 0);
        let tb3 = texts(&ShardSet::open(out3.join("train")).unwrap(), &bpe);
        assert!(va.iter().all(|t| !tb3.contains(t)));
        let _ = fs::remove_dir_all(&dir);
    }
}
