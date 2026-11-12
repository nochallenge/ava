// TESTING / DESIGN ONLY
//! v42_shards: v42 sources into train/val shards + a val slice per source + tag sidecars, with a mixture.
//!
//!   v42_shards --out DIR [--bpe FILE] [--shard-tokens N] [--val-per-mille V] [--seed S]
//!              [--max-docs-per-source M] [--tags-only] --src NAME:ORIGIN:SPEC[|val=SPEC] [--src ...]
//!              [--mix NAME=W,...] [--train-tokens T] [--budget NAME=N ...] [--val-tokens N] [--no-val]
//!              [--no-guard] [--guard-keep-questions] [--templates FILE] [--wikidata DIR] [--threads 2]
//!              [--max-bit-run N | --no-bit-guard]
//!   v42_shards --base [...]   the v42 base sources (listed below), out data/nn/shards-v42-base
//!   v42_shards --full [...]   the base sources + the new reading (below), out data/nn/shards-v42-full,
//!                             with FULL_MIX, --train-tokens 1G and --val-tokens 1M unless given
//!
//! See `v42_shards` (the module) for the order, split, mixture, guard, sidecar and resume rules. Defaults:
//! --bpe data/nn/bpe-4096.txt, --shard-tokens 50M, --val-per-mille 5, --seed ava_data's split seed.
//! --mix needs --train-tokens: each source's train budget is T * weight / sum of weights; every source
//! needs a weight, and weight 0 leaves a source out. --budget NAME=N sets one source's budget directly.
//! --val-tokens caps each source's val slice; --no-val writes no val (proxy runs). The guard (on unless
//! --no-guard) reads --templates (default data/nn/evalsets_v42/test_templates.txt, the only file of that
//! folder it opens) and fills them with every Wikidata (--wikidata) and hand-written fact name; a document
//! holding a filled test question is dropped (--guard-keep-questions: only one with the "Q:" frame).
//! --max-bit-run N (default 12) drops a document with a run of more than N single 0/1 words (number
//! tables, bit vectors), which can share a 13-word gram with the parity test items; --no-bit-guard keeps them.
//! --tags-only scores tags and prints the report without writing shards (and needs no --out).
//! After the build: bpe.txt, SHA256SUMS (hashed on --threads threads, default 2), then manifest.txt.
//!
//! --base adds these sources (all under the repo's data folder, `ava_data::DATA_DIR`):
//!   arc        sentence  nn/v42/arc_packed.docs          (ARC_Corpus packed by topic, deduped)
//!   tatoeba    sentence  eng_sentences.tsv@tab:2
//!   school     dialogue  school/dialogue.tsv@tab:2
//!   cornell    dialogue  nn/v42/dialogue_cornell.docs    (whole conversations; replaces cornell_lines)
//!   dailydialog dialogue nn/v42/dialogue_dd.docs
//!   books      book      pg19/train@list:nn/shards-v41-mix/books.tsv   (v41's 337 training books only)
//!   facts      practice  nn/v42/facts_restate.docs       (restated facts only; when the file exists)
//! --full adds to those:
//!   fineweb    web       nn/docs/fineweb_edu_000.docs    (FineWeb-Edu, ODC-By)
//!   simplewiki wiki      nn/docs/simplewiki.docs         (Simple English Wikipedia, CC BY-SA 4.0)
//!   childes, bnc_spoken, switchboard (speech), open_subtitles (dialogue), gutenberg (book):
//!              nn/docs/babylm/NAME_train.docs, val from nn/docs/babylm/NAME_dev.docs (BabyLM's dev files)
//! --lessons [FILE] (off by default) adds the owner's lessons as the LAST source, `lessons:qa:FILE` (default
//!   data/nn/docs/lessons.docs, written by v42_lessons). It takes no mix weight: under --mix it gets no budget
//!   (all of it, once), so every other source's budget and share target stay exactly as without it.
//! --lessons-repeat N (default 1, at most 8) reads the lessons N times, as sources lessons, lessons_r2, ...
//!   The owner chose 4 for v42 (10-29): up to about 4 repeats of the same text trains about as well as
//!   new text (Muennighoff et al. 2023). Every copy of a lesson lands on the same side of the val split.
//! --reason [DIR] (off by default) adds the seven Wave 5 reasoning families written by v42_reason, each one
//!   `reason_<family>:practice:DIR/reason_<family>.docs` (default DIR data/nn/docs; every file must exist).
//!   Under --mix, each family not already named in the mix gets its REASON_MIX weight (below, in the same
//!   percent units as FULL_MIX): search 0.6, causal 0.4, bayes 0.3, rules 0.3, state 0.4, analogy 0.2,
//!   library 0.3, 2.5 in all. They are added to the mix, so with FULL_MIX (sum 100) the reasoning text is
//!   2.5 / 102.5 = 2.44% of the train tokens and every other share shrinks by 100 / 102.5. A family holding
//!   less than its budget gives all it has. Without --mix each family goes in whole.

use ava_data::bpe::Bpe;
use ava_data::fact_ledger::{all_facts, guard_names, KEYS_DIR, TEMPLATES_FILE, WIKIDATA_DIR};
use ava_data::input::parse_count;
use ava_data::shards::{DEFAULT_SHARD_TOKENS, DEFAULT_SPLIT_SEED, DEFAULT_VAL_PER_MILLE};
use ava_data::v42_shards::{apply_mix, build, finish_out, parse_mix, prepare_out, write_sha256sums, BuildParams, SourceDef};
use ava_data::wikidata::Wikidata;
use ava_data::wording_guard::WordingGuard;
use std::io;
use std::path::PathBuf;
use std::time::Instant;

const DATA: &str = ava_data::DATA_DIR;

/// The first hand-set v42 mixture: shares of the train tokens, in percent (see the v42 notes).
pub const FULL_MIX: &str = "fineweb=50.6,arc=16.25,simplewiki=10.4,childes=6,books=4.5,gutenberg=4,open_subtitles=3.5,tatoeba=2.5,\
bnc_spoken=1.2,cornell=0.6,switchboard=0.24,dailydialog=0.11,facts=0.06,school=0.04";

/// The Wave 5 reasoning families (--reason) and their mix weights, in FULL_MIX's percent units.
pub const REASON_MIX: &str = "reason_search=0.6,reason_causal=0.4,reason_bayes=0.3,reason_rules=0.3,reason_state=0.4,\
reason_analogy=0.2,reason_library=0.3";

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn base_sources() -> Vec<String> {
    let mut v = vec![
        format!("arc:sentence:{DATA}/nn/v42/arc_packed.docs"),
        format!("tatoeba:sentence:{DATA}/eng_sentences.tsv@tab:2"),
        format!("school:dialogue:{DATA}/school/dialogue.tsv@tab:2"),
        format!("cornell:dialogue:{DATA}/nn/v42/dialogue_cornell.docs"),
        format!("dailydialog:dialogue:{DATA}/nn/v42/dialogue_dd.docs"),
        format!("books:book:{DATA}/pg19/train@list:{DATA}/nn/shards-v41-mix/books.tsv"),
    ];
    if std::path::Path::new(&format!("{DATA}/nn/v42/facts_restate.docs")).exists() {
        v.push(format!("facts:practice:{DATA}/nn/v42/facts_restate.docs"));
    }
    v
}

fn full_sources() -> Vec<String> {
    let mut v = vec![format!("fineweb:web:{DATA}/nn/docs/fineweb_edu_000.docs"), format!("simplewiki:wiki:{DATA}/nn/docs/simplewiki.docs")];
    v.extend(base_sources());
    for (name, origin) in [("childes", "speech"), ("bnc_spoken", "speech"), ("switchboard", "speech"), ("open_subtitles", "dialogue"), ("gutenberg", "book")] {
        v.push(format!("{name}:{origin}:{DATA}/nn/docs/babylm/{name}_train.docs|val={DATA}/nn/docs/babylm/{name}_dev.docs"));
    }
    v
}

fn run() -> io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut out: Option<PathBuf> = None;
    let mut bpe_path = PathBuf::from(format!("{DATA}/nn/bpe-4096.txt"));
    let mut shard_tokens = DEFAULT_SHARD_TOKENS;
    let mut val_per_mille = DEFAULT_VAL_PER_MILLE;
    let mut seed = DEFAULT_SPLIT_SEED;
    let mut max_docs = None;
    let mut tags_only = false;
    let mut no_val = false;
    let mut guard_on = true;
    let mut keep_q = false;
    let mut max_bit_run: Option<usize> = Some(12);
    let mut full = false;
    let mut mix: Option<String> = None;
    let mut train_tokens: Option<u64> = None;
    let mut val_tokens: Option<u64> = None;
    let mut budgets_given: Vec<(String, u64)> = Vec::new();
    let mut templates = PathBuf::from(KEYS_DIR).join(TEMPLATES_FILE);
    let mut wd_dir = PathBuf::from(WIKIDATA_DIR);
    let mut threads = 2usize;
    let mut lessons: Option<String> = None;
    let mut lessons_repeat = 1usize;
    let mut reason: Option<String> = None;
    let mut specs: Vec<String> = Vec::new();
    let mut i = 0;
    let bad = |m: String| io::Error::new(io::ErrorKind::InvalidInput, m);
    while i < args.len() {
        let a = args[i].as_str();
        let flag = |v: &mut bool| {
            *v = true;
        };
        match a {
            "--tags-only" => flag(&mut tags_only),
            "--no-val" => flag(&mut no_val),
            "--no-guard" => guard_on = false,
            "--guard-keep-questions" => flag(&mut keep_q),
            "--no-bit-guard" => max_bit_run = None,
            "--lessons" => {
                // an optional FILE: the next argument when it is not a flag
                match args.get(i + 1).filter(|v| !v.starts_with("--")) {
                    Some(f) => {
                        lessons = Some(f.clone());
                        i += 1;
                    }
                    None => lessons = Some(format!("{DATA}/nn/docs/lessons.docs")),
                }
            }
            "--lessons-repeat" => {
                i += 1;
                lessons_repeat = args.get(i).and_then(|v| v.parse().ok()).filter(|&n: &usize| (1..=8).contains(&n)).ok_or_else(|| bad("--lessons-repeat needs a count from 1 to 8".into()))?;
            }
            "--reason" => {
                // an optional DIR: the next argument when it is not a flag
                match args.get(i + 1).filter(|v| !v.starts_with("--")) {
                    Some(d) => {
                        reason = Some(d.clone());
                        i += 1;
                    }
                    None => reason = Some(format!("{DATA}/nn/docs")),
                }
            }
            "--base" => {
                specs.extend(base_sources());
                if out.is_none() {
                    out = Some(PathBuf::from(format!("{DATA}/nn/shards-v42-base")));
                }
            }
            "--full" => {
                full = true;
                specs.extend(full_sources());
                if out.is_none() {
                    out = Some(PathBuf::from(format!("{DATA}/nn/shards-v42-full")));
                }
            }
            _ => {
                let v = args.get(i + 1).cloned().ok_or_else(|| bad(format!("{a} needs a value")))?;
                match a {
                    "--out" => out = Some(PathBuf::from(v)),
                    "--bpe" => bpe_path = PathBuf::from(v),
                    "--shard-tokens" => shard_tokens = parse_count(&v).map_err(bad)? as usize,
                    "--val-per-mille" => val_per_mille = parse_count(&v).map_err(bad)? as u32,
                    "--seed" => seed = parse_count(&v).map_err(bad)?,
                    "--max-docs-per-source" => max_docs = Some(parse_count(&v).map_err(bad)?),
                    "--src" => specs.push(v),
                    "--mix" => mix = Some(v),
                    "--train-tokens" => train_tokens = Some(parse_count(&v).map_err(bad)?),
                    "--val-tokens" => val_tokens = Some(parse_count(&v).map_err(bad)?),
                    "--budget" => {
                        let (n, b) = v.split_once('=').ok_or_else(|| bad(format!("--budget wants NAME=N, got {v:?}")))?;
                        budgets_given.push((n.to_string(), parse_count(b).map_err(bad)?));
                    }
                    "--templates" => templates = PathBuf::from(v),
                    "--wikidata" => wd_dir = PathBuf::from(v),
                    "--threads" => threads = (parse_count(&v).map_err(bad)? as usize).max(1),
                    "--max-bit-run" => max_bit_run = Some(parse_count(&v).map_err(bad)? as usize),
                    _ => return Err(bad(format!("unknown argument {a}"))),
                }
                i += 1;
            }
        }
        i += 1;
    }
    if full {
        mix.get_or_insert_with(|| FULL_MIX.to_string());
        train_tokens.get_or_insert(1_000_000_000);
        val_tokens.get_or_insert(1_000_000);
    }
    if let Some(dir) = &reason {
        let named: Vec<String> = match &mix {
            Some(m) => parse_mix(m).map_err(bad)?.into_iter().map(|x| x.0).collect(),
            None => Vec::new(),
        };
        for (name, w) in parse_mix(REASON_MIX).map_err(bad)? {
            let path = format!("{dir}/{name}.docs");
            if !std::path::Path::new(&path).exists() {
                return Err(bad(format!("--reason: {path} is missing (write it with v42_reason first)")));
            }
            if !specs.iter().any(|s| s.starts_with(&format!("{name}:"))) {
                specs.push(format!("{name}:practice:{path}"));
            }
            if let Some(m) = &mut mix {
                if !named.contains(&name) {
                    m.push_str(&format!(",{name}={w}"));
                }
            }
        }
    }
    let mut sources: Vec<SourceDef> = specs.iter().map(|s| SourceDef::parse(s)).collect::<Result<_, _>>().map_err(bad)?;
    if sources.is_empty() && lessons.is_none() {
        return Err(bad("no sources (--src NAME:ORIGIN:SPEC, --base, --full or --lessons)".into()));
    }
    let (mut weights, mut budgets) = (Vec::new(), vec![None; sources.len()]);
    if let Some(m) = &mix {
        let t = train_tokens.ok_or_else(|| bad("--mix needs --train-tokens".into()))?;
        let (kept, ws, bs) = apply_mix(sources, &parse_mix(m).map_err(bad)?, t).map_err(bad)?;
        (sources, weights, budgets) = (kept, ws, bs);
    }
    if let Some(f) = &lessons {
        if sources.iter().any(|s| s.name == "lessons") {
            return Err(bad("--lessons given, but a source is already named lessons".into()));
        }
        // Each repeat is the same file read again as its own source; a document's split depends only on its
        // place in its source, so every copy of a lesson lands on the same side (no lesson leaks into val).
        for r in 1..=lessons_repeat {
            let name = if r == 1 { "lessons".to_string() } else { format!("lessons_r{r}") };
            sources.push(SourceDef::parse(&format!("{name}:qa:{f}")).map_err(bad)?);
            budgets.push(None);
        }
    } else if lessons_repeat > 1 {
        return Err(bad("--lessons-repeat needs --lessons".into()));
    }
    for (n, b) in &budgets_given {
        let k = sources.iter().position(|s| &s.name == n).ok_or_else(|| bad(format!("--budget names {n:?}, which is not a source")))?;
        budgets[k] = Some(*b);
    }
    let out = match (out, tags_only) {
        (Some(o), _) => o,
        (None, true) => PathBuf::from("."),
        (None, false) => return Err(bad("--out is required".into())),
    };
    let guard = if guard_on {
        let wd = Wikidata::load(&wd_dir)?;
        Some(WordingGuard::load(&templates, guard_names(&all_facts(), &wd))?)
    } else {
        None
    };
    if !tags_only && !prepare_out(&out)? {
        println!("{} is finished (manifest.txt exists); skipping", out.display());
        return Ok(());
    }
    let bpe = Bpe::load(&bpe_path)?;
    let mut p = BuildParams::new(out.clone(), sources, shard_tokens, val_per_mille, seed);
    (p.max_docs, p.tags_only, p.budgets, p.weights, p.val_tokens, p.no_val, p.guard) = (max_docs, tags_only, budgets, weights, val_tokens, no_val, guard);
    (p.guard_keep_questions, p.max_bit_run) = (keep_q, max_bit_run);
    let (_, mut report) = build(&p, &bpe)?;
    report.insert_str(
        0,
        &format!(
            "tokenizer {} + length marks (vocab 4112)\nmix {}  train_tokens {:?}\nguard templates {}\n",
            bpe_path.display(),
            mix.as_deref().unwrap_or("none (whole sources)"),
            train_tokens,
            if guard_on { templates.display().to_string() } else { "off".into() }
        ),
    );
    if !tags_only {
        bpe.save(out.join("bpe.txt"))?;
        let t = Instant::now();
        let (files, bytes) = write_sha256sums(&out, threads)?;
        let secs = t.elapsed().as_secs_f64();
        report.push_str(&format!(
            "SHA256SUMS: {files} files, {:.1} MB hashed in {secs:.1}s ({:.0} MB/s on {threads} threads)\n",
            bytes as f64 / 1e6,
            bytes as f64 / 1e6 / secs.max(1e-9)
        ));
    }
    print!("{report}");
    if !tags_only {
        finish_out(&out, &report)?;
    }
    Ok(())
}
