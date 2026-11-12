// TESTING / DESIGN ONLY
//! v42_facts: the facts' statuses, restatement documents, and the fact ledger (see `fact_ledger`).
//!
//!   v42_facts [--out DIR] [--seed S] [--dev-percent P] [--rounds R] [--per-doc K]
//!             [--keys-dir DIR] [--wikidata DIR] [--no-wikidata-elements]
//!             [--limit N] [--threads T] [--src NAME:SPEC ...] [--no-ledger]
//!
//! Reads from --keys-dir (default data/nn/evalsets_v42) only restate_fact_keys.txt,
//! control_fact_keys.txt and test_templates.txt, and the Wikidata tables from --wikidata (default
//! data/downloads/wikidata). Writes into --out (default data/nn/v42):
//!   facts_split.tsv      every hand-written fact: id, relation, subject, object, status, matched test key
//!                        (status: restate / control / unlisted for capitals and elements, train / dev for
//!                        animals and planets)
//!   restated_keys.tsv    every restated test key: key, where its names came from, subject and object forms
//!   facts_restate.docs   restatements of the restated facts (`rounds` rounds, `per_doc` sentences a doc);
//!                        not written when a sentence holds a filled test template or mentions a control fact
//!   facts_ledger.tsv     per fact and source: mentions and statements in her text (+ a report)
//! Ledger sources default to her old reading (v39-v41): ARC_Corpus, Tatoeba, Cornell lines, the
//! school dialogue, and v41's 337 training books (only those are opened). --limit N reads the first N
//! documents of each source (a trial; outputs get a `_trial` suffix). Sources are spread over
//! --threads workers (default 2).

use ava_data::docs::DocsWriter;
use ava_data::docs::SourceReader;
use ava_data::fact_ledger::{
    all_facts, check_restatements, guard_names, plan, restatement_docs, restatements, KeyLists, Ledger, Rel, DEFAULT_DEV_PERCENT, DEFAULT_SEED,
    KEYS_DIR, TEMPLATES_FILE, WIKIDATA_DIR,
};
use ava_data::input::parse_count;
use ava_data::wikidata::Wikidata;
use ava_data::wording_guard::WordingGuard;
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

fn num(args: &[String], flag: &str, d: u64) -> io::Result<u64> {
    arg(args, flag).map(|v| parse_count(&v).map_err(io::Error::other)).unwrap_or(Ok(d))
}

fn default_sources() -> Vec<(String, String)> {
    vec![
        ("arc".into(), format!("{DATA}/ai2-arc/ARC-V1-Feb2018-2/ARC_Corpus.txt")),
        ("tatoeba".into(), format!("{DATA}/eng_sentences.tsv@tab:2")),
        ("cornell".into(), format!("{DATA}/cornell_lines.txt@sep: +++$+++ :-1")),
        ("school".into(), format!("{DATA}/school/dialogue.tsv@tab:2")),
        ("books".into(), format!("{DATA}/pg19/train@list:{DATA}/nn/shards-v41-mix/books.tsv")),
    ]
}

fn run() -> io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let out = arg(&args, "--out").map(PathBuf::from).unwrap_or(PathBuf::from(format!("{DATA}/nn/v42")));
    let seed = num(&args, "--seed", DEFAULT_SEED)?;
    let dev_percent = num(&args, "--dev-percent", DEFAULT_DEV_PERCENT)?;
    let rounds = num(&args, "--rounds", 10)? as usize;
    let per_doc = num(&args, "--per-doc", 8)? as usize;
    let threads = num(&args, "--threads", 2)?.max(1) as usize;
    let limit = arg(&args, "--limit").map(|v| parse_count(&v).map_err(io::Error::other)).transpose()?;
    let keys_dir = arg(&args, "--keys-dir").map(PathBuf::from).unwrap_or(PathBuf::from(KEYS_DIR));
    let wd_dir = arg(&args, "--wikidata").map(PathBuf::from).unwrap_or(PathBuf::from(WIKIDATA_DIR));
    let wd_elements = !args.iter().any(|a| a == "--no-wikidata-elements");
    let suffix = if limit.is_some() { "_trial" } else { "" };
    std::fs::create_dir_all(&out)?;
    let facts = all_facts();
    let keys = KeyLists::load(&keys_dir)?;
    let wd = Wikidata::load(&wd_dir)?;
    let p = plan(facts.clone(), &keys, &wd, seed, dev_percent, wd_elements);

    // the statuses
    let mut split = String::from("id\trelation\tsubject\tobject\tstatus\ttest_key\n");
    let mut counts = [[0usize; 5]; 4];
    for (f, s, k) in &p.own {
        counts[f.rel as usize][*s as usize] += 1;
        split.push_str(&format!("{}\t{}\t{}\t{}\t{}\t{}\n", f.id(), f.rel.name(), f.a.join("|"), f.b.join("|"), s.name(), k.as_deref().unwrap_or("")));
    }
    std::fs::write(out.join("facts_split.tsv"), split)?;
    let mut report = format!(
        "v42_facts: seed {seed}  dev_percent {dev_percent}  rounds {rounds}  per_doc {per_doc}  limit {limit:?}  wikidata elements {wd_elements}\n\
         test keys: {} restate, {} control (from {}); wikidata: {} countries, {} elements\n",
        keys.restate.len(),
        keys.control.len(),
        keys_dir.display(),
        wd.countries.len(),
        wd.elements.len()
    );
    report.push_str("hand-written facts by status (restate control unlisted train dev):\n");
    for rel in [Rel::Capital, Rel::Symbol, Rel::Class, Rel::Planet] {
        let c = counts[rel as usize];
        report.push_str(&format!("  {:<8} {:>4} {:>4} {:>4} {:>4} {:>4}\n", rel.name(), c[0], c[1], c[2], c[3], c[4]));
    }

    // the restated keys
    let mut ktsv = String::from("key\tsource\tsubject\tobject\n");
    let mut by_src: std::collections::BTreeMap<(String, &str), usize> = std::collections::BTreeMap::new();
    for r in &p.keys {
        ktsv.push_str(&format!("{}\t{}\t{}\t{}\n", r.key, r.source, r.fact.a.join("|"), r.fact.b.join("|")));
        *by_src.entry((r.key.split(':').next().unwrap_or("").to_string(), r.source)).or_default() += 1;
    }
    std::fs::write(out.join("restated_keys.tsv"), ktsv)?;
    report.push_str(&format!("restated test keys: {} of {} (by kind and name source: ", p.keys.len(), keys.restate.len()));
    report.push_str(&by_src.iter().map(|((k, s), n)| format!("{k}/{s} {n}")).collect::<Vec<_>>().join(", "));
    report.push_str(")\n");
    for (k, why) in &p.skipped {
        report.push_str(&format!("  not restated: {k} ({why})\n"));
    }

    // the checks, then the restatement docs
    let guard = WordingGuard::load(keys_dir.join(TEMPLATES_FILE), guard_names(&facts, &wd))?;
    let mut sentences: Vec<String> = p.restate.iter().flat_map(restatements).collect();
    sentences.sort();
    sentences.dedup();
    let bad = check_restatements(&sentences, &guard, &p.control);
    report.push_str(&format!(
        "checks: {} distinct sentences x {} templates x {} names: filled test templates found {}; control facts ({}) mentioned {}\n",
        sentences.len(),
        guard.templates.len(),
        guard.n_names(),
        bad.iter().filter(|b| b.starts_with("template")).count(),
        p.control.len(),
        bad.iter().filter(|b| b.starts_with("control")).count()
    ));
    if !bad.is_empty() {
        print!("{report}");
        for b in bad.iter().take(20) {
            eprintln!("  {b}");
        }
        return Err(io::Error::other(format!("{} restatement problems; facts_restate.docs not written", bad.len())));
    }
    let docs = restatement_docs(&p.restate, seed, rounds, per_doc);
    let mut w = DocsWriter::create(out.join("facts_restate.docs"))?;
    let mut n_sent = 0usize;
    for d in &docs {
        n_sent += d.matches(". ").count() + d.matches("? ").count() + 1;
        w.write_doc(d)?;
    }
    let (nd, nb) = w.finish()?;
    let n_rel = |r: Rel| p.restate.iter().filter(|f| f.rel == r).count();
    report.push_str(&format!(
        "restatements: {} facts (capital {}, symbol {}, class {}, planet {}); {nd} docs, about {n_sent} sentences, {nb} bytes ({rounds} rounds of every phrasing)\n  first doc: {}\n",
        p.restate.len(),
        n_rel(Rel::Capital),
        n_rel(Rel::Symbol),
        n_rel(Rel::Class),
        n_rel(Rel::Planet),
        docs.first().map(|s| s.as_str()).unwrap_or("")
    ));

    if args.iter().any(|a| a == "--no-ledger") {
        print!("{report}");
        std::fs::write(out.join("facts_report.txt"), &report)?;
        return Ok(());
    }

    // the ledger
    let mut srcs: Vec<(String, String)> = Vec::new();
    for (i, a) in args.iter().enumerate() {
        if a == "--src" {
            let v = args.get(i + 1).ok_or_else(|| io::Error::other("--src needs NAME:SPEC"))?;
            let (n, s) = v.split_once(':').ok_or_else(|| io::Error::other("--src needs NAME:SPEC"))?;
            srcs.push((n.to_string(), s.to_string()));
        }
    }
    if srcs.is_empty() {
        srcs = default_sources();
    }
    let t0 = Instant::now();
    // one source per task; `threads` workers take the next task until none is left
    type Out = io::Result<(Ledger, u64, u64, f64)>;
    let next = std::sync::atomic::AtomicUsize::new(0);
    let slots: Vec<std::sync::Mutex<Option<Out>>> = srcs.iter().map(|_| std::sync::Mutex::new(None)).collect();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if i >= srcs.len() {
                    break;
                }
                let t = Instant::now();
                let res = (|| -> Out {
                    let mut r = SourceReader::open(&srcs[i].1).map_err(io::Error::other)?;
                    let mut l = Ledger::new(all_facts());
                    let (mut docs, mut bytes) = (0u64, 0u64);
                    while let Some(d) = r.next_doc()? {
                        if limit.is_some_and(|m| docs >= m) {
                            break;
                        }
                        l.add_text(&d);
                        docs += 1;
                        bytes += d.len() as u64;
                    }
                    Ok((l, docs, bytes, t.elapsed().as_secs_f64()))
                })();
                *slots[i].lock().unwrap() = Some(res);
            });
        }
    });
    let results: Vec<Out> = slots.into_iter().map(|m| m.into_inner().unwrap().unwrap()).collect();
    let secs = t0.elapsed().as_secs_f64();
    let mut per: Vec<Ledger> = Vec::new();
    report.push_str("ledger sources (docs, MB, sentences, seconds):\n");
    let (mut tot_bytes, mut tot_sent) = (0u64, 0u64);
    for ((name, spec), r) in srcs.iter().zip(results) {
        let (l, docs, bytes, s) = r?;
        report.push_str(&format!("  {name:<8} {docs:>9} docs {:>8.1} MB {:>10} sentences {s:>6.1}s  {spec}\n", bytes as f64 / 1e6, l.sentences));
        tot_bytes += bytes;
        tot_sent += l.sentences;
        per.push(l);
    }
    let mut total = Ledger::new(all_facts());
    for l in &per {
        total.merge(l);
    }
    report.push_str(&format!(
        "ledger time {secs:.1}s on {threads} threads ({:.1} MB/s, {:.0}k sentences/s)\n",
        tot_bytes as f64 / 1e6 / secs,
        tot_sent as f64 / 1e3 / secs
    ));

    // the table
    let mut tsv = String::from("id\tstatus\tmentions\tstated");
    for (n, _) in &srcs {
        tsv.push_str(&format!("\t{n}_mentions\t{n}_stated"));
    }
    tsv.push('\n');
    for (i, f) in facts.iter().enumerate() {
        tsv.push_str(&format!("{}\t{}\t{}\t{}", f.id(), p.own[i].1.name(), total.mentions[i], total.stated[i]));
        for l in &per {
            tsv.push_str(&format!("\t{}\t{}", l.mentions[i], l.stated[i]));
        }
        tsv.push('\n');
    }
    std::fs::write(out.join(format!("facts_ledger{suffix}.tsv")), tsv)?;

    // the summary: how often facts are stated, per relation
    let bins = [0u64, 1, 10, 100, 1000];
    report.push_str("statements per fact (stated = both sides + the relation cue in one sentence):\n");
    report.push_str("  relation   facts   =0   1-9  10-99  100-999  1000+   median   min .. max\n");
    for rel in ["capital", "symbol", "class", "planet"] {
        let mut v: Vec<u64> = facts.iter().enumerate().filter(|(_, f)| f.rel.name() == rel).map(|(i, _)| total.stated[i]).collect();
        v.sort_unstable();
        let mut h = [0usize; 5];
        for &x in &v {
            let b = bins.iter().rposition(|&e| x >= e).unwrap();
            h[b] += 1;
        }
        report.push_str(&format!(
            "  {rel:<9} {:>6} {:>5} {:>5} {:>6} {:>8} {:>6} {:>8} {:>6} .. {}\n",
            v.len(),
            h[0],
            h[1],
            h[2],
            h[3],
            h[4],
            v[v.len() / 2],
            v[0],
            v[v.len() - 1]
        ));
    }
    let show = |id: &str| -> String {
        let i = facts.iter().position(|f| f.id() == id).unwrap();
        format!("{id}: mentions {} stated {}", total.mentions[i], total.stated[i])
    };
    for id in ["capital:France", "capital:Spain", "capital:Australia", "symbol:gold", "symbol:sodium", "class:whale", "planet:Mars"] {
        report.push_str(&format!("  {}\n", show(id)));
    }
    let mut order: Vec<usize> = (0..facts.len()).collect();
    order.sort_by_key(|&i| (total.stated[i], facts[i].id()));
    let least: Vec<String> = order.iter().take(8).map(|&i| format!("{} ({})", facts[i].id(), total.stated[i])).collect();
    let most: Vec<String> = order.iter().rev().take(5).map(|&i| format!("{} ({})", facts[i].id(), total.stated[i])).collect();
    report.push_str(&format!("  least stated: {}\n  most stated: {}\n", least.join(", "), most.join(", ")));
    print!("{report}");
    std::fs::write(out.join(format!("facts_report{suffix}.txt")), report)?;
    Ok(())
}
