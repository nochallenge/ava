// TESTING / DESIGN ONLY — qa_split: v41's held-out question slices of the SciQ, ARC and OpenBookQA TRAIN splits (dev, sealed test, rest).
//
// qa_split [--out-dir data/v41] [--dev-per 200] [--test-per 400]
//
// v41 starts from v39, which never saw the question data, so v41's sealed QA test is a slice of the train splits
// held out from teaching. Per source (sciq-train, arc-train = ARC-Easy + ARC-Challenge train, obqa-train, loaded
// with ava_gpu::qa::eval_set and turned into items with ava_gpu::mc::from_qa(name, &qs, false): closed book, the
// prompt exactly what mc_eval scores):
//   every question's stem = ava_data::qa_stems::normalise_stem(question); the distinct stems are walked in
//   (split_hash(stem), stem) order; a stem's first item goes to the sealed test until --test-per items, then to
//   dev until --dev-per items; a stem already held out by an earlier source is skipped. Sources are taken in the
//   order sciq, arc, obqa. Every item whose stem is held out (dev or test, any source) is left out of the rest.
// Writes (MC TSV `id kind bin right prompt c0 c1 ...`, text escaped as ava_data::fact_items::escape_field;
// kind = sciq | arc | obqa; bin = easy | challenge for ARC, the kind otherwise; ids as from_qa gives them):
//   OUT/qa_dev.tsv               dev slice (read freely)
//   OUT/sealed/qa_test.tsv       sealed test slice (SEALED: counts and sha256 only)
//   OUT/reward_train.tsv         the rest (the reward phase's questions)
//   OUT/qa_exclude.txt           the held-out stems (dev + test), sorted, one per line (qa_prep --exclude)

use ava_data::fact_items::escape_field;
use ava_data::qa_stems::{normalise_stem, split_hash};
use ava_data::sha256::sha256_file;
use ava_gpu::mc::{from_qa, Item};
use ava_gpu::qa::{eval_set, load_arc, ARC_DIR};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

const USAGE: &str = "usage: qa_split [--out-dir data/v41] [--dev-per 200] [--test-per 400]";
/// (kind, eval_set name)
const SOURCES: [(&str, &str); 3] = [("sciq", "sciq-train"), ("arc", "arc-train"), ("obqa", "obqa-train")];

fn main() {
    if let Err(e) = run() {
        eprintln!("qa_split: {e}");
        std::process::exit(1);
    }
}

fn line(it: &Item) -> String {
    let mut s = format!("{}\t{}\t{}\t{}\t{}", it.id, it.kind, it.bin, it.right, escape_field(&it.prompt));
    for c in &it.choices {
        s.push('\t');
        s.push_str(&escape_field(c));
    }
    s
}

fn write(path: &Path, items: &[&Item]) -> Result<(), String> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    }
    let mut w = BufWriter::new(std::fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?);
    for it in items {
        writeln!(w, "{}", line(it)).map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| e.to_string())
}

fn sha(p: &Path) -> String {
    sha256_file(p).map(|h| format!("{h}  {} bytes", std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))).unwrap_or_else(|e| e.to_string())
}

fn run() -> Result<(), String> {
    let (mut out, mut dev_per, mut test_per) = (PathBuf::from("data/v41"), 200usize, 400usize);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.into());
        }
        let val = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?;
        match key {
            "--out-dir" => out = PathBuf::from(val),
            "--dev-per" => dev_per = val.parse().map_err(|_| format!("bad integer {val:?}"))?,
            "--test-per" => test_per = val.parse().map_err(|_| format!("bad integer {val:?}"))?,
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let easy_n = load_arc(&PathBuf::from(ARC_DIR).join("ARC-Easy/ARC-Easy-Train.jsonl"))?.len();

    // every source's items (kind and bin set) and stems
    let mut all: Vec<(Vec<Item>, Vec<String>)> = Vec::new();
    for (kind, name) in SOURCES {
        let qs = eval_set(name)?;
        let mut items = from_qa(name, &qs, false);
        for (j, it) in items.iter_mut().enumerate() {
            it.kind = kind.into();
            it.bin = if kind == "arc" { if j < easy_n { "easy" } else { "challenge" }.into() } else { kind.into() };
        }
        let stems = qs.iter().map(|q| normalise_stem(&q.question)).collect();
        all.push((items, stems));
    }

    // 0 = test, 1 = dev
    let mut held: HashMap<String, u8> = HashMap::new();
    let mut picked: Vec<[Vec<usize>; 2]> = Vec::new();
    let mut skipped_cross = [0usize; 3];
    for (s, (_, stems)) in all.iter().enumerate() {
        let mut first: HashMap<&str, usize> = HashMap::new();
        for (j, st) in stems.iter().enumerate() {
            if !st.is_empty() {
                first.entry(st.as_str()).or_insert(j);
            }
        }
        let mut order: Vec<(&str, usize)> = first.into_iter().collect();
        order.sort_by(|a, b| (split_hash(a.0), a.0).cmp(&(split_hash(b.0), b.0)));
        let mut p: [Vec<usize>; 2] = [Vec::new(), Vec::new()];
        for (st, j) in order {
            if p[0].len() >= test_per && p[1].len() >= dev_per {
                break;
            }
            if held.contains_key(st) {
                skipped_cross[s] += 1;
                continue;
            }
            let k = if p[0].len() < test_per { 0 } else { 1 };
            held.insert(st.to_string(), k as u8);
            p[k].push(j);
        }
        picked.push(p);
    }

    let mut dev: Vec<&Item> = Vec::new();
    let mut test: Vec<&Item> = Vec::new();
    let mut rest: Vec<&Item> = Vec::new();
    println!("source  items  distinct-stems  test  dev  rest  (items left out of rest: held stem)  stems already held by an earlier source");
    for (s, (items, stems)) in all.iter().enumerate() {
        let mut p = picked[s].clone();
        p[0].sort_unstable();
        p[1].sort_unstable();
        test.extend(p[0].iter().map(|&j| &items[j]));
        dev.extend(p[1].iter().map(|&j| &items[j]));
        let before = rest.len();
        rest.extend(items.iter().zip(stems).filter(|(_, st)| !held.contains_key(st.as_str())).map(|(it, _)| it));
        let n_rest = rest.len() - before;
        let distinct: HashSet<&String> = stems.iter().collect();
        println!(
            "{:<6} {:>6}  {:>14}  {:>4}  {:>3}  {:>5}  {:>5}  {:>3}",
            SOURCES[s].0,
            items.len(),
            distinct.len(),
            p[0].len(),
            p[1].len(),
            n_rest,
            items.len() - n_rest - p[0].len() - p[1].len(),
            skipped_cross[s]
        );
    }
    let cr = |v: &[&Item]| v.iter().filter(|it| it.prompt.contains('\r') || it.choices.iter().any(|c| c.contains('\r'))).count();
    let n_cr = cr(&dev) + cr(&test) + cr(&rest);
    if n_cr > 0 {
        println!("note: {n_cr} items hold a carriage return (escaped as \\r; mc.rs reads it back as two characters)");
    }

    let stems: BTreeSet<&String> = held.keys().collect();
    let (pd, pt, pr, pe) = (out.join("qa_dev.tsv"), out.join("sealed").join("qa_test.tsv"), out.join("reward_train.tsv"), out.join("qa_exclude.txt"));
    write(&pd, &dev)?;
    write(&pt, &test)?;
    write(&pr, &rest)?;
    let mut ex = String::new();
    for s in &stems {
        ex.push_str(s);
        ex.push('\n');
    }
    std::fs::write(&pe, ex).map_err(|e| format!("write {}: {e}", pe.display()))?;
    // checks: no held stem in the rest; dev and test share no stem
    let stem_of = |it: &Item| normalise_stem(it.prompt.trim_start_matches("Question: ").trim_end_matches("\nAnswer:"));
    let in_rest = rest.iter().filter(|it| held.contains_key(&stem_of(it))).count();
    let dev_s: HashSet<String> = dev.iter().map(|it| stem_of(it)).collect();
    let test_s: HashSet<String> = test.iter().map(|it| stem_of(it)).collect();
    println!(
        "dev {} items, sealed test {} items, rest {} items; {} held-out stems; checks: {in_rest} held stems in the rest, {} stems shared by dev and test",
        dev.len(),
        test.len(),
        rest.len(),
        stems.len(),
        dev_s.intersection(&test_s).count()
    );
    for (p, what) in [(&pd, "dev"), (&pt, "sealed test (SEALED)"), (&pr, "reward train"), (&pe, "exclude stems")] {
        println!("  {what:<22} sha256 {}  {}", sha(p), p.display());
    }
    if in_rest > 0 || !dev_s.is_disjoint(&test_s) {
        return Err("a check failed".into());
    }
    Ok(())
}
