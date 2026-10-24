// TESTING / DESIGN ONLY. (Rust since v27, replacing school/build.py byte for byte.)
// Build Ava's school corpora: tagged TSV lines "@domain<TAB>prompt<TAB>reply", held-out test
// lines FIRST, then training lines. compose.rs reads them with AVA_CORPUS / AVA_TEST.
//
//   cargo run --release --example school_build -- <ava-root> [out-dir]
//   (writes <out-dir>/*.tsv and counts.json; out-dir defaults to <ava-root>/data/school)
//
// Every test line comes from material never trained on:
//   code     whole FILES are held out (hash of the path), so no test line was seen in training
//   facts    each verified fact has 3 phrasings: 2 train, the 3rd is the exam
//   numbers  10% of problems are never shown; they test whether she can do arithmetic or
//            only remember it
// Lines are pre-filtered with the same length limits compose.rs applies (reply <= 16 tokens,
// prompt <= 20), using her own tokenizer (design_tests::seq::tokenize) and an UPPER BOUND of
// code_tokenize, so compose never drops a line and the test block is exactly the first
// AVA_TEST lines. Learner code is kept only if rustc type-checks it (nothing is run).
#[path = "common/school_util.rs"]
mod util;
use design_tests::seq::tokenize as seq_tok;
use std::path::{Path, PathBuf};
use std::process::Command;
use util::*;

type Pairs = Vec<(String, String)>;

/// an upper bound on code_tokenize's count: identifiers are one token, every other non-space
/// char is one token (code_tokenize merges operators/strings, so it is never more)
fn code_len(s: &str) -> usize {
    let (mut n, mut word) = (0, false);
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            if !word { n += 1; word = true; }
        } else {
            word = false;
            if !py_space(c) { n += 1; }
        }
    }
    n
}

fn strip_block_comments(src: &str) -> String {
    let b = src.as_bytes();
    let (mut out, mut i, mut depth) = (Vec::with_capacity(b.len()), 0usize, 0usize);
    while i < b.len() {
        if b[i..].starts_with(b"/*") { depth += 1; i += 2; continue; }
        if depth > 0 && b[i..].starts_with(b"*/") { depth -= 1; i += 2; continue; }
        if depth > 0 {
            if b[i] == b'\n' { out.push(b'\n'); }
            i += 1;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8(out).expect("comment stripping keeps UTF-8")
}

fn code_lines(src: &str) -> Vec<String> {
    let mut lines = Vec::new();
    for raw in splitlines(&strip_block_comments(src)) {
        let mut s = py_strip(raw).to_string();
        if s.is_empty() || s.starts_with("//") { continue; }
        if s.contains("//") && !s.contains('"') {
            s = py_rstrip(&s[..s.find("//").unwrap()]).to_string();
            if s.is_empty() { continue; }
        }
        if s.contains('\t') { s = s.replace('\t', " "); }
        lines.push(s);
    }
    lines
}

fn code_pairs(lines: &[String]) -> Pairs {
    lines.windows(2).filter(|w| code_len(&w[1]) <= 16 && code_len(&w[0]) <= 20)
        .map(|w| (w[0].clone(), w[1].clone())).collect()
}

/// a path relative to root, as Python's str() shows it (backslashes on Windows)
fn rel_str(f: &Path, root: &Path) -> String {
    let r = f.strip_prefix(root).expect("file under its root");
    r.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>()
        .join(if cfg!(windows) { "\\" } else { "/" })
}
fn rel_posix(f: &Path, root: &Path) -> String {
    f.strip_prefix(root).expect("file under its root").components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned()).collect::<Vec<_>>().join("/")
}

/// files -> (test pairs, train pairs); a file is held out whole when its path hashes to 0
fn code_domain(mut files: Vec<PathBuf>, root: &Path, hold: u128) -> (Pairs, Pairs) {
    files.sort_by_key(|f| path_key(f));
    let (mut test, mut train) = (Vec::new(), Vec::new());
    for f in &files {
        let Ok(src) = read_text(f) else { continue };
        let pairs = code_pairs(&code_lines(&src));
        if bucket(&rel_str(f, root), hold) == 0 { test.extend(pairs) } else { train.extend(pairs) }
    }
    (test, train)
}

// ---------------------------------------------------------------- verified learner code
fn typecheck(code: &str, edition: &str, n: usize) -> bool {
    let d = std::env::temp_dir().join(format!("ava-typecheck-{}-{n}", std::process::id()));
    let _ = std::fs::create_dir_all(&d);
    let p = d.join("x.rs");
    if write_text(&p, code).is_err() { return false; }
    let ok = Command::new("rustc").args(["--edition", edition, "--crate-type", "bin", "--emit=metadata", "--cap-lints", "allow", "-o"])
        .arg(d.join("x.rmeta")).arg(&p).output().map_or(false, |o| o.status.success());
    let _ = std::fs::remove_dir_all(&d);
    ok
}

/// runnable ```rust blocks from one Rust-by-Example page, rustdoc-style: '# ' hidden lines
/// included, and a block without fn main wrapped in one
fn rbe_blocks(md: &str) -> Vec<String> {
    let b = md.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i + 3 <= b.len() {
        if &b[i..i + 3] != b"```" { i += 1; continue; }
        // the regex ```([^\n]*)\n(.*?)``` : attributes to the end of the line, then the shortest body
        let Some(nl) = b[i + 3..].iter().position(|&c| c == b'\n').map(|k| i + 3 + k) else { i += 1; continue };
        let Some(end) = md[nl + 1..].find("```").map(|k| nl + 1 + k) else { i += 1; continue };
        let (attrs, body) = (&md[i + 3..nl], &md[nl + 1..end]);
        i = end + 3;
        let a: Vec<&str> = attrs.split(',').map(py_strip).collect();
        if a[0] != "rust" || a.iter().any(|x| ["ignore", "compile_fail", "edition2015", "text"].contains(x)) { continue; }
        let lines: Vec<&str> = splitlines(body).into_iter().map(|l| if l.starts_with("# ") || l == "#" { &l[l.len().min(2)..] } else { l }).collect();
        let mut code = lines.join("\n");
        if !code.contains("fn main") { code = format!("fn main() {{\n{code}\n}}"); }
        out.push(code);
    }
    out
}

fn first_named(dir: &Path, prefix: &str) -> Option<PathBuf> {
    std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path())
        .find(|p| p.file_name().map_or(false, |n| n.to_string_lossy().to_lowercase().starts_with(prefix)))
}

/// rustlings official solutions + Rust by Example blocks that TYPE-CHECK with rustc.
/// rustc only compiles; nothing is run. returns (checked, passed, name -> code)
fn learner_code(dl: &Path, cache: &Path) -> (usize, usize, Vec<(String, String)>) {
    if cache.exists() {
        let j = json_parse(&read_text(cache).expect("learner cache")).expect("learner cache is JSON");
        let files = match j.get("files") { Some(J::Obj(v)) => v.iter().map(|(k, x)| (k.clone(), x.str().unwrap_or("").to_string())).collect(), _ => Vec::new() };
        let n = |k: &str| j.get(k).and_then(|x| x.num()).unwrap_or(0.0) as usize;
        return (n("checked"), n("passed"), files);
    }
    let mut jobs: Vec<(String, String, String)> = Vec::new();
    if let Some(rl) = first_named(&dl.join("rustlings"), "rustlings-") {
        let ed = if read_text(&rl.join("Cargo.toml")).unwrap_or_default().contains("edition = \"2024\"") { "2024" } else { "2021" };
        let mut fs = Vec::new();
        rglob(&rl.join("solutions"), ".rs", &mut fs);
        fs.sort_by_key(|f| path_key(f));
        for f in fs { jobs.push((format!("rustlings/{}", rel_posix(&f, &rl)), read_text(&f).expect("rustlings file"), ed.into())); }
    }
    if let Some(rbe) = first_named(&dl.join("rust-by-example"), "rust-by-example-") {
        let mut fs = Vec::new();
        rglob(&rbe.join("src"), ".md", &mut fs);
        fs.sort_by_key(|f| path_key(f));
        for f in fs {
            for (i, code) in rbe_blocks(&read_text(&f).expect("rbe page")).into_iter().enumerate() {
                jobs.push((format!("rbe/{}#{i}", rel_posix(&f, &rbe)), code, "2021".into()));
            }
        }
    }
    // four at a time, as before
    let ok: Vec<bool> = std::thread::scope(|sc| {
        let chunks: Vec<Vec<usize>> = (0..4).map(|t| (t..jobs.len()).step_by(4).collect()).collect();
        let hs: Vec<_> = chunks.into_iter().map(|c| { let jobs = &jobs; sc.spawn(move || c.into_iter().map(|i| (i, typecheck(&jobs[i].1, &jobs[i].2, i))).collect::<Vec<_>>()) }).collect();
        let mut ok = vec![false; jobs.len()];
        for h in hs { for (i, k) in h.join().unwrap() { ok[i] = k; } }
        ok
    });
    let passed = ok.iter().filter(|&&k| k).count();
    let files: Vec<(String, String)> = jobs.iter().zip(&ok).filter(|x| *x.1).map(|(j, _)| (j.0.clone(), j.1.clone())).collect();
    let res = obj(vec![("checked", num(jobs.len())), ("passed", num(passed)),
        ("files", J::Obj(files.iter().map(|(k, v)| (k.clone(), s(v))).collect())),
        ("failed", J::Arr(jobs.iter().zip(&ok).filter(|x| !*x.1).map(|(j, _)| s(&j.0)).collect()))]);
    write_text(cache, &json_dumps(&res, None)).expect("write the learner cache");
    (jobs.len(), passed, files)
}

// ---------------------------------------------------------------- facts and numbers
fn facts(path: &Path, failed: &[String]) -> (Pairs, Pairs, usize, usize) {
    let (mut test, mut train, mut kept, mut dropped) = (Vec::new(), Vec::new(), 0, 0);
    if !path.exists() { return (test, train, kept, dropped); }
    for line in splitlines(&read_text(path).expect("facts file")) {
        if py_strip(line).is_empty() { continue; }
        let f = json_parse(line).expect("a fact line is JSON");
        let a = py_strip(f.get("a").and_then(|x| x.str()).unwrap_or("")).to_string();
        let qs: Vec<String> = f.get("q").and_then(|x| x.arr()).unwrap_or(&[]).iter().map(|x| x.str().unwrap_or("").to_string()).collect();
        let id = f.get("id").and_then(|x| x.str()).unwrap_or("");
        let at = seq_tok(&a).len();
        if failed.iter().any(|x| x == id) || qs.len() != 3 || at == 0 || at > 16 { dropped += 1; continue; }
        let ok: Vec<Option<String>> = qs.iter().map(|q| { let q = py_strip(q); let n = seq_tok(q).len(); (n > 0 && n <= 20).then(|| q.to_string()) }).collect();
        for q in ok[..2].iter().flatten() { train.push((q.clone(), a.clone())); }
        if let Some(q) = &ok[2] { test.push((q.clone(), a.clone())); }
        kept += 1;
    }
    (test, train, kept, dropped)
}

fn numbers() -> Vec<(String, (Pairs, Pairs))> {
    let mut probs: Vec<(i64, &str, i64, i64)> = Vec::new();
    for a in 0..50 { for b in 0..50 { probs.push((a, "+", b, a + b)); } }
    for a in 0..50 { for b in 0..=a { probs.push((a, "-", b, a - b)); } }
    for a in 0..13 { for b in 0..13 { probs.push((a, "*", b, a * b)); } }
    // independent re-check of every answer before anything is written
    assert!(probs.iter().all(|&(a, o, b, r)| match o { "+" => a.checked_add(b), "-" => a.checked_sub(b), _ => a.checked_mul(b) } == Some(r)));
    let dig = |n: i64| n.to_string().chars().map(|c| c.to_string()).collect::<Vec<_>>().join(" ");
    let (mut num, mut numd) = ((Vec::new(), Vec::new()), (Vec::new(), Vec::new()));
    for &(a, o, b, r) in &probs {
        let test = bucket(&format!("{a}{o}{b}"), 10) == 0;
        let (x, y) = (format!("what is {a} {o} {b} ?"), format!("{r} ."));
        let (xd, yd) = (format!("what is {} {o} {} ?", dig(a), dig(b)), format!("{} .", dig(r)));
        if test { num.0.push((x, y)); numd.0.push((xd, yd)); } else { num.1.push((x, y)); numd.1.push((xd, yd)); }
    }
    vec![("@num".into(), num), ("@numd".into(), numd)]
}

/// the same pairs compose.rs reads from triples.tsv: same filter, same order
fn dialogue(path: &Path, test_n: usize, train_n: usize) -> (Pairs, Pairs) {
    let text = read_text(path).expect("triples.tsv");
    let mut pairs: Pairs = Vec::new();
    for line in splitlines(&text) {
        let p: Vec<&str> = line.split('\t').collect();
        if p.len() < 3 { continue; }
        let (q, r) = (p[1], p[2]);
        let (qt, rt) = (seq_tok(q), seq_tok(r));
        if !qt.is_empty() && !rt.is_empty() && rt.len() <= 16 && qt.len() <= 20 {
            pairs.push((q.to_string(), r.to_string()));
            if pairs.len() >= test_n + train_n { break; }
        }
    }
    let train = pairs.split_off(test_n.min(pairs.len()));
    (pairs, train)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 2 { eprintln!("usage: school_build <ava-root> [out-dir]"); std::process::exit(1); }
    let ava = PathBuf::from(&a[1]);
    let school = ava.join("data/school");
    let out = a.get(2).map_or(school.clone(), PathBuf::from);
    let dl = school.join("downloads");
    let failed: Vec<String> = read_text(&school.join("facts/failed.json")).ok()
        .and_then(|t| json_parse(&t).ok()).and_then(|j| j.arr().map(|v| v.iter().filter_map(|x| x.str().map(String::from)).collect()))
        .unwrap_or_default();
    let sysroot = Command::new("rustc").args(["--print", "sysroot"]).output().expect("rustc --print sysroot");
    let std_root = PathBuf::from(String::from_utf8_lossy(&sysroot.stdout).trim()).join("lib").join("rustlib").join("src").join("rust").join("library");

    let mut doms: Vec<(String, (Pairs, Pairs))> = Vec::new();
    doms.push(("@dialogue".into(), dialogue(&ava.join("data/triples.tsv"), 1200, 30_000)));
    let mut std_files = Vec::new();
    for d in ["core", "alloc", "std"] { rglob(&std_root.join(d).join("src"), ".rs", &mut std_files); }
    doms.push(("@code_std".into(), code_domain(std_files, &std_root, 10)));
    let mut crate_files = Vec::new();
    rglob(&dl.join("crates"), ".rs", &mut crate_files);
    crate_files.retain(|f| !f.components().any(|c| c.as_os_str() == "target"));
    doms.push(("@code_crates".into(), code_domain(crate_files, &dl.join("crates"), 10)));
    // AVA_LEARNER_CACHE moves the type-check cache (a missing file = check everything again)
    let cache = std::env::var("AVA_LEARNER_CACHE").map_or(school.join("learner_verified.json"), PathBuf::from);
    let (checked, passed, mut files) = learner_code(&dl, &cache);
    println!("learner code: {passed} / {checked} type-check with rustc");
    files.sort_by(|x, y| x.0.cmp(&y.0));
    let (mut lt, mut ltr) = (Vec::new(), Vec::new());
    for (name, code) in &files {
        let pairs = code_pairs(&code_lines(code));
        if bucket(name, 10) == 0 { lt.extend(pairs) } else { ltr.extend(pairs) }
    }
    doms.push(("@code_learn".into(), (lt, ltr)));
    let mut ava_files = Vec::new();
    for d in ["design-tests/src", "design-tests/examples"] {
        if let Ok(rd) = std::fs::read_dir(ava.join(d)) {
            for e in rd.flatten() {
                let p = e.path();
                if p.file_name().map_or(false, |n| n.to_string_lossy().to_lowercase().ends_with(".rs")) { ava_files.push(p); }
            }
        }
    }
    doms.push(("@ava_code".into(), code_domain(ava_files, &ava.join("design-tests"), 10)));
    let mut report_facts: Vec<(String, J)> = Vec::new();
    for (key, tag) in [("ai", "@ai"), ("rustdoc", "@rustdoc"), ("refnomicon", "@refnomicon"), ("self", "@self")] {
        let (te, tr, kept, dropped) = facts(&school.join(format!("facts/{key}.jsonl")), &failed);
        report_facts.push((key.into(), obj(vec![("kept", num(kept)), ("dropped", num(dropped))])));
        if kept > 0 { doms.push((tag.into(), (te, tr))); }
    }
    doms.extend(numbers());

    std::fs::create_dir_all(&out).expect("out dir");
    let mut counts: Vec<(String, J)> = Vec::new();
    for (tag, (te, tr)) in &doms {
        counts.push((tag.clone(), obj(vec![("test", num(te.len())), ("train", num(tr.len()))])));
        let mut body = String::new();
        for (q, r) in te.iter().chain(tr) { body.push_str(&format!("{tag}\t{q}\t{r}\n")); }
        std::fs::write(out.join(format!("{}.tsv", &tag[1..])), body).expect("write a domain");
    }
    let report = obj(vec![("facts", J::Obj(report_facts.clone())), ("counts", J::Obj(counts.clone()))]);
    write_text(&out.join("counts.json"), &json_dumps(&report, Some(1))).expect("write counts.json");
    for (tag, c) in &counts {
        let n = |k: &str| c.get(k).and_then(|x| x.num()).unwrap_or(0.0) as usize;
        println!("  {:<14} test {:>7}   train {:>8}", tag, n("test"), n("train"));
    }
    // Python's repr of the facts report, as the old tool printed it
    let f = report_facts.iter().map(|(k, v)| {
        let n = |x: &str| v.get(x).and_then(|y| y.num()).unwrap_or(0.0) as usize;
        format!("'{k}': {{'kept': {}, 'dropped': {}}}", n("kept"), n("dropped"))
    }).collect::<Vec<_>>().join(", ");
    println!("facts: {{{f}}}");
}
