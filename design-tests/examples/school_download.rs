// TESTING / DESIGN ONLY. (Rust since v27, replacing school/download.py.)
// Fetch Ava's internet Rust sources into ava/data/school/downloads/, one empty folder per
// source. Nothing downloaded is executed. Verification:
//   crates      -> each .crate's SHA-256 must equal the checksum in the official crates.io index
//   rustlings / rust-by-example -> official rust-lang GitHub archives (checked later by rustc)
//   reference / nomicon -> pages crawled from doc.rust-lang.org (same-book links only)
//
//   cargo run --release --example school_download -- <downloads-dir>
//       live: needs the internet (https only, through the system's curl.exe)
//   cargo run --release --example school_download -- --replay <saved-book-dir> <book> <out-dir>
//       offline check: re-crawl a book already saved on disk; nothing is fetched
//   ... -- --sha256 <file>                  print a file's SHA-256 (as the crate check computes it)
//   ... -- --unpack <archive> <out-dir>     unpack a local .crate/.tar.gz/.zip with the same checks
//
// Archives are opened with the system's tar.exe (bsdtar), never run. An archive is refused
// whole if any entry is absolute, climbs out with "..", or is not a plain file or folder
// (links could point outside the folder).
#[path = "common/school_util.rs"]
mod util;
use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;
use util::*;

const UA: &str = "ava-design-tests/0.1 (research; one-off download)";
const CRATES: [&str; 30] = ["serde", "serde_json", "regex", "rand", "clap", "anyhow", "thiserror", "itertools",
    "log", "env_logger", "bytes", "tokio", "futures", "hashbrown", "indexmap", "once_cell",
    "smallvec", "chrono", "uuid", "url", "base64", "memchr", "either", "crossbeam-channel",
    "rayon", "num-traits", "semver", "toml", "walkdir", "glob"];

/// a system tool by absolute path, so nothing in the current folder or PATH can stand in for it
fn system_tool(name: &str) -> PathBuf {
    if cfg!(windows) {
        let root = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".into());
        Path::new(&root).join("System32").join(format!("{name}.exe"))
    } else {
        Path::new("/usr/bin").join(name)
    }
}

fn sleep(secs: f64) { std::thread::sleep(Duration::from_secs_f64(secs)); }

/// one https GET, three tries
fn get(url: &str) -> Result<Vec<u8>, String> {
    let mut last = String::new();
    for attempt in 0..3 {
        let out = Command::new(system_tool("curl"))
            .args(["-sS", "-f", "-L", "--proto", "=https", "--proto-redir", "=https", "--max-time", "60", "-A", UA, url])
            .output().map_err(|e| format!("curl: {e}"))?;
        if out.status.success() { return Ok(out.stdout); }
        last = String::from_utf8_lossy(&out.stderr).trim().to_string();
        if attempt < 2 { sleep(2.0 + 3.0 * attempt as f64); }
    }
    Err(last)
}

fn index_path(name: &str) -> String {
    let n = name.to_lowercase();
    match n.chars().count() {
        1 => format!("1/{n}"),
        2 => format!("2/{n}"),
        3 => format!("3/{}/{n}", &n[..1]),
        _ => format!("{}/{}/{n}", &n[..2], &n[2..4]),
    }
}

/// is an archive entry name safe to extract under the target folder?
fn safe_name(name: &str) -> bool {
    let b = name.as_bytes();
    let absolute = name.starts_with('/') || name.starts_with('\\') || (b.len() > 1 && b[1] == b':');
    !name.is_empty() && !absolute && !name.split(['/', '\\']).any(|p| p == "..")
}

/// extract an archive (gzip tar or zip) into `out` with the system tar, after checking every entry
fn extract(data: &[u8], ext: &str, gz: bool, out: &Path) -> Result<(), String> {
    let tmp = std::env::temp_dir().join(format!("ava-dl-{}-{}", std::process::id(), out.file_name().map_or("x".into(), |n| n.to_string_lossy().into_owned())));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
    let file = tmp.join(format!("archive.{ext}"));
    std::fs::write(&file, data).map_err(|e| e.to_string())?;
    let list = |verbose: bool| -> Result<Vec<String>, String> {
        let flags = format!("-t{}{}f", if verbose { "v" } else { "" }, if gz { "z" } else { "" });
        let o = Command::new(system_tool("tar")).arg(flags).arg(&file).output().map_err(|e| format!("tar: {e}"))?;
        if !o.status.success() { return Err(String::from_utf8_lossy(&o.stderr).trim().to_string()); }
        Ok(String::from_utf8_lossy(&o.stdout).lines().map(String::from).collect())
    };
    let res = (|| {
        let (names, long) = (list(false)?, list(true)?);
        if names.len() != long.len() { return Err("archive listing is ambiguous".to_string()); }
        if let Some(bad) = names.iter().zip(&long).find(|(n, l)| !safe_name(n) || !(l.starts_with('-') || l.starts_with('d'))) {
            return Err(format!("unsafe entry {}", py_repr(bad.0)));
        }
        let flags = format!("-x{}f", if gz { "z" } else { "" });
        let o = Command::new(system_tool("tar")).arg(flags).arg(&file).arg("-C").arg(out).output().map_err(|e| format!("tar: {e}"))?;
        if o.status.success() { Ok(()) } else { Err(String::from_utf8_lossy(&o.stderr).trim().to_string()) }
    })();
    let _ = std::fs::remove_dir_all(&tmp);
    res
}

fn crates(out: &Path) -> Vec<J> {
    std::fs::create_dir_all(out).expect("crates dir");
    let mut log = Vec::new();
    for name in CRATES {
        let idx = get(&format!("https://index.crates.io/{}", index_path(name))).expect("crates.io index");
        let idx = String::from_utf8(idx).expect("the index is UTF-8");
        let rel: Vec<J> = splitlines(&idx).into_iter().filter(|l| !py_strip(l).is_empty())
            .map(|l| json_parse(l).expect("an index line is JSON"))
            .filter(|r| !r.get("yanked").map_or(false, |y| y.truthy()) && !r.get("vers").and_then(|v| v.str()).unwrap_or("-").contains('-'))
            .collect();
        let latest = rel.last().expect("a released version");
        let vers = latest.get("vers").and_then(|v| v.str()).unwrap().to_string();
        let data = get(&format!("https://static.crates.io/crates/{name}/{name}-{vers}.crate")).expect("crate download");
        let ok = hex(&sha256(&data)) == latest.get("cksum").and_then(|v| v.str()).unwrap_or("");
        log.push(obj(vec![("crate", s(name)), ("version", s(&vers)), ("bytes", num(data.len())), ("sha256_ok", J::Bool(ok))]));
        println!("  {:<18} {:<10} {:>8.0} KB  sha256 {}", name, vers, data.len() as f64 / 1e3, if ok { "OK" } else { "MISMATCH" });
        if !ok { continue; } // an unverified crate is never extracted
        if let Err(e) = extract(&data, "tar.gz", true, out) { println!("    not extracted: {e}"); }
        sleep(0.5);
    }
    log
}

fn github(repo: &str, branch: &str, out: &Path) -> J {
    std::fs::create_dir_all(out).expect("github dir");
    let data = get(&format!("https://codeload.github.com/rust-lang/{repo}/zip/refs/heads/{branch}")).expect("github archive");
    if let Err(e) = extract(&data, "zip", false, out) { println!("    not extracted: {e}"); }
    println!("  {:<18} {:<10} {:>8.0} KB", repo, branch, data.len() as f64 / 1e3);
    obj(vec![("repo", s(&format!("rust-lang/{repo}"))), ("branch", s(branch)), ("bytes", num(data.len())), ("sha256", s(&hex(&sha256(&data))))])
}

/// every href="....html" target (no '#', '?' or quote), as the old regex href="([^"#?]+\.html) found them
fn html_links(html: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = html;
    while let Some(k) = rest.find("href=\"") {
        let tail = &rest[k + 6..];
        let run = &tail[..tail.find(['"', '#', '?']).unwrap_or(tail.len())];
        // the longest prefix of the run that ends in ".html" with at least one char before it
        if let Some(e) = run.rfind(".html").filter(|&e| e > 0) { out.push(&run[..e + 5]); }
        rest = &tail[run.len()..];
    }
    out
}

/// every page of one book on doc.rust-lang.org reachable from its index, same book only.
/// `fetch` gets one page (live: https; replay: from disk)
fn crawl(book: &str, out: &Path, limit: usize, fetch: &dyn Fn(&str, &str) -> Result<String, String>, pause: f64) -> J {
    std::fs::create_dir_all(out).expect("book dir");
    let base = format!("https://doc.rust-lang.org/{book}/");
    let (mut todo, mut seen, mut total) = (VecDeque::from(["index.html".to_string()]), HashSet::new(), 0usize);
    while !todo.is_empty() && seen.len() < limit {
        let page = todo.pop_front().unwrap();
        if !seen.insert(page.clone()) { continue; }
        let html = match fetch(&base, &page) {
            Ok(h) => h,
            Err(e) => { println!("    skip {page}: {e}"); continue; }
        };
        total += html.chars().count();
        let dest = out.join(&page);
        std::fs::create_dir_all(dest.parent().unwrap()).expect("page dir");
        write_text(&dest, &html).expect("write a page");
        for href in html_links(&html) {
            let nxt = urljoin(&format!("{base}{page}"), href);
            if let Some(rel) = nxt.strip_prefix(&base) {
                if !seen.contains(rel) { todo.push_back(rel.to_string()); }
            }
        }
        sleep(pause);
    }
    println!("  {:<18} {} pages {:.1} MB", book, seen.len(), total as f64 / 1e6);
    obj(vec![("book", s(book)), ("url", s(&base)), ("pages", num(seen.len())), ("bytes", num(total))])
}

fn live_page(base: &str, page: &str) -> Result<String, String> {
    get(&format!("{base}{page}")).map(|b| String::from_utf8_lossy(&b).into_owned())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() == 5 && a[1] == "--replay" {
        let saved = PathBuf::from(&a[2]);
        // a page that would sit outside the saved folder is never read
        let from_disk = move |_: &str, page: &str| -> Result<String, String> {
            if !safe_name(page) { return Err("outside the book".into()); }
            read_text(&saved.join(page)).map_err(|e| e.to_string())
        };
        crawl(&a[3], Path::new(&a[4]), 400, &from_disk, 0.0);
        return;
    }
    if a.len() == 3 && a[1] == "--sha256" {
        println!("{}", hex(&sha256(&std::fs::read(&a[2]).expect("read the file"))));
        return;
    }
    if a.len() == 4 && a[1] == "--unpack" {
        // a local .crate/.tar.gz or .zip, with the same entry checks as a download
        let gz = !a[2].to_lowercase().ends_with(".zip");
        std::fs::create_dir_all(&a[3]).expect("out dir");
        match extract(&std::fs::read(&a[2]).expect("read the archive"), if gz { "tar.gz" } else { "zip" }, gz, Path::new(&a[3])) {
            Ok(()) => println!("unpacked"),
            Err(e) => { println!("refused: {e}"); std::process::exit(2); }
        }
        return;
    }
    if a.len() != 2 { eprintln!("usage: school_download <downloads-dir> | --replay <saved-book-dir> <book> <out-dir> | --sha256 <file> | --unpack <archive> <out-dir>"); std::process::exit(1); }
    let root = PathBuf::from(&a[1]);
    std::fs::create_dir_all(&root).expect("downloads dir");
    let cr = crates(&root.join("crates"));
    let gh = vec![github("rustlings", "main", &root.join("rustlings")), github("rust-by-example", "master", &root.join("rust-by-example"))];
    let bk = vec![crawl("reference", &root.join("reference"), 400, &live_page, 0.3), crawl("nomicon", &root.join("nomicon"), 400, &live_page, 0.3)];
    let bad: Vec<String> = cr.iter().filter(|c| !c.get("sha256_ok").map_or(false, |x| x.truthy()))
        .map(|c| py_repr(c.get("crate").and_then(|x| x.str()).unwrap_or(""))).collect();
    let n = cr.len();
    let manifest = obj(vec![("crates", J::Arr(cr)), ("github", J::Arr(gh)), ("books", J::Arr(bk))]);
    write_text(&root.join("MANIFEST.json"), &json_dumps(&manifest, Some(1))).expect("write MANIFEST.json");
    println!("crates verified: {} / {} failed: [{}]", n - bad.len(), n, bad.join(", "));
}
