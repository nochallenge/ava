// TESTING / DESIGN ONLY
//! code_docs: C11's code reading, from Rust crates already on this machine (the local cargo registry), only
//! those under permissive licences. Nothing is downloaded or run; the files are only read as text.
//!
//! CRATES. Every folder of `<registry>/src/*/` with a Cargo.toml whose [package] gives a name, a version
//! and a `license` expression in which every licence is permissive (`PERMISSIVE`; operators OR / AND /
//! WITH and `/` are allowed; a crate with only `license-file` is left out, its terms not being checked).
//! Only the newest version of each crate is used.
//!
//! FILES. Every `.rs` file of a crate, in path order, skipping: any path part named like a test set
//! (test, sealed, spent, heldout: so `tests/` folders and `test.rs` files too), `target` and `.git`, files
//! over `max_file` bytes (generated bindings) and files that are not UTF-8. A crate gives at most
//! `max_crate` bytes; the whole output at most `max_total` bytes (crates in name order).
//!
//! DOC. One file per document: a first line `// crate NAME VERSION (LICENCE): PATH`, then the file.

use crate::docs::DocsWriter;
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

pub const PERMISSIVE: [&str; 14] = [
    "MIT",
    "MIT-0",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "0BSD",
    "ISC",
    "Zlib",
    "Unlicense",
    "CC0-1.0",
    "BSL-1.0",
    "Unicode-DFS-2016",
    "Unicode-3.0",
    "LLVM-exception",
];

/// True when every licence named in an SPDX-like expression is permissive.
pub fn permissive(expr: &str) -> bool {
    let toks: Vec<&str> = expr.split(|c: char| c.is_whitespace() || c == '(' || c == ')' || c == '/').filter(|t| !t.is_empty()).collect();
    !toks.is_empty() && toks.iter().all(|t| matches!(*t, "OR" | "AND" | "WITH" | "or" | "and") || PERMISSIVE.contains(t))
}

/// (name, version, license) from a Cargo.toml's [package] table.
pub fn package_info(toml: &str) -> Option<(String, String, Option<String>)> {
    let mut in_pkg = false;
    let (mut name, mut version, mut license) = (None, None, None);
    for line in toml.lines() {
        let l = line.trim();
        if l.starts_with('[') {
            in_pkg = l == "[package]";
            continue;
        }
        if !in_pkg {
            continue;
        }
        let Some((k, v)) = l.split_once('=') else { continue };
        let v = v.trim();
        let val = v.strip_prefix('"').and_then(|x| x.strip_suffix('"')).map(str::to_string);
        match k.trim() {
            "name" => name = val,
            "version" => version = val,
            "license" => license = val,
            _ => {}
        }
    }
    Some((name?, version?, license))
}

/// A version as numbers for ordering (pre-release parts after `-` ignored).
pub fn version_key(v: &str) -> Vec<u64> {
    v.split(['-', '+']).next().unwrap_or("").split('.').map(|p| p.parse().unwrap_or(0)).collect()
}

/// A path part that names a test set (skipped).
pub fn testlike(part: &str) -> bool {
    let l = part.to_lowercase();
    ["test", "sealed", "spent", "heldout"].iter().any(|w| l.contains(w))
}

fn rs_files(dir: &Path, root: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    let mut ents: Vec<_> = std::fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();
    ents.sort_by_key(|e| e.file_name());
    for e in ents {
        let p = e.path();
        let name = e.file_name().to_string_lossy().to_string();
        if testlike(&name) || name == "target" || name == ".git" {
            continue;
        }
        let ft = e.file_type()?;
        if ft.is_dir() {
            rs_files(&p, root, out)?;
        } else if ft.is_file() && name.ends_with(".rs") {
            out.push(p);
        }
    }
    Ok(())
}

#[derive(Default, Debug)]
pub struct CodeReport {
    pub crate_dirs: usize,
    pub not_permissive: usize,
    pub no_licence: usize,
    pub older_versions: usize,
    pub crates_used: usize,
    pub files: usize,
    pub bytes: u64,
    pub skipped_big: usize,
    pub skipped_not_utf8: usize,
    pub capped_crates: usize,
    pub licences: BTreeMap<String, usize>,
}

/// Writes the code docs from every registry source folder under `src_root` (e.g. `<cargo home>/registry/src`).
pub fn write_code_docs(src_root: &Path, out: &Path, max_file: u64, max_crate: u64, max_total: u64) -> io::Result<CodeReport> {
    let mut rep = CodeReport::default();
    // newest version per crate name
    let mut best: BTreeMap<String, (Vec<u64>, String, String, PathBuf)> = BTreeMap::new();
    for reg in std::fs::read_dir(src_root)?.filter_map(|e| e.ok()) {
        if !reg.file_type()?.is_dir() {
            continue;
        }
        for c in std::fs::read_dir(reg.path())?.filter_map(|e| e.ok()) {
            let Ok(toml) = std::fs::read_to_string(c.path().join("Cargo.toml")) else { continue };
            rep.crate_dirs += 1;
            let Some((name, version, lic)) = package_info(&toml) else { continue };
            let Some(lic) = lic else {
                rep.no_licence += 1;
                continue;
            };
            if !permissive(&lic) {
                rep.not_permissive += 1;
                continue;
            }
            let k = version_key(&version);
            match best.get(&name) {
                Some(b) if b.0 >= k => rep.older_versions += 1,
                Some(_) => {
                    rep.older_versions += 1;
                    best.insert(name, (k, version, lic, c.path()));
                }
                None => {
                    best.insert(name, (k, version, lic, c.path()));
                }
            }
        }
    }
    let mut w = DocsWriter::create(out)?;
    'crates: for (name, (_, version, lic, dir)) in &best {
        if rep.bytes >= max_total {
            break;
        }
        let mut files = Vec::new();
        rs_files(dir, dir, &mut files)?;
        let mut crate_bytes = 0u64;
        let mut used = false;
        for f in files {
            let len = std::fs::metadata(&f)?.len();
            if len > max_file {
                rep.skipped_big += 1;
                continue;
            }
            if crate_bytes + len > max_crate {
                rep.capped_crates += 1;
                break;
            }
            if rep.bytes + len > max_total {
                break 'crates;
            }
            let Ok(text) = std::fs::read_to_string(&f) else {
                rep.skipped_not_utf8 += 1;
                continue;
            };
            let rel = f.strip_prefix(dir).unwrap_or(&f).to_string_lossy().replace('\\', "/");
            w.write_doc(&format!("// crate {name} {version} ({lic}): {rel}\n{}", text.replace("\r\n", "\n").trim_end()))?;
            rep.files += 1;
            rep.bytes += len;
            crate_bytes += len;
            used = true;
        }
        if used {
            rep.crates_used += 1;
            *rep.licences.entry(lic.clone()).or_default() += 1;
        }
    }
    w.finish()?;
    Ok(rep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn licences() {
        assert!(permissive("MIT OR Apache-2.0"));
        assert!(permissive("MIT/Apache-2.0"));
        assert!(permissive("(MIT OR Apache-2.0) AND Unicode-DFS-2016"));
        assert!(permissive("Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT"));
        assert!(!permissive("MPL-2.0"));
        assert!(!permissive("MIT OR GPL-3.0"));
        assert!(!permissive(""));
    }

    #[test]
    fn toml_and_versions() {
        let t = "[package]\nname = \"foo\"\nversion = \"1.2.10\"\nlicense = \"MIT\"\n[dependencies]\nname = \"x\"\n";
        assert_eq!(package_info(t), Some(("foo".into(), "1.2.10".into(), Some("MIT".into()))));
        assert!(version_key("1.2.10") > version_key("1.2.9"));
        assert!(version_key("0.10.0-alpha.1") > version_key("0.9.9"));
        assert!(testlike("tests") && testlike("test_utils.rs") && !testlike("lib.rs"));
    }
}
