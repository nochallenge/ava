// TESTING / DESIGN ONLY
//! v42_code: C11's code reading from the Rust crates already in the local cargo registry (permissive
//! licences only; see the `code_docs` module). Nothing is downloaded or run.
//!
//!   v42_code [--registry DIR] [--out data/nn/docs/code_rust.docs] [--max-file 200000]
//!            [--max-crate 4000000] [--max-total 150000000]
//!
//! --registry defaults to `$CARGO_HOME/registry/src`, else `<home>/.cargo/registry/src`.

use ava_data::code_docs::write_code_docs;
use std::io;
use std::path::PathBuf;
use std::time::Instant;

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned()
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> io::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |k: &str, d: u64| arg(&args, k).and_then(|v| v.parse().ok()).unwrap_or(d);
    let registry = match arg(&args, "--registry") {
        Some(r) => PathBuf::from(r),
        None => {
            let home = std::env::var_os("CARGO_HOME").map(PathBuf::from).or_else(|| {
                std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME")).map(|h| PathBuf::from(h).join(".cargo"))
            });
            home.ok_or_else(|| io::Error::other("no --registry and no CARGO_HOME / home folder"))?.join("registry").join("src")
        }
    };
    let out = PathBuf::from(arg(&args, "--out").unwrap_or("data/nn/docs/code_rust.docs".into()));
    let (mf, mc, mt) = (num("--max-file", 200_000), num("--max-crate", 4_000_000), num("--max-total", 150_000_000));
    let t = Instant::now();
    let r = write_code_docs(&registry, &out, mf, mc, mt)?;
    let secs = t.elapsed().as_secs_f64();
    println!(
        "v42_code: {} crate folders; left out {} not permissive, {} without a licence field, {} older versions; {} crates used, {} files, {:.1} MB -> {} in {secs:.1}s ({:.0} MB/s)",
        r.crate_dirs,
        r.not_permissive,
        r.no_licence,
        r.older_versions,
        r.crates_used,
        r.files,
        r.bytes as f64 / 1e6,
        out.display(),
        r.bytes as f64 / 1e6 / secs.max(1e-9)
    );
    println!("  files skipped: {} over {mf} bytes, {} not UTF-8; {} crates cut at {mc} bytes", r.skipped_big, r.skipped_not_utf8, r.capped_crates);
    let mut lic: Vec<_> = r.licences.iter().collect();
    lic.sort_by(|a, b| b.1.cmp(a.1));
    println!("  licences (crates): {}", lic.iter().map(|(l, n)| format!("{l} {n}")).collect::<Vec<_>>().join(", "));
    Ok(())
}
