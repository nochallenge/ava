// TESTING / DESIGN ONLY — lookup (v43): build her library, or ask it a question. See src/lookup.rs.
//
// lookup build --docs data/nn/docs/simplewiki.docs [--facts data/v43/teach/facts.tsv] --out data/v43/library.tsv
// lookup ask --library data/v43/library.tsv [--k 2] QUESTION...

use ava_gpu::lookup::{build, Library};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    if let Err(e) = run() {
        eprintln!("lookup: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    match args.first().map(String::as_str) {
        Some("build") => {
            let docs = PathBuf::from(get("--docs").ok_or("--docs FILE")?);
            let out = PathBuf::from(get("--out").ok_or("--out FILE")?);
            let facts = get("--facts").map(PathBuf::from);
            let t = Instant::now();
            let (arts, n) = build(&docs, facts.as_deref(), &out)?;
            println!("{} passages ({arts} articles) -> {} in {:.1}s", n, out.display(), t.elapsed().as_secs_f64());
            Ok(())
        }
        Some("ask") => {
            let lib_path = PathBuf::from(get("--library").ok_or("--library FILE")?);
            let k: usize = get("--k").map(|v| v.parse().map_err(|_| "bad --k")).transpose()?.unwrap_or(2);
            let mut q: Vec<&str> = Vec::new();
            let mut i = 1;
            while i < args.len() {
                if args[i].starts_with("--") {
                    i += 2;
                } else {
                    q.push(&args[i]);
                    i += 1;
                }
            }
            let t = Instant::now();
            let lib = Library::load(&lib_path)?;
            println!("loaded {} passages in {:.1}s", lib.len(), t.elapsed().as_secs_f64());
            for question in q.join(" ").split(" | ") {
                let t = Instant::now();
                let hits = lib.search(question, k);
                println!("\nQ: {question}   ({:.1} ms)", t.elapsed().as_secs_f64() * 1e3);
                for (s, title, text) in hits {
                    println!("  {s:6.2}  {title}: {}", text.chars().take(160).collect::<String>());
                }
            }
            Ok(())
        }
        _ => Err("usage: lookup build --docs F [--facts F] --out F | lookup ask --library F [--k 2] QUESTION [| QUESTION ...]".into()),
    }
}
