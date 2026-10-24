// TESTING / DESIGN ONLY. (Rust since v27, replacing school/mix.py byte for byte.)
// Mix per-domain school corpora into one file for compose.rs: every chosen domain's test
// lines first (capped, spread evenly across the held-out material), then training lines
// interleaved round-robin, so a low-memory stop never starves one domain.
//
//   cargo run --release --example school_mix -- <school-dir> <out.tsv> @tag:train_cap[:test_cap] ...
//   (cap 0 = all). prints the compose command's n and AVA_TEST.
#[path = "common/school_util.rs"]
mod util;
use std::path::Path;
use util::*;

fn spread(xs: Vec<String>, cap: usize) -> Vec<String> {
    if cap == 0 || xs.len() <= cap { return xs; }
    let step = xs.len() as f64 / cap as f64;
    (0..cap).map(|i| xs[(i as f64 * step) as usize].clone()).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 4 { eprintln!("usage: school_mix <school-dir> <out.tsv> @tag:train_cap[:test_cap] ..."); std::process::exit(1); }
    let (school, out) = (Path::new(&a[1]), Path::new(&a[2]));
    let counts = json_parse(&read_text(&school.join("counts.json")).expect("counts.json")).expect("counts.json is JSON");
    let counts = counts.get("counts").expect("counts.json has no counts");
    let (mut tests, mut trains): (Vec<String>, Vec<Vec<String>>) = (Vec::new(), Vec::new());
    for spec in &a[3..] {
        let parts: Vec<&str> = spec.split(':').collect();
        let tag = parts[0];
        let tr_cap: usize = parts.get(1).and_then(|x| x.parse().ok()).expect("train cap");
        let te_cap: usize = parts.get(2).map_or(0, |x| x.parse().expect("test cap"));
        let text = read_text(&school.join(format!("{}.tsv", &tag[1..]))).expect("domain tsv");
        let lines: Vec<String> = splitlines(&text).into_iter().map(String::from).collect();
        let nt = counts.get(tag).and_then(|c| c.get("test")).and_then(|x| x.num()).expect("tag in counts") as usize;
        let te = spread(lines[..nt.min(lines.len())].to_vec(), te_cap);
        let tr = spread(lines[nt.min(lines.len())..].to_vec(), tr_cap);
        println!("  {:<14} test {:>6}  train {:>7}", tag, te.len(), tr.len());
        tests.extend(te);
        trains.push(tr);
    }
    let mut mixed: Vec<String> = Vec::new();
    for i in 0..trains.iter().map(|t| t.len()).max().unwrap_or(0) {
        for t in &trains { if i < t.len() { mixed.push(t[i].clone()); } }
    }
    let body = tests.iter().chain(&mixed).cloned().collect::<Vec<_>>().join("\n") + "\n";
    std::fs::write(out, body).expect("write the mix");
    // Python printed the path through Path(), which shows backslashes on Windows
    let shown = if cfg!(windows) { out.display().to_string().replace('/', "\\") } else { out.display().to_string() };
    println!("n={} AVA_TEST={}  -> {}", mixed.len(), tests.len(), shown);
}
