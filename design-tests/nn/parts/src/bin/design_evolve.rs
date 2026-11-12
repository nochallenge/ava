// TESTING / DESIGN ONLY
//! design_evolve: the H2 archive of design variants as a command-line tool, for proxy runs from the GPU
//! queue (it runs nothing itself).
//!
//!   design_evolve MENU.tsv ARCHIVE.tsv propose N      add N new variants; prints `id<TAB>flags`
//!   design_evolve MENU.tsv ARCHIVE.tsv record ID SCORE COST
//!   design_evolve MENU.tsv ARCHIVE.tsv best           the best scored variant
//!   design_evolve MENU.tsv ARCHIVE.tsv list
//!
//! MENU.tsv: one option per line, `name<TAB>value value ...`; the first value is today's default.
//! Flags print only what differs from the defaults: "on" → `--name`, "off" → nothing, else `--name value`.
//! Scores are higher-is-better (give −BPB). The archive is rewritten atomically after every change.

use ava_parts::evolve::{Archive, Menu, Variant};
use std::path::Path;

fn menu(path: &str) -> Result<Menu, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut m = Menu::new();
    for (n, line) in text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty() && !l.starts_with('#')) {
        let (name, vals) = line.split_once('\t').ok_or_else(|| format!("{path} line {}: name<TAB>values", n + 1))?;
        let vals: Vec<String> = vals.split_whitespace().map(String::from).collect();
        if vals.is_empty() {
            return Err(format!("{path} line {}: no values", n + 1));
        }
        m.push((name.trim().to_string(), vals));
    }
    Ok(m)
}

fn flags(m: &Menu, v: &Variant) -> String {
    let mut f = Vec::new();
    for (name, vals) in m {
        let x = &v[name];
        if *x == vals[0] || x == "off" {
            continue;
        }
        f.push(if x == "on" { format!("--{name}") } else { format!("--{name} {x}") });
    }
    if f.is_empty() {
        "(defaults)".into()
    } else {
        f.join(" ")
    }
}

fn save(path: &str, a: &Archive) -> Result<(), String> {
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, a.to_tsv()).map_err(|e| format!("{tmp}: {e}"))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{path}: {e}"))
}

fn run(args: &[String]) -> Result<(), String> {
    let usage = "design_evolve MENU.tsv ARCHIVE.tsv propose N | record ID SCORE COST | best | list";
    let (mp, ap, cmd) = match args {
        [m, a, c, ..] => (m.as_str(), a.as_str(), c.as_str()),
        _ => return Err(usage.into()),
    };
    let m = menu(mp)?;
    let mut a = if Path::new(ap).exists() {
        let text = std::fs::read_to_string(ap).map_err(|e| format!("{ap}: {e}"))?;
        let a = Archive::from_tsv(m.clone(), 42, &text)?;
        if a.entries.is_empty() {
            Archive::new(m.clone(), 42)
        } else {
            a
        }
    } else {
        Archive::new(m.clone(), 42)
    };
    let num = |i: usize| -> Result<f64, String> { args.get(i).and_then(|x| x.parse().ok()).ok_or_else(|| usage.to_string()) };
    match cmd {
        "propose" => {
            let n = num(3)? as usize;
            if a.entries.iter().all(|e| e.score.is_none()) && a.entries.len() == 1 {
                println!("0\t{}", flags(&m, &a.entries[0].variant));
                println!("# score the defaults first (record 0 SCORE COST); parents come from scored variants");
            }
            for id in a.propose(n) {
                println!("{id}\t{}", flags(&m, &a.entries[id].variant));
            }
            save(ap, &a)
        }
        "record" => {
            let id = num(3)? as usize;
            if id >= a.entries.len() {
                return Err(format!("no variant {id}"));
            }
            a.record(id, num(4)?, num(5)?);
            save(ap, &a)
        }
        "best" => {
            let b = a.best().ok_or("nothing scored yet")?;
            println!("{}\t{}\t{}", b.id, b.score.unwrap(), flags(&m, &b.variant));
            Ok(())
        }
        "list" => {
            for e in &a.entries {
                let s = e.score.map_or("-".into(), |x| format!("{x:.4}"));
                println!("{}\t{s}\t{}", e.id, flags(&m, &e.variant));
            }
            Ok(())
        }
        _ => Err(usage.into()),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = run(&args) {
        eprintln!("design_evolve: {e}");
        std::process::exit(1);
    }
}
