// TESTING / DESIGN ONLY — diary_day: make a sleep "day" (grow_gpu --phase sleep --facts) from one day of Ava's diary.
//
// diary_day --diary data/diary/2026-10-28.tsv --out data/diary/sleep/2026-10-28.diary-day.txt [--unrated heard|drop]
//
// Each turn gives at most one episode, in her taught format (rules and format in src/diary.rs):
//   corrected       -> "Question: {you}\nAnswer: {your correction}"   (your answer, never her wrong one)
//   rated /good     -> "Question: {you}\nAnswer: {her reply}"
//   rated /bad      -> left out (she does not rehearse her mistakes)
//   unrated         -> --unrated heard (default): your words alone, as text she heard (her unconfirmed reply is left
//                      out); --unrated drop: left out
// Episodes are separated by blank lines, which grow_gpu's episodes() reads one block per episode; the file is read
// back with a copy of that rule and checked before this tool reports success. The --out path must contain "diary"
// (so a model that slept on it can be recognised and kept from release) and may not be a diary day file itself.
// The diary is only read, never changed.

use ava_gpu::diary::{self, Unrated};
use std::path::PathBuf;

const USAGE: &str = "usage: diary_day --diary data/diary/YYYY-MM-DD.tsv --out data/diary/sleep/YYYY-MM-DD.diary-day.txt [--unrated heard|drop]";

fn main() {
    if let Err(e) = run() {
        eprintln!("diary_day: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let (mut input, mut out): (Option<PathBuf>, Option<PathBuf>) = (None, None);
    let mut unrated = Unrated::Heard;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.into());
        }
        let val = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        match key {
            "--diary" => input = Some(PathBuf::from(val)),
            "--out" => out = Some(PathBuf::from(val)),
            "--unrated" => {
                unrated = match val {
                    "heard" => Unrated::Heard,
                    "drop" => Unrated::Drop,
                    _ => return Err(format!("--unrated {val:?}: heard or drop")),
                }
            }
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let input = input.ok_or_else(|| format!("--diary is required\n{USAGE}"))?;
    let out = out.ok_or_else(|| format!("--out is required\n{USAGE}"))?;
    let date = diary::day_of(&input).ok_or_else(|| format!("{}: not a diary day file (YYYY-MM-DD.tsv)", input.display()))?;
    if !out.to_string_lossy().to_lowercase().contains("diary") {
        return Err(format!("--out {}: the path must contain \"diary\" (a day made from her diary must be recognisable)", out.display()));
    }
    if diary::day_of(&out).is_some() {
        return Err(format!("--out {}: that is a diary day file's name; the diary is never written by this tool", out.display()));
    }
    if !input.is_file() {
        return Err(format!("{}: no such diary file", input.display()));
    }

    let (lines, torn) = diary::read_day(&input)?;
    let (turns, orphans) = diary::resolve(&date, &lines);
    let (eps, c) = diary::day_episodes(&turns, unrated);
    if eps.is_empty() {
        return Err(format!(
            "{}: nothing to sleep on ({} turns: {} rated /bad, {} unrated dropped, {} empty); no file written",
            input.display(),
            c.turns,
            c.bad_dropped,
            c.unrated_dropped,
            c.empty
        ));
    }
    let text = diary::day_text(&eps);
    // what sleep will read must be exactly these episodes (a lone episode is written twice; see src/diary.rs)
    let back = diary::grow_episodes(&text);
    let want: Vec<String> = if eps.len() == 1 { vec![eps[0].clone(), eps[0].clone()] } else { eps.clone() };
    if back != want {
        return Err(format!("internal: the day file would be read as {} episodes, not the {} written", back.len(), want.len()));
    }
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(&out, &text).map_err(|e| format!("write {}: {e}", out.display()))?;
    println!(
        "{date}: {} turns -> {} episodes: {} corrected, {} confirmed (/good), {} heard (unrated: {}); left out: {} rated /bad, {} unrated, {} empty",
        c.turns,
        eps.len(),
        c.corrected,
        c.confirmed,
        c.heard,
        if unrated == Unrated::Heard { "your words kept as text" } else { "dropped" },
        c.bad_dropped,
        c.unrated_dropped,
        c.empty
    );
    if torn > 0 || orphans > 0 {
        println!("skipped {torn} torn or malformed lines and {orphans} commands that name no turn");
    }
    if eps.len() == 1 {
        println!("one episode: written twice so grow_gpu reads it as one block (identical copies sleep exactly like one)");
    }
    println!("wrote {} ({} bytes); sleep on it with grow_gpu --phase sleep --facts {}", out.display(), text.len(), out.display());
    Ok(())
}
