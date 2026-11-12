// TESTING / DESIGN ONLY — v43_prep: her lessons for answering from her library (v43, runs/v43/prereg_answers.md).
//
// v43_prep --library FILE --teach DIR --sciq FILE --squad FILE --out FILE [--glance] [--copies 3]
//          [--sciq-n 4000] [--squad-n 6000] [--squad-unans 2000] [--max-len 512] [--seed 1] [--bpe FILE]
//          [--self-q FILE --self-items FILE [--self-copies 3] [--self-skip CAT,..]] [--drills N] [--drills2 N]
//
// Every example is  [EOS] prompt answer [EOS]  as in qa_prep (AVASFT01, read by sft_gpu), with
//   prompt = "Context: {c}\nQuestion: {q}\nAnswer:"  ("Question: {q}\nAnswer:" when c is empty), answer = " {a}",
// encoded exactly as the chat app encodes (diary::encode; --glance for a model with the glance's number marks).
// Sources:
//   0 teach/qa.tsv     her simple questions; c = her library's best 2 passages for q (what she will see in chat);
//                      "I don't know." items included. Each written --copies times.
//   1 teach/chat.tsv   small talk and who she is; c = her library's best 2 passages (usually nothing). --copies times.
//   2 SciQ train       question -> correct answer, c = its support passage (items without one skipped); --sciq-n.
//   3 SQuAD 2.0 train  answerable: c = "Title: " + the answer's sentence and the one before it, plus a passage from
//                      another article, in random order; a = the first answer text; --squad-n.
//   4 SQuAD 2.0 train  unanswerable: c = "Title: " + the paragraph's first two sentences, plus another article's
//                      passage; a = "I don't know."; --squad-unans.
//   5 her own mistakes (round 2): questions she never trained on (--self-q: id, cat, question, accept, answer),
//                      run through the taught model by answer_eval (--self-items: id, cat, grade, answer, ...);
//                      c = her library's best 2 passages; a = the written answer where she was Right, else "I don't know."
//                      (learning what she does not know from what she got wrong). --self-copies times.
//                      --self-skip leaves out categories (round 2: math, which the drills teach instead).
//   6 maths practice   --drills N generated sums (add/take away within 100, times tables, exact division), asked
//                      the casual ways, a = "a plus b is r."; c = her library's best 2 passages, as in chat.
// Items are drawn with a seeded shuffle; all sources are mixed in one shuffled file.

use ava_data::bpe::{Bpe, EOS};
use ava_data::rng::SplitMix64;
use ava_gpu::diary;
use ava_gpu::lookup::Library;
use ava_gpu::qa::prompt_text;
use serde_json::Value;
use std::io::Write;
use std::path::PathBuf;

const SOURCES: [&str; 8] = ["teach-qa", "teach-chat", "sciq", "squad", "squad-unans", "self", "drills", "drills2"];
const IDK: &str = "I don't know.";

struct Ex {
    src: u8,
    np: usize,
    toks: Vec<u32>,
}

fn main() {
    if let Err(e) = run() {
        eprintln!("v43_prep: {e}");
        std::process::exit(1);
    }
}

fn sentences(p: &str) -> Vec<(usize, &str)> {
    // (start byte, sentence) at ". ", "! ", "? "
    let mut out = Vec::new();
    let b = p.as_bytes();
    let mut start = 0;
    for i in 0..b.len() {
        if matches!(b[i], b'.' | b'!' | b'?') && (i + 1 == b.len() || b[i + 1] == b' ') {
            out.push((start, p[start..=i].trim()));
            start = i + 1;
        }
    }
    if start < p.len() && !p[start..].trim().is_empty() {
        out.push((start, p[start..].trim()));
    }
    out
}

fn words_max(s: &str, n: usize) -> String {
    s.split_whitespace().take(n).collect::<Vec<_>>().join(" ")
}

/// 0..=999 in words ("twenty-four").
fn say_num(n: u32) -> String {
    const ONES: [&str; 20] = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"];
    const TENS: [&str; 10] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
    match n {
        0..=19 => ONES[n as usize].into(),
        20..=99 => if n % 10 == 0 { TENS[(n / 10) as usize].into() } else { format!("{}-{}", TENS[(n / 10) as usize], ONES[(n % 10) as usize]) },
        _ => if n % 100 == 0 { format!("{} hundred", ONES[(n / 100) as usize]) } else { format!("{} hundred and {}", ONES[(n / 100) as usize], say_num(n % 100)) },
    }
}

/// One maths practice question and its answer: add or take away within 100 (sums to 200), times tables to 12,
/// exact division; numbers in digits or (1 in 5) words; asked one of the casual ways.
fn drill(rng: &mut SplitMix64) -> (String, String) {
    let r = |rng: &mut SplitMix64, n: u64| rng.below(n) as u32;
    let op = r(rng, 4);
    let (a, b, res, word, syms): (u32, u32, u32, &str, &[&str]) = match op {
        0 => { let a = r(rng, 101); let b = if r(rng, 5) == 0 { r(rng, 101) } else { r(rng, (101 - a) as u64) }; (a, b, a + b, "plus", &["plus", "+", "add"]) }
        1 => { let a = r(rng, 101); let b = r(rng, a as u64 + 1); (a, b, a - b, "minus", &["minus", "-", "take away", "subtract"]) }
        2 => { let a = r(rng, 13); let b = r(rng, 13); (a, b, a * b, "times", &["times", "x", "*", "multiplied by", "multiply"]) }
        _ => { let b = 1 + r(rng, 12); let q = r(rng, 13); (b * q, b, q, "divided by", &["divided by", "/", "divide"]) }
    };
    let words = r(rng, 5) == 0;
    let n = |x: u32| if words { say_num(x) } else { x.to_string() };
    let (x, y) = (n(a), n(b));
    let sym = syms[r(rng, syms.len() as u64) as usize];
    let q = match sym {
        "add" => format!("add {x} and {y}"),
        "subtract" => format!("subtract {y} from {x}"),
        "multiply" => format!("multiply {x} by {y}"),
        "divide" => format!("divide {x} by {y}"),
        "+" | "-" | "*" | "/" | "x" if !words && r(rng, 2) == 0 => format!("{x}{sym}{y}"),
        _ => {
            let forms = ["what is {}", "whats {}", "what's {}?", "how much is {}", "ava what is {}", "can you tell me {}", "{} equals what", "{}", "what is {}?", "quick one: {}"];
            forms[r(rng, forms.len() as u64) as usize].replace("{}", &format!("{x} {sym} {y}"))
        }
    };
    (q, format!("{a} {word} {b} is {res}."))
}

const THINGS: [&str; 8] = ["candies", "apples", "toys", "books", "stickers", "cookies", "marbles", "pencils"];
const LEGS: [(&str, &str, u32); 10] = [("dog", "dogs", 4), ("cat", "cats", 4), ("cow", "cows", 4), ("horse", "horses", 4), ("spider", "spiders", 8), ("bird", "birds", 2), ("chicken", "chickens", 2), ("ant", "ants", 6), ("bee", "bees", 6), ("person", "people", 2)];

/// Round 3 maths practice: (question, answer, the number the answer must hold). Small numbers more often (a child's
/// sums), counting, word problems, legs, which is bigger, and drill()'s sums.
fn drill2(rng: &mut SplitMix64) -> (String, String, u32) {
    let r = |rng: &mut SplitMix64, n: u32| rng.below(n as u64) as u32;
    let small = |rng: &mut SplitMix64| if r(rng, 5) < 2 { r(rng, 21) } else { r(rng, 101) };
    match r(rng, 20) {
        0..=10 => {
            // sums, small numbers 2 times in 5
            let op = r(rng, 4);
            let (a, b, res, w) = match op {
                0 => { let a = small(rng); let b = r(rng, 101 - a); (a, b, a + b, "plus") }
                1 => { let a = small(rng); let b = r(rng, a + 1); (a, b, a - b, "minus") }
                2 => { let a = r(rng, 13); let b = r(rng, 13); (a, b, a * b, "times") }
                _ => { let b = 1 + r(rng, 12); let q = r(rng, 13); (b * q, b, q, "divided by") }
            };
            let sym = match (op, r(rng, 2)) { (0, 0) => "+", (1, 0) => "-", (2, 0) => "x", (3, 0) => "/", _ => w };
            let forms = ["what is {}", "whats {}", "what's {}?", "how much is {}", "{}", "ava what is {}"];
            (forms[r(rng, forms.len() as u32) as usize].replace("{}", &format!("{a} {sym} {b}")), format!("{a} {w} {b} is {res}."), res)
        }
        11..=12 => {
            let n = 1 + r(rng, 99);
            if r(rng, 2) == 0 {
                let f = ["what number comes after {}", "what comes after {}", "whats the number after {}"][r(rng, 3) as usize];
                (f.replace("{}", &n.to_string()), format!("{} comes after {n}.", n + 1), n + 1)
            } else {
                let f = ["what number comes before {}", "what comes before {}", "whats the number before {}"][r(rng, 3) as usize];
                (f.replace("{}", &n.to_string()), format!("{} comes before {n}.", n - 1), n - 1)
            }
        }
        13..=15 => {
            let t = THINGS[r(rng, THINGS.len() as u32) as usize];
            let a = 1 + r(rng, 20);
            if r(rng, 2) == 0 {
                let b = r(rng, a + 1);
                (format!("if you have {a} {t} and give away {b} how many do you have left"), format!("You have {} {t} left.", a - b), a - b)
            } else {
                let b = r(rng, 21);
                (format!("if you have {a} {t} and get {b} more how many do you have"), format!("You have {} {t}.", a + b), a + b)
            }
        }
        16..=17 => {
            let (one, many, l) = LEGS[r(rng, LEGS.len() as u32) as usize];
            let n = 1 + r(rng, 5);
            if n == 1 {
                (format!("how many legs does a {one} have"), format!("A {one} has {l} legs."), l)
            } else {
                (format!("how many legs do {n} {many} have"), format!("{n} {many} have {} legs.", n * l), n * l)
            }
        }
        _ => {
            let a = r(rng, 101);
            let mut b = r(rng, 101);
            if b == a {
                b = (a + 1 + r(rng, 9)) % 101;
            }
            let (hi, lo) = if a > b { (a, b) } else { (b, a) };
            (format!("which is bigger {a} or {b}"), format!("{hi} is bigger than {lo}."), hi)
        }
    }
}

fn tsv(path: &PathBuf) -> Result<Vec<(String, String)>, String> {
    let t = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(t.lines().skip(1).filter_map(|l| l.split_once('\t')).map(|(a, b)| (a.trim().to_string(), b.trim().to_string())).filter(|(a, b)| !a.is_empty() && !b.is_empty()).collect())
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let num = |k: &str, d: usize| -> Result<usize, String> { get(k).map(|v| v.parse().map_err(|_| format!("bad {k}"))).transpose().map(|v| v.unwrap_or(d)) };
    let need = |k: &str| get(k).map(PathBuf::from).ok_or_else(|| format!("{k} is required"));
    let lib = Library::load(&need("--library")?)?;
    let teach = need("--teach")?;
    let out = need("--out")?;
    let glance = args.iter().any(|a| a == "--glance");
    let copies = num("--copies", 3)?;
    let max_len = num("--max-len", 512)?;
    let mut rng = SplitMix64::new(num("--seed", 1)? as u64);
    let bpe_path = PathBuf::from(get("--bpe").unwrap_or_else(|| ava_gpu::DEFAULT_BPE.into()));
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;

    let mut exs: Vec<Ex> = Vec::new();
    let mut stats = [[0usize; 2]; 8]; // kept, dropped (too long)
    let mut add = |src: u8, ctx: &str, q: &str, a: &str, exs: &mut Vec<Ex>| {
        let mut toks = vec![EOS];
        toks.extend(diary::encode(&bpe, glance, &prompt_text(ctx, &diary::one_line(q))));
        let np = toks.len();
        toks.extend(diary::encode(&bpe, glance, &format!(" {}", diary::one_line(a))));
        toks.push(EOS);
        if toks.len() > max_len || toks.iter().any(|&t| t > u16::MAX as u32) {
            stats[src as usize][1] += 1;
            return;
        }
        stats[src as usize][0] += 1;
        exs.push(Ex { src, np, toks });
    };

    // 0, 1: written for her
    let qa = tsv(&teach.join("qa.tsv"))?;
    let chat = tsv(&teach.join("chat.tsv"))?;
    for _ in 0..copies {
        for (q, a) in &qa {
            add(0, &lib.context(q, 2), q, a, &mut exs);
        }
        for (u, a) in &chat {
            add(1, &lib.context(u, 2), u, a, &mut exs);
        }
    }

    // 2: SciQ train with its support passage
    let sciq: Value = serde_json::from_str(&std::fs::read_to_string(need("--sciq")?).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let mut items: Vec<&Value> = sciq.as_array().ok_or("sciq: not a list")?.iter().filter(|v| v["support"].as_str().is_some_and(|s| !s.trim().is_empty())).collect();
    rng.shuffle(&mut items);
    for v in items.into_iter().take(num("--sciq-n", 4000)?) {
        let (q, a, c) = (v["question"].as_str().unwrap_or(""), v["correct_answer"].as_str().unwrap_or(""), v["support"].as_str().unwrap_or(""));
        if !q.is_empty() && !a.is_empty() {
            add(2, &words_max(c, 120), q, a, &mut exs);
        }
    }

    // 3, 4: SQuAD 2.0 train
    let squad: Value = serde_json::from_str(&std::fs::read_to_string(need("--squad")?).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let mut ans: Vec<(String, String, String)> = Vec::new(); // (context, question, answer)
    let mut unans: Vec<(String, String)> = Vec::new();
    let mut openings: Vec<String> = Vec::new();
    for art in squad["data"].as_array().ok_or("squad: no data")? {
        let title = art["title"].as_str().unwrap_or("").replace('_', " ");
        for p in art["paragraphs"].as_array().into_iter().flatten() {
            let text = p["context"].as_str().unwrap_or("");
            let sents = sentences(text);
            if sents.is_empty() {
                continue;
            }
            let opening = format!("{title}: {}", words_max(&sents.iter().take(2).map(|s| s.1).collect::<Vec<_>>().join(" "), 60));
            openings.push(opening.clone());
            for qa in p["qas"].as_array().into_iter().flatten() {
                let q = qa["question"].as_str().unwrap_or("").to_string();
                if qa["is_impossible"].as_bool().unwrap_or(false) {
                    unans.push((opening.clone(), q));
                    continue;
                }
                let Some(a0) = qa["answers"].as_array().and_then(|a| a.first()) else { continue };
                let (at, start) = (a0["text"].as_str().unwrap_or(""), a0["answer_start"].as_u64().unwrap_or(0) as usize);
                let Some(k) = sents.iter().rposition(|s| s.0 <= start) else { continue };
                let mut c = sents[k].1.to_string();
                if k > 0 {
                    c = format!("{} {c}", sents[k - 1].1);
                }
                if !c.contains(at) || at.is_empty() {
                    continue;
                }
                ans.push((format!("{title}: {}", words_max(&c, 80)), q, at.to_string()));
            }
        }
    }
    rng.shuffle(&mut ans);
    rng.shuffle(&mut unans);
    let n_open = openings.len() as u64;
    for (c, q, a) in ans.into_iter().take(num("--squad-n", 6000)?) {
        let other = &openings[rng.below(n_open) as usize];
        let ctx = if rng.below(2) == 0 { format!("{c} {other}") } else { format!("{other} {c}") };
        add(3, &ctx, &q, &a, &mut exs);
    }
    for (c, q) in unans.into_iter().take(num("--squad-unans", 2000)?) {
        let other = &openings[rng.below(n_open) as usize];
        let ctx = if rng.below(2) == 0 { format!("{c} {other}") } else { format!("{other} {c}") };
        add(4, &ctx, &q, IDK, &mut exs);
    }

    // 5: her own mistakes (round 2)
    if let (Some(qf), Some(itf)) = (get("--self-q"), get("--self-items")) {
        let read = |p: &str| std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"));
        let (qt, it) = (read(&qf)?, read(&itf)?);
        let grades: std::collections::HashMap<&str, &str> =
            it.lines().skip(1).filter_map(|l| { let c: Vec<&str> = l.split('\t').collect(); (c.len() >= 3).then(|| (c[0], c[2])) }).collect();
        let skip_s = get("--self-skip").unwrap_or_default();
        let skip: Vec<&str> = skip_s.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
        let mut rows: Vec<(String, String)> = Vec::new();
        let (mut right, mut idk) = (0, 0);
        for l in qt.lines().skip(1) {
            let c: Vec<&str> = l.split('\t').collect();
            let Some(g) = (c.len() >= 5).then(|| grades.get(c[0])).flatten() else { continue };
            if skip.contains(&c[1].trim()) {
                continue;
            }
            let a = if *g == "Right" { right += 1; c[4] } else { idk += 1; IDK };
            rows.push((c[2].trim().to_string(), a.trim().to_string()));
        }
        println!("self: {} questions graded: {right} right -> her answer, {idk} wrong or unsure -> \"{IDK}\"", rows.len());
        for _ in 0..num("--self-copies", 3)? {
            for (q, a) in &rows {
                add(5, &lib.context(q, 2), q, a, &mut exs);
            }
        }
    }

    // 6: maths practice (round 2): --drills N sums asked the casual ways, answered "a plus b is r."
    for _ in 0..num("--drills", 0)? {
        let (q, a) = drill(&mut rng);
        add(6, &lib.context(&q, 2), &q, &a, &mut exs);
    }
    // 7: round 3 maths practice: --drills2 N (drill2: small sums, counting, word problems, legs, which is bigger)
    for _ in 0..num("--drills2", 0)? {
        let (q, a, _) = drill2(&mut rng);
        add(7, &lib.context(&q, 2), &q, &a, &mut exs);
    }

    rng.shuffle(&mut exs);
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"AVASFT01");
    buf.extend_from_slice(&(exs.len() as u64).to_le_bytes());
    let (mut tokens, mut scored) = (0usize, 0usize);
    for e in &exs {
        buf.push(e.src);
        buf.extend_from_slice(&(e.np as u32).to_le_bytes());
        buf.extend_from_slice(&(e.toks.len() as u32).to_le_bytes());
        for &t in &e.toks {
            buf.extend_from_slice(&(t as u16).to_le_bytes());
        }
        tokens += e.toks.len();
        scored += e.toks.len() - e.np;
    }
    if let Some(d) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(d).map_err(|e| e.to_string())?;
    }
    std::fs::File::create(&out).and_then(|mut f| f.write_all(&buf)).map_err(|e| format!("{}: {e}", out.display()))?;
    for (k, s) in SOURCES.iter().enumerate() {
        println!("{s:<12} kept {:>6}  too long {:>4}", stats[k][0], stats[k][1]);
    }
    println!("{} examples, {tokens} tokens ({scored} scored), glance {glance} -> {}", exs.len(), out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_are_said_in_words() {
        assert_eq!(say_num(0), "zero");
        assert_eq!(say_num(24), "twenty-four");
        assert_eq!(say_num(90), "ninety");
        assert_eq!(say_num(100), "one hundred");
        assert_eq!(say_num(144), "one hundred and forty-four");
    }

    #[test]
    fn every_practice_answer_is_right() {
        let mut rng = SplitMix64::new(7);
        let mut ops = [0usize; 4];
        for _ in 0..5000 {
            let (q, a) = drill(&mut rng);
            assert!(!q.is_empty());
            let w: Vec<&str> = a.trim_end_matches('.').split(' ').collect();
            let (x, y, r): (i64, i64, i64) = (w[0].parse().unwrap(), w[w.len() - 3].parse().unwrap(), w[w.len() - 1].parse().unwrap());
            let (k, want) = match w[1] {
                "plus" => (0, x + y),
                "minus" => (1, x - y),
                "times" => (2, x * y),
                _ => (3, x / y),
            };
            assert_eq!(r, want, "{q} -> {a}");
            assert!(r >= 0 && (k != 3 || x % y == 0), "{q} -> {a}");
            ops[k] += 1;
        }
        assert!(ops.iter().all(|&n| n > 1000), "{ops:?}");
    }

    #[test]
    fn every_round3_practice_answer_is_right() {
        let mut rng = SplitMix64::new(9);
        let nums = |s: &str| -> Vec<u32> { s.split(|c: char| !c.is_ascii_digit()).filter_map(|w| w.parse().ok()).collect() };
        let mut kinds = [0usize; 5];
        for _ in 0..5000 {
            let (q, a, res) = drill2(&mut rng);
            assert!(nums(&a).contains(&res), "{q} -> {a}");
            let n = nums(&q);
            let (k, want) = if q.contains("after") {
                (1, n[0] + 1)
            } else if q.contains("before") {
                (1, n[0] - 1)
            } else if q.contains("give away") {
                (2, n[0] - n[1])
            } else if q.contains("more how many") {
                (2, n[0] + n[1])
            } else if q.contains("legs") {
                let l = LEGS.iter().find(|x| q.contains(&format!(" {} ", x.0)) || q.ends_with(&format!(" {} have", x.1)) || q.contains(&format!(" {} have", x.1))).unwrap().2;
                (3, n.first().copied().unwrap_or(1) * l)
            } else if q.contains("bigger") {
                (4, n[0].max(n[1]))
            } else {
                let w: Vec<&str> = a.trim_end_matches('.').split(' ').collect();
                let (x, y) = (n[0], n[1]);
                (0, match w[1] { "plus" => x + y, "minus" => x - y, "times" => x * y, _ => x / y })
            };
            assert_eq!(res, want, "{q} -> {a}");
            kinds[k] += 1;
        }
        assert!(kinds.iter().all(|&c| c > 300), "{kinds:?}");
    }
}
