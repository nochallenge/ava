// TESTING / DESIGN ONLY.
// Teaching + sleep test (v24; Rust since v27, replacing school/teach_test.py byte for byte).
// Builds the talk scripts from her VERIFIED facts (facts the two checkers rejected are
// skipped), then scores the transcripts. Nothing here runs inside Ava: it writes what a
// teacher would say to her and grades what she said back.
//
//   teach_test make     <ava-root> <out-dir>
//       day1.txt: ask, teach ("= answer"), ask again (same day)
//       day2.txt: the 3 phrasings of each taught fact, then 20 UNTAUGHT facts (false recall)
//   teach_test score    <ava-root> <day1-output> <day2-output>
//   teach_test practice <ava-root> <out-dir>      (v26, retrieval practice)
//       day2p.txt: each taught fact asked in its 2nd phrasing, then corrected ("= answer")
//       day3.txt : each taught fact in its 3rd phrasing (never seen), then the 20 untaught
//   teach_test score3   <ava-root> <day2-output> <day3-output>
//       3rd phrasings + untaught: without practice (from day 2) vs after practice (day 3)
//   teach_test compare  <ava-root> <facts> <at> <arm-A day2> <arm-B day2> [<facts> <at> <A> <B> ...]
//       (v27) two arms on the same blocks: right, swaps, false recalls, net = right - wrong,
//       share right among taught answers given, and a paired bootstrap over facts of the
//       change in net (20,000 resamples, seed 1)
//   AVA_FACTS=<name> picks the fact file (self, ai, rustdoc, refnomicon; default self)
//   AVA_FACTS_AT=<k> skips the first k kept facts (a fresh set)
//
// run from design-tests:  cargo run --release --example teach_test -- <mode> ...

use std::env;
use std::fs;
use std::process::exit;

const N: usize = 20;
const NL: &str = if cfg!(windows) { "\r\n" } else { "\n" };

// ---- a small JSON reader (the fact files are JSON lines; no crates needed) ----
enum J { Null, Bool, Num, Str(String), Arr(Vec<J>), Obj(Vec<(String, J)>) }

struct P<'a> { s: &'a [u8], i: usize }
impl<'a> P<'a> {
    fn ws(&mut self) { while self.i < self.s.len() && b" \t\r\n".contains(&self.s[self.i]) { self.i += 1; } }
    fn val(&mut self) -> Result<J, String> {
        self.ws();
        match self.s.get(self.i) {
            Some(b'{') => {
                self.i += 1;
                let mut o = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b'}') { self.i += 1; return Ok(J::Obj(o)); }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.ws();
                    if self.s.get(self.i) != Some(&b':') { return Err(format!("expected ':' at {}", self.i)); }
                    self.i += 1;
                    let v = self.val()?;
                    o.push((k, v));
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => { self.i += 1; return Ok(J::Obj(o)); }
                        _ => return Err(format!("expected ',' or '}}' at {}", self.i)),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b']') { self.i += 1; return Ok(J::Arr(a)); }
                loop {
                    a.push(self.val()?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => { self.i += 1; return Ok(J::Arr(a)); }
                        _ => return Err(format!("expected ',' or ']' at {}", self.i)),
                    }
                }
            }
            Some(b'"') => Ok(J::Str(self.string()?)),
            Some(b't') if self.s[self.i..].starts_with(b"true") => { self.i += 4; Ok(J::Bool) }
            Some(b'f') if self.s[self.i..].starts_with(b"false") => { self.i += 5; Ok(J::Bool) }
            Some(b'n') if self.s[self.i..].starts_with(b"null") => { self.i += 4; Ok(J::Null) }
            Some(_) => {
                let st = self.i;
                while self.i < self.s.len() && b"+-0123456789.eE".contains(&self.s[self.i]) { self.i += 1; }
                let t = std::str::from_utf8(&self.s[st..self.i]).map_err(|e| e.to_string())?;
                t.parse::<f64>().map(|_| J::Num).map_err(|_| format!("bad value at {st}"))
            }
            None => Err("unexpected end".into()),
        }
    }
    fn hex4(&mut self) -> Result<u32, String> {
        let h = self.s.get(self.i..self.i + 4).ok_or("short \\u escape")?;
        self.i += 4;
        u32::from_str_radix(std::str::from_utf8(h).map_err(|e| e.to_string())?, 16).map_err(|e| e.to_string())
    }
    fn string(&mut self) -> Result<String, String> {
        if self.s.get(self.i) != Some(&b'"') { return Err(format!("expected a string at {}", self.i)); }
        self.i += 1;
        let mut b: Vec<u8> = Vec::new();
        loop {
            let c = *self.s.get(self.i).ok_or("unterminated string")?;
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = *self.s.get(self.i).ok_or("bad escape")?;
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"', b'\\' => '\\', b'/' => '/', b'b' => '\u{8}', b'f' => '\u{c}',
                        b'n' => '\n', b'r' => '\r', b't' => '\t',
                        b'u' => {
                            let mut u = self.hex4()?;
                            if (0xd800..0xdc00).contains(&u) && self.s[self.i..].starts_with(b"\\u") {
                                self.i += 2;
                                let lo = self.hex4()?;
                                u = 0x10000 + ((u - 0xd800) << 10) + (lo - 0xdc00);
                            }
                            char::from_u32(u).unwrap_or('\u{fffd}')
                        }
                        _ => return Err(format!("bad escape at {}", self.i)),
                    };
                    let mut t = [0u8; 4];
                    b.extend_from_slice(ch.encode_utf8(&mut t).as_bytes());
                }
                _ => b.push(c),
            }
        }
        String::from_utf8(b).map_err(|e| e.to_string())
    }
}
fn parse(s: &str) -> Result<J, String> { P { s: s.as_bytes(), i: 0 }.val() }
fn get<'a>(o: &'a J, k: &str) -> Option<&'a J> {
    if let J::Obj(v) = o { v.iter().find(|x| x.0 == k).map(|x| &x.1) } else { None }
}
fn as_str(j: Option<&J>) -> String { if let Some(J::Str(s)) = j { s.clone() } else { String::new() } }

struct Fact { id: String, q: Vec<String>, a: String }

/// the kept facts at [at, at + n): verified (not in failed.json) and with 3+ phrasings
fn facts(root: &str, which: &str, at: usize, n: usize) -> Vec<Fact> {
    let d = format!("{root}/data/school/facts");
    let bad: Vec<String> = match parse(&read(&format!("{d}/failed.json"))) {
        Ok(J::Arr(v)) => v.iter().filter_map(|x| if let J::Str(s) = x { Some(s.clone()) } else { None }).collect(),
        _ => die("failed.json is not a list"),
    };
    let mut out = Vec::new();
    for line in read(&format!("{d}/{which}.jsonl")).lines() {
        let f = parse(line).unwrap_or_else(|e| die(&format!("{which}.jsonl: {e}")));
        let id = as_str(get(&f, "id"));
        let q: Vec<String> = match get(&f, "q") {
            Some(J::Arr(v)) => v.iter().map(|x| if let J::Str(s) = x { s.clone() } else { String::new() }).collect(),
            _ => Vec::new(),
        };
        if !bad.contains(&id) && q.len() >= 3 {
            out.push(Fact { id, q, a: as_str(get(&f, "a")) });
        }
        if out.len() == at + n { break; }
    }
    out.into_iter().skip(at).collect()
}

/// the same split talk_tokenize makes for plain english: words and punctuation apart
fn norm(s: &str) -> String {
    let s = s.to_lowercase().replace('\u{2019}', "'");
    let mut toks: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '\'' { cur.push(c); continue; }
        if !cur.is_empty() { toks.push(std::mem::take(&mut cur)); }
        // what Python's \s counts as space: Unicode white space plus the separators 1c-1f
        if !(c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)) { toks.push(c.to_string()); }
    }
    if !cur.is_empty() { toks.push(cur); }
    toks.join(" ")
}

/// everything she said ("ava> ..."), normalised, in order
fn replies(path: &str) -> Vec<String> {
    let t = read(path).replace("\r\n", "\n").replace('\r', "\n");
    t.split('\n').filter_map(|l| l.find("ava> ").map(|i| norm(&l[i + 5..]))).collect()
}

/// a string as Python's repr() shows it (used in the per-fact lines)
fn py_repr(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut o = String::new();
    o.push(q);
    for c in s.chars() {
        let u = c as u32;
        let unprintable = u < 0x20 || (0x7f..0xa1).contains(&u) || u == 0xad || u == 0x1680
            || (0x2000..0x2010).contains(&u) || (0x2028..0x2030).contains(&u) || (0x205f..0x2065).contains(&u)
            || (0x2066..0x2070).contains(&u) || u == 0x3000 || u == 0xfeff || (0xfff9..0xfffc).contains(&u)
            || (0xe000..0xf900).contains(&u) || u >= 0xf0000;
        match c {
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if c == q => { o.push('\\'); o.push(c); }
            _ if unprintable => o.push_str(&if u < 0x100 { format!("\\x{u:02x}") } else if u < 0x10000 { format!("\\u{u:04x}") } else { format!("\\U{u:08x}") }),
            c => o.push(c),
        }
    }
    o.push(q);
    o
}

fn read(p: &str) -> String { fs::read_to_string(p).unwrap_or_else(|e| die(&format!("{p}: {e}"))) }
fn die(m: &str) -> ! { eprintln!("teach_test: {m}"); exit(1) }
fn write(path: String, lines: &[String]) {
    fs::write(&path, lines.iter().map(|l| format!("{l}{NL}")).collect::<String>()).unwrap_or_else(|e| die(&format!("{path}: {e}")));
}
fn at_or_die(v: &[String], i: usize, what: &str) -> String {
    v.get(i).cloned().unwrap_or_else(|| die(&format!("{what} is too short: needs reply {} of {}", i + 1, v.len())))
}

// ---- v27: two arms, the measure "right - wrong" ----
struct Arm { q0: usize, right: usize, swaps: usize, false_: usize, net: Vec<i32>, a1_pos: Vec<f32>, a1_neg: Vec<f32>, top1: Option<f64> }
fn arm(path: &str, taught: &[Fact], untaught: usize) -> Arm {
    let said = replies(path);
    let want: Vec<String> = taught.iter().map(|f| norm(&f.a)).collect();
    let n = taught.len();
    let txt = read(path);
    // the bracket reading after each reply: the recalled lesson's activation a1 (0 if none)
    let a1: Vec<f32> = txt.lines().filter(|l| l.contains("[reading:")).map(|l| {
        l.find("reminds her of lesson ").and_then(|i| {
            let rest = &l[i..];
            let p = rest.find('(')? + 1;
            rest[p..].split('%').next()?.trim().parse::<f32>().ok().map(|x| x / 100.0)
        }).unwrap_or(0.0)
    }).collect();
    let top1 = txt.lines().find_map(|l| l.split("next word : top-1 ").nth(1)).and_then(|r| r.split('%').next()?.trim().parse().ok());
    let s = |i: usize| at_or_die(&said, i, path);
    let mut a = Arm { q0: 0, right: 0, swaps: 0, false_: 0, net: Vec::new(), a1_pos: Vec::new(), a1_neg: Vec::new(), top1 };
    for i in 0..n {
        if s(3 * i) == want[i] { a.q0 += 1; }
        let mut net = 0;
        for k in [1, 2] {
            let x = s(3 * i + k);
            let r = a1.get(3 * i + k).copied().unwrap_or(0.0);
            if x == want[i] { a.right += 1; net += 1; a.a1_pos.push(r); }
            else if want.contains(&x) { a.swaps += 1; net -= 1; a.a1_neg.push(r); }
        }
        a.net.push(net);
    }
    for j in 0..untaught {
        let x = s(3 * n + j);
        if want.contains(&x) { a.false_ += 1; a.net.push(-1); a.a1_neg.push(a1.get(3 * n + j).copied().unwrap_or(0.0)); }
        else { a.net.push(0); }
    }
    a
}
fn auc(pos: &[f32], neg: &[f32]) -> f64 {
    if pos.is_empty() || neg.is_empty() { return f64::NAN; }
    let mut s = 0.0;
    for p in pos { for q in neg { s += if p > q { 1.0 } else if p == q { 0.5 } else { 0.0 }; } }
    s / (pos.len() * neg.len()) as f64
}
fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn main() {
    let a: Vec<String> = env::args().collect();
    if a.len() < 3 { die("usage: teach_test <make|practice|score|score3|compare> <ava-root> ..."); }
    let (mode, root) = (a[1].as_str(), a[2].as_str());
    let which = env::var("AVA_FACTS").unwrap_or_else(|_| "self".into());
    let at: usize = env::var("AVA_FACTS_AT").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    if mode == "compare" {
        return compare(root, &a[3..]);
    }
    let all = facts(root, &which, at, 2 * N);
    let (taught, untaught) = all.split_at(N.min(all.len()));
    let arg = |i: usize| a.get(i).cloned().unwrap_or_else(|| die(&format!("{mode} needs more arguments")));
    match mode {
        "make" => {
            let out = arg(3);
            let mut d1 = Vec::new();
            for f in taught { d1.push(f.q[0].clone()); d1.push(format!("= {}", f.a)); d1.push(f.q[0].clone()); }
            d1.push(":q".into());
            write(format!("{out}/day1.txt"), &d1);
            let mut d2: Vec<String> = taught.iter().flat_map(|f| f.q[..3].to_vec()).collect();
            d2.extend(untaught.iter().map(|f| f.q[0].clone()));
            d2.push(":q".into());
            write(format!("{out}/day2.txt"), &d2);
            println!("{} taught: {}", taught.len(), taught.iter().map(|f| f.id.as_str()).collect::<Vec<_>>().join(", "));
            println!("{} untaught: {}", untaught.len(), untaught.iter().map(|f| f.id.as_str()).collect::<Vec<_>>().join(", "));
        }
        "practice" => {
            let out = arg(3);
            let mut p: Vec<String> = Vec::new();
            for f in taught { p.push(f.q[1].clone()); p.push(format!("= {}", f.a)); }
            p.push(":q".into());
            write(format!("{out}/day2p.txt"), &p);
            let mut d3: Vec<String> = taught.iter().map(|f| f.q[2].clone()).collect();
            d3.extend(untaught.iter().map(|f| f.q[0].clone()));
            d3.push(":q".into());
            write(format!("{out}/day3.txt"), &d3);
            println!("practice: {} second phrasings with correction; day 3: {} third phrasings + {} untaught", taught.len(), taught.len(), untaught.len());
        }
        "score3" => {
            let (d2, d3) = (replies(&arg(3)), replies(&arg(4)));
            let want: Vec<String> = taught.iter().map(|f| norm(&f.a)).collect();
            let (n, u) = (taught.len(), untaught.len());
            let without: Vec<String> = (0..n).map(|i| at_or_die(&d2, 3 * i + 2, "day-2 output")).collect();
            let sl = |v: &[String], s: usize, e: usize| v[s.min(v.len())..e.min(v.len())].to_vec();
            for (name, q2, un) in [("without practice", without, sl(&d2, 3 * n, 3 * n + u)),
                                   ("after practice  ", sl(&d3, 0, n), sl(&d3, n, n + u))] {
                if q2.len() < n { die("day-3 output is too short"); }
                let right = (0..n).filter(|&i| q2[i] == want[i]).count();
                let swaps = (0..n).filter(|&i| want.contains(&q2[i]) && q2[i] != want[i]).count();
                let fals = un.iter().filter(|x| want.contains(x)).count();
                println!("{name}: 3rd phrasings right {right}/{n}, swaps {swaps}/{n}, false recalls {fals}/{u}, intrusions {}", swaps + fals);
            }
        }
        "score" => {
            let (d1, d2) = (replies(&arg(3)), replies(&arg(4)));
            let want: Vec<String> = taught.iter().map(|f| norm(&f.a)).collect();
            let n = taught.len();
            let s1 = |i: usize| at_or_die(&d1, i, "day-1 output");
            let s2 = |i: usize| at_or_die(&d2, i, "day-2 output");
            let two = d1.len() >= 2 * n;   // new day-1 format: ask, then ask again after teaching
            let before = (0..n).filter(|&i| s1(if two { 2 } else { 1 } * i) == want[i]).count();
            println!("before teaching, asked as taught : {before}/{n} exact");
            if two {
                let same = (0..n).filter(|&i| s1(2 * i + 1) == want[i]).count();
                println!("same day, right after teaching   : {same}/{n} exact");
            }
            let t1 = (0..n).filter(|&i| s2(3 * i) == want[i]).count();
            let para = (0..n).map(|i| [1, 2].iter().filter(|&&k| s2(3 * i + k) == want[i]).count()).sum::<usize>();
            println!("after one sleep, asked as taught : {t1}/{n} exact");
            println!("after one sleep, other phrasings : {para}/{} exact", 2 * n);
            if d2.len() >= 3 * n + untaught.len() {
                let wrong = (0..untaught.len()).filter(|&j| want.contains(&d2[3 * n + j])).count();
                println!("untaught questions answered with a taught answer (false recall): {wrong}/{}", untaught.len());
                let swaps = (0..n).map(|i| [1, 2].iter().filter(|&&k| { let x = &d2[3 * i + k]; want.contains(x) && *x != want[i] }).count()).sum::<usize>();
                println!("other phrasings answered with the WRONG taught answer (swaps): {swaps}/{}", 2 * n);
                println!("design score = paraphrases right - wrong taught answers = {}", para as i64 - swaps as i64 - wrong as i64);
            }
            for (i, f) in taught.iter().enumerate() {
                for k in 0..3 {
                    let x = s2(3 * i + k);
                    println!("  {} q{k} {} {} -> {}", f.id, if x == want[i] { "OK " } else { "no " }, py_repr(&f.q[k]), py_repr(&x));
                }
            }
        }
        _ => die(&format!("unknown mode {mode}")),
    }
}

fn compare(root: &str, rest: &[String]) {
    if rest.is_empty() || rest.len() % 4 != 0 { die("compare needs groups of: <facts> <at> <arm-A day2> <arm-B day2>"); }
    let (mut net_a, mut net_b) = (Vec::new(), Vec::new());
    let mut tot = [[0usize; 3]; 2];   // per arm: right, intrusions, as-taught misses
    let (mut pos, mut neg) = ([Vec::new(), Vec::new()], [Vec::new(), Vec::new()]);
    for g in rest.chunks(4) {
        let at: usize = g[1].parse().unwrap_or_else(|_| die("bad <at>"));
        let all = facts(root, &g[0], at, 2 * N);
        let (taught, untaught) = all.split_at(N.min(all.len()));
        let arms = [arm(&g[2], taught, untaught.len()), arm(&g[3], taught, untaught.len())];
        println!("{} {}-{}:", g[0], at, at + all.len() - 1);
        for (k, (name, m)) in ["A", "B"].iter().zip(&arms).enumerate() {
            let intr = m.swaps + m.false_;
            let net = m.right as i64 - intr as i64;
            let share = m.right as f64 / (m.right + intr).max(1) as f64 * 100.0;
            println!("  arm {name}: as-taught {}/{}  right {}/{}  swaps {}  false {}  intrusions {}  net {:+}  share right {:.1}%  | a1 AUC right-vs-intrusion {:.3}  | next-word top-1 {}",
                m.q0, taught.len(), m.right, 2 * taught.len(), m.swaps, m.false_, intr, net, share,
                auc(&m.a1_pos, &m.a1_neg), m.top1.map_or("-".into(), |t| format!("{t:.2}")));
            tot[k][0] += m.right; tot[k][1] += intr; tot[k][2] += taught.len() - m.q0;
            pos[k].extend_from_slice(&m.a1_pos); neg[k].extend_from_slice(&m.a1_neg);
        }
        let d: i64 = arms[1].net.iter().map(|&x| x as i64).sum::<i64>() - arms[0].net.iter().map(|&x| x as i64).sum::<i64>();
        println!("  change in net (B - A): {d:+}");
        net_a.extend_from_slice(&arms[0].net);
        net_b.extend_from_slice(&arms[1].net);
    }
    let d: Vec<i64> = net_b.iter().zip(&net_a).map(|(b, a)| (*b - *a) as i64).collect();
    let share = |k: usize| tot[k][0] as f64 / (tot[k][0] + tot[k][1]).max(1) as f64 * 100.0;
    println!("\npooled over {} facts:", d.len());
    for (k, name) in ["A", "B"].iter().enumerate() {
        println!("  arm {name}: right {}  intrusions {}  net {:+}  share right {:.1}%  as-taught misses {}  a1 AUC {:.3}",
            tot[k][0], tot[k][1], tot[k][0] as i64 - tot[k][1] as i64, share(k), tot[k][2], auc(&pos[k], &neg[k]));
    }
    let sum: i64 = d.iter().sum();
    let mut rng = 1u64;
    let mut boot: Vec<i64> = (0..20_000).map(|_| (0..d.len()).map(|_| d[(splitmix(&mut rng) % d.len() as u64) as usize]).sum()).collect();
    boot.sort_unstable();
    println!("  change in net (B - A): {sum:+}   95% CI {:+} .. {:+}   resamples <= 0: {:.3}   facts better {}, worse {}, same {}",
        boot[500], boot[19_499], boot.iter().filter(|&&x| x <= 0).count() as f64 / boot.len() as f64,
        d.iter().filter(|&&x| x > 0).count(), d.iter().filter(|&&x| x < 0).count(), d.iter().filter(|&&x| x == 0).count());
    println!("  share right (B - A): {:+.1} points", share(1) - share(0));
}
