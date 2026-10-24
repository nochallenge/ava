// TESTING / DESIGN ONLY.
//
// COMPOSE — a fork of deep.rs for the v16 agenda. Training and evaluation are
// deep.rs's, byte for byte; the only addition is a MEASUREMENT block in the eval loop
// (probe bit 1). It changes nothing the model does, so the TSV line must reproduce
// deep.rs exactly or the run is contaminated.
//
//   cargo run --release --example compose -- 15000 0 0 0 5 2 5 0 128 1
//
// KAPPA. A copy readout can only ever help at a position where the gold next word is
// literally present in the prompt (last 4 content words) or is the last content word of
// the reply so far, AND the model is currently wrong there. That share of the exam is
// the ceiling on the entire copy route, in points. Pre-registered in v16:
//   kappa < 3.0%  -> abandon the copy readout
//   3.0 .. 5.0%   -> build copy-only at 15k
//   >= 5.0%       -> build it and take it to 30k
// Reported twice: raw, and with the mandatory filter (gold outside the 200 most frequent
// training tokens), because is_content admits `not`, `what`, `me`, `just`, and a match
// on those is coincidence, not copying.
//
// probe bit 2 — the CLASS POOLER (v16 "day 1, class"). sum the vote over word classes,
// take the best class, emit its best word. pre-registered on the 0-rung bucket against
// 22.45%: inside [21.9, 23.1] the class lever is closed for good; >= 24.0 is a real gain
// but a class-marginal smoother, not composition.
//
// probe bit 4 — the SELF-MODEL baseline (v16 §2). does she know when she is right?
// AUC, coverage and calibration for margin, rung count, and a cross-fitted counting
// table. no faculty is built here: this is the bar a neuron-based one must beat.
//
// DOMAINS (school). AVA_CORPUS=<file> replaces the corpus, AVA_TEST=<k> holds out
// exactly the first k pairs. A line whose first tab field starts with '@' names its
// domain (@rust, @ava_code, @ai, ...); any other line is "dialogue", and its first field
// stays the ignored previous turn, exactly as triples.tsv has always been read. @rust and
// @ava_code are read by code_tokenize. With a tag or a second domain present, every exam
// is also reported per domain (+ one `TSV domain` line each); otherwise nothing new prints.
//
// ---- deep.rs's original header follows.
//
// DEPTH. The one gap [[v10]] named that is not decoration.
//
//   cargo run --release --example deep -- <triples> <stage2 0|1>
//
// Everything Ava does is one stage: features -> units -> one weighted vote. There is
// no layer that reads the OUTPUT of another layer. Dendritic depth turned out to be
// redundant with the specificity term (v10), but that was depth INSIDE a unit. This
// is depth between stages, which she genuinely does not have.
//
//   STAGE 1  the current model. features -> a guess.
//   STAGE 2  the same machinery, plus stage 1's GUESS as an input feature, trained on
//            the true answer. it can only learn one thing: when stage 1 is wrong in a
//            systematic way, and what to say instead.
//
// Stage 1's guess enters as an IDENTITY key, because identity keys are free — the
// competition returns a fixed number of winners however many keys you supply (v9 §3e).
// A general feature would tax every position to pay for corrections at a few.
use design_tests::reply;
use design_tests::seq::{self, BOS, EOS};
use design_tests::v4::{self, Params};
use design_tests::v6::*;
use std::env;
use std::fs;
use std::time::Instant;
#[path = "common/meaning.rs"]
mod meaning;

// WORD MEANING (v28 idea 1, AVA_MEANING=<file from meaning_build>). recall matches a word not
// only by identity but by how much its meaning (a sparse code learned from reading) overlaps
// the engram's words: credit = overlap^AVA_MEANING_POW (1 = as read, 2 = only strong
// similarity counts). unset = v26, bit for bit.
fn meaning_codes() -> Option<&'static (meaning::Meaning, i32)> {
    static M: std::sync::OnceLock<Option<(meaning::Meaning, i32)>> = std::sync::OnceLock::new();
    M.get_or_init(|| {
        let path = env::var("AVA_MEANING").ok()?;
        let m = meaning::Meaning::load(std::path::Path::new(&path)).unwrap_or_else(|e| panic!("AVA_MEANING {path}: {e}"));
        let pow = env::var("AVA_MEANING_POW").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        Some((m, pow))
    }).as_ref()
}
// WORD FORMS (v28 idea 2, AVA_LEMMA=<unimorph tsv>). two forms of one word ("designed",
// "design", "designs") are the same lexeme, so they match fully. the table is the one her
// morphology was learned from (she recalls 99.9% of its trained forms). unset = v26.
fn lemmas() -> Option<&'static std::collections::HashMap<String, Vec<u32>>> {
    static L: std::sync::OnceLock<Option<std::collections::HashMap<String, Vec<u32>>>> = std::sync::OnceLock::new();
    L.get_or_init(|| {
        let path = env::var("AVA_LEMMA").ok()?;
        let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("AVA_LEMMA {path}: {e}"));
        let (mut id, mut of): (std::collections::HashMap<String, u32>, std::collections::HashMap<String, Vec<u32>>) = Default::default();
        for l in text.lines() {
            let mut p = l.split('\t');
            let (Some(lemma), Some(form)) = (p.next(), p.next()) else { continue };
            if lemma.contains(' ') || form.contains(' ') { continue; }
            let n = id.len() as u32;
            let i = *id.entry(lemma.to_lowercase()).or_insert(n);
            for w in [lemma.to_lowercase(), form.to_lowercase()] {
                let v = of.entry(w).or_default();
                if !v.contains(&i) { v.push(i); }
            }
        }
        Some(of)
    }).as_ref()
}
// QUESTION KIND -> ANSWER KIND (v28 idea 3, AVA_QKIND=<school dialogue.tsv>). from what she
// heard (the 15,000 training exchanges after the 1,200 exam ones) she learns, per question
// word, how often a reply holds a number: P(num | w), shrunk toward the base rate P0 with 10
// pseudo-exchanges. a question's expectation combines its words naive-Bayes style (log-odds
// add). an engram whose answer kind the question makes LESS likely than usual is dimmed by
// P(kind | q) / P(kind); it is never boosted. unset = v26.
fn is_number(w: &str) -> bool {
    w.bytes().all(|c| c.is_ascii_digit()) && !w.is_empty() || matches!(w, "zero" | "one" | "two" | "three" | "four" | "five" | "six" | "seven" | "eight"
        | "nine" | "ten" | "eleven" | "twelve" | "thirteen" | "fourteen" | "fifteen" | "sixteen" | "seventeen" | "eighteen" | "nineteen"
        | "twenty" | "thirty" | "forty" | "fifty" | "sixty" | "seventy" | "eighty" | "ninety" | "hundred" | "thousand" | "million" | "billion")
}
struct QKind { n: std::collections::HashMap<String, (f32, f32)>, p0: f32 }
fn qkind() -> Option<&'static QKind> {
    static Q: std::sync::OnceLock<Option<QKind>> = std::sync::OnceLock::new();
    Q.get_or_init(|| {
        let path = env::var("AVA_QKIND").ok()?;
        let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("AVA_QKIND {path}: {e}"));
        let mut n: std::collections::HashMap<String, (f32, f32)> = Default::default();
        let (mut num, mut all) = (0f32, 0f32);
        for l in text.lines().skip(1200).take(15000) {
            let p: Vec<&str> = l.split('\t').collect();
            if p.len() < 3 { continue; }
            let k = seq::tokenize(p[2]).iter().any(|w| is_number(w)) as u8 as f32;
            num += k; all += 1.0;
            let mut q = seq::tokenize(p[1]);
            q.sort_unstable(); q.dedup();
            for w in q { let e = n.entry(w).or_default(); e.0 += k; e.1 += 1.0; }
        }
        Some(QKind { n, p0: (num + 1.0) / (all + 2.0) })
    }).as_ref()
}
/// P(the answer holds a number | question words), also counting her stored lessons
fn p_number(q: &[&str], lessons: &[(Vec<String>, Vec<String>)]) -> Option<f32> {
    let qk = qkind()?;
    let logit = |p: f32| (p / (1.0 - p)).ln();
    let mut z = logit(qk.p0);
    for w in q {
        let (mut k, mut c) = qk.n.get(*w).copied().unwrap_or((0.0, 0.0));
        for (lq, lr) in lessons {
            if lq.iter().any(|x| x == w) { c += 1.0; if lr.iter().any(|x| is_number(x)) { k += 1.0; } }
        }
        let p = (k + 10.0 * qk.p0) / (c + 10.0);
        z += logit(p) - logit(qk.p0);
    }
    Some(1.0 / (1.0 + (-z).exp()))
}
fn same_lexeme(a: &str, b: &str) -> bool {
    let Some(of) = lemmas() else { return false };
    match (of.get(a), of.get(b)) { (Some(x), Some(y)) => x.iter().any(|i| y.contains(i)), _ => false }
}

const CORPUS: &str = "../data/triples.tsv";
const MIN_FREE_GB: f64 = 3.0;
const ORDER: usize = 3;
/// A FIXED held-out set, taken from the FRONT of the corpus and never trained on.
///
/// It used to be the last 2% of whatever was loaded, which silently made every
/// corpus size a different exam. Measured: 60k exchanges scored 23.32% against 30k's
/// 36.47%, and doubling the neuron pool moved it by 0.09 — because the drop was never
/// about the model, it was two different test sets. triples.tsv is ordered by film, so
/// the tail of a 60k window is a different genre from the tail of a 30k one.
///
/// Comparisons across training size are meaningless unless the exam is held constant.
const TEST: usize = 1200;
const K_ID: usize = 6;

/// CODE TOKENIZER, for the @rust and @ava_code domains. seq::tokenize lowercases and
/// splits `foo_bar` into `foo _ bar` and `"a b"` into `" a b "`, which turns code into
/// noise. Here: case kept; identifiers (and numbers) are [A-Za-z0-9_]+; a string literal
/// "..." (escapes included) is ONE token, unterminated ones run to the end of the line;
/// a char literal ('a', '\n', '\'', '\u{..}') is one token and so is a lifetime or label
/// ('a, 'static) that has no closing quote; the operators below are single tokens
/// (longest first); every other non-space char stands alone.
fn code_tokenize(s: &str) -> Vec<String> {
    const OPS3: [&str; 3] = ["<<=", ">>=", "..="];
    const OPS2: [&str; 20] = ["::", "->", "=>", "==", "!=", "<=", ">=", "&&", "||",
        "+=", "-=", "*=", "/=", "%=", "^=", "&=", "|=", "<<", ">>", ".."];
    let c: Vec<char> = s.chars().collect();
    let ident = |ch: char| ch.is_ascii_alphanumeric() || ch == '_';
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < c.len() {
        let ch = c[i];
        if ch.is_whitespace() { i += 1; continue; }
        let end = if ident(ch) {
            (i..c.len()).find(|&j| !ident(c[j])).unwrap_or(c.len())
        } else if ch == '"' {
            let mut j = i + 1;
            while j < c.len() && c[j] != '"' { j += if c[j] == '\\' { 2 } else { 1 }; }
            (j + 1).min(c.len())
        } else if ch == '\'' {
            let close = if c.get(i + 1) == Some(&'\\') {
                (i + 3..c.len().min(i + 12)).find(|&j| c[j] == '\'')
            } else if c.get(i + 2) == Some(&'\'') { Some(i + 2) } else { None };
            match close {
                Some(j) => j + 1,
                None if c.get(i + 1).map_or(false, |&x| x.is_ascii_alphabetic() || x == '_') =>
                    (i + 1..c.len()).find(|&j| !ident(c[j])).unwrap_or(c.len()),
                None => i + 1,
            }
        } else {
            let at = |k: usize| -> String { c[i..(i + k).min(c.len())].iter().collect() };
            if OPS3.contains(&at(3).as_str()) { i + 3 }
            else if OPS2.contains(&at(2).as_str()) { i + 2 }
            else { i + 1 }
        };
        out.push(c[i..end].iter().collect());
        i = end;
    }
    out
}

#[cfg(test)]
mod code_tok {
    use super::*;
    fn t(s: &str) -> String { code_tokenize(s).join(" ") }
    #[test]
    fn rust_tokens() {
        assert_eq!(t("let x_1: Vec<u32> = a::b(&mut y);"), "let x_1 : Vec < u32 > = a :: b ( & mut y ) ;");
        assert_eq!(t(r#"println!("a \"b\" c", x)"#), r#"println ! ( "a \"b\" c" , x )"#);
        assert_eq!(t("fn f<'a>(s: &'a str) -> &'static str"), "fn f < 'a > ( s : & 'a str ) -> & 'static str");
        assert_eq!(t(r"let c = 'a'; let d = '\n'; let e = '\''; let u = '\u{1F600}';"),
            r"let c = 'a' ; let d = '\n' ; let e = '\'' ; let u = '\u{1F600}' ;");
        assert_eq!(t("a <<= 1; b >>= 2; 0..=9; x..y; p => q; i += 1 && j || k != l"),
            "a <<= 1 ; b >>= 2 ; 0 ..= 9 ; x .. y ; p => q ; i += 1 && j || k != l");
        assert_eq!(t("'outer: loop { break 'outer; }"), "'outer : loop { break 'outer ; }");
        assert_eq!(t("\"open"), "\"open");
        assert_eq!(t("Vec<Vec<u8>>"), "Vec < Vec < u8 >>");
    }
    #[test]
    fn talk_switch() {
        assert_eq!(talk_tokenize("Hello there ?"), seq::tokenize("Hello there ?"));
        assert_eq!(talk_tokenize("fn main() {}"), code_tokenize("fn main() {}"));
    }
}

/// talk mode: a line that looks like code is read the way code was learned.
fn talk_tokenize(s: &str) -> Vec<String> {
    if s.chars().any(|ch| "{}();:=<>&_".contains(ch)) { code_tokenize(s) } else { seq::tokenize(s) }
}

// SYNAPSES (v26). learn_bulk lays down Hebbian links among the units of every code she
// writes: 190M of them at 15k, most of her RAM. reading them (pruning + pattern completion,
// two pre-registered tries) added nothing: they link neurons of the SAME write, which her
// readout already counts. so they are OFF by default (same answers, 3.16 -> 0.89 GB);
// AVA_SYNAPSES=1 lays them down again.
fn synapses_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env::var("AVA_SYNAPSES").map_or(false, |v| v == "1"))
}

// DELAYED SYNAPSES (v26, on by default since it passed at 15k and 30k; AVA_DELAY=0 = off):
// links ACROSS time instead. the neurons of the moment
// before (pre) link to the word that arrives after the next one (post), the way activity
// persists for a moment and spike-timing plasticity bridges the gap. her readout sees only
// the present moment; this one remembers the moment before, skipping a word, which carries
// something new exactly when the present moment has never been seen.
fn delay_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env::var("AVA_DELAY").map_or(true, |v| v != "0"))
}

// ERROR-DRIVEN LINKS (v27, AVA_LINKS=1). every engram keeps one link per cue word, 0..1: one
// per word of its question (kQ) and one per answer word its question lacks (kA, the whole
// episode, v25's AVA_CUE=1). all start at 1. when she is shown "= answer", every OTHER engram
// the question pulled up loses η·a from exactly the links that pulled it (Rescorla-Wagner with
// target 0: reconsolidation by prediction error, retrieval weakening its competitors). a faded
// link stops counting on both sides, it is never held against the engram, so a taught question
// still reaches its own engram at a = 1. she learns at each "=" and in sleep (passes after the
// cortex replay). with every link at 1 this is AVA_CUE=1 exactly. η and the number of passes
// were fixed before any run (v27 pre-registration), not tuned.
type Links = Vec<(Vec<f32>, Vec<f32>)>;
const LINK_ETA: f32 = 0.5;
const LINK_PASSES: usize = 2;
fn links_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| env::var("AVA_LINKS").map_or(false, |v| v == "1"))
}
// AVA_LINK_ETA exists only for the implementation check (η = 0 must equal AVA_CUE=1)
fn link_eta() -> f32 {
    env::var("AVA_LINK_ETA").ok().and_then(|v| v.parse().ok()).unwrap_or(LINK_ETA)
}
fn word_set(ws: &[String]) -> Vec<&str> {
    let mut v: Vec<&str> = ws.iter().map(|w| w.as_str()).collect();
    v.sort_unstable();
    v.dedup();
    v
}
/// an engram's cue words: its question's, and its answer's that the question lacks
fn episode_sets<'a>(q: &'a [String], r: &'a [String]) -> (Vec<&'a str>, Vec<&'a str>) {
    let es = word_set(q);
    let mut ea: Vec<&str> = r.iter().map(|w| w.as_str()).filter(|w| es.binary_search(w).is_err()).collect();
    ea.sort_unstable();
    ea.dedup();
    (es, ea)
}
/// links aligned with the engrams: a lesson without links (new, or from an older brain) gets
/// all 1s, which is the whole-episode recall unchanged
fn fit_links(lessons: &Lessons, links: &mut Links) {
    links.truncate(lessons.len());
    for (i, (q, r)) in lessons.iter().enumerate() {
        let (es, ea) = episode_sets(q, r);
        let ok = links.get(i).map_or(false, |(kq, ka)| kq.len() == es.len() && ka.len() == ea.len());
        if !ok {
            let fresh = (vec![1.0; es.len()], vec![1.0; ea.len()]);
            if i < links.len() { links[i] = fresh; } else { links.push(fresh); }
        }
    }
}
/// one engram's activation by the asked word set qs (sorted): matched links count for it,
/// asked words it lacks count against it, a faded link counts on neither side
fn link_act(qs: &[&str], es: &[&str], ea: &[&str], kq: &[f32], ka: &[f32], spec: &dyn Fn(&str) -> f32) -> f32 {
    let (mut shared, mut all) = (0f32, 0f32);
    for w in qs {
        let x = spec(w);
        if let Ok(j) = es.binary_search(w) { let y = x * kq[j]; shared += y; all += y; }
        else if let Ok(j) = ea.binary_search(w) { let y = x * ka[j]; shared += y; all += y; }
        else { all += x; }
    }
    for (j, w) in es.iter().enumerate() {
        if qs.binary_search(w).is_err() { all += spec(w) * kq[j]; }
    }
    if all > 0.0 { shared / all } else { 0.0 }
}
/// one Rescorla-Wagner trial: she has just been shown that `ans` answers `q`. the engrams with
/// another answer that the question pulled up lose what pulled them; when the lesson is not
/// new (a replay), the engram holding the answer regains what the question used (no new links).
fn link_trial(lessons: &mut Lessons, links: &mut Links, q: &[String], ans: &[String], allow_new: bool,
              eta: f32, spec: &dyn Fn(&str) -> f32) {
    fit_links(lessons, links);
    let qs = word_set(q);
    let (mut target, mut exact) = (false, false);
    {
        let sets: Vec<(Vec<&str>, Vec<&str>)> = lessons.iter().map(|(eq, er)| episode_sets(eq, er)).collect();
        // what she expected before learning
        let acts: Vec<f32> = sets.iter().zip(links.iter())
            .map(|((es, ea), (kq, ka))| link_act(&qs, es, ea, kq, ka, spec)).collect();
        for (i, ((es, ea), (kq, ka))) in sets.iter().zip(links.iter_mut()).enumerate() {
            if lessons[i].1.as_slice() == ans {
                target = true;
                if *es == qs { exact = true; }
                continue;
            }
            let a = acts[i];
            if a <= 0.0 { continue; }
            for w in &qs {
                if let Ok(j) = es.binary_search(w) { kq[j] = (kq[j] - eta * a).max(0.0); }
                if let Ok(j) = ea.binary_search(w) { ka[j] = (ka[j] - eta * a).max(0.0); }
            }
        }
        if target && !allow_new {
            for (i, ((es, ea), (kq, ka))) in sets.iter().zip(links.iter_mut()).enumerate() {
                if lessons[i].1.as_slice() != ans { continue; }
                let gain = eta * (1.0 - acts[i]);
                for w in &qs {
                    if let Ok(j) = es.binary_search(w) { kq[j] = (kq[j] + gain).min(1.0); }
                    else if let Ok(j) = ea.binary_search(w) { ka[j] = (ka[j] + gain).min(1.0); }
                }
            }
        }
    }
    if !target || (allow_new && !exact) {
        let (es, ea) = episode_sets(q, ans);
        links.push((vec![1.0; es.len()], vec![1.0; ea.len()]));
        lessons.push((q.to_vec(), ans.to_vec()));
    }
}
/// a reading, never spoken: how much of an engram's question still links to it, rarity-weighted
fn link_kept(lessons: &Lessons, links: &Links, i: usize, spec: &dyn Fn(&str) -> f32) -> f32 {
    let (Some((q, _)), Some((kq, _))) = (lessons.get(i), links.get(i)) else { return 1.0 };
    let es = word_set(q);
    if kq.len() != es.len() { return 1.0; }
    let (mut k, mut t) = (0f32, 0f32);
    for (j, w) in es.iter().enumerate() { let x = spec(w); k += x * kq[j]; t += x; }
    if t > 0.0 { k / t } else { 1.0 }
}
/// how specific a word is to her: ln(n / (1 + times it was a reply word)), rare words weigh more
fn spec_of(s1: &Stage, w: &str) -> f32 {
    let n = s1.ro.n_learned.max(1) as f32;
    let c = s1.cont.as_ref().and_then(|ct| s1.ro.op_id.get(w).and_then(|o| ct.r0.get(o))).copied().unwrap_or(0);
    (n / (1.0 + c as f32)).ln().max(0.0)
}

fn mem_gb() -> (f64, f64) {
    use std::process::Command;
    let pid = std::process::id();
    let script = format!(
        "$p=(Get-Process -Id {pid}).WorkingSet64; \
         $f=(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory*1KB; \
         Write-Output \"$p $f\""
    );
    let out = Command::new("powershell").args(["-NoProfile", "-Command", &script])
        .output().ok().and_then(|o| String::from_utf8(o.stdout).ok());
    let Some(t) = out else { return (0.0, f64::INFINITY) };
    let mut it = t.split_whitespace();
    (it.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0) / 1e9,
     it.next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(f64::INFINITY) / 1e9)
}

/// Area under the ROC curve by ranks (Mann-Whitney), ties given their average rank.
/// 0.5 = the signal knows nothing about whether she is right; 1.0 = it knows perfectly.
/// the `k` largest entries of a dense vector (index, value), largest first.
fn topk_idx(p: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut best: Vec<(usize, f32)> = Vec::with_capacity(k + 1);
    for (i, &x) in p.iter().enumerate() {
        if x > 0.0 && (best.len() < k || x > best[best.len() - 1].1) {
            let at = best.iter().position(|b| x > b.1).unwrap_or(best.len());
            best.insert(at, (i, x));
            best.truncate(k);
        }
    }
    best
}

fn auc(xs: &[(f64, bool)]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut i, mut sum, mut pos) = (0usize, 0f64, 0usize);
    while i < v.len() {
        let mut j = i;
        while j < v.len() && v[j].0 == v[i].0 { j += 1; }
        let avg = (i + j + 1) as f64 / 2.0;
        for k in i..j { if v[k].1 { sum += avg; pos += 1; } }
        i = j;
    }
    let neg = v.len() - pos;
    if pos == 0 || neg == 0 { return 0.5; }
    (sum - (pos * (pos + 1)) as f64 / 2.0) / (pos as f64 * neg as f64)
}

/// 95% interval for the AUC, resampling whole EXCHANGES. the ~12k positions come from
/// 1,200 exchanges and share a prompt within one, so the effective n is ~1,200, not 12k.
/// fixed-seed xorshift, so the interval is the same every run.
fn boot(xs: &[(usize, f64, bool)], n_ex: usize) -> (f64, f64) {
    let mut by: Vec<Vec<(f64, bool)>> = vec![Vec::new(); n_ex];
    for &(e, s, y) in xs { by[e].push((s, y)); }
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut aucs: Vec<f64> = (0..200).map(|_| {
        let mut v = Vec::with_capacity(xs.len());
        for _ in 0..n_ex {
            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
            v.extend_from_slice(&by[(seed % n_ex as u64) as usize]);
        }
        auc(&v)
    }).collect();
    aucs.sort_by(|a, b| a.total_cmp(b));
    (aucs[5], aucs[194])
}

/// Accuracy on the most-confident `frac` of positions. stable sort, so ties keep exam order.
fn cover(xs: &[(f64, bool)], frac: f64) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| b.0.total_cmp(&a.0));
    let k = ((v.len() as f64 * frac) as usize).max(1);
    v[..k].iter().filter(|x| x.1).count() as f64 / k as f64 * 100.0
}

/// Expected calibration error in points: in 10 equal-width bins of predicted p, how far
/// the actual hit rate sits from the predicted one, weighted by bin size.
fn ece(xs: &[(f64, bool)]) -> f64 {
    let mut b = [[0f64; 3]; 10];
    for &(p, y) in xs {
        let i = ((p * 10.0) as usize).min(9);
        b[i][0] += 1.0; b[i][1] += p; if y { b[i][2] += 1.0; }
    }
    let n = xs.len() as f64;
    b.iter().filter(|c| c[0] > 0.0).map(|c| c[0] / n * (c[2] / c[0] - c[1] / c[0]).abs()).sum::<f64>() * 100.0
}

/// Interpolated Kneser-Ney (Chen & Goodman 1998), the standard count-based language
/// model. Here as a YARDSTICK: it says how much of the exam any count-based predictor
/// can get, so ava's 22% on unaddressed positions can be read as "weak" or "near the floor".
/// The highest order uses raw counts; every lower order uses CONTINUATION counts (how
/// many distinct words precede an n-gram), which is what stops a backed-off guess from
/// being "the most common word".
struct Kn {
    ids: std::collections::HashMap<String, u32>,
    words: Vec<String>,
    n: usize,
    raw: Vec<std::collections::HashMap<Vec<u32>, KnCell>>,
    cont: Vec<std::collections::HashMap<Vec<u32>, KnCell>>,
    d_raw: Vec<f32>,
    d_cont: Vec<f32>,
}
struct KnCell { tot: f32, n1: f32, next: Vec<(u32, u32)> }
const KN_BOS: u32 = 0;
const KN_SEP: u32 = 1;
const KN_UNK: u32 = 2;   // ids below 3 are never predicted

impl Kn {
    fn build(pairs: &[(Vec<String>, Vec<String>)], concat: bool, n: usize) -> Kn {
        use std::collections::HashMap;
        let mut ids: HashMap<String, u32> = HashMap::new();
        let mut words: Vec<String> = Vec::new();
        for w in ["<bos>", "<sep>", "<unk>"] { ids.insert(w.into(), words.len() as u32); words.push(w.into()); }
        let mut acc: Vec<HashMap<Vec<u32>, HashMap<u32, u32>>> = (0..n).map(|_| HashMap::new()).collect();
        for (q, r) in pairs {
            let mut s = vec![KN_BOS; n - 1];
            let mut id = |w: &str| -> u32 {
                if let Some(&i) = ids.get(w) { return i; }
                let i = words.len() as u32; ids.insert(w.to_string(), i); words.push(w.to_string()); i
            };
            if concat { for w in q { s.push(id(w)); } s.push(KN_SEP); }
            for w in r { s.push(id(w)); }
            s.push(id(EOS));
            for i in (n - 1)..s.len() {
                for k in 0..n {
                    *acc[k].entry(s[i - k..i].to_vec()).or_default().entry(s[i]).or_default() += 1;
                }
            }
        }
        // continuation counts for every order below the top: N1+(. h w)
        let mut cacc: Vec<HashMap<Vec<u32>, HashMap<u32, u32>>> = (0..n).map(|_| HashMap::new()).collect();
        for k in 0..n - 1 {
            for (h, nx) in &acc[k + 1] {
                let short = h[1..].to_vec();
                let e = cacc[k].entry(short).or_default();
                for &w in nx.keys() { *e.entry(w).or_default() += 1; }
            }
        }
        let fin = |m: Vec<HashMap<Vec<u32>, HashMap<u32, u32>>>| -> (Vec<HashMap<Vec<u32>, KnCell>>, Vec<f32>) {
            let mut ds = Vec::new();
            let t = m.into_iter().map(|tab| {
                let (mut c1, mut c2) = (0f32, 0f32);
                let out: HashMap<Vec<u32>, KnCell> = tab.into_iter().map(|(h, nx)| {
                    let mut next: Vec<(u32, u32)> = nx.into_iter().collect();
                    next.sort_unstable();
                    for &(_, c) in &next { if c == 1 { c1 += 1.0 } else if c == 2 { c2 += 1.0 } }
                    let tot = next.iter().map(|x| x.1 as f32).sum();
                    (h, KnCell { tot, n1: next.len() as f32, next })
                }).collect();
                ds.push(if c1 + c2 > 0.0 { (c1 / (c1 + 2.0 * c2)).clamp(0.1, 0.95) } else { 0.75 });
                out
            }).collect();
            (t, ds)
        };
        let (raw, d_raw) = fin(acc);
        let (cont, d_cont) = fin(cacc);
        Kn { ids, words, n, raw, cont, d_raw, d_cont }
    }
    fn id(&self, w: &str) -> u32 { self.ids.get(w).copied().unwrap_or(KN_UNK) }
    /// the n-1 tokens before reply position `pos`, padded with <bos>.
    fn ctx(&self, q: &[String], concat: bool, rep: &[String], pos: usize) -> Vec<u32> {
        let mut s = vec![KN_BOS; self.n - 1];
        if concat { s.extend(q.iter().map(|w| self.id(w))); s.push(KN_SEP); }
        s.extend(rep[..pos].iter().map(|w| self.id(w)));
        s[s.len() - (self.n - 1)..].to_vec()
    }
    /// top-k word ids under P_KN(w | last order-1 tokens of ctx).
    fn top(&self, ctx: &[u32], order: usize, k: usize, p: &mut Vec<f32>) -> Vec<u32> {
        let v = self.words.len();
        p.clear();
        p.resize(v, 1.0 / v as f32);
        for len in 0..order {
            let (tab, d) = if len == order - 1 { (&self.raw[len], self.d_raw[len]) }
                           else { (&self.cont[len], self.d_cont[len]) };
            let Some(c) = tab.get(&ctx[ctx.len() - len..]) else { continue };
            let g = d * c.n1 / c.tot;
            for x in p.iter_mut() { *x *= g; }
            for &(w, n) in &c.next { p[w as usize] += (n as f32 - d).max(0.0) / c.tot; }
        }
        let mut best: Vec<(f32, u32)> = Vec::with_capacity(k + 1);
        for (w, &s) in p.iter().enumerate().skip(3) {
            if best.len() < k || s > best[best.len() - 1].0 {
                let at = best.iter().position(|b| s > b.0).unwrap_or(best.len());
                best.insert(at, (s, w as u32));
                best.truncate(k);
            }
        }
        best.into_iter().map(|(_, w)| w).collect()
    }
}

type Nbrs = std::collections::HashMap<String, Vec<String>>;
struct Stage { fac: Factored, ro: OpReadout, nbrs: Option<Nbrs>, mode: u8, rungs: u32, per: usize,
              cls: Option<std::collections::HashMap<String, u16>>, bits: u32, cont: Option<Cont>,
              rod: OpReadout, prev: Option<Vec<NeuronId>> }

/// NOVELTY-GATED COUNTS (probe bit 16) — kneser-ney's continuation counts, grown by a
/// neuron-local rule instead of a corpus pass. When a word is written in context `x y`,
/// the `B1:y` units count it ONLY IF the `B2:x y` units had never voted for it before:
/// "y has been followed by w in one more DIFFERENT context". The same gate one level down
/// feeds a unigram that counts in how many different contexts a word has appeared, not how
/// often. `r0` is the plain frequency, kept so interpolation and continuation can be
/// measured apart.
#[derive(Default)]
struct Cont {
    c1: std::collections::HashMap<NeuronId, std::collections::HashMap<u32, u32>>,
    c0: std::collections::HashMap<u32, u32>,
    r0: std::collections::HashMap<u32, u32>,
    /// v22: a 3-word level that lives OUTSIDE the code. `b3` = raw votes of `B3:x y z`
    /// units; `c2` = `B2:y z` counted only when `B3` was surprised. kept out of the code
    /// because every general feature in the code dilutes her main vote (v8).
    b3: std::collections::HashMap<NeuronId, std::collections::HashMap<u32, u32>>,
    c2: std::collections::HashMap<NeuronId, std::collections::HashMap<u32, u32>>,
}

/// FAMILIARITY (v24, probe 64): the perirhinal "have I heard this before?" signal, which in
/// the brain fires before (and apart from) recalling anything. one unit per ordered word pair
/// of every prompt she has heard, sentence edges included (`<s> w0` ... `wn </s>`); a unit's
/// count is how often she heard that pair. ordered, because "you are who" is strange and
/// unordered pairs can't see it (measured: unordered AUC 0.50 on reversed prompts, ordered 0.94).
///   F(q) = mean ln(1 + count) over q's pairs
///   φ(q) = where F(q) falls among her own heard prompts, each judged without itself
/// no threshold anywhere: φ is a continuous reading, never a rule on what she says.
#[derive(Default)]
struct Familiar { bi: std::collections::HashMap<u64, u32>, loo: Vec<f32> }
impl Familiar {
    fn keys(q: &[String]) -> Vec<u64> {
        let w = |i: usize| -> &str { if i == 0 { "<s>" } else if i > q.len() { "</s>" } else { q[i - 1].as_str() } };
        (0..=q.len()).map(|i| {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for b in w(i).bytes().chain([0x1f]).chain(w(i + 1).bytes()) { h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3); }
            h
        }).collect()
    }
    fn hear(&mut self, q: &[String]) { for k in Self::keys(q) { *self.bi.entry(k).or_default() += 1; } }
    /// `own` = leave this prompt's own counts out (it was heard once, as itself).
    fn f(&self, q: &[String], own: bool) -> f32 {
        let ks = Self::keys(q);
        let s: f32 = ks.iter().map(|k| {
            let mine = if own { ks.iter().filter(|x| *x == k).count() as u32 } else { 0 };
            (1.0 + self.bi.get(k).copied().unwrap_or(0).saturating_sub(mine) as f32).ln()
        }).sum();
        s / ks.len().max(1) as f32
    }
    fn settle(&mut self, heard: &[(Vec<String>, Vec<String>)]) {
        self.loo = heard.iter().map(|(q, _)| self.f(q, true)).collect();
        self.loo.sort_by(|a, b| a.total_cmp(b));
    }
    fn phi(&self, q: &[String]) -> f64 {
        let x = self.f(q, false);
        let (lt, le) = (self.loo.partition_point(|&v| v < x), self.loo.partition_point(|&v| v <= x));
        (lt + le) as f64 / 2.0 / self.loo.len().max(1) as f64
    }
}

/// PERSISTENCE sidecar (v24). store.rs keeps the synapses, the addressing and the readout;
/// this keeps what compose's brain grew since: her fallback's count tables (`Cont`) and
/// her familiarity units. plain little-endian u64s; brain files stay local, never published.
type Nested = std::collections::HashMap<NeuronId, std::collections::HashMap<u32, u32>>;
fn put(w: &mut impl std::io::Write, x: u64) -> std::io::Result<()> { w.write_all(&x.to_le_bytes()) }
fn get(r: &mut impl std::io::Read) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn put_nested(w: &mut impl std::io::Write, t: &Nested) -> std::io::Result<()> {
    put(w, t.len() as u64)?;
    for (&u, m) in t {
        put(w, u as u64)?;
        put(w, m.len() as u64)?;
        for (&o, &c) in m { put(w, o as u64)?; put(w, c as u64)?; }
    }
    Ok(())
}
fn get_nested(r: &mut impl std::io::Read) -> std::io::Result<Nested> {
    let n = get(r)? as usize;
    let mut t = Nested::with_capacity(n);
    for _ in 0..n {
        let u = get(r)? as NeuronId;
        let k = get(r)? as usize;
        let mut m = std::collections::HashMap::with_capacity(k);
        for _ in 0..k { let o = get(r)? as u32; m.insert(o, get(r)? as u32); }
        t.insert(u, m);
    }
    Ok(t)
}
fn put_flat(w: &mut impl std::io::Write, t: &std::collections::HashMap<u32, u32>) -> std::io::Result<()> {
    put(w, t.len() as u64)?;
    for (&o, &c) in t { put(w, o as u64)?; put(w, c as u64)?; }
    Ok(())
}
fn get_flat(r: &mut impl std::io::Read) -> std::io::Result<std::collections::HashMap<u32, u32>> {
    let n = get(r)? as usize;
    let mut t = std::collections::HashMap::with_capacity(n);
    for _ in 0..n { let o = get(r)? as u32; t.insert(o, get(r)? as u32); }
    Ok(t)
}
fn put_str(w: &mut impl std::io::Write, s: &str) -> std::io::Result<()> {
    put(w, s.len() as u64)?;
    w.write_all(s.as_bytes())
}
fn get_str(r: &mut impl std::io::Read) -> std::io::Result<String> {
    let mut b = vec![0u8; get(r)? as usize];
    r.read_exact(&mut b)?;
    String::from_utf8(b).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}
type Lessons = Vec<(Vec<String>, Vec<String>)>;
fn save_more(path: &str, cont: &Option<Cont>, fam: &Option<Familiar>, lessons: &Lessons, rod: &OpReadout,
             links: &Links) -> std::io::Result<()> {
    use std::io::Write;
    let mut w = std::io::BufWriter::with_capacity(1 << 22, fs::File::create(path)?);
    w.write_all(b"AVAMORE4")?;
    put(&mut w, cont.is_some() as u64)?;
    if let Some(c) = cont {
        put_nested(&mut w, &c.c1)?; put_flat(&mut w, &c.c0)?; put_flat(&mut w, &c.r0)?;
        put_nested(&mut w, &c.b3)?; put_nested(&mut w, &c.c2)?;
    }
    put(&mut w, fam.is_some() as u64)?;
    if let Some(f) = fam {
        put(&mut w, f.bi.len() as u64)?;
        for (&k, &c) in &f.bi { put(&mut w, k)?; put(&mut w, c as u64)?; }
        put(&mut w, f.loo.len() as u64)?;
        for &x in &f.loo { put(&mut w, x.to_bits() as u64)?; }
    }
    put(&mut w, lessons.len() as u64)?;
    for (q, r) in lessons { put_str(&mut w, &q.join("\u{1f}"))?; put_str(&mut w, &r.join("\u{1f}"))?; }
    // AVAMORE3 (v26): the delayed synapses, a readout of their own
    put(&mut w, rod.n_learned as u64)?;
    put(&mut w, rod.ops.len() as u64)?;
    for o in &rod.ops { put_str(&mut w, o)?; }
    put(&mut w, rod.votes.len() as u64)?;
    for (&u, v) in &rod.votes {
        put(&mut w, u as u64)?;
        put(&mut w, *rod.total.get(&u).unwrap_or(&0) as u64)?;
        put(&mut w, v.len() as u64)?;
        for (&o, &c) in v { put(&mut w, o as u64)?; put(&mut w, c as u32 as u64)?; }
    }
    // AVAMORE4 (v27): each engram's cue links
    put(&mut w, links.len() as u64)?;
    for (kq, ka) in links {
        put(&mut w, kq.len() as u64)?;
        for &k in kq { put(&mut w, k.to_bits() as u64)?; }
        put(&mut w, ka.len() as u64)?;
        for &k in ka { put(&mut w, k.to_bits() as u64)?; }
    }
    w.flush()
}
fn load_more(path: &str, cont: &mut Option<Cont>, fam: &mut Option<Familiar>, lessons: &mut Lessons, rod: &mut OpReadout,
             links: &mut Links) -> std::io::Result<()> {
    use std::io::Read;
    let mut r = std::io::BufReader::with_capacity(1 << 22, fs::File::open(path)?);
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    let v4 = &magic == b"AVAMORE4";
    let v3 = v4 || &magic == b"AVAMORE3";
    let v2 = v3 || &magic == b"AVAMORE2";
    if !v2 && &magic != b"AVAMORE1" { return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not an ava sidecar")); }
    // a table she never grew stays empty: the probe bits of this run decide what is read
    if get(&mut r)? == 1 {
        let c = Cont { c1: get_nested(&mut r)?, c0: get_flat(&mut r)?, r0: get_flat(&mut r)?,
                       b3: get_nested(&mut r)?, c2: get_nested(&mut r)? };
        if cont.is_some() { *cont = Some(c); }
    }
    if get(&mut r)? == 1 {
        let n = get(&mut r)? as usize;
        let mut bi = std::collections::HashMap::with_capacity(n);
        for _ in 0..n { let k = get(&mut r)?; bi.insert(k, get(&mut r)? as u32); }
        let m = get(&mut r)? as usize;
        let mut loo = Vec::with_capacity(m);
        for _ in 0..m { loo.push(f32::from_bits(get(&mut r)? as u32)); }
        if fam.is_some() { *fam = Some(Familiar { bi, loo }); }
    }
    if v2 {
        let n = get(&mut r)? as usize;
        for _ in 0..n {
            let q = get_str(&mut r)?.split('\u{1f}').map(String::from).collect();
            let a = get_str(&mut r)?.split('\u{1f}').map(String::from).collect();
            lessons.push((q, a));
        }
    }
    if v3 {
        let mut d = OpReadout::default();
        d.n_learned = get(&mut r)? as u32;
        let n = get(&mut r)? as usize;
        for _ in 0..n { let o = get_str(&mut r)?; d.op(&o); }
        let m = get(&mut r)? as usize;
        for _ in 0..m {
            let u = get(&mut r)? as NeuronId;
            d.total.insert(u, get(&mut r)? as u32);
            let k = get(&mut r)? as usize;
            let mut v = std::collections::HashMap::with_capacity(k);
            for _ in 0..k { let o = get(&mut r)? as u32; v.insert(o, get(&mut r)? as u32 as i32); }
            d.votes.insert(u, v);
        }
        *rod = d;
    }
    if v4 {
        let n = get(&mut r)? as usize;
        let mut ks = Links::with_capacity(n);
        for _ in 0..n {
            let m = get(&mut r)? as usize;
            let mut kq = Vec::with_capacity(m);
            for _ in 0..m { kq.push(f32::from_bits(get(&mut r)? as u32)); }
            let m = get(&mut r)? as usize;
            let mut ka = Vec::with_capacity(m);
            for _ in 0..m { ka.push(f32::from_bits(get(&mut r)? as u32)); }
            ks.push((kq, ka));
        }
        *links = ks;
    }
    Ok(())
}

// TALK CACHE (v28, AVA_TALKCACHE=<file>). a woken brain re-takes her whole next-word exam
// (18,848 positions, ~25 s) before she says a word, only for what talk reads from it: the margin
// deciles, the (rung x margin) cells her switch and her confidence come from, the delayed vote's
// weight per rung. what the exam measures depends on her cortex, the corpus, the arguments and
// the switches the exam reads; never on her engrams or how she recalls them. so a run keeps
// everything the exam loop measured in a file, and a later run with the same key reads it back
// instead of re-taking the exam. everything after the loop (every reading printed, every table
// talk uses) is then computed by the same code from the very same numbers. a file made under any
// other key is never used: the key names this program, her brain, its sidecar and ledger and the
// corpus (each by path, bytes, modified time and a hash of the content), every argument, and
// every AVA_* switch except the ones that only change recall. unset = as before.

/// one value's trip to or from the cache. the same call writes it (saving) or reads it back in
/// place (loading), so the two directions can never disagree on the order.
enum Tape<'a> { W(Vec<u8>), R(&'a [u8]) }
trait Taped { fn tape(&mut self, t: &mut Tape) -> std::io::Result<()>; }
impl Taped for u64 {
    fn tape(&mut self, t: &mut Tape) -> std::io::Result<()> {
        match t { Tape::W(w) => put(w, *self), Tape::R(r) => { *self = get(r)?; Ok(()) } }
    }
}
impl Taped for usize {
    fn tape(&mut self, t: &mut Tape) -> std::io::Result<()> { let mut x = *self as u64; x.tape(t)?; *self = x as usize; Ok(()) }
}
impl Taped for u8 {
    fn tape(&mut self, t: &mut Tape) -> std::io::Result<()> { let mut x = *self as u64; x.tape(t)?; *self = x as u8; Ok(()) }
}
impl Taped for bool {
    fn tape(&mut self, t: &mut Tape) -> std::io::Result<()> { let mut x = *self as u64; x.tape(t)?; *self = x != 0; Ok(()) }
}
impl Taped for f32 {
    // exact bits: a reading computed from it is the reading the exam gave
    fn tape(&mut self, t: &mut Tape) -> std::io::Result<()> {
        let mut x = self.to_bits() as u64;
        x.tape(t)?;
        *self = f32::from_bits(x as u32);
        Ok(())
    }
}
impl Taped for String {
    fn tape(&mut self, t: &mut Tape) -> std::io::Result<()> {
        match t { Tape::W(w) => put_str(w, self), Tape::R(r) => { *self = get_str(r)?; Ok(()) } }
    }
}
impl<T: Taped, const N: usize> Taped for [T; N] {
    fn tape(&mut self, t: &mut Tape) -> std::io::Result<()> { for x in self.iter_mut() { x.tape(t)?; } Ok(()) }
}
impl<T: Taped + Default> Taped for Vec<T> {
    fn tape(&mut self, t: &mut Tape) -> std::io::Result<()> {
        let mut n = self.len();
        n.tape(t)?;
        if let Tape::R(r) = t {
            // every element takes at least one byte: a length past the end is a damaged file
            if n > r.len() { return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "damaged talk cache")); }
            *self = std::iter::repeat_with(T::default).take(n).collect();
        }
        for x in self.iter_mut() { x.tape(t)?; }
        Ok(())
    }
}
impl<A: Taped, B: Taped> Taped for (A, B) {
    fn tape(&mut self, t: &mut Tape) -> std::io::Result<()> { self.0.tape(t)?; self.1.tape(t) }
}
impl<A: Taped, B: Taped, C: Taped> Taped for (A, B, C) {
    fn tape(&mut self, t: &mut Tape) -> std::io::Result<()> { self.0.tape(t)?; self.1.tape(t)?; self.2.tape(t) }
}
impl<A: Taped, B: Taped, C: Taped, D: Taped> Taped for (A, B, C, D) {
    fn tape(&mut self, t: &mut Tape) -> std::io::Result<()> { self.0.tape(t)?; self.1.tape(t)?; self.2.tape(t)?; self.3.tape(t) }
}
/// everything her next-word exam loop measures, under the loop's own names
#[derive(Default)]
struct Exam {
    n_all: usize, t1: usize, t5: usize, changed: usize, fixed: usize, broke: usize,
    rung: [usize; 4], hit: [usize; 4], reach: [usize; 5], reach_rare: [usize; 5], kappa: [usize; 4],
    kappa_rare: [usize; 4], pool_hit: [usize; 4], pool_one: [usize; 4],
    kn1: [[[usize; 4]; 3]; 2], kn5: [[usize; 3]; 2], nat: [[usize; 4]; 3],
    pdom: Vec<usize>, pok: Vec<bool>, cal: Vec<(usize, u8, f32, bool)>, natr: Vec<[bool; 3]>, knr: Vec<[bool; 2]>,
    mixv: Vec<(Vec<(String, f32)>, Vec<(String, f32)>, String)>, compv: Vec<(Vec<(String, f32)>, usize)>,
    delv: Vec<Vec<(String, f32)>>,
}
const EXAM_MAGIC: &[u8; 8] = b"AVAEXAM1";
impl Exam {
    fn tape(&mut self, t: &mut Tape) -> std::io::Result<()> {
        let e = self;
        e.n_all.tape(t)?; e.t1.tape(t)?; e.t5.tape(t)?; e.changed.tape(t)?; e.fixed.tape(t)?; e.broke.tape(t)?;
        e.rung.tape(t)?; e.hit.tape(t)?; e.reach.tape(t)?; e.reach_rare.tape(t)?; e.kappa.tape(t)?;
        e.kappa_rare.tape(t)?; e.pool_hit.tape(t)?; e.pool_one.tape(t)?;
        e.kn1.tape(t)?; e.kn5.tape(t)?; e.nat.tape(t)?;
        e.pdom.tape(t)?; e.pok.tape(t)?; e.cal.tape(t)?; e.natr.tape(t)?; e.knr.tape(t)?;
        e.mixv.tape(t)?; e.compv.tape(t)?; e.delv.tape(t)
    }
    /// magic, key, the exam, then a hash of all of it. written beside the file and renamed over
    /// it, so a killed run never leaves half a cache; a file that is not a talk cache is left alone.
    fn save(&mut self, path: &str, key: &str) -> std::io::Result<()> {
        use std::io::Read;
        if let Ok(mut f) = fs::File::open(path) {
            let mut m = [0u8; 8];
            if f.read_exact(&mut m).is_err() || &m != EXAM_MAGIC {
                return Err(std::io::Error::new(std::io::ErrorKind::AlreadyExists, "a file that is not a talk cache is there"));
            }
        }
        let mut w = EXAM_MAGIC.to_vec();
        put_str(&mut w, key)?;
        let mut t = Tape::W(w);
        self.tape(&mut t)?;
        let Tape::W(mut w) = t else { unreachable!() };
        let h = words(0xcbf2_9ce4_8422_2325, &w);
        put(&mut w, h)?;
        let part = format!("{path}.{}.part", std::process::id());
        // a write or rename that fails leaves nothing behind
        let done = fs::write(&part, &w).and_then(|_| fs::rename(&part, path));
        if done.is_err() { fs::remove_file(&part).ok(); }
        done
    }
    /// None unless the file is whole, undamaged, and was made under exactly this key
    fn load(path: &str, key: &str) -> Option<Exam> {
        let b = fs::read(path).ok()?;
        let (body, h) = b.split_at(b.len().checked_sub(8)?);
        if words(0xcbf2_9ce4_8422_2325, body) != u64::from_le_bytes(h.try_into().ok()?) { return None; }
        let mut r = body.strip_prefix(EXAM_MAGIC.as_slice())?;
        if get_str(&mut r).ok()? != key { return None; }
        let mut e = Exam::default();
        let mut t = Tape::R(r);
        e.tape(&mut t).ok()?;
        // and nothing left over: the file holds this exam and nothing else
        matches!(t, Tape::R(rest) if rest.is_empty()).then_some(e)
    }
}
/// a 64-bit hash of bytes, read 8 at a time (one changed word always changes it)
fn words(mut h: u64, b: &[u8]) -> u64 {
    let mut it = b.chunks_exact(8);
    for c in &mut it { h = (h ^ u64::from_le_bytes(c.try_into().unwrap())).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(31); }
    let r = it.remainder();
    if !r.is_empty() {
        let mut w = [0u8; 8];
        w[..r.len()].copy_from_slice(r);
        h = (h ^ u64::from_le_bytes(w)).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(31);
    }
    h
}
fn file_hash(path: &str) -> std::io::Result<u64> {
    use std::io::Read;
    let mut f = fs::File::open(path)?;
    let mut buf = vec![0u8; 1 << 22];
    let mut h = 0xcbf2_9ce4_8422_2325;
    loop {
        // whole buffers (a multiple of 8) until the last, so the hash never depends on how reads split
        let mut n = 0;
        while n < buf.len() { match f.read(&mut buf[n..])? { 0 => break, k => n += k } }
        h = words(h, &buf[..n]);
        if n < buf.len() { return Ok(h); }
    }
}
/// the cache's key, as text: everything the exam depends on (see TALK CACHE above)
fn talk_key(args: &[String], brain: &str, corpus: &str) -> String {
    // a file by where it is, its bytes, when it last changed, and a hash of what it holds
    let file = |p: &str| -> String {
        let at = fs::canonicalize(p).map_or_else(|_| p.to_string(), |c| c.display().to_string());
        match (fs::metadata(p), file_hash(p)) {
            (Ok(m), Ok(h)) => format!("{at}\t{}\t{}\t{h:016x}", m.len(), m.modified().ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos())),
            _ => format!("{at}\tmissing"),
        }
    };
    let exe = env::current_exe().map_or_else(|_| "unknown".to_string(), |p| file(&p.to_string_lossy()));
    let mut k = vec![format!("exe\t{exe}"), format!("brain\t{}", file(brain)),
        format!("more\t{}", file(&format!("{brain}.more"))), format!("ledger\t{}", file(&format!("{brain}.ledger"))),
        format!("corpus\t{}", file(corpus)), format!("args\t{}", args.get(1..).unwrap_or(&[]).join("\t"))];
    // left out: the switches read only by recall (after the exam) or only on a night replayed in
    // this run (such a run never uses the cache), and the cache's own name. Windows reads
    // ava_lemma as AVA_LEMMA, so names are compared in capitals.
    let recall_only = |n: &str| ["AVA_LEMMA", "AVA_MEANING", "AVA_MEANING_POW", "AVA_MEANING_FREE", "AVA_QKIND",
        "AVA_CUE", "AVA_COMPETE", "AVA_SEPARATE", "AVA_LINKS", "AVA_LINK_ETA", "AVA_LINK_DUMP", "AVA_TALKCACHE"].contains(&n);
    let mut vars: Vec<String> = env::vars_os()
        .map(|(n, v)| (n.to_string_lossy().to_ascii_uppercase(), v.to_string_lossy().into_owned()))
        .filter(|(n, _)| n.starts_with("AVA_") && !recall_only(n))
        .map(|(n, v)| format!("env\t{n}={v}")).collect();
    vars.sort();
    k.extend(vars);
    k.join("\n")
}

// THREADS (v28, AVA_THREADS=<n>). the reply exam asks every held-out prompt for whole replies,
// one prompt after another. each prompt's replies depend only on her brain and that prompt, so n
// cores can work them out at once; they are put back in the prompts' order, and every count and
// sum after that runs one prompt at a time in that order, exactly as before. so the numbers are
// the 1-core numbers, bit for bit, whatever the threads' timing. unset = 1 core, the old loop.
fn threads() -> usize {
    env::var("AVA_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1).max(1)
}
/// f over every item on n threads, each taking the next item nobody has taken; results in the
/// items' order
fn par_map<T: Sync, R: Send>(items: &[T], n: usize, f: impl Fn(usize, &T) -> R + Sync) -> Vec<R> {
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut out: Vec<Option<R>> = (0..items.len()).map(|_| None).collect();
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..n.min(items.len())).map(|_| std::thread::Builder::new().stack_size(16 << 20)
            .spawn_scoped(sc, || {
                let mut mine = Vec::new();
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if i >= items.len() { return mine; }
                    mine.push((i, f(i, &items[i])));
                }
            }).expect("AVA_THREADS: could not start a thread")).collect();
        for h in hs { for (i, r) in h.join().expect("AVA_THREADS: a thread failed") { out[i] = Some(r); } }
    });
    out.into_iter().map(|r| r.expect("AVA_THREADS: an item was never done")).collect()
}
/// her engrams and their links, which the reply exam's threads read at once: RefCell's borrow /
/// borrow_mut over a read-write lock (only talk's "=" and sleep write, never during the exam)
struct Shared<T>(std::sync::RwLock<T>);
impl<T> Shared<T> {
    fn new(x: T) -> Shared<T> { Shared(std::sync::RwLock::new(x)) }
    fn borrow(&self) -> std::sync::RwLockReadGuard<'_, T> { self.0.read().unwrap() }
    fn borrow_mut(&self) -> std::sync::RwLockWriteGuard<'_, T> { self.0.write().unwrap() }
}

/// count-min read of one feature from a per-unit table: a gram feature owns `per_tag`
/// hashed units, any of which may be shared, so the smallest count is the honest one.
fn cm<V: Copy + Into<f64>>(t: &std::collections::HashMap<NeuronId, std::collections::HashMap<u32, V>>,
    us: &[NeuronId]) -> Vec<(u32, f32)>
{
    let Some(first) = us.first().and_then(|u| t.get(u)) else { return Vec::new() };
    let mut out: Vec<(u32, f32)> = first.iter().filter_map(|(&o, _)| {
        let m = us.iter().map(|u| t.get(u).and_then(|v| v.get(&o)).map_or(0.0, |&x| x.into()))
            .fold(f64::INFINITY, f64::min);
        if m > 0.0 { Some((o, m as f32)) } else { None }
    }).collect();
    out.sort_unstable_by_key(|x| x.0);
    out
}

impl Stage {
    fn new(n_id: u32, n_gram: u32) -> Stage {
        Stage { fac: Factored::new(n_id, n_gram, K_ID, 2), ro: OpReadout::default(),
                nbrs: None, mode: 0, rungs: 0, per: 0, cls: None, bits: 0, cont: None,
                rod: OpReadout::default(), prev: None }
    }
    /// `extra` is stage 1's guess, empty for stage 1 itself.
    ///
    /// `read` distinguishes the two directions, which matters only in neighbour mode 2:
    /// backoff is a READ-time fallback. Writes stay literal so a memory is always filed
    /// under its own spelling, and the approximate addresses are tried only when the
    /// literal one turns out not to exist.
    fn feats(&self, q: &[String], ctx: &[&str], pos: usize, extra: Option<&str>, read: bool)
        -> (Vec<String>, Vec<String>)
    {
        let (mut id, g) = match (&self.nbrs, self.mode, read) {
            (Some(n), 1, _) => reply::feats_near(q, ctx, pos, n),          // v12: flat
            (Some(n), 2, true) => reply::feats_backoff(q, ctx, pos, n, &self.fac.id.feat_id),
            // mode 3: ordered ladder. writes lay down every rung, reads take the
            // finest rung that exists and only fall through when there is none.
            (n, 3, true) => reply::feats_ladder(
                q, ctx, pos, self.rungs, n.as_ref(), &self.fac.id.feat_id),
            _ => match &self.cls {
                Some(c) => reply::feats_class(q, ctx, pos, self.rungs, self.bits, c),
                None => reply::feats_rungs(q, ctx, pos, self.rungs),
            },
        };
        if let Some(w) = extra {
            id.push(format!("S1:{w}|R:{}", ctx.join(" ")));
            id.push(format!("S1:{w}|P1:{}", ctx.last().copied().unwrap_or(BOS)));
        }
        (id, g)
    }
    fn settle(&mut self, q: &[String], ctx: &[&str], pos: usize, extra: Option<&str>) {
        let (id, g) = self.feats(q, ctx, pos, extra, false);
        if self.per > 0 { self.fac.learn_feats_each(&id, &g, self.per); }
        else { self.fac.learn_feats(&id, &g); }
    }
    fn code(&self, id: &[String], g: &[String]) -> Vec<NeuronId> {
        let (idc, gc) = if self.per > 0 { self.fac.query_feats_each(id, g, self.per) }
                        else { self.fac.query_feats(id, g, K_ID) };
        seq::code_of(idc, &gc)
    }
    fn write(&mut self, brain: &mut v4::Brain, q: &[String], ctx: &[&str], pos: usize,
             extra: Option<&str>, target: &str) {
        let (id, g) = self.feats(q, ctx, pos, extra, false);
        let code = self.code(&id, &g);
        if synapses_on() { brain.learn_bulk(&code, 1); }
        if self.cont.is_some() {
            let o = self.ro.op(target);
            // "never voted for w" under count-min: at least one of the feature's units is at 0
            let fresh = |ro: &OpReadout, gram: &GramAllocator, f: &str| gram.units(f).iter()
                .any(|u| ro.votes.get(u).and_then(|v| v.get(&o)).map_or(true, |&c| c <= 0));
            let b2 = g.iter().find(|f| f.starts_with("B2:"));
            let b1 = g.iter().find(|f| f.starts_with("B1:"));
            if let (Some(b2), Some(b1)) = (b2, b1) {
                let new2 = fresh(&self.ro, &self.fac.gram, b2);
                let new1 = fresh(&self.ro, &self.fac.gram, b1);
                let us = self.fac.gram.units(b1);
                let u2 = self.fac.gram.units(b2);
                let u3 = self.fac.gram.units(&format!("B3:{}", ctx.join(" ")));
                let ct = self.cont.as_mut().unwrap();
                let new3 = u3.iter().any(|u| ct.b3.get(u).and_then(|v| v.get(&o)).map_or(true, |&c| c == 0));
                if new3 { for &u in &u2 { *ct.c2.entry(u).or_default().entry(o).or_default() += 1; } }
                for u in u3 { *ct.b3.entry(u).or_default().entry(o).or_default() += 1; }
                if new2 { for u in us { *ct.c1.entry(u).or_default().entry(o).or_default() += 1; } }
                if new1 { *ct.c0.entry(o).or_default() += 1; }
                *ct.r0.entry(o).or_default() += 1;
            }
        }
        if delay_on() {
            if pos > 0 { if let Some(p) = &self.prev { self.rod.learn(p, target); } }
            self.prev = Some(code.clone());
        }
        self.ro.learn(&code, target);
    }
    /// KN-style interpolation over HER OWN units: `B2:x y` votes (count-min), then `B1:y`,
    /// then a unigram, trusting the most specific level and passing down only the
    /// discounted mass. `cont` = novelty-gated counts at the lower two levels (kneser-ney);
    /// otherwise raw votes everywhere (interpolation alone).
    ///
    /// `deep` (v22) puts `B3:x y z` on top, read from its own table, and turns `B2` into
    /// a novelty-gated level beneath it — kneser-ney with a 3-word context.
    fn native(&self, g: &[String], ctx: &[&str], cont: bool, deep: bool, p: &mut Vec<f32>) -> Option<String> {
        self.native_k(g, ctx, cont, deep, 1, p).and_then(|v| v.into_iter().next())
    }
    /// the same decode, returning her top `k` words (for trying several openings).
    fn native_k(&self, g: &[String], ctx: &[&str], cont: bool, deep: bool, k: usize, p: &mut Vec<f32>)
        -> Option<Vec<String>>
    {
        const D: f32 = 0.75;
        let ct = self.cont.as_ref()?;
        let v = self.ro.ops.len();
        if v == 0 { return None; }
        let (b2, b1) = (g.iter().find(|f| f.starts_with("B2:"))?, g.iter().find(|f| f.starts_with("B1:"))?);
        let mut l0: Vec<(u32, f32)> = (if cont { &ct.c0 } else { &ct.r0 }).iter().map(|(&o, &c)| (o, c as f32)).collect();
        l0.sort_unstable_by_key(|x| x.0);
        let l1 = if cont { cm(&ct.c1, &self.fac.gram.units(b1)) } else { cm(&self.ro.votes, &self.fac.gram.units(b1)) };
        let (l2, l3) = if deep {
            (cm(&ct.c2, &self.fac.gram.units(b2)), cm(&ct.b3, &self.fac.gram.units(&format!("B3:{}", ctx.join(" ")))))
        } else {
            (cm(&self.ro.votes, &self.fac.gram.units(b2)), Vec::new())
        };
        p.clear();
        p.resize(v, 1.0 / v as f32);
        for lvl in [&l0, &l1, &l2, &l3] {
            let tot: f32 = lvl.iter().map(|x| x.1).sum();
            if tot <= 0.0 { continue; }
            let gam = D * lvl.len() as f32 / tot;
            for x in p.iter_mut() { *x *= gam; }
            for &(o, c) in lvl.iter() { p[o as usize] += (c - D).max(0.0) / tot; }
        }
        // top-k by insertion; ties keep the lower index, so top-1 is the old argmax exactly
        let mut best: Vec<(f32, usize)> = Vec::with_capacity(k + 1);
        for (i, &x) in p.iter().enumerate() {
            if best.len() < k || x > best[best.len() - 1].0 {
                let at = best.iter().position(|b| x > b.0).unwrap_or(best.len());
                best.insert(at, (x, i));
                best.truncate(k);
            }
        }
        Some(best.into_iter().map(|(_, i)| self.ro.ops[i].clone()).collect())
    }
    fn ask(&self, q: &[String], ctx: &[&str], pos: usize, extra: Option<&str>, top: usize)
        -> Vec<(String, f32)>
    {
        let (id, g) = self.feats(q, ctx, pos, extra, true);
        self.ro.decode(&self.code(&id, &g), top)
    }
}

// ---- v29: WP2 working memory (AVA_WM), WP3 the previous turn (AVA_PREV), WP5 hearing each exchange
// once (AVA_DEDUP, in main), and the clean exam's halves (AVA_MASK, AVA_MASK_HALF, AVA_DUMP). each is
// off unless set, and then nothing here runs: v28, bit for bit.

/// THE CONVERSATION VOTE (v29 WP2 `AVA_WM=1`, WP3 `AVA_PREV=1`). people keep a running sense of what
/// is being talked about (prefrontal persistent activity) and remember what was said just before
/// (hippocampal short-term memory); her cortex reads only the prompt and her last 3 words.
///   THE TRACE. every word in play holds an activation that fades x`decay` with each later word heard
///   or said. AVA_WM: the prompt's words, a = decay^(words she has said), and each word she has said,
///   a = decay^(words said since). AVA_PREV: the turn before the prompt, the older part of the same
///   conversation, a = decay^(prompt length + words she has said). a word in two places keeps its
///   fresher activation.
///   THE LINKS (Hebbian, from the stretch she heard, never the exam). each time a reply word w came,
///   every word u then in play (the same parts of the trace, counted by presence) is linked to it.
///   c = times w came with u in play, n(u) = reply words that came with u in play, m(w) = times w came,
///   N = reply words heard, P(w) = m(w)/N. the link (default, AVA_WM_LINK=raise) is how much u in play
///   raises w's chance, in probability:
///     s(u, w) = P~(w | u) - P(w),  P~(w | u) = (c + α·P(w)) / (n(u) + α),  α = CONVO_ALPHA = 100 events
///   (the estimate shrunk toward the base rate, so a word seen once with u counts for little, and a word
///   that comes as often with u in play as without it gains nothing). positive links only; each word
///   keeps its `keep` strongest (pruning). AVA_WM_LINK=pmi: the screening tries s1-s3, v26's priming link
///   widened from "question word -> answer word" to "word in play -> the word that came":
///     s(u, w) = max(0, ln(c·N / (n(u)·m(w)))) · c/(c+1)   (positive PMI, discounted when thin)
///   it ranked rare words first (after "what time is it ?": weeding, bothers, haired, icy, ...) and was
///   replaced by raise on the screen half (SCREEN_LOG.md, v29 wp2).
///   THE VOTE. v(w) = Σ_u a(u)·s(u, w), its top 10 normalised to sum 1. it reaches what she says as
///   the delayed synapses do, never as a rule: vote(w) + b·v(w), b per rung learned on the exam (the
///   other half for the exam's own lines, the whole exam when she speaks). b = 0 is possible.
const CONVO_DECAY: f32 = 0.9;
const CONVO_KEEP: usize = 64;
///   TWO FORMS of reaching her choice (AVA_WM_FORM, both graded, the weight learned per rung as above):
///     vote (default): added beside her vote, as above; it can bring in a word her vote lacks.
///     gain: the support raises the gain of words her vote already holds (v26's priming form):
///       score(w) = vote(w) · exp(g · support(w)), support(w) = Σ_u a(u)·s(u, w) / Σ_u a(u),
///       over her vote's top 10, g per rung from {0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2}.
///   (the gain form was screened with pmi links only, s3; raise links are in probability units, far
///   below what its g range was set for.)
const CONVO_ALPHA: f64 = 100.0;
const CONVO_B: [f32; 8] = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 1000.0];
const CONVO_G: [f32; 10] = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0];
struct Convo { ids: std::collections::HashMap<String, u32>, words: Vec<String>, links: Vec<Vec<(u32, f32)>>,
               wm: bool, prev: bool, decay: f32, gain: bool }
impl Convo {
    fn build(train: &[(Vec<String>, Vec<String>)], prevs: &[Vec<String>], wm: bool, prev: bool, decay: f32, keep: usize, raise: bool) -> Convo {
        use std::collections::HashMap;
        let (mut ids, mut words): (HashMap<String, u32>, Vec<String>) = (HashMap::new(), Vec::new());
        fn id_of(w: &str, ids: &mut HashMap<String, u32>, words: &mut Vec<String>) -> u32 {
            if let Some(&i) = ids.get(w) { return i; }
            let i = words.len() as u32;
            ids.insert(w.to_string(), i);
            words.push(w.to_string());
            i
        }
        let mut co: HashMap<u64, u32> = HashMap::new();
        let (mut n, mut m, mut heard) = (Vec::<u32>::new(), Vec::<u32>::new(), 0u64);
        let mut play: Vec<u32> = Vec::new();
        for (i, (q, r)) in train.iter().enumerate() {
            play.clear();
            if prev { for w in prevs.get(i).map_or(&[][..], |p| p.as_slice()) { play.push(id_of(w, &mut ids, &mut words)); } }
            if wm { for w in q { play.push(id_of(w, &mut ids, &mut words)); } }
            play.sort_unstable();
            play.dedup();
            for w in r {
                let wi = id_of(w, &mut ids, &mut words);
                if n.len() < words.len() { n.resize(words.len(), 0); m.resize(words.len(), 0); }
                heard += 1;
                m[wi as usize] += 1;
                for &u in &play { n[u as usize] += 1; *co.entry((u as u64) << 32 | wi as u64).or_default() += 1; }
                if wm { if let Err(at) = play.binary_search(&wi) { play.insert(at, wi); } }
            }
        }
        n.resize(words.len(), 0);
        m.resize(words.len(), 0);
        let mut links: Vec<Vec<(u32, f32)>> = vec![Vec::new(); words.len()];
        for (&k, &c) in &co {
            let (u, w, c) = ((k >> 32) as usize, (k & 0xffff_ffff) as usize, c as f64);
            if raise {
                // how much u in play raises w's chance, in probability: P~(w | u) - P(w), with
                // P~(w | u) = (c + α·P(w)) / (n(u) + α), shrunk toward the base rate by α events
                let pw = m[w] as f64 / heard as f64;
                let s = (c - n[u] as f64 * pw) / (n[u] as f64 + CONVO_ALPHA);
                if s > 0.0 { links[u].push((w as u32, s as f32)); }
            } else {
                let pmi = (c * heard as f64 / (n[u] as f64 * m[w] as f64)).ln();
                if pmi > 0.0 { links[u].push((w as u32, (pmi * c / (c + 1.0)) as f32)); }
            }
        }
        // strongest first, ties by word (so the order never depends on the hash map's)
        for l in links.iter_mut() {
            l.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            l.truncate(keep);
            l.shrink_to_fit();
        }
        Convo { ids, words, links, wm, prev, decay, gain: false }
    }
    /// the trace after `said` (her reply so far): Σ_u a(u)·s(u, w) for every word w it reaches, and Σ_u a(u)
    fn support(&self, q: &[String], said: &[String], prev: &[String]) -> (std::collections::HashMap<u32, f32>, f32) {
        let t = said.len();
        let mut act: Vec<(u32, f32)> = Vec::new();
        let mut put = |w: &str, a: f32| {
            if let Some(&u) = self.ids.get(w) {
                match act.iter_mut().find(|x| x.0 == u) { Some(x) => x.1 = x.1.max(a), None => act.push((u, a)) }
            }
        };
        if self.prev { let a = self.decay.powi((q.len() + t) as i32); for w in prev { put(w, a); } }
        if self.wm {
            let a = self.decay.powi(t as i32);
            for w in q { put(w, a); }
            for (s, w) in said.iter().enumerate() { put(w, self.decay.powi((t - 1 - s) as i32)); }
        }
        let mut acc: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
        for &(u, a) in &act { for &(w, s) in &self.links[u as usize] { *acc.entry(w).or_default() += a * s; } }
        (acc, act.iter().map(|x| x.1).sum())
    }
    /// GAIN form (AVA_WM_FORM=gain): how strongly the words in play support each of `ws`, the mean
    /// link weighted by activation, Σ_u a(u)·s(u, w) / Σ_u a(u) (0 for a word no link reaches)
    fn gains(&self, q: &[String], said: &[String], prev: &[String], ws: &[(String, f32)]) -> Vec<f32> {
        let (acc, asum) = self.support(q, said, prev);
        ws.iter().map(|(w, _)| if asum > 0.0 { self.ids.get(w).and_then(|u| acc.get(u)).map_or(0.0, |x| x / asum) } else { 0.0 }).collect()
    }
    /// VOTE form (default): the trace's vote for the next word, its top 10 summing to 1
    fn vote(&self, q: &[String], said: &[String], prev: &[String]) -> Vec<(String, f32)> {
        let (acc, _) = self.support(q, said, prev);
        let mut v: Vec<(u32, f32)> = acc.into_iter().filter(|x| x.1 > 0.0).collect();
        v.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(10);
        let tot: f32 = v.iter().map(|x| x.1).sum();
        v.into_iter().map(|(w, x)| (self.words[w as usize].clone(), x / tot)).collect()
    }
}

/// THE CLEAN EXAM's HALVES (v29, `AVA_MASK=<mask from exam_clean>`, `AVA_MASK_HALF=screen|confirm`,
/// default screen). clean = an exchange whose pair and reply she never heard in training. the clean
/// exchanges, in exam order, alternate: screen = the 0th, 2nd, 4th, ...; confirm = the 1st, 3rd, ....
/// a reading only: it chooses which positions and prompts a line is ALSO counted over, and changes
/// nothing she does. the mask must match this exam line by line (index, domain, reply length).
struct Mask { path: String, half: &'static str, on: Vec<bool>, clean: usize }
fn load_mask(test: &[(Vec<String>, Vec<String>)], test_dom: &[usize], doms: &[String]) -> std::io::Result<Option<Mask>> {
    let bad = |m: String| std::io::Error::new(std::io::ErrorKind::InvalidData, m);
    let Ok(path) = env::var("AVA_MASK") else {
        if env::var("AVA_MASK_HALF").is_ok() || env::var("AVA_DUMP").is_ok() { return Err(bad("AVA_MASK_HALF and AVA_DUMP need AVA_MASK".into())); }
        return Ok(None);
    };
    let half: &'static str = match env::var("AVA_MASK_HALF").as_deref() {
        Err(_) | Ok("screen") => "screen",
        Ok("confirm") => "confirm",
        Ok(h) => return Err(bad(format!("AVA_MASK_HALF must be screen or confirm, got {h:?}"))),
    };
    let text = fs::read_to_string(&path)?;
    let mut lines = text.lines();
    if lines.next() != Some("index\tdomain\tpair_heard\treply_heard\tprompt_heard\tsaid_heard\treply_tokens") {
        return Err(bad(format!("AVA_MASK {path}: not a mask from exam_clean")));
    }
    let (mut clean, mut k) = (Vec::new(), 0usize);
    for (i, l) in lines.enumerate() {
        let f: Vec<&str> = l.split('\t').collect();
        let bit = |s: &str| s == "0" || s == "1";
        let ok = f.len() == 7 && i < test.len() && f[0] == i.to_string() && f[1] == doms[test_dom[i]]
            && f[2..6].iter().all(|s| bit(s)) && f[6] == test[i].1.len().to_string();
        if !ok { return Err(bad(format!("AVA_MASK {path}: line {} does not match this exam", i + 2))); }
        if f[2] == "0" && f[3] == "0" { clean.push(i); }
        k = i + 1;
    }
    if k != test.len() { return Err(bad(format!("AVA_MASK {path}: {k} exchanges, this exam has {}", test.len()))); }
    let mut on = vec![false; test.len()];
    let first = if half == "screen" { 0 } else { 1 };
    for (j, &i) in clean.iter().enumerate() { if j % 2 == first { on[i] = true; } }
    Ok(Some(Mask { path, half, on, clean: clean.len() }))
}

fn main() -> std::io::Result<()> {
    let a: Vec<String> = env::args().collect();
    let n: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(30_000);
    let two: bool = a.get(2).and_then(|s| s.parse::<u32>().ok()).unwrap_or(0) != 0;
    // CREDIT ASSIGNMENT: how many error-correcting epochs after the Hebbian write.
    let epochs: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    // NEIGHBOUR KEYS: address each memory by distributionally similar words too.
    //   0 off   1 flat (v12: neighbours compete with the exact key at write AND read)
    //   2 ordered backoff (v13: read-time only, and only when no exact address exists)
    let near: u8 = a.get(4).and_then(|s| s.parse::<u8>().ok()).unwrap_or(0);
    // REPLY-SIDE LADDER RUNGS: bitmask, see reply::feats_rungs.
    let rungs: u32 = a.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);
    // PER-KEY UNITS: 0 = the old shared budget of K_ID winners across all identity
    // keys; >0 = each key competes alone for this many units and the code is the union.
    let per: usize = a.get(6).and_then(|s| s.parse().ok()).unwrap_or(0);
    // IDENTITY POOL, in millions. 5M was hard-coded and it is a CAPACITY, not a
    // constant: doubling the corpus to 60k exchanges against the same 5M neurons took
    // top-1 from 36.47% to 23.32% and accuracy on fully-addressed positions from
    // 87.37% to 16.92%. An addressed position scoring 17% is not a model that has
    // learned the wrong thing, it is a model whose addresses have collided — the
    // reply-side rungs alone intern roughly 5 keys per position, so 58,800 exchanges
    // ask for ~4.7M unit-slots out of 5M neurons and every neuron is claimed.
    // v7's reachability law said this in one line: reachable = facts x 16.
    // CLASS-CROSSED KEYS: bitmask, see reply::push_class_keys. 0 = off.
    let bits: u32 = a.get(8).and_then(|s| s.parse().ok()).unwrap_or(0);
    let nclass: usize = a.get(9).and_then(|s| s.parse().ok()).unwrap_or(128);
    // PROBES: measurement only, never read by the model. bit 1 = the copy oracle / kappa.
    let probe: u32 = a.get(10).and_then(|s| s.parse().ok()).unwrap_or(0);
    // TALK: after the exam, chat with the routed model (needs probe bits 4 and 8: 12).
    let talk = a.get(11).map_or(false, |s| s == "talk");
    let pool: u32 = a.get(7).and_then(|s| s.parse::<u32>().ok())
        .map(|m| m * 1_000_000)
        .unwrap_or_else(|| 5_000_000 * ((n as u32 + 29_999) / 30_000).max(1));
    let (base, _) = mem_gb();

    // AVA_CORPUS replaces the corpus file; AVA_TEST, when set, is the EXACT held-out count.
    let corpus = env::var("AVA_CORPUS").unwrap_or_else(|_| CORPUS.to_string());
    let ava_test: Option<usize> = match env::var("AVA_TEST") {
        Ok(s) => Some(s.trim().parse().map_err(|_| std::io::Error::new(
            std::io::ErrorKind::InvalidInput, format!("AVA_TEST must be a whole number, got {s:?}")))?),
        Err(_) => None,
    };
    let raw = fs::read_to_string(&corpus)?;
    let want = n + ava_test.unwrap_or(TEST);
    let mut pairs: Vec<(Vec<String>, Vec<String>)> = Vec::with_capacity(want);
    // DOMAINS. `dom[i]` indexes `doms` for pair i, test and train alike. an untagged line
    // is dialogue and its first field is the previous turn, ignored as it always was.
    let (mut doms, mut dom, mut tagged) = (Vec::<String>::new(), Vec::<usize>::with_capacity(want), false);
    // tagged lines the length filter dropped. a school corpus is pre-filtered, so any drop
    // means the builder and this reader disagree; a drop inside the test block shifts it.
    let mut dropped: Vec<(String, usize)> = Vec::new();
    // v24: she HEARS EVERYTHING. a child hears full-length speech; the old filter (replies
    // <= 16 words, prompts <= 20) threw away 27% of her dialogue, and it is gone. what stays
    // the same is WHICH stretch of conversation she lives and WHICH questions the exam asks:
    //   exam = the first `k` short exchanges (exactly the old exam), plus the long exchanges
    //          among them, held out too and reported as "<domain> long";
    //   life = onward to the n-th short exchange, the long ones in that stretch now included.
    // so a run hears a strict superset of what the same run heard before. AVA_SHORT_ONLY=1
    // brings the old filter back, only to re-run old experiments as controls.
    let short_only = env::var("AVA_SHORT_ONLY").map_or(false, |v| v == "1");
    let k_test = ava_test.unwrap_or(TEST);
    let (mut seen_short, mut n_region) = (0usize, 0usize);
    // v29 (AVA_PREV=1): the turn before each prompt, kept beside its pair (empty when the switch is
    // off, or for a tagged line, which has no previous turn). see `Convo`.
    let (wm_on, prev_on) = (env::var("AVA_WM").map_or(false, |v| v == "1"), env::var("AVA_PREV").map_or(false, |v| v == "1"));
    let mut prevs: Vec<Vec<String>> = Vec::with_capacity(want);
    for line in raw.lines() {
        let mut p = line.split('\t');
        let (Some(f0), Some(q), Some(r)) = (p.next(), p.next(), p.next()) else { continue };
        let tag = f0.strip_prefix('@').map(str::trim);
        let code = matches!(tag, Some("rust") | Some("ava_code")) || tag.map_or(false, |t| t.starts_with("code_"));
        let tok = |s: &str| if code { code_tokenize(s) } else { seq::tokenize(s) };
        let (qt, rt) = (tok(q), tok(r));
        let short = rt.len() <= 16 && qt.len() <= 20;
        if !qt.is_empty() && !rt.is_empty() && (short || !short_only) {
            let held = !short_only && seen_short < k_test;
            if held { n_region += 1; }
            if short { seen_short += 1; }
            pairs.push((qt, rt));
            prevs.push(if prev_on && tag.is_none() { tok(f0) } else { Vec::new() });
            let base = tag.unwrap_or("dialogue");
            let name = if held && !short { format!("{base} long") } else { base.to_string() };
            let di = match doms.iter().position(|d| *d == name) {
                Some(d) => d,
                None => { doms.push(name); doms.len() - 1 }
            };
            dom.push(di);
            tagged |= tag.is_some();
            if if short_only { pairs.len() >= want } else { seen_short >= k_test + n } { break; }
        } else if let Some(t) = tag {
            match dropped.iter_mut().find(|d| d.0 == t) { Some(d) => d.1 += 1, None => dropped.push((t.to_string(), 1)) }
        }
    }
    if !dropped.is_empty() { println!("   !! tagged lines dropped by the length filter: {dropped:?}"); }
    drop(raw);
    let n_test = if !short_only { n_region.min(pairs.len().saturating_sub(1)) } else {
        match ava_test {
            Some(k) => k.min(pairs.len().saturating_sub(1)),
            None => TEST.min(pairs.len() / 4),
        }
    };
    // SPEAKING has no length rule either. the only stop is a watchdog far past anything she
    // has heard (her longest heard reply is 112 words), so a loop cannot hang the program;
    // every reply that reaches it is counted and printed. AVA_SHORT_ONLY keeps the old 16.
    let say_stop: usize = if short_only { 16 } else { 128 };
    let (test, train) = pairs.split_at(n_test);
    let (test_dom, train_dom) = dom.split_at(n_test);
    let (test_prev, train_prev) = prevs.split_at(n_test);
    // HEARING EACH EXCHANGE ONCE (v29 WP5, AVA_DEDUP=1). the corpus repeats itself: an exchange she
    // has already heard, word for word (same prompt, same reply), is left out of the stretch she
    // lives, everywhere it is read. a data switch, not a behaviour. it changes what she hears while
    // she lives, so it needs a brain that lives in this run: refused with AVA_LOAD.
    let dedup = env::var("AVA_DEDUP").map_or(false, |v| v == "1");
    if dedup && env::var("AVA_LOAD").is_ok() {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput,
            "AVA_DEDUP=1 changes what she hears while she lives; a woken brain (AVA_LOAD) has already lived"));
    }
    let kept: Vec<usize> = if dedup {
        let mut seen: std::collections::HashSet<(&[String], &[String])> = std::collections::HashSet::new();
        (0..train.len()).filter(|&i| seen.insert((train[i].0.as_slice(), train[i].1.as_slice()))).collect()
    } else { Vec::new() };
    let train_v: Vec<(Vec<String>, Vec<String>)> = kept.iter().map(|&i| train[i].clone()).collect();
    let train_dom_v: Vec<usize> = kept.iter().map(|&i| train_dom[i]).collect();
    let train_prev_v: Vec<Vec<String>> = kept.iter().map(|&i| train_prev[i].clone()).collect();
    let n_heard = train.len();
    let (train, train_dom, train_prev) = if dedup { (&train_v[..], &train_dom_v[..], &train_prev_v[..]) }
        else { (train, train_dom, train_prev) };
    // per-domain reports print only for a tagged or mixed corpus, so a plain triples run
    // prints exactly what it always did.
    let multi = doms.len() > 1 || tagged;
    println!("== depth {} : {} train / {} held out ==",
        if two { 2 } else { 1 }, train.len(), test.len());
    if dedup {
        println!("   AVA_DEDUP: she hears each exchange once: {} of {n_heard} were repeats of an earlier one, left out ({:.1}%)",
            n_heard - train.len(), (n_heard - train.len()) as f64 / n_heard.max(1) as f64 * 100.0);
    }
    if multi {
        let cnt = |ds: &[usize], d: usize| ds.iter().filter(|&&x| x == d).count();
        println!("   {} domains:{}", doms.len(), doms.iter().enumerate()
            .map(|(d, name)| format!("  {name} {}/{}", cnt(train_dom, d), cnt(test_dom, d)))
            .collect::<String>());
        println!("   (train/held-out pairs; @rust, @code_* and @ava_code read by the code tokenizer)");
    }
    // the clean exam's halves (v29, AVA_MASK): a reading only, see `Mask`
    let mask = load_mask(test, test_dom, &doms)?;
    if let Some(mk) = &mask {
        println!("   clean exam (AVA_MASK {}): {} of {} exchanges clean; lines below also count the {} half ({} exchanges)",
            mk.path, mk.clean, test.len(), mk.half, mk.on.iter().filter(|&&x| x).count());
        if probe & 20 != 20 { println!("   (the clean exam's next-word lines need probe bits 4 and 16; its reply line needs 32)"); }
    }
    let dump_path = env::var("AVA_DUMP").ok();
    let mut dump = String::new();

    let t0 = Instant::now();
    println!("   identity pool {:.1}M neurons for {} exchanges",
        pool as f64 / 1e6, train.len());
    let mut s1 = Stage::new(pool, 500_000);
    s1.rungs = rungs;
    s1.per = per;
    s1.bits = bits;
    if probe & 16 != 0 { s1.cont = Some(Cont::default()); }
    let mut fam: Option<Familiar> = if probe & 64 != 0 { Some(Familiar::default()) } else { None };
    // ENGRAMS (v24): every lesson she is taught is also kept whole, as the hippocampus keeps an
    // episode. it exists from the moment she is taught (talk), is rebuilt when she sleeps, and
    // is saved with her brain. probe 128 lets a question reactivate one (see `recall`).
    // (Shared, not RefCell, since v28: the reply exam's threads read them at once; AVA_THREADS)
    let lessons = Shared::new(Lessons::new());
    let links = Shared::new(Links::new());
    let mut heard = 0usize;
    if bits != 0 {
        let mut docs: Vec<&[String]> = Vec::with_capacity(train.len() * 2);
        for (q, r) in train { docs.push(q.as_slice()); docs.push(r.as_slice()); }
        let (prof, freq, _) = design_tests::wordclass::profiles(docs.into_iter(), 2, 20);
        let (cls, ex) = design_tests::wordclass::exemplar_classes(&prof, &freq, nclass);
        println!("   {} words -> {nclass} classes at {:?}", cls.len(), t0.elapsed());
        // what did the classes actually carve up? print a few, largest first.
        let mut members: std::collections::HashMap<u16, Vec<&String>> = Default::default();
        for (w, c) in &cls { members.entry(*c).or_default().push(w); }
        let mut ks: Vec<u16> = members.keys().copied().collect();
        ks.sort_unstable_by_key(|k| std::cmp::Reverse(members[k].len()));
        for k in ks.iter().take(5) {
            let mut m: Vec<&str> = members[k].iter().map(|s| s.as_str()).collect();
            m.sort_unstable();
            println!("     [{}] {:>4} words: {}", ex[*k as usize], m.len(),
                m.iter().take(12).cloned().collect::<Vec<_>>().join(" "));
        }
        println!("   class bits 0b{bits:04b}");
        s1.cls = Some(cls);
    }
    if per > 0 { println!("   per-key identity units: {per} (union, not a shared budget)"); }
    if rungs != 0 { println!("   reply-side rungs: 0b{rungs:04b}"); }
    if near == 3 { s1.mode = 3; }
    if near == 1 || near == 2 || near == 3 {
        let mut docs: Vec<&[String]> = Vec::with_capacity(train.len() * 2);
        for (q, r) in train { docs.push(q.as_slice()); docs.push(r.as_slice()); }
        let (prof, _, _) = design_tests::wordclass::profiles(docs.into_iter(), 2, 20);
        let nb = design_tests::wordclass::neighbours(&prof, 3, 0.08);
        println!("   learned neighbours for {} words at {:?}", nb.len(), t0.elapsed());
        for probe in ["monday", "eat", "happy"] {
            if let Some(n) = nb.get(probe) { println!("     {probe} ~ {}", n.join(" ")); }
        }
        s1.nbrs = Some(nb);
        if near != 3 { s1.mode = near; }
        println!("   neighbour mode {} ({})", near,
            if near == 1 { "flat, competes with the exact key" }
            else { "ordered backoff, read-time only" });
    }
    let mut b1 = v4::Brain::new(s1.fac.n_total, Params::default());
    // PERSISTENCE (v24): she is the same Ava tomorrow. AVA_LOAD=<file> wakes her from a saved
    // brain instead of living her life again; AVA_SAVE=<file> keeps her brain after she has
    // lived. <file> holds the synapses, addressing and readout (store.rs), <file>.more her
    // fallback's count tables and familiarity units. brain files stay local, never published.
    let woke = match env::var("AVA_LOAD") {
        Ok(path) => {
            let t = Instant::now();
            let (b, f, r) = design_tests::store::load(&path, Params::default())?;
            b1 = b; s1.fac = f; s1.ro = r;
            // a brain saved before v26 carries same-moment synapses nothing reads; let them go
            // (kept for AVA_SYNAPSES=1, and for AVA_COMPLETE, which reads them)
            if !synapses_on() && env::var("AVA_COMPLETE").is_err() {
                let n = b1.total_synapses();
                for u in b1.units.iter_mut() { u.out = Vec::new(); }
                if n > 0 { println!("   released {n} same-moment synapses (off since v26)"); }
            }
            load_more(&format!("{path}.more"), &mut s1.cont, &mut fam, &mut lessons.borrow_mut(), &mut s1.rod, &mut links.borrow_mut())?;
            fit_links(&lessons.borrow(), &mut links.borrow_mut());
            if !lessons.borrow().is_empty() { println!("   {} engrams (lessons kept whole)", lessons.borrow().len()); }
            println!("   woke from {path} in {:?} (no relearning)", t.elapsed());
            true
        }
        Err(_) => false,
    };
    let mut buf = Vec::new();

    // LIVE (v24, AVA_LIVE=1): no warm-up passes and no preview of her life. one pass; at
    // every position she first THINKS (guesses the next word, and the guess is scored),
    // then LEARNS it (settle + write). the lab protocol below settles twice over the
    // whole corpus before writing anything; a brain never gets to do that.
    let live = env::var("AVA_LIVE").map_or(false, |v| v == "1");
    let (mut life, mut block) = ([0usize; 2], [0usize; 2]);
    // RECAST (v24, AVA_RECAST=1|2, live only): see the loop below.
    let recast: u8 = if live { env::var("AVA_RECAST").ok().and_then(|v| v.parse().ok()).unwrap_or(0) } else { 0 };
    // her own replies over her life: [spoken, ended by herself, sum |her length - theirs|, corrections]
    let (mut rc_life, mut rc_block, mut rc_first) = ([0f64; 4], [0f64; 4], None::<[f64; 4]>);
    for pass in 0..3 {
        if woke { break; }
        if live && pass < 2 { continue; }
        for (i, (q, r)) in train.iter().enumerate() {
            if i % 5_000 == 0 && i > 0 && mem_gb().1 < MIN_FREE_GB {
                println!("   !! stopping, low memory"); break;
            }
            if live && i % 5_000 == 0 && i > 0 {
                println!("     lived {i:>7} exchanges: {:.2}% of words guessed before learning them (last 5k: {:.2}%)",
                    life[1] as f64 / life[0].max(1) as f64 * 100.0, block[1] as f64 / block[0].max(1) as f64 * 100.0);
                block = [0, 0];
                if recast > 0 {
                    let b = rc_block;
                    println!("       her own replies, last 5k: {:.1}% ended by herself, length off by {:.2} words, \
{:.2} corrections per reply", b[1] / b[0].max(1.0) * 100.0, b[2] / b[0].max(1.0), b[3] / b[0].max(1.0));
                    if rc_first.is_none() { rc_first = Some(b); }
                    rc_block = [0.0; 4];
                }
            }
            // RECAST (v24, AVA_RECAST). a child talks, hears how an adult would have said it,
            // and learns from the difference; nobody hands them a rule like "stop at 16 words".
            // before she hears this reply, she SAYS her own (memory only, greedy, nothing
            // learned yet). wherever her words have left the real reply, she learns, in the
            // state SHE talked herself into, what the appropriate next step was:
            //   1 = only the ending: past the point where the person stopped, the next step
            //       is to stop. this is how she learns how long a reply should be.
            //   2 = the ending, and otherwise the person's word at that position.
            // while she is on track (her words so far = theirs) nothing extra is written; the
            // normal live step below learns those very states. the better she gets, the less
            // she leaves the track and the less there is to correct: progressive by itself.
            if recast > 0 {
                let mut out: Vec<String> = Vec::new();
                let mut fixed = 0usize;
                while out.len() < say_stop {
                    let pos = out.len();
                    let have = pos.min(ORDER);
                    let mut c: Vec<&str> = vec![BOS; ORDER];
                    for k in 0..have { c[ORDER - have + k] = out[pos - have + k].as_str(); }
                    let w = s1.ask(q, &c, pos, None, 1).into_iter().next().map_or_else(|| EOS.to_string(), |x| x.0);
                    let on_track = pos <= r.len() && out[..] == r[..pos];
                    let should = r.get(pos).map_or(EOS, |s| s.as_str());
                    if !on_track && w != should && (recast == 2 || should == EOS) {
                        s1.settle(q, &c, pos, None);
                        s1.write(&mut b1, q, &c, pos, None, should);
                        fixed += 1;
                    }
                    if w == EOS { break; }
                    out.push(w);
                }
                let ended = out.len() < say_stop;
                let x = [1.0, ended as u8 as f64, (out.len() as f64 - r.len() as f64).abs(), fixed as f64];
                for k in 0..4 { rc_life[k] += x[k]; rc_block[k] += x[k]; }
            }
            // familiarity hears each prompt once, as she lives it (lab protocol: the write pass)
            if live || pass == 2 { if let Some(f) = fam.as_mut() { f.hear(q); heard += 1; } }
            reply::turns(r, ORDER, &mut buf);
            for t in &buf {
                if live {
                    let hit = s1.ask(q, &t.ctx, t.pos, None, 1).into_iter().next()
                        .map_or(false, |(w, _)| w == t.next);
                    life[0] += 1; block[0] += 1;
                    if hit { life[1] += 1; block[1] += 1; }
                    s1.settle(q, &t.ctx, t.pos, None);
                    s1.write(&mut b1, q, &t.ctx, t.pos, None, t.next);
                }
                else if pass < 2 { s1.settle(q, &t.ctx, t.pos, None); }
                else { s1.write(&mut b1, q, &t.ctx, t.pos, None, t.next); }
            }
        }
    }
    if live && !woke {
        println!("   LIVE: one pass, think then learn. lifetime {:.2}% of {} words guessed right before learning them",
            life[1] as f64 / life[0].max(1) as f64 * 100.0, life[0]);
        println!("TSV\tlife\t{}\t{:.2}", train.len(), life[1] as f64 / life[0].max(1) as f64 * 100.0);
    }
    if recast > 0 {
        let (l, f) = (rc_life, rc_first.unwrap_or(rc_life));
        println!("   RECAST {recast}: over her life {:.1}% of her own replies ended by herself, length off by {:.2} words, \
{:.2} corrections per reply (first 5k: {:.1}%, {:.2}, {:.2}; last 5k: {:.1}%, {:.2}, {:.2})",
            l[1] / l[0].max(1.0) * 100.0, l[2] / l[0].max(1.0), l[3] / l[0].max(1.0),
            f[1] / f[0].max(1.0) * 100.0, f[2] / f[0].max(1.0), f[3] / f[0].max(1.0),
            rc_block[1] / rc_block[0].max(1.0) * 100.0, rc_block[2] / rc_block[0].max(1.0), rc_block[3] / rc_block[0].max(1.0));
        println!("TSV\trecast\t{recast}\t{}\t{:.2}\t{:.3}\t{:.3}", rc_life[0], l[1] / l[0].max(1.0) * 100.0,
            l[2] / l[0].max(1.0), l[3] / l[0].max(1.0));
    }
    println!("   stage 1 trained at {:?}, {:.2} GB", t0.elapsed(), mem_gb().0 - base);
    if let Some(f) = fam.as_mut() {
        if !woke { f.settle(&train[..heard.min(train.len())]); }
        println!("   familiarity: {} word-pair units, judged against {} heard prompts", f.bi.len(), f.loo.len());
    }
    // SLEEP (v24, AVA_SLEEP=<log>[;<log>...]). the day's lessons wait in a log, the
    // hippocampus's pending store; sleep replays them into long-term memory with the very
    // settle/write she uses when she lives. a ledger (name, length, fnv of the bytes) beside
    // the brain file makes each log replay exactly once: counts add, so a second replay would
    // count twice. her learning is order-free (v17), so for her sleep is transfer and
    // keeping, not protection from interference.
    let mut ledger: Vec<String> = env::var("AVA_LOAD").ok()
        .and_then(|p| fs::read_to_string(format!("{p}.ledger")).ok())
        .map_or_else(Vec::new, |s| s.lines().map(String::from).collect());
    // a log replayed tonight is a new ledger line: her cortex is then not the saved one, and the
    // talk cache is neither read nor written (v28)
    let ledger0 = ledger.len();
    if let Ok(list) = env::var("AVA_SLEEP") {
        let mut night: Vec<(Vec<String>, Vec<String>)> = Vec::new();
        for file in list.split(';').map(str::trim).filter(|f| !f.is_empty()) {
            let bytes = fs::read(file)?;
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for &b in &bytes { h = (h ^ b as u64).wrapping_mul(0x100_0000_01b3); }
            let name = std::path::Path::new(file).file_name().map_or(file.to_string(), |n| n.to_string_lossy().into_owned());
            let sig = format!("{name}\t{}\t{h:016x}", bytes.len());
            if ledger.contains(&sig) { println!("   sleep: {name} was consolidated before; nothing replayed"); continue; }
            let (mut taught, mut words) = (0usize, 0usize);
            for line in String::from_utf8_lossy(&bytes).lines() {
                let f: Vec<&str> = line.split('\t').collect();
                if f.len() < 4 || f[0] != "@sleep" || f[3] != "teach" { continue; }
                let q: Vec<String> = f[1].split('\u{1f}').map(String::from).collect();
                let r: Vec<String> = f[2].split('\u{1f}').map(String::from).collect();
                let mut sb = Vec::new();
                reply::turns(&r, ORDER, &mut sb);
                for t in &sb { s1.settle(&q, &t.ctx, t.pos, None); s1.write(&mut b1, &q, &t.ctx, t.pos, None, t.next); words += 1; }
                if let Some(fm) = fam.as_mut() {
                    fm.hear(&q);
                    let x = fm.f(&q, true);
                    let at = fm.loo.partition_point(|&v| v < x);
                    fm.loo.insert(at, x);
                }
                taught += 1;
                // the replayed episode is also (re)kept whole, as an engram
                let e = (q, r);
                if links_on() { night.push(e); }
                else if !lessons.borrow().contains(&e) { lessons.borrow_mut().push(e); }
            }
            println!("   sleep: replayed {taught} lessons ({words} words) from {name} into long-term memory");
            ledger.push(sig);
        }
        // LINKS (v27): after the whole night's replay, so word rarity is the night's. pass 1
        // re-enacts the day's "=" events in order (the talk session is not saved); pass 2
        // replays each lesson once more. exactly LINK_PASSES, nothing else.
        if links_on() && !night.is_empty() {
            let eta = link_eta();
            let spec = |w: &str| spec_of(&s1, w);
            let (mut lb, mut kb) = (lessons.borrow_mut(), links.borrow_mut());
            for p in 0..LINK_PASSES {
                for (q, r) in &night { link_trial(&mut lb, &mut kb, q, r, p == 0, eta, &spec); }
            }
            let kept: Vec<f32> = (0..lb.len()).map(|i| link_kept(&lb, &kb, i, &spec)).collect();
            let gone = kb.iter().map(|(kq, ka)| kq.iter().chain(ka).filter(|&&k| k <= 0.0).count()).sum::<usize>();
            let all = kb.iter().map(|(kq, ka)| kq.len() + ka.len()).sum::<usize>();
            println!("   links (v27, η {eta}): {LINK_PASSES} passes over {} lessons; {} engrams keep {:.0}% of their question's links on average (least {:.0}%); {gone} of {all} links faded to 0",
                night.len(), lb.len(), kept.iter().sum::<f32>() / kept.len().max(1) as f32 * 100.0,
                kept.iter().copied().fold(f32::INFINITY, f32::min) * 100.0);
            if let Ok(path) = env::var("AVA_LINK_DUMP") {
                let mut d = String::new();
                for (i, ((q, r), (kq, ka))) in lb.iter().zip(kb.iter()).enumerate() {
                    let (es, ea) = episode_sets(q, r);
                    for (j, w) in es.iter().enumerate() { d += &format!("{i}\tq\t{w}\t{:.6}\t{:.6}\n", kq[j], spec(w)); }
                    for (j, w) in ea.iter().enumerate() { d += &format!("{i}\ta\t{w}\t{:.6}\t{:.6}\n", ka[j], spec(w)); }
                }
                fs::write(&path, d)?;
                println!("   links dumped to {path}");
            }
        }
    }
    if let Ok(path) = env::var("AVA_SAVE") {
        let t = Instant::now();
        let bytes = design_tests::store::save(&path, &b1, &s1.fac, &s1.ro)?;
        fit_links(&lessons.borrow(), &mut links.borrow_mut());
        save_more(&format!("{path}.more"), &s1.cont, &fam, &lessons.borrow(), &s1.rod, &links.borrow())?;
        fs::write(format!("{path}.ledger"), ledger.iter().map(|l| format!("{l}\n")).collect::<String>())?;
        let more = fs::metadata(format!("{path}.more")).map_or(0, |m| m.len());
        println!("   saved her brain to {path}: {:.2} GB + {:.2} GB sidecar in {:?}",
            bytes as f64 / 1e9, more as f64 / 1e9, t.elapsed());
    }

    // ---- ERROR-CORRECTING EPOCHS. The Hebbian pass above counted co-occurrence and
    // never asked whether the answer was right. These passes ask: predict, and only
    // when wrong, strengthen the true answer and weaken the one that won.
    for ep in 0..epochs {
        let (mut seen, mut ok) = (0usize, 0usize);
        for (i, (q, r)) in train.iter().enumerate() {
            if i % 5_000 == 0 && i > 0 && mem_gb().1 < MIN_FREE_GB {
                println!("   !! stopping epoch, low memory"); break;
            }
            reply::turns(r, ORDER, &mut buf);
            for t in &buf {
                let (id, g) = s1.feats(q, &t.ctx, t.pos, None, false);
                let code = s1.code(&id, &g);
                if s1.ro.learn_delta(&code, t.next, 1) { ok += 1; }
                seen += 1;
            }
        }
        println!("   epoch {}: {:.2}% of training positions already correct before update",
            ep + 1, ok as f64 / seen as f64 * 100.0);
    }

    // ---- STAGE 2: same machinery, plus stage 1's guess as an input.
    let mut s2 = Stage::new(pool, 500_000);
    let mut b2 = v4::Brain::new(s2.fac.n_total, Params::default());
    if two {
        for pass in 0..3 {
            for (i, (q, r)) in train.iter().enumerate() {
                if i % 5_000 == 0 && i > 0 && mem_gb().1 < MIN_FREE_GB {
                    println!("   !! stopping stage 2, low memory"); break;
                }
                reply::turns(r, ORDER, &mut buf);
                for t in &buf {
                    let guess = s1.ask(q, &t.ctx, t.pos, None, 1)
                        .into_iter().next().map(|(w, _)| w);
                    let g = guess.as_deref();
                    if pass < 2 { s2.settle(q, &t.ctx, t.pos, g); }
                    else { s2.write(&mut b2, q, &t.ctx, t.pos, g, t.next); }
                }
            }
        }
        println!("   stage 2 trained at {:?}, {:.2} GB", t0.elapsed(), mem_gb().0 - base);
    }

    // THE CONVERSATION VOTE (v29 WP2/WP3, AVA_WM / AVA_PREV): its links, learned from the stretch she
    // heard (see `Convo`). off = not built, and nothing below reads it.
    let convo: Option<Convo> = if wm_on || prev_on {
        let t = Instant::now();
        let decay = env::var("AVA_WM_DECAY").ok().and_then(|v| v.parse().ok()).unwrap_or(CONVO_DECAY);
        let keep = env::var("AVA_WM_KEEP").ok().and_then(|v| v.parse().ok()).unwrap_or(CONVO_KEEP);
        let gain = match env::var("AVA_WM_FORM").as_deref() {
            Err(_) | Ok("vote") => false,
            Ok("gain") => true,
            Ok(f) => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("AVA_WM_FORM must be vote or gain, got {f:?}"))),
        };
        let raise = match env::var("AVA_WM_LINK").as_deref() {
            Ok("pmi") => false,
            Err(_) | Ok("raise") => true,
            Ok(f) => return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("AVA_WM_LINK must be pmi or raise, got {f:?}"))),
        };
        let mut cv = Convo::build(train, train_prev, wm_on, prev_on, decay, keep, raise);
        cv.gain = gain;
        println!("   conversation vote (v29: {}{}): {} words, {} links kept (at most {keep} a word, {} links), fading x{decay} a word, {} form, learned in {:.1}s",
            if wm_on { "working memory" } else { "" }, if prev_on { if wm_on { " + the previous turn" } else { "the previous turn" } } else { "" },
            cv.words.len(), cv.links.iter().map(|l| l.len()).sum::<usize>(), if raise { "raise" } else { "pmi" },
            if gain { "gain" } else { "vote" }, t.elapsed().as_secs_f64());
        // what it does, on one line of conversation (as the prompt, or as the turn before an empty prompt)
        let ex = seq::tokenize("what time is it ?");
        let (q_ex, p_ex): (&[String], &[String]) = if wm_on { (&ex, &[]) } else { (&[], &ex) };
        println!("     e.g. after \"what time is it ?\" its vote opens with: {}", cv.vote(q_ex, &[], p_ex).iter().take(6)
            .map(|(w, x)| format!("{w} {x:.2}")).collect::<Vec<_>>().join(", "));
        Some(cv)
    } else { None };

    let answer = |q: &[String], ctx: &[&str], pos: usize, top: usize| -> Vec<(String, f32)> {
        let first = s1.ask(q, ctx, pos, None, 1).into_iter().next().map(|(w, _)| w);
        if !two { return s1.ask(q, ctx, pos, None, top); }
        let out = s2.ask(q, ctx, pos, first.as_deref(), top);
        if out.is_empty() { s1.ask(q, ctx, pos, None, top) } else { out }
    };

    // ---- how often does stage 2 CHANGE stage 1, and is it right to?
    let (mut n_all, mut t1, mut t5) = (0usize, 0usize, 0usize);
    let (mut changed, mut fixed, mut broke) = (0usize, 0usize, 0usize);
    let mut b3 = Vec::new();
    // IS REACHABILITY STILL THE BOTTLENECK? Every fix that ever worked in this project
    // made stored knowledge easier to REACH. So before adding another addressing
    // scheme, ask what fraction of held-out positions have no literal address at all,
    // and whether having one is what separates the right answers from the wrong ones.
    // rung[r] = positions where exactly r of {Q:, QS:, QW:} are interned; hit[r] = right.
    let (mut rung, mut hit) = ([0usize; 4], [0usize; 4]);
    // ---- COPY ORACLE. the 200 most frequent training tokens, ties broken by spelling
    // so the set is identical every run (HashMap order is randomised per process).
    let common: std::collections::HashSet<String> = {
        let mut f: std::collections::HashMap<&str, usize> = Default::default();
        for (q, r) in train { for w in q.iter().chain(r.iter()) { *f.entry(w.as_str()).or_default() += 1; } }
        let mut v: Vec<(&str, usize)> = f.into_iter().collect();
        v.sort_unstable_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(y.0)));
        v.into_iter().take(200).map(|(w, _)| w.to_string()).collect()
    };
    // slot: 0..4 = q's content words, last first; 4 = reply context's last content word.
    let (mut reach, mut reach_rare) = ([0usize; 5], [0usize; 5]);
    let (mut kappa, mut kappa_rare) = ([0usize; 4], [0usize; 4]);
    // ---- CLASS POOLER (probe bit 2). classes feed the READOUT only, never an address:
    // s1.cls stays None, so feats() and the rung histogram are untouched.
    let pcls: std::collections::HashMap<String, u16> = if probe & 2 != 0 {
        let mut docs: Vec<&[String]> = Vec::with_capacity(train.len() * 2);
        for (q, r) in train { docs.push(q.as_slice()); docs.push(r.as_slice()); }
        let (prof, freq, _) = design_tests::wordclass::profiles(docs.into_iter(), 2, 20);
        design_tests::wordclass::exemplar_classes(&prof, &freq, nclass).0
    } else { Default::default() };
    // strict = v16's definition, an unclassed word (EOS, rare words) can never win.
    // singleton = not pre-registered, a sensitivity check: an unclassed word competes
    // as a class of one, so the pooler is not handicapped on every end of sentence.
    let (mut pool_hit, mut pool_one) = ([0usize; 4], [0usize; 4]);
    // ---- SELF-MODEL (probe bit 4): (exchange, rung, margin, right) per position.
    let mut cal: Vec<(usize, u8, f32, bool)> = Vec::new();
    // ---- KNESER-NEY YARDSTICK (probe bit 8). the textbook count-based language model,
    // trained on exactly the same exchanges and graded on exactly the same positions.
    // [0] = reply only, [1] = prompt <sep> reply as one sequence. orders 3, 4, 5.
    let kns: Vec<Kn> = if probe & 8 != 0 {
        let t = Instant::now();
        let k = vec![Kn::build(train, false, 5), Kn::build(train, true, 5)];
        println!("   kneser-ney built at {:?}, {:.2} GB", t.elapsed(), mem_gb().0 - base);
        k
    } else { Vec::new() };
    let mut kp: Vec<f32> = Vec::new();
    let (mut kn1, mut kn5) = ([[[0usize; 4]; 3]; 2], [[0usize; 3]; 2]);
    let (mut np, mut nat, mut natr) = (Vec::<f32>::new(), [[0usize; 4]; 3], Vec::<[bool; 3]>::new());
    let mut knr: Vec<[bool; 2]> = Vec::new();
    let mut mixv: Vec<(Vec<(String, f32)>, Vec<(String, f32)>, String)> = Vec::new();   // router candidates: prompt+reply, orders 4 and 5
    // ---- READING HER SYNAPSES (v26, AVA_COMPLETE=<D>). every write lays Hebbian links among
    // the neurons of its code, and since v5 nothing read them: grammar neurons became hubs
    // with millions of partners and spiking recall drowned. two brain mechanisms bring them
    // back. SYNAPTIC PRUNING: each neuron keeps only its D strongest synapses, as development
    // does. PATTERN COMPLETION: the neurons a question activates wake the neurons they were
    // wired to; a neuron joins only if at least frac (her recall bar, 0.55) of the active
    // neurons reach it, at most as many as were active (k-winners). the woken neurons then
    // vote through the same readout. pre-registered in the v26 note.
    let complete_d: usize = env::var("AVA_COMPLETE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    if complete_d > 0 {
        let t = Instant::now();
        let before = b1.total_synapses();
        // AVA_COVARY=1: the COVARIANCE RULE (Sejnowski 1977; BCM). a synapse is worth keeping
        // only if its two neurons fired together MORE than chance: co-firings s(a,b) minus
        // f(a)·f(b)/N, where f counts the writes a neuron took part in. a hub fires with
        // everything, so its links sit near chance and are pruned first. 0 = keep the strongest.
        let covary = env::var("AVA_COVARY").map_or(false, |v| v == "1");
        let cut = if covary {
            let mut f = vec![0u32; b1.units.len()];
            for (&u, &t) in &s1.ro.total { if (u as usize) < f.len() { f[u as usize] = t; } }
            let nl = s1.ro.n_learned.max(1) as f64;
            let mut cut = 0u64;
            for (a, unit) in b1.units.iter_mut().enumerate() {
                if unit.out.is_empty() { continue; }
                unit.compact();
                let fa = f[a] as f64;
                let before = unit.out.len();
                let mut sc: Vec<(f64, (NeuronId, u16))> = unit.out.iter()
                    .map(|&(b, w)| (w as f64 - fa * f[b as usize] as f64 / nl, (b, w)))
                    .filter(|x| x.0 > 0.0).collect();
                sc.sort_unstable_by(|x, y| y.0.total_cmp(&x.0).then(x.1 .0.cmp(&y.1 .0)));
                sc.truncate(complete_d);
                let mut keep: Vec<(NeuronId, u16)> = sc.into_iter().map(|x| x.1).collect();
                keep.sort_unstable();
                keep.shrink_to_fit();
                unit.out = keep;
                cut += (before - unit.out.len()) as u64;
            }
            cut
        } else { b1.compact_pruned(complete_d) };
        println!("   synaptic pruning: {} -> {} synapses (each neuron keeps its {} {}), {:?}, {:.2} GB",
            before, before - cut, complete_d, if covary { "most above chance" } else { "strongest" },
            t.elapsed(), mem_gb().0 - base);
    }
    let frac = Params::default().frac;
    let complete = |code: &[NeuronId]| -> Vec<NeuronId> {
        let mut own: Vec<NeuronId> = code.to_vec();
        own.sort_unstable();
        own.dedup();
        let mut drive: std::collections::HashMap<NeuronId, (u32, u32)> = std::collections::HashMap::new();
        for &c in &own {
            for &(to, w) in &b1.units[c as usize].out {
                let e = drive.entry(to).or_default();
                e.0 += 1;
                e.1 += w as u32;
            }
        }
        let bar = ((own.len() as f32) * frac).ceil().max(1.0) as u32;
        let mut v: Vec<(NeuronId, u32, u32)> = drive.into_iter()
            .filter(|(x, d)| d.0 >= bar && own.binary_search(x).is_err())
            .map(|(x, d)| (x, d.0, d.1)).collect();
        v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(b.2.cmp(&a.2)).then(a.0.cmp(&b.0)));
        v.truncate(own.len());
        v.into_iter().map(|x| x.0).collect()
    };
    // per position: the completed neurons' own vote (normalised top 10), and how many woke
    let mut compv: Vec<(Vec<(String, f32)>, usize)> = Vec::new();
    // per position: the delayed synapses' vote from the moment before (empty at a reply's start)
    let mut delv: Vec<Vec<(String, f32)>> = Vec::new();
    let mut prev_code: Option<Vec<NeuronId>> = None;
    // per position: its domain, and whether ava's top-1 was right. measurement only.
    let (mut pdom, mut pok) = (Vec::<usize>::new(), Vec::<bool>::new());
    // TALK CACHE (v28, see Exam): only on a brain she woke from and did not sleep on in this run.
    // with a file made under the same key the loop below has nothing to ask, and what it would
    // have measured is read back right after it.
    let tc_path: Option<String> = match env::var("AVA_TALKCACHE") {
        Ok(p) if p.trim().is_empty() => { println!("   talk cache: not used (AVA_TALKCACHE names no file)"); None }
        Ok(_) if !woke => { println!("   talk cache: not used (she did not wake from a saved brain)"); None }
        Ok(_) if ledger.len() > ledger0 => { println!("   talk cache: not used (she slept on new lessons in this run)"); None }
        Ok(p) => Some(p),
        Err(_) => None,
    };
    let tc_key: Option<String> = tc_path.as_ref().map(|_| talk_key(&a, &env::var("AVA_LOAD").unwrap_or_default(), &corpus));
    let cached: Option<Exam> = tc_path.as_deref().zip(tc_key.as_deref()).and_then(|(p, k)| Exam::load(p, k));
    if let (Some(_), Some(p)) = (&cached, &tc_path) {
        println!("   talk cache: her exam read back from {p} (same program, brain, corpus, arguments and switches)");
    }
    let asked: &[(Vec<String>, Vec<String>)] = if cached.is_some() { &[] } else { test };
    for (ei, (q, rep)) in asked.iter().enumerate() {
        reply::turns(rep, ORDER, &mut b3);
        for t in &b3 {
            let one = s1.ask(q, &t.ctx, t.pos, None, 1).into_iter().next().map(|(w, _)| w);
            let got = answer(q, &t.ctx, t.pos, 5);
            let top = got.first().map(|(w, _)| w.clone());
            if top.as_deref() == Some(t.next) { t1 += 1; }
            pdom.push(test_dom[ei]);
            pok.push(top.as_deref() == Some(t.next));
            if got.iter().any(|(w, _)| w == t.next) { t5 += 1; }
            {
                let (id, gf) = s1.feats(q, &t.ctx, t.pos, None, true);
                if delay_on() {
                    let now = s1.code(&id, &gf);
                    let before = if t.pos > 0 { prev_code.take() } else { None };
                    let mut v = before.map_or(Vec::new(), |c| s1.rod.decode(&c, 10));
                    let tot: f32 = v.iter().map(|x| x.1.max(0.0)).sum();
                    for x in v.iter_mut() { x.1 = if tot > 0.0 { x.1.max(0.0) / tot } else { 0.0 }; }
                    delv.push(v);
                    prev_code = Some(now);
                }
                // count only the LITERAL rungs. class keys (`C...`) are interned for
                // nearly every position, so counting them would redefine the buckets
                // and make this incomparable to the runs it exists to be compared with.
                let r = id.iter().filter(|k| !k.starts_with('C'))
                    .filter(|k| s1.fac.id.feat_id.contains_key(k.as_str())).count();
                let r = r.min(3);
                rung[r] += 1;
                if top.as_deref() == Some(t.next) { hit[r] += 1; }
                if probe & 1 != 0 {
                    let qc: Vec<&str> = q.iter().rev().map(|s| s.as_str())
                        .filter(|w| reply::is_content(w)).take(4).collect();
                    let rc = t.ctx.iter().rev().find(|w| **w != BOS && reply::is_content(w)).copied();
                    let slot = qc.iter().position(|c| *c == t.next)
                        .or(if rc == Some(t.next) { Some(4) } else { None });
                    if let Some(j) = slot {
                        let rare = !common.contains(t.next);
                        reach[j] += 1;
                        if rare { reach_rare[j] += 1; }
                        if top.as_deref() != Some(t.next) {
                            kappa[r] += 1;
                            if rare { kappa_rare[r] += 1; }
                        }
                    }
                }
                if probe & 2 != 0 {
                    // S(class) = sum of its words' scores, over the FULL accumulator.
                    let all = s1.ask(q, &t.ctx, t.pos, None, usize::MAX);
                    let mut pool: std::collections::HashMap<u16, f32> = Default::default();
                    for (w, s) in &all {
                        if let Some(&c) = pcls.get(w) { *pool.entry(c).or_insert(0.0) += *s; }
                    }
                    let best = pool.iter().max_by(|a, b| a.1.total_cmp(b.1).then(b.0.cmp(a.0)))
                        .map(|(c, s)| (*c, *s));
                    let inclass = |c: u16| all.iter().find(|(w, _)| pcls.get(w) == Some(&c))
                        .map(|(w, _)| w.as_str());
                    let strict = best.and_then(|(c, _)| inclass(c));
                    let lone = all.iter().find(|(w, _)| !pcls.contains_key(w));
                    let single = match (best, lone) {
                        (Some((c, s)), Some((w, ls))) => if *ls > s { Some(w.as_str()) } else { inclass(c) },
                        (Some((c, _)), None) => inclass(c),
                        (None, Some((w, _))) => Some(w.as_str()),
                        (None, None) => None,
                    };
                    if strict == Some(t.next) { pool_hit[r] += 1; }
                    if single == Some(t.next) { pool_one[r] += 1; }
                }
                if probe & 8 != 0 {
                    let mut kr = [false; 2];
                    for (v, kn) in kns.iter().enumerate() {
                        let ctx = kn.ctx(q, v == 1, rep, t.pos);
                        for (oi, order) in [3usize, 4, 5].into_iter().enumerate() {
                            let got = kn.top(&ctx, order, 5, &mut kp);
                            let ok1 = got.first().map(|&w| kn.words[w as usize] == t.next) == Some(true);
                            if ok1 { kn1[v][oi][r] += 1; }
                            if got.iter().any(|&w| kn.words[w as usize] == t.next) { kn5[v][oi] += 1; }
                            if v == 1 && order >= 4 { kr[order - 4] = ok1; }
                        }
                    }
                    knr.push(kr);
                }
                if probe & 16 != 0 {
                    let mut nr = [false; 3];
                    for (vi, (c, d)) in [(false, false), (true, false), (true, true)].into_iter().enumerate() {
                        nr[vi] = s1.native(&gf, &t.ctx, c, d, &mut np).as_deref() == Some(t.next);
                        if nr[vi] { nat[vi][r] += 1; }
                    }
                    natr.push(nr);
                    // RELIABILITY-WEIGHTED VOTING (v24): keep both votes, each normalised over
                    // its own top 10. np still holds the 3-word fallback's distribution.
                    let norm = |v: Vec<(String, f32)>| -> Vec<(String, f32)> {
                        let t: f32 = v.iter().map(|x| x.1.max(0.0)).sum();
                        v.into_iter().map(|(w, x)| (w, if t > 0.0 { x.max(0.0) / t } else { 0.0 })).collect()
                    };
                    let av = norm(s1.ask(q, &t.ctx, t.pos, None, 10));
                    let nv = norm(topk_idx(&np, 10).into_iter().map(|(i, x)| (s1.ro.ops[i].clone(), x)).collect());
                    mixv.push((av, nv, t.next.to_string()));
                    if complete_d > 0 {
                        let (id, g) = s1.feats(q, &t.ctx, t.pos, None, true);
                        let woke = complete(&s1.code(&id, &g));
                        compv.push((norm(s1.ro.decode(&woke, 10)), woke.len()));
                    }
                }
                if probe & 28 != 0 {
                    // margin = s1/(s1+s2): the one signal v16 found NOT to be a re-encoding
                    // of rung count. computed from the top-5 the eval already asked for.
                    let m = match (got.first(), got.get(1)) {
                        (Some(a), Some(b)) if a.1 + b.1 > 0.0 => a.1 / (a.1 + b.1),
                        (Some(_), _) => 1.0,
                        _ => 0.0,
                    };
                    cal.push((ei, r as u8, m, top.as_deref() == Some(t.next)));
                }
            }
            if two && one != top {
                changed += 1;
                let was = one.as_deref() == Some(t.next);
                let now = top.as_deref() == Some(t.next);
                if !was && now { fixed += 1 }
                if was && !now { broke += 1 }
            }
            n_all += 1;
        }
    }
    // TALK CACHE: a hit hands back everything the loop measured; a miss keeps it for next time,
    // unless her files changed while she sat the exam
    let Exam { n_all, t1, t5, changed, fixed, broke, rung, hit, reach, reach_rare, kappa, kappa_rare,
               pool_hit, pool_one, kn1, kn5, nat, pdom, pok, cal, natr, knr, mixv, compv, delv } = match cached {
        Some(e) => e,
        None => {
            let mut e = Exam { n_all, t1, t5, changed, fixed, broke, rung, hit, reach, reach_rare, kappa, kappa_rare,
                               pool_hit, pool_one, kn1, kn5, nat, pdom, pok, cal, natr, knr, mixv, compv, delv };
            if let (Some(p), Some(k)) = (&tc_path, &tc_key) {
                if talk_key(&a, &env::var("AVA_LOAD").unwrap_or_default(), &corpus) != *k {
                    println!("   talk cache: not written (her files changed while she sat the exam)");
                } else {
                    match e.save(p, k) {
                        Ok(()) => println!("   talk cache: her exam kept in {p} for the next run"),
                        Err(err) => println!("   talk cache: not written to {p} ({err})"),
                    }
                }
            }
            e
        }
    };
    println!("\n   next word : top-1 {:.2}%   top-5 {:.2}%   ({n_all} positions)",
        t1 as f64 / n_all as f64 * 100.0, t5 as f64 / n_all as f64 * 100.0);
    println!("   by how many identity keys are interned:");
    for r in 0..4 {
        if rung[r] == 0 { continue }
        let lab = if r == 3 { "3+" } else { "  " };
        println!("     {r}{lab} interned : {:>5} positions ({:>5.1}%)   top-1 {:.2}%",
            rung[r], rung[r] as f64 / n_all as f64 * 100.0,
            hit[r] as f64 / rung[r] as f64 * 100.0);
    }
    // ---- PER DOMAIN, next word. the same positions and the same decisions as the global
    // numbers, only split by tag. native = the 3-word novelty-gated fallback; switch = v22's
    // ava/native choice per (rung x margin) cell, cross-fitted on exchange parity over the
    // WHOLE exam (exactly sw_b(2) below), sliced by domain afterwards; kn5 = kneser-ney
    // prompt+reply n=5. `None` = that probe bit is off.
    let nd = doms.len();
    let tally = |ok: &dyn Fn(usize) -> bool| -> Vec<usize> {
        let mut v = vec![0usize; nd];
        for i in 0..pdom.len() { if ok(i) { v[pdom[i]] += 1; } }
        v
    };
    let d_pos = tally(&|_| true);
    let d_ava = tally(&|i| pok[i]);
    let has_nat = probe & 16 != 0 && natr.len() == pdom.len() && cal.len() == pdom.len() && !cal.is_empty();
    let sw3: Vec<bool> = if has_nat {
        let mut ms: Vec<f32> = cal.iter().map(|c| c.2).collect();
        ms.sort_by(|a, b| a.total_cmp(b));
        let edges: Vec<f32> = (1..10).map(|k| ms[ms.len() * k / 10]).collect();
        let cell = |c: &(usize, u8, f32, bool)| c.1 as usize * 10 + edges.iter().filter(|e| c.2 >= **e).count();
        let mut tab = [[[0usize; 2]; 40]; 2];
        for (c, n) in cal.iter().zip(&natr) {
            let t = &mut tab[c.0 & 1][cell(c)];
            if c.3 { t[0] += 1; }
            if n[2] { t[1] += 1; }
        }
        cal.iter().zip(&natr).map(|(c, n)| {
            let o = tab[1 - (c.0 & 1)][cell(c)];
            if o[1] > o[0] { n[2] } else { c.3 }
        }).collect()
    } else { Vec::new() };
    let d_nat = has_nat.then(|| tally(&|i| natr[i][2]));
    let d_sw = has_nat.then(|| tally(&|i| sw3[i]));
    let d_kn = (probe & 8 != 0 && knr.len() == pdom.len()).then(|| tally(&|i| knr[i][1]));
    // a hit count as a percentage of domain d's positions (None = all domains), or "-".
    let dpc = |h: Option<&Vec<usize>>, d: Option<usize>| -> String {
        let (x, of) = match d {
            Some(d) => (h.map(|v| v[d]), d_pos[d]),
            None => (h.map(|v| v.iter().sum()), d_pos.iter().sum()),
        };
        match x { Some(x) if of > 0 => format!("{:.2}", x as f64 / of as f64 * 100.0), _ => "-".into() }
    };
    if multi {
        println!("\n   PER DOMAIN, next word  (top-1 on the positions above, split by domain tag)");
        println!("     native = B3>B2>B1>uni novelty-gated; switch = ava/native per cell, cross-fitted; kn5 = kn prompt+reply n=5");
        println!("     {:<16} {:>9} {:>8} {:>8} {:>8} {:>8}", "domain", "positions", "ava", "native", "switch", "kn5");
        let pcs = |s: String| if s == "-" { s } else { s + "%" };
        for d in (0..nd).map(Some).chain([None]) {
            println!("     {:<16} {:>9} {:>8} {:>8} {:>8} {:>8}", d.map_or("(all)", |d| doms[d].as_str()),
                d.map_or(pdom.len(), |d| d_pos[d]), pcs(dpc(Some(&d_ava), d)),
                pcs(dpc(d_nat.as_ref(), d)), pcs(dpc(d_sw.as_ref(), d)), pcs(dpc(d_kn.as_ref(), d)));
        }
    }
    // per held-out exchange, filled by the reply exam (probe bit 32): (word-F1, proper,
    // exact) for ava greedy, switch greedy and repaired.
    let mut rex: Vec<[(f64, bool, bool); 3]> = Vec::new();
    if probe & 1 != 0 {
        let pc = |x: usize| x as f64 / n_all as f64 * 100.0;
        let (k, kr): (usize, usize) = (kappa.iter().sum(), kappa_rare.iter().sum());
        println!("\n   COPY ORACLE  (share of all {n_all} positions; 'rare' = gold outside the top-200)");
        let lab = ["q0 (last)", "q1", "q2", "q3", "reply ctx"];
        for j in 0..5 {
            println!("     gold = {:<10}: {:>5.2}%   rare {:>5.2}%", lab[j], pc(reach[j]), pc(reach_rare[j]));
        }
        println!("     copy-reachable  : {:>5.2}%   rare {:>5.2}%",
            pc(reach.iter().sum()), pc(reach_rare.iter().sum()));
        println!("   KAPPA = reachable AND top-1 wrong : {:.2}%   rare {:.2}%", pc(k), pc(kr));
        for r in 0..4 {
            if rung[r] == 0 { continue }
            println!("     {r}{} interned : kappa {:>5.2}% of bucket   rare {:>5.2}%",
                if r == 3 { "+" } else { " " },
                kappa[r] as f64 / rung[r] as f64 * 100.0, kappa_rare[r] as f64 / rung[r] as f64 * 100.0);
        }
        let verdict = |x: f64| if x < 3.0 { "ABANDON the copy readout" }
            else if x < 5.0 { "build copy-only at 15k" } else { "build it, take it to 30k" };
        println!("   pre-registered verdict: raw -> {}   |   rare -> {}", verdict(pc(k)), verdict(pc(kr)));
        println!("TSV\tkappa\t{}\t{:.2}\t{:.2}", train.len(), pc(k), pc(kr));
    }
    if probe & 2 != 0 {
        let pc = |x: usize| x as f64 / n_all as f64 * 100.0;
        println!("\n   CLASS POOLER  ({} words in {nclass} classes; readout only, never an address)",
            pcls.len());
        println!("     all positions : strict {:>5.2}%   singleton {:>5.2}%   vs baseline {:>5.2}%",
            pc(pool_hit.iter().sum()), pc(pool_one.iter().sum()), pc(t1));
        for r in 0..4 {
            if rung[r] == 0 { continue }
            let f = |x: usize| x as f64 / rung[r] as f64 * 100.0;
            println!("     {r}{} interned : strict {:>5.2}%   singleton {:>5.2}%   vs baseline {:>5.2}%",
                if r == 3 { "+" } else { " " }, f(pool_hit[r]), f(pool_one[r]), f(hit[r]));
        }
        // pre-registered in v16, on the 0-rung bucket, strict pooler, against 22.45%.
        let z = pool_hit[0] as f64 / rung[0].max(1) as f64 * 100.0;
        println!("   pre-registered verdict (0-rung, strict = {z:.2}%): {}",
            if (21.9..=23.1).contains(&z) { "class lever CLOSED (inside the noise band)" }
            else if z < 21.9 { "class lever CLOSED (worse than the vote it re-pools)" }
            else if z >= 24.0 { "real gain, but a class-marginal smoother, not composition" }
            else { "between 23.1 and 24.0: ambiguous, needs the 30k run" });
        println!("TSV\tpool\t{}\t{:.2}\t{:.2}\t{z:.2}", train.len(),
            pc(pool_hit.iter().sum()), pc(pool_one.iter().sum()));
    }
    if probe & 4 != 0 && !cal.is_empty() {
        // three curves over the same positions (v16 §2): (i) raw margin, the honest
        // null; (ii) rung count; (iii) a plain counting table over rung x margin decile,
        // CROSS-FITTED by exchange parity so no position is scored by a table it filled.
        // that table is the bar any neuron-based self-model has to beat.
        let mut ms: Vec<f32> = cal.iter().map(|c| c.2).collect();
        ms.sort_by(|a, b| a.total_cmp(b));
        let edges: Vec<f32> = (1..10).map(|k| ms[ms.len() * k / 10]).collect();
        let dec = |m: f32| edges.iter().filter(|e| m >= **e).count();
        let mut tab = [[[0usize; 2]; 40]; 2];
        for c in &cal {
            let cell = &mut tab[c.0 & 1][c.1 as usize * 10 + dec(c.2)];
            cell[1] += 1;
            if c.3 { cell[0] += 1; }
        }
        let base: [f64; 2] = [0, 1].map(|f| {
            let (h, n) = tab[f].iter().fold((0, 0), |a, c| (a.0 + c[0], a.1 + c[1]));
            h as f64 / n.max(1) as f64
        });
        let p_tab: Vec<f64> = cal.iter().map(|c| {
            let o = 1 - (c.0 & 1);
            let [h, n] = tab[o][c.1 as usize * 10 + dec(c.2)];
            if n == 0 { base[o] } else { (h as f64 + 1.0) / (n as f64 + 2.0) }
        }).collect();
        let sig: [(&str, Vec<f64>); 3] = [
            ("margin s1/(s1+s2)", cal.iter().map(|c| c.2 as f64).collect()),
            ("rung count", cal.iter().map(|c| c.1 as f64).collect()),
            ("table rung x margin", p_tab.clone()),
        ];
        let right = cal.iter().filter(|c| c.3).count();
        println!("\n   SELF-MODEL  (does she know when her top-1 is right? base rate {:.2}%)",
            right as f64 / cal.len() as f64 * 100.0);
        println!("     {:<20} {:>6}  {:<16} {:>8} {:>8} {:>8}", "signal", "AUC",
            "95% CI (by exch)", "acc@25%", "acc@50%", "acc@75%");
        let mut tsv = Vec::new();
        for (name, s) in &sig {
            let xs: Vec<(f64, bool)> = s.iter().zip(&cal).map(|(v, c)| (*v, c.3)).collect();
            let ex: Vec<(usize, f64, bool)> = s.iter().zip(&cal).map(|(v, c)| (c.0, *v, c.3)).collect();
            let (lo, hi) = boot(&ex, test.len());
            let a = auc(&xs);
            println!("     {:<20} {:>6.3}  [{:.3}, {:.3}]   {:>7.2}% {:>7.2}% {:>7.2}%", name, a, lo, hi,
                cover(&xs, 0.25), cover(&xs, 0.50), cover(&xs, 0.75));
            tsv.push(format!("{a:.3}"));
        }
        // does margin carry anything rung count does not? AUC of margin INSIDE each bucket.
        let within: Vec<String> = (0..4u8).map(|r| {
            let xs: Vec<(f64, bool)> = cal.iter().filter(|c| c.1 == r).map(|c| (c.2 as f64, c.3)).collect();
            format!("{r}: {:.3}", auc(&xs))
        }).collect();
        println!("     margin AUC within each rung bucket   {}", within.join("   "));
        let xs: Vec<(f64, bool)> = p_tab.iter().zip(&cal).map(|(p, c)| (*p, c.3)).collect();
        println!("     table calibration error (ECE, 10 bins): {:.2} points", ece(&xs));
        println!("TSV\tselfmodel\t{}\t{}\t{:.2}", train.len(), tsv.join("\t"), ece(&xs));
    }
    if probe & 16 != 0 && natr.len() == cal.len() && !cal.is_empty() {
        let pc = |x: usize| x as f64 / n_all as f64 * 100.0;
        let pb = |x: usize, r: usize| x as f64 / rung[r].max(1) as f64 * 100.0;
        println!("\n   NATIVE BACKOFF  (kn's two ideas inside her own units; 0-rung = no identity key)");
        println!("     {:<30} {:>7}   {:>7} {:>7} {:>7} {:>7}", "decode", "all", "0-rung", "1", "2", "3+");
        let row = |name: &str, h: &[usize; 4]| println!("     {:<30} {:>6.2}%   {:>6.2}% {:>6.2}% {:>6.2}% {:>6.2}%",
            name, pc(h.iter().sum()), pb(h[0], 0), pb(h[1], 1), pb(h[2], 2), pb(h[3], 3));
        row("ava (sum of every unit)", &hit);
        row("B2>B1>uni, raw votes", &nat[0]);
        row("B2>B1>uni, novelty-gated", &nat[1]);
        row("B3>B2>B1>uni, novelty-gated", &nat[2]);
        // switch A, PRE-DECLARED and fitted on nothing: the native decode wherever no
        // identity key exists, ava's own vote everywhere else.
        let sw_a = |j: usize| cal.iter().zip(&natr).filter(|(c, n)| if c.1 == 0 { n[j] } else { c.3 }).count();
        let (a, a4) = (sw_a(1), sw_a(2));
        // switch B: per (rung x margin) cell, cross-fitted on the other half, {ava, native}.
        let mut ms: Vec<f32> = cal.iter().map(|c| c.2).collect();
        ms.sort_by(|a, b| a.total_cmp(b));
        let edges: Vec<f32> = (1..10).map(|k| ms[ms.len() * k / 10]).collect();
        let cell = |c: &(usize, u8, f32, bool)| c.1 as usize * 10 + edges.iter().filter(|e| c.2 >= **e).count();
        let sw_b = |j: usize| -> usize {
            let mut tab = [[[0usize; 2]; 40]; 2];
            for (c, n) in cal.iter().zip(&natr) {
                let t = &mut tab[c.0 & 1][cell(c)];
                if c.3 { t[0] += 1; }
                if n[j] { t[1] += 1; }
            }
            cal.iter().zip(&natr).filter(|(c, n)| {
                let o = tab[1 - (c.0 & 1)][cell(c)];
                if o[1] > o[0] { n[j] } else { c.3 }
            }).count()
        };
        let (b, b4) = (sw_b(1), sw_b(2));
        println!("   ava + native where rung = 0 (fitted on nothing) : top-1 {:.2}%  ({:+.2})", pc(a), pc(a) - pc(t1));
        println!("   ava / native routed per cell (cross-fitted)     : top-1 {:.2}%  ({:+.2})", pc(b), pc(b) - pc(t1));
        println!("   with the 3-word level:  rung = 0 {:.2}% ({:+.2})   per cell {:.2}% ({:+.2})",
            pc(a4), pc(a4) - pc(t1), pc(b4), pc(b4) - pc(t1));
        println!("TSV\tnative\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}", train.len(),
            pb(nat[0][0], 0), pb(nat[1][0], 0), pc(nat[0].iter().sum()), pc(nat[1].iter().sum()), pc(a), pc(b));
        println!("TSV\tnative3\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.2}", train.len(),
            pb(nat[2][0], 0), pc(nat[2].iter().sum()), pc(a4), pc(b4));

        // ---- RELIABILITY-WEIGHTED VOTING (v24). the brain weighs each cue by how reliable it
        // is (Ernst & Banks 2002) instead of switching one off. here both votes stay on:
        //   score(w) = a * ava(w) + (1 - a) * fallback(w),  a in {0, 0.1, ..., 1}
        // with `a` chosen per (rung x margin) cell on the OTHER exam half (cross-fitted, like
        // the switch). a = 1 or 0 IS the switch, so this can only add the in-between.
        // PRE-REGISTERED: better only if it beats the switch AND the exchange-bootstrap 95%
        // interval of the difference lies above zero.
        // the delayed synapses' weight per rung when she SPEAKS: the b this exam liked best on
        // ALL its positions (as the switch's own cells are), set in the verdict below.
        let mut del_b = [0f32; 4];
        // v29: the conversation vote's weight per rung when she speaks, and its per-position line
        let mut ct_b = [0f32; 4];
        let mut ct_line: Option<Vec<bool>> = None;
        if mixv.len() == cal.len() {
            let pick = |m: &(Vec<(String, f32)>, Vec<(String, f32)>, String), a: f32| -> bool {
                let mut best: (&str, f32) = ("", -1.0);
                for (w, x) in &m.0 {
                    let y = a * x + (1.0 - a) * m.1.iter().find(|z| &z.0 == w).map_or(0.0, |z| z.1);
                    if y > best.1 { best = (w.as_str(), y); }
                }
                for (w, x) in &m.1 {
                    if m.0.iter().any(|z| &z.0 == w) { continue; }
                    let y = (1.0 - a) * x;
                    if y > best.1 { best = (w.as_str(), y); }
                }
                best.0 == m.2
            };
            let mut acc = vec![[[0usize; 11]; 40]; 2];
            let mut hard = [[[0usize; 2]; 40]; 2];
            for ((c, n), m) in cal.iter().zip(&natr).zip(&mixv) {
                let (h, ce) = (c.0 & 1, cell(c));
                for g in 0..=10 { if pick(m, g as f32 / 10.0) { acc[h][ce][g] += 1; } }
                if c.3 { hard[h][ce][0] += 1; }
                if n[2] { hard[h][ce][1] += 1; }
            }
            // per cell, the weight the other half liked best; ties go to the switch's own choice
            let choose = |h: usize, ce: usize| -> usize {
                let o = &acc[1 - h][ce];
                let sw = if hard[1 - h][ce][1] > hard[1 - h][ce][0] { 0 } else { 10 };
                let top = *o.iter().max().unwrap();
                if o[sw] == top { sw } else { (0..=10).filter(|&g| o[g] == top).min_by_key(|&g| (g as i32 - sw as i32).abs()).unwrap() }
            };
            let n_ex = cal.iter().map(|c| c.0).max().map_or(0, |m| m + 1);
            let mut per_ex = vec![[0i64; 3]; n_ex];   // soft hits, switch hits, positions
            let mut wsum = 0f64;
            for ((c, n), m) in cal.iter().zip(&natr).zip(&mixv) {
                let (h, ce) = (c.0 & 1, cell(c));
                let g = choose(h, ce);
                wsum += g as f64 / 10.0;
                let o = hard[1 - h][ce];
                let sw_ok = if o[1] > o[0] { n[2] } else { c.3 };
                per_ex[c.0][0] += pick(m, g as f32 / 10.0) as i64;
                per_ex[c.0][1] += sw_ok as i64;
                per_ex[c.0][2] += 1;
            }
            let tot = |k: usize| per_ex.iter().map(|e| e[k]).sum::<i64>() as f64;
            let (soft, swi, np_) = (tot(0) / tot(2) * 100.0, tot(1) / tot(2) * 100.0, tot(2));
            let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
            let mut ds: Vec<f64> = (0..1000).map(|_| {
                let (mut d, mut m) = (0i64, 0i64);
                for _ in 0..n_ex {
                    seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                    let e = per_ex[(seed % n_ex as u64) as usize];
                    d += e[0] - e[1]; m += e[2];
                }
                d as f64 / m.max(1) as f64 * 100.0
            }).collect();
            ds.sort_by(|a, b| a.total_cmp(b));
            let (lo, hi) = (ds[25], ds[974]);
            println!("   reliability-weighted vote (cross-fitted)       : top-1 {:.2}%  vs switch {:.2}%  diff {:+.2} \
(95% CI {:+.2} .. {:+.2}), mean weight on memory {:.2} -> {}", soft, swi, soft - swi, lo, hi, wsum / np_,
                if soft > swi && lo > 0.0 { "BETTER" } else { "not better" });
            println!("TSV\tsoftmix\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.2}", train.len(), soft, swi, lo, hi);

            // ---- SEMANTIC PRIMING (v26, AVA_PRIME=1). hearing a word pre-activates the words that
            // tend to come up in answers to it (spreading activation; Meyer & Schvaneveldt 1971).
            // a Hebbian association between prompt words and reply words over the exchanges she
            // heard (built here from exactly those exchanges, never the exam):
            //   assoc(u, w) = max(0, ln(c(u,w)·N / (n_prompt(u)·n_reply(w)))) · c/(c+1)
            //   prime(w | prompt) = mean over the prompt's distinct words u of assoc(u, w)
            // priming raises the gain of words already active: the switch's chosen vote (her
            // memory or her fallback, per cell as before) is re-ranked over its top 10 by
            // vote(w) · exp(g · prime(w)), with g picked on the OTHER exam half (cross-fitted).
            // PRE-REGISTERED: better only if it beats the switch AND the exchange-bootstrap 95%
            // interval of the difference lies above zero.
            if env::var("AVA_PRIME").map_or(false, |v| v == "1") {
                let t0 = std::time::Instant::now();
                let mut ids: std::collections::HashMap<&str, u32> = std::collections::HashMap::new();
                let mut np_: Vec<u32> = Vec::new();
                let mut nr_: Vec<u32> = Vec::new();
                let mut co: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
                fn uniq<'a>(ws: &'a [String], ids: &mut std::collections::HashMap<&'a str, u32>,
                            np_: &mut Vec<u32>, nr_: &mut Vec<u32>) -> Vec<u32> {
                    let mut v: Vec<u32> = ws.iter().map(|w| {
                        let n = ids.len() as u32;
                        let id = *ids.entry(w.as_str()).or_insert(n);
                        if id as usize == np_.len() { np_.push(0); nr_.push(0); }
                        id
                    }).collect();
                    v.sort_unstable();
                    v.dedup();
                    v
                }
                for (q, r) in train.iter() {
                    let us = uniq(q, &mut ids, &mut np_, &mut nr_);
                    let ws = uniq(r, &mut ids, &mut np_, &mut nr_);
                    for &u in &us { np_[u as usize] += 1; }
                    for &w in &ws { nr_[w as usize] += 1; }
                    for &u in &us { for &w in &ws { *co.entry((u as u64) << 32 | w as u64).or_default() += 1; } }
                }
                let n_tr = train.len() as f32;
                let prime = |q: &[String], w: &str| -> f32 {
                    let Some(&wi) = ids.get(w) else { return 0.0 };
                    let mut us: Vec<u32> = q.iter().filter_map(|u| ids.get(u.as_str()).copied()).collect();
                    us.sort_unstable();
                    us.dedup();
                    if us.is_empty() { return 0.0; }
                    let s: f32 = us.iter().map(|&u| {
                        let c = co.get(&((u as u64) << 32 | wi as u64)).copied().unwrap_or(0) as f32;
                        if c == 0.0 { return 0.0; }
                        let pmi = (c * n_tr / (np_[u as usize] as f32 * nr_[wi as usize] as f32)).ln();
                        pmi.max(0.0) * c / (c + 1.0)
                    }).sum();
                    s / us.len() as f32
                };
                const G: [f32; 10] = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0];
                // per position: the switch's source (other-half cell table, as above), and for
                // each g whether the primed re-rank of that source's top 10 is right
                let n_ex = cal.iter().map(|c| c.0).max().map_or(0, |m| m + 1);
                let mut hits = [[0usize; 10]; 2];   // [half][g]
                let mut rows: Vec<(usize, u8, [bool; 10], bool)> = Vec::with_capacity(cal.len());
                for ((c, n), m) in cal.iter().zip(&natr).zip(&mixv) {
                    let (h, ce) = (c.0 & 1, cell(c));
                    let o = hard[1 - h][ce];
                    let use_nat = o[1] > o[0];
                    let sw_ok = if use_nat { n[2] } else { c.3 };
                    let src = if use_nat { &m.1 } else { &m.0 };
                    let q = &test[c.0].0;
                    let pr: Vec<f32> = src.iter().map(|(w, _)| prime(q, w)).collect();
                    let mut ok = [false; 10];
                    for (gi, &g) in G.iter().enumerate() {
                        let best = src.iter().zip(&pr).map(|((w, x), p)| (w, x * (g * p).exp()))
                            .fold(("", -1.0f32), |b, (w, y)| if y > b.1 { (w.as_str(), y) } else { b });
                        // g = 0 is the switch itself (an empty top 10 falls back to its answer)
                        ok[gi] = if gi == 0 || src.is_empty() { sw_ok } else { best.0 == m.2 };
                        if ok[gi] { hits[h][gi] += 1; }
                    }
                    rows.push((c.0, c.1, ok, sw_ok));
                }
                let pick_g = |h: usize| -> usize {
                    let o = &hits[1 - h];
                    (0..10).max_by_key(|&g| (o[g], std::cmp::Reverse(g))).unwrap()
                };
                let mut per_ex = vec![[0i64; 3]; n_ex];
                let mut by_rung = [[0usize; 3]; 4];   // primed hits, switch hits, positions
                for (ei, r, ok, sw_ok) in &rows {
                    let g = pick_g(ei & 1);
                    per_ex[*ei][0] += ok[g] as i64;
                    per_ex[*ei][1] += *sw_ok as i64;
                    per_ex[*ei][2] += 1;
                    let rr = (*r as usize).min(3);
                    by_rung[rr][0] += ok[g] as usize;
                    by_rung[rr][1] += *sw_ok as usize;
                    by_rung[rr][2] += 1;
                }
                let tot = |k: usize| per_ex.iter().map(|e| e[k]).sum::<i64>() as f64;
                let (pri, swi, np2) = (tot(0) / tot(2) * 100.0, tot(1) / tot(2) * 100.0, tot(2));
                let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
                let mut ds: Vec<f64> = (0..1000).map(|_| {
                    let (mut d, mut m) = (0i64, 0i64);
                    for _ in 0..n_ex {
                        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                        let e = per_ex[(seed % n_ex as u64) as usize];
                        d += e[0] - e[1]; m += e[2];
                    }
                    d as f64 / m.max(1) as f64 * 100.0
                }).collect();
                ds.sort_by(|a, b| a.total_cmp(b));
                let (lo, hi) = (ds[25], ds[974]);
                println!("   semantic priming (cross-fitted)                : top-1 {:.2}%  vs switch {:.2}%  diff {:+.2} \
(95% CI {:+.2} .. {:+.2}), gain g = {} / {} (halves) -> {}", pri, swi, pri - swi, lo, hi,
                    G[pick_g(1)], G[pick_g(0)], if pri > swi && lo > 0.0 { "BETTER" } else { "not better" });
                for (r, b) in by_rung.iter().enumerate() {
                    if b[2] == 0 { continue; }
                    println!("     {}{} interned: {:>5} positions  switch {:.2}%  primed {:.2}%", r, if r == 3 { "+" } else { " " },
                        b[2], b[1] as f64 / b[2] as f64 * 100.0, b[0] as f64 / b[2] as f64 * 100.0);
                }
                println!("     ({} words, {} associations, built in {:.1}s; {} positions)", ids.len(), co.len(),
                    t0.elapsed().as_secs_f64(), np2);
                println!("TSV\tprime\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.2}", train.len(), pri, swi, lo, hi);
            }

            // ---- PATTERN COMPLETION, the verdict. where the switch uses her memory, the woken
            // neurons add their vote: score(w) = memory(w) + b · woken(w), both normalised over
            // their top 10, b per rung chosen on the OTHER exam half. where the switch uses her
            // fallback, nothing changes. PRE-REGISTERED: better only if it beats the switch AND
            // the exchange-bootstrap 95% interval of the difference lies above zero.
            if complete_d > 0 && compv.len() == cal.len() {
                const B: [f32; 8] = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 1000.0];
                let mut hits = [[[0usize; 8]; 4]; 2];   // [half][rung][b]
                let mut rows: Vec<(usize, usize, [bool; 8], bool)> = Vec::with_capacity(cal.len());
                let (mut woke_n, mut woke_any) = (0usize, 0usize);
                let (mut alone_ok, mut mem_ok, mut n_mem) = (0usize, 0usize, 0usize);
                for (((c, n), m), cv) in cal.iter().zip(&natr).zip(&mixv).zip(&compv) {
                    let (h, ce, rr) = (c.0 & 1, cell(c), (c.1 as usize).min(3));
                    let o = hard[1 - h][ce];
                    let use_nat = o[1] > o[0];
                    let sw_ok = if use_nat { n[2] } else { c.3 };
                    woke_n += cv.1;
                    woke_any += (cv.1 > 0) as usize;
                    let mut ok = [sw_ok; 8];
                    if !use_nat && !cv.0.is_empty() {
                        alone_ok += (cv.0[0].0 == m.2) as usize;
                        mem_ok += c.3 as usize;
                        n_mem += 1;
                        for (bi, &b) in B.iter().enumerate().skip(1) {
                            let mut best: (&str, f32) = ("", -1.0);
                            for (w, x) in &m.0 {
                                let y = x + b * cv.0.iter().find(|z| &z.0 == w).map_or(0.0, |z| z.1);
                                if y > best.1 { best = (w.as_str(), y); }
                            }
                            for (w, x) in &cv.0 {
                                if m.0.iter().any(|z| &z.0 == w) { continue; }
                                let y = b * x;
                                if y > best.1 { best = (w.as_str(), y); }
                            }
                            ok[bi] = if m.0.is_empty() && best.1 <= 0.0 { sw_ok } else { best.0 == m.2 };
                        }
                    }
                    for bi in 0..8 { if ok[bi] { hits[h][rr][bi] += 1; } }
                    rows.push((c.0, rr, ok, sw_ok));
                }
                let pick_b = |h: usize, rr: usize| -> usize {
                    let o = &hits[1 - h][rr];
                    (0..8).max_by_key(|&b| (o[b], std::cmp::Reverse(b))).unwrap()
                };
                let n_ex = cal.iter().map(|c| c.0).max().map_or(0, |m| m + 1);
                let mut per_ex = vec![[0i64; 3]; n_ex];
                let mut by_rung = [[0usize; 3]; 4];
                for (ei, rr, ok, sw_ok) in &rows {
                    let b = pick_b(ei & 1, *rr);
                    per_ex[*ei][0] += ok[b] as i64;
                    per_ex[*ei][1] += *sw_ok as i64;
                    per_ex[*ei][2] += 1;
                    by_rung[*rr][0] += ok[b] as usize;
                    by_rung[*rr][1] += *sw_ok as usize;
                    by_rung[*rr][2] += 1;
                }
                let tot = |k: usize| per_ex.iter().map(|e| e[k]).sum::<i64>() as f64;
                let (cpl, swi) = (tot(0) / tot(2) * 100.0, tot(1) / tot(2) * 100.0);
                let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
                let mut ds: Vec<f64> = (0..1000).map(|_| {
                    let (mut d, mut m) = (0i64, 0i64);
                    for _ in 0..n_ex {
                        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                        let e = per_ex[(seed % n_ex as u64) as usize];
                        d += e[0] - e[1]; m += e[2];
                    }
                    d as f64 / m.max(1) as f64 * 100.0
                }).collect();
                ds.sort_by(|a, b| a.total_cmp(b));
                let (lo, hi) = (ds[25], ds[974]);
                println!("   pattern completion through her synapses (cross-fitted): top-1 {:.2}%  vs switch {:.2}%  diff {:+.2} \
(95% CI {:+.2} .. {:+.2}) -> {}", cpl, swi, cpl - swi, lo, hi, if cpl > swi && lo > 0.0 { "BETTER" } else { "not better" });
                println!("     neurons woken per position: {:.1} on average; any woken at {:.1}% of positions",
                    woke_n as f64 / rows.len().max(1) as f64, woke_any as f64 / rows.len().max(1) as f64 * 100.0);
                println!("     where the switch uses her memory and neurons woke ({} positions): memory alone {:.2}%, woken neurons alone {:.2}%",
                    n_mem, mem_ok as f64 / n_mem.max(1) as f64 * 100.0, alone_ok as f64 / n_mem.max(1) as f64 * 100.0);
                for (r, b) in by_rung.iter().enumerate() {
                    if b[2] == 0 { continue; }
                    println!("     {}{} interned: {:>5} positions  switch {:.2}%  completed {:.2}%   b = {} / {}", r, if r == 3 { "+" } else { " " },
                        b[2], b[1] as f64 / b[2] as f64 * 100.0, b[0] as f64 / b[2] as f64 * 100.0, B[pick_b(1, r)], B[pick_b(0, r)]);
                }
                println!("TSV\tcomplete\t{}\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.2}", train.len(), complete_d, cpl, swi, lo, hi);
            }

            // ---- DELAYED SYNAPSES, the verdict. whichever vote the switch uses (memory or
            // fallback), the delayed synapses add theirs: score(w) = vote(w) + b · delayed(w),
            // both normalised over their top 10, b per rung chosen on the OTHER exam half.
            // PRE-REGISTERED: better only if it beats the switch AND the exchange-bootstrap 95%
            // interval of the difference lies above zero.
            if delay_on() && delv.len() == cal.len() {
                const B: [f32; 8] = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 1000.0];
                let mut hits = [[[0usize; 8]; 4]; 2];
                let mut rows: Vec<(usize, usize, [bool; 8], bool)> = Vec::with_capacity(cal.len());
                let mut alone = [[0usize; 2]; 4];   // delayed vote alone right, positions with one
                for (((c, n), m), dv) in cal.iter().zip(&natr).zip(&mixv).zip(&delv) {
                    let (h, ce, rr) = (c.0 & 1, cell(c), (c.1 as usize).min(3));
                    let o = hard[1 - h][ce];
                    let use_nat = o[1] > o[0];
                    let sw_ok = if use_nat { n[2] } else { c.3 };
                    let src = if use_nat { &m.1 } else { &m.0 };
                    let mut ok = [sw_ok; 8];
                    if !dv.is_empty() {
                        alone[rr][0] += (dv[0].0 == m.2) as usize;
                        alone[rr][1] += 1;
                        for (bi, &b) in B.iter().enumerate().skip(1) {
                            let mut best: (&str, f32) = ("", -1.0);
                            for (w, x) in src {
                                let y = x + b * dv.iter().find(|z| &z.0 == w).map_or(0.0, |z| z.1);
                                if y > best.1 { best = (w.as_str(), y); }
                            }
                            for (w, x) in dv {
                                if src.iter().any(|z| &z.0 == w) { continue; }
                                let y = b * x;
                                if y > best.1 { best = (w.as_str(), y); }
                            }
                            ok[bi] = if src.is_empty() && best.1 <= 0.0 { sw_ok } else { best.0 == m.2 };
                        }
                    }
                    for bi in 0..8 { if ok[bi] { hits[h][rr][bi] += 1; } }
                    rows.push((c.0, rr, ok, sw_ok));
                }
                let pick_b = |h: usize, rr: usize| -> usize {
                    let o = &hits[1 - h][rr];
                    (0..8).max_by_key(|&b| (o[b], std::cmp::Reverse(b))).unwrap()
                };
                let n_ex = cal.iter().map(|c| c.0).max().map_or(0, |m| m + 1);
                let mut per_ex = vec![[0i64; 3]; n_ex];
                let mut by_rung = [[0usize; 3]; 4];
                for (ei, rr, ok, sw_ok) in &rows {
                    let b = pick_b(ei & 1, *rr);
                    per_ex[*ei][0] += ok[b] as i64;
                    per_ex[*ei][1] += *sw_ok as i64;
                    per_ex[*ei][2] += 1;
                    by_rung[*rr][0] += ok[b] as usize;
                    by_rung[*rr][1] += *sw_ok as usize;
                    by_rung[*rr][2] += 1;
                }
                let tot = |k: usize| per_ex.iter().map(|e| e[k]).sum::<i64>() as f64;
                let (dl, swi) = (tot(0) / tot(2) * 100.0, tot(1) / tot(2) * 100.0);
                let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
                let mut ds: Vec<f64> = (0..1000).map(|_| {
                    let (mut d, mut m) = (0i64, 0i64);
                    for _ in 0..n_ex {
                        seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                        let e = per_ex[(seed % n_ex as u64) as usize];
                        d += e[0] - e[1]; m += e[2];
                    }
                    d as f64 / m.max(1) as f64 * 100.0
                }).collect();
                ds.sort_by(|a, b| a.total_cmp(b));
                let (lo, hi) = (ds[25], ds[974]);
                println!("   delayed synapses, the moment before (cross-fitted): top-1 {:.2}%  vs switch {:.2}%  diff {:+.2} \
(95% CI {:+.2} .. {:+.2}) -> {}", dl, swi, dl - swi, lo, hi, if dl > swi && lo > 0.0 { "BETTER" } else { "not better" });
                for (r, b) in by_rung.iter().enumerate() {
                    if b[2] == 0 { continue; }
                    println!("     {}{} interned: {:>5} positions  switch {:.2}%  with delayed {:.2}%  (delayed alone {:.2}%)   b = {} / {}",
                        r, if r == 3 { "+" } else { " " }, b[2], b[1] as f64 / b[2] as f64 * 100.0, b[0] as f64 / b[2] as f64 * 100.0,
                        alone[r][0] as f64 / alone[r][1].max(1) as f64 * 100.0, B[pick_b(1, r)], B[pick_b(0, r)]);
                }
                println!("TSV\tdelay\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.2}", train.len(), dl, swi, lo, hi);
                for r in 0..4 {
                    let both_halves = |b: usize| hits[0][r][b] + hits[1][r][b];
                    del_b[r] = B[(0..8).max_by_key(|&b| (both_halves(b), std::cmp::Reverse(b))).unwrap()];
                }
                println!("     when she speaks, the delayed vote's weight per rung: {:?}", del_b);
            }

            // ---- THE CONVERSATION VOTE (v29 WP2/WP3), the verdict, in exactly the delayed synapses'
            // form: whichever vote the switch uses, the conversation adds its own, score(w) = vote(w) +
            // b · conversation(w), both normalised over their top 10, b per rung chosen on the OTHER
            // exam half (cross-fitted). the vote reads only the prompt, the reply so far (here the
            // person's, word by word) and the turn before; never her cortex.
            if let Some(cv) = convo.as_ref() {
                // where each exchange's positions start in the exam's order
                let starts: Vec<usize> = test.iter().scan(0usize, |s, (_, r)| { let x = *s; *s += r.len() + 1; Some(x) }).collect();
                // per position: the vote (vote form), or the support of each word in her two candidate
                // votes, memory then fallback (gain form)
                type Read = (Vec<(String, f32)>, Vec<f32>, Vec<f32>);
                let per: Vec<Vec<Read>> = par_map(test, threads(), |ei, (q, rep)| {
                    (0..=rep.len()).map(|t| {
                        let (said, pv) = (&rep[..t], &test_prev[ei]);
                        match mixv.get(starts[ei] + t) {
                            Some(m) if cv.gain => (Vec::new(), cv.gains(q, said, pv, &m.0), cv.gains(q, said, pv, &m.1)),
                            _ => (cv.vote(q, said, pv), Vec::new(), Vec::new()),
                        }
                    }).collect()
                });
                let ctv: Vec<Read> = per.into_iter().flatten().collect();
                if ctv.len() == cal.len() {
                    let opts: &[f32] = if cv.gain { &CONVO_G } else { &CONVO_B };
                    let no = opts.len();
                    let mut hits = [[[0usize; 10]; 4]; 2];
                    let mut rows: Vec<(usize, usize, [bool; 10], bool)> = Vec::with_capacity(cal.len());
                    let mut alone = [[0usize; 2]; 4];   // the conversation alone right, positions where it reads anything
                    for (((c, n), m), (cvv, g0, g1)) in cal.iter().zip(&natr).zip(&mixv).zip(&ctv) {
                        let (h, ce, rr) = (c.0 & 1, cell(c), (c.1 as usize).min(3));
                        let o = hard[1 - h][ce];
                        let use_nat = o[1] > o[0];
                        let sw_ok = if use_nat { n[2] } else { c.3 };
                        let src = if use_nat { &m.1 } else { &m.0 };
                        let mut ok = [sw_ok; 10];
                        if cv.gain {
                            // option 0 is the switch itself; the others re-rank her chosen vote's top 10
                            let gs = if use_nat { g1 } else { g0 };
                            if !src.is_empty() && gs.iter().any(|&x| x > 0.0) {
                                let top = gs.iter().enumerate().fold((0usize, -1.0f32), |b, (i, &x)| if x > b.1 { (i, x) } else { b });
                                alone[rr][0] += (src[top.0].0 == m.2) as usize;
                                alone[rr][1] += 1;
                                for (gi, &g) in opts.iter().enumerate().skip(1) {
                                    let best = src.iter().zip(gs).map(|((w, x), s)| (w, x * (g * s).exp()))
                                        .fold(("", -1.0f32), |b, (w, y)| if y > b.1 { (w.as_str(), y) } else { b });
                                    ok[gi] = best.0 == m.2;
                                }
                            }
                        } else if !cvv.is_empty() {
                            alone[rr][0] += (cvv[0].0 == m.2) as usize;
                            alone[rr][1] += 1;
                            for (bi, &b) in opts.iter().enumerate().skip(1) {
                                let mut best: (&str, f32) = ("", -1.0);
                                for (w, x) in src {
                                    let y = x + b * cvv.iter().find(|z| &z.0 == w).map_or(0.0, |z| z.1);
                                    if y > best.1 { best = (w.as_str(), y); }
                                }
                                for (w, x) in cvv {
                                    if src.iter().any(|z| &z.0 == w) { continue; }
                                    let y = b * x;
                                    if y > best.1 { best = (w.as_str(), y); }
                                }
                                ok[bi] = if src.is_empty() && best.1 <= 0.0 { sw_ok } else { best.0 == m.2 };
                            }
                        }
                        for bi in 0..no { if ok[bi] { hits[h][rr][bi] += 1; } }
                        rows.push((c.0, rr, ok, sw_ok));
                    }
                    let pick_b = |h: usize, rr: usize| -> usize {
                        let o = &hits[1 - h][rr];
                        (0..no).max_by_key(|&b| (o[b], std::cmp::Reverse(b))).unwrap()
                    };
                    let n_ex = cal.iter().map(|c| c.0).max().map_or(0, |m| m + 1);
                    let mut per_ex = vec![[0i64; 3]; n_ex];
                    let mut by_rung = [[0usize; 3]; 4];
                    for (ei, rr, ok, sw_ok) in &rows {
                        let b = pick_b(ei & 1, *rr);
                        per_ex[*ei][0] += ok[b] as i64;
                        per_ex[*ei][1] += *sw_ok as i64;
                        per_ex[*ei][2] += 1;
                        by_rung[*rr][0] += ok[b] as usize;
                        by_rung[*rr][1] += *sw_ok as usize;
                        by_rung[*rr][2] += 1;
                    }
                    let tot = |k: usize| per_ex.iter().map(|e| e[k]).sum::<i64>() as f64;
                    let (cvl, swi) = (tot(0) / tot(2) * 100.0, tot(1) / tot(2) * 100.0);
                    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
                    let mut ds: Vec<f64> = (0..1000).map(|_| {
                        let (mut d, mut m) = (0i64, 0i64);
                        for _ in 0..n_ex {
                            seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                            let e = per_ex[(seed % n_ex as u64) as usize];
                            d += e[0] - e[1]; m += e[2];
                        }
                        d as f64 / m.max(1) as f64 * 100.0
                    }).collect();
                    ds.sort_by(|a, b| a.total_cmp(b));
                    let (lo, hi) = (ds[25], ds[974]);
                    println!("   conversation vote, v29 (cross-fitted)          : top-1 {:.2}%  vs switch {:.2}%  diff {:+.2} \
(95% CI {:+.2} .. {:+.2}) -> {}", cvl, swi, cvl - swi, lo, hi, if cvl > swi && lo > 0.0 { "BETTER" } else { "not better" });
                    for (r, b) in by_rung.iter().enumerate() {
                        if b[2] == 0 { continue; }
                        println!("     {}{} interned: {:>5} positions  switch {:.2}%  with the conversation {:.2}%  (it alone {:.2}%)   {} = {} / {}",
                            r, if r == 3 { "+" } else { " " }, b[2], b[1] as f64 / b[2] as f64 * 100.0, b[0] as f64 / b[2] as f64 * 100.0,
                            alone[r][0] as f64 / alone[r][1].max(1) as f64 * 100.0, if cv.gain { "g" } else { "b" }, opts[pick_b(1, r)], opts[pick_b(0, r)]);
                    }
                    println!("TSV\tconvo\t{}\t{:.2}\t{:.2}\t{:.2}\t{:.2}", train.len(), cvl, swi, lo, hi);
                    for r in 0..4 {
                        let both_halves = |b: usize| hits[0][r][b] + hits[1][r][b];
                        ct_b[r] = opts[(0..no).max_by_key(|&b| (both_halves(b), std::cmp::Reverse(b))).unwrap()];
                    }
                    println!("     when she speaks, the conversation vote's weight per rung: {:?}", ct_b);
                    ct_line = Some(rows.iter().map(|(ei, rr, ok, _)| ok[pick_b(ei & 1, *rr)]).collect());
                }
            }
        }

        // ---- THE CLEAN EXAM (v29, AVA_MASK): the very same per-position decisions as the lines
        // above (cells, halves and weights all fitted on the whole exam as before), counted over the
        // chosen half's positions beside the whole exam. her switch line is the switch, or the switch
        // plus the conversation vote when AVA_WM / AVA_PREV is on.
        if let Some(mk) = &mask {
            // per position, exactly sw_b(j) above: ava or native j per cell, the cell's table from the
            // other exchange-parity half
            let sw_line = |j: usize| -> Vec<bool> {
                let mut tab = [[[0usize; 2]; 40]; 2];
                for (c, n) in cal.iter().zip(&natr) {
                    let t = &mut tab[c.0 & 1][cell(c)];
                    if c.3 { t[0] += 1; }
                    if n[j] { t[1] += 1; }
                }
                cal.iter().zip(&natr).map(|(c, n)| {
                    let o = tab[1 - (c.0 & 1)][cell(c)];
                    if o[1] > o[0] { n[j] } else { c.3 }
                }).collect()
            };
            let (routed, switch) = (sw_line(1), sw_line(2));
            let line: &Vec<bool> = ct_line.as_ref().unwrap_or(&switch);
            let mine: Vec<usize> = (0..cal.len()).filter(|&i| mk.on[cal[i].0]).collect();
            let all: Vec<usize> = (0..cal.len()).collect();
            let pc = |v: &dyn Fn(usize) -> bool, of: &[usize]| of.iter().filter(|&&i| v(i)).count() as f64 / of.len().max(1) as f64 * 100.0;
            let rows: [(&str, &dyn Fn(usize) -> bool); 3] = [("plain top-1 (ava)", &|i| cal[i].3),
                ("ava / native routed per cell", &|i| routed[i]), ("switch (3-word level, per cell)", &|i| switch[i])];
            println!("\n   CLEAN EXAM, {} half  (next word top-1; {} of {} positions, {} exchanges)", mk.half, mine.len(), cal.len(),
                mk.on.iter().filter(|&&x| x).count());
            println!("     {:<44} {:>10} {:>12}", "line", "whole exam", format!("{} half", mk.half));
            for (name, f) in rows.iter() { println!("     {:<44} {:>9.2}% {:>11.2}%", name, pc(*f, &all), pc(*f, &mine)); }
            let lname = if ct_line.is_some() { "her switch line = switch + conversation vote" } else { "her switch line = switch" };
            println!("     {:<44} {:>9.2}% {:>11.2}%", lname, pc(&|i| line[i], &all), pc(&|i| line[i], &mine));
            println!("TSV\tmask_next\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}", mk.half, mine.len(),
                pc(&|i| line[i], &mine), pc(&|i| cal[i].3, &mine), pc(&|i| routed[i], &mine), pc(&|i| switch[i], &mine), all.len(),
                pc(&|i| line[i], &all), pc(&|i| cal[i].3, &all), pc(&|i| routed[i], &all), pc(&|i| switch[i], &all));
            if dump_path.is_some() {
                for &i in &mine {
                    dump += &format!("P\t{i}\t{}\t{}\t{}\t{}\t{}\n", cal[i].0, line[i] as u8, cal[i].3 as u8, routed[i] as u8, switch[i] as u8);
                }
            }
        }

        if (talk && probe & 8 == 0) || probe & 32 != 0 {
            // ---- THE SWITCHED MODEL AS A GENERATOR, shared by the reply exam and talk.
            // per (rung x margin) cell: her own vote or her native fallback (3-word,
            // novelty-gated), whichever was right more often there; her confidence is that
            // cell's measured hit rate. nothing outside her.
            let (mut both, mut ncell) = ([[0usize; 2]; 40], [0usize; 40]);
            for (c, n) in cal.iter().zip(&natr) {
                let i = cell(c);
                ncell[i] += 1;
                if c.3 { both[i][0] += 1; }
                if n[2] { both[i][1] += 1; }
            }
            // one step: the top-k words from the source this cell trusts (or her own vote,
            // if forced), her confidence in this state, and which source spoke.
            // RECOLLECTION (v24, probe 128): pattern completion from a partial cue. a question
            // reactivates each engram by the content they share, rare words weighing more
            // (no stop list: frequency grades it); a = shared / all, from 0 to 1. returns the
            // most active engram. nothing here is a threshold.
            let spec = |w: &str| -> f32 { spec_of(&s1, w) };
            // the most active engram, its activation, and the runner-up's
            // PATTERN SEPARATION (v25, AVA_SEPARATE=1), the dentate gyrus's job: a word that many
            // engrams share tells them apart poorly, so its weight in recall shrinks by
            // d(w) = 1 + ln((L + 1) / (1 + engrams containing w)). similar lessons ("neurons",
            // "units", "facts" in every lesson about her insides) get more distinct cues.
            let separate = env::var("AVA_SEPARATE").map_or(false, |v| v == "1");
            // THE CUE (v25, AVA_CUE bits). 1 = the WHOLE EPISODE is the engram: the answer's words
            // can complete a match (a paraphrase often names what the answer names), but a
            // question is never blamed for lacking its own answer. 2 = word STEMS: inflected
            // forms prime each other (morphological priming, "siblings" ~ "sibling"), graded by
            // shared prefix / longer word, from 4 letters. 0 = v24, bit for bit.
            // LINKS (v27) read the whole episode through each engram's learned links
            let links_on = links_on();
            let cue: u8 = if links_on { 1 } else { env::var("AVA_CUE").ok().and_then(|v| v.parse().ok()).unwrap_or(0) };
            let activate = |q: &[String]| -> Option<(usize, f32, f32)> {
                if probe & 128 == 0 { return None; }
                let lb = lessons.borrow();
                fn set<'a>(ws: impl Iterator<Item = &'a str>) -> Vec<&'a str> {
                    let mut v: Vec<&str> = ws.collect();
                    v.sort_unstable();
                    v.dedup();
                    v
                }
                let qs = set(q.iter().map(|w| w.as_str()));
                // per lesson: its question's words, and (cue 1) the answer's words not in it
                let sets: Vec<(Vec<&str>, Vec<&str>)> = lb.iter().map(|(eq, er)| {
                    let es = set(eq.iter().map(|w| w.as_str()));
                    let ea = if cue & 1 == 0 { Vec::new() } else {
                        set(er.iter().map(|w| w.as_str()).filter(|w| es.binary_search(w).is_err()))
                    };
                    (es, ea)
                }).collect();
                let l = sets.len() as f32;
                let sep = |w: &str| -> f32 {
                    if !separate { return 1.0; }
                    let n = sets.iter().filter(|(es, ea)| es.binary_search(&w).is_ok() || ea.binary_search(&w).is_ok()).count() as f32;
                    1.0 + ((l + 1.0) / (1.0 + n)).ln()
                };
                let wt = |w: &str| spec(w) * sep(w);
                let stem = |a: &str, b: &str| -> f32 {
                    let alpha = |x: &str| x.bytes().all(|c| c.is_ascii_alphabetic());
                    if cue & 2 == 0 || !alpha(a) || !alpha(b) { return 0.0; }
                    let p = a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count();
                    if p >= 4 { p as f32 / a.len().max(b.len()) as f32 } else { 0.0 }
                };
                // how well word w is matched by a sorted word set: 1 if present, else its best
                // stem, or (v28, AVA_MEANING) its best meaning overlap
                // AVA_MEANING_FREE=1: meaning only bridges a word that NO lesson contains (a gap in
                // what she was taught); two words both stored in lessons are told apart by identity
                let free_only = env::var("AVA_MEANING_FREE").map_or(false, |v| v == "1");
                let stored: std::collections::HashSet<&str> = if free_only {
                    sets.iter().flat_map(|(es, ea)| es.iter().chain(ea.iter()).copied()).collect()
                } else { Default::default() };
                let mean = |a: &str, b: &str| -> f32 {
                    if same_lexeme(a, b) { return 1.0; }
                    if free_only && stored.contains(a) && stored.contains(b) { return 0.0; }
                    match meaning_codes() { Some((m, p)) => m.overlap(a, b).powi(*p), None => 0.0 }
                };
                let hit = |w: &str, ws: &[&str]| -> f32 {
                    if ws.binary_search(&w).is_ok() { 1.0 } else { ws.iter().map(|v| stem(w, v).max(mean(w, v))).fold(0.0, f32::max) }
                };
                let (mut best, mut second): (Option<(usize, f32)>, f32) = (None, 0.0);
                let lk = links.borrow();
                // v28 idea 3: how much this question expects a number (None = off)
                let pn = p_number(&qs, &lb);
                let p0 = qkind().map_or(0.5, |k| k.p0);
                for (i, (es, ea)) in sets.iter().enumerate() {
                    let kind = |a: f32| -> f32 {
                        let Some(pn) = pn else { return a };
                        let f = if lb[i].1.iter().any(|w| is_number(w)) { pn / p0 } else { (1.0 - pn) / (1.0 - p0) };
                        a * f.min(1.0)
                    };
                    let a = kind(match lk.get(i) {
                        Some((kq, ka)) if links_on && kq.len() == es.len() && ka.len() == ea.len() =>
                            link_act(&qs, es, ea, kq, ka, &spec),
                        _ => {
                            let (mut shared, mut all) = (0f32, 0f32);
                            for w in &qs { let x = wt(w); all += x; shared += x * hit(w, es).max(hit(w, ea)); }
                            for w in es { all += wt(w) * (1.0 - hit(w, &qs)); }
                            if all > 0.0 { shared / all } else { 0.0 }
                        }
                    });
                    if a > best.map_or(0.0, |b| b.1) { second = best.map_or(0.0, |b| b.1); best = Some((i, a)); }
                    else if a > second { second = a; }
                }
                best.map(|(i, a)| (i, a, second))
            };
            // COMPETITION (v25, AVA_COMPETE). engrams inhibit each other, the way neighbouring
            // neurons do, so what counts is how far the winner stands out, not its raw strength:
            //   0 = none (v24: the winner's activation a1, bit for bit)
            //   1 = margin a1 - a2       2 = share a1·a1/(a1 + a2)       3 = steep a1²
            // a weak or ambiguous cue then reaches her vote with little or no weight.
            let compete: u8 = env::var("AVA_COMPETE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            let recall = |q: &[String]| -> Option<(usize, f32)> {
                let (i, a1, a2) = activate(q)?;
                let a = match compete {
                    1 => a1 - a2,
                    2 => a1 * a1 / (a1 + a2),
                    3 => a1 * a1,
                    _ => a1,
                };
                (a > 0.0).then_some((i, a))
            };
            // `prev` = the turn before the prompt (v29, read only by the conversation vote under AVA_PREV)
            let step = |q: &[String], prev: &[String], out: &[String], k: usize, force_ava: bool| -> (Vec<(String, f32)>, f64, bool) {
                let have = out.len().min(ORDER);
                let mut c: Vec<&str> = vec![BOS; ORDER];
                for i in 0..have { c[ORDER - have + i] = out[out.len() - have + i].as_str(); }
                // with an engram active, both votes are read over her top 10 so they can mix
                let rec = recall(q);
                let (k_out, k) = (k, if rec.is_some() || delay_on() { k.max(10) } else { k });
                let got = s1.ask(q, &c, out.len(), None, k.max(2));
                let m = match (got.first(), got.get(1)) {
                    (Some(a), Some(b)) if a.1 + b.1 > 0.0 => a.1 / (a.1 + b.1),
                    (Some(_), _) => 1.0,
                    _ => 0.0,
                };
                let (id, g) = s1.feats(q, &c, out.len(), None, true);
                let r = id.iter().filter(|k| !k.starts_with('C'))
                    .filter(|k| s1.fac.id.feat_id.contains_key(k.as_str())).count().min(3);
                let ci = r * 10 + edges.iter().filter(|e| m >= **e).count();
                let nat = !force_ava && both[ci][1] > both[ci][0];
                let mut ws: Vec<(String, f32)> = if nat {
                    let mut pv = Vec::new();
                    let w = s1.native_k(&g, &c, true, true, k, &mut pv).unwrap_or_default();
                    w.into_iter().map(|w| {
                        let x = s1.ro.op_id.get(&w).and_then(|&i| pv.get(i as usize)).copied().unwrap_or(0.0);
                        (w, x)
                    }).collect()
                } else { got.iter().take(k).map(|x| (x.0.clone(), x.1)).collect() };
                // DELAYED SYNAPSES (v26): the neurons of the moment before vote for this word too,
                // ws + b·delayed, both normalised over their top 10, b per rung from the exam.
                let b = del_b[r];
                if delay_on() && b > 0.0 && !out.is_empty() {
                    let pl = out.len() - 1;
                    let hp = pl.min(ORDER);
                    let mut cp: Vec<&str> = vec![BOS; ORDER];
                    for i in 0..hp { cp[ORDER - hp + i] = out[pl - hp + i].as_str(); }
                    let (idp, gp) = s1.feats(q, &cp, pl, None, true);
                    let dv = s1.rod.decode(&s1.code(&idp, &gp), 10);
                    let dt: f32 = dv.iter().map(|x| x.1.max(0.0)).sum();
                    let wt: f32 = ws.iter().map(|x| x.1.max(0.0)).sum::<f32>().max(1e-12);
                    for x in ws.iter_mut() { x.1 = x.1.max(0.0) / wt; }
                    if dt > 0.0 {
                        for (w, x) in dv {
                            let y = b * x.max(0.0) / dt;
                            match ws.iter_mut().find(|z| z.0 == w) { Some(z) => z.1 += y, None => ws.push((w, y)) }
                        }
                    }
                    ws.sort_by(|x, y| y.1.total_cmp(&x.1));
                }
                // THE CONVERSATION VOTE (v29, AVA_WM / AVA_PREV): the words in play vote too, ws + b·vote,
                // ws normalised to sum 1 first (the scale the exam learned b on), b per rung from the exam.
                if let Some(cv) = convo.as_ref() {
                    let b = ct_b[r];
                    if b > 0.0 && cv.gain {
                        // gain form: the words she already holds, each x exp(g · support), g = b here
                        let gs = cv.gains(q, out, prev, &ws);
                        for (x, s) in ws.iter_mut().zip(gs) { x.1 = x.1.max(0.0) * (b * s).exp(); }
                        ws.sort_by(|x, y| y.1.total_cmp(&x.1));
                    } else if b > 0.0 {
                        let v = cv.vote(q, out, prev);
                        if !v.is_empty() {
                            let wt: f32 = ws.iter().map(|x| x.1.max(0.0)).sum::<f32>().max(1e-12);
                            for x in ws.iter_mut() { x.1 = x.1.max(0.0) / wt; }
                            for (w, x) in v {
                                let y = b * x;
                                match ws.iter_mut().find(|z| z.0 == w) { Some(z) => z.1 += y, None => ws.push((w, y)) }
                            }
                            ws.sort_by(|x, y| y.1.total_cmp(&x.1));
                        }
                    }
                }
                // the reactivated engram votes for its own next word while her reply so far
                // follows it: final = (1 - a)·cortex + a·engram, the cortex vote normalised.
                if let Some((e, a)) = rec {
                    let lb = lessons.borrow();
                    let r = &lb[e].1;
                    if out.len() <= r.len() && out[..] == r[..out.len()] {
                        let h = r.get(out.len()).map_or(EOS, |s| s.as_str());
                        let tot: f32 = ws.iter().map(|x| x.1.max(0.0)).sum::<f32>().max(1e-12);
                        for x in ws.iter_mut() { x.1 = (1.0 - a) * x.1.max(0.0) / tot; }
                        match ws.iter_mut().find(|x| x.0 == h) { Some(x) => x.1 += a, None => ws.push((h.to_string(), a)) }
                        ws.sort_by(|x, y| y.1.total_cmp(&x.1));
                    }
                }
                ws.truncate(k_out.max(1));
                if ws.is_empty() { ws.push((EOS.to_string(), 1.0)); }
                (ws, (both[ci][nat as usize] as f64 + 1.0) / (ncell[ci] as f64 + 2.0), nat)
            };
            // finish a reply greedily from a prefix: words, per-step confidences, whether she
            // ENDED it (EOS before the watchdog), and how many words came from each source.
            type Reply = (Vec<String>, Vec<f64>, bool, [usize; 2]);
            // ADAPTATION (v24, `adapt` > 0). no bans: humans repeat ("no no no"). a neuron that
            // just fired is temporarily LESS excitable and recovers. each word she says leaves
            // a trace (1.0, fading x ADAPT_DECAY per later word, capped at 1); her vote for it
            // is damped by adapt x trace. strong learned evidence still wins; only weak,
            // generic votes (where loops come from) get tipped elsewhere. EOS never adapts.
            // adapt = 0 is the old greedy step, bit for bit.
            const ADAPT_DECAY: f32 = 0.8;   // fixed before any run, not tuned
            let finish = |q: &[String], prev: &[String], mut out: Vec<String>, mut ps: Vec<f64>, mut by: [usize; 2], force_ava: bool,
                          adapt: f32| -> Reply {
                let mut trace: Vec<(String, f32)> = Vec::new();
                while out.len() < say_stop {
                    let (ws, p, nat) = step(q, prev, &out, if adapt > 0.0 { 4 } else { 1 }, force_ava);
                    ps.push(p);
                    let w = if adapt > 0.0 {
                        let tot: f32 = ws.iter().map(|x| x.1.max(0.0)).sum::<f32>().max(1e-12);
                        let mut best = (0usize, f32::MIN);
                        for (i, (w, x)) in ws.iter().enumerate() {
                            let a = trace.iter().find(|t| &t.0 == w).map_or(0.0, |t| t.1.min(1.0));
                            let y = x.max(0.0) / tot * (1.0 - adapt * a);
                            if y > best.1 { best = (i, y); }
                        }
                        ws[best.0].0.clone()
                    } else { ws[0].0.clone() };
                    if w == EOS { return (out, ps, true, by); }
                    by[nat as usize] += 1;
                    for t in trace.iter_mut() { t.1 *= ADAPT_DECAY; }
                    match trace.iter_mut().find(|t| t.0 == w) { Some(t) => t.1 += 1.0, None => trace.push((w.clone(), 1.0)) }
                    out.push(w);
                }
                (out, ps, false, by)
            };
            // a PROPER reply: she ended it, and it ends like a sentence.
            let proper = |out: &[String], ended: bool| ended
                && out.last().map_or(false, |w| matches!(w.as_str(), "." | "?" | "!"));
            let mean = |ps: &[f64]| ps.iter().sum::<f64>() / ps.len().max(1) as f64;
            // RERANK — a whole reply as the goal. the opening word decides most of a reply,
            // so try several: her own top 4 and her fallback's top 4. finish each, and keep
            // the PROPER sentence she is most confident in as a whole. an improper one wins
            // only if nothing proper exists.
            const OPEN: usize = 4;
            let rerank = |q: &[String], prev: &[String]| -> Reply {
                let bos: Vec<&str> = vec![BOS; ORDER];
                let (_, p0, _) = step(q, prev, &[], 1, false);
                let g0 = s1.feats(q, &bos, 0, None, true).1;
                let mut firsts: Vec<(String, usize)> = s1.ask(q, &bos, 0, None, OPEN)
                    .into_iter().map(|(w, _)| (w, 0)).collect();
                for w in s1.native_k(&g0, &bos, true, true, OPEN, &mut Vec::new()).unwrap_or_default() {
                    if !firsts.iter().any(|f| f.0 == w) { firsts.push((w, 1)); }
                }
                let mut best: Option<(Reply, (bool, f64))> = None;
                for (w, src) in firsts {
                    if w == EOS { continue; }
                    let mut by = [0usize; 2];
                    by[src] += 1;
                    let cand = finish(q, prev, vec![w], vec![p0], by, false, 0.0);
                    let key = (proper(&cand.0, cand.2), mean(&cand.1));
                    if best.as_ref().map_or(true, |b| key.0 > b.1 .0 || (key.0 == b.1 .0 && key.1 > b.1 .1)) {
                        best = Some((cand, key));
                    }
                }
                best.map(|b| b.0).unwrap_or_else(|| finish(q, prev, Vec::new(), Vec::new(), [0; 2], false, 0.0))
            };
            // REPAIR (v23, post-hoc: her first reply, or the reranker's if it did not end as
            // a sentence) was RETIRED from talk in v24. rerank and repair are hard rules of
            // ours ("a reply must end in . ? !"), and on code repair cut exact replies to ~1%.
            // both survive only as exam rows below, so the record of what they did stays
            // measurable; nothing she says goes through them.

            if probe & 32 != 0 {
                // ---- REPLY EXAM. whole replies to every held-out prompt, scored against what
                // the person actually said. PRE-REGISTERED: the reranker counts as better only
                // if word-F1 rises over switch-greedy AND the proper-sentence rate does not fall.
                let f1 = |out: &[String], r: &[String]| -> f64 {
                    if out.is_empty() || r.is_empty() { return 0.0; }
                    let mut m: std::collections::HashMap<&str, i32> = Default::default();
                    for w in r { *m.entry(w.as_str()).or_default() += 1; }
                    let mut common = 0usize;
                    for w in out {
                        if let Some(c) = m.get_mut(w.as_str()) { if *c > 0 { *c -= 1; common += 1; } }
                    }
                    if common == 0 { return 0.0; }
                    let (pr, rc) = (common as f64 / out.len() as f64, common as f64 / r.len() as f64);
                    2.0 * pr * rc / (pr + rc)
                };
                let t = Instant::now();
                let names = ["ava greedy (her original)", "switch greedy (v22 talk)", "switch + whole-reply rerank",
                    "switch, repaired (post-hoc)", "switch + adaptation (soft)"];
                const ADAPT: f32 = 0.5;   // damping strength, fixed before any run, not tuned
                let mut sums = [[0f64; 5]; 5];   // F1, proper, exact, length, reached the watchdog
                let mut conf: Vec<(f64, f64, bool, bool)> = Vec::new();
                // three readings of her confidence in one reply: the mean over its words, the
                // FIRST word only (her read of the prompt itself), and the weakest word.
                let sig = |o: &Reply| [mean(&o.1), o.1[0], o.1.iter().cloned().fold(f64::INFINITY, f64::min)];
                let mut normal_sig: Vec<[f64; 3]> = Vec::new();
                let (mut phi_n, mut phi_r, mut phi_m) = (Vec::<f64>::new(), Vec::<f64>::new(), Vec::<f64>::new());
                let all4 = |q: &[String], prev: &[String]| -> [Reply; 5] {
                    let (sg, rr) = (finish(q, prev, Vec::new(), Vec::new(), [0; 2], false, 0.0), rerank(q, prev));
                    let rp = if proper(&sg.0, sg.2) { sg.clone() } else { rr.clone() };
                    [finish(q, prev, Vec::new(), Vec::new(), [0; 2], true, 0.0), sg, rr, rp,
                     finish(q, prev, Vec::new(), Vec::new(), [0; 2], false, ADAPT)]
                };
                // AVA_THREADS (v28): with n > 1 every prompt's replies are worked out on n cores
                // first (par_map); the loops below still count them one prompt at a time, in order
                let n_th = threads();
                let mut pre = (if n_th > 1 { par_map(test, n_th, |i, (q, _)| all4(q, &test_prev[i])) } else { Vec::new() }).into_iter();
                for (ti, (q, r)) in test.iter().enumerate() {
                    let outs = pre.next().unwrap_or_else(|| all4(q, &test_prev[ti]));
                    let mut ex = [(0f64, false, false); 5];
                    for (i, o) in outs.iter().enumerate() {
                        ex[i] = (f1(&o.0, r), proper(&o.0, o.2), &o.0 == r);
                        sums[i][0] += ex[i].0;
                        if ex[i].1 { sums[i][1] += 1.0; }
                        if ex[i].2 { sums[i][2] += 1.0; }
                        sums[i][3] += o.0.len() as f64;
                        if !o.2 { sums[i][4] += 1.0; }
                    }
                    rex.push([ex[0], ex[1], ex[3]]);
                    // does her own confidence know when a switch-greedy reply broke?
                    let o = &outs[1];
                    normal_sig.push(sig(o));
                    if let Some(f) = &fam { phi_n.push(f.phi(q)); }
                    let lo = o.1.iter().cloned().fold(f64::INFINITY, f64::min);
                    conf.push((mean(&o.1), lo, proper(&o.0, o.2), f1(&o.0, r) >= 0.5));
                }
                // STRANGE PROMPTS: the same prompts with their words reversed. no true reply
                // exists, so only two questions: does she still finish a sentence, and does
                // her confidence drop (the v22 fragments came from prompts unlike her data)?
                let mut odd = [0usize; 5];
                let (mut rev, mut mix): (Vec<[f64; 3]>, Vec<[f64; 3]>) = (Vec::new(), Vec::new());
                let reversed = |q: &[String]| -> Vec<String> { q.iter().rev().cloned().collect() };
                let mut pre = (if n_th > 1 { par_map(test, n_th, |i, (q, _)| all4(&reversed(q), &test_prev[i])) } else { Vec::new() }).into_iter();
                for (ti, (q, _)) in test.iter().enumerate() {
                    let q: Vec<String> = reversed(q);
                    let outs = pre.next().unwrap_or_else(|| all4(&q, &test_prev[ti]));
                    for (i, o) in outs.iter().enumerate() { if proper(&o.0, o.2) { odd[i] += 1; } }
                    rev.push(sig(&outs[1]));
                    if let Some(f) = &fam { phi_r.push(f.phi(&q)); }
                }
                // MIXED-UP prompts: real words, real lengths, no meaning. word j of prompt i is
                // drawn by a fixed stride from all test-prompt words.
                let flat: Vec<&String> = test.iter().flat_map(|(q, _)| q.iter()).collect();
                let mut mix_proper = 0usize;
                let mixed_up = |i: usize, q: &[String]| -> Vec<String> {
                    (0..q.len()).map(|j| flat[(i * 7919 + j * 104_729) % flat.len()].clone()).collect()
                };
                let mut pre = (if n_th > 1 {
                    par_map(test, n_th, |i, (q, _)| finish(&mixed_up(i, q), &test_prev[i], Vec::new(), Vec::new(), [0; 2], false, 0.0))
                } else { Vec::new() }).into_iter();
                for (i, (q, _)) in test.iter().enumerate() {
                    let q: Vec<String> = mixed_up(i, q);
                    let o = pre.next().unwrap_or_else(|| finish(&q, &test_prev[i], Vec::new(), Vec::new(), [0; 2], false, 0.0));
                    if proper(&o.0, o.2) { mix_proper += 1; }
                    mix.push(sig(&o));
                    if let Some(f) = &fam { phi_m.push(f.phi(&q)); }
                }
                let norm: Vec<[f64; 3]> = normal_sig.clone();
                let vs = |odd: &[[f64; 3]], k: usize| auc(&norm.iter().map(|x| (x[k], true))
                    .chain(odd.iter().map(|x| (x[k], false))).collect::<Vec<_>>());
                let mut normal: Vec<f64> = norm.iter().map(|c| c[0]).collect();
                normal.sort_by(|a, b| a.total_cmp(b));
                let line = normal[normal.len() / 4];
                let below = mix.iter().filter(|x| x[0] < line).count();
                let n = test.len() as f64;
                let real_proper = test.iter().filter(|(_, r)| proper(r, true)).count() as f64 / n * 100.0;
                let real_len = test.iter().map(|(_, r)| r.len()).sum::<usize>() as f64 / n;
                println!("\n   REPLY EXAM  ({} held-out prompts, whole replies vs what the person said, {:?})",
                    test.len(), t.elapsed());
                println!("     {:<30} {:>8} {:>9} {:>7} {:>8} {:>17}", "strategy", "word-F1", "proper", "exact", "avg len",
                    "proper, reversed");
                println!("     {:<30} {:>8} {:>8.1}% {:>7} {:>8.1} {:>17}", "(the real replies)", "-", real_proper, "-", real_len, "-");
                for i in 0..5 {
                    println!("     {:<30} {:>8.2} {:>8.1}% {:>6.1}% {:>8.1} {:>16.1}%", names[i],
                        sums[i][0] / n * 100.0, sums[i][1] / n * 100.0, sums[i][2] / n * 100.0, sums[i][3] / n,
                        odd[i] as f64 / n * 100.0);
                }
                if !short_only {
                    println!("     replies that never ended and reached the {say_stop}-word watchdog: {}",
                        (0..5).map(|i| format!("{:.1}%", sums[i][4] / n * 100.0)).collect::<Vec<_>>().join(" / "));
                }
                // THE CLEAN EXAM (v29, AVA_MASK): switch greedy on the chosen half's prompts, beside the whole exam
                if let Some(mk) = &mask {
                    let mine: Vec<usize> = (0..rex.len()).filter(|&i| mk.on[i]).collect();
                    let stat = |of: &[usize]| -> [f64; 3] {
                        let m = of.len().max(1) as f64;
                        [of.iter().map(|&i| rex[i][1].0).sum::<f64>() / m * 100.0,
                         of.iter().filter(|&&i| rex[i][1].1).count() as f64 / m * 100.0,
                         of.iter().filter(|&&i| rex[i][1].2).count() as f64 / m * 100.0]
                    };
                    let all: Vec<usize> = (0..rex.len()).collect();
                    let (h, w) = (stat(&mine), stat(&all));
                    println!("   CLEAN EXAM, {} half ({} prompts), switch greedy: word-F1 {:.2}  proper {:.2}%  exact {:.2}%   (whole exam: {:.2}  {:.2}%  {:.2}%)",
                        mk.half, mine.len(), h[0], h[1], h[2], w[0], w[1], w[2]);
                    println!("TSV\tmask_reply\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{}\t{:.4}\t{:.4}\t{:.4}", mk.half, mine.len(), h[0], h[1], h[2],
                        all.len(), w[0], w[1], w[2]);
                    if dump_path.is_some() {
                        for &i in &mine {
                            dump += &format!("E\t{i}\t{:.9}\t{}\t{}\n", rex[i][1].0, rex[i][1].1 as u8, rex[i][1].2 as u8);
                        }
                    }
                }
                let cut = |f: &dyn Fn(&(f64, f64, bool, bool)) -> (f64, bool)| auc(&conf.iter().map(f).collect::<Vec<_>>());
                println!("   does reply confidence know a broken reply? (switch greedy)");
                println!("     AUC for 'proper sentence'  : mean {:.3}   weakest word {:.3}",
                    cut(&|c| (c.0, c.2)), cut(&|c| (c.1, c.2)));
                println!("     AUC for 'word-F1 >= 0.5'   : mean {:.3}   weakest word {:.3}",
                    cut(&|c| (c.0, c.3)), cut(&|c| (c.1, c.3)));
                let (d_f1, d_pr) = ((sums[2][0] - sums[1][0]) / n * 100.0, (sums[2][1] - sums[1][1]) / n * 100.0);
                println!("   pre-registered verdict: rerank vs switch greedy  F1 {:+.2}, proper {:+.1} -> {}", d_f1, d_pr,
                    if d_f1 > 0.0 && d_pr >= 0.0 { "BETTER" } else { "not better" });
                println!("   post-hoc: repaired vs switch greedy  F1 {:+.2}, proper {:+.1}",
                    (sums[3][0] - sums[1][0]) / n * 100.0, (sums[3][1] - sums[1][1]) / n * 100.0);
                // PRE-REGISTERED (v24): adaptation counts as better only if the proper-sentence
                // rate RISES and word-F1 does NOT fall, against switch greedy.
                let (r_f1, r_pr) = ((sums[4][0] - sums[1][0]) / n * 100.0, (sums[4][1] - sums[1][1]) / n * 100.0);
                println!("   pre-registered (v24): adaptation vs switch greedy  F1 {:+.2}, proper {:+.1} -> {}", r_f1, r_pr,
                    if r_f1 >= 0.0 && r_pr > 0.0 { "BETTER" } else { "not better" });
                println!("TSV\tadapt\t{}\t{:.2}\t{:.1}\t{:.1}", train.len(), sums[4][0] / n * 100.0,
                    sums[4][1] / n * 100.0, sums[4][2] / n * 100.0);
                println!("   does she know a strange prompt? AUC of her confidence, normal vs strange (0.5 = can't tell)");
                println!("     {:<26} {:>7} {:>11} {:>13}", "", "mean", "first word", "weakest word");
                for (name, odd) in [("words reversed", &rev), ("words mixed up", &mix)] {
                    println!("     {:<26} {:>7.3} {:>11.3} {:>13.3}", name, vs(odd, 0), vs(odd, 1), vs(odd, 2));
                }
                println!("     mixed-up prompts: {:.1}% get a proper reply; {:.1}% fall under her unsure line (normal: 25%)",
                    mix_proper as f64 / n * 100.0, below as f64 / n * 100.0);
                if fam.is_some() && !phi_n.is_empty() {
                    // PRE-REGISTERED (v24, hippocampus design §2, fixed before any run):
                    //   F1: the feeling of knowing c^ = conf x u(phi), u = 1 - exp(-phi/0.1), tells
                    //       normal from mixed-up prompts with AUC >= 0.85 (confidence alone: 0.659);
                    //   F2: c^ still knows a good reply: AUC for 'word-F1 >= 0.5' stays >= 0.94;
                    //   F3: report only. c^ is a READING of her; it never changes what she says.
                    let u = |p: f64| 1.0 - (-p / 0.1).exp();
                    let fok = |s: &[[f64; 3]], p: &[f64]| -> Vec<f64> { s.iter().zip(p).map(|(s, &p)| s[0] * u(p)).collect() };
                    let (cn, cr, cx) = (fok(&norm, &phi_n), fok(&rev, &phi_r), fok(&mix, &phi_m));
                    let a = |x: &[f64], y: &[f64]| auc(&x.iter().map(|&v| (v, true))
                        .chain(y.iter().map(|&v| (v, false))).collect::<Vec<_>>());
                    let c0 = |s: &[[f64; 3]]| s.iter().map(|x| x[0]).collect::<Vec<f64>>();
                    let med = |v: &[f64]| { let mut s = v.to_vec(); s.sort_by(|a, b| a.total_cmp(b)); s.get(s.len() / 2).copied().unwrap_or(0.0) };
                    let f1g = auc(&conf.iter().zip(&cn).map(|(c, &v)| (v, c.3)).collect::<Vec<_>>());
                    let (am, ar) = (a(&cn, &cx), a(&cn, &cr));
                    let mut q4 = cn.clone();
                    q4.sort_by(|a, b| a.total_cmp(b));
                    let line4 = q4[q4.len() / 4];
                    println!("   FAMILIARITY (hippocampus): \"have I heard this kind of question before?\"  AUC normal vs strange");
                    println!("     {:<40} {:>9} {:>9}", "", "reversed", "mixed up");
                    println!("     {:<40} {:>9.3} {:>9.3}", "her reply confidence alone (before)", a(&c0(&norm), &c0(&rev)), a(&c0(&norm), &c0(&mix)));
                    println!("     {:<40} {:>9.3} {:>9.3}", "familiarity phi alone", a(&phi_n, &phi_r), a(&phi_n, &phi_m));
                    println!("     {:<40} {:>9.3} {:>9.3}", "feeling of knowing c^ = conf x u(phi)", ar, am);
                    println!("     median phi: normal {:.2}, reversed {:.2}, mixed up {:.2}; mixed-up prompts in normal's bottom quarter of c^: {:.1}%",
                        med(&phi_n), med(&phi_r), med(&phi_m), cx.iter().filter(|&&v| v < line4).count() as f64 / cx.len().max(1) as f64 * 100.0);
                    println!("   pre-registered (v24): F1 c^ mixed-up AUC {:.3} (>= 0.85) -> {};  F2 c^ AUC for 'word-F1 >= 0.5' {:.3} (>= 0.94) -> {}",
                        am, if am >= 0.85 { "PASS" } else { "FAIL" }, f1g, if f1g >= 0.94 { "PASS" } else { "FAIL" });
                    println!("TSV\tfam\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}", train.len(), am, ar, a(&phi_n, &phi_m), f1g);
                }
                println!("TSV\treply\t{}\t{}", train.len(), (0..4).map(|i| format!("{:.2}\t{:.1}", sums[i][0] / n * 100.0,
                    sums[i][1] / n * 100.0)).collect::<Vec<_>>().join("\t"));
                if multi {
                    println!("\n   PER DOMAIN, reply exam  (word-F1 / proper % / exact % per strategy)");
                    println!("     {:<16} {:>7}   {:<23} {:<23} {}", "domain", "prompts",
                        "ava greedy", "switch greedy", "repaired (post-hoc)");
                    for d in (0..nd).map(Some).chain([None]) {
                        let mine: Vec<&[(f64, bool, bool); 3]> = rex.iter().zip(test_dom)
                            .filter(|(_, &x)| d.map_or(true, |d| x == d)).map(|(e, _)| e).collect();
                        let m = mine.len().max(1) as f64;
                        let col = |k: usize| if mine.is_empty() { "-".to_string() } else { format!("{:>6.2} {:>6.1}% {:>6.1}%",
                            mine.iter().map(|e| e[k].0).sum::<f64>() / m * 100.0,
                            mine.iter().filter(|e| e[k].1).count() as f64 / m * 100.0,
                            mine.iter().filter(|e| e[k].2).count() as f64 / m * 100.0) };
                        println!("     {:<16} {:>7}   {:<23} {:<23} {}", d.map_or("(all)", |d| doms[d].as_str()),
                            mine.len(), col(0), col(1), col(2));
                    }
                }
            }

            if talk && probe & 8 == 0 {
                // ---- TALK, AVA ALONE (v24: no hard rules). she says her own first reply, word by
                // word, and nothing is redone or added. the v23 "repair" (redo a reply that does
                // not end in . ? !) and the canned "i'm not sure. my best guess:" were rules
                // written by us, not things she learned; both are gone. how sure she was stays
                // as a READING in brackets, the way a scan shows a brain, never as her words.
                println!("\n   talk mode: ava alone — memory + her own fallback, nothing added or redone. \
the bracket is a reading of her, not her words. :q to leave.");
                use std::io::{BufRead, Write};
                // TEACHING (v24). a line starting with "=" is the person showing her what an
                // appropriate reply to their last question would have been, the way a parent
                // says "you could say: ...". she does not change on the spot: the lesson goes
                // into today's log (the hippocampus's pending store, ../data/sleep/, local and
                // never published) and becomes long-term memory when she sleeps (AVA_SLEEP).
                let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
                let log_path = format!("../data/sleep/talk-{stamp}.tsv");
                let mut log: Option<fs::File> = None;
                let mut last_q: Option<Vec<String>> = None;
                // v29 (AVA_PREV): the turn before the person's line is her own last reply
                let mut last_said: Vec<String> = Vec::new();
                let mut lessons_n = 0usize;
                let stdin = std::io::stdin();
                loop {
                    print!("you> ");
                    std::io::stdout().flush().ok();
                    let mut s = String::new();
                    if stdin.lock().read_line(&mut s).unwrap_or(0) == 0 { break; }
                    let s = s.trim();
                    if s.is_empty() { continue; }
                    if s == ":q" { break; }
                    if let Some(t) = s.strip_prefix('=') {
                        let r = talk_tokenize(t);
                        match &last_q {
                            Some(q) if !r.is_empty() => {
                                if log.is_none() {
                                    fs::create_dir_all("../data/sleep").ok();
                                    log = fs::OpenOptions::new().create(true).append(true).open(&log_path).ok();
                                }
                                if let Some(f) = log.as_mut() {
                                    writeln!(f, "@sleep\t{}\t{}\tteach", q.join("\u{1f}"), r.join("\u{1f}")).ok();
                                    f.flush().ok();
                                }
                                lessons_n += 1;
                                // the hippocampus holds it from now on; the cortex learns it tonight
                                if links_on {
                                    // (v27) and every other engram this question pulled up loses the links that pulled it
                                    link_trial(&mut lessons.borrow_mut(), &mut links.borrow_mut(), q, &r, true, link_eta(), &spec);
                                } else {
                                    let e = (q.clone(), r.clone());
                                    if !lessons.borrow().contains(&e) { lessons.borrow_mut().push(e); }
                                }
                                println!("     [noted for tonight: \"{}\" -> \"{}\"]", q.join(" "), r.join(" "));
                            }
                            _ => println!("     [nothing to teach yet: ask her something first]"),
                        }
                        continue;
                    }
                    let qt = talk_tokenize(s);
                    let (out, ps, ended, by) = finish(&qt, &last_said, Vec::new(), Vec::new(), [0; 2], false, 0.0);
                    let sure = mean(&ps);
                    let said = if out.is_empty() { "...".to_string() } else { out.join(" ") };
                    let mut feel = fam.as_ref().map_or(String::new(), |f| format!("; the question feels {:.0}% familiar", f.phi(&qt) * 100.0));
                    if let Some((e, a1, a2)) = activate(&qt) {
                        let w = recall(&qt).map_or(0.0, |x| x.1);
                        feel += &format!("; it reminds her of lesson {} ({:.0}%, runner-up {:.0}%, weight {:.0}%)", e + 1, a1 * 100.0, a2 * 100.0, w * 100.0);
                        if links_on {
                            feel += &format!("; its links kept {:.0}%", link_kept(&lessons.borrow(), &links.borrow(), e, &spec) * 100.0);
                        }
                    }
                    println!("ava> {said}\n     [reading: ~{:.0}% sure{feel}; words from memory {}, from her fallback {}{}]",
                        sure * 100.0, by[0], by[1], if ended { String::new() } else { format!("; stopped by the watchdog at {say_stop} words") });
                    last_q = Some(qt);
                    last_said = out;
                }
                if lessons_n > 0 {
                    println!("   {lessons_n} lessons waiting in {log_path}; she learns them when she sleeps (AVA_SLEEP={log_path})");
                }
            }
        }
    }
    if probe & 8 != 0 && knr.len() == cal.len() && !cal.is_empty() {
        let pc = |x: usize| x as f64 / n_all as f64 * 100.0;
        let pb = |x: usize, r: usize| x as f64 / rung[r].max(1) as f64 * 100.0;
        println!("\n   KNESER-NEY YARDSTICK  (same {} exchanges, same {n_all} positions)", train.len());
        println!("     {:<22} {:>7} {:>7}   {:>7} {:>7} {:>7} {:>7}", "model", "top-1", "top-5",
            "0-rung", "1", "2", "3+");
        println!("     {:<22} {:>6.2}% {:>6.2}%   {:>6.2}% {:>6.2}% {:>6.2}% {:>6.2}%", "ava",
            pc(t1), pc(t5), pb(hit[0], 0), pb(hit[1], 1), pb(hit[2], 2), pb(hit[3], 3));
        let mut tsv = Vec::new();
        for v in 0..2 {
            for (oi, order) in [3, 4, 5].into_iter().enumerate() {
                let k = &kn1[v][oi];
                println!("     {:<22} {:>6.2}% {:>6.2}%   {:>6.2}% {:>6.2}% {:>6.2}% {:>6.2}%",
                    format!("kn {} n={order}", if v == 0 { "reply-only" } else { "prompt+reply" }),
                    pc(k.iter().sum()), pc(kn5[v][oi]), pb(k[0], 0), pb(k[1], 1), pb(k[2], 2), pb(k[3], 3));
                tsv.push(format!("{:.2}", pc(k.iter().sum())));
            }
        }
        // ROUTER. per (rung x margin-decile) cell, answer with whichever candidate was
        // right most often — decided on the OTHER half of the exam (exchange parity), so
        // no position is routed by a rule it helped choose. ties go to ava.
        //   pre-registered: {ava, kn n=4}.   wider: {ava, kn n=4, kn n=5}, same discipline.
        let mut ms: Vec<f32> = cal.iter().map(|c| c.2).collect();
        ms.sort_by(|a, b| a.total_cmp(b));
        let edges: Vec<f32> = (1..10).map(|k| ms[ms.len() * k / 10]).collect();
        let cell = |c: &(usize, u8, f32, bool)| c.1 as usize * 10 + edges.iter().filter(|e| c.2 >= **e).count();
        let mut tab = [[[0usize; 3]; 40]; 2];
        for (c, k) in cal.iter().zip(&knr) {
            let t = &mut tab[c.0 & 1][cell(c)];
            for (i, ok) in [c.3, k[0], k[1]].into_iter().enumerate() { if ok { t[i] += 1; } }
        }
        let route = |m: usize| -> (usize, usize) {
            let (mut right, mut used) = (0usize, 0usize);
            for (c, k) in cal.iter().zip(&knr) {
                let o = tab[1 - (c.0 & 1)][cell(c)];
                let pick = (0..m).fold(0, |b, i| if o[i] > o[b] { i } else { b });
                if pick != 0 { used += 1; }
                if [c.3, k[0], k[1]][pick] { right += 1; }
            }
            (right, used)
        };
        let oracle = cal.iter().zip(&knr).filter(|(c, k)| c.3 || k[0] || k[1]).count();
        println!("     any of the three right (ceiling of any switch) : {:.2}%", pc(oracle));
        let (r2, u2) = route(2);
        let (r3, u3) = route(3);
        println!("   ROUTED {{ava, kn4}}      cross-fitted : top-1 {:.2}%  ({:+.2} vs ava; kn answers {:.1}%)",
            pc(r2), pc(r2) - pc(t1), pc(u2));
        println!("   ROUTED {{ava, kn4, kn5}} cross-fitted : top-1 {:.2}%  ({:+.2} vs ava; kn answers {:.1}%)",
            pc(r3), pc(r3) - pc(t1), pc(u3));
        let (oracle, routed) = (oracle, r3);
        println!("TSV\tkn\t{}\t{}\t{:.2}\t{:.2}", train.len(), tsv.join("\t"), pc(oracle), pc(routed));

        if talk {
            // ---- TALK. the routed model, live. each word comes from whichever of ava /
            // kn4 / kn5 this (rung x margin) cell trusted most across the whole exam, and
            // her confidence is that cell's measured hit rate — not a guess about herself,
            // a count of how often she was right in exactly this state.
            let mut ncell = [0usize; 40];
            for c in &cal { ncell[cell(c)] += 1; }
            let both: Vec<[usize; 3]> = (0..40).map(|i| {
                let mut s = [0usize; 3];
                for f in 0..2 { for k in 0..3 { s[k] += tab[f][i][k]; } }
                s
            }).collect();
            let kn = &kns[1];
            let step = |q: &[String], out: &[String]| -> (String, f64, usize) {
                let have = out.len().min(ORDER);
                let mut c: Vec<&str> = vec![BOS; ORDER];
                for k in 0..have { c[ORDER - have + k] = out[out.len() - have + k].as_str(); }
                let got = s1.ask(q, &c, out.len(), None, 2);
                let m = match (got.first(), got.get(1)) {
                    (Some(a), Some(b)) if a.1 + b.1 > 0.0 => a.1 / (a.1 + b.1),
                    (Some(_), _) => 1.0,
                    _ => 0.0,
                };
                let (id, _) = s1.feats(q, &c, out.len(), None, true);
                let r = id.iter().filter(|k| !k.starts_with('C'))
                    .filter(|k| s1.fac.id.feat_id.contains_key(k.as_str())).count().min(3);
                let ci = r * 10 + edges.iter().filter(|e| m >= **e).count();
                let s = both[ci];
                let pick = (0..3).fold(0, |b, i| if s[i] > s[b] { i } else { b });
                let w = if pick == 0 { got.first().map(|x| x.0.clone()) } else {
                    let mut kp = Vec::new();
                    kn.top(&kn.ctx(q, true, out, out.len()), pick + 3, 1, &mut kp)
                        .first().map(|&i| kn.words[i as usize].clone())
                };
                (w.unwrap_or_else(|| EOS.to_string()), (s[pick] as f64 + 1.0) / (ncell[ci] as f64 + 2.0), pick)
            };
            let speak = |q: &[String]| -> (Vec<String>, f64, [usize; 3]) {
                let (mut out, mut ps, mut by) = (Vec::new(), Vec::new(), [0usize; 3]);
                for _ in 0..say_stop {
                    let (w, p, pick) = step(q, &out);
                    ps.push(p);
                    if w == EOS { break; }
                    by[pick] += 1;
                    out.push(w);
                }
                (out, ps.iter().sum::<f64>() / ps.len().max(1) as f64, by)
            };
            // v24: no canned "i'm not sure" (a rule of ours, not her words); the bracket is a reading.
            println!("\n   talk mode: ava + kneser-ney, routed by her own confidence. \
the bracket is a reading of her, not her words. :q to leave.");
            use std::io::{BufRead, Write};
            let stdin = std::io::stdin();
            loop {
                print!("you> ");
                std::io::stdout().flush().ok();
                let mut s = String::new();
                if stdin.lock().read_line(&mut s).unwrap_or(0) == 0 { break; }
                let s = s.trim();
                if s.is_empty() { continue; }
                if s == ":q" { break; }
                let q = talk_tokenize(s);
                let (out, sure, by) = speak(&q);
                let said = if out.is_empty() { "...".to_string() } else { out.join(" ") };
                println!("ava> {said}\n     [reading: ~{:.0}% sure; words from ava {}, kn4 {}, kn5 {}]",
                    sure * 100.0, by[0], by[1], by[2]);
            }
        }
    }
    if two {
        println!("   stage 2 overruled stage 1 {changed} times: {fixed} fixed, {broke} broke, \
{} neutral", changed - fixed - broke);
        println!("   -> net {:+} correct answers from having a second layer",
            fixed as i64 - broke as i64);
    }
    if multi {
        // one machine line per domain: name, positions, next-word top-1 for ava / switch /
        // kn5, then reply exact and word-F1 for ava greedy and repaired. "-" = not measured.
        let rx = |d: usize, k: usize, f1: bool| -> String {
            let mine: Vec<&(f64, bool, bool)> = rex.iter().zip(test_dom).filter(|(_, &x)| x == d)
                .map(|(e, _)| &e[k]).collect();
            if mine.is_empty() || rex.len() != test.len() { return "-".into(); }
            let v = if f1 { mine.iter().map(|e| e.0).sum::<f64>() } else { mine.iter().filter(|e| e.2).count() as f64 };
            format!("{:.2}", v / mine.len() as f64 * 100.0)
        };
        println!();
        for d in 0..nd {
            println!("TSV\tdomain\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", doms[d], d_pos[d],
                dpc(Some(&d_ava), Some(d)), dpc(d_sw.as_ref(), Some(d)), dpc(d_kn.as_ref(), Some(d)),
                rx(d, 0, false), rx(d, 0, true), rx(d, 2, false), rx(d, 2, true));
        }
    }

    let gen = |text: &str| -> String {
        let q = seq::tokenize(text);
        let mut out: Vec<String> = Vec::new();
        for _ in 0..12 {
            let have = out.len().min(ORDER);
            let mut c: Vec<&str> = vec![BOS; ORDER];
            for k in 0..have { c[ORDER - have + k] = out[out.len() - have + k].as_str(); }
            let Some((w, _)) = answer(&q, &c, out.len(), 1).into_iter().next() else { break };
            if w == EOS { break; }
            out.push(w);
        }
        if out.is_empty() { "(silence)".into() } else { out.join(" ") }
    };
    println!();
    for p in ["hello", "thank you very much", "i love you", "what time is it ?",
              "i am hungry", "where are you going ?"] {
        println!("   you> {:<22} ava> {}", p, gen(p));
    }
    println!("\nTSV\tdepth{}\t{}\t{:.2}\t{:.2}", if two { 2 } else { 1 }, train.len(),
        t1 as f64 / n_all as f64 * 100.0, t5 as f64 / n_all as f64 * 100.0);
    // v29 (AVA_DUMP, with AVA_MASK): the chosen half, per position (P) and per prompt (E), for paired_boot
    if let (Some(p), Some(mk)) = (&dump_path, &mask) {
        let sw = |v: &str| env::var(v).unwrap_or_else(|_| "-".into());
        let head = format!("# ava dump v1\thalf={}\tmask={}\tAVA_WM={}\tAVA_PREV={}\tAVA_DEDUP={}\tAVA_WM_DECAY={}\tAVA_WM_KEEP={}\tAVA_WM_FORM={}\tAVA_WM_LINK={}\n\
# P\tposition\texchange\ther_switch_line\tplain_top1\trouted_per_cell\tswitch\n# E\texchange\tF1_switch_greedy\tproper\texact\n",
            mk.half, mk.path, sw("AVA_WM"), sw("AVA_PREV"), sw("AVA_DEDUP"), sw("AVA_WM_DECAY"), sw("AVA_WM_KEEP"), sw("AVA_WM_FORM"), sw("AVA_WM_LINK"));
        fs::write(p, head + &dump)?;
    }
    Ok(())
}
