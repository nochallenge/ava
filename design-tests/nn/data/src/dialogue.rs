// TESTING / DESIGN ONLY
//! dialogue: v42's threaded conversations, one whole conversation per document (V5).
//!
//! WHY. Her old dialogue text was single lines (each Cornell line its own document), so nothing
//! earlier in a document could help predict a reply. Here every conversation is one document, so
//! the memory has something to look back at.
//!
//! FORMAT. One turn per line, each line starting with a plain speaker mark: the first speaker of
//! the conversation is "A:", the second "B:", any further speaker "C:", "D:", ... e.g.
//!   A: They do not!
//!   B: They do to!
//! The text of a turn has its whitespace collapsed to single spaces. Turns with no text are left out.
//!
//! CORNELL. `cornell_lines.txt` rows are `Lid +++$+++ uid +++$+++ mid +++$+++ NAME +++$+++ text`
//! (Windows-1252 bytes, read with `docs::decode_legacy`); `cornell_convs.txt` rows are
//! `uid1 +++$+++ uid2 +++$+++ mid +++$+++ ['L194', 'L195', ...]`, the line ids in spoken order.
//! A conversation keeps its turns in that order; it is kept when at least 2 turns have text.
//!
//! DAILYDIALOG. `triples.tsv` rows are three consecutive turns of one dialogue (`t1 \t t2 \t t3`),
//! sliding by one turn. A row continues the dialogue of the row before it when its first two turns
//! are that row's last two; otherwise a new dialogue starts. Turns alternate A / B. The text is
//! tokenised with spaces around punctuation ("Say , Jim ."); `detok` removes those spaces. Exact
//! repeats of a whole dialogue are dropped (counted). NOTE: only the first 48,507 rows of the local
//! `triples.tsv` are DailyDialog; the rest is the Cornell movie corpus again (its first row is the
//! first Cornell conversation). So a rebuilt dialogue is dropped (counted as `known`) when more than
//! half of its turns are Cornell lines (`cornell_line_set`): those are already in the Cornell documents.

use crate::docs::{collapse_ws, decode_legacy};
use crate::hash::{new_map, FastMap, FastSet};
use std::collections::hash_map::Entry;
use std::fs::File;
use std::hash::{BuildHasher, Hash};
use std::io::{self, BufRead, BufReader};
use std::path::Path;

const CORNELL_SEP: &str = " +++$+++ ";

/// Speaker mark for the n-th distinct speaker of a conversation (0 -> "A").
pub fn mark(n: usize) -> char {
    (b'A' + (n.min(25)) as u8) as char
}

/// Join turns `(speaker index, text)` into one document: "A: ...\nB: ...".
pub fn format_turns(turns: &[(usize, String)]) -> String {
    let mut out = String::new();
    for (i, (who, text)) in turns.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push(mark(*who));
        out.push_str(": ");
        out.push_str(text);
    }
    out
}

/// One Cornell line: (line id, speaker id, text).
pub fn parse_cornell_line(raw: &[u8]) -> Option<(String, String, String)> {
    let s = decode_legacy(raw);
    let s = s.trim_end_matches(['\n', '\r']);
    let f: Vec<&str> = s.splitn(5, CORNELL_SEP).collect();
    if f.len() < 4 {
        return None;
    }
    let text = if f.len() == 5 { collapse_ws(f[4]) } else { String::new() };
    Some((f[0].trim().to_string(), f[1].trim().to_string(), text))
}

/// One Cornell conversation row: (movie id, line ids in order).
pub fn parse_cornell_conv(line: &str) -> Option<(String, Vec<String>)> {
    let f: Vec<&str> = line.trim_end_matches(['\n', '\r']).split(CORNELL_SEP).collect();
    if f.len() != 4 {
        return None;
    }
    let ids: Vec<String> = f[3].split('\'').filter(|p| p.starts_with('L') && p.len() > 1).map(|p| p.to_string()).collect();
    if ids.is_empty() {
        return None;
    }
    Some((f[2].trim().to_string(), ids))
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConvStats {
    /// Conversation rows read.
    pub rows: u64,
    /// Documents made.
    pub docs: u64,
    /// Turns in the documents.
    pub turns: u64,
    /// Turns whose line id was missing or whose text was empty (left out).
    pub missing_turns: u64,
    /// Conversations left with fewer than 2 turns (dropped).
    pub too_short: u64,
    /// Exact repeats of an earlier dialogue (dropped).
    pub repeats: u64,
    /// Bad rows.
    pub bad_rows: u64,
    /// Rebuilt dialogues made mostly of known (Cornell) lines (dropped).
    pub known: u64,
}

/// Hashes of every Cornell line's text (whitespace collapsed), to spot Cornell turns elsewhere.
pub fn cornell_line_set(lines_path: &Path) -> io::Result<FastSet<u64>> {
    let mut set: FastSet<u64> = FastSet::default();
    let r = BufReader::with_capacity(1 << 20, File::open(lines_path)?);
    for line in r.split(b'\n') {
        if let Some((_, _, text)) = parse_cornell_line(&line?) {
            if !text.is_empty() {
                set.insert(fast_hash(text.as_str()));
            }
        }
    }
    Ok(set)
}

/// Turn a list of (speaker id, text) into turns with speaker indices by order of first appearance.
fn index_speakers(raw: Vec<(String, String)>) -> Vec<(usize, String)> {
    let mut seen: Vec<String> = Vec::new();
    raw.into_iter()
        .map(|(who, text)| {
            let i = match seen.iter().position(|s| *s == who) {
                Some(i) => i,
                None => {
                    seen.push(who);
                    seen.len() - 1
                }
            };
            (i, text)
        })
        .collect()
}

/// Build Cornell conversations: `emit(doc)` is called once per kept conversation, in file order.
/// `limit` stops after that many conversation rows.
pub fn cornell(lines_path: &Path, convs_path: &Path, limit: Option<u64>, mut emit: impl FnMut(&str) -> io::Result<()>) -> io::Result<ConvStats> {
    // line id -> (speaker, text); about 300k short lines, a few tens of MB
    let mut lines: FastMap<String, (String, String)> = new_map();
    let mut r = BufReader::with_capacity(1 << 20, File::open(lines_path)?);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        if r.read_until(b'\n', &mut buf)? == 0 {
            break;
        }
        if let Some((id, who, text)) = parse_cornell_line(&buf) {
            lines.insert(id, (who, text));
        }
    }
    let mut st = ConvStats::default();
    let r = BufReader::with_capacity(1 << 20, File::open(convs_path)?);
    for line in r.split(b'\n') {
        if limit.is_some_and(|l| st.rows >= l) {
            break;
        }
        let line = decode_legacy(&line?);
        if line.trim().is_empty() {
            continue;
        }
        st.rows += 1;
        let Some((_movie, ids)) = parse_cornell_conv(&line) else {
            st.bad_rows += 1;
            continue;
        };
        let mut raw = Vec::new();
        for id in &ids {
            match lines.get(id) {
                Some((who, text)) if !text.is_empty() => raw.push((who.clone(), text.clone())),
                _ => st.missing_turns += 1,
            }
        }
        if raw.len() < 2 {
            st.too_short += 1;
            continue;
        }
        let turns = index_speakers(raw);
        st.docs += 1;
        st.turns += turns.len() as u64;
        emit(&format_turns(&turns))?;
    }
    Ok(st)
}

/// Contraction endings that tokenised text splits off with a spaced apostrophe ("you ’ re").
const CONTRACTIONS: [&str; 8] = ["s", "re", "ll", "ve", "t", "d", "m", "S"];

/// Remove the spaces that tokenised text puts around punctuation:
/// "Say , Jim , how ?" -> "Say, Jim, how?"; "( a )" -> "(a)"; "you ’ re" -> "you're"
/// (the curly apostrophe becomes a plain one); whitespace collapsed.
pub fn detok(s: &str) -> String {
    let s = collapse_ws(&s.replace('\u{2019}', "'"));
    // join "word ' re" back into "word're"
    let words: Vec<&str> = s.split(' ').collect();
    let mut joined: Vec<String> = Vec::with_capacity(words.len());
    let mut i = 0;
    while i < words.len() {
        let prev_letter = joined.last().and_then(|w| w.chars().last()).is_some_and(|c| c.is_alphabetic());
        if words[i] == "'" && prev_letter && i + 1 < words.len() && CONTRACTIONS.contains(&words[i + 1]) {
            let last = joined.last_mut().unwrap();
            last.push('\'');
            last.push_str(words[i + 1]);
            i += 2;
            continue;
        }
        joined.push(words[i].to_string());
        i += 1;
    }
    let s = joined.join(" ");
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == ' ' {
            let next = chars.get(i + 1).copied();
            let prev = out.chars().last();
            if matches!(next, Some(',' | '.' | '?' | '!' | ';' | ':' | ')' | '%')) || prev == Some('(') {
                i += 1;
                continue;
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

fn fast_hash<T: Hash + ?Sized>(x: &T) -> u64 {
    crate::hash::FastBuild::default().hash_one(x)
}

/// Rebuild DailyDialog dialogues from sliding triples (see the module docs). `emit` is called once
/// per kept dialogue (at least 3 turns, not an exact repeat, not mostly `known` lines). `limit` stops
/// after that many rows.
pub fn dailydialog_triples(path: &Path, limit: Option<u64>, known: &FastSet<u64>, mut emit: impl FnMut(&str) -> io::Result<()>) -> io::Result<ConvStats> {
    let mut st = ConvStats::default();
    let mut seen: FastSet<u64> = FastSet::default();
    let mut cur: Vec<String> = Vec::new();
    let mut flush = |cur: &mut Vec<String>, st: &mut ConvStats| -> io::Result<()> {
        if cur.is_empty() {
            return Ok(());
        }
        let n_known = cur.iter().filter(|t| known.contains(&fast_hash(collapse_ws(t).as_str()))).count();
        if 2 * n_known > cur.len() {
            st.known += 1;
            cur.clear();
            return Ok(());
        }
        let turns: Vec<(usize, String)> = cur.drain(..).enumerate().map(|(i, t)| (i % 2, detok(&t))).filter(|(_, t)| !t.is_empty()).collect();
        if turns.len() < 2 {
            st.too_short += 1;
            return Ok(());
        }
        let doc = format_turns(&turns);
        if !seen.insert(fast_hash(doc.as_str())) {
            st.repeats += 1;
            return Ok(());
        }
        st.docs += 1;
        st.turns += turns.len() as u64;
        emit(&doc)
    };
    let r = BufReader::with_capacity(1 << 20, File::open(path)?);
    for line in r.split(b'\n') {
        if limit.is_some_and(|l| st.rows >= l) {
            break;
        }
        let line = decode_legacy(&line?);
        let line = line.trim_end_matches(['\n', '\r']);
        if line.trim().is_empty() {
            continue;
        }
        st.rows += 1;
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 3 {
            st.bad_rows += 1;
            continue;
        }
        let n = cur.len();
        let continues = n >= 2 && cur[n - 2] == f[0] && cur[n - 1] == f[1];
        if continues {
            cur.push(f[2].to_string());
        } else {
            flush(&mut cur, &mut st)?;
            cur.extend(f.iter().map(|s| s.to_string()));
        }
    }
    flush(&mut cur, &mut st)?;
    Ok(st)
}

/// Count distinct values (used by the report for turns-per-doc spread).
pub fn histogram(values: &[usize], edges: &[usize]) -> Vec<(String, usize)> {
    let mut counts: FastMap<usize, usize> = new_map();
    for &v in values {
        let b = edges.iter().position(|&e| v < e).unwrap_or(edges.len());
        match counts.entry(b) {
            Entry::Occupied(mut o) => *o.get_mut() += 1,
            Entry::Vacant(e) => {
                e.insert(1);
            }
        }
    }
    (0..=edges.len())
        .map(|b| {
            let lo = if b == 0 { 0 } else { edges[b - 1] };
            let label = if b == edges.len() { format!("{lo}+") } else { format!("{lo}-{}", edges[b] - 1) };
            (label, counts.get(&b).copied().unwrap_or(0))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(name: &str, body: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ava_dlg_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        std::fs::File::create(&p).unwrap().write_all(body).unwrap();
        p
    }

    #[test]
    fn marks_and_format() {
        assert_eq!(mark(0), 'A');
        assert_eq!(mark(1), 'B');
        assert_eq!(mark(2), 'C');
        let t = vec![(0, "hi".to_string()), (1, "yo".to_string()), (0, "bye".to_string())];
        assert_eq!(format_turns(&t), "A: hi\nB: yo\nA: bye");
    }

    #[test]
    fn detok_fixes_spaced_punctuation() {
        assert_eq!(detok("Say , Jim , how about going for a few beers after dinner ?"), "Say, Jim, how about going for a few beers after dinner?");
        assert_eq!(detok("I don't . It will ( maybe ) help !"), "I don't. It will (maybe) help!");
        assert_eq!(detok("  50 %  of  it ; ok  "), "50% of it; ok");
        assert_eq!(detok("Yeah , you \u{2019} re big . Frank \u{2019} s here , ' s"), "Yeah, you're big. Frank's here, ' s");
        assert_eq!(detok("he said ' hello ' to me"), "he said ' hello ' to me");
    }

    #[test]
    fn cornell_threads_conversations() {
        let lines = tmp(
            "lines.txt",
            b"L3 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ Not  me.\nL2 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ Who?\nL1 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ Caf\xe9 time\nL9 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ \nL8 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ Alone\n",
        );
        let convs = tmp(
            "convs.txt",
            b"u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L1', 'L2', 'L3']\nu0 +++$+++ u2 +++$+++ m0 +++$+++ ['L8', 'L9']\nu2 +++$+++ u0 +++$+++ m0 +++$+++ ['L2', 'L3']\n",
        );
        let mut docs = Vec::new();
        let st = cornell(&lines, &convs, None, |d| {
            docs.push(d.to_string());
            Ok(())
        })
        .unwrap();
        assert_eq!(docs, vec!["A: Caf\u{e9} time\nB: Who?\nA: Not me.".to_string(), "A: Who?\nB: Not me.".to_string()]);
        assert_eq!(st.rows, 3);
        assert_eq!(st.docs, 2);
        assert_eq!(st.turns, 5);
        assert_eq!(st.too_short, 1);
        assert_eq!(st.missing_turns, 1);
        let mut n = 0;
        let st = cornell(&lines, &convs, Some(1), |_| {
            n += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!((st.rows, n), (1, 1));
    }

    #[test]
    fn dailydialog_rebuilds_dialogues_from_triples() {
        let p = tmp("tri.tsv", b"a1 ?\tb1 .\ta2 !\nb1 .\ta2 !\tb2\nx1\tx2\tx3\na1 ?\tb1 .\ta2 !\nb1 .\ta2 !\tb2\nWho?\tNot  me.\tWhy\n");
        // a Cornell file holding two of the last row's three turns
        let lines = tmp("known.txt", b"L3 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ Not me.\nL2 +++$+++ u2 +++$+++ m0 +++$+++ CAMERON +++$+++ Who?\n");
        let known = cornell_line_set(&lines).unwrap();
        assert_eq!(known.len(), 2);
        let mut docs = Vec::new();
        let st = dailydialog_triples(&p, None, &known, |d| {
            docs.push(d.to_string());
            Ok(())
        })
        .unwrap();
        assert_eq!(docs, vec!["A: a1?\nB: b1.\nA: a2!\nB: b2".to_string(), "A: x1\nB: x2\nA: x3".to_string()]);
        assert_eq!(st.rows, 6);
        assert_eq!(st.repeats, 1);
        assert_eq!(st.known, 1);
        assert_eq!(st.turns, 7);
    }

    #[test]
    fn histogram_buckets() {
        let h = histogram(&[1, 2, 2, 5, 9], &[2, 5]);
        assert_eq!(h, vec![("0-1".to_string(), 1), ("2-4".to_string(), 2), ("5+".to_string(), 2)]);
    }
}
