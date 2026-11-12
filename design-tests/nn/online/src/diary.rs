// TESTING / DESIGN ONLY
//! A small reader for the diary the chat writes (`gpu/src/diary.rs` has the full rules), turning
//! each turn into episodes for the learner, with the same choices `diary_day` makes:
//!
//! - your words are always kept, as something she heard (trust 0.6);
//! - her reply is kept as something she said, as "Question: ...\nAnswer: ..." (trust 0.9 if you
//!   rated it /good, 0.3 if unrated, 0 if rated /bad or corrected, so only good answers are ever
//!   learned: the trust gate's floor is 0.5);
//! - your correction is kept as something she was taught (trust 1.0): she learns your answer, not
//!   her own wrong one.
//!
//! The diary file format: a header line, then `time model turn user ava` lines (TSV; `\\ \t \n \r`
//! escaped, a turn's text starting with '/' written as `\/`). Command lines have the user field
//! `/correct` (the correction in the last field), `/good` or `/bad`, and refer to the latest turn
//! with that id above them. A last line with no newline (a write cut short) and lines closed with
//! the torn mark are skipped.

use crate::clock::parse_time;
use crate::episodes::{Episode, Origin};

pub const HEADER: &str = "time\tmodel\tturn\tuser\tava";

/// Like `unesc`, but also reads `\/` and keeps unknown escapes as they are (the diary's rule).
pub fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('\\') => out.push('\\'),
            Some('/') => out.push('/'),
            Some(o) => {
                out.push('\\');
                out.push(o);
            }
            None => out.push('\\'),
        }
    }
    out
}

#[derive(Clone, Debug, PartialEq)]
pub struct Turn {
    pub time: String,
    pub turn: u64,
    pub user: String,
    pub ava: String,
    pub correction: Option<String>,
    pub rating: Option<bool>,
}

/// The turns of one day file's bytes, resolved, and the count of skipped lines.
pub fn read_turns(bytes: &[u8]) -> (Vec<Turn>, usize) {
    let mut pieces: Vec<&[u8]> = bytes.split(|&b| b == b'\n').collect();
    let tail = pieces.pop().unwrap_or(&[]);
    let mut bad = usize::from(!tail.is_empty());
    let mut turns: Vec<Turn> = Vec::new();
    let mut latest = std::collections::HashMap::<u64, usize>::new();
    for raw in pieces {
        let raw = raw.strip_suffix(b"\r").unwrap_or(raw);
        let Ok(s) = std::str::from_utf8(raw) else {
            bad += 1;
            continue;
        };
        if s.is_empty() || s == HEADER {
            continue;
        }
        let f: Vec<&str> = s.split('\t').collect();
        let Some(id) = f.get(2).and_then(|x| x.parse::<u64>().ok()).filter(|_| f.len() == 5) else {
            bad += 1;
            continue;
        };
        match f[3] {
            "/correct" | "/good" | "/bad" => match latest.get(&id) {
                Some(&i) => match f[3] {
                    "/correct" => turns[i].correction = Some(unescape(f[4])),
                    "/good" => turns[i].rating = Some(true),
                    _ => turns[i].rating = Some(false),
                },
                None => bad += 1,
            },
            u if u.starts_with('/') => bad += 1,
            u => {
                latest.insert(id, turns.len());
                turns.push(Turn { time: unescape(f[0]), turn: id, user: unescape(u), ava: unescape(f[4]), correction: None, rating: None });
            }
        }
    }
    (turns, bad)
}

/// Whitespace runs become one space (an episode's only newline is the Question/Answer one).
pub fn one_line(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn qa(q: &str, a: &str) -> String {
    format!("Question: {}\nAnswer: {}", one_line(q), one_line(a))
}

/// All episodes of a day: each turn's, with the natural feedback of the turn after it (C3) where no
/// explicit command was given.
pub fn day_episodes(day: &str, turns: &[Turn], user: &str) -> Vec<Episode> {
    let sig = crate::feedback::signals(turns);
    let mut out = Vec::new();
    for (t, s) in turns.iter().zip(sig) {
        let mut eps = turn_episodes(day, t, user);
        let explicit = t.correction.is_some() || t.rating.is_some();
        if let (Some(s), false) = (s, explicit) {
            use crate::feedback::Signal;
            let said_trust = match &s {
                Signal::Thanks => 0.7,
                Signal::Disagree | Signal::Corrects(_) => 0.1,
                Signal::ReAsk => 0.15,
            };
            eps[1].trust = said_trust;
            if let Signal::Corrects(a) = &s {
                let mut e = eps[1].clone();
                e.id = e.id.replace(":said", ":taught-implicit");
                e.origin = Origin::Taught;
                e.text = qa(&t.user, a);
                e.trust = 0.8;
                eps.push(e);
            }
        }
        out.extend(eps);
    }
    out
}

/// The episodes of one turn for `user` (the person who talked). Ids are stable, so reading the same
/// day again finds the same episodes, and a correction that comes later adds the taught one and
/// lowers the trust of her reply.
pub fn turn_episodes(day: &str, t: &Turn, user: &str) -> Vec<Episode> {
    let time = parse_time(&t.time).unwrap_or(0);
    let base = format!("diary/{day}#{}@{}", t.turn, t.time);
    let source = format!("diary {day}, turn {}", t.turn);
    let said_trust = match (&t.correction, t.rating) {
        (Some(_), _) | (None, Some(false)) => 0.0,
        (None, Some(true)) => 0.9,
        (None, None) => 0.3,
    };
    let mut v = vec![
        Episode::new(&format!("{base}:heard"), user, Origin::Heard, &source, time, time, &one_line(&t.user), 0.6),
        Episode::new(&format!("{base}:said"), user, Origin::Said, &source, time, time, &qa(&t.user, &t.ava), said_trust),
    ];
    if let Some(c) = &t.correction {
        v.push(Episode::new(&format!("{base}:taught"), user, Origin::Taught, &source, time, time, &qa(&t.user, c), 1.0));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_turns_commands_and_torn_lines() {
        let day = "time\tmodel\tturn\tuser\tava\n\
2026-10-28T14:03:05\tv41/m\t3\tWhat is 2 and 3?\t6\n\
2026-10-28T14:03:20\tv41/m\t3\t/correct\t5\n\
2026-10-28T14:04:00\tv41/m\t4\t\\/not a command\tok\n\
2026-10-28T14:04:10\tv41/m\t4\t/good\t\n\
2026-10-28T14:04:11\tv41/m\t9\t/bad\t\n\
broken line\t\t\t\t\t#torn\n\
2026-10-28T14:05:00\tv41/m\t5\tcut sho";
        let (t, bad) = read_turns(day.as_bytes());
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].correction.as_deref(), Some("5"));
        assert_eq!(t[1].user, "/not a command");
        assert_eq!(t[1].rating, Some(true));
        // the orphan /bad, the torn line, the cut last line
        assert_eq!(bad, 3);
    }

    #[test]
    fn episodes_follow_the_diary_day_rules() {
        let t = Turn { time: "2026-10-28T14:03:05".into(), turn: 3, user: "What is 2 and 3?".into(), ava: "6".into(), correction: Some("5".into()), rating: None };
        let e = turn_episodes("2026-10-28", &t, "owner");
        assert_eq!(e.len(), 3);
        assert_eq!(e[0].origin, Origin::Heard);
        assert_eq!(e[1].trust, 0.0);
        assert_eq!(e[2].text, "Question: What is 2 and 3?\nAnswer: 5");
        assert_eq!(e[2].source, "diary 2026-10-28, turn 3");
        let good = Turn { correction: None, rating: Some(true), ..t };
        let e = turn_episodes("2026-10-28", &good, "owner");
        assert_eq!(e.len(), 2);
        assert_eq!(e[1].trust, 0.9);
    }

    #[test]
    fn natural_feedback_counts_unless_a_command_was_given() {
        let turn = |time: &str, id: u64, user: &str, ava: &str, rating: Option<bool>| Turn {
            time: time.into(),
            turn: id,
            user: user.into(),
            ava: ava.into(),
            correction: None,
            rating,
        };
        let turns = vec![
            turn("2026-10-28T10:00:00", 1, "What is 2 and 3?", "6", None),
            turn("2026-10-28T10:00:20", 2, "No, it's 5.", "Oh, 5.", None),
            turn("2026-10-28T10:01:00", 3, "What colour is grass?", "Green.", None),
            turn("2026-10-28T10:01:10", 4, "Thanks!", "You are welcome.", Some(false)),
            turn("2026-10-28T10:02:00", 5, "Who wrote it?", "Me.", Some(true)),
            turn("2026-10-28T10:02:10", 6, "No.", "Sorry.", None),
        ];
        let e = day_episodes("2026-10-28", &turns, "owner");
        let get = |id: &str| e.iter().find(|x| x.id.ends_with(id)).unwrap();
        assert_eq!(get("#1@2026-10-28T10:00:00:said").trust, 0.1);
        let taught = get("#1@2026-10-28T10:00:00:taught-implicit");
        assert_eq!((taught.trust, taught.text.as_str()), (0.8, "Question: What is 2 and 3?\nAnswer: 5"));
        assert_eq!(get("#3@2026-10-28T10:01:00:said").trust, 0.7);
        // Turn 5 was rated /good: the "No." after it does not override the command.
        assert_eq!(get("#5@2026-10-28T10:02:00:said").trust, 0.9);
    }
}
