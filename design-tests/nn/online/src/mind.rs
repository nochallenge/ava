// TESTING / DESIGN ONLY
//! Her own mind, small versions (Wave 4): the parts that need no GPU. Each one chooses or gathers
//! material from her memories. Writing in her own words is the gpu side's job (the think bin), and
//! what she writes there is tagged imagined, so it is never learned as fact.
//!
//! - F8 reflection: which of today's moments to think back on. Corrections first, then what
//!   surprised her, then what people thanked her for.
//! - F11 autobiography: the plain facts of her life so far, from her own logs: when she met each
//!   person, how much she has talked and read, what she was taught, and how each skill has grown.
//! - F14 self-posed questions (retrieval practice): she quizzes herself on what she was taught. The
//!   due-est memories go first, i.e. those she is most likely to be losing. Answering a question from
//!   memory strengthens a memory more than reading it again (the testing effect).
//! - B13 a model of each person, kept separate per person (C4): how much they talk with her, what
//!   about, how often they correct or thank her. It helps her pitch replies; it is never shared
//!   with anyone else.
//! - J8 teaching console: lessons from a `question<TAB>answer` file become taught episodes.

use crate::content_words;
use crate::drives::Competence;
use crate::episodes::{Episode, Origin, Store};
use crate::replay::{need, recall_prob};
use std::collections::{BTreeMap, BTreeSet};

// ------------------------------------------------------------------------------------------- F8

/// Why a moment is worth thinking back on.
#[derive(Clone, Debug, PartialEq)]
pub enum Notable {
    /// Someone corrected her (explicitly, or "no, it's X").
    Corrected,
    /// Her reply was doubted (a plain "no", or the same question asked again).
    Doubted,
    /// It surprised her (mean surprise in nats per token).
    Surprising(f32),
    /// Someone thanked her or marked the reply good.
    Thanked,
}

/// The day's moments worth reflecting on, best first: (episode id, why). `t0..t1` is the day.
/// Her dreams and daydreams are left out: reflection is on what happened.
pub fn reflect_pick(store: &Store, t0: i64, t1: i64, n: usize) -> Vec<(String, Notable)> {
    let mut scored: Vec<(f64, String, Notable)> = Vec::new();
    for e in store.eps.values().filter(|e| e.t0 >= t0 && e.t0 < t1) {
        let why = match e.origin {
            Origin::Taught => Some((3.0, Notable::Corrected)),
            Origin::Said if e.trust <= 0.2 => Some((2.0, Notable::Doubted)),
            Origin::Said if e.trust >= 0.7 => Some((0.5, Notable::Thanked)),
            Origin::Heard if e.surprise > 0.0 => Some((1.0 + e.surprise as f64 / 10.0, Notable::Surprising(e.surprise))),
            _ => None,
        };
        if let Some((s, w)) = why {
            scored.push((s, e.id.clone(), w));
        }
    }
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.into_iter().take(n).map(|(_, id, w)| (id, w)).collect()
}

// ------------------------------------------------------------------------------------------- F11

fn day_of(t: i64) -> String {
    let (y, m, d) = crate::clock::civil_from_days(t.div_euclid(86_400));
    format!("{y:04}-{m:02}-{d:02}")
}

/// The facts of her life so far, one line each, oldest first. `brains_live` is how many brains have
/// been live (the registry's history), `born` the time of her first memory if known.
pub fn autobiography(store: &Store, comp: &Competence, brains_live: usize) -> Vec<String> {
    let mut out = Vec::new();
    let eps: Vec<&Episode> = store.eps.values().filter(|e| !matches!(e.origin, Origin::Dreamt | Origin::Imagined)).collect();
    if let Some(first) = eps.iter().map(|e| e.t0).filter(|&t| t > 0).min() {
        out.push(format!("My first memory is from {}.", day_of(first)));
    }
    let mut met: BTreeMap<&str, i64> = BTreeMap::new();
    let mut talks: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    for e in &eps {
        if !e.user.is_empty() && e.origin == Origin::Heard {
            let m = met.entry(&e.user).or_insert(e.t0);
            *m = (*m).min(e.t0);
            talks.entry(&e.user).or_default().insert(day_of(e.t0));
        }
    }
    let mut people: Vec<(&str, i64)> = met.into_iter().collect();
    people.sort_by_key(|p| p.1);
    for (who, t) in &people {
        out.push(format!("I first talked with {who} on {}, and we have talked on {} days.", day_of(*t), talks[who].len()));
    }
    let count = |o: Origin| eps.iter().filter(|e| e.origin == o).count();
    let (heard, taught, read) = (count(Origin::Heard), count(Origin::Taught), count(Origin::Read));
    if heard + taught + read > 0 {
        out.push(format!("I have been asked {heard} things, been taught {taught}, and read {read} passages that I remember."));
    }
    if brains_live > 1 {
        out.push(format!("My brain has been updated {} times while I kept talking.", brains_live - 1));
    }
    out.extend(comp.lines().into_iter().map(|l| format!("My {l}.")));
    out
}

// ------------------------------------------------------------------------------------------- F14

#[derive(Clone, Debug, PartialEq)]
pub struct SelfQuiz {
    pub id: String,
    pub question: String,
    pub answer: String,
    /// Her chance of still knowing it now (lower = more due).
    pub recall: f64,
}

/// Splits "Question: ...\nAnswer: ..." text.
pub fn split_qa(text: &str) -> Option<(String, String)> {
    let (q, a) = text.split_once("\nAnswer: ")?;
    let q = q.strip_prefix("Question: ")?;
    (!q.trim().is_empty() && !a.trim().is_empty()).then(|| (q.trim().to_string(), a.trim().to_string()))
}

/// Up to `n` self-questions from what she was taught (shared lessons, or `user`'s own), the ones she
/// is most likely losing first; among equals, the ones needed most.
pub fn self_questions(store: &Store, user: &str, now: i64, n: usize) -> Vec<SelfQuiz> {
    let mut qs: Vec<(f64, f64, SelfQuiz)> = store
        .eps
        .values()
        .filter(|e| e.origin == Origin::Taught && (e.user.is_empty() || e.user == user) && e.has_text())
        .filter_map(|e| {
            let (question, answer) = split_qa(&e.text)?;
            let last = if e.last_replay > 0 { e.last_replay } else { e.t1 };
            let r = recall_prob(e.strength_days, ((now - last) as f64 / 86_400.0).max(0.0));
            Some((r, need(&e.uses, now), SelfQuiz { id: e.id.clone(), question, answer, recall: r }))
        })
        .collect();
    qs.sort_by(|a, b| a.0.total_cmp(&b.0).then(b.1.total_cmp(&a.1)).then(a.2.id.cmp(&b.2.id)));
    qs.into_iter().take(n).map(|q| q.2).collect()
}

// ------------------------------------------------------------------------------------------- B13

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Person {
    pub user: String,
    pub turns: usize,
    pub days: usize,
    pub first: i64,
    pub last: i64,
    /// Share of her replies they doubted or corrected.
    pub corrected: f64,
    /// Share of her replies they thanked her for.
    pub thanked: f64,
    /// Mean words per message.
    pub words: f64,
    /// Their most frequent content words (topics), most frequent first.
    pub topics: Vec<String>,
}

/// One person's model, from their episodes only.
pub fn person_model(store: &Store, user: &str) -> Person {
    let mut p = Person { user: user.into(), first: i64::MAX, ..Default::default() };
    let (mut said, mut low, mut high, mut words) = (0usize, 0usize, 0usize, 0usize);
    let mut days = BTreeSet::new();
    let mut topic: BTreeMap<String, usize> = BTreeMap::new();
    for e in store.eps.values().filter(|e| e.user == user) {
        match e.origin {
            Origin::Heard => {
                p.turns += 1;
                p.first = p.first.min(e.t0);
                p.last = p.last.max(e.t0);
                days.insert(e.t0.div_euclid(86_400));
                let q = split_qa(&e.text).map_or(e.text.clone(), |(q, _)| q);
                words += q.split_whitespace().count();
                for w in content_words(&q) {
                    if w.len() >= 3 && !w.chars().any(|c| c.is_ascii_digit()) {
                        *topic.entry(w).or_default() += 1;
                    }
                }
            }
            Origin::Said => {
                said += 1;
                low += (e.trust <= 0.2) as usize;
                high += (e.trust >= 0.7) as usize;
            }
            _ => {}
        }
    }
    if p.turns == 0 {
        p.first = 0;
    }
    p.days = days.len();
    p.corrected = low as f64 / said.max(1) as f64;
    p.thanked = high as f64 / said.max(1) as f64;
    p.words = words as f64 / p.turns.max(1) as f64;
    let mut t: Vec<(String, usize)> = topic.into_iter().filter(|x| x.1 >= 2).collect();
    t.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    p.topics = t.into_iter().take(5).map(|x| x.0).collect();
    p
}

impl Person {
    /// Plain lines she can be given as context when talking with this person (and only them).
    pub fn lines(&self) -> Vec<String> {
        if self.turns == 0 {
            return vec![format!("I have not talked with {} before.", self.user)];
        }
        let mut v = vec![format!("I have talked with {} on {} days ({} messages, about {:.0} words each).", self.user, self.days, self.turns, self.words)];
        if !self.topics.is_empty() {
            v.push(format!("They often ask about: {}.", self.topics.join(", ")));
        }
        v.push(format!("They corrected or doubted {:.0}% of my replies and thanked me for {:.0}%.", self.corrected * 100.0, self.thanked * 100.0));
        v
    }
}

// ------------------------------------------------------------------------------------------- J8

/// Lessons from a `question<TAB>answer` file (a first line `question<TAB>answer` is a header) as
/// taught episodes. Shared lessons have no user. Ids are `lesson/{name}#{line}`, so importing the
/// same file again adds nothing new. Bad lines are counted, not fatal.
pub fn lessons(name: &str, text: &str, user: &str, t: i64) -> (Vec<Episode>, usize) {
    let (mut out, mut bad) = (Vec::new(), 0);
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() || (i == 0 && line.to_lowercase().starts_with("question\t")) {
            continue;
        }
        match line.split_once('\t') {
            Some((q, a)) if !q.trim().is_empty() && !a.trim().is_empty() => {
                let src = format!("lesson {name}, line {}", i + 1);
                out.push(Episode::new(&format!("lesson/{name}#{}", i + 1), user, Origin::Taught, &src, t, t, &crate::diary::qa(q, a), 1.0));
            }
            _ => bad += 1,
        }
    }
    (out, bad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log::test_dir;

    const DAY: i64 = 86_400;

    fn store(tag: &str) -> Store {
        Store::open(&test_dir(tag).join("episodes.log")).unwrap()
    }

    fn ep(id: &str, user: &str, o: Origin, t: i64, text: &str, trust: f32) -> Episode {
        Episode::new(id, user, o, "test", t, t, text, trust)
    }

    #[test]
    fn reflection_puts_corrections_first_and_skips_dreams() {
        let mut s = store("mind-reflect");
        let t = 20_754 * DAY;
        s.add(ep("a", "u", Origin::Said, t + 10, "x", 0.9)).unwrap();
        s.add(ep("b", "u", Origin::Taught, t + 20, "Question: q\nAnswer: a", 1.0)).unwrap();
        s.add(ep("c", "u", Origin::Said, t + 30, "y", 0.1)).unwrap();
        let mut h = ep("d", "u", Origin::Heard, t + 40, "z", 0.6);
        h.surprise = 4.0;
        s.add(h).unwrap();
        s.add(ep("e", "", Origin::Dreamt, t + 50, "dream", 0.0)).unwrap();
        s.add(ep("f", "u", Origin::Taught, t + DAY + 1, "Question: q\nAnswer: a", 1.0)).unwrap();
        let ids: Vec<String> = reflect_pick(&s, t, t + DAY, 10).into_iter().map(|x| x.0).collect();
        assert_eq!(ids, vec!["b", "c", "d", "a"]);
    }

    #[test]
    fn autobiography_tells_firsts_and_counts() {
        let mut s = store("mind-life");
        let t = 20_754 * DAY;
        s.add(ep("h1", "sam", Origin::Heard, t + 5, "Question: hi\nAnswer: hello", 0.6)).unwrap();
        s.add(ep("h2", "sam", Origin::Heard, t + DAY, "Question: hi again\nAnswer: hello", 0.6)).unwrap();
        s.add(ep("l1", "", Origin::Taught, t + 9, "Question: q\nAnswer: a", 1.0)).unwrap();
        s.add(ep("d1", "", Origin::Dreamt, t - DAY, "a dream", 0.0)).unwrap();
        let mut c = Competence::default();
        c.note("numbers", t, 0.5);
        let life = autobiography(&s, &c, 3);
        assert_eq!(life[0], "My first memory is from 2026-10-28.");
        assert_eq!(life[1], "I first talked with sam on 2026-10-28, and we have talked on 2 days.");
        assert_eq!(life[2], "I have been asked 2 things, been taught 1, and read 0 passages that I remember.");
        assert_eq!(life[3], "My brain has been updated 2 times while I kept talking.");
        assert!(life[4].starts_with("My numbers: 50% now"));
    }

    #[test]
    fn self_questions_put_the_fading_ones_first_and_respect_walls() {
        let mut s = store("mind-quiz");
        let now = 20_760 * DAY;
        let mut old = ep("old", "", Origin::Taught, now - 9 * DAY, "Question: What is 7 times 8?\nAnswer: 56.", 1.0);
        old.strength_days = 2.0;
        let mut fresh = ep("fresh", "", Origin::Taught, now - DAY, "Question: Why is the sky blue?\nAnswer: Air scatters blue light most.", 1.0);
        fresh.strength_days = 30.0;
        s.add(old).unwrap();
        s.add(fresh).unwrap();
        s.add(ep("theirs", "kim", Origin::Taught, now - 9 * DAY, "Question: My cat?\nAnswer: Tom.", 1.0)).unwrap();
        s.add(ep("noqa", "", Origin::Taught, now, "just text", 1.0)).unwrap();
        let q = self_questions(&s, "sam", now, 5);
        assert_eq!(q.iter().map(|x| x.id.as_str()).collect::<Vec<_>>(), vec!["old", "fresh"]);
        assert_eq!((q[0].question.as_str(), q[0].answer.as_str()), ("What is 7 times 8?", "56."));
        assert!(q[0].recall < 0.05 && q[1].recall > 0.9);
        assert_eq!(self_questions(&s, "kim", now, 5).len(), 3);
    }

    #[test]
    fn person_model_counts_only_that_person() {
        let mut s = store("mind-person");
        let t = 20_754 * DAY;
        for (i, q) in ["Tell me about planets", "Which planets have rings", "How big are planets"].iter().enumerate() {
            s.add(ep(&format!("h{i}"), "sam", Origin::Heard, t + i as i64 * DAY, &crate::diary::qa(q, "..."), 0.6)).unwrap();
            s.add(ep(&format!("s{i}"), "sam", Origin::Said, t + i as i64 * DAY, "...", [0.9, 0.1, 0.3][i])).unwrap();
        }
        s.add(ep("other", "kim", Origin::Heard, t, &crate::diary::qa("planets planets planets", "..."), 0.6)).unwrap();
        let p = person_model(&s, "sam");
        assert_eq!((p.turns, p.days), (3, 3));
        assert_eq!(p.topics, vec!["planets"]);
        assert!((p.corrected - 1.0 / 3.0).abs() < 1e-9 && (p.thanked - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(p.lines()[1], "They often ask about: planets.");
        assert_eq!(person_model(&s, "nobody").lines(), vec!["I have not talked with nobody before."]);
    }

    #[test]
    fn lessons_import_once_and_count_bad_lines() {
        let text = "question\tanswer\nWhat is 2 and 2?\t4.\nno tab here\n\nWhy?\tBecause.\n";
        let (eps, bad) = lessons("lessons_001", text, "", 100);
        assert_eq!((eps.len(), bad), (2, 1));
        assert_eq!(eps[0].id, "lesson/lessons_001#2");
        assert_eq!(split_qa(&eps[0].text), Some(("What is 2 and 2?".into(), "4.".into())));
        let mut s = store("mind-lessons");
        assert!(s.add(eps[0].clone()).unwrap());
        assert!(!s.add(eps[0].clone()).unwrap());
    }
}
