// TESTING / DESIGN ONLY
//! F5: standing goals over days. People carry intentions for days ("ask her how the exam went", "learn
//! to count past 100") and remember them when something brings them to mind (prospective memory:
//! Einstein & McDaniel). Unfinished tasks stay active (the Zeigarnik effect) and goals compete for time.
//!
//! - A goal has an owner (a person, or "" for her own), words, cue words that bring it to mind, an
//!   optional due time, a value (0..1) and progress (0..1).
//! - `cued`: goals whose cue words appear in what was just said (the "when X, do Y" part).
//! - `pull`: how strongly a goal asks for attention now, graded, never a switch:
//!     pull = value · (1 − progress) · (urgency + cue)
//!     urgency = 1/(1 + days left) with a due time, else 0.2·(1 − e^(−days since touched / 7))
//!     cue = share of its cue words just heard (0 if none)
//! - Stored as a checksummed append-only log (`crate::log`), so nothing is lost or half-written.

use crate::log::Log;
use crate::{content_words, fields, rec};
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct Goal {
    pub id: String,
    pub owner: String,
    pub text: String,
    pub cues: Vec<String>,
    pub created: i64,
    pub due: Option<i64>,
    pub value: f64,
    pub progress: f64,
    pub touched: i64,
    pub done: Option<i64>,
}

pub struct Goals {
    log: Log,
    pub goals: BTreeMap<String, Goal>,
}

impl Goals {
    pub fn open(path: &Path) -> Result<Goals, String> {
        let o = Log::open(path)?;
        let mut g = Goals { log: o.log, goals: BTreeMap::new() };
        for r in &o.records {
            let f = fields(r);
            g.apply(&f.iter().map(String::as_str).collect::<Vec<_>>())?;
        }
        Ok(g)
    }

    fn apply(&mut self, f: &[&str]) -> Result<(), String> {
        let bad = || format!("bad goal record {:?}", f.iter().take(2).collect::<Vec<_>>());
        let num = |i: usize| -> Result<i64, String> { f.get(i).and_then(|x| x.parse().ok()).ok_or_else(bad) };
        let flt = |i: usize| -> Result<f64, String> { f.get(i).and_then(|x| x.parse().ok()).ok_or_else(bad) };
        match f.first().copied() {
            Some("goal") if f.len() == 8 => {
                let due = if f[6].is_empty() { None } else { Some(num(6)?) };
                let cues = f[4].split(' ').filter(|w| !w.is_empty()).map(String::from).collect();
                let g = Goal { id: f[1].into(), owner: f[2].into(), text: f[3].into(), cues, created: num(5)?, due, value: flt(7)?, progress: 0.0, touched: num(5)?, done: None };
                self.goals.insert(g.id.clone(), g);
            }
            Some("progress") if f.len() == 4 => {
                if let Some(g) = self.goals.get_mut(f[1]) {
                    g.progress = flt(2)?.clamp(0.0, 1.0);
                    g.touched = num(3)?;
                }
            }
            Some("done") if f.len() == 3 => {
                if let Some(g) = self.goals.get_mut(f[1]) {
                    g.progress = 1.0;
                    g.done = Some(num(2)?);
                    g.touched = num(2)?;
                }
            }
            Some("drop") if f.len() == 2 => {
                self.goals.remove(f[1]);
            }
            _ => return Err(bad()),
        }
        Ok(())
    }

    fn write(&mut self, f: &[&str]) -> Result<(), String> {
        self.apply(f)?;
        self.log.append(&rec(f))
    }

    /// Adds a goal; its id is "g{n}". Cue words are taken from `cues` (content words only).
    pub fn add(&mut self, owner: &str, text: &str, cues: &str, due: Option<i64>, value: f64, t: i64) -> Result<String, String> {
        let id = format!("g{}", self.goals.keys().filter_map(|k| k.strip_prefix('g')?.parse::<u64>().ok()).max().map_or(1, |n| n + 1));
        let cues = content_words(cues).join(" ");
        let due = due.map(|d| d.to_string()).unwrap_or_default();
        self.write(&["goal", &id, owner, text, &cues, &t.to_string(), &due, &value.clamp(0.0, 1.0).to_string()])?;
        Ok(id)
    }

    pub fn progress(&mut self, id: &str, p: f64, t: i64) -> Result<(), String> {
        self.known(id)?;
        self.write(&["progress", id, &p.to_string(), &t.to_string()])
    }

    pub fn done(&mut self, id: &str, t: i64) -> Result<(), String> {
        self.known(id)?;
        self.write(&["done", id, &t.to_string()])
    }

    pub fn drop_goal(&mut self, id: &str) -> Result<(), String> {
        self.known(id)?;
        self.write(&["drop", id])
    }

    fn known(&self, id: &str) -> Result<(), String> {
        if self.goals.contains_key(id) {
            Ok(())
        } else {
            Err(format!("no goal {id}"))
        }
    }

    pub fn open_goals(&self) -> impl Iterator<Item = &Goal> {
        self.goals.values().filter(|g| g.done.is_none())
    }
}

/// The share of a goal's cue words found in `text` (0 when it has none).
pub fn cue(g: &Goal, text: &str) -> f64 {
    if g.cues.is_empty() {
        return 0.0;
    }
    let words = content_words(text);
    g.cues.iter().filter(|c| words.iter().any(|w| w == *c)).count() as f64 / g.cues.len() as f64
}

pub fn urgency(g: &Goal, now: i64) -> f64 {
    match g.due {
        Some(d) => 1.0 / (1.0 + ((d - now) as f64 / 86_400.0).max(0.0)),
        None => 0.2 * (1.0 - (-((now - g.touched) as f64 / 86_400.0).max(0.0) / 7.0).exp()),
    }
}

pub fn pull(g: &Goal, now: i64, heard: &str) -> f64 {
    if g.done.is_some() {
        return 0.0;
    }
    g.value * (1.0 - g.progress) * (urgency(g, now) + cue(g, heard))
}

/// Open goals by pull, strongest first, for one person (and her own goals).
pub fn arbitrate<'a>(goals: &'a Goals, user: &str, now: i64, heard: &str) -> Vec<(&'a Goal, f64)> {
    let mut v: Vec<(&Goal, f64)> = goals.open_goals().filter(|g| g.owner.is_empty() || g.owner == user).map(|g| (g, pull(g, now, heard))).collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.id.cmp(&b.0.id)));
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log::test_dir;

    const DAY: i64 = 86_400;

    #[test]
    fn goals_persist_come_to_mind_on_cue_and_compete() {
        let d = test_dir("goals");
        let p = d.join("goals.log");
        let t0 = 1_800_000_000;
        {
            let mut g = Goals::open(&p).unwrap();
            let a = g.add("sam", "ask how the exam went", "exam test", None, 0.8, t0).unwrap();
            let b = g.add("", "learn to count past 100", "count numbers", Some(t0 + 10 * DAY), 0.6, t0).unwrap();
            let c = g.add("kim", "send the recipe", "recipe", None, 0.9, t0).unwrap();
            assert_eq!((a.as_str(), b.as_str(), c.as_str()), ("g1", "g2", "g3"));
            g.progress("g2", 0.5, t0 + DAY).unwrap();
            assert!(g.progress("g9", 0.5, t0).is_err());
        }
        // Reopened from the log: the same state.
        let mut g = Goals::open(&p).unwrap();
        assert_eq!(g.goals.len(), 3);
        assert_eq!(g.goals["g2"].progress, 0.5);
        // Sam mentions the exam: that goal comes to mind and leads; Kim's goal is never shown to Sam.
        let now = t0 + 2 * DAY;
        let r = arbitrate(&g, "sam", now, "I have a big exam tomorrow");
        assert_eq!(r.iter().map(|(g, _)| g.id.as_str()).collect::<Vec<_>>(), vec!["g1", "g2"]);
        assert!(r[0].1 > 2.0 * r[1].1);
        // With nothing said, the unfinished goal still pulls a little, and more as days pass.
        let quiet = pull(&g.goals["g1"], now, "");
        assert!(quiet > 0.0 && quiet < pull(&g.goals["g1"], now + 20 * DAY, ""));
        // A due date makes a goal more urgent as it nears.
        assert!(urgency(&g.goals["g2"], t0 + 9 * DAY) > 2.0 * urgency(&g.goals["g2"], t0 + DAY));
        // Done goals pull nothing; dropped goals are gone after reopening too.
        g.done("g1", now).unwrap();
        g.drop_goal("g3").unwrap();
        assert_eq!(pull(&g.goals["g1"], now, "exam"), 0.0);
        let g = Goals::open(&p).unwrap();
        assert_eq!(g.goals.len(), 2);
        assert_eq!(g.open_goals().count(), 1);
        let _ = std::fs::remove_dir_all(d);
    }
}
