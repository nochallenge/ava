// TESTING / DESIGN ONLY
//! Memory of experiences (Wave 4): what happened, to whom, from where, and how sure she can be.
//!
//! - L1 events: `surprise_cuts` cuts a token stream where her surprise jumps (EM-LLM); `gap_cuts`
//!   cuts turns into conversations at long pauses. Recall also brings back time neighbours.
//! - F2 / C2 origin tags: every episode says whether she heard it, said it, read it, was taught it,
//!   dreamt it or imagined it. Dreamt and imagined text is never recalled or learned as fact, and her
//!   own words are capped in replay (see `replay.rs`).
//! - C4 per-user walls: an episode belongs to one person (or to nobody, for books and lessons).
//!   Recall for one person never returns another person's episodes.
//! - C7 sources: every recalled episode names where it came from (a diary day and turn, a book).
//! - C1 trust: what may go into her weights is gated by trust and capped per person per day, so no
//!   one person can flood her with planted "facts".
//! - G4 tiers: verbatim text, then a gist when the text budget is full (lowest value first, and
//!   episodes already learned into the weights first), with the weights as the last tier.
//! - G6 erasure: erasing an episode or a person removes the text from this store for good (the log
//!   is rewritten without it), returns the ids so the registry can taint brains that learned them,
//!   and refuses to take the same episodes in again. The diary itself is append-only by design;
//!   erasing from the diary is the owner's own decision and is not done here.
//!
//! All state lives in a record log (`ROOT/episodes.log`), so it survives crashes and can be rebuilt.

use crate::log::Log;
use crate::{content_words, fields, rec};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Origin {
    Heard,
    Said,
    Read,
    Taught,
    Dreamt,
    Imagined,
}

impl Origin {
    pub const ALL: [Origin; 6] = [Origin::Heard, Origin::Said, Origin::Read, Origin::Taught, Origin::Dreamt, Origin::Imagined];
    pub fn name(self) -> &'static str {
        match self {
            Origin::Heard => "heard",
            Origin::Said => "said",
            Origin::Read => "read",
            Origin::Taught => "taught",
            Origin::Dreamt => "dreamt",
            Origin::Imagined => "imagined",
        }
    }
    pub fn parse(s: &str) -> Option<Origin> {
        Origin::ALL.into_iter().find(|o| o.name() == s)
    }
    /// Her own words (C2).
    pub fn is_hers(self) -> bool {
        matches!(self, Origin::Said | Origin::Dreamt | Origin::Imagined)
    }
    /// Only things from outside her can become facts (F2).
    pub fn can_be_fact(self) -> bool {
        matches!(self, Origin::Heard | Origin::Read | Origin::Taught)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// The text is kept and not yet learned.
    Verbatim,
    /// Only a short gist is kept.
    Gist,
    /// Learned into the weights (the text may still be kept until the budget needs the room).
    Weights,
}

impl Tier {
    fn name(self) -> &'static str {
        match self {
            Tier::Verbatim => "verbatim",
            Tier::Gist => "gist",
            Tier::Weights => "weights",
        }
    }
    fn parse(s: &str) -> Option<Tier> {
        [Tier::Verbatim, Tier::Gist, Tier::Weights].into_iter().find(|t| t.name() == s)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Episode {
    pub id: String,
    /// The person it belongs to; empty for shared things (books, lessons, her dreams).
    pub user: String,
    pub origin: Origin,
    /// Where it came from, in words she can say (e.g. "diary 2026-10-28, turn 3").
    pub source: String,
    pub t0: i64,
    pub t1: i64,
    pub text: String,
    pub gist: String,
    /// Mean surprise (nats per token) when it happened, if known.
    pub surprise: f32,
    /// 0 to 1.
    pub trust: f32,
    pub tier: Tier,
    pub replays: u32,
    /// How long (days) until she would likely forget it (see `replay.rs`).
    pub strength_days: f64,
    pub last_replay: i64,
    /// Times it was recalled or rehearsed (for "need").
    pub uses: Vec<i64>,
}

impl Episode {
    pub fn new(id: &str, user: &str, origin: Origin, source: &str, t0: i64, t1: i64, text: &str, trust: f32) -> Episode {
        Episode {
            id: id.into(),
            user: user.into(),
            origin,
            source: source.into(),
            t0,
            t1,
            text: text.into(),
            gist: String::new(),
            surprise: 0.0,
            trust,
            tier: Tier::Verbatim,
            replays: 0,
            strength_days: 1.0,
            last_replay: 0,
            uses: Vec::new(),
        }
    }
    pub fn has_text(&self) -> bool {
        !self.text.is_empty()
    }
    /// The text if kept, else the gist.
    pub fn shown(&self) -> &str {
        if self.has_text() {
            &self.text
        } else {
            &self.gist
        }
    }
    fn record(&self) -> Vec<u8> {
        let uses: Vec<String> = self.uses.iter().map(|u| u.to_string()).collect();
        rec(&[
            "ep",
            &self.id,
            &self.user,
            self.origin.name(),
            &self.source,
            &self.t0.to_string(),
            &self.t1.to_string(),
            &self.surprise.to_string(),
            &self.trust.to_string(),
            self.tier.name(),
            &self.text,
            &self.gist,
            &self.replays.to_string(),
            &self.strength_days.to_string(),
            &self.last_replay.to_string(),
            &uses.join(","),
        ])
    }
}

/// A default gist: the first sentence, at most 160 characters.
pub fn first_sentence_gist(text: &str) -> String {
    let t = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let end = t.find(['.', '?', '!']).map(|i| i + 1).unwrap_or(t.len());
    let s: String = t[..end].chars().take(160).collect();
    if s.len() < t.len() {
        format!("{s} …")
    } else {
        s
    }
}

/// Why an episode may not go into her weights (C1, F2).
#[derive(Clone, Debug)]
pub struct TrustPolicy {
    pub min_trust: f32,
    /// At most this many episodes per person per day go into the weights.
    pub per_user_daily_cap: usize,
}

impl Default for TrustPolicy {
    fn default() -> Self {
        TrustPolicy { min_trust: 0.5, per_user_daily_cap: 200 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hit {
    pub id: String,
    pub source: String,
    pub when: i64,
    pub text: String,
    pub score: f64,
    /// Brought back because it happened right before or after a hit (L1).
    pub neighbour: bool,
}

pub struct Store {
    log: Option<Log>,
    pub eps: BTreeMap<String, Episode>,
    pub erased_ids: BTreeSet<String>,
    /// Person -> erased up to this time (their later episodes are welcome).
    pub erased_users: BTreeMap<String, i64>,
    pub repaired_bytes: u64,
}

impl Store {
    pub fn open(path: &Path) -> Result<Store, String> {
        let o = Log::open(path)?;
        let mut s = Store { log: Some(o.log), eps: BTreeMap::new(), erased_ids: BTreeSet::new(), erased_users: BTreeMap::new(), repaired_bytes: o.cut_bytes };
        for r in &o.records {
            let f = fields(r);
            s.apply(&f.iter().map(String::as_str).collect::<Vec<_>>())?;
        }
        Ok(s)
    }

    fn apply(&mut self, f: &[&str]) -> Result<(), String> {
        let bad = || format!("bad record {:?}", f.iter().take(3).collect::<Vec<_>>());
        let num = |i: usize| -> Result<i64, String> { f.get(i).and_then(|x| x.parse().ok()).ok_or_else(bad) };
        let flt = |i: usize| -> Result<f64, String> { f.get(i).and_then(|x| x.parse().ok()).ok_or_else(bad) };
        match f.first().copied() {
            Some("ep") if f.len() == 16 => {
                let ep = Episode {
                    id: f[1].into(),
                    user: f[2].into(),
                    origin: Origin::parse(f[3]).ok_or_else(bad)?,
                    source: f[4].into(),
                    t0: num(5)?,
                    t1: num(6)?,
                    surprise: flt(7)? as f32,
                    trust: flt(8)? as f32,
                    tier: Tier::parse(f[9]).ok_or_else(bad)?,
                    text: f[10].into(),
                    gist: f[11].into(),
                    replays: num(12)? as u32,
                    strength_days: flt(13)?,
                    last_replay: num(14)?,
                    uses: f[15].split(',').filter_map(|x| x.parse().ok()).collect(),
                };
                self.eps.insert(ep.id.clone(), ep);
            }
            Some("trust") if f.len() == 3 => {
                let v = flt(2)? as f32;
                if let Some(e) = self.eps.get_mut(f[1]) {
                    e.trust = v;
                }
            }
            Some("surprise") if f.len() == 3 => {
                let v = flt(2)? as f32;
                if let Some(e) = self.eps.get_mut(f[1]) {
                    e.surprise = v;
                }
            }
            Some("gist") if f.len() == 3 => {
                if let Some(e) = self.eps.get_mut(f[1]) {
                    e.gist = f[2].into();
                    e.text.clear();
                    if e.tier == Tier::Verbatim {
                        e.tier = Tier::Gist;
                    }
                }
            }
            Some("weights") if f.len() == 2 => {
                if let Some(e) = self.eps.get_mut(f[1]) {
                    e.tier = Tier::Weights;
                }
            }
            Some("replay") if f.len() == 4 => {
                let (t, s) = (num(2)?, flt(3)?);
                if let Some(e) = self.eps.get_mut(f[1]) {
                    e.replays += 1;
                    e.last_replay = t;
                    e.strength_days = s;
                    e.uses.push(t);
                }
            }
            Some("use") if f.len() == 3 => {
                let t = num(2)?;
                if let Some(e) = self.eps.get_mut(f[1]) {
                    e.uses.push(t);
                }
            }
            Some("erase") if f.len() == 2 => {
                self.eps.remove(f[1]);
                self.erased_ids.insert(f[1].into());
            }
            Some("erase-user") if f.len() == 3 => {
                let t = num(2)?;
                self.eps.retain(|_, e| !(e.user == f[1] && e.t0 <= t));
                let v = self.erased_users.entry(f[1].into()).or_insert(t);
                *v = (*v).max(t);
            }
            _ => return Err(bad()),
        }
        Ok(())
    }

    fn put(&mut self, f: &[&str]) -> Result<(), String> {
        self.log.as_mut().ok_or("store is closed")?.append(&rec(f))?;
        self.apply(f)
    }

    /// Adds an episode. Returns false (and adds nothing) if it is already there or was erased.
    pub fn add(&mut self, ep: Episode) -> Result<bool, String> {
        if self.eps.contains_key(&ep.id) || self.erased_ids.contains(&ep.id) {
            return Ok(false);
        }
        if self.erased_users.get(&ep.user).is_some_and(|&t| ep.t0 <= t) {
            return Ok(false);
        }
        self.log.as_mut().ok_or("store is closed")?.append(&ep.record())?;
        self.eps.insert(ep.id.clone(), ep);
        Ok(true)
    }

    pub fn set_trust(&mut self, id: &str, trust: f32) -> Result<(), String> {
        match self.eps.get(id) {
            Some(e) if (e.trust - trust).abs() > 1e-6 => self.put(&["trust", id, &trust.to_string()]),
            _ => Ok(()),
        }
    }
    /// Records how much an episode surprised her (nats per token), measured after the fact.
    pub fn set_surprise(&mut self, id: &str, nats: f32) -> Result<(), String> {
        match self.eps.get(id) {
            Some(_) if nats.is_finite() && nats > 0.0 => self.put(&["surprise", id, &nats.to_string()]),
            _ => Ok(()),
        }
    }
    pub fn mark_learned(&mut self, id: &str) -> Result<(), String> {
        self.put(&["weights", id])
    }
    pub fn note_replay(&mut self, id: &str, t: i64, strength_days: f64) -> Result<(), String> {
        self.put(&["replay", id, &t.to_string(), &strength_days.to_string()])
    }
    pub fn note_use(&mut self, id: &str, t: i64) -> Result<(), String> {
        self.put(&["use", id, &t.to_string()])
    }

    /// Rewrites the log from the current state, so dropped text is gone from disk too.
    pub fn compact(&mut self) -> Result<(), String> {
        let mut recs: Vec<Vec<u8>> = Vec::new();
        for (u, t) in &self.erased_users {
            recs.push(rec(&["erase-user", u, &t.to_string()]));
        }
        for id in &self.erased_ids {
            recs.push(rec(&["erase", id]));
        }
        recs.extend(self.eps.values().map(Episode::record));
        let log = self.log.take().ok_or("store is closed")?;
        self.log = Some(log.rewrite(&recs)?);
        Ok(())
    }

    /// G6: erases one episode for good. Returns whether it existed.
    pub fn erase(&mut self, id: &str) -> Result<bool, String> {
        let had = self.eps.contains_key(id);
        self.put(&["erase", id])?;
        self.compact()?;
        Ok(had)
    }

    /// G6: erases everything a person said or was told up to time `t`. Returns the erased ids.
    pub fn erase_user(&mut self, user: &str, t: i64) -> Result<Vec<String>, String> {
        if user.is_empty() {
            return Err("give a person's name (shared memories are not a person's)".into());
        }
        let ids: Vec<String> = self.eps.values().filter(|e| e.user == user && e.t0 <= t).map(|e| e.id.clone()).collect();
        self.put(&["erase-user", user, &t.to_string()])?;
        for id in &ids {
            self.erased_ids.insert(id.clone());
        }
        self.compact()?;
        Ok(ids)
    }

    /// C1 + F2: may this episode go into her weights now? `learned_today` = this person's episodes
    /// already learned today.
    pub fn may_learn(&self, e: &Episode, p: &TrustPolicy, learned_today: usize) -> Result<(), String> {
        if !e.has_text() {
            return Err("only a gist is left".into());
        }
        if !e.origin.can_be_fact() && e.origin != Origin::Said {
            return Err(format!("{} text is never learned as fact", e.origin.name()));
        }
        if e.trust < p.min_trust {
            return Err(format!("trust {:.2} is below {:.2}", e.trust, p.min_trust));
        }
        if !e.user.is_empty() && learned_today >= p.per_user_daily_cap {
            return Err(format!("{} already gave {learned_today} today", e.user));
        }
        Ok(())
    }

    /// C4 + C7 + L1: the episodes that best match `query` for this person (their own and shared
    /// ones, never anyone else's), each naming its source, plus their time neighbours.
    /// Dreamt and imagined episodes are left out: they are not memories of the world.
    pub fn recall(&self, user: &str, query: &str, n: usize, neighbours: bool) -> Vec<Hit> {
        let visible: Vec<&Episode> =
            self.eps.values().filter(|e| (e.user == user || e.user.is_empty()) && e.origin.can_be_fact() || (e.user == user && e.origin == Origin::Said)).collect();
        let words: Vec<Vec<String>> = visible.iter().map(|e| content_words(e.shown())).collect();
        let mut df: HashMap<&str, usize> = HashMap::new();
        for w in &words {
            let set: BTreeSet<&str> = w.iter().map(String::as_str).collect();
            for x in set {
                *df.entry(x).or_default() += 1;
            }
        }
        let q: BTreeSet<String> = content_words(query).into_iter().collect();
        let nv = visible.len().max(1) as f64;
        let mut hits: Vec<Hit> = Vec::new();
        for (e, w) in visible.iter().zip(&words) {
            let set: BTreeSet<&str> = w.iter().map(String::as_str).collect();
            let s: f64 = q.iter().filter(|x| set.contains(x.as_str())).map(|x| (1.0 + nv / df[x.as_str()] as f64).ln()).sum();
            if s > 0.0 {
                let score = s / (1.0 + (1.0 + w.len() as f64).ln());
                hits.push(Hit { id: e.id.clone(), source: e.source.clone(), when: e.t0, text: e.shown().into(), score, neighbour: false });
            }
        }
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
        hits.truncate(n);
        if neighbours {
            // The person's own timeline: what happened right before and after each hit.
            let mut line: Vec<&Episode> = visible.iter().copied().filter(|e| e.user == user).collect();
            line.sort_by_key(|e| (e.t0, e.id.clone()));
            let mut extra = Vec::new();
            for h in &hits {
                if let Some(i) = line.iter().position(|e| e.id == h.id) {
                    for j in [i.wrapping_sub(1), i + 1] {
                        if let Some(e) = line.get(j) {
                            if !hits.iter().chain(extra.iter()).any(|x: &Hit| x.id == e.id) {
                                extra.push(Hit { id: e.id.clone(), source: e.source.clone(), when: e.t0, text: e.shown().into(), score: h.score * 0.5, neighbour: true });
                            }
                        }
                    }
                }
            }
            hits.extend(extra);
        }
        hits
    }

    /// G4: when kept text is over `budget` bytes, turns the lowest-value texts into gists (episodes
    /// already learned into the weights first). The newest day is always kept whole. Returns the ids.
    pub fn plan_forgetting(&mut self, budget: usize, now: i64, gist: &dyn Fn(&str) -> String) -> Result<Vec<String>, String> {
        let mut total: usize = self.eps.values().filter(|e| e.has_text()).map(|e| e.text.len()).sum();
        if total <= budget {
            return Ok(Vec::new());
        }
        let mut c: Vec<(bool, f64, String, usize)> = self
            .eps
            .values()
            .filter(|e| e.has_text() && now - e.t1 > 86_400)
            .map(|e| (e.tier != Tier::Weights, value(e, now), e.id.clone(), e.text.len()))
            .collect();
        c.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.total_cmp(&b.1)).then(a.2.cmp(&b.2)));
        let mut done = Vec::new();
        for (_, _, id, len) in c {
            if total <= budget {
                break;
            }
            let g = gist(&self.eps[&id].text);
            self.put(&["gist", &id, &g])?;
            total -= len;
            done.push(id);
        }
        if !done.is_empty() {
            self.compact()?;
        }
        Ok(done)
    }
}

/// How much an episode is worth keeping: how likely it is to be needed (`replay::need`), times
/// how much it can be trusted, times how surprising (new) it was.
pub fn value(e: &Episode, now: i64) -> f64 {
    let mut uses = e.uses.clone();
    uses.push(e.t1);
    crate::replay::need(&uses, now) * (0.5 + e.trust as f64) * (1.0 + (e.surprise as f64).min(5.0) / 5.0)
}

/// L1: event starts in a token stream: a cut before token i when its surprise is more than `k`
/// standard deviations above the mean of the last `window` tokens of the current event, the event
/// is at least `min_len` long; and always at `max_len`. Always starts with 0.
pub fn surprise_cuts(s: &[f32], window: usize, k: f32, min_len: usize, max_len: usize) -> Vec<usize> {
    let mut cuts = vec![0usize];
    let mut start = 0usize;
    for i in 1..s.len() {
        let len = i - start;
        if len >= max_len {
            cuts.push(i);
            start = i;
            continue;
        }
        if len < min_len {
            continue;
        }
        let w = &s[i.saturating_sub(window).max(start)..i];
        let mean = w.iter().sum::<f32>() / w.len() as f32;
        let var = w.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / w.len() as f32;
        if s[i] > mean + k * var.sqrt().max(1e-3) {
            cuts.push(i);
            start = i;
        }
    }
    cuts
}

/// L1 for conversations: a new event starts after a pause longer than `gap` seconds.
pub fn gap_cuts(times: &[i64], gap: i64) -> Vec<usize> {
    let mut cuts = Vec::new();
    for i in 0..times.len() {
        if i == 0 || times[i] - times[i - 1] > gap {
            cuts.push(i);
        }
    }
    cuts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log::test_dir;

    fn ep(id: &str, user: &str, o: Origin, t: i64, text: &str) -> Episode {
        Episode::new(id, user, o, &format!("diary turn {id}"), t, t, text, 0.9)
    }

    #[test]
    fn people_never_see_each_others_memories() {
        let d = test_dir("eps_wall");
        let mut s = Store::open(&d.join("e.log")).unwrap();
        s.add(ep("a1", "ana", Origin::Heard, 10, "My locker code is 4417.")).unwrap();
        s.add(ep("b1", "ben", Origin::Heard, 11, "I lost my locker key.")).unwrap();
        s.add(ep("book", "", Origin::Read, 1, "A locker is a small cupboard that locks.")).unwrap();
        s.add(ep("dream", "", Origin::Dreamt, 12, "The locker code is 9999.")).unwrap();
        let h = s.recall("ben", "what is the locker code", 10, true);
        assert!(h.iter().all(|x| x.id != "a1"), "{h:?}");
        assert!(h.iter().all(|x| x.id != "dream"), "dreams are not memories: {h:?}");
        assert!(h.iter().any(|x| x.id == "book"));
        let h = s.recall("ana", "locker code", 1, false);
        assert_eq!(h[0].id, "a1");
        assert_eq!(h[0].source, "diary turn a1");
    }

    #[test]
    fn time_neighbours_come_back_with_a_hit() {
        let d = test_dir("eps_nb");
        let mut s = Store::open(&d.join("e.log")).unwrap();
        s.add(ep("t1", "ana", Origin::Heard, 100, "We went to the zoo.")).unwrap();
        s.add(ep("t2", "ana", Origin::Heard, 200, "A giraffe ate from my hand.")).unwrap();
        s.add(ep("t3", "ana", Origin::Heard, 300, "Then it started to rain.")).unwrap();
        let h = s.recall("ana", "giraffe", 1, true);
        let ids: Vec<&str> = h.iter().map(|x| x.id.as_str()).collect();
        assert_eq!(ids, vec!["t2", "t1", "t3"]);
        assert!(h[1].neighbour && h[2].neighbour);
    }

    #[test]
    fn erasure_removes_text_from_disk_and_refuses_it_again() {
        let d = test_dir("eps_erase");
        let p = d.join("e.log");
        let mut s = Store::open(&p).unwrap();
        s.add(ep("a1", "ana", Origin::Heard, 10, "zebra-secret-123")).unwrap();
        s.add(ep("a2", "ana", Origin::Heard, 20, "another-private-thing")).unwrap();
        s.add(ep("b1", "ben", Origin::Heard, 15, "ben stays")).unwrap();
        s.note_use("a1", 30).unwrap();
        let gone = s.erase_user("ana", 25).unwrap();
        assert_eq!(gone.len(), 2);
        assert!(!s.add(ep("a1", "ana", Origin::Heard, 10, "zebra-secret-123")).unwrap());
        assert!(s.add(ep("a3", "ana", Origin::Heard, 40, "later and welcome")).unwrap());
        drop(s);
        let raw = std::fs::read(&p).unwrap();
        for secret in [b"zebra-secret".as_slice(), b"another-private"] {
            assert!(!raw.windows(secret.len()).any(|w| w == secret));
        }
        let s = Store::open(&p).unwrap();
        assert_eq!(s.eps.keys().cloned().collect::<Vec<_>>(), vec!["a3", "b1"]);
        assert!(s.erased_ids.contains("a1"));
    }

    #[test]
    fn trust_gate_and_per_person_cap() {
        let s = Store::open(&test_dir("eps_trust").join("e.log")).unwrap();
        let p = TrustPolicy { min_trust: 0.5, per_user_daily_cap: 2 };
        let mut e = ep("x", "ana", Origin::Heard, 1, "Paris is in France.");
        assert!(s.may_learn(&e, &p, 0).is_ok());
        assert!(s.may_learn(&e, &p, 2).unwrap_err().contains("today"));
        e.trust = 0.2;
        assert!(s.may_learn(&e, &p, 0).unwrap_err().contains("trust"));
        let d = ep("d", "", Origin::Dreamt, 1, "Cats can fly.");
        assert!(s.may_learn(&d, &p, 0).unwrap_err().contains("never learned"));
    }

    #[test]
    fn forgetting_keeps_the_newest_day_and_drops_learned_text_first() {
        let d = test_dir("eps_forget");
        let mut s = Store::open(&d.join("e.log")).unwrap();
        let day = 86_400;
        let now = 100 * day;
        s.add(ep("old_learned", "ana", Origin::Heard, 10 * day, "aaaaaaaaaa. more")).unwrap();
        s.add(ep("old", "ana", Origin::Heard, 11 * day, "bbbbbbbbbb. more")).unwrap();
        s.add(ep("new", "ana", Origin::Heard, now - 10, "cccccccccc. more")).unwrap();
        s.mark_learned("old_learned").unwrap();
        let g = s.plan_forgetting(35, now, &first_sentence_gist).unwrap();
        assert_eq!(g, vec!["old_learned"]);
        assert_eq!(s.eps["old_learned"].shown(), "aaaaaaaaaa. …");
        assert_eq!(s.eps["old_learned"].tier, Tier::Weights);
        let g = s.plan_forgetting(10, now, &first_sentence_gist).unwrap();
        assert_eq!(g, vec!["old"]);
        assert_eq!(s.eps["old"].tier, Tier::Gist);
        assert!(s.eps["new"].has_text());
    }

    #[test]
    fn surprise_and_gap_cuts() {
        let mut s = vec![1.0f32; 40];
        s[20] = 9.0;
        assert_eq!(surprise_cuts(&s, 10, 2.0, 5, 100), vec![0, 20]);
        assert_eq!(surprise_cuts(&s, 10, 2.0, 5, 15), vec![0, 15, 20, 35]);
        assert_eq!(gap_cuts(&[0, 10, 20, 5000, 5010], 1800), vec![0, 3]);
    }
}
