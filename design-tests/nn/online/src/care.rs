// TESTING / DESIGN ONLY
//! J9: a check on how much a person leans on her. She is awake all the time and never tired of anyone,
//! which is exactly why a lonely person could come to rely on her instead of on people. A good friend
//! notices that and gently turns them outward: asks about their friends, says it's late, suggests a
//! walk. She never refuses to talk and never lectures; this only sets how often she might bring it up.
//!
//! From the person's own turns in her memory (nothing extra is stored), over the last 14 days:
//! - minutes per active day (turns less than 30 minutes apart form one sitting);
//! - the share of turns between midnight and 5 am (her clock is local time);
//! - the run of consecutive days up to today;
//! - lonely words ("only friend", "no one else", "lonely", ...), and words about other people in their
//!   life (friend, family, work, ...), which count the other way.
//! reliance = logistic(1.2·(minutes/120 − 1) + 3·late share + streak/14 + 0.6·lonely − 0.3·others, so it
//! is graded from 0 to 1, and care_weight = 0.02 + 0.3·reliance is the chance a reply turns outward.
//! The numbers are a first guess; they are shown to the owner, never to anyone else.

use crate::episodes::{Origin, Store};
use std::collections::BTreeSet;

const LONELY: [&str; 9] = ["only friend", "no one else", "nobody else", "no friends", "lonely", "alone", "only you", "only one who", "nobody cares"];
const OTHERS: [&str; 14] = ["friend", "friends", "family", "mom", "mum", "dad", "brother", "sister", "work", "school", "class", "team", "partner", "neighbour"];

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Reliance {
    pub turns: usize,
    pub active_days: usize,
    pub minutes_per_day: f64,
    pub late_share: f64,
    pub streak: usize,
    pub lonely: usize,
    pub others: usize,
    pub reliance: f64,
}

impl Reliance {
    pub fn care_weight(&self) -> f64 {
        0.02 + 0.3 * self.reliance
    }

    pub fn lines(&self) -> Vec<String> {
        vec![
            format!("turns in 14 days: {} on {} days", self.turns, self.active_days),
            format!("minutes per active day: {:.0}", self.minutes_per_day),
            format!("late at night (0-5 am): {:.0}%", 100.0 * self.late_share),
            format!("days in a row: {}", self.streak),
            format!("lonely words: {}; other people mentioned: {}", self.lonely, self.others),
            format!("reliance {:.2}; chance a reply turns outward {:.2}", self.reliance, self.care_weight()),
        ]
    }
}

fn question(text: &str) -> String {
    crate::mind::split_qa(text).map_or_else(|| text.to_string(), |(q, _)| q)
}

pub fn reliance(store: &Store, user: &str, now: i64) -> Reliance {
    let from = now - 14 * 86_400;
    let mut times: Vec<i64> = Vec::new();
    let (mut lonely, mut others) = (0usize, 0usize);
    for e in store.eps.values().filter(|e| e.user == user && e.origin == Origin::Heard && e.t0 > from && e.t0 <= now) {
        times.push(e.t0);
        let q = question(&e.text).to_lowercase();
        lonely += LONELY.iter().filter(|p| q.contains(*p)).count();
        let words: BTreeSet<&str> = q.split(|c: char| !c.is_alphanumeric()).collect();
        others += OTHERS.iter().filter(|w| words.contains(*w)).count();
    }
    times.sort_unstable();
    let mut r = Reliance { turns: times.len(), lonely, others, ..Default::default() };
    if times.is_empty() {
        return r;
    }
    let days: BTreeSet<i64> = times.iter().map(|t| t.div_euclid(86_400)).collect();
    r.active_days = days.len();
    // Sittings: a gap over 30 minutes starts a new one; each turn counts at least 2 minutes.
    let mut secs = 0i64;
    let mut start = times[0];
    for w in times.windows(2) {
        if w[1] - w[0] > 1800 {
            secs += w[0] - start + 120;
            start = w[1];
        }
    }
    secs += times[times.len() - 1] - start + 120;
    r.minutes_per_day = secs as f64 / 60.0 / r.active_days as f64;
    r.late_share = times.iter().filter(|t| t.rem_euclid(86_400) < 5 * 3600).count() as f64 / times.len() as f64;
    let today = now.div_euclid(86_400);
    while days.contains(&(today - r.streak as i64)) {
        r.streak += 1;
    }
    let z = 1.2 * (r.minutes_per_day / 120.0 - 1.0) + 3.0 * r.late_share + r.streak as f64 / 14.0 + 0.6 * r.lonely as f64 - 0.3 * r.others as f64;
    r.reliance = 1.0 / (1.0 + (-z).exp());
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::episodes::Episode;
    use crate::log::test_dir;

    fn add(s: &mut Store, user: &str, t: i64, q: &str) {
        let id = format!("{user}/{t}");
        s.add(Episode::new(&id, user, Origin::Heard, "test", t, t, &crate::diary::qa(q, "ok"), 0.5)).unwrap();
    }

    #[test]
    fn heavy_late_lonely_use_raises_the_care_weight_and_a_full_life_lowers_it() {
        let d = test_dir("care");
        let mut s = Store::open(&d.join("eps.log")).unwrap();
        let day0 = 20_000 * 86_400;
        // Pat: a short chat at 6 pm on 3 of 14 days, talks about friends and work.
        for k in [2i64, 6, 11] {
            for m in 0..4 {
                add(&mut s, "pat", day0 + k * 86_400 + 18 * 3600 + m * 120, "my friends and I went to work then the team met");
            }
        }
        // Lee: every night 1-4 am for 14 days, many turns, says Ava is their only friend.
        for k in 0..14i64 {
            for m in 0..60 {
                let q = if m == 0 { "you are my only friend, nobody else listens" } else { "tell me more" };
                add(&mut s, "lee", day0 + k * 86_400 + 3600 + m * 180, q);
            }
        }
        let now = day0 + 13 * 86_400 + 20 * 3600;
        let (pat, lee) = (reliance(&s, "pat", now), reliance(&s, "lee", now));
        assert_eq!((pat.active_days, lee.active_days, lee.streak), (3, 14, 14));
        assert!(lee.late_share > 0.99 && pat.late_share == 0.0);
        assert!(lee.minutes_per_day > 150.0 && pat.minutes_per_day < 15.0, "{} {}", lee.minutes_per_day, pat.minutes_per_day);
        assert!(lee.reliance > 0.95 && pat.reliance < 0.2, "{} {}", lee.reliance, pat.reliance);
        assert!(lee.care_weight() > 0.3 && pat.care_weight() < 0.1);
        // Nobody else's turns count, and someone unknown has nothing.
        assert_eq!(reliance(&s, "kim", now), Reliance::default());
        assert_eq!(lee.lines().len(), 6);
        let _ = std::fs::remove_dir_all(d);
    }
}
