// TESTING / DESIGN ONLY
//! F6: drives that choose what the learner does next, and F11: what she can and cannot do yet.
//!
//! She is always online: talking is never scheduled here. The live server answers at once, whatever
//! the learner is doing in the background. The learner's own choices:
//! - memory pressure (new episodes waiting) -> Consolidate (train a candidate on a replay batch);
//! - curiosity (grows with idle time) -> Think about recent episodes (F1) if there are some new
//!   ones, else Dream (B11, sample freely, tagged as dreamt);
//! - fatigue (each GPU job adds some; idle time takes it away) -> Rest: no background compute.
//!   It keeps her from running the GPU all day; the owner sets how fast it builds.
//! GPU work only runs when `gpu_share` says the GPU is free.

use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activity {
    Rest,
    Consolidate,
    Think,
    Dream,
}

#[derive(Clone, Debug)]
pub struct Rates {
    /// Memory pressure per new episode (1.0 = one batch is waiting).
    pub per_episode: f64,
    pub curiosity_per_hour: f64,
    pub fatigue_per_job: f64,
    pub rest_per_hour: f64,
}

impl Default for Rates {
    fn default() -> Self {
        Rates { per_episode: 1.0 / 64.0, curiosity_per_hour: 0.5, fatigue_per_job: 0.25, rest_per_hour: 0.5 }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Drives {
    pub memory: f64,
    pub curiosity: f64,
    pub fatigue: f64,
}

impl Drives {
    /// Time passed (hours) and new episodes since the last tick.
    pub fn tick(&mut self, hours: f64, new_episodes: usize, r: &Rates) {
        self.memory += new_episodes as f64 * r.per_episode;
        self.curiosity += hours * r.curiosity_per_hour;
        self.fatigue = (self.fatigue - hours * r.rest_per_hour).max(0.0);
    }

    pub fn choose(&self, gpu_free: bool, fresh_to_think: bool) -> Activity {
        if self.fatigue >= 1.0 || !gpu_free {
            Activity::Rest
        } else if self.memory >= 1.0 {
            Activity::Consolidate
        } else if self.curiosity >= 1.0 {
            if fresh_to_think {
                Activity::Think
            } else {
                Activity::Dream
            }
        } else {
            Activity::Rest
        }
    }

    pub fn done(&mut self, a: Activity, r: &Rates) {
        match a {
            Activity::Rest => {}
            Activity::Consolidate => {
                self.memory = (self.memory - 1.0).max(0.0);
                self.fatigue += r.fatigue_per_job;
            }
            Activity::Think | Activity::Dream => {
                self.curiosity = 0.0;
                self.fatigue += r.fatigue_per_job / 2.0;
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SkillLine {
    pub first: (i64, f64),
    pub last: (i64, f64),
    pub best: f64,
    pub checks: u32,
}

/// F11: her running record of each skill, from every gate check.
#[derive(Clone, Debug, Default)]
pub struct Competence {
    pub skills: BTreeMap<String, SkillLine>,
}

impl Competence {
    pub fn note(&mut self, skill: &str, t: i64, score: f64) {
        let e = self.skills.entry(skill.to_string()).or_insert(SkillLine { first: (t, score), last: (t, score), best: score, checks: 0 });
        e.last = (t, score);
        e.best = e.best.max(score);
        e.checks += 1;
    }

    /// One plain line per skill: now, at the first check, and the best so far.
    pub fn lines(&self) -> Vec<String> {
        self.skills
            .iter()
            .map(|(k, s)| {
                let (y, m, d) = crate::clock::civil_from_days(s.first.0.div_euclid(86_400));
                format!("{k}: {:.0}% now, {:.0}% on {y:04}-{m:02}-{d:02}, best {:.0}% ({} checks)", s.last.1 * 100.0, s.first.1 * 100.0, s.best * 100.0, s.checks)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drives_pick_work_and_rest() {
        let r = Rates::default();
        let mut d = Drives::default();
        d.tick(0.5, 64, &r);
        assert_eq!(d.choose(true, false), Activity::Consolidate);
        assert_eq!(d.choose(false, false), Activity::Rest);
        d.done(Activity::Consolidate, &r);
        assert_eq!(d.choose(true, false), Activity::Rest);
        d.tick(2.0, 0, &r);
        assert_eq!(d.choose(true, true), Activity::Think);
        assert_eq!(d.choose(true, false), Activity::Dream);
        d.fatigue = 1.0;
        assert_eq!(d.choose(true, true), Activity::Rest);
        d.tick(2.0, 0, &r);
        assert_eq!(d.fatigue, 0.0);
    }

    #[test]
    fn competence_remembers_first_last_and_best() {
        let mut c = Competence::default();
        let day = 86_400;
        c.note("numbers", 20_754 * day, 0.41);
        c.note("numbers", 20_755 * day, 0.65);
        c.note("numbers", 20_756 * day, 0.62);
        assert_eq!(c.lines(), vec!["numbers: 62% now, 41% on 2026-10-28, best 65% (3 checks)"]);
    }
}
