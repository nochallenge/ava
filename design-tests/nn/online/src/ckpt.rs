// TESTING / DESIGN ONLY
//! K6 + G2 + G6: the checkpoint registry of the always-online learner.
//!
//! Folder layout (ROOT is the learner's own folder, private, never published):
//! - `ROOT/ckpt/NNNNNN.safetensors`: every brain the learner made, copied in once and never changed;
//! - `ROOT/registry.log`: the record log (see `log.rs`); the registry's state is rebuilt from it;
//! - `ROOT/CURRENT`: one line, `gen<TAB>ckpt/NNNNNN.safetensors`, the live brain. The live server
//!   watches this file and hot-swaps when it changes. It is written atomically after each promote or
//!   rollback; if it ever disagrees with the log, the log wins and `CURRENT` is rewritten.
//!
//! Records: `cand gen parent hash note episodes` (episodes = the ids it learned from, comma
//! separated), `live gen why` (promote or rollback), `reject gen why`, `taint gen why`.
//!
//! Lineage: a brain trained from another inherits what that one learned. When episodes are erased
//! (G6), every brain that learned from them, and every brain trained from such a brain, is tainted:
//! its file is deleted and it can never be live again; if the live brain is tainted, the registry
//! rolls back to the newest clean brain that was live before.

use crate::log::{atomic_copy, atomic_write, hash_file, Log};
use crate::{fields, rec, safet};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct Ckpt {
    pub gen: u64,
    pub parent: Option<u64>,
    pub hash: u64,
    pub note: String,
    pub episodes: Vec<String>,
    pub rejected: Option<String>,
    pub tainted: Option<String>,
}

pub struct Registry {
    pub root: PathBuf,
    log: Log,
    pub ckpts: BTreeMap<u64, Ckpt>,
    pub live: Option<u64>,
    /// Every brain that has been live, in order (a rollback adds the brain again).
    pub live_history: Vec<u64>,
    /// Bytes the log cut from a torn end when it was opened.
    pub repaired_bytes: u64,
}

pub fn ckpt_rel(gen: u64) -> String {
    format!("ckpt/{gen:06}.safetensors")
}

impl Registry {
    pub fn open(root: &Path) -> Result<Registry, String> {
        std::fs::create_dir_all(root.join("ckpt")).map_err(|e| e.to_string())?;
        let o = Log::open(&root.join("registry.log"))?;
        let mut r = Registry {
            root: root.to_path_buf(),
            log: o.log,
            ckpts: BTreeMap::new(),
            live: None,
            live_history: Vec::new(),
            repaired_bytes: o.cut_bytes,
        };
        for raw in &o.records {
            let f = fields(raw);
            r.apply(&f.iter().map(String::as_str).collect::<Vec<_>>())?;
        }
        r.write_current()?;
        Ok(r)
    }

    fn apply(&mut self, f: &[&str]) -> Result<(), String> {
        let num = |i: usize| -> Result<u64, String> { f.get(i).and_then(|x| x.parse().ok()).ok_or(format!("bad record {f:?}")) };
        match f.first().copied() {
            Some("cand") if f.len() == 6 => {
                let gen = num(1)?;
                let parent = if f[2] == "-" { None } else { Some(num(2)?) };
                let hash = u64::from_str_radix(f[3], 16).map_err(|_| format!("bad hash in {f:?}"))?;
                let episodes = f[5].split(',').filter(|s| !s.is_empty()).map(String::from).collect();
                self.ckpts.insert(gen, Ckpt { gen, parent, hash, note: f[4].to_string(), episodes, rejected: None, tainted: None });
            }
            Some("live") if f.len() == 3 => {
                let gen = num(1)?;
                self.live = Some(gen);
                self.live_history.push(gen);
            }
            Some("reject") if f.len() == 3 => {
                let gen = num(1)?;
                if let Some(c) = self.ckpts.get_mut(&gen) {
                    c.rejected = Some(f[2].to_string());
                }
            }
            Some("taint") if f.len() == 3 => {
                let gen = num(1)?;
                if let Some(c) = self.ckpts.get_mut(&gen) {
                    c.tainted = Some(f[2].to_string());
                }
            }
            _ => return Err(format!("unknown record {f:?}")),
        }
        Ok(())
    }

    fn put(&mut self, f: &[&str]) -> Result<(), String> {
        self.log.append(&rec(f))?;
        self.apply(f)
    }

    pub fn path(&self, gen: u64) -> PathBuf {
        self.root.join(ckpt_rel(gen))
    }

    /// Rewrites CURRENT to match the log (only when it differs).
    fn write_current(&self) -> Result<(), String> {
        let p = self.root.join("CURRENT");
        let want = match self.live {
            Some(g) => format!("{g}\t{}\n", ckpt_rel(g)),
            None => String::new(),
        };
        if std::fs::read_to_string(&p).unwrap_or_default() != want {
            atomic_write(&p, want.as_bytes())?;
        }
        Ok(())
    }

    /// Copies a brain file in as a new candidate. `episodes` are the ids it learned from.
    pub fn add_candidate(&mut self, src: &Path, parent: Option<u64>, note: &str, episodes: &[String]) -> Result<u64, String> {
        if let Some(p) = parent {
            if !self.ckpts.contains_key(&p) {
                return Err(format!("parent {p} is unknown"));
            }
        }
        let gen = self.ckpts.keys().next_back().map_or(1, |g| g + 1);
        let hash = atomic_copy(src, &self.path(gen))?;
        let parent_s = parent.map_or("-".to_string(), |p| p.to_string());
        let eps = episodes.join(",");
        self.put(&["cand", &gen.to_string(), &parent_s, &format!("{hash:016x}"), note, &eps])?;
        Ok(gen)
    }

    /// True when the file on disk is the one that was registered.
    pub fn verify(&self, gen: u64) -> bool {
        match self.ckpts.get(&gen) {
            Some(c) => hash_file(&self.path(gen)).map(|h| h == c.hash).unwrap_or(false),
            None => false,
        }
    }

    fn usable(&self, gen: u64) -> Result<&Ckpt, String> {
        let c = self.ckpts.get(&gen).ok_or(format!("brain {gen} is unknown"))?;
        if let Some(w) = &c.tainted {
            return Err(format!("brain {gen} is tainted ({w})"));
        }
        if !self.verify(gen) {
            return Err(format!("brain {gen}: file is missing or changed"));
        }
        Ok(c)
    }

    /// Makes a candidate the live brain. Unless `migration` is set, it must still read the live
    /// brain's codes (F4).
    pub fn promote(&mut self, gen: u64, migration: bool) -> Result<(), String> {
        let c = self.usable(gen)?;
        if let Some(w) = &c.rejected {
            return Err(format!("brain {gen} was rejected ({w})"));
        }
        if let (Some(live), false) = (self.live, migration) {
            let p = safet::problems(&safet::read_header(&self.path(live))?, &safet::read_header(&self.path(gen))?);
            if !p.is_empty() {
                return Err(format!("brain {gen} cannot read brain {live}'s codes: {}", p.join("; ")));
            }
        }
        self.put(&["live", &gen.to_string(), "promote"])?;
        self.write_current()
    }

    pub fn reject(&mut self, gen: u64, why: &str) -> Result<(), String> {
        if !self.ckpts.contains_key(&gen) {
            return Err(format!("brain {gen} is unknown"));
        }
        self.put(&["reject", &gen.to_string(), why])?;
        // A rejected candidate is never used again, so its file only takes space.
        let _ = std::fs::remove_file(self.path(gen));
        Ok(())
    }

    /// Goes back to a brain that was live before.
    pub fn rollback(&mut self, to: u64) -> Result<(), String> {
        if !self.live_history.contains(&to) {
            return Err(format!("brain {to} was never live"));
        }
        self.usable(to)?;
        self.put(&["live", &to.to_string(), "rollback"])?;
        self.write_current()
    }

    /// G6: taints every brain that learned from any of `erased`, and everything trained from it;
    /// deletes their files; rolls the live brain back if needed. Returns the newly tainted brains.
    pub fn taint_by_episodes(&mut self, erased: &BTreeSet<String>) -> Result<Vec<u64>, String> {
        let mut bad: BTreeSet<u64> = self.ckpts.values().filter(|c| c.tainted.is_some()).map(|c| c.gen).collect();
        let mut newly = Vec::new();
        let gens: Vec<u64> = self.ckpts.keys().copied().collect();
        for g in gens {
            let c = &self.ckpts[&g];
            if c.tainted.is_some() {
                continue;
            }
            let from_parent = c.parent.is_some_and(|p| bad.contains(&p));
            if from_parent || c.episodes.iter().any(|e| erased.contains(e)) {
                bad.insert(g);
                newly.push(g);
                let why = if from_parent { "trained from a tainted brain" } else { "learned from erased episodes" };
                self.put(&["taint", &g.to_string(), why])?;
                let _ = std::fs::remove_file(self.path(g));
            }
        }
        if let Some(live) = self.live {
            if bad.contains(&live) {
                let clean = self.live_history.iter().rev().copied().find(|g| !bad.contains(g) && self.verify(*g));
                match clean {
                    Some(g) => self.rollback(g)?,
                    None => return Err("no clean brain is left to go back to".into()),
                }
            }
        }
        Ok(newly)
    }

    /// The chain of brains from the first one to `gen`.
    pub fn lineage(&self, gen: u64) -> Vec<u64> {
        let mut v = vec![gen];
        let mut g = gen;
        while let Some(p) = self.ckpts.get(&g).and_then(|c| c.parent) {
            v.push(p);
            g = p;
        }
        v.reverse();
        v
    }
}

/// Reads ROOT/CURRENT: (gen, path of the live brain).
pub fn read_current(root: &Path) -> Option<(u64, PathBuf)> {
    let s = std::fs::read_to_string(root.join("CURRENT")).ok()?;
    let (g, p) = s.trim_end().split_once('\t')?;
    Some((g.parse().ok()?, root.join(p)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::log::test_dir;
    use crate::safet::write_dummy;

    fn brain(d: &Path, name: &str, extra: bool) -> PathBuf {
        let p = d.join(name);
        if extra {
            write_dummy(&p, &[("emb", &[8, 4]), ("engram", &[2, 4])]).unwrap();
        } else {
            write_dummy(&p, &[("emb", &[8, 4])]).unwrap();
        }
        p
    }

    #[test]
    fn promote_reject_rollback_and_rebuild_from_the_log() {
        let d = test_dir("ckpt");
        let root = d.join("learner");
        let mut r = Registry::open(&root).unwrap();
        let g1 = r.add_candidate(&brain(&d, "a", false), None, "import", &[]).unwrap();
        r.promote(g1, false).unwrap();
        assert_eq!(read_current(&root).unwrap().0, g1);
        let g2 = r.add_candidate(&brain(&d, "b", true), Some(g1), "day 1", &["e1".into()]).unwrap();
        r.promote(g2, false).unwrap();
        let g3 = r.add_candidate(&brain(&d, "c", false), Some(g2), "day 2", &["e2".into()]).unwrap();
        // Brain 3 lost the engram, so it cannot read brain 2's codes.
        assert!(r.promote(g3, false).unwrap_err().contains("engram is missing"));
        r.reject(g3, "gate: numbers dropped").unwrap();
        assert!(r.promote(g3, false).is_err());
        r.rollback(g1).unwrap();
        assert_eq!(r.live, Some(g1));
        assert_eq!(r.lineage(g3), vec![g1, g2, g3]);
        drop(r);
        // Rebuilt from the log alone, with CURRENT damaged.
        std::fs::write(root.join("CURRENT"), "junk").unwrap();
        let r = Registry::open(&root).unwrap();
        assert_eq!(r.live, Some(g1));
        assert_eq!(r.live_history, vec![g1, g2, g1]);
        assert_eq!(r.ckpts[&g3].rejected.as_deref(), Some("gate: numbers dropped"));
        assert_eq!(read_current(&root).unwrap().0, g1);
    }

    #[test]
    fn a_changed_file_never_goes_live() {
        let d = test_dir("ckpt_verify");
        let mut r = Registry::open(&d.join("l")).unwrap();
        let g = r.add_candidate(&brain(&d, "a", false), None, "import", &[]).unwrap();
        let p = r.path(g);
        let mut b = std::fs::read(&p).unwrap();
        *b.last_mut().unwrap() ^= 1;
        std::fs::write(&p, b).unwrap();
        assert!(r.promote(g, false).unwrap_err().contains("missing or changed"));
    }

    #[test]
    fn erasure_taints_children_and_rolls_back() {
        let d = test_dir("ckpt_taint");
        let mut r = Registry::open(&d.join("l")).unwrap();
        let g1 = r.add_candidate(&brain(&d, "a", false), None, "import", &[]).unwrap();
        r.promote(g1, false).unwrap();
        let g2 = r.add_candidate(&brain(&d, "b", false), Some(g1), "", &["secret".into()]).unwrap();
        r.promote(g2, false).unwrap();
        let g3 = r.add_candidate(&brain(&d, "c", false), Some(g2), "", &["other".into()]).unwrap();
        r.promote(g3, false).unwrap();
        let erased: BTreeSet<String> = ["secret".to_string()].into();
        let t = r.taint_by_episodes(&erased).unwrap();
        assert_eq!(t, vec![g2, g3]);
        assert_eq!(r.live, Some(g1));
        assert!(!r.path(g2).exists() && !r.path(g3).exists());
        assert!(r.rollback(g3).is_err());
        // Running it again finds nothing new.
        assert!(r.taint_by_episodes(&erased).unwrap().is_empty());
    }
}
