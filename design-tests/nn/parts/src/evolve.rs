// TESTING / DESIGN ONLY
//! H2: evolving her design with an archive of variants (open-ended search as in the Darwin Gödel Machine,
//! Zhang et al. 2025, and AlphaEvolve). Brains were not designed in one go; many variants were tried and
//! the good ones kept and varied again. Here a variant is a choice for each design option (the gpu
//! crate's `arch` flags and these parts' switches), scored by a short proxy run. Every variant ever scored
//! stays in the archive, not only the best, because a weaker one can be the parent of a later winner.
//!
//! - Parent choice: weight = e^(−rank/3) / (1 + children), rank 0 = the best so far. The children term
//!   is DGM's: good variants are used often but not forever, and other branches get their turn. (DGM's
//!   own sigmoid around the median treated the whole top half alike and did no better than random
//!   search in the test below, so it was replaced.)
//! - A child is either a crossover of two parents (each option from one or the other, then maybe one
//!   change) or one parent with one or two options changed. Crossover joins improvements found on
//!   different branches. Variants already scored or already proposed are never proposed again (proxy
//!   runs are expensive; the archive makes the search resumable).
//! - Scores are "higher is better" (pass −BPB for a language-model proxy).
//! - The archive is a TSV file: `id<TAB>parent<TAB>score<TAB>cost<TAB>name=value ...`, append-only.

use std::collections::BTreeMap;

/// The design options: name → allowed values.
pub type Menu = Vec<(String, Vec<String>)>;
pub type Variant = BTreeMap<String, String>;

#[derive(Clone, Debug)]
pub struct Entry {
    pub id: usize,
    pub parent: Option<usize>,
    pub variant: Variant,
    pub score: Option<f64>,
    pub cost: f64,
}

pub struct Archive {
    pub menu: Menu,
    pub entries: Vec<Entry>,
    rng: u64,
}

pub fn key(v: &Variant) -> String {
    v.iter().map(|(k, x)| format!("{k}={x}")).collect::<Vec<_>>().join(" ")
}

impl Archive {
    /// A new archive whose first entry is the base design (the first value of every option).
    pub fn new(menu: Menu, seed: u64) -> Archive {
        let base: Variant = menu.iter().map(|(k, vals)| (k.clone(), vals[0].clone())).collect();
        Archive { menu, entries: vec![Entry { id: 0, parent: None, variant: base, score: None, cost: 0.0 }], rng: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1 }
    }

    fn next(&mut self) -> u64 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        self.rng
    }

    fn unif(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn record(&mut self, id: usize, score: f64, cost: f64) {
        let e = &mut self.entries[id];
        e.score = Some(score);
        e.cost = cost;
    }

    pub fn best(&self) -> Option<&Entry> {
        self.entries.iter().filter(|e| e.score.is_some()).max_by(|a, b| a.score.unwrap().total_cmp(&b.score.unwrap()))
    }

    fn parent_weights(&self) -> Vec<(usize, f64)> {
        let mut scored: Vec<(usize, f64)> = self.entries.iter().filter_map(|e| Some((e.id, e.score?))).collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scored
            .iter()
            .enumerate()
            .map(|(rank, &(id, _))| {
                let children = self.entries.iter().filter(|c| c.parent == Some(id)).count();
                (id, (-(rank as f64) / 3.0).exp() / (1.0 + children as f64))
            })
            .collect()
    }

    fn pick(&mut self, w: &[(usize, f64)]) -> usize {
        if w.is_empty() {
            return 0;
        }
        let total: f64 = w.iter().map(|x| x.1).sum();
        let mut r = self.unif() * total;
        for &(id, x) in w {
            if r < x {
                return id;
            }
            r -= x;
        }
        w[w.len() - 1].0
    }

    fn mutate(&mut self, v: &mut Variant) {
        let o = (self.next() % self.menu.len() as u64) as usize;
        let (name, vals) = self.menu[o].clone();
        let x = vals[(self.next() % vals.len() as u64) as usize].clone();
        v.insert(name, x);
    }

    /// Up to `n` new variants to run next (fewer if the menu is nearly used up). Each is added to the
    /// archive unscored; record its score when its run is done.
    pub fn propose(&mut self, n: usize) -> Vec<usize> {
        let mut out = Vec::new();
        let mut tries = 0;
        while out.len() < n && tries < 200 * n.max(1) {
            tries += 1;
            let w = self.parent_weights();
            let parent = self.pick(&w);
            let mut v = self.entries[parent].variant.clone();
            if w.len() >= 2 && self.unif() < 0.5 {
                let other = self.pick(&w);
                let o = self.entries[other].variant.clone();
                for (k, x) in o {
                    if self.unif() < 0.5 {
                        v.insert(k, x);
                    }
                }
                if self.unif() < 0.5 {
                    self.mutate(&mut v);
                }
            } else {
                let changes = 1 + (self.next() % 2) as usize;
                for _ in 0..changes {
                    self.mutate(&mut v);
                }
            }
            let k = key(&v);
            if self.entries.iter().any(|e| key(&e.variant) == k) {
                continue;
            }
            let id = self.entries.len();
            self.entries.push(Entry { id, parent: Some(parent), variant: v, score: None, cost: 0.0 });
            out.push(id);
        }
        out
    }

    pub fn to_tsv(&self) -> String {
        let mut s = String::new();
        for e in &self.entries {
            let p = e.parent.map_or("-".into(), |p| p.to_string());
            let sc = e.score.map_or("-".into(), |x| x.to_string());
            s.push_str(&format!("{}\t{p}\t{sc}\t{}\t{}\n", e.id, e.cost, key(&e.variant)));
        }
        s
    }

    pub fn from_tsv(menu: Menu, seed: u64, text: &str) -> Result<Archive, String> {
        let mut a = Archive::new(menu, seed);
        a.entries.clear();
        for (n, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() != 5 || f[0].parse::<usize>().ok() != Some(n) {
                return Err(format!("archive line {}: bad", n + 1));
            }
            let variant = f[4].split(' ').filter_map(|kv| kv.split_once('=')).map(|(k, v)| (k.to_string(), v.to_string())).collect();
            a.entries.push(Entry { id: n, parent: f[1].parse().ok(), variant, score: f[2].parse().ok(), cost: f[3].parse().map_err(|_| "bad cost")? });
        }
        // Advance the generator past the entries already made, so a resumed search does not repeat.
        for _ in 0..a.entries.len() * 7 {
            a.next();
        }
        Ok(a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn menu() -> Menu {
        let opt = |n: &str, v: &[&str]| (n.to_string(), v.iter().map(|x| x.to_string()).collect());
        vec![
            opt("delays", &["off", "on"]),
            opt("wm_slots", &["0", "4", "8", "16"]),
            opt("halting", &["off", "on"]),
            opt("stp", &["off", "on"]),
            opt("mix", &["a", "b", "c"]),
            opt("lr", &["1e-3", "2e-3", "4e-3"]),
        ]
    }

    /// A made-up proxy score with an interaction: halting only helps with working memory of 8+ slots.
    fn proxy(v: &Variant) -> f64 {
        let mut s = 0.0;
        s += if v["delays"] == "on" { 0.02 } else { 0.0 };
        s += match v["wm_slots"].as_str() {
            "8" => 0.03,
            "16" => 0.02,
            "4" => 0.01,
            _ => 0.0,
        };
        let big_wm = v["wm_slots"] == "8" || v["wm_slots"] == "16";
        s += if v["halting"] == "on" { if big_wm { 0.05 } else { -0.03 } } else { 0.0 };
        s += if v["stp"] == "on" { -0.01 } else { 0.0 };
        s += if v["mix"] == "b" { 0.015 } else { 0.0 };
        s += if v["lr"] == "2e-3" { 0.01 } else { 0.0 };
        -1.5 + s
    }

    fn search(seed: u64, runs: usize) -> f64 {
        let mut a = Archive::new(menu(), seed);
        let base = a.entries[0].variant.clone();
        a.record(0, proxy(&base), 1.0);
        while a.entries.len() < runs {
            for id in a.propose(4) {
                let v = a.entries[id].variant.clone();
                a.record(id, proxy(&v), 1.0);
            }
        }
        a.best().unwrap().score.unwrap()
    }

    /// The same number of distinct designs, drawn at random (the base design included).
    fn random_search(seed: u64, runs: usize) -> f64 {
        let mut a = Archive::new(menu(), seed);
        let mut seen: std::collections::BTreeSet<String> = [key(&a.entries[0].variant)].into();
        let mut best = proxy(&a.entries[0].variant);
        while seen.len() < runs {
            let v: Variant = a.menu.clone().iter().map(|(k, vals)| (k.clone(), vals[(a.next() % vals.len() as u64) as usize].clone())).collect();
            if seen.insert(key(&v)) {
                best = best.max(proxy(&v));
            }
        }
        best
    }

    #[test]
    fn the_archive_finds_the_interaction_and_never_repeats() {
        let best = -1.5 + 0.02 + 0.03 + 0.05 + 0.015 + 0.01;
        for runs in [40, 80, 120] {
            let (mut arch, mut rand, mut hits, mut rhits) = (0.0, 0.0, 0, 0);
            for s in 0..20 {
                let (x, y) = (search(s, runs), random_search(1000 + s, runs));
                arch += x / 20.0;
                rand += y / 20.0;
                hits += ((x - best).abs() < 1e-9) as usize;
                rhits += ((y - best).abs() < 1e-9) as usize;
            }
            eprintln!("{runs} proxy runs of 288 designs: archive gap to best {:.4} ({hits}/20 reach it), random {:.4} ({rhits}/20)", best - arch, best - rand);
            if runs >= 80 {
                assert!(best - arch < 0.5 * (best - rand) && hits > rhits, "{arch} {rand}");
            }
        }
        let mut a = Archive::new(menu(), 3);
        a.record(0, -1.5, 1.0);
        let ids = a.propose(30);
        let keys: std::collections::BTreeSet<String> = a.entries.iter().map(|e| key(&e.variant)).collect();
        assert_eq!(keys.len(), a.entries.len());
        assert_eq!(ids.len(), 30);
    }

    #[test]
    fn the_archive_file_round_trips_and_resumes() {
        let mut a = Archive::new(menu(), 5);
        a.record(0, -1.5, 2.0);
        for id in a.propose(5) {
            let v = a.entries[id].variant.clone();
            a.record(id, proxy(&v), 1.5);
        }
        let t = a.to_tsv();
        let mut b = Archive::from_tsv(menu(), 5, &t).unwrap();
        assert_eq!(b.to_tsv(), t);
        let new = b.propose(5);
        assert!(new.iter().all(|&i| b.entries[i].score.is_none()));
        let keys: std::collections::BTreeSet<String> = b.entries.iter().map(|e| key(&e.variant)).collect();
        assert_eq!(keys.len(), b.entries.len());
    }
}
