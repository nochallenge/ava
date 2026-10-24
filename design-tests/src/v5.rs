//! v5 — neurons earn their memories, and a readout so she can answer.
//!
//! Every version so far assigned units by hash: `fnv1a(feature) % N`. That is fixed
//! at compile time, identical on the first exposure and the millionth, and it cannot
//! reflect anything the network has learned. Section 11.4 of the first note says the
//! biology does the opposite — Han et al 2007, neurons that are more EXCITABLE at the
//! moment of learning win a competition to be recruited into the engram.
//!
//! So: competitive allocation with homeostasis.
//!   - each feature has a small pool of CANDIDATE units with LEARNABLE weights
//!   - a fact's code is the k units that win on (overlap x boost)
//!   - boost favours units that have not been winning, so load spreads itself
//!   - winners tune their weights toward the features that selected them
//!
//! Hash is still used, but only to decide which units are ALLOWED to compete for a
//! feature — the biological equivalent of random initial connectivity. Which of them
//! actually wins is learned.
//!
//! Plus a decoder, because a memory you cannot read out cannot answer a question.

use std::collections::HashMap;

pub type NeuronId = u32;

#[inline]
pub fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub const CAND: usize = 16; // units allowed to compete for one feature

pub struct Allocator {
    /// feature id -> its candidate units and the learned strength of each
    pub cand: Vec<[(NeuronId, f32); CAND]>,
    pub feat_id: HashMap<String, u32>,
    /// how often each unit has won recently (exponential moving average)
    pub duty: Vec<f32>,
    pub n: u32,
    pub k: usize,
    pub lr: f32,
    pub boost_beta: f32,
    pub wins: Vec<u32>,
}

impl Allocator {
    pub fn new(n: u32, k: usize) -> Allocator {
        Allocator {
            cand: Vec::new(),
            feat_id: HashMap::new(),
            duty: vec![0.0; n as usize],
            wins: vec![0; n as usize],
            n,
            k,
            lr: 0.08,
            boost_beta: 1.0,
        }
    }

    /// Register a feature and give it a candidate pool. Random-ish initial
    /// connectivity, initial strengths jittered so ties are broken by the initial
    /// wiring rather than by unit index.
    pub fn intern(&mut self, f: &str) -> u32 {
        if let Some(&id) = self.feat_id.get(f) {
            return id;
        }
        let id = self.cand.len() as u32;
        let mut pool = [(0u32, 0.0f32); CAND];
        for i in 0..CAND {
            let h = fnv1a(f.as_bytes(), i as u64);
            let unit = (h % self.n as u64) as u32;
            let init = 0.40 + ((h >> 32) % 1000) as f32 / 5000.0; // 0.40..0.60
            pool[i] = (unit, init);
        }
        self.cand.push(pool);
        self.feat_id.insert(f.to_string(), id);
        id
    }

    /// Score every unit that any active feature can reach, then pick the winners.
    /// `boost` is the homeostasis term: a unit that has been winning a lot is
    /// handicapped, so the population spreads its load instead of a few hub units
    /// absorbing everything.
    fn compete(&self, fids: &[u32], target_duty: f32) -> Vec<(NeuronId, f32)> {
        let mut score: HashMap<NeuronId, f32> = HashMap::with_capacity(fids.len() * CAND);
        for &fid in fids {
            for &(u, w) in self.cand[fid as usize].iter() {
                *score.entry(u).or_insert(0.0) += w;
            }
        }
        let mut v: Vec<(NeuronId, f32)> = score
            .into_iter()
            .map(|(u, s)| {
                let d = self.duty[u as usize];
                let boost = ((target_duty - d) * self.boost_beta).exp().clamp(0.25, 4.0);
                (u, s * boost)
            })
            .collect();
        v.sort_unstable_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| fnv1a(&a.0.to_le_bytes(), 0xA11).cmp(&fnv1a(&b.0.to_le_bytes(), 0xA11)))
        });
        v.truncate(self.k);
        v
    }

    /// Allocate a code AND learn from it. This is the step that makes the assignment
    /// earned rather than fixed.
    pub fn allocate_learn(&mut self, features: &[String]) -> Vec<NeuronId> {
        let fids: Vec<u32> = features.iter().map(|f| self.intern(f)).collect();
        let target = self.k as f32 / self.n as f32;
        let winners = self.compete(&fids, target);
        let win_ids: Vec<NeuronId> = winners.iter().map(|&(u, _)| u).collect();

        // winners strengthen their link to the features that selected them;
        // losers in the same pool weaken slightly. classic competitive learning.
        for &fid in &fids {
            for slot in self.cand[fid as usize].iter_mut() {
                if win_ids.contains(&slot.0) {
                    slot.1 += self.lr * (1.0 - slot.1);
                } else {
                    slot.1 -= self.lr * 0.10 * slot.1;
                }
            }
        }
        // homeostasis: winners get less excitable, everyone else drifts back
        let decay = 0.999_f32;
        for &u in &win_ids {
            self.duty[u as usize] = self.duty[u as usize] * decay + (1.0 - decay);
            self.wins[u as usize] += 1;
        }
        let mut out = win_ids;
        out.sort_unstable();
        out
    }

    /// Allocate WITHOUT learning — used at query time, so asking a question does not
    /// rewire the brain.
    pub fn allocate(&self, features: &[String]) -> Vec<NeuronId> {
        let fids: Vec<u32> = features.iter().filter_map(|f| self.feat_id.get(f).copied()).collect();
        if fids.is_empty() { return Vec::new(); }
        let target = self.k as f32 / self.n as f32;
        let mut out: Vec<NeuronId> = self.compete(&fids, target).into_iter().map(|(u, _)| u).collect();
        out.sort_unstable();
        out
    }

    /// Same but return the top `c` only — a partial cue from a partial question.
    pub fn cue(&self, features: &[String], c: usize) -> Vec<NeuronId> {
        let fids: Vec<u32> = features.iter().filter_map(|f| self.feat_id.get(f).copied()).collect();
        if fids.is_empty() { return Vec::new(); }
        let target = self.k as f32 / self.n as f32;
        let mut v = self.compete(&fids, target);
        v.truncate(c);
        let mut out: Vec<NeuronId> = v.into_iter().map(|(u, _)| u).collect();
        out.sort_unstable();
        out
    }
}

// ------------------------------------------------------------------ readout

/// The problem nobody mentioned until now: v3 and v4 deleted the index, which means
/// the network can complete a pattern but cannot SAY what it completed. A recalled
/// code is a set of integers.
///
/// So: a second population where one unit stands for one surface form — word
/// neurons, the "concept cells" of section 6.3. The memory code drives them
/// Hebbianly, and whichever word unit gets the most drive is the answer.
/// This is a learned readout, not a lookup table: every unit of the code votes,
/// the mapping is many-to-many, and it degrades gracefully.
pub struct Decoder {
    /// memory unit -> (word unit, strength)
    pub out: Vec<Vec<(u32, u16)>>,
    pub words: Vec<String>,
    pub word_id: HashMap<String, u32>,
}

impl Decoder {
    pub fn new(n: u32) -> Decoder {
        Decoder { out: vec![Vec::new(); n as usize], words: Vec::new(), word_id: HashMap::new() }
    }

    pub fn word(&mut self, w: &str) -> u32 {
        if let Some(&i) = self.word_id.get(w) { return i; }
        let i = self.words.len() as u32;
        self.words.push(w.to_string());
        self.word_id.insert(w.to_string(), i);
        i
    }

    pub fn learn(&mut self, code: &[NeuronId], word: &str) {
        let wid = self.word(word);
        for &u in code {
            let slot = &mut self.out[u as usize];
            match slot.binary_search_by_key(&wid, |&(w, _)| w) {
                Ok(i) => slot[i].1 = slot[i].1.saturating_add(1),
                Err(i) => slot.insert(i, (wid, 1)),
            }
        }
    }

    /// Every unit that fired votes for the words it is connected to.
    pub fn decode(&self, fired: &[NeuronId], top: usize) -> Vec<(String, u32)> {
        let mut vote: HashMap<u32, u32> = HashMap::new();
        for &u in fired {
            for &(w, s) in &self.out[u as usize] {
                *vote.entry(w).or_insert(0) += s as u32;
            }
        }
        let mut v: Vec<(u32, u32)> = vote.into_iter().collect();
        v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(top);
        v.into_iter().map(|(w, s)| (self.words[w as usize].clone(), s)).collect()
    }
}
