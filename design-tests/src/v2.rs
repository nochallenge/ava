//! v2 — the 8.1 design with the measured failures fixed.
//!
//! v1 failed three ways and they turned out to share one root cause: nothing in the
//! network ever said "enough units have fired, stop." Everything was excitation.
//!
//! The fix is one mechanism: a global inhibitory pool that every spike feeds and
//! that raises everyone's threshold. It is modelled on dentate gyrus / CA3 feedback
//! inhibition, and it fixes all three failures at once:
//!   - runaway      -> inhibition rises until nothing more can fire
//!   - interference -> by the time a rival chunk's shared units have fired, the bar
//!                     is too high for the rival to complete. first pattern wins.
//!   - k-WTA        -> sparsity becomes a dynamic property, not a truncate(k) call
//!
//! Plus two synapse types, which is what gives a chain of thought:
//!   WITHIN a chunk  — symmetric, strong, no delay      -> pattern completion
//!   BETWEEN chunks  — directed, weaker, LONG delay     -> sequence
//! The long delay is not decoration. A successor chunk must wait for inhibition to
//! decay or it can never fire. See `SEQ_DELAY` and the `chain_of_thought` test.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

pub type Tick = u64;
pub type NeuronId = usize;

pub const US: Tick = 1;
pub const MS: Tick = 1_000 * US;

pub const N: usize = 2000;
pub const K: usize = 40;
pub const FANOUT: usize = 8;

/// mV above rest, scaled x10 from the textbook 15 mV so integer weights have
/// enough resolution to land inside the operating window. v1's window was
/// [1.5, 1.67) and weights were whole millivolts, which is why nothing worked.
pub const V_TH: f64 = 150.0;
pub const TAU_M: f64 = (10 * MS) as f64;

// ---------------------------------------------------------------- inhibition

/// The global inhibitory pool. One per population. Same lazy-decay trick as the
/// membrane: store a level and the time it was true, catch up on demand.
#[derive(Debug, Clone)]
pub struct Inhibition {
    level: f64,
    t_last: Tick,
    pub tau: f64,
    pub per_spike: f64,
    pub gain: f64,
}

impl Inhibition {
    pub fn new(tau: f64, per_spike: f64, gain: f64) -> Self {
        Inhibition { level: 0.0, t_last: 0, tau, per_spike, gain }
    }
    pub fn level_at(&self, now: Tick) -> f64 {
        let dt = now.saturating_sub(self.t_last) as f64;
        self.level * (-dt / self.tau).exp()
    }
    /// how much the firing threshold is raised right now
    pub fn offset(&self, now: Tick) -> f64 {
        self.gain * self.level_at(now)
    }
    pub fn bump(&mut self, now: Tick) {
        self.level = self.level_at(now) + self.per_spike;
        self.t_last = now;
    }
}

// ---------------------------------------------------------------- the neuron

#[derive(Debug, Clone)]
pub struct Neuron {
    v_last: f64,
    t_last: Tick,
    tau: f64,
    v_th: f64,
    v_reset: f64,
    refractory: Tick,
    fireable_at: Tick,
}

impl Neuron {
    pub fn new(tau: f64, v_th: f64, refractory: Tick) -> Self {
        Neuron { v_last: 0.0, t_last: 0, tau, v_th, v_reset: 0.0, refractory, fireable_at: 0 }
    }
    pub fn v_at(&self, now: Tick) -> f64 {
        let dt = now.saturating_sub(self.t_last) as f64;
        self.v_last * (-dt / self.tau).exp()
    }
    /// `extra_th` is the inhibitory offset. That single argument is the whole fix.
    pub fn receive(&mut self, now: Tick, w: f64, extra_th: f64) -> bool {
        self.v_last = self.v_at(now);
        self.t_last = now;
        if now < self.fireable_at {
            return false;
        }
        self.v_last += w;
        if self.v_last >= self.v_th + extra_th {
            self.v_last = self.v_reset;
            self.fireable_at = now + self.refractory;
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------- events

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event {
    pub at: Tick,
    pub target: NeuronId,
    pub weight_mv: i32,
}

impl Ord for Event {
    fn cmp(&self, o: &Self) -> Ordering {
        self.at.cmp(&o.at).then(self.target.cmp(&o.target)).then(self.weight_mv.cmp(&o.weight_mv))
    }
}
impl PartialOrd for Event {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}

// ---------------------------------------------------------------- unit + brain

#[derive(Debug, Clone, Copy)]
pub struct Synapse { pub to: NeuronId, pub weight_mv: i32, pub delay: Tick }

#[derive(Debug, Clone)]
pub struct Unit {
    /// the thinking layer: integrate, decide, fire
    pub cell: Neuron,
    /// completion wiring, symmetric, within one chunk
    pub within: Vec<Synapse>,
    /// sequence wiring, directed and delayed, to the next chunk
    pub next: Vec<Synapse>,
    /// the memory layer: which chunks this unit holds a piece of
    pub tag: Vec<u32>,
}

pub struct Brain {
    pub units: Vec<Unit>,
    pub inh: Inhibition,
    pub queue: BinaryHeap<Reverse<Event>>,
    pub refractory: Tick,
}

impl Brain {
    pub fn new(n: usize, inh: Inhibition, refractory: Tick) -> Self {
        Brain {
            units: vec![
                Unit {
                    cell: Neuron::new(TAU_M, V_TH, refractory),
                    within: Vec::new(),
                    next: Vec::new(),
                    tag: Vec::new(),
                };
                n
            ],
            inh,
            queue: BinaryHeap::new(),
            refractory,
        }
    }

    pub fn push(&mut self, e: Event) { self.queue.push(Reverse(e)); }

    pub fn run(&mut self, until: Tick, max_spikes: usize) -> Vec<(Tick, NeuronId)> {
        let mut fired = Vec::new();
        while let Some(Reverse(e)) = self.queue.pop() {
            if e.at > until { self.queue.push(Reverse(e)); break; }
            if fired.len() >= max_spikes { break; }
            let extra = self.inh.offset(e.at);
            let unit = &mut self.units[e.target];
            if unit.cell.receive(e.at, e.weight_mv as f64, extra) {
                fired.push((e.at, e.target));
                for s in unit.within.iter().chain(unit.next.iter()) {
                    self.queue.push(Reverse(Event {
                        at: e.at + s.delay,
                        target: s.to,
                        weight_mv: s.weight_mv,
                    }));
                }
                self.inh.bump(e.at);
            }
        }
        fired
    }
}

// ---------------------------------------------------------------- encoding

pub fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub const STOP: &[&str] = &[
    "the", "a", "an", "of", "in", "to", "and", "or", "is", "it", "its", "on", "with",
    "for", "that", "this", "as", "at", "by", "be", "are", "was", "were", "must", "how",
    "more", "than", "we", "you", "i", "he", "she", "they", "but", "if", "then", "so",
];

pub fn normalize(w: &str) -> String {
    w.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase()
}

/// Pattern separation happens HERE, before anything is stored. Stopword removal
/// plus word-pair features took max pairwise overlap from 9 to 1 in v1's
/// experiment 4, which was worth more than every weight tweak combined.
pub fn features(chunk: &str) -> Vec<String> {
    let kept: Vec<String> = chunk
        .split_whitespace()
        .map(normalize)
        .filter(|w| !w.is_empty() && !STOP.contains(&w.as_str()))
        .collect();
    let mut f = kept.clone();
    for p in kept.windows(2) {
        f.push(format!("{}_{}", p[0], p[1]));
    }
    f
}

/// Hash tie-break, not index tie-break. v1's index tie-break dragged every code
/// toward low-numbered units (mean index 577 where unbiased is ~1000).
pub fn top_k(score: &[u32], k: usize) -> Vec<NeuronId> {
    let mut idx: Vec<usize> = (0..score.len()).filter(|&i| score[i] > 0).collect();
    idx.sort_unstable_by(|&a, &b| {
        score[b].cmp(&score[a]).then_with(|| {
            fnv1a(&a.to_le_bytes(), 0xA11).cmp(&fnv1a(&b.to_le_bytes(), 0xA11))
        })
    });
    idx.truncate(k);
    idx.sort_unstable();
    idx
}

pub fn encode(chunk: &str) -> Vec<NeuronId> {
    let mut score = vec![0u32; N];
    for f in features(chunk) {
        for i in 0..FANOUT {
            score[(fnv1a(f.as_bytes(), i as u64) % N as u64) as usize] += 1;
        }
    }
    top_k(&score, K)
}

pub fn overlap(a: &[NeuronId], b: &[NeuronId]) -> usize {
    let (mut i, mut j, mut n) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => { n += 1; i += 1; j += 1; }
        }
    }
    n
}

// ---------------------------------------------------------------- installing

/// Wire one chunk for completion. Weight is normalised to the cue size we expect
/// to be given, rather than hand tuned: w = V_th / cue.
pub fn install_chunk(brain: &mut Brain, chunk_id: u32, code: &[NeuronId], expected_cue: usize) {
    install_chunk_drive(brain, chunk_id, code, expected_cue, 1.0)
}

/// `drive` is headroom over bare threshold. It must exceed 1.0, because the cue's
/// own spikes raise inhibition before the completion current arrives. Setting
/// drive = 1.0 was the first thing I tried and nothing fired at all.
pub fn install_chunk_drive(
    brain: &mut Brain,
    chunk_id: u32,
    code: &[NeuronId],
    expected_cue: usize,
    drive: f64,
) {
    let w = ((V_TH * drive) / expected_cue as f64).ceil() as i32;
    for &id in code {
        brain.units[id].tag.push(chunk_id);
        for &other in code {
            if other != id {
                brain.units[id].within.push(Synapse { to: other, weight_mv: w, delay: MS });
            }
        }
    }
}

/// Wire chunk A -> chunk B as a sequence link. The delay MUST outlast the
/// inhibitory transient or B can never fire; that is the whole reason this is a
/// separate synapse type and not just another recurrent weight.
pub fn install_sequence(
    brain: &mut Brain,
    from: &[NeuronId],
    to: &[NeuronId],
    delay: Tick,
    drive_frac: f64,
) {
    // enough total drive that a fully-active A pushes B over threshold once
    // inhibition has decayed: from.len() * w >= V_th * drive_frac
    let w = ((V_TH * drive_frac) / from.len() as f64).ceil() as i32;
    for &a in from {
        for &b in to {
            brain.units[a].next.push(Synapse { to: b, weight_mv: w, delay });
        }
    }
}

pub fn cue(brain: &mut Brain, seed: &[NeuronId], at: Tick) {
    for &id in seed {
        brain.push(Event { at, target: id, weight_mv: (V_TH * 2.0) as i32 });
    }
}

/// Which chunks fired, and how completely.
pub fn readout(spikes: &[(Tick, NeuronId)], codes: &[Vec<NeuronId>]) -> Vec<usize> {
    let mut who: Vec<NeuronId> = spikes.iter().map(|&(_, id)| id).collect();
    who.sort_unstable();
    who.dedup();
    codes.iter().map(|c| overlap(&who, c)).collect()
}

/// Separation deliberately DISABLED: raw words, no stopword removal, no word pairs.
/// Used only to test whether inhibition alone can cope with high overlap.
pub fn encode_raw(chunk: &str) -> Vec<NeuronId> {
    let mut score = vec![0u32; N];
    for word in chunk.split_whitespace() {
        let w = normalize(word);
        if w.is_empty() { continue; }
        for i in 0..FANOUT {
            score[(fnv1a(w.as_bytes(), i as u64) % N as u64) as usize] += 1;
        }
    }
    top_k(&score, K)
}

/// encode into a population of arbitrary size, so N and K can be swept.
pub fn encode_in(chunk: &str, n: usize, k: usize) -> Vec<NeuronId> {
    let mut score = vec![0u32; n];
    for f in features(chunk) {
        for i in 0..FANOUT {
            score[(fnv1a(f.as_bytes(), i as u64) % n as u64) as usize] += 1;
        }
    }
    top_k(&score, k)
}

// ---------------------------------------------------------------- the encoder

use std::collections::HashMap;

/// Pattern separation, done properly. Stopword removal is a hand-written guess at
/// which features are uninformative; IDF measures it. A feature that appears in
/// every chunk gets weight ~0 and contributes nothing to the code, so a code is
/// built out of what makes its chunk DIFFERENT.
///
/// Measured on a 400-chunk corpus at N=20000, k=40:
///   plain features -> mean pairwise overlap 15.92, max 28   (unusable)
///   IDF weighted   -> mean pairwise overlap  0.18, max 15   (usable at cue >= 16)
pub struct Encoder {
    idf: HashMap<String, f64>,
    pub n: usize,
    pub k: usize,
}

impl Encoder {
    pub fn fit(corpus: &[String], n: usize, k: usize) -> Self {
        let total = corpus.len() as f64;
        let mut df: HashMap<String, f64> = HashMap::new();
        for text in corpus {
            let mut seen = features(text);
            seen.sort();
            seen.dedup();
            for f in seen {
                *df.entry(f).or_insert(0.0) += 1.0;
            }
        }
        let idf = df.into_iter().map(|(f, d)| (f, (total / d).ln().max(0.0))).collect();
        Encoder { idf, n, k }
    }

    pub fn encode(&self, text: &str) -> Vec<NeuronId> {
        let mut score = vec![0f64; self.n];
        for f in features(text) {
            let w = *self.idf.get(&f).unwrap_or(&1.0);
            if w <= 0.0 { continue; }
            for i in 0..FANOUT {
                score[(fnv1a(f.as_bytes(), i as u64) % self.n as u64) as usize] += w;
            }
        }
        // scores are f64, which is not Ord. total_cmp is the escape hatch from 5.2,
        // and this is the first place the project actually needs it.
        let mut idx: Vec<usize> = (0..self.n).filter(|&i| score[i] > 0.0).collect();
        idx.sort_unstable_by(|&a, &b| {
            score[b].total_cmp(&score[a]).then_with(|| {
                fnv1a(&a.to_le_bytes(), 0xA11).cmp(&fnv1a(&b.to_le_bytes(), 0xA11))
            })
        });
        idx.truncate(self.k);
        idx.sort_unstable();
        idx
    }
}
