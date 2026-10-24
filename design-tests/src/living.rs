//! v3 — "living" memory. No frozen index anywhere.
//!
//! v2 had `tag: Vec<u32>` on every unit: a literal list of which chunks that unit
//! belonged to. That is a lookup table bolted onto the side of a neural network.
//! Delete it and v2 cannot recall anything, which means the memory was never really
//! in the neurons at all.
//!
//! Here there is no tag, no chunk table, no index. A unit owns exactly one thing:
//! a set of outgoing connection STRENGTHS. Learning a fact increments the strengths
//! between the units that co-fired. Learning it again increments them further.
//! The same synapse carries contributions from every fact that ever used it, so the
//! weights are a superposition and no single memory is stored anywhere in particular.
//!
//! The only way to read a memory back out is to stimulate the network and watch what
//! fires. That is the test this module exists to pass.

use std::collections::HashMap;

pub type Tick = u64;
pub type NeuronId = u32;

pub const US: Tick = 1;
pub const MS: Tick = 1_000 * US;

pub fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ------------------------------------------------------------------ encoding

/// A UniMorph row: lemma, inflected form, morphological features.
/// `eat  ate  V;PST`
#[derive(Debug, Clone)]
pub struct Fact {
    pub lemma: String,
    pub form: String,
    pub tags: Vec<String>,
}

impl Fact {
    pub fn parse(line: &str) -> Option<Fact> {
        let mut it = line.split('\t');
        let lemma = it.next()?.trim();
        let form = it.next()?.trim();
        let tags = it.next()?.trim();
        if lemma.is_empty() || form.is_empty() || tags.is_empty() {
            return None;
        }
        Some(Fact {
            lemma: lemma.to_lowercase(),
            form: form.to_lowercase(),
            tags: tags.split(';').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
        })
    }

    /// The features this fact is made of. Lemma and form are rare (high IDF, they
    /// pin down WHICH word). Tags like V or PST are ubiquitous (low IDF, they say
    /// what KIND of fact it is). IDF sorts that out on its own.
    pub fn features(&self) -> Vec<String> {
        let mut f = Vec::with_capacity(8 + self.tags.len());
        f.push(format!("L:{}", self.lemma));
        f.push(format!("F:{}", self.form));
        for t in &self.tags {
            f.push(format!("T:{t}"));
        }
        // the lemma paired with each tag: what makes THIS row different from the
        // other rows about the same lemma
        for t in &self.tags {
            f.push(format!("L:{}|T:{}", self.lemma, t));
        }
        f
    }
}

/// Sparse-scoring IDF encoder. v2 allocated a `vec![0f64; N]` per chunk, which is
/// O(N) per encode and hopeless at N=200000. Only ~features*FANOUT slots are ever
/// touched, so score them in a small map instead. This is also the fix for the
/// "retrieval is O(all units)" problem left open in the v2 note.
pub struct Encoder {
    pub idf: HashMap<String, f32>,
    pub n: u32,
    pub k: usize,
    pub fanout: usize,
}

impl Encoder {
    pub fn fit(facts: &[Fact], n: u32, k: usize, fanout: usize) -> Encoder {
        let total = facts.len() as f32;
        let mut df: HashMap<String, f32> = HashMap::new();
        for fact in facts {
            let mut seen = fact.features();
            seen.sort();
            seen.dedup();
            for f in seen {
                *df.entry(f).or_insert(0.0) += 1.0;
            }
        }
        let idf = df.into_iter().map(|(f, d)| (f, (total / d).ln().max(0.0))).collect();
        Encoder { idf, n, k, fanout }
    }

    pub fn encode(&self, fact: &Fact) -> Vec<NeuronId> {
        let mut score: HashMap<u32, f32> = HashMap::with_capacity(64);
        for f in fact.features() {
            let w = *self.idf.get(&f).unwrap_or(&1.0);
            if w <= 0.0 {
                continue;
            }
            for i in 0..self.fanout {
                let slot = (fnv1a(f.as_bytes(), i as u64) % self.n as u64) as u32;
                *score.entry(slot).or_insert(0.0) += w;
            }
        }
        let mut v: Vec<(u32, f32)> = score.into_iter().collect();
        v.sort_unstable_by(|a, b| {
            b.1.total_cmp(&a.1).then_with(|| {
                fnv1a(&a.0.to_le_bytes(), 0xA11).cmp(&fnv1a(&b.0.to_le_bytes(), 0xA11))
            })
        });
        v.truncate(self.k);
        let mut out: Vec<NeuronId> = v.into_iter().map(|(i, _)| i).collect();
        out.sort_unstable();
        out
    }
}

pub fn overlap(a: &[NeuronId], b: &[NeuronId]) -> usize {
    let (mut i, mut j, mut n) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => { n += 1; i += 1; j += 1; }
        }
    }
    n
}

// ------------------------------------------------------------------ the unit

/// A unit owns connection strengths and nothing else. There is deliberately no
/// `tag`, no chunk id, no list of what it "contains".
#[derive(Debug, Default, Clone)]
pub struct Unit {
    /// (target, strength), kept sorted by target so co-firing pairs ACCUMULATE
    /// into an existing synapse instead of appending a duplicate.
    pub out: Vec<(NeuronId, u16)>,
    pub v: f32,
    pub t_last: Tick,
    pub fired_at: Tick,
}

impl Unit {
    /// Hebbian: strengthen an existing connection, or grow a new one.
    /// This is the entire storage mechanism.
    fn strengthen(&mut self, to: NeuronId, by: u16) {
        match self.out.binary_search_by_key(&to, |&(t, _)| t) {
            Ok(i) => self.out[i].1 = self.out[i].1.saturating_add(by),
            Err(i) => self.out.insert(i, (to, by)),
        }
    }
    pub fn strength_to(&self, to: NeuronId) -> u16 {
        self.out.binary_search_by_key(&to, |&(t, _)| t).map(|i| self.out[i].1).unwrap_or(0)
    }
    /// how many distinct partners this unit is wired to
    pub fn degree(&self) -> usize { self.out.len() }
    /// total strength it carries — a proxy for how much it has learned
    pub fn load(&self) -> u64 { self.out.iter().map(|&(_, w)| w as u64).sum() }
}

pub struct Brain {
    pub units: Vec<Unit>,
    pub tau_m: f32,
    pub v_th: f32,
}

impl Brain {
    pub fn new(n: u32) -> Brain {
        Brain { units: vec![Unit::default(); n as usize], tau_m: (10 * MS) as f32, v_th: 1.0 }
    }

    /// Learn one fact. No id is recorded. Nothing is appended to any table.
    /// The only change to the world is that some synapses got stronger.
    pub fn learn(&mut self, code: &[NeuronId], increment: u16) {
        for &a in code {
            for &b in code {
                if a != b {
                    self.units[a as usize].strengthen(b, increment);
                }
            }
        }
    }

    /// Recall: clamp the cue on, let one volley of activation spread, and report
    /// which units fired. No lookup, no index, no stored answer to compare against.
    ///
    /// Threshold is RELATIVE - a unit fires if its input reaches `frac` of the
    /// strongest input in the population. That is the E%-max winner-take-all rule
    /// the dynamics critique pointed at, and it makes the network self-normalising:
    /// no absolute weight has to be tuned as facts accumulate.
    pub fn recall(&self, cue: &[NeuronId], frac: f32, cap: usize) -> Vec<NeuronId> {
        let mut input: HashMap<NeuronId, u64> = HashMap::with_capacity(cue.len() * 64);
        for &c in cue {
            for &(to, w) in &self.units[c as usize].out {
                *input.entry(to).or_insert(0) += w as u64;
            }
        }
        let peak = input.values().copied().max().unwrap_or(0) as f32;
        if peak == 0.0 { return Vec::new(); }
        let bar = (peak * frac) as u64;
        let mut v: Vec<(NeuronId, u64)> =
            input.into_iter().filter(|&(_, s)| s >= bar).collect();
        v.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(cap);
        let mut out: Vec<NeuronId> = v.into_iter().map(|(i, _)| i).collect();
        out.sort_unstable();
        out
    }

    pub fn total_synapses(&self) -> u64 {
        self.units.iter().map(|u| u.out.len() as u64).sum()
    }
}
