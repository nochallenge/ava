//! v6 — factored codes, so grammar and identity live in different parts of the same
//! pattern, and a compositional readout so she can say a word she has never seen.
//!
//! The v5 wug test failed at exactly 0.0%, and the failure was informative:
//!     overmarch + PST  ->  "overmarches"     (wanted "overmarched")
//! Right word family, wrong grammatical form. Identity was in the code; grammar was
//! not. Homeostasis had deliberately decorrelated everything, including the category
//! structure that generalization needs (same-tag overlap fell from 3.33 to 0.0008).
//!
//! TWO changes, and both are needed:
//!
//! 1. FACTORED CODES. Split the population. Identity units are allocated
//!    competitively with homeostasis, as in v5, so distinct facts stay separate.
//!    Grammar units are a FIXED shared set per tag with no homeostatic pressure at
//!    all, so every past-tense fact literally shares the same past-tense units.
//!    Overlap between two unrelated facts stays ~0; overlap between two past-tense
//!    facts is exactly the grammar part. Similarity structure is back, on purpose,
//!    in a place where it cannot corrupt identity.
//!
//! 2. COMPOSITIONAL READOUT. v5 decoded to a whole word, so a form never seen had no
//!    unit and could never be produced — the wug test was unwinnable by construction.
//!    v6 decodes to an EDIT OPERATION ("delete 0, append ed") which is then applied
//!    to the lemma. Regular forms become derivable; irregulars are still memorised
//!    as their own operation. That split is the dual-route account of inflection,
//!    and here it falls out of the architecture rather than being imposed.

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

// ---------------------------------------------------------------- edit ops

/// The transformation from lemma to form. `walk -> walked` is "0|ed";
/// `try -> tried` is "1|ied"; `eat -> ate` is "3|ate", i.e. memorised whole.
pub fn edit_op(lemma: &str, form: &str) -> String {
    let l: Vec<char> = lemma.chars().collect();
    let f: Vec<char> = form.chars().collect();
    let mut p = 0;
    while p < l.len() && p < f.len() && l[p] == f[p] {
        p += 1;
    }
    let remove = l.len() - p;
    let add: String = f[p..].iter().collect();
    format!("{remove}|{add}")
}

pub fn apply_op(lemma: &str, op: &str) -> Option<String> {
    let (r, add) = op.split_once('|')?;
    let remove: usize = r.parse().ok()?;
    let l: Vec<char> = lemma.chars().collect();
    if remove > l.len() {
        return None;
    }
    let keep: String = l[..l.len() - remove].iter().collect();
    Some(format!("{keep}{add}"))
}

// ---------------------------------------------------------------- features

pub struct Fields<'a> {
    pub lemma: &'a str,
    pub form: Option<&'a str>,
    pub tags: &'a [String],
}

impl<'a> Fields<'a> {
    /// Identity features: what makes THIS fact this fact. The lemma crossed with each
    /// tag, plus the surface form when we have it. Deliberately NOT the bare lemma on
    /// its own — that is shared across all of a word's forms and would drag a past
    /// tense query toward the present tense, which is exactly how v5 failed.
    pub fn identity(&self) -> Vec<String> {
        // ONLY the full lemma+tagset conjunction, and the form when we have it.
        // Using the per-tag conjunctions here was the bug: `L:lead|T:V` is shared by
        // every verb form of "lead", so a PAST query still retrieved the units that
        // vote for "leads". With the full conjunction, an unseen lemma+tagset simply
        // has no identity units at all and the query falls through to the rule.
        let mut ts: Vec<&str> = self.tags.iter().map(|s| s.as_str()).collect();
        ts.sort();
        vec![format!("L:{}|A:{}", self.lemma, ts.join(","))]
        // NOTE: the surface form is deliberately NOT an identity feature.
        // It was, and it was a real bug: at LEARN time the identity features were
        // [L:eat|A:PST,V, F:ate] but at QUERY time only [L:eat|A:PST,V] is known,
        // so competition picked a DIFFERENT set of winners and the query looked up a
        // code that had never been stored. She answered "eats" for eat+PST while the
        // correct memory sat there unreachable. Whatever addresses a memory at write
        // time must be exactly what is available at read time.
    }
    /// Grammar features: the tags, and the whole tag-set as one symbol so that
    /// V;PST and V;PRS;3;SG are different categories rather than overlapping bags.
    pub fn grammar(&self) -> Vec<String> {
        // bare per-tag symbols are kept, but they are the WEAK, generic part: they
        // give "both are verbs" similarity. The tagset symbol carries the specific
        // category, and the crossed shape features carry the conditioning.
        let mut v: Vec<String> = self.tags.iter().map(|t| format!("T:{t}")).collect();
        let mut ts: Vec<&str> = self.tags.iter().map(|s| s.as_str()).collect();
        ts.sort();
        let tagset = ts.join(",");
        v.push(format!("A:{tagset}"));

        // SHAPE: the end of the stem. English inflection is conditioned on it --
        // "bake"+PST is +d, "walk"+PST is +ed, "dax"+PL is +es not +s. Without this
        // the network learns only the single most frequent operation and produces
        // "wugd" and "daxs". These are not identity (they are shared by thousands of
        // words) and not pure syntax either; they are the phonological class.
        // The shape features are ALWAYS crossed with the category. Uncrossed ones
        // ("stems ending in g") appear under every tag, so they vote for whatever
        // operation is globally most common and drown out the category-specific
        // answer -- measured: nonce past tense came back as "wugs". Crossing them
        // means shape only ever speaks within a grammatical category.
        let ch: Vec<char> = self.lemma.chars().collect();
        for n in 1..=3usize {
            if ch.len() >= n {
                let end: String = ch[ch.len() - n..].iter().collect();
                v.push(format!("E{n}:{end}|A:{tagset}"));
            }
        }
        v
    }
}

// ---------------------------------------------------------------- allocator

pub const CAND: usize = 16;

/// Competitive allocation with homeostasis — v5's mechanism, used now ONLY for the
/// identity half of the code.
pub struct IdAllocator {
    pub cand: Vec<[(NeuronId, f32); CAND]>,
    pub feat_id: HashMap<String, u32>,
    pub duty: Vec<f32>,
    pub n: u32,
    pub k: usize,
    pub lr: f32,
    pub beta: f32,
}

impl IdAllocator {
    pub fn new(n: u32, k: usize) -> Self {
        IdAllocator {
            cand: Vec::new(),
            feat_id: HashMap::new(),
            duty: vec![0.0; n as usize],
            n,
            k,
            lr: 0.08,
            beta: 1.0,
        }
    }

    fn intern(&mut self, f: &str) -> u32 {
        if let Some(&i) = self.feat_id.get(f) {
            return i;
        }
        let id = self.cand.len() as u32;
        let mut pool = [(0u32, 0.0f32); CAND];
        for i in 0..CAND {
            let h = fnv1a(f.as_bytes(), i as u64);
            pool[i] = ((h % self.n as u64) as u32, 0.40 + ((h >> 32) % 1000) as f32 / 5000.0);
        }
        self.cand.push(pool);
        self.feat_id.insert(f.to_string(), id);
        id
    }

    fn compete(&self, fids: &[u32], k: usize) -> Vec<NeuronId> {
        if fids.is_empty() { return Vec::new(); }
        let target = self.k as f32 / self.n as f32;
        let mut score: HashMap<NeuronId, f32> = HashMap::with_capacity(fids.len() * CAND);
        for &f in fids {
            for &(u, w) in self.cand[f as usize].iter() {
                *score.entry(u).or_insert(0.0) += w;
            }
        }
        let mut v: Vec<(NeuronId, f32)> = score
            .into_iter()
            .map(|(u, s)| {
                let b = ((target - self.duty[u as usize]) * self.beta).exp().clamp(0.25, 4.0);
                (u, s * b)
            })
            .collect();
        v.sort_unstable_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| fnv1a(&a.0.to_le_bytes(), 0xA11).cmp(&fnv1a(&b.0.to_le_bytes(), 0xA11)))
        });
        v.truncate(k);
        v.into_iter().map(|(u, _)| u).collect()
    }

    pub fn allocate_learn(&mut self, feats: &[String]) -> Vec<NeuronId> {
        let fids: Vec<u32> = feats.iter().map(|f| self.intern(f)).collect();
        let win = self.compete(&fids, self.k);
        for &f in &fids {
            for slot in self.cand[f as usize].iter_mut() {
                if win.contains(&slot.0) {
                    slot.1 += self.lr * (1.0 - slot.1);
                } else {
                    slot.1 -= self.lr * 0.10 * slot.1;
                }
            }
        }
        for &u in &win {
            self.duty[u as usize] = self.duty[u as usize] * 0.999 + 0.001;
        }
        win
    }

    /// query time: no learning, and unseen features simply contribute nothing.
    /// That is what makes an unseen lemma+tag combination fall through to grammar.
    pub fn allocate(&self, feats: &[String], k: usize) -> Vec<NeuronId> {
        let fids: Vec<u32> = feats.iter().filter_map(|f| self.feat_id.get(f).copied()).collect();
        self.compete(&fids, k)
    }

    // ---------------------------------------------------------- v13: PER-KEY units.
    //
    // MEASURED, and it overturns the rule this project has been running on since v9.
    //
    // v9 §3e said identity features are FREE: `allocate` returns a fixed k winners
    // however many keys you supply, so a better key simply wins and costs nothing.
    // Every later design leaned on that. It is false, and the failure mode is visible
    // the moment you add a key that fires OFTEN. Adding a coarse reply-side rung to
    // the reply encoder raised the fraction of held-out positions with SOME address
    // from 29.3% to 93.0% and made her WORSE:
    //
    //     rungs              positions with 3+ addresses      top-1 on them     overall
    //     baseline (3 keys)  11.8%                            89.93%            35.53%
    //     + QW|R1            12.7%                            86.00%            35.46%
    //     + QW|R1 + QW|slot  29.3%                            53.58%            34.64%
    //
    // Reachability went up and precision collapsed, because `compete` sums every
    // supplied key's candidate weights into ONE pool and truncates to k. The slots are
    // a fixed budget. A coarse key does not sit under the exact key, it takes seats
    // from it — and a key that fires everywhere is a hub INSIDE the identity route,
    // which is the same bug as `U:word` and `S:slot|L:len`, in the one place the
    // project had convinced itself the bug could not occur.
    //
    // The fix is not to ration keys, it is to stop making them share a budget. Each
    // key competes on its OWN for `per` units and the code is the union, exactly as
    // `GramAllocator` already gives each grammar feature its own units. Then:
    //
    //   * the exact key keeps the same units no matter how many coarse keys exist,
    //   * a coarse key adds units instead of displacing them,
    //   * and which of them decides is left to the specificity weight at readout,
    //     where ln(N_total / N_served) already knows that a unit serving one memory
    //     is worth more than a unit serving ten thousand.
    //
    // It also makes write and read SYMMETRIC per key, which the pooled version is not:
    // pooled winners depend on which OTHER keys happened to be present, so a key
    // written alongside an exact match addressed different units than the same key
    // queried alone. That is v6's addressing bug yet again, hidden one level down.
    pub fn allocate_each_learn(&mut self, feats: &[String], per: usize) -> Vec<NeuronId> {
        let fids: Vec<u32> = feats.iter().map(|f| self.intern(f)).collect();
        let mut out: Vec<NeuronId> = Vec::with_capacity(fids.len() * per);
        for &f in &fids {
            let win = self.compete(&[f], per);
            for slot in self.cand[f as usize].iter_mut() {
                if win.contains(&slot.0) {
                    slot.1 += self.lr * (1.0 - slot.1);
                } else {
                    slot.1 -= self.lr * 0.10 * slot.1;
                }
            }
            for &u in &win {
                self.duty[u as usize] = self.duty[u as usize] * 0.999 + 0.001;
            }
            out.extend(win);
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn allocate_each(&self, feats: &[String], per: usize) -> Vec<NeuronId> {
        let mut out: Vec<NeuronId> = Vec::with_capacity(feats.len() * per);
        for f in feats {
            let Some(&fid) = self.feat_id.get(f) else { continue };
            out.extend(self.compete(&[fid], per));
        }
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// Grammar units: a FIXED set per tag, deterministic, never subject to homeostasis.
/// Every past-tense fact gets literally the same past-tense units, which is the whole
/// point — that shared substructure IS the grammatical category.
pub struct GramAllocator {
    pub base: u32,
    pub span: u32,
    pub per_tag: usize,
}

impl GramAllocator {
    pub fn units(&self, feat: &str) -> Vec<NeuronId> {
        (0..self.per_tag)
            .map(|i| self.base + (fnv1a(feat.as_bytes(), 0x6_0000_u64 + i as u64) % self.span as u64) as u32)
            .collect()
    }
    pub fn code(&self, feats: &[String]) -> Vec<NeuronId> {
        let mut v: Vec<NeuronId> = feats.iter().flat_map(|f| self.units(f)).collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}

pub struct Factored {
    pub id: IdAllocator,
    pub gram: GramAllocator,
    pub n_total: u32,
}

impl Factored {
    pub fn new(n_id: u32, n_gram: u32, k_id: usize, per_tag: usize) -> Self {
        Factored {
            id: IdAllocator::new(n_id, k_id),
            gram: GramAllocator { base: n_id, span: n_gram, per_tag },
            n_total: n_id + n_gram,
        }
    }

    pub fn learn_code(&mut self, f: &Fields) -> (Vec<NeuronId>, Vec<NeuronId>) {
        self.learn_feats(&f.identity(), &f.grammar())
    }

    pub fn query_code(&self, f: &Fields, k_id: usize) -> (Vec<NeuronId>, Vec<NeuronId>) {
        self.query_feats(&f.identity(), &f.grammar(), k_id)
    }

    /// The same machinery, taking raw feature lists instead of English morphology.
    ///
    /// This is the seam that makes the substrate general. Nothing below this line
    /// knows or cares that the features were a lemma and a tag-set — it only needs
    /// "features that identify this one fact" and "features it shares with a class".
    /// A second domain reuses every neuron and supplies a different pair.
    pub fn learn_feats(&mut self, id: &[String], gram: &[String]) -> (Vec<NeuronId>, Vec<NeuronId>) {
        (self.id.allocate_learn(id), self.gram.code(gram))
    }

    pub fn query_feats(&self, id: &[String], gram: &[String], k_id: usize)
        -> (Vec<NeuronId>, Vec<NeuronId>)
    {
        (self.id.allocate(id, k_id), self.gram.code(gram))
    }

    /// v13: every identity key gets its own `per` units instead of all keys sharing a
    /// fixed budget. See `IdAllocator::allocate_each_learn`.
    pub fn learn_feats_each(&mut self, id: &[String], gram: &[String], per: usize)
        -> (Vec<NeuronId>, Vec<NeuronId>)
    {
        (self.id.allocate_each_learn(id, per), self.gram.code(gram))
    }

    pub fn query_feats_each(&self, id: &[String], gram: &[String], per: usize)
        -> (Vec<NeuronId>, Vec<NeuronId>)
    {
        (self.id.allocate_each(id, per), self.gram.code(gram))
    }
}

// ---------------------------------------------------------------- readout

/// Votes for an edit operation, normalised per unit.
///
/// Normalisation is essential. A grammar unit sits in tens of thousands of codes, so
/// raw counts would let it drown out everything. Dividing by the unit's total vote
/// mass turns each unit into a probability distribution over operations: a grammar
/// unit says "past tense is +ed about 70% of the time", an identity unit that has
/// only ever seen one fact says "this one, 100%".
#[derive(Default)]
pub struct OpReadout {
    pub votes: HashMap<NeuronId, HashMap<u32, i32>>,
    pub total: HashMap<NeuronId, u32>,
    pub ops: Vec<String>,
    pub op_id: HashMap<String, u32>,
    /// how many facts have been learned in total — needed to judge how SPECIFIC
    /// any one unit is
    pub n_learned: u32,
}

impl OpReadout {
    pub fn op(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.op_id.get(s) {
            return i;
        }
        let i = self.ops.len() as u32;
        self.ops.push(s.to_string());
        self.op_id.insert(s.to_string(), i);
        i
    }

    pub fn learn(&mut self, code: &[NeuronId], op: &str) {
        let o = self.op(op);
        self.n_learned += 1;
        for &u in code {
            *self.votes.entry(u).or_default().entry(o).or_insert(0) += 1;
            *self.total.entry(u).or_insert(0) += 1;
        }
    }

    /// Cheap argmax over op ids, for use inside a training loop.
    pub fn argmax(&self, fired: &[NeuronId]) -> Option<u32> {
        let n = self.n_learned.max(1) as f32;
        let mut acc: HashMap<u32, f32> = HashMap::new();
        for &u in fired {
            let (Some(v), Some(&t)) = (self.votes.get(&u), self.total.get(&u)) else { continue };
            if t == 0 { continue; }
            let spec = (n / t as f32).ln().max(0.0);
            for (&o, &c) in v {
                *acc.entry(o).or_insert(0.0) += (c as f32 / t as f32) * spec;
            }
        }
        acc.into_iter().max_by(|a, b| a.1.total_cmp(&b.1)).map(|(o, _)| o)
    }

    /// CREDIT ASSIGNMENT. The delta rule (Rosenblatt, 1958), which is the one thing
    /// this architecture has never had.
    ///
    /// `learn` above is Hebbian counting: it records what co-occurred and never asks
    /// whether the answer was right. That is why every mechanism added downstream of
    /// it in this project failed — working memory, conversational history, dendritic
    /// compartments, a second layer. None of them could be TRAINED to be useful,
    /// because nothing here can answer "which of my parameters should have been
    /// different, given that I was wrong". They could only be hand-designed and hoped
    /// for, and four out of four were not worth having.
    ///
    /// This asks. Predict first; if the prediction is wrong, strengthen the units'
    /// vote for the answer that was actually right and weaken their vote for the one
    /// that won. Blame lands exactly on the units that caused the error.
    ///
    /// Returns whether the prediction was already correct, so a training loop can
    /// report its own error curve.
    pub fn learn_delta(&mut self, code: &[NeuronId], op: &str, lr: i32) -> bool {
        let target = self.op(op);
        let pred = self.argmax(code);
        if pred == Some(target) {
            return true; // already right: leave it alone
        }
        for &u in code {
            let e = self.votes.entry(u).or_default();
            *e.entry(target).or_insert(0) += lr;
            if let Some(p) = pred {
                *e.entry(p).or_insert(0) -= lr;
            }
        }
        false
    }

    /// Vote, weighted by how SPECIFIC each unit is.
    ///
    /// Normalising alone is not enough. A grammar unit sits in ~40,000 codes and puts
    /// ~95% of its mass on the regular operation, so it contributes ~0.95. An identity
    /// unit sits in ~5 codes and puts 100% on the right answer, contributing ~1.0.
    /// Ten of each and the grammar route wins by a nose — which is exactly what
    /// happened: irregulars scored 0.0% and she said "childs", "mouses", "thinked".
    /// Over-regularisation, the error a three-year-old makes.
    ///
    /// The missing term is specificity: ln(facts_total / facts_this_unit_serves).
    /// The identity unit scores ln(120000/5) = 10.1, the grammar unit ln(120000/40000)
    /// = 1.1. A remembered exception now beats the rule nine to one, while a word with
    /// NO identity units still falls through to the rule. That is the dual-route
    /// account of inflection, and it is one multiplication.
    /// DENDRITIC READOUT. The vote, but computed in compartments instead of one flat sum.
    ///
    /// `decode` above sums every unit's contribution into a single pool. That is a
    /// point neuron: one linear summation, depth 1. Beniaguev, Segev & London (Neuron,
    /// 2021) measured how much is lost by that assumption — mimicking one real layer-5
    /// pyramidal cell needs a temporally-convolutional network 5-8 layers deep, and
    /// **removing the NMDA receptors collapses that requirement to a single hidden
    /// layer.** Essentially all of a cortical neuron's computational depth lives in
    /// nonlinear dendritic integration.
    ///
    /// Poirazi & Mel's subunit model says what that nonlinearity is: summation is
    /// SUPRALINEAR within a dendritic branch (NMDA spikes: coincident input on one
    /// branch produces more than the sum of its parts) and roughly LINEAR across
    /// branches at the soma. So a branch where many inputs agree can dominate one
    /// where the same total drive is spread thin.
    ///
    /// Here the branches are the feature families the code was already factored into
    /// — identity, grammar/window, prompt. Within a branch: raise the agreed vote to
    /// a power > 1. Across branches: add. One line of biology, and it turns the
    /// readout from depth 1 into depth 2.
    /// `gate` is the fraction of a branch's own peak an operation must reach to be
    /// reported at all. 0.0 = every branch reports everything (a point neuron with
    /// extra steps); 0.5 = a branch only speaks up for candidates at least half as
    /// strong as its best.
    pub fn decode_dendritic(
        &self, branches: &[&[NeuronId]], power: f32, gate: f32, top: usize,
    ) -> Vec<(String, f32)> {
        let n = self.n_learned.max(1) as f32;
        let mut soma: HashMap<u32, f32> = HashMap::new();
        for branch in branches {
            // within-branch: the ordinary specificity-weighted sum...
            let mut local: HashMap<u32, f32> = HashMap::new();
            for &u in branch.iter() {
                let (Some(v), Some(&t)) = (self.votes.get(&u), self.total.get(&u)) else { continue };
                if t == 0 { continue; }
                let spec = (n / t as f32).ln().max(0.0);
                for (&o, &c) in v {
                    *local.entry(o).or_insert(0.0) += (c as f32 / t as f32) * spec;
                }
            }
            // ...then LOCAL COMPETITION on the branch before anything reaches the soma.
            //
            // Measured first without this: raising each branch's vote to a power did
            // nothing at all (140 distinct replies and 8.5% on-topic at powers 0, 1.3
            // and 1.8; top-1 24.51 -> 24.42 -> 23.84). Of course it did — x^p is
            // MONOTONE, so it widens the gap between winner and runner-up inside a
            // branch but can never change which one wins. Supralinear summation alone
            // is cosmetic.
            //
            // The part of a dendritic spike that can change an outcome is that it is
            // closer to ALL-OR-NONE. A branch does not report a graded opinion about
            // every possibility; it either commits to its strong candidates or stays
            // quiet. So: keep only the ops within `gate` of this branch's own peak and
            // drop the rest. That is a k-winner-take-all on the branch, it is what
            // local inhibition does on a real dendrite, and unlike a power it is
            // non-monotone across ops — a diffuse branch is silenced instead of
            // contributing a smear of mass to whatever is globally most common.
            let peak = local.values().cloned().fold(0.0f32, f32::max);
            if peak <= 0.0 { continue; }
            let bar = peak * gate;
            for (o, x) in local {
                if x >= bar {
                    *soma.entry(o).or_insert(0.0) += x.powf(power);
                }
            }
        }
        let mut v: Vec<(u32, f32)> = soma.into_iter().collect();
        v.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(top);
        v.into_iter().map(|(o, s)| (self.ops[o as usize].clone(), s)).collect()
    }

    pub fn decode(&self, fired: &[NeuronId], top: usize) -> Vec<(String, f32)> {
        let n = self.n_learned.max(1) as f32;
        let mut acc: HashMap<u32, f32> = HashMap::new();
        for &u in fired {
            let (Some(v), Some(&t)) = (self.votes.get(&u), self.total.get(&u)) else { continue };
            if t == 0 { continue; }
            let specificity = (n / t as f32).ln().max(0.0);
            for (&o, &c) in v {
                *acc.entry(o).or_insert(0.0) += (c as f32 / t as f32) * specificity;
            }
        }
        let mut v: Vec<(u32, f32)> = acc.into_iter().collect();
        v.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(top);
        v.into_iter().map(|(o, s)| (self.ops[o as usize].clone(), s)).collect()
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
