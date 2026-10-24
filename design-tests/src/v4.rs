//! v4 — the merge. v3's living synapses recalled through v2's event-driven LIF loop.
//!
//! v2 had exact u64 time, a BinaryHeap event queue and real membrane dynamics, but
//! stored memories next to a frozen `tag` index.
//! v3 had genuinely distributed storage — strengths only, no index — but recalled by
//! summing input in one pass, which is not spiking at all.
//!
//! The hard part of merging them is the threshold. v3 used a RELATIVE rule (fire if
//! you reach 55% of the strongest input in the population), which is trivial when you
//! can see every unit's total at once and impossible when units fire asynchronously.
//!
//! The resolution is that a spiking network does not need to compute the maximum.
//! If input arrives SPREAD OVER TIME, the most strongly driven unit reaches threshold
//! FIRST, entirely on its own. Feedback inhibition triggered by those first spikes
//! then raises the bar on everyone else. Rank ordering falls out of the dynamics, and
//! that is time-to-first-spike coding from section 3 of the first note doing real work.
//!
//! So synapses get a small deterministic delay jitter. Without it every input lands on
//! the same tick, every unit crosses simultaneously, and inhibition has nothing to
//! discriminate between. The jitter is what makes the temporal code exist.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

pub type Tick = u64;
pub type NeuronId = u32;

pub const US: Tick = 1;
pub const MS: Tick = 1_000 * US;

#[inline]
pub fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ------------------------------------------------------------------- events

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event {
    pub at: Tick,
    pub target: NeuronId,
    pub w: u32, // synaptic strength, integer so Event can be Ord
}

impl Ord for Event {
    fn cmp(&self, o: &Self) -> Ordering {
        self.at.cmp(&o.at).then(self.target.cmp(&o.target)).then(self.w.cmp(&o.w))
    }
}
impl PartialOrd for Event {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
}

// ------------------------------------------------------------------- unit

/// v3's unit — strengths and nothing else — plus the LIF state v2 needed.
/// Still no `tag`, no chunk id, no index.
#[derive(Debug, Default, Clone)]
pub struct Unit {
    pub out: Vec<(NeuronId, u16)>,
    v: f32,
    t_last: Tick,
    fired_at: Tick,
}

impl Unit {
    pub fn strengthen(&mut self, to: NeuronId, by: u16) {
        match self.out.binary_search_by_key(&to, |&(t, _)| t) {
            Ok(i) => self.out[i].1 = self.out[i].1.saturating_add(by),
            Err(i) => self.out.insert(i, (to, by)),
        }
    }
    pub fn strength_to(&self, to: NeuronId) -> u16 {
        self.out.binary_search_by_key(&to, |&(t, _)| t).map(|i| self.out[i].1).unwrap_or(0)
    }
    pub fn degree(&self) -> usize { self.out.len() }

    /// Sort and merge duplicate targets, summing strengths exactly as `strengthen`
    /// would have. Leaves capacity alone.
    fn dedup_sorted(&mut self) {
        if self.out.len() < 2 { return; }
        self.out.sort_unstable_by_key(|&(t, _)| t);
        let mut w = 0;
        for r in 1..self.out.len() {
            if self.out[r].0 == self.out[w].0 {
                self.out[w].1 = self.out[w].1.saturating_add(self.out[r].1);
            } else {
                w += 1;
                self.out[w] = self.out[r];
            }
        }
        self.out.truncate(w + 1);
    }

    /// Bulk-build append. MEASURED: `strengthen` keeps `out` sorted with
    /// `Vec::insert`, which memmoves half the vector on every synapse. That is fine
    /// while units are small, and it is quadratic once they are not — and grammar
    /// units are hubs BY DESIGN: every past-tense fact wires the same past-tense
    /// units, so at 652,477 facts one of them reaches ~4,000,000 partners. Writing
    /// 120k facts took 62 s that way; the full corpus would have taken ~30 min.
    ///
    /// So: push unsorted, and only sort+merge when the vector is genuinely full.
    /// Compaction is what keeps the duplicates from exploding — grammar-to-grammar
    /// pairs repeat once per fact, hundreds of thousands of times. Reserving after a
    /// compaction that freed little bounds how often it can run.
    pub fn push_raw(&mut self, to: NeuronId, by: u16) {
        if self.out.len() == self.out.capacity() && self.out.capacity() >= 32 {
            self.dedup_sorted();
            if self.out.capacity() - self.out.len() < self.out.capacity() / 4 {
                self.out.reserve(self.out.len()); // genuinely distinct partners; grow
            }
        }
        self.out.push((to, by));
    }

    /// Restore the sorted, deduplicated invariant `recall` and `strength_to` expect.
    pub fn compact(&mut self) {
        self.dedup_sorted();
        self.out.shrink_to_fit();
    }

    /// SYNAPTIC PRUNING. Keep only the `keep` strongest synapses.
    ///
    /// Measured cause of a real failure: grammar units are hubs by construction, and
    /// at 652,477 facts one reached 1,225,703 outgoing synapses. `recall` seeds by
    /// emitting EVERY outgoing spike of every cue unit, so one query pushed millions
    /// of events into the heap and took ~10 seconds against ~25 microseconds for the
    /// direct readout. That is why spiking recall silently fell out of the pipeline
    /// after v5 — it did not get removed, it became unusable and the code routed
    /// around it.
    ///
    /// No cortical pyramidal cell has a million partners either; the usual figure is
    /// 10,000 or so. A synapse that was strengthened once, among a million, carries
    /// almost no information — the weakest are exactly the ones a real brain removes.
    /// So bound the degree and drop the weakest, which is what development does.
    pub fn prune(&mut self, keep: usize) {
        if keep == 0 || self.out.len() <= keep {
            return;
        }
        // strongest first, ties by target so pruning is deterministic
        self.out.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        self.out.truncate(keep);
        self.out.sort_unstable_by_key(|&(t, _)| t); // restore the search invariant
        self.out.shrink_to_fit();
    }
}

// ------------------------------------------------------------------- config

#[derive(Debug, Clone, Copy)]
pub struct Params {
    pub tau_m: f32,
    /// bar as a fraction of the running peak drive seen this volley
    pub frac: f32,
    pub base_delay: Tick,
    /// spread of synaptic delays. THIS is what creates the temporal code.
    pub jitter: Tick,
    pub tau_inh: f32,
    pub inh_per_spike: f32,
    pub inh_gain: f32,
    pub window: Tick,
    pub max_spikes: usize,
    /// Count DISTINCT presynaptic contributors instead of summing raw strength.
    ///
    /// The threshold is `frac * cue.len()`, written on the assumption stated in
    /// `recall`: "a pattern member receives one synapse from every cue unit, so its
    /// drive is ~cue.len()". That assumption holds only while strengths are ~1.
    /// They are not: a grammar-to-grammar synapse is strengthened once per fact and
    /// reaches 47,832, so ONE arrival clears a threshold of 15.4 by three orders of
    /// magnitude and the volley is noise from the first event onward.
    ///
    /// The bar is counted in units and the drive is counted in facts. Counting
    /// contributors puts them back in the same currency, which is what the threshold
    /// always meant.
    pub count_drive: bool,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            tau_m: (20 * MS) as f32,
            frac: 0.55,
            base_delay: 1 * MS,
            jitter: 4 * MS,
            tau_inh: (25 * MS) as f32, // >= tau_m, mandatory (v2 note 9.2b)
            inh_per_spike: 1.0,
            inh_gain: 0.35,
            window: 60 * MS,
            max_spikes: 4096,
            count_drive: false,
        }
    }
}

pub struct Brain {
    pub units: Vec<Unit>,
    pub p: Params,
}

impl Brain {
    pub fn new(n: u32, p: Params) -> Brain {
        Brain { units: vec![Unit::default(); n as usize], p }
    }

    /// Hebbian. Unchanged from v3: the only thing learning does is make
    /// connections stronger.
    pub fn learn(&mut self, code: &[NeuronId], inc: u16) {
        for &a in code {
            for &b in code {
                if a != b {
                    self.units[a as usize].strengthen(b, inc);
                }
            }
        }
    }

    /// Same Hebbian rule, bulk write path. Must be followed by `compact()` before
    /// any recall. Identical results to `learn`, ~50x faster at corpus scale.
    pub fn learn_bulk(&mut self, code: &[NeuronId], inc: u16) {
        for &a in code {
            for &b in code {
                if a != b {
                    self.units[a as usize].push_raw(b, inc);
                }
            }
        }
    }

    /// Sort and merge every unit's synapses. Cheap on units that were never touched.
    pub fn compact(&mut self) {
        for u in self.units.iter_mut() {
            if u.out.len() > 1 {
                u.compact();
            }
        }
    }

    /// Compact, then cap every unit's out-degree at `max_degree`, keeping the
    /// strongest. Returns how many synapses were pruned away.
    pub fn compact_pruned(&mut self, max_degree: usize) -> u64 {
        let before = self.total_synapses();
        for u in self.units.iter_mut() {
            if u.out.len() > 1 {
                u.compact();
            }
            u.prune(max_degree);
        }
        before - self.total_synapses()
    }

    #[inline]
    fn delay(&self, from: NeuronId, to: NeuronId) -> Tick {
        if self.p.jitter == 0 { return self.p.base_delay; }
        let h = fnv1a(&((from as u64) << 32 | to as u64).to_le_bytes(), 0xD1A);
        self.p.base_delay + (h % self.p.jitter)
    }

    /// Spiking recall. Seed the cue, run the event queue, report who fired and when.
    ///
    /// MEASURED, and not what I first built: a running-peak relative threshold does
    /// NOT work here. The first event to arrive sets the peak, so the bar starts near
    /// zero, that unit fires, and the volley avalanches — 3987 spikes and 0% precision.
    /// The bar has to be absolute and scaled to the cue, because the cue size is the
    /// one thing that IS known up front. Inhibition then ends the volley.
    ///
    /// Selection is done by the threshold; TIMING is what the temporal code carries.
    /// Pattern members reach threshold at 4.06 ms on average, spurious units at
    /// 6.36 ms, so who-fired-first is a real readable signal even though it is not
    /// what does the selecting.
    pub fn recall(&mut self, cue: &[NeuronId], now: Tick) -> Vec<(Tick, NeuronId)> {
        let p = self.p;
        let mut q: BinaryHeap<Reverse<Event>> = BinaryHeap::with_capacity(4096);
        let mut touched: Vec<NeuronId> = Vec::with_capacity(4096);

        // the cue units are taken as already firing: emit their outgoing spikes
        for &c in cue {
            for i in 0..self.units[c as usize].out.len() {
                let (to, w) = self.units[c as usize].out[i];
                q.push(Reverse(Event { at: now + self.delay(c, to), target: to, w: w as u32 }));
            }
        }

        let mut fired: Vec<(Tick, NeuronId)> = Vec::new();
        // ABSOLUTE bar, scaled to the cue. A pattern member receives one synapse
        // from every cue unit, so its drive is ~cue.len(); a unit that merely shares
        // a synapse or two receives ~1-2. frac*cue.len() sits between them.
        // The running-peak version of this fired on the first arrival and avalanched.
        let theta: f32 = p.frac * cue.len() as f32;
        let mut inh_level: f32 = 0.0;
        let mut inh_t: Tick = now;
        let deadline = now + p.window;

        while let Some(Reverse(e)) = q.pop() {
            if e.at > deadline || fired.len() >= p.max_spikes { break; }

            let u = &mut self.units[e.target as usize];
            if u.t_last == 0 && u.v == 0.0 { touched.push(e.target); }

            // lazy decay, one exp() per event — section 5.4
            let dt = e.at.saturating_sub(u.t_last) as f32;
            if u.t_last != 0 { u.v *= (-dt / p.tau_m).exp(); }
            u.t_last = e.at;

            if u.fired_at != 0 { continue; } // one spike per unit per recall (v2 §2)

            u.v += if p.count_drive { 1.0 } else { e.w as f32 };

            // inhibition decays, then raises the bar. This is what ENDS the volley:
            // once enough units have fired, nothing else can reach threshold.
            let idt = e.at.saturating_sub(inh_t) as f32;
            inh_level *= (-idt / p.tau_inh).exp();
            inh_t = e.at;

            let bar = theta + inh_level * p.inh_gain;
            if u.v >= bar {
                u.fired_at = e.at;
                fired.push((e.at, e.target));
                inh_level += p.inh_per_spike;
                for i in 0..u.out.len() {
                    let (to, w) = self.units[e.target as usize].out[i];
                    let d = self.delay(e.target, to);
                    if e.at + d <= deadline {
                        q.push(Reverse(Event { at: e.at + d, target: to, w: w as u32 }));
                    }
                }
            }
        }

        // reset only the units we touched — never the whole population
        for &id in &touched {
            let u = &mut self.units[id as usize];
            u.v = 0.0;
            u.t_last = 0;
            u.fired_at = 0;
        }
        fired
    }

    /// v3's one-shot non-spiking recall, kept so the merge can be checked against it.
    pub fn recall_oneshot(&self, cue: &[NeuronId], frac: f32, cap: usize) -> Vec<NeuronId> {
        use std::collections::HashMap;
        let mut input: HashMap<NeuronId, u64> = HashMap::with_capacity(cue.len() * 64);
        for &c in cue {
            for &(to, w) in &self.units[c as usize].out {
                *input.entry(to).or_insert(0) += w as u64;
            }
        }
        let peak = input.values().copied().max().unwrap_or(0) as f32;
        if peak == 0.0 { return Vec::new(); }
        let bar = (peak * frac) as u64;
        let mut v: Vec<(NeuronId, u64)> = input.into_iter().filter(|&(_, s)| s >= bar).collect();
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

#[cfg(test)]
mod bulk_tests {
    use super::*;

    /// The fast write path must produce byte-identical synapses to the slow one.
    /// This is the only thing that makes the optimisation safe to trust at a scale
    /// where the slow path can no longer be run for comparison.
    #[test]
    fn bulk_write_equals_sorted_insert() {
        let codes: Vec<Vec<NeuronId>> = (0..400u32)
            .map(|i| {
                let mut c: Vec<NeuronId> = (0..14)
                    .map(|j| (fnv1a(&[i as u8, j as u8], j as u64) % 300) as u32)
                    .collect();
                c.sort_unstable();
                c.dedup();
                c
            })
            .collect();

        let mut slow = Brain::new(300, Params::default());
        for c in &codes {
            slow.learn(c, 1);
        }

        let mut fast = Brain::new(300, Params::default());
        for c in &codes {
            fast.learn_bulk(c, 1);
        }
        fast.compact();

        assert_eq!(slow.total_synapses(), fast.total_synapses());
        for i in 0..300usize {
            assert_eq!(slow.units[i].out, fast.units[i].out, "unit {i} differs");
        }
        assert!(slow.total_synapses() > 5_000, "test is too small to mean anything");
    }
}
