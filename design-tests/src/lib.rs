pub mod wordclass;
pub mod reply;
pub mod seq;
pub mod store;
pub mod v6;
pub mod v5;
pub mod v4;
pub mod living;
pub mod v2;

use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub type Tick = u64;
pub type NeuronId = usize;

pub const US: Tick = 1;
pub const MS: Tick = 1_000 * US;

// ---------------- 12.1 the neuron ----------------

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
    pub fn new(tau_ticks: f64, v_th: f64) -> Self {
        Neuron {
            v_last: 0.0,
            t_last: 0,
            tau: tau_ticks,
            v_th,
            v_reset: 0.0,
            refractory: 2 * MS,
            fireable_at: 0,
        }
    }
    pub fn v_at(&self, now: Tick) -> f64 {
        let dt = now.saturating_sub(self.t_last) as f64;
        self.v_last * (-dt / self.tau).exp()
    }
    fn decay_to(&mut self, now: Tick) {
        self.v_last = self.v_at(now);
        self.t_last = now;
    }
    pub fn receive(&mut self, now: Tick, w: f64) -> bool {
        self.decay_to(now);
        if now < self.fireable_at {
            return false;
        }
        self.v_last += w;
        if self.v_last >= self.v_th {
            self.v_last = self.v_reset;
            self.fireable_at = now + self.refractory;
            true
        } else {
            false
        }
    }
}

// ---------------- 12.2 the event queue ----------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Event {
    pub at: Tick,
    pub target: NeuronId,
    pub weight_mv: i32,
}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        self.at
            .cmp(&other.at)
            .then(self.target.cmp(&other.target))
            .then(self.weight_mv.cmp(&other.weight_mv))
    }
}
impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Default)]
pub struct Queue {
    heap: BinaryHeap<Reverse<Event>>,
}

impl Queue {
    pub fn new() -> Self {
        Queue { heap: BinaryHeap::new() }
    }
    pub fn push(&mut self, e: Event) {
        self.heap.push(Reverse(e));
    }
    pub fn pop(&mut self) -> Option<Event> {
        self.heap.pop().map(|Reverse(e)| e)
    }
    pub fn len(&self) -> usize {
        self.heap.len()
    }
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

// ---------------- 12.3 what a unit holds ----------------

#[derive(Debug, Clone, Copy)]
pub struct Synapse {
    pub to: NeuronId,
    pub weight_mv: i32,
    pub delay: Tick,
}

#[derive(Debug, Clone)]
pub struct Unit {
    pub cell: Neuron,
    pub out: Vec<Synapse>,
    pub tag: Vec<u32>,
}

pub struct Brain {
    pub units: Vec<Unit>,
    pub queue: Queue,
    pub now: Tick,
}

impl Brain {
    pub fn new(n: usize, tau_ticks: f64, v_th: f64) -> Self {
        Brain {
            units: vec![
                Unit {
                    cell: Neuron::new(tau_ticks, v_th),
                    out: Vec::new(),
                    tag: Vec::new()
                };
                n
            ],
            queue: Queue::new(),
            now: 0,
        }
    }

    pub fn run(&mut self, until: Tick, max_spikes: usize) -> Vec<(Tick, NeuronId)> {
        let mut fired = Vec::new();
        while let Some(e) = self.queue.pop() {
            if e.at > until {
                self.queue.push(e);
                break;
            }
            if fired.len() >= max_spikes {
                break;
            }
            self.now = e.at;
            let unit = &mut self.units[e.target];
            if unit.cell.receive(e.at, e.weight_mv as f64) {
                fired.push((e.at, e.target));
                for s in &unit.out {
                    self.queue.push(Event {
                        at: e.at + s.delay,
                        target: s.to,
                        weight_mv: s.weight_mv,
                    });
                }
            }
        }
        fired
    }
}

// ---------------- 12.4 the splitter ----------------

pub const N: usize = 2000;
pub const K: usize = 40;
pub const FANOUT: usize = 8;

pub fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

pub fn sentences(tutorial: &str) -> Vec<&str> {
    tutorial
        .split(|c| c == '.' || c == '?' || c == '!' || c == '\n')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

pub fn normalize(word: &str) -> String {
    word.chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase()
}

pub fn top_k(score: &[u32], k: usize) -> Vec<NeuronId> {
    let mut idx: Vec<usize> = (0..score.len()).filter(|&i| score[i] > 0).collect();
    idx.sort_unstable_by(|&a, &b| score[b].cmp(&score[a]).then(a.cmp(&b)));
    idx.truncate(k);
    idx.sort_unstable();
    idx
}

pub fn encode(chunk: &str) -> Vec<NeuronId> {
    let mut score = vec![0u32; N];
    for word in chunk.split_whitespace() {
        let w = normalize(word);
        if w.is_empty() {
            continue;
        }
        for i in 0..FANOUT {
            let idx = (fnv1a(w.as_bytes(), i as u64) % N as u64) as usize;
            score[idx] += 1;
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
            Ordering::Equal => {
                n += 1;
                i += 1;
                j += 1;
            }
        }
    }
    n
}

pub fn install(brain: &mut Brain, chunk_id: u32, code: &[NeuronId], w_mv: i32, delay: Tick) {
    for &id in code {
        brain.units[id].tag.push(chunk_id);
        for &other in code {
            if other != id {
                brain.units[id].out.push(Synapse {
                    to: other,
                    weight_mv: w_mv,
                    delay,
                });
            }
        }
    }
}

pub fn recall(brain: &mut Brain, seed: &[NeuronId], now: Tick, window: Tick) -> Vec<NeuronId> {
    for &id in seed {
        brain.queue.push(Event { at: now, target: id, weight_mv: 100 });
    }
    let mut who: Vec<NeuronId> = brain
        .run(now + window, 10_000)
        .into_iter()
        .map(|(_, id)| id)
        .collect();
    who.sort_unstable();
    who.dedup();
    who
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decay_is_exponential() {
        let mut n = Neuron::new(10.0 * MS as f64, 15.0);
        n.receive(0, 10.0);
        let v = n.v_at(10 * MS);
        assert!((v - 10.0 * (-1.0f64).exp()).abs() < 1e-12, "got {v}");
    }

    #[test]
    fn fires_only_at_threshold() {
        let mut n = Neuron::new(10.0 * MS as f64, 15.0);
        assert!(!n.receive(0, 10.0));
        assert!(n.receive(1, 10.0));
    }

    #[test]
    fn queue_pops_earliest_first() {
        let mut q = Queue::new();
        q.push(Event { at: 50, target: 1, weight_mv: 1 });
        q.push(Event { at: 10, target: 2, weight_mv: 1 });
        q.push(Event { at: 30, target: 3, weight_mv: 1 });
        assert_eq!(q.pop().unwrap().at, 10);
        assert_eq!(q.pop().unwrap().at, 30);
        assert_eq!(q.pop().unwrap().at, 50);
    }

    #[test]
    fn code_is_sparse_sorted_and_deterministic() {
        let c = encode("the cat sat on the mat");
        assert!(c.len() <= K);
        assert!(c.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(c, encode("the cat sat on the mat"));
    }

    #[test]
    fn related_text_overlaps_more_than_unrelated() {
        let a = encode("a neuron fires when its membrane potential crosses threshold");
        let b = encode("a neuron fires when its membrane voltage passes the threshold");
        let c = encode("cookery involves butter sugar flour and a hot oven");
        let (ab, ac) = (overlap(&a, &b), overlap(&a, &c));
        println!("related={ab} unrelated={ac}");
        assert!(ab > ac, "related={ab} unrelated={ac}");
    }

    #[test]
    fn partial_cue_completes_the_pattern() {
        let mut brain = Brain::new(N, 10.0 * MS as f64, 15.0);
        let text = "the verb agrees with the subject in number and person";
        let code = encode(text);
        install(&mut brain, 0, &code, 2, MS);
        let cue: Vec<NeuronId> = code.iter().take(code.len() / 4).copied().collect();
        let out = recall(&mut brain, &cue, 0, 50 * MS);
        let hit = overlap(&out, &code);
        println!("cue={} recovered={} of {}", cue.len(), hit, code.len());
        assert!(hit > cue.len(), "pattern completion failed: {hit} <= {}", cue.len());
    }
}

// appended: lets an experiment change the refractory period without rebuilding the type
impl Brain {
    pub fn set_refractory(&mut self, r: Tick) {
        for u in &mut self.units {
            u.cell.refractory = r;
        }
    }
}

/// corrected top_k. the original broke score ties with `.then(a.cmp(&b))`, i.e.
/// lowest index wins, which systematically drags every code toward low unit
/// numbers and manufactures overlap that has nothing to do with meaning.
/// tie-break on a hash of the index instead: still deterministic, but unbiased.
pub fn top_k_hashed(score: &[u32], k: usize) -> Vec<NeuronId> {
    let mut idx: Vec<usize> = (0..score.len()).filter(|&i| score[i] > 0).collect();
    idx.sort_unstable_by(|&a, &b| {
        score[b]
            .cmp(&score[a])
            .then_with(|| fnv1a(&a.to_le_bytes(), 0xA11).cmp(&fnv1a(&b.to_le_bytes(), 0xA11)))
    });
    idx.truncate(k);
    idx.sort_unstable();
    idx
}
