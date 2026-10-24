// TESTING / DESIGN ONLY.
//
// HDC — true / false BY COMMITTEE. Binary spatter codes (Kanerva) on chain.rs's family tree.
//
//   cargo run --release --example hdc
//
// The brain basis, in plain words. A neuron says only yes (a spike) or no (silence). And
// the wire between two neurons is bad: a synapse passes a spike only ~10-50% of the time.
// So no single neuron can be trusted with a fact. The brain gets truth from a COMMITTEE:
// thousands of neurons each cast a yes/no vote, and the SIZE of the majority is the degree
// of belief — a landslide is "true", a dead heat is "not sure", a landslide the other way
// is "false". Losing half the votes on the wire only makes the landslide a bit smaller.
//
// Here that committee is a hypervector of D bits (D ~ 8192), one bit per neuron:
//
//   concept   a random D-bit pattern. two random patterns agree on ~half their bits, so
//             their similarity 1 - 2*hamming/D is ~0, give or take 1/sqrt(D).
//   BIND      role (x) value = XOR. "mother is p2_10" is one new pattern, unlike both
//             parts. XOR undoes itself, so  (role x value) x role = value.
//   BUNDLE    a person's RECORD = bitwise majority vote of their bound facts (ties broken
//             by one fixed random pattern). one D-bit vector holds several facts at once.
//   CLEANUP   unbinding gives a NOISY copy of the answer. compare it to every person
//             (the item memory) and take the best — with a confidence that comes from the
//             known random-similarity distribution, not from a hand-set threshold.
//
// The confidence (no magic numbers). For D bits, a random person's z = sim * sqrt(n) is
// N(0,1); the true answer's z is N(s*sqrt(n), 1-s^2), where s is the similarity a bundle of
// k facts leaves on each member — known exactly from k by counting votes. With even odds
// that the fact exists at all (spread evenly over the N people), the chance the best match
// is right is  LR_best / (N + sum LR),  LR = likelihood ratio true/random. She answers when
// that chance is above 1/2: the break-even point under right-minus-wrong scoring. Below it
// she says "I don't know". The same rule is used everywhere below; nothing is tuned.
//
// Same world as chain.rs (same seed, sizes, 5% withheld links), replayed draw for draw so
// the missing links and godparents are the same people. The derived (grandparent) facts are
// NOT stored here: every chained answer is computed.
use std::time::Instant;

const GENS: usize = 6;
const PER_GEN: usize = 2000;
const BASE: [&str; 2] = ["mother", "father"];
/// derived relations and their TRUE programs — copied from chain.rs, used to generate the
/// questions and to grade. unlike chain.rs, the program is GIVEN here (see section 3).
const DERIVED: [(&str, &[usize]); 5] = [
    ("maternal_grandmother", &[0, 0]),
    ("maternal_grandfather", &[0, 1]),
    ("paternal_grandmother", &[1, 0]),
    ("paternal_grandfather", &[1, 1]),
    ("great_grandmother", &[0, 0, 0]),
];
/// chain.rs defaults: no deliberately wrong derived facts, 5% of base links withheld.
const NOISE: f64 = 0.0;
const WITHHELD: f64 = 5.0;
/// roles bound into a record. godparent is chain.rs's no-rule control, stored for the
/// training half as there.
const ROLES: [&str; 3] = ["mother", "father", "godparent"];

/// chain.rs's generator, unchanged, so the world is the same world.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
    fn chance(&mut self, pct: f64) -> bool { ((self.next() % 10_000) as f64) < pct * 100.0 }
}

/// splitmix64 for the hypervectors, the synapse failures and the samples — xorshift's
/// neighbouring outputs are too alike to be trusted as independent neurons.
struct Mix(u64);
impl Mix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize { (self.next() % n as u64) as usize }
    fn bits(&mut self, w: usize) -> Vec<u64> { (0..w).map(|_| self.next()).collect() }
}

// ---------------------------------------------------------------- the world (chain.rs)

struct World {
    n: usize,
    parent: Vec<[usize; 2]>,
    train: Vec<bool>,
    stored: Vec<[bool; 2]>,
    god: Vec<usize>,
    god_stored: Vec<bool>,
    held: usize,
    n_base: usize,
}
fn walk(parent: &[[usize; 2]], prog: &[usize], x: usize) -> Option<usize> {
    let mut c = x;
    for &b in prog { c = parent[c][b]; if c == usize::MAX { return None; } }
    Some(c)
}
fn name(p: usize) -> String { format!("p{}_{}", p / PER_GEN, p % PER_GEN) }

fn world() -> World {
    let mut rng = Rng(0xC4A1_2026_1013_0001);
    let n = GENS * PER_GEN;
    let mut parent = vec![[usize::MAX; 2]; n];
    for g in 1..GENS {
        for i in 0..PER_GEN {
            let p = g * PER_GEN + i;
            parent[p][0] = (g - 1) * PER_GEN + 2 * rng.below(PER_GEN / 2);
            parent[p][1] = (g - 1) * PER_GEN + 2 * rng.below(PER_GEN / 2) + 1;
        }
    }
    let train: Vec<bool> = (0..n).map(|_| rng.below(2) == 0).collect();
    let mut stored = vec![[false; 2]; n];
    let (mut held, mut n_base) = (0usize, 0usize);
    for p in PER_GEN..n {
        for b in 0..2 {
            if rng.chance(WITHHELD) { held += 1; continue; }
            stored[p][b] = true;
            n_base += 1;
        }
    }
    // chain.rs stores derived facts here. they are not stored in this test, but their
    // draws are replayed so the godparents below are the same people.
    for (_, prog) in DERIVED {
        for p in 0..n {
            if !train[p] || walk(&parent, prog, p).is_none() { continue; }
            if rng.chance(NOISE) { rng.below(PER_GEN); }
        }
    }
    let mut god = vec![usize::MAX; n];
    let mut god_stored = vec![false; n];
    for p in PER_GEN..n {
        god[p] = (p / PER_GEN - 1) * PER_GEN + rng.below(PER_GEN);
        god_stored[p] = train[p];
    }
    World { n, parent, train, stored, god, god_stored, held, n_base }
}

// ---------------------------------------------------------------- the committee

struct Hdc {
    d: usize,
    w: usize,
    people: Vec<u64>,   // item memory: one random pattern per person
    role: Vec<Vec<u64>>,
    tie: Vec<u64>,
    rec: Vec<u64>,      // one record per person: the majority vote of their bound facts
    k: Vec<usize>,      // how many facts went into each record
}
impl Hdc {
    fn person(&self, i: usize) -> &[u64] { &self.people[i * self.w..(i + 1) * self.w] }
    fn record(&self, i: usize) -> &[u64] { &self.rec[i * self.w..(i + 1) * self.w] }
}

fn xor(a: &[u64], b: &[u64]) -> Vec<u64> { a.iter().zip(b).map(|(x, y)| x ^ y).collect() }

/// bitwise majority. an even number of voters gets the fixed random tie-breaker as one
/// more voter; zero voters leaves just the tie-breaker (a record of nothing is noise).
fn bundle(items: &[Vec<u64>], tie: &[u64]) -> Vec<u64> {
    let w = tie.len();
    let mut v: Vec<&[u64]> = items.iter().map(|x| x.as_slice()).collect();
    if v.len() % 2 == 0 { v.push(tie); }
    if v.len() == 1 { return v[0].to_vec(); }
    if v.len() == 3 {
        return (0..w).map(|j| { let (a, b, c) = (v[0][j], v[1][j], v[2][j]); (a & b) | (a & c) | (b & c) }).collect();
    }
    let need = (v.len() / 2 + 1) as u16;
    let mut out = vec![0u64; w];
    let mut cnt = [0u16; 64];
    for j in 0..w {
        cnt.fill(0);
        for x in &v { let x = x[j]; for b in 0..64 { cnt[b] += ((x >> b) & 1) as u16; } }
        for b in 0..64 { if cnt[b] >= need { out[j] |= 1 << b; } }
    }
    out
}

fn build(d: usize, wd: &World) -> Hdc {
    let w = d / 64;
    let mut mx = Mix(0x4D0C_2026_1015_0000 ^ d as u64);
    let people = mx.bits(wd.n * w);
    let role: Vec<Vec<u64>> = (0..ROLES.len()).map(|_| mx.bits(w)).collect();
    let tie = mx.bits(w);
    let mut h = Hdc { d, w, people, role, tie, rec: Vec::with_capacity(wd.n * w), k: Vec::with_capacity(wd.n) };
    for x in 0..wd.n {
        let mut pairs: Vec<Vec<u64>> = Vec::new();
        for r in 0..2 { if wd.stored[x][r] { pairs.push(xor(&h.role[r], h.person(wd.parent[x][r]))); } }
        if wd.god_stored[x] { pairs.push(xor(&h.role[2], h.person(wd.god[x]))); }
        let b = bundle(&pairs, &h.tie);
        h.rec.extend_from_slice(&b);
        h.k.push(pairs.len());
    }
    h
}

/// the similarity a bundle of k facts leaves on each member: the chance the member's bit
/// survives the vote, counted exactly over the other voters. k=1 -> 1, k=2,3 -> 0.5.
fn member_sim(k: usize, lnf: &[f64]) -> f64 {
    if k == 0 { return 0.0; }
    let v = if k % 2 == 0 { k + 1 } else { k };
    let m = v - 1;
    let mut p = 0.0;
    for j in m / 2..=m { p += (lnf[m] - lnf[j] - lnf[m - j] - m as f64 * std::f64::consts::LN_2).exp(); }
    (2.0 * p - 1.0).clamp(0.0, 1.0)
}

/// hamming distance of q to every row, over the transmitted bits m only.
fn hams(q: &[u64], m: &[u64], items: &[u64], out: &mut [u32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("popcnt") {
            unsafe { return hams_fast(q, m, items, out); }
        }
    }
    hams_plain(q, m, items, out)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt")]
unsafe fn hams_fast(q: &[u64], m: &[u64], items: &[u64], out: &mut [u32]) { hams_plain(q, m, items, out) }
#[inline(always)]
fn hams_plain(q: &[u64], m: &[u64], items: &[u64], out: &mut [u32]) {
    for (o, it) in out.iter_mut().zip(items.chunks_exact(q.len())) {
        *o = q.iter().zip(it).zip(m).map(|((a, b), c)| ((a ^ b) & c).count_ones()).sum();
    }
}
fn ham1(a: &[u64], b: &[u64], m: &[u64]) -> u32 {
    a.iter().zip(b).zip(m).map(|((x, y), c)| ((x ^ y) & c).count_ones()).sum()
}

/// which bits reach the reader: each one independently with probability p. a failed bit
/// casts no vote. (k redundant synapses per bit pass it if ANY passes: p -> 1-(1-p)^k.)
fn mask(mx: &mut Mix, w: usize, p: f64) -> (Vec<u64>, u32) {
    if p >= 1.0 { return (vec![!0u64; w], (w * 64) as u32); }
    let t = (p * 4_294_967_296.0) as u64;
    let m: Vec<u64> = (0..w).map(|_| {
        let mut v = 0u64;
        for b in (0..64).step_by(2) {
            let r = mx.next();
            if (r & 0xFFFF_FFFF) < t { v |= 1 << b; }
            if (r >> 32) < t { v |= 1 << (b + 1); }
        }
        v
    }).collect();
    let n = m.iter().map(|x| x.count_ones()).sum();
    (m, n)
}

/// log likelihood ratio "this is the stored member" vs "this is a random pattern", for a
/// z-score read over n transmitted bits from a bundle whose members keep similarity s.
struct Llr { mu: f64, v: f64, c: f64 }
impl Llr {
    fn new(s: f64, n: u32) -> Llr {
        let n = n.max(1) as f64;
        let v = (1.0 - s * s).max(1.0 / n);   // floor: one bit's worth of resolution
        Llr { mu: s * n.sqrt(), v, c: -0.5 * v.ln() }
    }
    fn at(&self, z: f64) -> f64 { -(z - self.mu).powi(2) / (2.0 * self.v) + self.c + 0.5 * z * z }
}

#[derive(Clone, Copy)]
struct Ans { who: usize, conf: f64 }

/// cleanup against the item memory of all N people. conf = P(best match is the answer).
fn cleanup(h: &Hdc, q: &[u64], m: &[u64], n: u32, s: f64, buf: &mut Vec<u32>) -> Ans {
    let np = h.people.len() / h.w;
    buf.resize(np, 0);
    if n == 0 { return Ans { who: 0, conf: 0.0 }; }
    hams(q, m, &h.people, buf);
    let (nf, l) = (n as f64, Llr::new(s, n));
    let sq = nf.sqrt();
    let (mut best, mut bl) = (0usize, f64::NEG_INFINITY);
    for (i, &hd) in buf.iter().enumerate() {
        let v = l.at((nf - 2.0 * hd as f64) / sq);
        if v > bl { bl = v; best = i; }
    }
    let ln_n = (np as f64).ln();
    let top = bl.max(ln_n);
    let mut sum = (ln_n - top).exp();
    for &hd in buf.iter() {
        let v = l.at((nf - 2.0 * hd as f64) / sq) - top;
        if v > -40.0 { sum += v.exp(); }
    }
    Ans { who: best, conf: (bl - top).exp() / sum }
}

/// one read of "role of x": unbind x's record, pass it over unreliable synapses, clean up.
fn read(h: &Hdc, x: usize, r: usize, p: f64, mx: &mut Mix, st: &[f64], buf: &mut Vec<u32>) -> Ans {
    let q = xor(h.record(x), &h.role[r]);
    let (m, n) = mask(mx, h.w, p);
    cleanup(h, &q, &m, n, st[h.k[x]], buf)
}

/// a chain of reads: each hop cleans up to a whole person before the next unbind. the
/// chain's confidence is the product of its hops'.
fn chain(h: &Hdc, x: usize, prog: &[usize], p: f64, mx: &mut Mix, st: &[f64], buf: &mut Vec<u32>) -> Ans {
    let mut a = Ans { who: x, conf: 1.0 };
    for &r in prog {
        let b = read(h, a.who, r, p, mx, st, buf);
        a = Ans { who: b.who, conf: a.conf * b.conf };
    }
    a
}

/// right / wrong / abstain tally, scored the user's way.
#[derive(Default, Clone, Copy)]
struct T { q: usize, right: usize, wrong: usize }
impl T {
    fn add(&mut self, a: Ans, truth: Option<usize>) {
        self.q += 1;
        if a.conf > 0.5 { if Some(a.who) == truth { self.right += 1; } else { self.wrong += 1; } }
    }
    fn pc(&self, x: usize) -> f64 { x as f64 / self.q.max(1) as f64 * 100.0 }
    fn row(&self) -> String {
        let ans = self.right + self.wrong;
        format!("{:>7} {:>8.2}% {:>7.2}% {:>8.2}% {:>8.2}% {:>9.2}%", self.q, self.pc(self.right), self.pc(self.wrong),
            self.pc(self.q - ans), self.pc(self.right) - self.pc(self.wrong),
            self.right as f64 / ans.max(1) as f64 * 100.0)
    }
    fn cell(&self) -> String { format!("{:.1}/{:.1}", self.pc(self.right), self.pc(self.wrong)) }
}
const HEAD: &str = "  asked    right    wrong  abstain  right-wrong precision";

fn auc(t: &[f64], f: &[f64]) -> f64 {
    let mut s = 0.0;
    for a in t { for b in f { if a > b { s += 1.0; } else if a == b { s += 0.5; } } }
    s / (t.len() * f.len()).max(1) as f64
}

fn main() {
    let t0 = Instant::now();
    let wd = world();
    let n = wd.n;
    let lnf: Vec<f64> = (0..4096).scan(0.0, |a, i: usize| { if i > 0 { *a += (i as f64).ln(); } Some(*a) }).collect();
    let st: Vec<f64> = (0..=2048).map(|k| member_sim(k, &lnf)).collect();
    let n_god = wd.god_stored.iter().filter(|&&b| b).count();
    println!("== hdc: true/false by committee — binary spatter codes on chain.rs's family tree ==");
    println!("   {n} people, {GENS} generations; stored {} mother/father facts ({} withheld = missing links) \
+ {n_god} godparent facts (training half)", wd.n_base, wd.held);
    println!("   a person's RECORD = majority vote of bind(ROLE, value) over their stored facts (0-3 pairs)");
    println!("   answer rule everywhere: answer iff P(best match is right) > 1/2, from the random-similarity law\n");

    let mut hs: Vec<Hdc> = [256usize, 1024, 4096, 8192, 16384].iter().map(|&d| build(d, &wd)).collect();
    let hd = |hs: &Vec<Hdc>, d: usize| hs.iter().position(|h| h.d == d).unwrap();
    println!("   built item memory + records for D = 256, 1024, 4096, 8192, 16384 at {:?}\n", t0.elapsed());
    let mut buf: Vec<u32> = Vec::new();
    let mut mx = Mix(0x5A3E_2026_1015_0001);

    // ---------------------------------------------------------------- 1. forward
    let h = &hs[hd(&hs, 8192)];
    println!("== 1. FORWARD recall: unbind x's record with ROLE, clean up (D=8192, every synapse works) ==");
    let memo: Vec<[Ans; 2]> = (0..n).map(|x| [read(h, x, 0, 1.0, &mut mx, &st, &mut buf), read(h, x, 1, 1.0, &mut mx, &st, &mut buf)]).collect();
    println!("   ({} cleanups against {n} people at {:?})", 2 * n, t0.elapsed());
    println!("   {:<38}{HEAD}", "query");
    for r in 0..2 {
        let mut t = T::default();
        for x in 0..n { if wd.stored[x][r] { t.add(memo[x][r], Some(wd.parent[x][r])); } }
        println!("   {:<38}{}", format!("{}(x), stored", BASE[r]), t.row());
    }
    let mut gs = T::default();
    let mut gn = T::default();
    for _ in 0..2000 {
        let x = loop { let x = PER_GEN + mx.below(n - PER_GEN); if wd.god_stored[x] { break x; } };
        gs.add(read(h, x, 2, 1.0, &mut mx, &st, &mut buf), Some(wd.god[x]));
        let x = loop { let x = PER_GEN + mx.below(n - PER_GEN); if !wd.god_stored[x] { break x; } };
        gn.add(read(h, x, 2, 1.0, &mut mx, &st, &mut buf), None);
    }
    println!("   {:<38}{}", "godparent(x), stored (2000 sampled)", gs.row());
    let mut miss = T::default();
    let mut g0 = T::default();
    for x in 0..n {
        for r in 0..2 {
            if x < PER_GEN { g0.add(memo[x][r], None); } else if !wd.stored[x][r] { miss.add(memo[x][r], None); }
        }
    }
    println!("   -- never stored, so the right reply is \"I don't know\" (any answer counts wrong):");
    println!("   {:<38}{}", "mother/father, link withheld", miss.row());
    println!("   {:<38}{}", "mother/father of generation 0", g0.row());
    println!("   {:<38}{}", "godparent(x), held-out half (2000)", gn.row());
    println!();

    // ---------------------------------------------------------------- 2. reverse
    println!("== 2. REVERSE recall (never stored in reverse): \"whose mother is y?\" ==");
    println!("   probe = bind(ROLE_mother, y), compared with every RECORD (XOR binding is symmetric).");
    println!("   a record is named when P(it holds this pair) > 1/2, same law. graded against the");
    println!("   children whose link was STORED (a withheld link cannot be in any record).");
    println!("   {:<8} {:>6} {:>10} {:>8} {:>7} {:>7} {:>8} {:>10} {:>8}",
        "role", "asked", "exact-set", "named", "right", "wrong", "missed", "precision", "recall");
    let nrec = n as f64;
    let mut rev_examples: Vec<(usize, usize, Vec<usize>)> = Vec::new();
    for r in 0..2 {
        let (mut asked, mut exact, mut named, mut right, mut wrong, mut missed, mut want) = (0usize, 0, 0, 0, 0, 0, 0);
        let ones = vec![!0u64; h.w];
        let mut rb = vec![0u32; n];
        for _ in 0..1000 {
            // a woman (even index) or man (odd) from generations 0..4 — some have no children
            let y = (mx.below((GENS - 1) * PER_GEN / 2)) * 2 + r;
            let probe = xor(&h.role[r], h.person(y));
            hams(&probe, &ones, &h.rec, &mut rb);
            let nf = h.d as f64;
            let said: Vec<usize> = (0..n).filter(|&x| {
                let l = Llr::new(st[h.k[x]], h.d as u32);
                l.at((nf - 2.0 * rb[x] as f64) / nf.sqrt()) > nrec.ln()
            }).collect();
            let kids: Vec<usize> = (0..n).filter(|&x| wd.parent[x][r] == y && wd.stored[x][r]).collect();
            asked += 1;
            if said == kids { exact += 1; }
            named += said.len();
            want += kids.len();
            let hit = said.iter().filter(|x| kids.contains(x)).count();
            right += hit;
            wrong += said.len() - hit;
            missed += kids.len() - hit;
            if rev_examples.len() < 2 + 2 * r && !kids.is_empty() { rev_examples.push((r, y, said.clone())); }
        }
        println!("   {:<8} {:>6} {:>9.2}% {:>8} {:>7} {:>7} {:>8} {:>9.2}% {:>7.2}%", BASE[r], asked,
            exact as f64 / asked as f64 * 100.0, named, right, wrong, missed,
            right as f64 / named.max(1) as f64 * 100.0, right as f64 / want.max(1) as f64 * 100.0);
    }
    for (r, y, said) in &rev_examples {
        let all: Vec<String> = (0..n).filter(|&x| wd.parent[x][*r] == *y)
            .map(|x| format!("{}{}", name(x), if wd.stored[x][*r] { "" } else { "*" })).collect();
        let s: Vec<String> = said.iter().map(|&x| name(x)).collect();
        println!("   you> whose {} is {}?   ava> {}   (truth {}; * = link withheld)", BASE[*r], name(*y),
            if s.is_empty() { "nobody I know".into() } else { s.join(", ") }, all.join(", "));
    }
    println!();

    // ---------------------------------------------------------------- 3. chaining
    println!("== 3. CHAINING (never taught together): unbind -> clean up -> unbind -> clean up ==");
    println!("   every held-out person of chain.rs (the half with no derived facts), same five relations.");
    println!("   {:<22}{HEAD}  ceiling", "relation");
    let mut tot = T::default();
    let mut ceil_tot = 0usize;
    for (rel, prog) in DERIVED {
        let mut t = T::default();
        let mut ceil = 0usize;
        for x in 0..n {
            if wd.train[x] { continue; }
            let Some(y) = walk(&wd.parent, prog, x) else { continue };
            // the memo IS the read: at p = 1 a read is deterministic
            let mut a = Ans { who: x, conf: 1.0 };
            for &r in prog { let b = memo[a.who][r]; a = Ans { who: b.who, conf: a.conf * b.conf }; }
            t.add(a, Some(y));
            // ceiling: every link on the true path was stored
            let mut c = x;
            let mut ok = true;
            for &r in prog { if !wd.stored[c][r] { ok = false; break; } c = wd.parent[c][r]; }
            if ok { ceil += 1; }
        }
        println!("   {:<22}{}  {:>6.2}%", rel, t.row(), ceil as f64 / t.q.max(1) as f64 * 100.0);
        tot.q += t.q; tot.right += t.right; tot.wrong += t.wrong; ceil_tot += ceil;
    }
    println!("   {:<22}{}  {:>6.2}%", "ALL FIVE", tot.row(), ceil_tot as f64 / tot.q.max(1) as f64 * 100.0);
    println!("   chain.rs reported: composed 90.11%, 0 wrong. NOT the same task: chain.rs had to INDUCE the");
    println!("   program from examples; here the program is GIVEN. both are capped by the withheld links (ceiling).");
    for x in [3 * PER_GEN + 7, 4 * PER_GEN + 12, 5 * PER_GEN + 33] {
        for (rel, prog) in [("maternal_grandfather", &[0usize, 1][..]), ("great_grandmother", &[0, 0, 0][..])] {
            let mut a = Ans { who: x, conf: 1.0 };
            for &r in prog { let b = memo[a.who][r]; a = Ans { who: b.who, conf: a.conf * b.conf }; }
            let said = if a.conf > 0.5 { format!("{} (conf {:.3})", name(a.who), a.conf) } else { format!("I don't know (conf {:.3})", a.conf) };
            let real = walk(&wd.parent, prog, x).map_or("-".into(), name);
            println!("   you> who is {}'s {rel}?   ava> {said}   (truth {real})", name(x));
        }
    }
    println!("   ({:?})\n", t0.elapsed());

    // ---------------------------------------------------------------- 4. capacity
    println!("== 4. CAPACITY of one committee: N random role/value pairs bundled into ONE vector ==");
    println!("   recover each value (unbind, clean up against all {n} people). cells: right% / wrong%, rest abstain.");
    let cap_d = [1024usize, 4096, 8192];
    print!("   {:>6} {:>8}", "N", "sim/fact");
    for d in cap_d { print!("  {:>15}", format!("D={d}")); }
    println!("   (z of the true answer at D=8192)");
    let mut cmx = Mix(0xCA9A_2026_1015_0004);
    for e in 0..=10 {
        let nn = 1usize << e;
        let trials = (512 / nn).max(1);
        print!("   {:>6} {:>8.3}", nn, st[nn]);
        for d in cap_d {
            let h = &hs[hd(&hs, d)];
            let ones = vec![!0u64; h.w];
            let mut t = T::default();
            for _ in 0..trials {
                let roles: Vec<Vec<u64>> = (0..nn).map(|_| cmx.bits(h.w)).collect();
                let vals: Vec<usize> = (0..nn).map(|_| cmx.below(n)).collect();
                let pairs: Vec<Vec<u64>> = (0..nn).map(|i| xor(&roles[i], h.person(vals[i]))).collect();
                let b = bundle(&pairs, &h.tie);
                for i in 0..nn {
                    let q = xor(&b, &roles[i]);
                    t.add(cleanup(h, &q, &ones, h.d as u32, st[nn], &mut buf), Some(vals[i]));
                }
            }
            print!("  {:>15}", t.cell());
        }
        println!("   {:>6.1}", st[nn] * 8192f64.sqrt());
    }
    println!("   ({:?})\n", t0.elapsed());

    // ---------------------------------------------------------------- 5. noisy synapses
    println!("== 5. NOISY SYNAPSES: each bit of a read reaches the reader with probability p ==");
    println!("   a failed bit casts no vote; z is taken over the bits that arrived. 300 forward + 300 two-hop");
    println!("   queries whose links ARE stored (so p = 1 is 100%); fresh failures on every read.");
    let ps = [1.0, 0.9, 0.7, 0.5, 0.3, 0.1];
    let nd = [1024usize, 4096, 8192, 16384];
    let fq: Vec<(usize, usize)> = (0..300).map(|_| loop {
        let x = PER_GEN + mx.below(n - PER_GEN);
        let r = mx.below(2);
        if wd.stored[x][r] { break (x, r); }
    }).collect();
    let cq: Vec<(usize, usize)> = (0..300).map(|_| loop {
        let x = 2 * PER_GEN + mx.below(n - 2 * PER_GEN);
        let pi = mx.below(4);
        let prog = DERIVED[pi].1;
        if wd.train[x] { continue; }
        if wd.stored[x][prog[0]] && wd.stored[wd.parent[x][prog[0]]][prog[1]] { break (x, pi); }
    }).collect();
    let mut grid = vec![vec![(T::default(), T::default()); nd.len()]; ps.len()];
    for (di, &d) in nd.iter().enumerate() {
        let h = &hs[hd(&hs, d)];
        for (pi, &p) in ps.iter().enumerate() {
            let mut nmx = Mix(0x5E7A_2026_1015_0005 ^ ((d as u64) << 8) ^ pi as u64);
            let (mut f, mut c) = (T::default(), T::default());
            for &(x, r) in &fq { f.add(read(h, x, r, p, &mut nmx, &st, &mut buf), Some(wd.parent[x][r])); }
            for &(x, i) in &cq {
                let prog = DERIVED[i].1;
                c.add(chain(h, x, prog, p, &mut nmx, &st, &mut buf), walk(&wd.parent, prog, x));
            }
            grid[pi][di] = (f, c);
        }
    }
    for (label, which) in [("forward  mother/father(x)", 0usize), ("two-hop chain", 1)] {
        println!("   {label}: right% / wrong%  (abstain = the rest)");
        print!("   {:>6}", "p");
        for d in nd { print!("  {:>13}", format!("D={d}")); }
        println!();
        for (pi, &p) in ps.iter().enumerate() {
            print!("   {:>6.1}", p);
            for di in 0..nd.len() { let t = if which == 0 { grid[pi][di].0 } else { grid[pi][di].1 }; print!("  {:>13}", t.cell()); }
            println!();
        }
    }
    println!("   redundant synapses: k copies of each wire, the vote arrives if ANY copy fires (p' = 1-(1-p)^k), D=1024");
    println!("   {:>5} {:>4} {:>7}  {:>13}  {:>13}", "p", "k", "p'", "forward", "two-hop");
    let h = &hs[hd(&hs, 1024)];
    for (p, k) in [(0.1, 1), (0.1, 2), (0.1, 4), (0.1, 8), (0.1, 16), (0.3, 1), (0.3, 2), (0.3, 4)] {
        let pe = 1.0 - (1.0f64 - p).powi(k);
        let mut nmx = Mix(0x4EDD_2026_1015_0006 ^ ((k as u64) << 8) ^ (p * 10.0) as u64);
        let (mut f, mut c) = (T::default(), T::default());
        for &(x, r) in &fq { f.add(read(h, x, r, pe, &mut nmx, &st, &mut buf), Some(wd.parent[x][r])); }
        for &(x, i) in &cq { let prog = DERIVED[i].1; c.add(chain(h, x, prog, pe, &mut nmx, &st, &mut buf), walk(&wd.parent, prog, x)); }
        println!("   {:>5.1} {:>4} {:>7.3}  {:>13}  {:>13}", p, k, pe, f.cell(), c.cell());
    }
    println!("   ({:?})\n", t0.elapsed());

    // ---------------------------------------------------------------- 6. graded truth
    println!("== 6. GRADED TRUTH: \"is y the mother of x?\" — 2000 true statements, 2000 false ==");
    println!("   belief = P(true | z) from the same two laws, even odds. AUC is threshold-free. the bands");
    println!("   >=0.95 \"true\", <=0.05 \"false\", between \"not sure\" are for display only.");
    let tq: Vec<(usize, usize, usize)> = (0..2000).map(|_| loop {
        let x = PER_GEN + mx.below(n - PER_GEN);
        if !wd.stored[x][0] { continue; }
        let m = wd.parent[x][0];
        let f = loop { let f = (m / PER_GEN) * PER_GEN + 2 * mx.below(PER_GEN / 2); if f != m { break f; } };
        break (x, m, f);
    }).collect();
    println!("   {:>6} {:>4} {:>8}   {:>26}   {:>26}", "D", "p", "AUC", "TRUE said true/unsure/false", "FALSE said true/unsure/false");
    let mut hists: Vec<(usize, f64, Vec<f64>, Vec<f64>)> = Vec::new();
    for d in [256usize, 1024, 8192] {
        let h = &hs[hd(&hs, d)];
        for p in [1.0, 0.3, 0.1] {
            let mut gmx = Mix(0x7E57_2026_1015_0007 ^ ((d as u64) << 8) ^ (p * 10.0) as u64);
            let (mut ts, mut fs, mut tz, mut fz) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let (mut tb, mut fb) = ([0usize; 3], [0usize; 3]);
            for &(x, m, f) in &tq {
                let q = xor(h.record(x), &h.role[0]);
                for (y, is_true) in [(m, true), (f, false)] {
                    let (mk, nb) = mask(&mut gmx, h.w, p);
                    let nf = nb.max(1) as f64;
                    let z = (nf - 2.0 * ham1(&q, h.person(y), &mk) as f64) / nf.sqrt();
                    let l = Llr::new(st[h.k[x]], nb).at(z);
                    let belief = 1.0 / (1.0 + (-l).exp());
                    let band = if belief >= 0.95 { 0 } else if belief <= 0.05 { 2 } else { 1 };
                    if is_true { ts.push(l); tz.push(z); tb[band] += 1; } else { fs.push(l); fz.push(z); fb[band] += 1; }
                }
            }
            let pc = |c: usize| c as f64 / tq.len() as f64 * 100.0;
            println!("   {:>6} {:>4.1} {:>8.4}   {:>26}   {:>26}", d, p, auc(&ts, &fs),
                format!("{:.1} / {:.1} / {:.1}", pc(tb[0]), pc(tb[1]), pc(tb[2])),
                format!("{:.1} / {:.1} / {:.1}", pc(fb[0]), pc(fb[1]), pc(fb[2])));
            if (d == 8192 && p == 1.0) || (d == 256 && p == 0.1) || (d == 1024 && p == 0.1) { hists.push((d, p, tz, fz)); }
        }
    }
    let edges = [f64::NEG_INFINITY, -2.0, 0.0, 2.0, 4.0, 8.0, 16.0, 32.0, f64::INFINITY];
    let labels = ["<-2", "-2..0", "0..2", "2..4", "4..8", "8..16", "16..32", ">=32"];
    for (d, p, tz, fz) in &hists {
        println!("   z-score distribution, D={d} p={p}  (count of TRUE | FALSE statements)");
        let mut line = String::from("     ");
        for b in 0..8 {
            let c = |v: &Vec<f64>| v.iter().filter(|&&z| z >= edges[b] && z < edges[b + 1]).count();
            line += &format!(" {:>6}:{:>4}|{:<4}", labels[b], c(tz), c(fz));
        }
        println!("{line}");
    }
    println!("\n   total {:?}", t0.elapsed());
    println!("\nTSV\thdc\t{:.2}\t{}\t{:.2}", tot.pc(tot.right), tot.wrong, tot.pc(tot.right + tot.wrong));
    hs.clear();
}
