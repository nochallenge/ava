// TESTING / DESIGN ONLY. (v29 WP9 + WP7) Arithmetic as a learned procedure, learned from
// demonstration or from outcomes alone.
//
//   cargo run --release --example procedure
//
// Her arithmetic today is memorised text: 3.4% on sums she never heard (v24 school brain). People
// don't add by recalling every sum; they run a small procedure column by column. Here the column
// layout is given, as a teacher's page gives it to a child: right to left, missing digits are 0,
// one carry held in mind, a last carry written in front. What happens INSIDE a column is learned:
// a habit from what she sees in the column, (digit, digit, carry), to what she does (write a
// digit, carry or not). 200 situations, 20 possible acts.
//
//   M  memory:        every whole sum she was taught, answered only when seen exactly
//   D  demonstration: worked examples show each column's digit and carry; she counts them
//   R  reward only:   she answers "a + b" by running her habits and learns only whether the WHOLE
//                     answer was right. Each habit she used moves toward the outcome by
//                     (r - value) / experience, a running mean: the dopamine reward-prediction
//                     error with no learning-rate constant. She acts in proportion to her values,
//                     so a habit that keeps failing fades and no exploration rate is set by hand.
//                     Staged: 1-digit sums, then 2, then 3 (a curriculum).
// Pre-registered in "ava v29 - in depth" (WP9 + WP7).
//
// WP9b adds subtraction (the same column shape with a borrow) and long multiplication, which is a
// procedure that calls procedures: "number x digit" columns are learned (D or R), the page layout
// is given, and the rows are added by the addition habits learned above. Pre-registered as WP9b.
//
// WP9e: long division found from her own skills (ADD, SUB as comparison, TIMES, her number x digit
// rows): which way of choosing a quotient digit, and which page, explain 20 worked answers?
// WP9f: her own add-back check as the only teacher. WP9g: AVA_EXPLORE=uncertain, exploring where
// experience is thin (Thompson sampling from her own counts) instead of in proportion to value;
// it PASSED and is the default (AVA_EXPLORE=proportional reproduces the WP9-WP9f runs).
// WP17a: AVA_CURRICULUM=self, she chooses what to practise by her learning progress. WP17b:
// AVA_CURRICULUM=clear, the same pull divided by how many habits a problem there uses.
use std::collections::HashMap;

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// a number of exactly `d` digits (1-digit numbers include 0)
fn number(d: u32, x: &mut u64) -> u64 {
    if d == 1 { return splitmix(x) % 10; }
    let lo = 10u64.pow(d - 1);
    lo + splitmix(x) % (9 * lo)
}

fn digits(n: u64) -> Vec<usize> {
    let mut v = Vec::new();
    let mut m = n;
    loop { v.push((m % 10) as usize); m /= 10; if m == 0 { break; } }
    v
}

/// a column situation: (digit of a, digit of b, carry held)
fn state(a: usize, b: usize, c: usize) -> usize { (a * 10 + b) * 2 + c }
/// an act: (digit written, carry passed on)
fn act(d: usize, c: usize) -> usize { d * 2 + c }

/// run the column procedure with a chooser; returns the answer and the (state, act) pairs used
fn run(a: u64, b: u64, choose: &mut dyn FnMut(usize) -> usize) -> (u64, Vec<(usize, usize)>) {
    let (da, db) = (digits(a), digits(b));
    let n = da.len().max(db.len());
    let (mut carry, mut out, mut used, mut place) = (0usize, 0u64, Vec::with_capacity(n), 1u64);
    for i in 0..n {
        let s = state(*da.get(i).unwrap_or(&0), *db.get(i).unwrap_or(&0), carry);
        let x = choose(s);
        used.push((s, x));
        out += (x / 2) as u64 * place;
        carry = x % 2;
        place *= 10;
    }
    if carry == 1 { out += place; }
    (out, used)
}

/// subtraction (a >= b): the same column shape, (digit, digit, borrow held) -> (digit, borrow on);
/// a borrow left over at the end means the answer is wrong
fn sub_run(a: u64, b: u64, choose: &mut dyn FnMut(usize) -> usize) -> (Option<u64>, Vec<(usize, usize)>) {
    let (da, db) = (digits(a), digits(b));
    let n = da.len().max(db.len());
    let (mut borrow, mut out, mut used, mut place) = (0usize, 0u64, Vec::with_capacity(n), 1u64);
    for i in 0..n {
        let s = state(*da.get(i).unwrap_or(&0), *db.get(i).unwrap_or(&0), borrow);
        let x = choose(s);
        used.push((s, x));
        out += (x / 2) as u64 * place;
        borrow = x % 2;
        place *= 10;
    }
    ((borrow == 0).then_some(out), used)
}

/// the right subtraction act in a column
fn sub_act(p: usize, q: usize, w: usize) -> usize {
    let t = p as i64 - q as i64 - w as i64;
    if t < 0 { act((t + 10) as usize, 1) } else { act(t as usize, 0) }
}

/// "number x digit", a column situation: (digit of a, the multiplier digit, carry held 0-8)
fn mstate(x: usize, m: usize, c: usize) -> usize { (x * 10 + m) * 9 + c }
/// an act: (digit written, carry passed on 0-8)
fn mact(d: usize, c: usize) -> usize { d * 9 + c }
/// a carry can only reach m - 1 (9m + m - 1 < 10m), so 460 of the 900 situations can occur
fn mreachable(s: usize) -> bool { let (m, c) = ((s / 9) % 10, s % 9); c <= m.saturating_sub(1) }

/// procedure 1: one row, a times one digit, right to left; the last carry is written in front
fn mul_row(a: u64, m: usize, choose: &mut dyn FnMut(usize) -> usize) -> (u64, Vec<(usize, usize)>) {
    let (mut carry, mut out, mut used, mut place) = (0usize, 0u64, Vec::new(), 1u64);
    for x in digits(a) {
        let s = mstate(x, m, carry);
        let y = choose(s);
        used.push((s, y));
        out += (y / 9) as u64 * place;
        carry = y % 9;
        place *= 10;
    }
    out += carry as u64 * place;
    (out, used)
}

/// procedure 2, the long multiplication page (given): one row per digit of b, each row shifted
/// one place further left (zeros written in), and the rows added by HER addition procedure.
/// No product is stored anywhere; the (state, act) pairs returned are procedure 1's.
fn mul_run(a: u64, b: u64, row: &mut dyn FnMut(usize) -> usize, add: &mut dyn FnMut(usize) -> usize) -> (u64, Vec<(usize, usize)>) {
    let (mut total, mut used, mut shift) = (0u64, Vec::new(), 1u64);
    for (i, m) in digits(b).into_iter().enumerate() {
        let (r, u) = mul_row(a, m, &mut *row);
        used.extend(u);
        total = if i == 0 { r * shift } else { run(total, r * shift, &mut *add).0 };
        shift *= 10;
    }
    (total, used)
}

/// (v29 WP9g, PASSED and adopted) she explores where experience is thin (see Habits::sample).
/// AVA_EXPLORE=proportional brings back the old choice in proportion to value (the WP9-WP9f runs).
fn uncertain() -> bool {
    static U: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *U.get_or_init(|| std::env::var("AVA_EXPLORE").map_or(true, |v| v != "proportional"))
}

fn unif(x: &mut u64) -> f64 { ((splitmix(x) >> 11) as f64 + 0.5) / (1u64 << 53) as f64 }

/// Gamma(a, 1): Marsaglia-Tsang, with the a < 1 boost
fn gamma(a: f64, x: &mut u64) -> f64 {
    if a < 1.0 { return gamma(a + 1.0, x) * unif(x).powf(1.0 / a); }
    let d = a - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let z = (-2.0 * unif(x).ln()).sqrt() * (std::f64::consts::TAU * unif(x)).cos();
        let v = 1.0 + c * z;
        if v <= 0.0 { continue; }
        let v = v * v * v;
        if unif(x).ln() < 0.5 * z * z + d - d * v + d * v.ln() { return d * v; }
    }
}

fn beta(a: f64, b: f64, x: &mut u64) -> f64 {
    let (g, h) = (gamma(a.max(1e-9), x), gamma(b.max(1e-9), x));
    if g + h > 0.0 { g / (g + h) } else { 0.5 }
}

/// (v29 WP17a) AVA_CURRICULUM=self: she chooses what to practise
fn self_chosen() -> bool { curriculum() != 0 }

/// 0 the teacher's order, 1 her own choice by learning progress (WP17a, `self`), 2 the same with
/// each level's pull divided by how many habits its problems use (WP17b, `clear`)
fn curriculum() -> u8 {
    static C: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *C.get_or_init(|| match std::env::var("AVA_CURRICULUM").as_deref() { Ok("self") => 1, Ok("clear") => 2, _ => 0 })
}

/// the teacher's stages as levels: each stage's new problems, with no order between them
fn levels(stages: &[Vec<(u64, u64)>]) -> Vec<Vec<(u64, u64)>> {
    let mut had = std::collections::HashSet::new();
    stages.iter().map(|st| st.iter().copied().filter(|p| had.insert(*p)).collect()).collect()
}

/// (v29 WP17a) she practises where she is improving: each problem's level is drawn in proportion
/// to its learning progress (how much her success changed between her last 100 problems there and
/// the 100 before) plus 1 / (1 + problems done there), a pull toward what she knows little about.
/// `practise` answers one problem and says whether it was right; every 20,000 answers `check`
/// reports and says whether she is done. Returns the number answered.
/// (WP17b, AVA_CURRICULUM=clear) each level's pull is divided by the mean number of habits a
/// problem there used (1 before any): a lesson whose right/wrong lands on fewer habits is clearer.
/// `practise` returns (right, habits used).
fn practise_by_progress<T>(levels: &[Vec<(u64, u64)>], x: &mut u64, cap: usize, t: &mut T,
                           mut practise: impl FnMut(&mut T, u64, u64) -> (bool, usize),
                           mut check: impl FnMut(&T, usize, &[usize]) -> bool) -> usize {
    const W: usize = 100;
    let mut hist: Vec<std::collections::VecDeque<bool>> = vec![std::collections::VecDeque::new(); levels.len()];
    let mut done = vec![0usize; levels.len()];
    let mut used_sum = vec![0usize; levels.len()];
    let mut answered = 0usize;
    loop {
        let w: Vec<f64> = (0..levels.len()).map(|k| {
            if levels[k].is_empty() { return 0.0; }
            let h = &hist[k];
            let progress = if h.len() == 2 * W {
                let (old, new) = (h.iter().take(W).filter(|&&b| b).count(), h.iter().skip(W).filter(|&&b| b).count());
                (new as f64 - old as f64).abs() / W as f64
            } else { 0.0 };
            let pull = progress + 1.0 / (1.0 + done[k] as f64);
            if curriculum() == 2 { pull / if done[k] == 0 { 1.0 } else { used_sum[k] as f64 / done[k] as f64 } } else { pull }
        }).collect();
        let mut r = unif(x) * w.iter().sum::<f64>();
        let k = (0..w.len()).find(|&k| { r -= w[k]; r <= 0.0 && w[k] > 0.0 }).unwrap_or_else(|| (0..w.len()).rev().find(|&k| w[k] > 0.0).unwrap());
        let (a, b) = levels[k][(splitmix(x) % levels[k].len() as u64) as usize];
        let (ok, used) = practise(t, a, b);
        used_sum[k] += used;
        hist[k].push_back(ok);
        if hist[k].len() > 2 * W { hist[k].pop_front(); }
        done[k] += 1;
        answered += 1;
        if answered % 20_000 == 0 && (check(t, answered, &done) || answered >= cap) { return answered; }
    }
}

#[derive(Clone)]
struct Habits { acts: usize, value: Vec<Vec<f64>>, seen: Vec<Vec<f64>> }

impl Habits {
    /// every habit starts even (0.5), counted as one experience
    fn new(states: usize, acts: usize) -> Self { Habits { acts, value: vec![vec![0.5; acts]; states], seen: vec![vec![1.0; acts]; states] } }
    fn best(&self, s: usize) -> usize {
        (0..self.acts).max_by(|&i, &j| self.value[s][i].total_cmp(&self.value[s][j]).then(j.cmp(&i))).unwrap()
    }
    /// act in proportion to value; or (WP9g, AVA_EXPLORE=uncertain) by a value drawn from what her
    /// experience supports, Beta(0.5 + successes, 0.5 + failures): her running mean and count hold
    /// exactly that (value * seen = 0.5 + rewards). Little experience, a wide draw; much, a narrow one.
    fn sample(&self, s: usize, x: &mut u64) -> usize {
        if uncertain() {
            return (0..self.acts).map(|i| {
                let (v, n) = (self.value[s][i], self.seen[s][i]);
                (i, beta(v * n, (1.0 - v) * n, x))
            }).max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0))).unwrap().0;
        }
        let tot: f64 = self.value[s].iter().sum();
        if tot <= 0.0 { return (splitmix(x) % self.acts as u64) as usize; }
        let mut r = (splitmix(x) >> 11) as f64 / (1u64 << 53) as f64 * tot;
        for i in 0..self.acts { r -= self.value[s][i]; if r <= 0.0 { return i; } }
        self.acts - 1
    }
    fn learn(&mut self, used: &[(usize, usize)], reward: f64) {
        for &(s, a) in used {
            self.seen[s][a] += 1.0;
            self.value[s][a] += (reward - self.value[s][a]) / self.seen[s][a];
        }
    }
}

fn score(tests: &[(u64, u64)], truth: fn(u64, u64) -> u64, mut answer: impl FnMut(u64, u64) -> Option<u64>) -> f64 {
    let ok = tests.iter().filter(|&&(a, b)| answer(a, b) == Some(truth(a, b))).count();
    100.0 * ok as f64 / tests.len() as f64
}

fn accuracy(tests: &[(u64, u64)], answer: impl FnMut(u64, u64) -> Option<u64>) -> f64 { score(tests, |a, b| a + b, answer) }

/// print the learning curve at these points (and when 95% on the unseen test is first reached)
const MILESTONES: [usize; 10] = [20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000];

fn main() {
    // taught sums: operands of 1-3 digits, seed 1
    let mut x: u64 = 1;
    let mut taught: Vec<(u64, u64)> = Vec::new();
    while taught.len() < 2000 {
        let (da, db) = (1 + (splitmix(&mut x) % 3) as u32, 1 + (splitmix(&mut x) % 3) as u32);
        taught.push((number(da, &mut x), number(db, &mut x)));
    }
    let known: HashMap<(u64, u64), u64> = taught.iter().map(|&(a, b)| ((a, b), a + b)).collect();
    // tests, seed 2, never a taught sum
    let mut y: u64 = 2;
    let mut make = |n: usize, lo: u32, hi: u32| -> Vec<(u64, u64)> {
        let mut v = Vec::new();
        while v.len() < n {
            let (da, db) = (lo + (splitmix(&mut y) % (hi - lo + 1) as u64) as u32, lo + (splitmix(&mut y) % (hi - lo + 1) as u64) as u32);
            let p = (number(da, &mut y), number(db, &mut y));
            if !known.contains_key(&p) { v.push(p); }
        }
        v
    };
    let (t3, t6, t12) = (make(2000, 1, 3), make(1000, 4, 6), make(1000, 7, 12));
    println!("taught {} sums (1-3 digits); tests: {} unseen 1-3 digit, {} of 4-6, {} of 7-12", taught.len(), t3.len(), t6.len(), t12.len());

    // M: memory of whole sums
    let m = |a: u64, b: u64| known.get(&(a, b)).copied();
    println!("\nM memory          1-3: {:>6.2}%   4-6: {:>6.2}%   7-12: {:>6.2}%   (taught sums: {:.2}%)",
             accuracy(&t3, m), accuracy(&t6, m), accuracy(&t12, m), accuracy(&taught, m));

    // D: counting worked examples, column by column
    let mut count = vec![[0u32; 20]; 200];
    for &(a, b) in &taught {
        let (da, db) = (digits(a), digits(b));
        let mut c = 0;
        for i in 0..da.len().max(db.len()) {
            let (p, q) = (*da.get(i).unwrap_or(&0), *db.get(i).unwrap_or(&0));
            let t = p + q + c;
            count[state(p, q, c)][act(t % 10, t / 10)] += 1;
            c = t / 10;
        }
    }
    let covered = count.iter().filter(|r| r.iter().any(|&n| n > 0)).count();
    let d = |a: u64, b: u64| -> Option<u64> {
        let mut gap = false;
        let (s, _) = run(a, b, &mut |s| { if count[s].iter().all(|&n| n == 0) { gap = true; } (0..20).max_by_key(|&i| (count[s][i], std::cmp::Reverse(i))).unwrap() });
        (!gap).then_some(s)
    };
    println!("D demonstration   1-3: {:>6.2}%   4-6: {:>6.2}%   7-12: {:>6.2}%   ({covered} of 200 column situations seen)",
             accuracy(&t3, d), accuracy(&t6, d), accuracy(&t12, d));

    // R: reward only, staged 1 -> 2 -> 3 digits, practising on the taught sums
    let mut h = Habits::new(200, 20);
    let mut z: u64 = 3;
    let (mut answered, mut reached) = (0usize, None);
    let stages: [Vec<(u64, u64)>; 3] = [1u64, 2, 3].map(|k| taught.iter().copied().filter(|&(a, b)| digits(a).len().max(digits(b).len()) as u64 <= k).collect());
    println!("\nR reward only (answered sums -> unseen 1-3 / 4-6 / 7-12):");
    if self_chosen() {
        let mut x: u64 = 90;
        answered = practise_by_progress(&levels(&stages), &mut x, 2_000_000, &mut h,
            |h, a, b| { let (ans, used) = run(a, b, &mut |s| h.sample(s, &mut z)); let ok = ans == a + b; h.learn(&used, if ok { 1.0 } else { 0.0 }); (ok, used.len()) },
            |h, n, done| {
                let r = |a: u64, b: u64| Some(run(a, b, &mut |s| h.best(s)).0);
                let acc = accuracy(&t3, r);
                println!("  self  {:>8} answered: {:>6.2}% / {:>6.2}% / {:>6.2}%   practised per level {:?}", n, acc, accuracy(&t6, r), accuracy(&t12, r), done);
                if acc >= 95.0 && reached.is_none() { reached = Some(n); }
                taught.iter().all(|&(a, b)| run(a, b, &mut |s| h.best(s)).0 == a + b)
            });
    } else {
    'outer: for (si, stage) in stages.iter().enumerate() {
        for _pass in 0..400 {
            for &(a, b) in stage {
                let (ans, used) = run(a, b, &mut |s| h.sample(s, &mut z));
                h.learn(&used, if ans == a + b { 1.0 } else { 0.0 });
                answered += 1;
                if answered % 20_000 == 0 {
                    let r = |a: u64, b: u64| Some(run(a, b, &mut |s| h.best(s)).0);
                    let acc = accuracy(&t3, r);
                    println!("  stage {} {:>8} answered: {:>6.2}% / {:>6.2}% / {:>6.2}%", si + 1, answered, acc, accuracy(&t6, r), accuracy(&t12, r));
                    if acc >= 95.0 && reached.is_none() { reached = Some(answered); }
                    if answered >= 2_000_000 { break 'outer; }
                }
            }
            // move on once this stage's own sums are all right with her best habits
            if stage.iter().all(|&(a, b)| run(a, b, &mut |s| h.best(s)).0 == a + b) { break; }
        }
    }
    }
    let r = |a: u64, b: u64| Some(run(a, b, &mut |s| h.best(s)).0);
    println!("R reward only     1-3: {:>6.2}%   4-6: {:>6.2}%   7-12: {:>6.2}%   ({answered} sums answered; 95% on 1-3 reached at {})",
             accuracy(&t3, r), accuracy(&t6, r), accuracy(&t12, r), reached.map_or("never".into(), |n| n.to_string()));
    // which column habits did reward teach, and are any wrong?
    let wrong: Vec<String> = (0..200).filter_map(|s| {
        let (p, q, c) = (s / 20, (s / 2) % 10, s % 2);
        let t = p + q + c;
        let b = h.best(s);
        (b != act(t % 10, t / 10)).then(|| format!("{p}+{q}+carry{c} -> wrote {} carry {}", b / 2, b % 2))
    }).collect();
    println!("  column habits wrong after reward learning: {} of 200{}", wrong.len(), if wrong.is_empty() { String::new() } else { format!(": {}", wrong.iter().take(10).cloned().collect::<Vec<_>>().join("; ")) });

    let (sub_h, sub_t6, sub_t12) = subtraction();
    multiplication(&count, &h);
    let facts = times_table(&count, &h);
    discover(&h, &facts);
    let found = division(&h, &facts, &sub_h);
    self_taught(&h, &facts, &sub_h, &sub_t6, &sub_t12, found);
}

/// (v29 WP9c) one column of "number x digit" as a teacher says it: multiply, then add what you
/// carried. The times fact (x, m) -> product (tens, units) is learned: 100 situations, 100 acts.
/// Adding the carry is her addition procedure. Returns the row and the times-fact pairs used.
fn fact_row(a: u64, m: usize, fact: &mut dyn FnMut(usize) -> usize, add: &mut dyn FnMut(usize) -> usize) -> (u64, Vec<(usize, usize)>) {
    let (mut carry, mut out, mut used, mut place) = (0u64, 0u64, Vec::new(), 1u64);
    for x in digits(a) {
        let s = x * 10 + m;
        let f = fact(s);
        used.push((s, f));
        let t = run(f as u64, carry, &mut *add).0;
        out += (t % 10) * place;
        carry = t / 10;
        place *= 10;
    }
    out += carry * place;
    (out, used)
}

/// the WP9b page layout over fact_row: one row per digit of b, shifted, added by her addition
fn fact_run(a: u64, b: u64, fact: &mut dyn FnMut(usize) -> usize, add: &mut dyn FnMut(usize) -> usize) -> (u64, Vec<(usize, usize)>) {
    let (mut total, mut used, mut shift) = (0u64, Vec::new(), 1u64);
    for (i, m) in digits(b).into_iter().enumerate() {
        let (r, u) = fact_row(a, m, &mut *fact, &mut *add);
        used.extend(u);
        total = if i == 0 { r * shift } else { run(total, r * shift, &mut *add).0 };
        shift *= 10;
    }
    (total, used)
}

/// (v29 WP9c) long multiplication from a learned times table plus her addition. Fresh seeds 31-33.
fn times_table(add_count: &[[u32; 20]], add_h: &Habits) -> Habits {
    let mul = |a: u64, b: u64| a * b;
    let mut x: u64 = 31;
    let mut taught: Vec<(u64, u64)> = Vec::new();
    while taught.len() < 2000 {
        let (da, db) = (1 + (splitmix(&mut x) % 3) as u32, 1 + (splitmix(&mut x) % 2) as u32);
        taught.push((number(da, &mut x), number(db, &mut x)));
    }
    let known: HashMap<(u64, u64), u64> = taught.iter().map(|&(a, b)| ((a, b), a * b)).collect();
    let mut y: u64 = 32;
    let mut make = |n: usize, (alo, ahi): (u32, u32), (blo, bhi): (u32, u32)| -> Vec<(u64, u64)> {
        let mut v = Vec::new();
        while v.len() < n {
            let (da, db) = (alo + (splitmix(&mut y) % (ahi - alo + 1) as u64) as u32, blo + (splitmix(&mut y) % (bhi - blo + 1) as u64) as u32);
            let p = (number(da, &mut y), number(db, &mut y));
            if !known.contains_key(&p) { v.push(p); }
        }
        v
    };
    let (tu, tl) = (make(2000, (1, 3), (1, 2)), make(1000, (4, 6), (3, 3)));
    println!("\n==== MULTIPLICATION: TIMES TABLE + ADD THE CARRY (WP9c, fresh seeds) ====");
    println!("taught {} products (1-3 x 1-2 digits); tests: {} unseen, {} longer (4-6 x 3 digits)", taught.len(), tu.len(), tl.len());
    println!("products stored by D and R: 0 (they hold 100 times facts and 200 addition habits)");
    let m = |a: u64, b: u64| known.get(&(a, b)).copied();
    println!("M memory          unseen: {:>6.2}%   longer: {:>6.2}%   (taught: {:.2}%)", score(&tu, mul, m), score(&tl, mul, m), score(&taught, mul, m));

    // D: worked examples show each column's times fact ("7 x 8 = 56") before the carry is added
    let mut count = vec![[0u32; 100]; 100];
    for &(a, b) in &taught {
        for md in digits(b) { for xd in digits(a) { count[xd * 10 + md][xd * md] += 1; } }
    }
    let covered = count.iter().filter(|r| r.iter().any(|&n| n > 0)).count();
    let d = |a: u64, b: u64| -> Option<u64> {
        let (mut g1, mut g2) = (false, false);
        let (p, _) = fact_run(a, b,
            &mut |s| { if count[s].iter().all(|&n| n == 0) { g1 = true; } (0..100).max_by_key(|&i| (count[s][i], std::cmp::Reverse(i))).unwrap() },
            &mut |s| { if add_count[s].iter().all(|&n| n == 0) { g2 = true; } (0..20).max_by_key(|&i| (add_count[s][i], std::cmp::Reverse(i))).unwrap() });
        (!g1 && !g2).then_some(p)
    };
    println!("D demonstration   unseen: {:>6.2}%   longer: {:>6.2}%   ({covered} of 100 times facts seen)", score(&tu, mul, d), score(&tl, mul, d));

    // R: right/wrong of the WHOLE product only; credit goes to the times facts she used
    let mut h = Habits::new(100, 100);
    let mut z: u64 = 33;
    let (mut answered, mut reached) = (0usize, None);
    let stages: [Vec<(u64, u64)>; 4] = [(1usize, 1usize), (2, 1), (3, 1), (3, 2)]
        .map(|(ka, kb)| taught.iter().copied().filter(|&(a, b)| digits(a).len() <= ka && digits(b).len() <= kb).collect());
    println!("R reward only (answered -> unseen / longer):");
    if self_chosen() {
        let mut x: u64 = 91;
        let last = stages.last().unwrap().clone();
        answered = practise_by_progress(&levels(&stages), &mut x, 20_000_000, &mut h,
            |h, a, b| { let (ans, used) = fact_run(a, b, &mut |s| h.sample(s, &mut z), &mut |s| add_h.best(s)); let ok = ans == a * b; h.learn(&used, if ok { 1.0 } else { 0.0 }); (ok, used.len()) },
            |h, n, done| {
                let r = |a: u64, b: u64| Some(fact_run(a, b, &mut |s| h.best(s), &mut |s| add_h.best(s)).0);
                let acc = score(&tu, mul, r);
                let first = acc >= 95.0 && reached.is_none();
                if first { reached = Some(n); }
                if first || MILESTONES.contains(&n) { println!("  self  {:>9} answered: {:>6.2}% / {:>6.2}%   practised per level {:?}", n, acc, score(&tl, mul, r), done); }
                last.iter().all(|&(a, b)| fact_run(a, b, &mut |s| h.best(s), &mut |s| add_h.best(s)).0 == a * b)
            });
    } else {
    'outer: for (si, stage) in stages.iter().enumerate() {
        for _pass in 0..400 {
            for &(a, b) in stage {
                let (ans, used) = fact_run(a, b, &mut |s| h.sample(s, &mut z), &mut |s| add_h.best(s));
                h.learn(&used, if ans == a * b { 1.0 } else { 0.0 });
                answered += 1;
                if answered % 20_000 == 0 {
                    let r = |a: u64, b: u64| Some(fact_run(a, b, &mut |s| h.best(s), &mut |s| add_h.best(s)).0);
                    let acc = score(&tu, mul, r);
                    let first = acc >= 95.0 && reached.is_none();
                    if first { reached = Some(answered); }
                    if first || MILESTONES.contains(&answered) {
                        println!("  stage {} {:>9} answered: {:>6.2}% / {:>6.2}%", si + 1, answered, acc, score(&tl, mul, r));
                    }
                    if answered >= 20_000_000 { break 'outer; }
                }
            }
            if stage.iter().all(|&(a, b)| fact_run(a, b, &mut |s| h.best(s), &mut |s| add_h.best(s)).0 == a * b) { break; }
        }
    }
    }
    let r = |a: u64, b: u64| Some(fact_run(a, b, &mut |s| h.best(s), &mut |s| add_h.best(s)).0);
    println!("R reward only     unseen: {:>6.2}%   longer: {:>6.2}%   ({answered} answered; 95% on unseen reached at {})",
             score(&tu, mul, r), score(&tl, mul, r), reached.map_or("never".into(), |n| n.to_string()));
    let wrong: Vec<String> = (0..100).filter(|&s| h.best(s) != (s / 10) * (s % 10))
        .map(|s| format!("{}x{} -> {}", s / 10, s % 10, h.best(s))).collect();
    println!("  times facts wrong after reward learning: {} of 100{}", wrong.len(), if wrong.is_empty() { String::new() } else { format!(": {}", wrong.iter().take(10).cloned().collect::<Vec<_>>().join("; ")) });
    h
}

/// (v29 WP9d) a column value built from her own skills: the column's digit x, the multiplier m,
/// the carry c, ADD (her addition procedure) and TIMES (her times facts, single digits only)
#[derive(Clone)]
enum Ex { X, M, C, Add(Box<Ex>, Box<Ex>), Times(Box<Ex>, Box<Ex>) }

impl Ex {
    fn show(&self) -> String {
        match self {
            Ex::X => "x".into(),
            Ex::M => "m".into(),
            Ex::C => "c".into(),
            Ex::Add(a, b) => format!("ADD({}, {})", a.show(), b.show()),
            Ex::Times(a, b) => format!("TIMES({}, {})", a.show(), b.show()),
        }
    }
}

/// every expression of up to 3 leaves, smallest first (3 + 18 + 216 = 237)
fn expressions() -> Vec<Ex> {
    let mk = |op: usize, a: Ex, b: Ex| if op == 0 { Ex::Add(Box::new(a), Box::new(b)) } else { Ex::Times(Box::new(a), Box::new(b)) };
    let leaves = vec![Ex::X, Ex::M, Ex::C];
    let mut s3 = Vec::new();
    for op in 0..2 { for a in &leaves { for b in &leaves { s3.push(mk(op, a.clone(), b.clone())); } } }
    let mut s5 = Vec::new();
    for op in 0..2 { for a in &s3 { for b in &leaves { s5.push(mk(op, a.clone(), b.clone())); s5.push(mk(op, b.clone(), a.clone())); } } }
    leaves.into_iter().chain(s3).chain(s5).collect()
}

/// the skills she has: addition and the times table, both learned from right/wrong alone (R)
struct Skills<'a> { add: &'a Habits, fact: &'a Habits }

impl Skills<'_> {
    fn add(&self, p: u64, q: u64) -> u64 { run(p, q, &mut |s| self.add.best(s)).0 }
    fn times(&self, p: u64, q: u64) -> Option<u64> { (p < 10 && q < 10).then(|| self.fact.best((p * 10 + q) as usize) as u64) }
    fn eval(&self, e: &Ex, x: u64, m: u64, c: u64) -> Option<u64> {
        Some(match e {
            Ex::X => x,
            Ex::M => m,
            Ex::C => c,
            Ex::Add(a, b) => self.add(self.eval(a, x, m, c)?, self.eval(b, x, m, c)?),
            Ex::Times(a, b) => self.times(self.eval(a, x, m, c)?, self.eval(b, x, m, c)?)?,
        })
    }
    /// "number x digit" by a column expression: right to left, the value's units written, its
    /// tens carried, the last carry written in front (the layout, given)
    fn row(&self, e: &Ex, a: u64, m: u64) -> Option<u64> {
        let (mut carry, mut out, mut place) = (0u64, 0u64, 1u64);
        for x in digits(a) {
            let t = self.eval(e, x as u64, m, carry)?;
            if t > 1_000 { return None; } // a runaway carry: this idea has already gone wrong
            out += (t % 10) * place;
            carry = t / 10;
            place *= 10;
        }
        Some(out + carry * place)
    }
    /// the long multiplication page, five ways of laying out her rows
    fn page(&self, k: usize, e: &Ex, a: u64, b: u64) -> Option<u64> {
        let db = digits(b);
        let n = db.len() as u32;
        match k {
            0 => db.iter().enumerate().try_fold(0, |t, (i, &d)| Some(self.add(t, self.row(e, a, d as u64)? * 10u64.pow(i as u32)))),
            1 => db.iter().try_fold(0, |t, &d| Some(self.add(t, self.row(e, a, d as u64)?))),
            2 => db.iter().enumerate().try_fold(0, |t, (i, &d)| Some(self.add(t, self.row(e, a, d as u64)? * 10u64.pow(n - 1 - i as u32)))),
            3 => {
                let mut s = String::new();
                for &d in db.iter().rev() { s.push_str(&self.row(e, a, d as u64)?.to_string()); }
                s.parse().ok()
            }
            _ => digits(a).iter().enumerate().try_fold(0, |t, (i, &d)| Some(self.add(t, self.row(e, b, d as u64)? * 10u64.pow(i as u32)))),
        }
    }
}

const LAYOUTS: [&str; 5] = ["rows shifted by place, added", "rows added unshifted", "rows shifted the wrong way", "rows side by side", "rows of b by a's digits, shifted"];

/// (v29 WP9d) can she FIND the split "times fact, then add the carry" from worked answers alone?
fn discover(add_h: &Habits, fact_h: &Habits) {
    println!("\n==== FINDING THE COMPOSITION HERSELF (WP9d, fresh seeds) ====");
    let sk = Skills { add: add_h, fact: fact_h };
    let exprs = expressions();
    let truth = |e: &Ex| (0..10u64).all(|x| (0..10u64).all(|m| (0..=m.saturating_sub(1)).all(|c| sk.eval(e, x, m, c) == Some(x * m + c))));
    let n_true = exprs.iter().filter(|e| truth(e)).count();
    println!("{} column ideas (expressions over x, m, c with ADD and TIMES); {} of them are right", exprs.len(), n_true);
    let mut x: u64 = 41;
    let ex1: Vec<(u64, u64)> = (0..20).map(|_| { let d = 1 + (splitmix(&mut x) % 3) as u32; (number(d, &mut x), number(1, &mut x)) }).collect();
    let ex2: Vec<(u64, u64)> = (0..20).map(|_| { let d = 1 + (splitmix(&mut x) % 3) as u32; (number(d, &mut x), number(2, &mut x)) }).collect();
    let mut y: u64 = 42;
    let t1: Vec<(u64, u64)> = (0..1000).map(|_| { let d = 4 + (splitmix(&mut y) % 3) as u32; (number(d, &mut y), number(1, &mut y)) }).collect();
    let t2: Vec<(u64, u64)> = (0..1000).map(|_| { let d = 4 + (splitmix(&mut y) % 3) as u32; (number(d, &mut y), number(3, &mut y)) }).collect();
    // the teacher's slips: 2 of the 20 answers wrong, in each level
    let slip = |ex: &[(u64, u64)], z: &mut u64| -> Vec<u64> {
        let mut ans: Vec<u64> = ex.iter().map(|&(a, b)| a * b).collect();
        let i = (splitmix(z) % 20) as usize;
        let j = (i + 1 + (splitmix(z) % 19) as usize) % 20;
        for k in [i, j] { ans[k] += 1 + splitmix(z) % 9; }
        ans
    };
    let mut z: u64 = 43;
    let (noisy1, noisy2) = (slip(&ex1, &mut z), slip(&ex2, &mut z));
    let clean1: Vec<u64> = ex1.iter().map(|&(a, b)| a * b).collect();
    let clean2: Vec<u64> = ex2.iter().map(|&(a, b)| a * b).collect();

    for (label, ans1, ans2, need) in [("clean examples", &clean1, &clean2, 20usize), ("2 of 20 answers wrong", &noisy1, &noisy2, 16)] {
        println!("\n-- {label} (an idea must fit {need} of 20) --");
        // level 1: which column idea explains the answers?
        let fits1 = |e: &Ex, k: usize| ex1[..k].iter().zip(&ans1[..k]).filter(|(&(a, m), &ans)| sk.row(e, a, m) == Some(ans)).count();
        print!("level 1, ideas still fitting after 1/2/3/5/10/20 examples:");
        for k in [1, 2, 3, 5, 10, 20] { print!(" {}", exprs.iter().filter(|e| fits1(e, k) >= (need * k).div_ceil(20)).count()); }
        println!();
        let Some(e) = exprs.iter().find(|e| fits1(e, 20) >= need) else { println!("  no idea fits: FAIL"); continue };
        let acc1 = t1.iter().filter(|&&(a, m)| sk.row(e, a, m) == Some(a * m)).count();
        println!("  chosen: t = {}   right on all 460 situations: {}   unseen 4-6 digit x 1 digit: {:.1}%", e.show(), truth(e), acc1 as f64 / 10.0);
        // level 2: how do her rows make a long product?
        let fits2 = |k: usize, n: usize| ex2[..n].iter().zip(&ans2[..n]).filter(|(&(a, b), &ans)| sk.page(k, e, a, b) == Some(ans)).count();
        print!("level 2, layouts still fitting after 1/2/3/5/10/20 examples:");
        for n in [1, 2, 3, 5, 10, 20] { print!(" {}", (0..5).filter(|&k| fits2(k, n) >= (need * n).div_ceil(20)).count()); }
        println!();
        let Some(k) = (0..5).find(|&k| fits2(k, 20) >= need) else { println!("  no layout fits: FAIL"); continue };
        let acc2 = t2.iter().filter(|&&(a, b)| sk.page(k, e, a, b) == Some(a * b)).count();
        println!("  chosen: {}   unseen 4-6 x 3 digits: {:.1}%", LAYOUTS[k], acc2 as f64 / 10.0);
    }
}

/// (v29 WP9b) subtraction, a >= b: borrowing instead of carrying. Pre-registered in "ava v29 - in depth".
fn subtraction() -> (Habits, Vec<(u64, u64)>, Vec<(u64, u64)>) {
    let sub = |a: u64, b: u64| a - b;
    let mut x: u64 = 21;
    let mut taught: Vec<(u64, u64)> = Vec::new();
    while taught.len() < 2000 {
        let (da, db) = (1 + (splitmix(&mut x) % 3) as u32, 1 + (splitmix(&mut x) % 3) as u32);
        let (p, q) = (number(da, &mut x), number(db, &mut x));
        taught.push((p.max(q), p.min(q)));
    }
    let known: HashMap<(u64, u64), u64> = taught.iter().map(|&(a, b)| ((a, b), a - b)).collect();
    let mut y: u64 = 22;
    let mut make = |n: usize, lo: u32, hi: u32| -> Vec<(u64, u64)> {
        let mut v = Vec::new();
        while v.len() < n {
            let (da, db) = (lo + (splitmix(&mut y) % (hi - lo + 1) as u64) as u32, lo + (splitmix(&mut y) % (hi - lo + 1) as u64) as u32);
            let (p, q) = (number(da, &mut y), number(db, &mut y));
            if !known.contains_key(&(p.max(q), p.min(q))) { v.push((p.max(q), p.min(q))); }
        }
        v
    };
    let (t3, t6, t12) = (make(2000, 1, 3), make(1000, 4, 6), make(1000, 7, 12));
    println!("\n==== SUBTRACTION (a >= b) ====");
    println!("taught {} differences (1-3 digits); tests: {} unseen 1-3 digit, {} of 4-6, {} of 7-12", taught.len(), t3.len(), t6.len(), t12.len());
    let m = |a: u64, b: u64| known.get(&(a, b)).copied();
    println!("M memory          1-3: {:>6.2}%   4-6: {:>6.2}%   7-12: {:>6.2}%   (taught: {:.2}%)",
             score(&t3, sub, m), score(&t6, sub, m), score(&t12, sub, m), score(&taught, sub, m));

    // D: worked examples show each column's digit and borrow
    let mut count = vec![[0u32; 20]; 200];
    for &(a, b) in &taught {
        let (da, db) = (digits(a), digits(b));
        let mut w = 0;
        for i in 0..da.len() {
            let (p, q) = (da[i], *db.get(i).unwrap_or(&0));
            let x = sub_act(p, q, w);
            count[state(p, q, w)][x] += 1;
            w = x % 2;
        }
    }
    let covered = count.iter().filter(|r| r.iter().any(|&n| n > 0)).count();
    let d = |a: u64, b: u64| -> Option<u64> {
        let mut gap = false;
        let (s, _) = sub_run(a, b, &mut |s| { if count[s].iter().all(|&n| n == 0) { gap = true; } (0..20).max_by_key(|&i| (count[s][i], std::cmp::Reverse(i))).unwrap() });
        if gap { None } else { s }
    };
    println!("D demonstration   1-3: {:>6.2}%   4-6: {:>6.2}%   7-12: {:>6.2}%   ({covered} of 200 column situations seen)",
             score(&t3, sub, d), score(&t6, sub, d), score(&t12, sub, d));

    // R: right/wrong only, staged by the length of a
    let mut h = Habits::new(200, 20);
    let mut z: u64 = 23;
    let (mut answered, mut reached) = (0usize, None);
    let stages: [Vec<(u64, u64)>; 3] = [1usize, 2, 3].map(|k| taught.iter().copied().filter(|&(a, _)| digits(a).len() <= k).collect());
    println!("R reward only (answered -> unseen 1-3 / 4-6 / 7-12):");
    if self_chosen() {
        let mut x: u64 = 92;
        let last = stages.last().unwrap().clone();
        answered = practise_by_progress(&levels(&stages), &mut x, 20_000_000, &mut h,
            |h, a, b| { let (ans, used) = sub_run(a, b, &mut |s| h.sample(s, &mut z)); let ok = ans == Some(a - b); h.learn(&used, if ok { 1.0 } else { 0.0 }); (ok, used.len()) },
            |h, n, done| {
                let r = |a: u64, b: u64| sub_run(a, b, &mut |s| h.best(s)).0;
                let acc = score(&t3, sub, r);
                let first = acc >= 95.0 && reached.is_none();
                if first { reached = Some(n); }
                if first || MILESTONES.contains(&n) { println!("  self  {:>9} answered: {:>6.2}% / {:>6.2}% / {:>6.2}%   practised per level {:?}", n, acc, score(&t6, sub, r), score(&t12, sub, r), done); }
                last.iter().all(|&(a, b)| sub_run(a, b, &mut |s| h.best(s)).0 == Some(a - b))
            });
    } else {
    'outer: for (si, stage) in stages.iter().enumerate() {
        for _pass in 0..400 {
            for &(a, b) in stage {
                let (ans, used) = sub_run(a, b, &mut |s| h.sample(s, &mut z));
                h.learn(&used, if ans == Some(a - b) { 1.0 } else { 0.0 });
                answered += 1;
                if answered % 20_000 == 0 {
                    let r = |a: u64, b: u64| sub_run(a, b, &mut |s| h.best(s)).0;
                    let acc = score(&t3, sub, r);
                    let first = acc >= 95.0 && reached.is_none();
                    if first { reached = Some(answered); }
                    if first || MILESTONES.contains(&answered) {
                        println!("  stage {} {:>9} answered: {:>6.2}% / {:>6.2}% / {:>6.2}%", si + 1, answered, acc, score(&t6, sub, r), score(&t12, sub, r));
                    }
                    if answered >= 20_000_000 { break 'outer; }
                }
            }
            if stage.iter().all(|&(a, b)| sub_run(a, b, &mut |s| h.best(s)).0 == Some(a - b)) { break; }
        }
    }
    }
    let r = |a: u64, b: u64| sub_run(a, b, &mut |s| h.best(s)).0;
    println!("R reward only     1-3: {:>6.2}%   4-6: {:>6.2}%   7-12: {:>6.2}%   ({answered} answered; 95% on 1-3 reached at {})",
             score(&t3, sub, r), score(&t6, sub, r), score(&t12, sub, r), reached.map_or("never".into(), |n| n.to_string()));
    let wrong: Vec<String> = (0..200).filter_map(|s| {
        let (p, q, w) = (s / 20, (s / 2) % 10, s % 2);
        let b = h.best(s);
        (b != sub_act(p, q, w)).then(|| format!("{p}-{q}-borrow{w} -> wrote {} borrow {}", b / 2, b % 2))
    }).collect();
    println!("  column habits wrong after reward learning: {} of 200{}", wrong.len(), if wrong.is_empty() { String::new() } else { format!(": {}", wrong.iter().take(10).cloned().collect::<Vec<_>>().join("; ")) });
    (h, t6, t12)
}

/// (v29 WP9b) long multiplication: procedure 1 ("number x digit") is learned; the page layout is
/// given and adds the rows with the addition procedure she learned above (D with D's, R with R's).
fn multiplication(add_count: &[[u32; 20]], add_h: &Habits) {
    let mul = |a: u64, b: u64| a * b;
    let mut x: u64 = 11;
    let mut taught: Vec<(u64, u64)> = Vec::new();
    while taught.len() < 2000 {
        let (da, db) = (1 + (splitmix(&mut x) % 3) as u32, 1 + (splitmix(&mut x) % 2) as u32);
        taught.push((number(da, &mut x), number(db, &mut x)));
    }
    let known: HashMap<(u64, u64), u64> = taught.iter().map(|&(a, b)| ((a, b), a * b)).collect();
    let mut y: u64 = 12;
    let mut make = |n: usize, (alo, ahi): (u32, u32), (blo, bhi): (u32, u32)| -> Vec<(u64, u64)> {
        let mut v = Vec::new();
        while v.len() < n {
            let (da, db) = (alo + (splitmix(&mut y) % (ahi - alo + 1) as u64) as u32, blo + (splitmix(&mut y) % (bhi - blo + 1) as u64) as u32);
            let p = (number(da, &mut y), number(db, &mut y));
            if !known.contains_key(&p) { v.push(p); }
        }
        v
    };
    let (tu, tl) = (make(2000, (1, 3), (1, 2)), make(1000, (4, 6), (3, 3)));
    println!("\n==== MULTIPLICATION ====");
    println!("taught {} products (1-3 digits x 1-2 digits); tests: {} unseen of the same sizes, {} longer (4-6 x 3 digits)", taught.len(), tu.len(), tl.len());
    println!("products stored by D and R: 0 (they hold only column habits: 460 reachable 'number x digit' situations, 200 addition ones)");
    let m = |a: u64, b: u64| known.get(&(a, b)).copied();
    println!("M memory          unseen: {:>6.2}%   longer: {:>6.2}%   (taught: {:.2}%)", score(&tu, mul, m), score(&tl, mul, m), score(&taught, mul, m));

    // D: worked examples show each row's column digit and carry; the rows are added with D's addition
    let mut count = vec![[0u32; 90]; 900];
    for &(a, b) in &taught {
        for mdig in digits(b) {
            let mut c = 0;
            for xd in digits(a) {
                let t = xd * mdig + c;
                count[mstate(xd, mdig, c)][mact(t % 10, t / 10)] += 1;
                c = t / 10;
            }
        }
    }
    let covered = (0..900).filter(|&s| mreachable(s) && count[s].iter().any(|&n| n > 0)).count();
    let d = |a: u64, b: u64| -> Option<u64> {
        let (mut g1, mut g2) = (false, false);
        let (p, _) = mul_run(a, b,
            &mut |s| { if count[s].iter().all(|&n| n == 0) { g1 = true; } (0..90).max_by_key(|&i| (count[s][i], std::cmp::Reverse(i))).unwrap() },
            &mut |s| { if add_count[s].iter().all(|&n| n == 0) { g2 = true; } (0..20).max_by_key(|&i| (add_count[s][i], std::cmp::Reverse(i))).unwrap() });
        (!g1 && !g2).then_some(p)
    };
    println!("D demonstration   unseen: {:>6.2}%   longer: {:>6.2}%   ({covered} of 460 reachable column situations seen)", score(&tu, mul, d), score(&tl, mul, d));
    let gaps: Vec<String> = (0..900).filter(|&s| mreachable(s) && count[s].iter().all(|&n| n == 0))
        .map(|s| format!("{}x{}+carry{}", s / 90, (s / 9) % 10, s % 9)).collect();
    if !gaps.is_empty() { println!("  never shown ({}): {}", gaps.len(), gaps.iter().take(12).cloned().collect::<Vec<_>>().join(", ")); }

    // R: right/wrong of the WHOLE product only; credit goes to the procedure-1 habits she used
    let mut h = Habits::new(900, 90);
    let mut z: u64 = 13;
    let (mut answered, mut reached) = (0usize, None);
    let stages: [Vec<(u64, u64)>; 4] = [(1usize, 1usize), (2, 1), (3, 1), (3, 2)]
        .map(|(ka, kb)| taught.iter().copied().filter(|&(a, b)| digits(a).len() <= ka && digits(b).len() <= kb).collect());
    println!("R reward only (answered -> unseen / longer):");
    if self_chosen() {
        let mut x: u64 = 93;
        let last = stages.last().unwrap().clone();
        answered = practise_by_progress(&levels(&stages), &mut x, 20_000_000, &mut h,
            |h, a, b| { let (ans, used) = mul_run(a, b, &mut |s| h.sample(s, &mut z), &mut |s| add_h.best(s)); let ok = ans == a * b; h.learn(&used, if ok { 1.0 } else { 0.0 }); (ok, used.len()) },
            |h, n, done| {
                let r = |a: u64, b: u64| Some(mul_run(a, b, &mut |s| h.best(s), &mut |s| add_h.best(s)).0);
                let acc = score(&tu, mul, r);
                let first = acc >= 95.0 && reached.is_none();
                if first { reached = Some(n); }
                if first || MILESTONES.contains(&n) { println!("  self  {:>9} answered: {:>6.2}% / {:>6.2}%   practised per level {:?}", n, acc, score(&tl, mul, r), done); }
                last.iter().all(|&(a, b)| mul_run(a, b, &mut |s| h.best(s), &mut |s| add_h.best(s)).0 == a * b)
            });
    } else {
    'outer: for (si, stage) in stages.iter().enumerate() {
        for _pass in 0..400 {
            for &(a, b) in stage {
                let (ans, used) = mul_run(a, b, &mut |s| h.sample(s, &mut z), &mut |s| add_h.best(s));
                h.learn(&used, if ans == a * b { 1.0 } else { 0.0 });
                answered += 1;
                if answered % 20_000 == 0 {
                    let r = |a: u64, b: u64| Some(mul_run(a, b, &mut |s| h.best(s), &mut |s| add_h.best(s)).0);
                    let acc = score(&tu, mul, r);
                    let first = acc >= 95.0 && reached.is_none();
                    if first { reached = Some(answered); }
                    if first || MILESTONES.contains(&answered) {
                        println!("  stage {} {:>9} answered: {:>6.2}% / {:>6.2}%", si + 1, answered, acc, score(&tl, mul, r));
                    }
                    if answered >= 20_000_000 { break 'outer; }
                }
            }
            if stage.iter().all(|&(a, b)| mul_run(a, b, &mut |s| h.best(s), &mut |s| add_h.best(s)).0 == a * b) { break; }
        }
    }
    }
    let r = |a: u64, b: u64| Some(mul_run(a, b, &mut |s| h.best(s), &mut |s| add_h.best(s)).0);
    println!("R reward only     unseen: {:>6.2}%   longer: {:>6.2}%   ({answered} answered; 95% on unseen reached at {})",
             score(&tu, mul, r), score(&tl, mul, r), reached.map_or("never".into(), |n| n.to_string()));
    let wrong: Vec<String> = (0..900).filter(|&s| mreachable(s)).filter_map(|s| {
        let (xd, md, c) = (s / 90, (s / 9) % 10, s % 9);
        let t = xd * md + c;
        let b = h.best(s);
        (b != mact(t % 10, t / 10)).then(|| format!("{xd}x{md}+carry{c} -> wrote {} carry {}", b / 9, b % 9))
    }).collect();
    println!("  'number x digit' habits wrong after reward learning: {} of 460{}", wrong.len(), if wrong.is_empty() { String::new() } else { format!(": {}", wrong.iter().take(10).cloned().collect::<Vec<_>>().join("; ")) });
}

/// (v29 WP9e) how a quotient digit is chosen: the largest / smallest q (0-9) with LHS REL RHS, or
/// the loop "count how many times d can be taken from r" (repeated subtraction)
#[derive(Clone, Copy)]
enum Pick { Trial { largest: bool, lhs: u8, rel: u8, rhs: u8 }, Count }

impl Pick {
    fn show(&self) -> String {
        match *self {
            Pick::Count => "count how many times d can be taken from r".into(),
            Pick::Trial { largest, lhs, rel, rhs } => format!("the {} q with {} {} {}", if largest { "largest" } else { "smallest" },
                ["q", "ADD(q, d)", "TIMES(q, d)", "TIMES(q, q)"][lhs as usize], ["<=", "<", ">=", ">"][rel as usize], ["r", "d"][rhs as usize]),
        }
    }
}

/// the 65 ideas, simplest first: LHS q, ADD, TIMES(q, d), TIMES(q, q); largest before smallest;
/// <=, <, >=, >; r before d; the loop last
fn picks() -> Vec<Pick> {
    let mut v = Vec::new();
    for lhs in 0..4 { for largest in [true, false] { for rel in 0..4 { for rhs in 0..2 { v.push(Pick::Trial { largest, lhs, rel, rhs }); } } } }
    v.push(Pick::Count);
    v
}

const PAGES: [&str; 4] = ["bring the next digit down, left to right", "bring the next digit down, right to left", "each digit divided alone", "the whole number in one step"];

/// her skills for dividing: WP9d's skills, her subtraction (also her comparison), and the column
/// idea she found in WP9d for "number x digit"
struct Div<'a> { sk: Skills<'a>, sub: &'a Habits, col: Ex }

impl Div<'_> {
    fn sub(&self, p: u64, q: u64) -> Option<u64> { sub_run(p, q, &mut |s| self.sub.best(s)).0 }
    /// p <= q: she can take p from q with no borrow left over
    fn le(&self, p: u64, q: u64) -> bool { self.sub(q, p).is_some() }
    fn rel(&self, p: u64, k: u8, q: u64) -> bool {
        match k { 0 => self.le(p, q), 1 => p != q && self.le(p, q), 2 => self.le(q, p), _ => p != q && self.le(q, p) }
    }
    /// n x one digit: a times fact when both are single digits, else her number x digit row
    fn nd(&self, n: u64, q: u64) -> Option<u64> {
        if n < 10 { self.sk.times(n, q) } else { self.sk.row(&self.col, n, q) }
    }
    fn lhs(&self, k: u8, q: u64, d: u64) -> Option<u64> {
        match k { 0 => Some(q), 1 => Some(self.sk.add(q, d)), 2 => self.nd(d, q), _ => self.sk.times(q, q) }
    }
    /// one step on what is in hand, r: (quotient digit, what is left)
    fn step(&self, p: Pick, r: u64, d: u64) -> Option<(u64, u64)> {
        match p {
            Pick::Count => {
                let (mut left, mut n) = (r, 0u64);
                while let Some(t) = self.sub(left, d) {
                    left = t;
                    n += 1;
                    if n > 100_000 { return None; }
                }
                Some((n, left))
            }
            Pick::Trial { largest, lhs, rel, rhs } => {
                let order: Vec<u64> = if largest { (0..10).rev().collect() } else { (0..10).collect() };
                let mut found = None;
                for q in order {
                    let l = self.lhs(lhs, q, d)?;
                    if self.rel(l, rel, if rhs == 0 { r } else { d }) { found = Some(q); break; }
                }
                let q = found?;
                Some((q, self.sub(r, self.nd(d, q)?)?))
            }
        }
    }
    /// a page: (quotient, remainder)
    fn page(&self, k: usize, p: Pick, a: u64, d: u64) -> Option<(u64, u64)> {
        let mut da = digits(a);
        match k {
            0 | 1 => {
                if k == 0 { da.reverse(); }
                let (mut r, mut qd) = (0u64, Vec::new());
                for x in da {
                    // the next digit is written after what is left (writing, not arithmetic)
                    let (q, left) = self.step(p, r * 10 + x as u64, d)?;
                    if q > 9 { return None; }
                    qd.push(q);
                    r = left;
                }
                if k == 1 { qd.reverse(); }
                Some((qd.iter().fold(0, |t, &q| t * 10 + q), r))
            }
            2 => {
                da.reverse();
                let (mut r, mut qd) = (0u64, Vec::new());
                for x in da {
                    let (q, left) = self.step(p, x as u64, d)?;
                    if q > 9 { return None; }
                    qd.push(q);
                    r = left;
                }
                Some((qd.iter().fold(0, |t, &q| t * 10 + q), r))
            }
            _ => self.step(p, a, d),
        }
    }
    /// her check: q x d by her multiplication page, ADD the remainder, and the remainder below d
    fn check(&self, a: u64, d: u64, (q, r): (u64, u64)) -> bool {
        let Some(back) = self.sk.page(0, &self.col, q, d) else { return false };
        self.sk.add(back, r) == a && r != d && self.le(r, d)
    }
}

/// (v29 WP9e) can she find long division from worked answers, using only skills she has?
fn division(add_h: &Habits, fact_h: &Habits, sub_h: &Habits) -> Option<(Pick, usize)> {
    println!("\n==== LONG DIVISION FROM HER OWN SKILLS (WP9e, fresh seeds) ====");
    let col = Ex::Add(Box::new(Ex::Times(Box::new(Ex::X), Box::new(Ex::M))), Box::new(Ex::C));
    let dv = Div { sk: Skills { add: add_h, fact: fact_h }, sub: sub_h, col };
    let ideas = picks();
    let situations: Vec<(u64, u64)> = (1..10u64).flat_map(|d| (0..10 * d).map(move |r| (r, d))).collect();
    let truth = |p: Pick| situations.iter().all(|&(r, d)| dv.step(p, r, d) == Some((r / d, r % d)));
    println!("{} ways to choose a quotient digit; {} of them right on all {} single steps", ideas.len(), ideas.iter().filter(|&&p| truth(p)).count(), situations.len());
    let mut x: u64 = 61;
    let ex1: Vec<(u64, u64)> = (0..20).map(|_| { let d = 1 + splitmix(&mut x) % 9; (splitmix(&mut x) % (10 * d), d) }).collect();
    let ex2: Vec<(u64, u64)> = (0..20).map(|_| { let n = 2 + (splitmix(&mut x) % 3) as u32; let a = number(n, &mut x); (a, 1 + splitmix(&mut x) % 9) }).collect();
    let mut y: u64 = 62;
    let t2: Vec<(u64, u64)> = (0..1000).map(|_| { let n = 4 + (splitmix(&mut y) % 5) as u32; let a = number(n, &mut y); (a, 1 + splitmix(&mut y) % 9) }).collect();
    let t3: Vec<(u64, u64)> = (0..1000).map(|_| { let n = 4 + (splitmix(&mut y) % 5) as u32; let a = number(n, &mut y); (a, number(2, &mut y)) }).collect();
    // the teacher's slips: 2 of the 20 quotients wrong, in each level
    let slip = |ex: &[(u64, u64)], z: &mut u64| -> Vec<(u64, u64)> {
        let mut ans: Vec<(u64, u64)> = ex.iter().map(|&(a, d)| (a / d, a % d)).collect();
        let i = (splitmix(z) % 20) as usize;
        let j = (i + 1 + (splitmix(z) % 19) as usize) % 20;
        for k in [i, j] { ans[k].0 += 1 + splitmix(z) % 9; }
        ans
    };
    let mut z: u64 = 63;
    let (noisy1, noisy2) = (slip(&ex1, &mut z), slip(&ex2, &mut z));
    let clean1: Vec<(u64, u64)> = ex1.iter().map(|&(a, d)| (a / d, a % d)).collect();
    let clean2: Vec<(u64, u64)> = ex2.iter().map(|&(a, d)| (a / d, a % d)).collect();
    let pct = |k: usize, n: usize| 100.0 * k as f64 / n as f64;
    let mut chosen = None;

    for (label, ans1, ans2, need) in [("clean examples", &clean1, &clean2, 20usize), ("2 of 20 answers wrong", &noisy1, &noisy2, 16)] {
        println!("\n-- {label} (an idea must fit {need} of 20) --");
        let fits1 = |p: Pick, k: usize| ex1[..k].iter().zip(&ans1[..k]).filter(|(&(r, d), &ans)| dv.step(p, r, d) == Some(ans)).count();
        print!("level 1, ways still fitting after 1/2/3/5/10/20 examples:");
        for k in [1, 2, 3, 5, 10, 20] { print!(" {}", ideas.iter().filter(|&&p| fits1(p, k) >= (need * k).div_ceil(20)).count()); }
        println!();
        let Some(&p) = ideas.iter().find(|&&p| fits1(p, 20) >= need) else { println!("  no way fits: FAIL"); continue };
        println!("  chosen: {}   right on all {} single steps: {}", p.show(), situations.len(), truth(p));
        let fits2 = |k: usize, n: usize| ex2[..n].iter().zip(&ans2[..n]).filter(|(&(a, d), &ans)| dv.page(k, p, a, d) == Some(ans)).count();
        print!("level 2, pages still fitting after 1/2/3/5/10/20 examples:");
        for n in [1, 2, 3, 5, 10, 20] { print!(" {}", (0..4).filter(|&k| fits2(k, n) >= (need * n).div_ceil(20)).count()); }
        println!();
        let Some(k) = (0..4).find(|&k| fits2(k, 20) >= need) else { println!("  no page fits: FAIL"); continue };
        chosen.get_or_insert((p, k));
        for (name, tests) in [("unseen 4-8 digit / 1 digit", &t2), ("transfer: unseen 4-8 digit / 2 digits", &t3)] {
            let (mut right, mut answered, mut right_answered, mut refused_right, mut caught) = (0, 0, 0, 0, 0);
            for &(a, d) in tests.iter() {
                let got = dv.page(k, p, a, d);
                let ok = got == Some((a / d, a % d));
                right += ok as usize;
                let kept = got.map_or(false, |g| dv.check(a, d, g));
                if kept { answered += 1; right_answered += ok as usize; }
                if !kept && ok { refused_right += 1; }
                if !kept && !ok { caught += 1; }
            }
            println!("  chosen page: {}   {name}: {:.1}% right; her check kept {} ({:.1}% right among them), refused {} right, caught {} of {} wrong",
                     PAGES[k], pct(right, tests.len()), answered, pct(right_answered, answered.max(1)), refused_right, caught, tests.len() - right);
        }
    }
    chosen
}

/// (v29 WP9f) her own check as the teacher: she makes up subtraction problems, answers with her
/// habits (acting in proportion to value, as in R), and the only reward is her own check, adding
/// the answer back to b with her addition. No teacher and no true answer are used.
fn self_taught(add_h: &Habits, fact_h: &Habits, sub_h: &Habits, t6: &[(u64, u64)], t12: &[(u64, u64)], found: Option<(Pick, usize)>) {
    println!("\n==== HER OWN CHECK AS THE TEACHER (WP9f) ====");
    let sub = |a: u64, b: u64| a - b;
    let wrong_habits = |h: &Habits| -> Vec<String> {
        (0..200).filter_map(|s| {
            let (p, q, w) = (s / 20, (s / 2) % 10, s % 2);
            let b = h.best(s);
            (b != sub_act(p, q, w)).then(|| format!("{p}-{q}-borrow{w} -> wrote {} borrow {}", b / 2, b % 2))
        }).collect()
    };
    let mut h = sub_h.clone();
    let r = |h: &Habits, a: u64, b: u64| sub_run(a, b, &mut |s| h.best(s)).0;
    println!("before: unseen 4-6 {:.2}%   7-12 {:.2}%   habits wrong {:?}", score(t6, sub, |a, b| r(&h, a, b)), score(t12, sub, |a, b| r(&h, a, b)), wrong_habits(&h));
    // problems she makes up (seed 71); her choices (seed 72)
    let (mut x, mut z): (u64, u64) = (71, 72);
    let (mut passed, mut first_right) = (0usize, None);
    let s99 = state(9, 9, 1);
    for i in 1..=200_000usize {
        let (n1, n2) = (4 + (splitmix(&mut x) % 9) as u32, 4 + (splitmix(&mut x) % 9) as u32);
        let (p, q) = (number(n1, &mut x), number(n2, &mut x));
        let (a, b) = (p.max(q), p.min(q));
        let (ans, used) = sub_run(a, b, &mut |s| h.sample(s, &mut z));
        // her check: add the answer back to b with her own addition
        let ok = ans.map_or(false, |d| run(d, b, &mut |s| add_h.best(s)).0 == a);
        h.learn(&used, if ok { 1.0 } else { 0.0 });
        passed += ok as usize;
        if first_right.is_none() && h.best(s99) == sub_act(9, 9, 1) { first_right = Some(i); }
        if MILESTONES.contains(&i) || i == 200_000 {
            println!("  {:>7} made up: her check passed {:.1}% of them so far; unseen 7-12 {:.2}%; habits wrong {}", i, 100.0 * passed as f64 / i as f64,
                     score(t12, sub, |a, b| r(&h, a, b)), wrong_habits(&h).len());
        }
    }
    let wrong = wrong_habits(&h);
    println!("after:  unseen 4-6 {:.2}%   7-12 {:.2}%   habits wrong {} {:?}   9-9-borrow1 right from problem {}",
             score(t6, sub, |a, b| r(&h, a, b)), score(t12, sub, |a, b| r(&h, a, b)), wrong.len(), wrong, first_right.map_or("never".into(), |n| n.to_string()));
    // WP9e's transfer again, with the habits her check taught (the same tests, seed 62)
    let Some((p, k)) = found else { println!("  (WP9e found no procedure; no transfer re-run)"); return };
    let col = Ex::Add(Box::new(Ex::Times(Box::new(Ex::X), Box::new(Ex::M))), Box::new(Ex::C));
    let dv = Div { sk: Skills { add: add_h, fact: fact_h }, sub: &h, col };
    let mut y: u64 = 62;
    let _t2: Vec<(u64, u64)> = (0..1000).map(|_| { let n = 4 + (splitmix(&mut y) % 5) as u32; let a = number(n, &mut y); (a, 1 + splitmix(&mut y) % 9) }).collect();
    let t3: Vec<(u64, u64)> = (0..1000).map(|_| { let n = 4 + (splitmix(&mut y) % 5) as u32; let a = number(n, &mut y); (a, number(2, &mut y)) }).collect();
    let (mut right, mut kept, mut refused_right) = (0, 0, 0);
    for &(a, d) in &t3 {
        let got = dv.page(k, p, a, d);
        let ok = got == Some((a / d, a % d));
        let keep = got.map_or(false, |g| dv.check(a, d, g));
        right += ok as usize;
        kept += keep as usize;
        if ok && !keep { refused_right += 1; }
    }
    println!("WP9e transfer again (4-8 digit / 2 digits): {:.1}% right; her check kept {kept}, refused {refused_right} right", 100.0 * right as f64 / t3.len() as f64);
}
