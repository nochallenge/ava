// TESTING / DESIGN ONLY. The paired bootstrap for v29's switches (WP2 AVA_WM, WP3 AVA_PREV, WP5
// AVA_DEDUP). std only.
//
//   cargo run --release --example paired_boot -- <off.dump> <on.dump>
//   cargo run --release --example paired_boot -- --selftest
//
// The dumps are compose's AVA_DUMP files (with AVA_MASK and the same AVA_MASK_HALF) from two runs
// that differ only in the switch. Per position (P lines): her switch line, 0 or 1. Per prompt
// (E lines): switch-greedy word-F1, proper, exact.
//
// Rule 1 (pre-registered): next-word top-1 of her switch line on the half's positions, on minus off,
// in points, with a PAIRED bootstrap 95% CI: 20,000 resamples of the n positions with replacement,
// each resample drawing the same positions for both arms (so it resamples the per-position
// difference), indices drawn by splitmix64 seeded with 1 (index = next() % n, in order, resample
// after resample). CI = the 2.5% and 97.5% points of the sorted resampled differences: sorted[500]
// and sorted[19499] (500 resamples lie below the low end and 500 above the high end).
// Rules 2 and 3: switch-greedy word-F1 and the proper-reply rate on the half's prompts, on minus off,
// in points (point estimates; the pre-registration sets no interval for them).
//
// Nothing here reads a model; it only compares two files.
use std::fs;

const RESAMPLES: usize = 20_000;
const SEED: u64 = 1;

/// splitmix64 (Steele, Lea & Flood 2014; Vigna's reference constants)
struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// mean(on - off) in points, and its paired bootstrap 95% interval
fn paired(off: &[f64], on: &[f64], seed: u64) -> (f64, f64, f64) {
    assert_eq!(off.len(), on.len());
    let n = off.len();
    assert!(n > 0, "nothing to compare");
    let d: Vec<f64> = on.iter().zip(off).map(|(a, b)| a - b).collect();
    let mean = d.iter().sum::<f64>() / n as f64 * 100.0;
    let mut rng = SplitMix(seed);
    let mut means: Vec<f64> = (0..RESAMPLES).map(|_| {
        let mut s = 0f64;
        for _ in 0..n { s += d[(rng.next() % n as u64) as usize]; }
        s / n as f64 * 100.0
    }).collect();
    means.sort_by(|a, b| a.total_cmp(b));
    (mean, means[RESAMPLES * 25 / 1000], means[RESAMPLES * 975 / 1000 - 1])
}

#[derive(Default)]
struct Dump { half: String, mask: String, head: String, pos: Vec<(usize, usize, u8)>, ex: Vec<(usize, f64, u8, u8)> }

fn read(path: &str) -> Result<Dump, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let mut d = Dump::default();
    let bit = |s: &str, at: &str| -> Result<u8, String> {
        match s { "0" => Ok(0), "1" => Ok(1), _ => Err(format!("{at}: {s:?} is not 0 or 1")) }
    };
    for (ln, l) in text.lines().enumerate() {
        let at = format!("{path}:{}", ln + 1);
        if let Some(h) = l.strip_prefix("# ava dump v1\t") {
            d.head = h.to_string();
            for kv in h.split('\t') {
                if let Some(v) = kv.strip_prefix("half=") { d.half = v.to_string(); }
                if let Some(v) = kv.strip_prefix("mask=") { d.mask = v.to_string(); }
            }
            continue;
        }
        if l.is_empty() || l.starts_with('#') { continue; }
        let f: Vec<&str> = l.split('\t').collect();
        let num = |s: &str| s.parse::<usize>().map_err(|_| format!("{at}: {s:?} is not a whole number"));
        match (f[0], f.len()) {
            ("P", 7) => d.pos.push((num(f[1])?, num(f[2])?, bit(f[3], &at)?)),
            ("E", 5) => d.ex.push((num(f[1])?, f[2].parse::<f64>().map_err(|_| format!("{at}: bad F1"))?, bit(f[3], &at)?, bit(f[4], &at)?)),
            _ => return Err(format!("{at}: not a dump line")),
        }
    }
    if d.half.is_empty() { return Err(format!("{path}: no '# ava dump v1' header")); }
    d.pos.sort_by_key(|p| p.0);
    d.ex.sort_by_key(|e| e.0);
    Ok(d)
}

struct Verdict { n_pos: usize, line: [f64; 2], r1: (f64, f64, f64), n_ex: usize, f1: [f64; 2], proper: [f64; 2], exact: [f64; 2] }

fn compare(off: &Dump, on: &Dump) -> Result<Verdict, String> {
    if off.half != on.half { return Err(format!("the dumps are of different halves ({} vs {})", off.half, on.half)); }
    if off.mask != on.mask { eprintln!("note: the dumps name different mask files ({} vs {})", off.mask, on.mask); }
    if off.pos.len() != on.pos.len() || off.pos.iter().zip(&on.pos).any(|(a, b)| a.0 != b.0 || a.1 != b.1) {
        return Err("the dumps do not hold the same positions (position, exchange)".into());
    }
    if off.ex.len() != on.ex.len() || off.ex.iter().zip(&on.ex).any(|(a, b)| a.0 != b.0) {
        return Err("the dumps do not hold the same prompts".into());
    }
    if off.pos.is_empty() { return Err("no positions (P lines) in the dumps".into()); }
    let (a, b): (Vec<f64>, Vec<f64>) = (off.pos.iter().map(|p| p.2 as f64).collect(), on.pos.iter().map(|p| p.2 as f64).collect());
    let pc = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64 * 100.0;
    let r1 = paired(&a, &b, SEED);
    let m = off.ex.len().max(1) as f64;
    let ex = |d: &Dump, k: usize| -> f64 {
        d.ex.iter().map(|e| match k { 0 => e.1, 1 => e.2 as f64, _ => e.3 as f64 }).sum::<f64>() / m * 100.0
    };
    Ok(Verdict { n_pos: a.len(), line: [pc(&a), pc(&b)], r1, n_ex: off.ex.len(),
                 f1: [ex(off, 0), ex(on, 0)], proper: [ex(off, 1), ex(on, 1)], exact: [ex(off, 2), ex(on, 2)] })
}

fn report(v: &Verdict, half: &str) {
    let yes = |b: bool| if b { "HOLDS" } else { "FAILS" };
    let (d, lo, hi) = v.r1;
    println!("paired comparison, {half} half (off = first dump, on = second)");
    println!("  rule 1  next word top-1, her switch line: off {:.2}%  on {:.2}%  diff {:+.3} points, 95% CI {:+.3} .. {:+.3}",
        v.line[0], v.line[1], d, lo, hi);
    println!("          ({} positions, {RESAMPLES} paired resamples, splitmix64 seed {SEED})  higher with the CI above 0 -> {}",
        v.n_pos, yes(d > 0.0 && lo > 0.0));
    if v.n_ex > 0 {
        let (df, dp) = (v.f1[1] - v.f1[0], v.proper[1] - v.proper[0]);
        println!("  rule 2  switch-greedy word-F1: off {:.3}  on {:.3}  diff {:+.3} points (needed >= -0.30) -> {}", v.f1[0], v.f1[1], df, yes(df >= -0.3));
        println!("  rule 3  proper replies: off {:.3}%  on {:.3}%  diff {:+.3} points (needed >= -0.50) -> {}", v.proper[0], v.proper[1], dp, yes(dp >= -0.5));
        println!("  (info)  exact replies: off {:.3}%  on {:.3}%  diff {:+.3}   ({} prompts)", v.exact[0], v.exact[1], v.exact[1] - v.exact[0], v.n_ex);
    } else {
        println!("  rules 2-3: no prompts (E lines) in the dumps; they come from a reply run (probe bit 32)");
    }
    println!("TSV\tpaired\t{half}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}", v.n_pos, v.line[0], v.line[1], d, lo, hi,
        v.n_ex, v.f1[0], v.f1[1], v.proper[0], v.proper[1]);
}

/// synthetic cases with answers known in advance
fn selftest() -> bool {
    let mut all = true;
    let mut check = |name: &str, ok: bool, got: String| {
        println!("  {} {name}: {got}", if ok { "PASS" } else { "FAIL" });
        all &= ok;
    };
    // 1. the generator: Vigna's splitmix64.c, seed 1234567, first five outputs
    let mut g = SplitMix(1_234_567);
    let first: Vec<u64> = (0..5).map(|_| g.next()).collect();
    check("splitmix64 seed 1234567 matches the reference sequence",
        first == [6457827717110365317, 3203168211198807973, 9817491932198370423, 4593380528125082431, 16408922859458223821],
        format!("{first:?}"));
    // a deterministic stream of 0/1 for the cases below (not the bootstrap's own seed)
    let mut s = SplitMix(42);
    let coin: Vec<f64> = (0..5000).map(|_| (s.next() >> 63) as f64).collect();
    // 2. two identical arms: a paired bootstrap has nothing to resample, the CI is exactly [0, 0]
    //    (an unpaired one would be about ±2 points wide here)
    let (d, lo, hi) = paired(&coin, &coin, SEED);
    check("identical arms -> diff 0, CI exactly 0 .. 0", d == 0.0 && lo == 0.0 && hi == 0.0, format!("{d} [{lo}, {hi}]"));
    // 3. on right everywhere, off wrong everywhere -> exactly +100, CI [100, 100]
    let (z, o) = (vec![0.0; 1000], vec![1.0; 1000]);
    let (d, lo, hi) = paired(&z, &o, SEED);
    check("on always right, off never -> +100, CI 100 .. 100", d == 100.0 && lo == 100.0 && hi == 100.0, format!("{d} [{lo}, {hi}]"));
    // 4. on gains exactly 10% of 10,000 positions, loses none: the resampled mean is Binomial(n, 0.1)/n,
    //    so the 95% interval is 10 ± 1.96·sqrt(0.1·0.9/10000)·100 = 10 ± 0.588 points
    let off: Vec<f64> = (0..10_000).map(|i| ((i * 7) % 3 == 0) as u8 as f64).collect();
    let on: Vec<f64> = off.iter().enumerate().map(|(i, &x)| if i % 10 == 0 { 1.0 } else { x }).collect();
    // positions where off is already right cannot gain: make the gained ones off-wrong ones
    let off: Vec<f64> = off.iter().enumerate().map(|(i, &x)| if i % 10 == 0 { 0.0 } else { x }).collect();
    let (d, lo, hi) = paired(&off, &on, SEED);
    check("+10% of positions -> +10.000, CI ≈ 9.412 .. 10.588 (normal theory, ±0.05)",
        (d - 10.0).abs() < 1e-9 && (lo - 9.412).abs() < 0.05 && (hi - 10.588).abs() < 0.05, format!("{d:.3} [{lo:.3}, {hi:.3}]"));
    // 5. gains at 15%, losses at 5% of 10,000: mean +10, sd of the difference sqrt(0.20 - 0.01) = 0.4359,
    //    so the interval is 10 ± 1.96·0.4359/100·100 = 10 ± 0.854 points
    let mut off = vec![0.0; 10_000];
    let mut on = vec![0.0; 10_000];
    for i in 0..10_000 { match i % 20 { 0..=2 => on[i] = 1.0, 3 => off[i] = 1.0, _ => {} } }
    let (d, lo, hi) = paired(&off, &on, SEED);
    check("+15% / -5% -> +10.000, CI ≈ 9.146 .. 10.854 (normal theory, ±0.05)",
        (d - 10.0).abs() < 1e-9 && (lo - 9.146).abs() < 0.05 && (hi - 10.854).abs() < 0.05, format!("{d:.3} [{lo:.3}, {hi:.3}]"));
    // 6. the same seed gives the same interval twice; another seed a slightly different one
    let again = paired(&off, &on, SEED);
    let other = paired(&off, &on, 2);
    check("seed 1 repeats exactly; seed 2 differs", again == (d, lo, hi) && other != again, format!("{again:?} vs {other:?}"));
    // 7. end to end through files: two dumps as compose writes them
    let dir = std::env::temp_dir();
    let (pa, pb) = (dir.join(format!("paired_boot_selftest_{}_off.dump", std::process::id())),
                    dir.join(format!("paired_boot_selftest_{}_on.dump", std::process::id())));
    let head = "# ava dump v1\thalf=screen\tmask=test.tsv\tAVA_WM=-\n# P\tposition\texchange\ther_switch_line\tplain_top1\trouted_per_cell\tswitch\n";
    let (mut ta, mut tb) = (head.to_string(), head.to_string());
    for i in 0..10_000usize {
        let (a, b) = (0, (i % 10 == 0) as u8);
        ta += &format!("P\t{}\t{}\t{a}\t0\t0\t{a}\n", i * 2, i / 10);
        tb += &format!("P\t{}\t{}\t{b}\t0\t0\t{a}\n", i * 2, i / 10);
    }
    for e in 0..1000 {
        ta += &format!("E\t{e}\t0.500000000\t1\t0\n");
        tb += &format!("E\t{e}\t0.600000000\t{}\t0\n", (e % 100 != 0) as u8);
    }
    let wrote = fs::write(&pa, ta).and_then(|_| fs::write(&pb, tb));
    let got = wrote.map_err(|e| e.to_string()).and_then(|_| {
        let (a, b) = (read(&pa.to_string_lossy())?, read(&pb.to_string_lossy())?);
        compare(&a, &b)
    });
    fs::remove_file(&pa).ok();
    fs::remove_file(&pb).ok();
    match got {
        Ok(v) => check("files: rule 1 +10, F1 +10 points, proper -1 point",
            (v.r1.0 - 10.0).abs() < 1e-9 && (v.f1[1] - v.f1[0] - 10.0).abs() < 1e-6 && (v.proper[1] - v.proper[0] + 1.0).abs() < 1e-9
                && v.n_pos == 10_000 && v.n_ex == 1000,
            format!("rule 1 {:+.3} [{:+.3}, {:+.3}], F1 {:+.3}, proper {:+.3}", v.r1.0, v.r1.1, v.r1.2, v.f1[1] - v.f1[0], v.proper[1] - v.proper[0])),
        Err(e) => check("files", false, e),
    }
    // 8. dumps of different halves are refused
    let a = Dump { half: "screen".into(), pos: vec![(0, 0, 1)], ..Default::default() };
    let b = Dump { half: "confirm".into(), pos: vec![(0, 0, 1)], ..Default::default() };
    check("different halves are refused", compare(&a, &b).is_err(), "refused".into());
    all
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.get(1).map(|s| s.as_str()) == Some("--selftest") {
        println!("paired_boot self-test");
        let ok = selftest();
        println!("{}", if ok { "self-test: ALL PASS" } else { "self-test: FAILED" });
        std::process::exit(if ok { 0 } else { 1 });
    }
    if a.len() != 3 { eprintln!("usage: paired_boot <off.dump> <on.dump>   |   paired_boot --selftest"); std::process::exit(2); }
    let run = || -> Result<(), String> {
        let (off, on) = (read(&a[1])?, read(&a[2])?);
        println!("  off: {}  ({})", a[1], off.head);
        println!("  on : {}  ({})", a[2], on.head);
        report(&compare(&off, &on)?, &off.half);
        Ok(())
    };
    if let Err(e) = run() { eprintln!("paired_boot: {e}"); std::process::exit(1); }
}
