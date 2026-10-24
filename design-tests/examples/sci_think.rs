// TESTING / DESIGN ONLY. (v30) Offline readouts over read_think's dumped memories (Train + Dev only).
//   sci_think <plain.tsv> <contrast.tsv>
// Each row: set, id, key, choice, margin, scores, question, choices, tops (per choice its 10
// strongest memory strengths, ';' inside a choice, ',' between choices).
// Everything here is fitted on Train (Easy-Train + Chal-Train) and only reported on Dev.
//   views      plain recall (question + all choice words) and contrast recall (only the words that
//              set a choice apart); each choice's evidence = sum of its k strongest memories,
//              divided by the best choice's (so the two views are on one scale)
//   two views  evidence = a * plain + (1 - a) * contrast, a and k chosen on Train
//   polarity   a word in the question that, on Train, more often means "the least supported
//              choice is right" than "the most supported is" (a one-sided sign test on the
//              questions where the two differ, p < 0.01, at least 20 such questions) turns the
//              reading round for questions holding it. Learned, graded by evidence, not listed.
use std::collections::HashMap;

/// masks: per choice, per memory, which cue words it held (question words first, then the choice's);
/// qspec / cspec: those words' specificities (present in dumps from the coverage build)
struct Row { set: String, key: usize, q: String, tops: Vec<Vec<f64>>, masks: Vec<Vec<u64>>, qspec: Vec<f64>, cspec: Vec<Vec<f64>> }

fn load(p: &str) -> Vec<Row> {
    std::fs::read_to_string(p).expect("read tsv").lines().filter(|l| !l.is_empty()).map(|l| {
        let f: Vec<&str> = l.split('\t').collect();
        let cell = |x: &str| -> (f64, u64) { let mut it = x.split(':'); (it.next().unwrap_or("0").parse().unwrap_or(0.0), it.next().map_or(0, |m| u64::from_str_radix(m, 16).unwrap_or(0))) };
        let tops = f[8].split(',').map(|c| c.split(';').map(|x| cell(x).0).collect()).collect();
        let masks = f[8].split(',').map(|c| c.split(';').map(|x| cell(x).1).collect()).collect();
        let nums = |s: &str| -> Vec<f64> { s.split(';').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap_or(0.0)).collect() };
        let qspec = f.get(9).map_or(Vec::new(), |s| nums(s));
        let cspec = f.get(10).map_or(Vec::new(), |s| s.split(',').map(nums).collect());
        Row { set: f[0].to_string(), key: f[2].parse().unwrap_or(usize::MAX), q: f[6].to_lowercase(), tops, masks, qspec, cspec }
    }).collect()
}

fn ev(tops: &[Vec<f64>], k: usize) -> Vec<f64> {
    let s: Vec<f64> = tops.iter().map(|t| t.iter().take(k).sum()).collect();
    let m = s.iter().cloned().fold(0.0, f64::max);
    if m > 0.0 { s.iter().map(|x| x / m).collect() } else { s }
}

fn argbest(s: &[f64], max: bool) -> usize {
    let mut b = 0;
    for i in 1..s.len() { if (max && s[i] > s[b]) || (!max && s[i] < s[b]) { b = i; } }
    b
}

fn words(q: &str) -> Vec<String> {
    let mut v: Vec<String> = q.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(String::from).collect();
    v.sort();
    v.dedup();
    v
}

/// one-sided sign test: P(X >= x) for X ~ Binomial(n, 1/2)
fn sign_p(x: usize, n: usize) -> f64 {
    let mut lc = 0.0f64; // ln C(n, i), accumulated
    let mut p = 0.0;
    for i in 0..=n {
        if i > 0 { lc += ((n - i + 1) as f64).ln() - (i as f64).ln(); }
        if i >= x { p += (lc - n as f64 * std::f64::consts::LN_2).exp(); }
    }
    p
}

const SETS: [&str; 4] = ["Easy-Train", "Easy-Dev", "Chal-Train", "Chal-Dev"];

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.get(1).map(String::as_str) == Some("cover") { return cover_main(&a[2]); }
    let (pl, co) = (load(&a[1]), load(&a[2]));
    assert_eq!(pl.len(), co.len());
    let train = |r: &Row| r.set.ends_with("Train");
    let n = pl.len();
    let report = |name: &str, pick: &dyn Fn(usize) -> usize| {
        let mut out = format!("{name:<44}");
        for s in SETS {
            let ids: Vec<usize> = (0..n).filter(|&i| pl[i].set == s).collect();
            let r = ids.iter().filter(|&&i| pick(i) == pl[i].key).count();
            out += &format!(" {:>10.2}", 100.0 * r as f64 / ids.len().max(1) as f64);
        }
        println!("{out}");
    };
    println!("{:<44} {:>10} {:>10} {:>10} {:>10}", "readout (accuracy %, every question answered)", SETS[0], SETS[1], SETS[2], SETS[3]);
    report("plain, sum of 10 (today)", &|i| argbest(&ev(&pl[i].tops, 10), true));
    report("contrast, sum of 10", &|i| argbest(&ev(&co[i].tops, 10), true));
    // two views: a and k chosen on Train (all Train questions pooled)
    let mut best = (0usize, 0.0f64, 0usize);
    for k in [1, 3, 5, 10] {
        for ai in 0..=10 {
            let al = ai as f64 / 10.0;
            let r = (0..n).filter(|&i| train(&pl[i])).filter(|&i| {
                let (p, c) = (ev(&pl[i].tops, k), ev(&co[i].tops, k));
                let s: Vec<f64> = p.iter().zip(&c).map(|(x, y)| al * x + (1.0 - al) * y).collect();
                argbest(&s, true) == pl[i].key
            }).count();
            if r > best.0 { best = (r, al, k); }
        }
    }
    let (_, al, k) = best;
    let two = |i: usize| -> Vec<f64> { let (p, c) = (ev(&pl[i].tops, k), ev(&co[i].tops, k)); p.iter().zip(&c).map(|(x, y)| al * x + (1.0 - al) * y).collect() };
    report(&format!("two views (a={al:.1} plain, k={k}; chosen on Train)"), &|i| argbest(&two(i), true));
    // polarity, learned on Train under the plain reading
    let mut tally: HashMap<String, (usize, usize)> = HashMap::new(); // (min right & max wrong, max right & min wrong)
    for i in (0..n).filter(|&i| train(&pl[i])) {
        let s = ev(&pl[i].tops, 10);
        let (mx, mn) = (argbest(&s, true) == pl[i].key, argbest(&s, false) == pl[i].key);
        if mx == mn { continue; }
        for w in words(&pl[i].q) { let e = tally.entry(w).or_default(); if mn { e.0 += 1 } else { e.1 += 1 } }
    }
    let mut look: Vec<(String, usize, usize)> = tally.iter().filter(|(_, (x, y))| x + y >= 8).map(|(w, &(x, y))| (w.clone(), x, y)).collect();
    look.sort_by(|a, b| ((b.1 as f64 + 1.0) / (b.2 as f64 + 1.0)).partial_cmp(&((a.1 as f64 + 1.0) / (a.2 as f64 + 1.0))).unwrap());
    println!("
most min-leaning words on Train (min right / max right), for inspection:");
    for l in look.iter().take(15) { println!("  {:<12} {:>4} / {:<4} p={:.3}", l.0, l.1, l.2, sign_p(l.1, l.1 + l.2)); }
    let mut flips: Vec<(String, usize, usize, f64)> = tally.into_iter().filter(|(_, (x, y))| x + y >= 20 && x > y)
        .map(|(w, (x, y))| { let p = sign_p(x, x + y); (w, x, y, p) }).filter(|f| f.3 < 0.01).collect();
    flips.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());
    println!("\nwords learned on Train to mean 'the least supported choice' (min right / max right, p):");
    for f in &flips { println!("  {:<12} {:>4} / {:<4} p={:.2e}", f.0, f.1, f.2, f.3); }
    let flip = |i: usize| words(&pl[i].q).iter().any(|w| flips.iter().any(|f| &f.0 == w));
    report("plain + learned polarity", &|i| argbest(&ev(&pl[i].tops, 10), !flip(i)));
    report("two views + learned polarity", &|i| argbest(&two(i), !flip(i)));
    let nf: Vec<usize> = SETS.iter().map(|s| (0..n).filter(|&i| &pl[i].set == s && flip(i)).count()).collect();
    println!("questions turned round: {:?}", nf);
}

/// how much of the question and the choice the k strongest memories cover together: the summed
/// specificity of every cue word held by at least one of them (an explanation, not a vote)
fn cover(r: &Row, k: usize) -> Vec<f64> { cover_q(r, k, false) }

/// q_only: count only the question's words (every choice is judged against the same question)
fn cover_q(r: &Row, k: usize, q_only: bool) -> Vec<f64> {
    let nq = r.qspec.len();
    let s: Vec<f64> = (0..r.tops.len()).map(|ch| {
        let u = r.masks[ch].iter().take(k).fold(0u64, |a, &m| a | m);
        let mut v = 0.0;
        for b in 0..64 { if q_only && b >= nq { break; } if u >> b & 1 == 1 { v += if b < nq { r.qspec[b] } else { r.cspec[ch].get(b - nq).copied().unwrap_or(0.0) }; } }
        v
    }).collect();
    let m = s.iter().cloned().fold(0.0, f64::max);
    if m > 0.0 { s.iter().map(|x| x / m).collect() } else { s }
}

/// `sci_think cover <dump.tsv>`: plain evidence + lambda x coverage, lambda and k chosen on Train
pub fn cover_main(p: &str) {
    let rows = load(p);
    let n = rows.len();
    let train = |r: &Row| r.set.ends_with("Train");
    let report = |name: &str, pick: &dyn Fn(usize) -> usize| {
        let mut out = format!("{name:<52}");
        for s in SETS {
            let ids: Vec<usize> = (0..n).filter(|&i| rows[i].set == s).collect();
            let r = ids.iter().filter(|&&i| pick(i) == rows[i].key).count();
            out += &format!(" {:>10.2}", 100.0 * r as f64 / ids.len().max(1) as f64);
        }
        println!("{out}");
    };
    println!("{:<52} {:>10} {:>10} {:>10} {:>10}", "readout (accuracy %)", SETS[0], SETS[1], SETS[2], SETS[3]);
    report("plain, sum of 10 (today)", &|i| argbest(&ev(&rows[i].tops, 10), true));
    for k in [1, 2, 3, 5, 10] { report(&format!("coverage of the {k} strongest"), &|i| argbest(&cover(&rows[i], k), true)); }
    for k in [1, 2, 3, 5, 10] { report(&format!("question coverage of the {k} strongest"), &|i| argbest(&cover_q(&rows[i], k, true), true)); }
    let mut bq = (0usize, 0.0f64, 0usize);
    for k in [1, 2, 3, 5, 10] { for li in 0..=20 { let l = li as f64 / 10.0;
        let r = (0..n).filter(|&i| train(&rows[i])).filter(|&i| { let (p, c) = (ev(&rows[i].tops, 10), cover_q(&rows[i], k, true)); argbest(&p.iter().zip(&c).map(|(x, y)| x + l * y).collect::<Vec<_>>(), true) == rows[i].key }).count();
        if r > bq.0 { bq = (r, l, k); } } }
    let (_, lq, kq) = bq;
    report(&format!("plain + {lq:.1} x question coverage of {kq} (Train)"), &|i| { let (p, c) = (ev(&rows[i].tops, 10), cover_q(&rows[i], kq, true)); argbest(&p.iter().zip(&c).map(|(x, y)| x + lq * y).collect::<Vec<_>>(), true) });
    let mut best = (0usize, 0.0f64, 0usize);
    for k in [1, 2, 3, 5, 10] {
        for li in 0..=20 {
            let l = li as f64 / 10.0;
            let r = (0..n).filter(|&i| train(&rows[i])).filter(|&i| {
                let (p, c) = (ev(&rows[i].tops, 10), cover(&rows[i], k));
                argbest(&p.iter().zip(&c).map(|(x, y)| x + l * y).collect::<Vec<_>>(), true) == rows[i].key
            }).count();
            if r > best.0 { best = (r, l, k); }
        }
    }
    let (_, l, k) = best;
    report(&format!("plain + {l:.1} x coverage of {k} (chosen on Train)"), &|i| {
        let (p, c) = (ev(&rows[i].tops, 10), cover(&rows[i], k));
        argbest(&p.iter().zip(&c).map(|(x, y)| x + l * y).collect::<Vec<_>>(), true)
    });
}
