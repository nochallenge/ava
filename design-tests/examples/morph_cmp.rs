// TESTING / DESIGN ONLY. (v29 WP5a) Compare two morph_backoff runs on the same split, form by form.
//
//   cargo run --release --example morph_cmp -- <base dump> <new dump>
//
// Each dump is AVA_MORPH_DUMP from morph_backoff (one line per withheld past-tense form, in the
// same order for the same split). Prints overall / regular / irregular right for both and the
// paired-bootstrap 95% CI of the change in forms right (20,000 resamples, splitmix64 seed 1).
use std::fs;

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// (lemma, tags, gold, kind, ok)
fn read(p: &str) -> Vec<(String, String, String, String, bool)> {
    fs::read_to_string(p).expect("read a dump").lines().skip(1).map(|l| {
        let c: Vec<&str> = l.split('\t').collect();
        (c[0].to_string(), c[1].to_string(), c[2].to_string(), c[3].to_string(), c[5] == "1")
    }).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() < 3 { eprintln!("usage: morph_cmp <base dump> <new dump>"); std::process::exit(1); }
    let (x, y) = (read(&a[1]), read(&a[2]));
    assert_eq!(x.len(), y.len(), "the two dumps hold different numbers of forms: not the same split");
    for (p, q) in x.iter().zip(&y) { assert!(p.0 == q.0 && p.1 == q.1 && p.2 == q.2, "forms differ: {} vs {}", p.0, q.0); }
    let n = x.len();
    let pc = |v: &[(String, String, String, String, bool)], kind: Option<&str>| {
        let s: Vec<_> = v.iter().filter(|r| kind.map_or(true, |k| r.3 == k)).collect();
        (s.iter().filter(|r| r.4).count(), s.len())
    };
    for (name, kind) in [("all withheld", None), ("regular", Some("regular")), ("irregular", Some("irregular"))] {
        let ((b, nb), (c, _)) = (pc(&x, kind), pc(&y, kind));
        println!("{name:<14} {nb:>5}: base {:>6.2}%   new {:>6.2}%   ({:+} forms)", 100.0 * b as f64 / nb as f64, 100.0 * c as f64 / nb as f64, c as i64 - b as i64);
    }
    let d: Vec<i64> = x.iter().zip(&y).map(|(p, q)| q.4 as i64 - p.4 as i64).collect();
    let mut s: u64 = 1;
    let mut boots: Vec<i64> = (0..20_000).map(|_| (0..n).map(|_| d[(splitmix(&mut s) % n as u64) as usize]).sum()).collect();
    boots.sort_unstable();
    println!("change {:+} forms, 95% CI {:+} .. {:+} (paired bootstrap, 20,000, seed 1)", d.iter().sum::<i64>(), boots[500], boots[19_499]);
    let flips: Vec<String> = x.iter().zip(&y).filter(|(p, q)| p.4 != q.4).take(24)
        .map(|(p, q)| format!("{} {} -> {} [{}]: {}", p.0, p.1, p.2, p.3, if q.4 { "now right" } else { "now wrong" })).collect();
    println!("first changes:\n  {}", flips.join("\n  "));
}
