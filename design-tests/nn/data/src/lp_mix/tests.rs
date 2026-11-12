// TESTING / DESIGN ONLY — tests of src/lp_mix.rs (v42 B6, learning-progress data choice), CPU only.

use super::*;
use crate::shards::ShardWriter;
use std::io::Write;
use std::path::PathBuf;

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ava_lpmix_{tag}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Docs as (source, len); tokens encode the source (100 · source + position % 100) so a window shows where it came
/// from. Shards of `per_shard` tokens. Writes train/ shards and train/tags.tsv, returns the train dir.
fn build(tag: &str, docs: &[(u16, usize)], per_shard: usize) -> PathBuf {
    let d = tmp(tag);
    let train = d.join("train");
    let mut w = ShardWriter::new(&train, 4096, per_shard).unwrap();
    let mut tags = String::from("# v42 tags, one row per document in written order: start\tlen\tsource\torigin\tlevel\tper_mille\n# sources: 0=web(web) 1=wiki(wiki) 2=chat(dialogue)\n");
    let mut start = 0u64;
    for &(s, n) in docs {
        let toks: Vec<u32> = (0..n).map(|i| 100 * s as u32 + (i % 100) as u32).collect();
        w.push_doc(&toks).unwrap();
        tags.push_str(&format!("{start}\t{n}\t{s}\tweb\t0\t0\n"));
        start += n as u64;
    }
    w.finish().unwrap();
    std::fs::File::create(train.join("tags.tsv")).unwrap().write_all(tags.as_bytes()).unwrap();
    train
}

/// Long web pages, mid-length wiki articles, very short chat lines in between.
fn docs_mix() -> Vec<(u16, usize)> {
    let mut v = Vec::new();
    for i in 0..60 {
        v.push((0, 300 + (i * 37) % 200));
        v.push((2, 7));
        v.push((2, 9));
        if i % 2 == 0 {
            v.push((1, 150 + (i * 13) % 90));
        }
    }
    v
}

fn index(train: &Path, ctx: usize) -> (ShardSet, SourceIndex) {
    let set = ShardSet::open(train).unwrap();
    let (names, docs) = read_tags(&train.join("tags.tsv")).unwrap();
    assert_eq!(names, ["web", "wiki", "chat"]);
    let ix = SourceIndex::build(&set, &names, &docs, ctx);
    (set, ix)
}

fn probed(m: &LpMix) -> Vec<usize> {
    (0..m.base.len()).filter(|&c| !m.probes[c].is_empty()).collect()
}

#[test]
fn every_slot_has_one_class_and_the_class_holds_most_of_it() {
    let train = build("classes", &docs_mix(), 5000);
    let (set, ix) = index(&train, 32);
    // the same slots as the shard sampler: (n - 1) / ctx per shard
    let want: u64 = set.shards.iter().map(|s| (s.header.n_tokens - 1) / 32).sum();
    assert_eq!(ix.total() as u64, want);
    assert_eq!(ix.names.last().unwrap(), "mixed");
    let mut r = WindowReader::open(&set).unwrap();
    for (c, ws) in ix.windows.iter().enumerate() {
        for &s in ws {
            let w = r.read(s, 33).unwrap();
            let mut n = [0usize; 3];
            for t in &w {
                n[(*t / 100) as usize] += 1;
            }
            let (best, &most) = n.iter().enumerate().max_by_key(|x| *x.1).unwrap();
            if c < 3 {
                assert_eq!(c, best);
                assert!(2 * most > 33, "class {c} window {s}: {n:?}");
            } else {
                assert!(2 * most <= 33, "mixed window {s}: {n:?}");
            }
        }
    }
    // web and wiki windows exist; the short chat lines never hold a window's majority
    assert!(!ix.windows[0].is_empty() && !ix.windows[1].is_empty() && ix.windows[2].is_empty());
    println!("B6 classes: {:?} windows, purity {:?}", ix.windows.iter().map(|w| w.len()).collect::<Vec<_>>(), ix.purity);
}

#[test]
fn the_reader_reads_what_the_shards_hold_across_shard_files() {
    let train = build("reader", &docs_mix(), 1000);
    let (set, ix) = index(&train, 16);
    assert!(set.len() > 10);
    let mut r = WindowReader::open(&set).unwrap();
    let mut off = 0u64;
    for (j, sh) in set.shards.iter().enumerate() {
        let toks = set.load_shard(j).unwrap();
        for k in [0usize, 3, (sh.header.n_tokens as usize - 1) / 16 - 1] {
            assert_eq!(r.read(off + (k * 16) as u64, 17).unwrap(), toks.window(k * 16, 17));
        }
        off += sh.header.n_tokens;
    }
    // no window crosses a shard file
    assert!(r.read(off - 5, 17).is_err());
    assert!(ix.total() > 0);
}

#[test]
fn with_no_tilt_the_draw_is_the_natural_mix_and_each_class_is_a_shuffled_pass() {
    let train = build("natural", &docs_mix(), 5000);
    let (_, ix) = index(&train, 32);
    let mut m = LpMix::new(&ix, LpCfg::parse("0,250,4,1").unwrap(), 11);
    for (c, p) in m.probes.iter().enumerate() {
        assert_eq!(p.len(), if ix.windows[c].len() >= 100 { 1 } else { 0 });
    }
    // one pass through a class draws each of its training windows once
    let c0 = probed(&m)[0];
    let n0 = ix.windows[c0].len() - 1;
    let mut seen = std::collections::HashSet::new();
    let mut counts = vec![0usize; ix.windows.len()];
    let steps = 400u64;
    for s in 0..steps {
        for (c, st) in m.draw(s, 16) {
            counts[c] += 1;
            assert!(!m.probes[c].contains(&st), "a probe window was drawn for training");
            if c == c0 && seen.len() < n0 {
                assert!(seen.insert(st), "a window came twice in one pass");
            }
        }
    }
    assert_eq!(seen.len(), n0, "the pass through class {c0} was not finished in the test");
    let total = (steps * 16) as f64;
    for (c, &k) in counts.iter().enumerate() {
        let want = m.base[c] * total;
        assert!((k as f64 - want).abs() < 4.0 * want.sqrt() + 1.0, "class {c}: {k} vs {want:.0}");
    }
    // eps 0 never moves the mix, whatever the probes say
    let pr = probed(&m);
    for l in [5.0, 4.0, 2.0, 1.0, 0.5] {
        m.observe(&pr.iter().map(|&c| (c, if c == c0 { l } else { 4.0 })).collect::<Vec<_>>());
    }
    assert_eq!(m.p, m.base);
}

#[test]
fn the_tilt_follows_progress_within_the_floor_and_the_cap() {
    let train = build("tilt", &docs_mix(), 5000);
    let (_, ix) = index(&train, 32);
    let cfg = LpCfg::parse("0.5,100,2,1").unwrap();
    let mut m = LpMix::new(&ix, cfg, 3);
    let pr = probed(&m);
    assert!(pr.len() >= 2, "{pr:?}");
    let (fast, slow) = (pr[0], pr[1]);
    // three probes: not enough history, the natural mix
    for l in [5.0, 4.0, 3.0] {
        m.observe(&pr.iter().map(|&c| (c, if c == fast { l } else { 4.0 })).collect::<Vec<_>>());
        assert_eq!(m.p, m.base);
    }
    m.observe(&pr.iter().map(|&c| (c, if c == fast { 2.0 } else { 4.0 })).collect::<Vec<_>>());
    assert!(m.p[fast] > m.base[fast] && m.p[slow] < m.base[slow], "{:?} vs {:?}", m.p, m.base);
    for &c in &pr {
        assert!(m.p[c] >= (1.0 - cfg.eps) * m.base[c] - 1e-12, "floor");
        assert!(m.p[c] <= (1.0 - cfg.eps) * m.base[c] + cfg.eps * cfg.cap * m.base[c] + 1e-12, "cap");
    }
    assert!((m.p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    // a class being forgotten (its loss rising) is visited more too: progress is absolute
    let mut f = LpMix::new(&ix, cfg, 3);
    for l in [3.0, 3.0, 4.0, 4.0] {
        f.observe(&pr.iter().map(|&c| (c, if c == slow { l } else { 4.0 })).collect::<Vec<_>>());
    }
    assert!(f.p[slow] > f.base[slow]);
}

#[test]
fn a_resume_from_the_state_line_draws_exactly_the_same() {
    let train = build("resume", &docs_mix(), 5000);
    let (_, ix) = index(&train, 32);
    let cfg = LpCfg::parse("0.3,50,4,1").unwrap();
    let mut a = LpMix::new(&ix, cfg, 9);
    let pr = probed(&a);
    for s in 0..300u64 {
        a.draw(s, 8);
        if s % 50 == 49 {
            a.observe(&pr.iter().map(|&c| (c, 5.0 / (1.0 + s as f64 * (c + 1) as f64 * 1e-3))).collect::<Vec<_>>());
        }
    }
    let line = a.state_line();
    let mut b = LpMix::new(&ix, cfg, 9);
    b.restore(&line).unwrap();
    assert_eq!(a.p, b.p);
    for s in 300..400u64 {
        assert_eq!(a.draw(s, 8), b.draw(s, 8), "step {s}");
    }
    assert!(b.restore("ava-lpmix v1 cursors=0:0 hist=").is_err());
    assert!(LpCfg::parse("0.95").is_err() && LpCfg::parse("0.2,0").is_err() && LpCfg::parse("0.2,10,0.5").is_err());
}

/// B6b: with END the tilt fades linearly with training and is the natural mix exactly from END on; END 0 keeps it.
#[test]
fn the_curriculum_fades_the_tilt_to_the_natural_mix() {
    let train = build("fade", &docs_mix(), 5000);
    let (_, ix) = index(&train, 32);
    let cfg = LpCfg::parse("0.4,100,2,1,0.5").unwrap();
    assert_eq!(cfg.end, 0.5);
    assert_eq!(LpCfg::parse("0.4").unwrap().end, 0.0);
    assert!(LpCfg::parse("0.4,100,2,1,1.5").is_err());
    let mut m = LpMix::new(&ix, cfg, 3);
    let pr = probed(&m);
    let (fast, _) = (pr[0], pr[1]);
    for (f, e) in [(0.0, 0.4), (0.25, 0.2), (0.5, 0.0), (0.9, 0.0)] {
        m.set_frac(f);
        assert!((m.eps() - e).abs() < 1e-12, "frac {f}: eps {}", m.eps());
    }
    let feed = |m: &mut LpMix, f: f64| {
        m.set_frac(f);
        for l in [5.0, 4.0, 3.0, 2.0] {
            m.observe(&pr.iter().map(|&c| (c, if c == fast { l } else { 4.0 })).collect::<Vec<_>>());
        }
    };
    feed(&mut m, 0.1);
    assert!(m.p[fast] > m.base[fast], "tilted early");
    let early = m.p[fast];
    feed(&mut m, 0.3);
    assert!(m.p[fast] > m.base[fast] && m.p[fast] < early, "fading");
    feed(&mut m, 0.5);
    assert_eq!(m.p, m.base, "the natural mix exactly from END on");
    let mut k = LpMix::new(&ix, LpCfg::parse("0.4,100,2,1").unwrap(), 3);
    k.set_frac(0.9);
    assert_eq!(k.eps(), 0.4, "END 0: no fade");
}
