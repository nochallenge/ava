//! TESTING / DESIGN ONLY — integration tests for ava_data::shards (files on disk).
//!
//! Every test writes into its own directory under cargo's per-crate test temp dir
//! (`CARGO_TARGET_TMPDIR`, i.e. inside this crate's `target/`), never a system temp folder.

use ava_data::bpe::PAD;
use ava_data::shards::*;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

fn tmp(name: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("shards-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d); // only this test's own scratch dir
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write_docs(dir: &PathBuf, vocab: usize, max_per_shard: usize, docs: &[Vec<u32>]) -> ShardSet {
    let mut w = ShardWriter::new(dir, vocab, max_per_shard).unwrap();
    for d in docs {
        w.push_doc(d).unwrap();
    }
    w.finish().unwrap();
    ShardSet::open(dir).unwrap()
}

/// Shards whose tokens are their own global positions (vocab 2^20 -> u32 width), so a
/// window's first token says exactly where it started.
fn positional_set(name: &str, sizes: &[usize]) -> (ShardSet, Vec<usize>) {
    // Shards of unequal sizes: write each with its own writer, then move it into place.
    let dir = tmp(name);
    let total: usize = sizes.iter().sum();
    let mut start = 0usize;
    let mut starts = vec![];
    for (i, &n) in sizes.iter().enumerate() {
        let sub = dir.join(format!("w{i}"));
        let mut w = ShardWriter::new(&sub, 1 << 20, n.max(1)).unwrap();
        let doc: Vec<u32> = (start..start + n).map(|x| x as u32).collect();
        w.push_doc(&doc).unwrap();
        w.finish().unwrap();
        if n > 0 {
            std::fs::rename(sub.join(shard_name(0)), dir.join(shard_name(i))).unwrap();
        }
        starts.push(start);
        start += n;
    }
    assert_eq!(start, total);
    (ShardSet::open(&dir).unwrap(), starts)
}

#[test]
fn header_roundtrip_and_layout() {
    let h = ShardHeader::new(4096, 123_456_789_012);
    assert_eq!(h.width, 2);
    let b = h.to_bytes();
    assert_eq!(b.len(), HEADER_LEN);
    assert_eq!(&b[0..8], MAGIC);
    assert_eq!(u32::from_le_bytes(b[8..12].try_into().unwrap()), VERSION);
    assert_eq!(u32::from_le_bytes(b[12..16].try_into().unwrap()), 4096);
    assert_eq!(u64::from_le_bytes(b[16..24].try_into().unwrap()), 123_456_789_012);
    assert_eq!(b[24], 2);
    assert!(b[25..32].iter().all(|&x| x == 0));
    assert_eq!(ShardHeader::from_bytes(&b).unwrap(), h);

    let h4 = ShardHeader::new(70_000, 5);
    assert_eq!(h4.width, 4);
    assert_eq!(ShardHeader::from_bytes(&h4.to_bytes()).unwrap(), h4);

    let mut bad = b;
    bad[0] = b'X';
    assert!(ShardHeader::from_bytes(&bad).is_err());
    let mut bad = b;
    bad[8] = 2; // version 2
    assert!(ShardHeader::from_bytes(&bad).is_err());
    let mut bad = b;
    bad[24] = 4; // width does not match the vocab rule
    assert!(ShardHeader::from_bytes(&bad).is_err());
    let mut bad = b;
    bad[30] = 1; // padding must be zero
    assert!(ShardHeader::from_bytes(&bad).is_err());
}

#[test]
fn width_rule() {
    assert_eq!(width_for_vocab(258), 2);
    assert_eq!(width_for_vocab(4096), 2);
    assert_eq!(width_for_vocab(65_536), 2);
    assert_eq!(width_for_vocab(65_537), 4);
    assert_eq!(width_for_vocab(1 << 20), 4);
}

#[test]
fn u16_and_u32_roundtrip() {
    for (name, vocab, width) in [("u16", 4096usize, 2u8), ("u32", 100_000, 4)] {
        let dir = tmp(name);
        let docs: Vec<Vec<u32>> = (0..20)
            .map(|d| (0..(d * 7 % 13 + 1)).map(|i| ((d * 1000 + i * 37) % vocab) as u32).collect())
            .collect();
        let set = write_docs(&dir, vocab, 1_000_000, &docs);
        assert_eq!(set.len(), 1);
        assert_eq!(set.vocab_size as usize, vocab);
        assert_eq!(set.shards[0].header.width, width);
        let all: Vec<u32> = docs.concat();
        assert_eq!(set.total_tokens, all.len() as u64);
        assert_eq!(set.load_shard(0).unwrap().to_vec(), all);
        let len = std::fs::metadata(&set.shards[0].path).unwrap().len();
        assert_eq!(len, HEADER_LEN as u64 + all.len() as u64 * width as u64);
    }
    // the largest ids survive
    let dir = tmp("maxid");
    let set = write_docs(&dir, 65_536, 100, &[vec![65_535, 0, 65_535]]);
    assert_eq!(set.load_shard(0).unwrap().to_vec(), vec![65_535, 0, 65_535]);
    let dir = tmp("maxid32");
    let set = write_docs(&dir, 1 << 24, 100, &[vec![(1 << 24) - 1, 7]]);
    assert_eq!(set.load_shard(0).unwrap().to_vec(), vec![(1 << 24) - 1, 7]);
}

#[test]
fn writer_splits_into_shards_at_the_limit() {
    let dir = tmp("split");
    let docs: Vec<Vec<u32>> = vec![(0..4).collect(), (4..11).collect(), (11..20).collect(), (20..25).collect()];
    let set = write_docs(&dir, 4096, 10, &docs);
    assert_eq!(set.len(), 3);
    let sizes: Vec<u64> = set.shards.iter().map(|s| s.header.n_tokens).collect();
    assert_eq!(sizes, vec![10, 10, 5]);
    let names: Vec<String> = set.shards.iter().map(|s| s.path.file_name().unwrap().to_string_lossy().into_owned()).collect();
    assert_eq!(names, vec!["shard_00000.tok", "shard_00001.tok", "shard_00002.tok"]);
    let mut all = vec![];
    for i in 0..set.len() {
        all.extend(set.load_shard(i).unwrap().to_vec());
    }
    assert_eq!(all, (0..25).collect::<Vec<u32>>());
    // no temp files left behind
    for e in std::fs::read_dir(&dir).unwrap() {
        let n = e.unwrap().file_name().to_string_lossy().into_owned();
        assert!(n.ends_with(".tok"), "leftover file {n}");
    }
    // exact multiple: no empty trailing shard
    let dir = tmp("split-exact");
    let set = write_docs(&dir, 4096, 5, &[(0..10).collect()]);
    assert_eq!(set.len(), 2);
    // stats
    let dir = tmp("split-stats");
    let mut w = ShardWriter::new(&dir, 4096, 10).unwrap();
    for d in &docs {
        w.push_doc(d).unwrap();
    }
    let st = w.finish().unwrap();
    assert_eq!(st, WriterStats { shards: 3, tokens: 25, docs: 4 });
}

#[test]
fn writer_rejects_bad_input() {
    let dir = tmp("reject");
    let mut w = ShardWriter::new(&dir, 300, 10).unwrap();
    assert!(w.push_doc(&[1, 2, 300]).is_err(), "token id >= vocab must be rejected");
    drop(w);
    let dir = tmp("reject-existing");
    write_docs(&dir, 300, 10, &[vec![1, 2, 3]]);
    assert!(ShardWriter::new(&dir, 300, 10).is_err(), "must refuse to mix with existing shards");
    let dir = tmp("reject-zero");
    assert!(ShardWriter::new(&dir, 300, 0).is_err());
}

#[test]
fn open_rejects_corrupt_sets() {
    let dir = tmp("corrupt-trunc");
    let set = write_docs(&dir, 4096, 100, &[(0..50).collect()]);
    let p = set.shards[0].path.clone();
    let bytes = std::fs::read(&p).unwrap(); // a 132-byte test shard
    std::fs::write(&p, &bytes[..bytes.len() - 1]).unwrap();
    assert!(ShardSet::open(&dir).is_err(), "truncated shard must be rejected");

    let dir = tmp("corrupt-magic");
    let set = write_docs(&dir, 4096, 100, &[(0..50).collect()]);
    let p = set.shards[0].path.clone();
    let mut bytes = std::fs::read(&p).unwrap();
    bytes[0] = b'Z';
    std::fs::write(&p, &bytes).unwrap();
    assert!(ShardSet::open(&dir).is_err());

    let dir = tmp("corrupt-mixed");
    write_docs(&dir, 4096, 100, &[(0..50).collect()]);
    let other = tmp("corrupt-mixed-other");
    let s2 = write_docs(&other, 5000, 100, &[(0..50).collect()]);
    std::fs::copy(&s2.shards[0].path, dir.join(shard_name(1))).unwrap();
    assert!(ShardSet::open(&dir).is_err(), "mixed vocab sizes must be rejected");

    assert!(ShardSet::open(tmp("missing").join("nope")).is_err());
}

#[test]
fn open_sorts_numerically_and_ignores_other_files() {
    let dir = tmp("sorting");
    let set = write_docs(&dir, 4096, 3, &[(0..9).collect()]);
    assert_eq!(set.len(), 3);
    std::fs::rename(dir.join(shard_name(2)), dir.join("shard_100000.tok")).unwrap();
    std::fs::write(dir.join("notes.txt"), b"x").unwrap();
    std::fs::write(dir.join("shard_00009.tok.tmp"), b"x").unwrap();
    let set = ShardSet::open(&dir).unwrap();
    let names: Vec<String> = set.shards.iter().map(|s| s.path.file_name().unwrap().to_string_lossy().into_owned()).collect();
    assert_eq!(names, vec!["shard_00000.tok", "shard_00001.tok", "shard_100000.tok"]);
}

#[test]
fn train_val_split_is_deterministic() {
    let mut n_val = 0;
    for i in 0..200_000u64 {
        if is_val_doc(DEFAULT_SPLIT_SEED, i, 5) {
            n_val += 1;
        }
    }
    assert!((850..1150).contains(&n_val), "0.5% of 200k should be ~1000, got {n_val}");
    assert!(!(0..1000).any(|i| is_val_doc(1, i, 0)), "val_per_mille 0 -> no val docs");
    assert!((0..1000).all(|i| is_val_doc(1, i, 1000)), "val_per_mille 1000 -> all val docs");
    let a: Vec<bool> = (0..5000).map(|i| is_val_doc(7, i, 100)).collect();
    let b: Vec<bool> = (0..5000).map(|i| is_val_doc(7, i, 100)).collect();
    let c: Vec<bool> = (0..5000).map(|i| is_val_doc(8, i, 100)).collect();
    assert_eq!(a, b);
    assert_ne!(a, c);

    // make_shards: the same docs + seed give byte-identical output, and the split follows the rule
    let docs: Vec<Vec<u32>> = (0..3000u32).map(|d| vec![d % 4000, (d * 7) % 4000, 256]).collect();
    let d1 = tmp("mk1");
    let d2 = tmp("mk2");
    let s1 = make_shards(&d1, 4096, 1000, 50, 42, docs.clone()).unwrap();
    let s2 = make_shards(&d2, 4096, 1000, 50, 42, docs.clone()).unwrap();
    assert_eq!(s1, s2);
    assert_eq!(s1.train.docs + s1.val.docs, 3000);
    assert_eq!(s1.train.tokens + s1.val.tokens, 9000);
    for sub in ["train", "val"] {
        let a = ShardSet::open(d1.join(sub)).unwrap();
        let b = ShardSet::open(d2.join(sub)).unwrap();
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert_eq!(std::fs::read(&a.shards[i].path).unwrap(), std::fs::read(&b.shards[i].path).unwrap());
        }
    }
    let val = ShardSet::open(d1.join("val")).unwrap();
    let mut val_tokens = vec![];
    for i in 0..val.len() {
        val_tokens.extend(val.load_shard(i).unwrap().to_vec());
    }
    let want: Vec<u32> = docs.iter().enumerate().filter(|(i, _)| is_val_doc(42, *i as u64, 50)).flat_map(|(_, d)| d.clone()).collect();
    assert_eq!(val_tokens, want);
    assert!(s1.val.docs > 100 && s1.val.docs < 200, "5% of 3000 ~ 150, got {}", s1.val.docs);
}

#[test]
fn sampler_is_deterministic_and_seed_dependent() {
    let (set, _) = positional_set("det", &[40, 33, 57]);
    let ctx = 6;
    let run = |seed: u64| -> Vec<Vec<u32>> {
        let mut s = Sampler::new(&set, ctx, seed).unwrap();
        (0..60).map(|_| s.next_window()).collect()
    };
    let a = run(1);
    assert_eq!(a, run(1));
    assert_ne!(a, run(2));
    for w in &a {
        assert_eq!(w.len(), ctx + 1);
        for k in 1..w.len() {
            assert_eq!(w[k], w[0] + k as u32, "window must be consecutive tokens");
        }
    }
}

#[test]
fn every_slot_visited_once_per_epoch() {
    let sizes = [50usize, 23, 5, 41, 9, 8];
    let ctx = 8;
    let (set, starts) = positional_set("slots", &sizes);
    // slots: (n - 1) / ctx for n > ctx, windows start at shard_start + k * ctx
    let mut expected: HashSet<u32> = HashSet::new();
    for (i, &n) in sizes.iter().enumerate() {
        if n > ctx {
            for k in 0..(n - 1) / ctx {
                expected.insert((starts[i] + k * ctx) as u32);
            }
        }
    }
    let mut s = Sampler::new(&set, ctx, 99).unwrap();
    assert_eq!(s.windows_per_epoch(), expected.len() as u64);
    let shard_of = |tok: u32| -> usize { starts.iter().rposition(|&st| st as u32 <= tok).unwrap() };
    let mut orders = vec![];
    for epoch in 0..3u64 {
        assert_eq!(s.epoch(), epoch);
        let mut seen: HashMap<u32, usize> = HashMap::new();
        let mut shard_seq: Vec<usize> = vec![];
        let mut order = vec![];
        for _ in 0..expected.len() {
            let w = s.next_window();
            *seen.entry(w[0]).or_default() += 1;
            let sh = shard_of(w[0]);
            assert!(shard_of(w[ctx]) == sh, "window crosses a shard boundary");
            assert_eq!(s.last_position(), Some((sh, w[0] as usize - starts[sh])));
            if shard_seq.last() != Some(&sh) {
                assert!(!shard_seq.contains(&sh), "shard {sh} revisited within an epoch");
                shard_seq.push(sh);
            }
            order.push(w[0]);
        }
        assert_eq!(seen.len(), expected.len());
        assert!(seen.values().all(|&c| c == 1));
        assert_eq!(seen.keys().copied().collect::<HashSet<u32>>(), expected);
        orders.push(order);
    }
    assert_eq!(s.epoch(), 3);
    assert_ne!(orders[0], orders[1], "each epoch reshuffles");
    // windows inside a shard are not in file order
    let first: Vec<u32> = orders[0].iter().copied().filter(|&t| shard_of(t) == 0).collect();
    let mut sorted = first.clone();
    sorted.sort();
    assert_ne!(first, sorted, "slots within a shard should be shuffled");
}

#[test]
fn resume_reproduces_the_stream_exactly() {
    let (set, _) = positional_set("resume", &[30, 17, 4, 26]);
    let ctx = 5;
    let seed = 2024;
    let wpe = Sampler::new(&set, ctx, seed).unwrap().windows_per_epoch() as usize;
    assert!(wpe > 8);
    // every step count over 2.5 epochs: covers every shard boundary and two epoch boundaries
    for steps in 0..(wpe * 5 / 2) {
        let mut a = Sampler::new(&set, ctx, seed).unwrap();
        for _ in 0..steps {
            a.next_window();
        }
        let line = a.cursor().to_line();
        assert!(!line.contains('\n'));
        let cur = Cursor::parse(&line).unwrap();
        assert_eq!(cur, a.cursor());
        let mut b = Sampler::resume(&set, cur).unwrap();
        for k in 0..(wpe + 3) {
            assert_eq!(a.next_window(), b.next_window(), "diverged after {steps} steps + {k}");
            assert_eq!(a.cursor(), b.cursor());
        }
    }
}

#[test]
fn single_shard_reshuffles_each_epoch_and_resumes() {
    // the same shard is last of epoch e and first of epoch e+1: slot order must still change
    let (set, _) = positional_set("single", &[61]);
    let ctx = 4;
    let mut s = Sampler::new(&set, ctx, 3).unwrap();
    let wpe = s.windows_per_epoch() as usize;
    assert_eq!(wpe, 15);
    let e0: Vec<u32> = (0..wpe).map(|_| s.next_window()[0]).collect();
    let cur = s.cursor();
    assert_eq!((cur.epoch, cur.shard_pos, cur.slot_pos), (1, 0, 0));
    let e1: Vec<u32> = (0..wpe).map(|_| s.next_window()[0]).collect();
    assert_ne!(e0, e1);
    let (mut a, mut b) = (e0.clone(), e1.clone());
    a.sort();
    b.sort();
    assert_eq!(a, b);
    let mut r = Sampler::resume(&set, cur).unwrap();
    let again: Vec<u32> = (0..wpe).map(|_| r.next_window()[0]).collect();
    assert_eq!(again, e1);
}

#[test]
fn cursor_line_format() {
    let c = Cursor { seed: u64::MAX, ctx_len: 1024, epoch: 3, shard_pos: 17, slot_pos: 123_456 };
    let line = c.to_line();
    assert!(line.starts_with("ava-cursor v1 "), "{line}");
    assert_eq!(Cursor::parse(&line).unwrap(), c);
    assert_eq!(Cursor::parse(&format!("  {line}\n")).unwrap(), c);
    for bad in ["", "ava-cursor v2 seed=1 ctx_len=2 epoch=0 shard_pos=0 slot_pos=0", "ava-cursor v1 seed=1", "ava-cursor v1 seed=x ctx_len=2 epoch=0 shard_pos=0 slot_pos=0"] {
        assert!(Cursor::parse(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn resume_rejects_mismatched_cursor() {
    let (set, _) = positional_set("resume-bad", &[30, 17]);
    let good = Sampler::new(&set, 5, 1).unwrap().cursor();
    assert!(Sampler::resume(&set, Cursor { shard_pos: 2, ..good }).is_err());
    assert!(Sampler::resume(&set, Cursor { ctx_len: 0, ..good }).is_err());
    assert!(Sampler::new(&set, 0, 1).is_err());
    assert!(Sampler::new(&set, 30, 1).is_err(), "no shard can hold a window of 31 tokens");
}

#[test]
fn batch_matches_next_window() {
    let (set, _) = positional_set("batch", &[30, 17]);
    let mut a = Sampler::new(&set, 4, 5).unwrap();
    let mut b = Sampler::new(&set, 4, 5).unwrap();
    let bat = a.batch(9);
    let one: Vec<Vec<u32>> = (0..9).map(|_| b.next_window()).collect();
    assert_eq!(bat, one);
}

#[test]
fn sequential_covers_every_target_once() {
    // 27 tokens over shards of 10, 10, 7; ctx 4 -> windows start at 0, 4, 8, ...
    let dir = tmp("seq");
    let set = write_docs(&dir, 1 << 20, 10, &[(0..27).collect()]);
    assert_eq!(set.len(), 3);
    let ws: Vec<Vec<u32>> = set.iter_sequential(4).collect();
    assert_eq!(ws[0], vec![0, 1, 2, 3, 4]);
    assert_eq!(ws[1], vec![4, 5, 6, 7, 8]);
    assert_eq!(ws[2], vec![8, 9, 10, 11, 12], "windows run across shard boundaries");
    assert_eq!(ws.last().unwrap(), &vec![24, 25, 26, PAD, PAD], "last partial window is PAD-padded");
    assert_eq!(ws.len(), 7);
    let mut targets: Vec<u32> = ws.iter().flat_map(|w| w[1..].iter().copied().filter(|&t| t != PAD)).collect();
    targets.sort();
    assert_eq!(targets, (1..27).collect::<Vec<u32>>(), "every token after the first is a target exactly once");

    // exactly k * ctx + 1 tokens: no padded window
    let dir = tmp("seq-exact");
    let set = write_docs(&dir, 1 << 20, 4, &[(0..9).collect()]);
    let ws: Vec<Vec<u32>> = set.iter_sequential(4).collect();
    assert_eq!(ws, vec![vec![0, 1, 2, 3, 4], vec![4, 5, 6, 7, 8]]);

    // fewer than 2 tokens: nothing to predict
    let dir = tmp("seq-tiny");
    let set = write_docs(&dir, 1 << 20, 4, &[vec![5]]);
    assert_eq!(set.iter_sequential(4).count(), 0);
}
