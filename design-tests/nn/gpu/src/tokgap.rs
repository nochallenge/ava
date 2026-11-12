// TESTING / DESIGN ONLY — tokgap (v42 T4): where does she lose to the transformer yardstick T?
//
// Both models score the same validation windows; every target token gets its NLL under each (probe::token_nll), and
// the gap (Ava − T) is split by token type before any fix is chosen (Zoology 2023: 82% of a gated-convolution model's
// gap to attention was associative recall). Four splits, each a partition of the targets, so within a split the shares
// of the total gap add up to 1:
// - recall: `recall` when the bigram (previous token, target) already occurred earlier in the window (an associative-
//   recall hit: the target could be copied from the context, Zoology's definition); `repeat` when only the target
//   itself occurred before; `new` otherwise.
// - freq: target types sorted by how often they occur among all scored targets (ties by id); the commonest types that
//   make up the first half of the occurrences are `common`, the next 40% `mid`, the last 10% `rare`.
// - kind: `eos`; `number` (a digit or a length mark); `punct` (no letter or digit in its bytes); `word-start` (its
//   bytes start with a space or a newline, then a letter or digit); `continuation` (the rest: the inside of a word).
// - pos: the target's position in the window (tokens read before it − 1): 0-15, 16-63, 64-255, 256-1023, 1024+.
// For each class: targets, their share of all targets, Ava's and T's mean NLL (nats per token), the mean gap, and the
// class's share of the total gap (sum of the class's gaps / sum of all gaps; negative = she is ahead there).

use ava_data::bpe::{Bpe, EOS};
use ava_data::numbers::{is_digit, is_mark};
use std::collections::HashSet;

pub const SPLITS: [&str; 4] = ["recall", "freq", "kind", "pos"];
pub const CLASSES: [&[&str]; 4] = [
    &["recall", "repeat", "new"],
    &["common", "mid", "rare"],
    &["eos", "number", "punct", "word-start", "continuation"],
    &["0-15", "16-63", "64-255", "256-1023", "1024+"],
];

/// Recall class of each target of one window (`w`: the window's tokens; target p is w[p + 1], read after w[..=p]):
/// 0 recall, 1 repeat, 2 new.
pub fn recall_classes(w: &[u32]) -> Vec<u8> {
    let mut seen_tok: HashSet<u32> = HashSet::new();
    let mut seen_bi: HashSet<(u32, u32)> = HashSet::new();
    let mut out = Vec::with_capacity(w.len().saturating_sub(1));
    for p in 0..w.len().saturating_sub(1) {
        seen_tok.insert(w[p]);
        if p >= 1 {
            seen_bi.insert((w[p - 1], w[p]));
        }
        let (prev, tgt) = (w[p], w[p + 1]);
        out.push(if seen_bi.contains(&(prev, tgt)) {
            0
        } else if seen_tok.contains(&tgt) {
            1
        } else {
            2
        });
    }
    out
}

/// Position class of target p (0-15, 16-63, 64-255, 256-1023, 1024+).
pub fn pos_class(p: usize) -> u8 {
    match p {
        0..=15 => 0,
        16..=63 => 1,
        64..=255 => 2,
        256..=1023 => 3,
        _ => 4,
    }
}

/// Kind class of every id below `vocab` from the tokenizer's bytes (ids it does not have, other than marks, are
/// `continuation`).
pub fn kind_table(bpe: &Bpe, vocab: usize) -> Vec<u8> {
    (0..vocab as u32)
        .map(|id| {
            if id == EOS {
                return 0;
            }
            if is_digit(id) || is_mark(id) {
                return 1;
            }
            if (id as usize) >= bpe.vocab_size() {
                return 4;
            }
            let b = bpe.token_bytes(id);
            let wordy = |c: &u8| c.is_ascii_alphanumeric() || *c >= 0x80;
            if !b.iter().any(wordy) {
                2
            } else if matches!(b.first(), Some(b' ' | b'\n' | b'\t')) && b.get(1).is_some_and(wordy) {
                3
            } else {
                4
            }
        })
        .collect()
}

/// Frequency class (0 common, 1 mid, 2 rare) of every id from its count among the scored targets (see the top).
pub fn freq_table(counts: &[u64]) -> Vec<u8> {
    let total: u64 = counts.iter().sum();
    let mut ids: Vec<usize> = (0..counts.len()).filter(|&i| counts[i] > 0).collect();
    ids.sort_by(|&a, &b| counts[b].cmp(&counts[a]).then(a.cmp(&b)));
    let mut out = vec![2u8; counts.len()];
    let mut before = 0u64; // occurrences of the commoner types
    for i in ids {
        let f = before as f64 / total.max(1) as f64;
        out[i] = if f < 0.5 {
            0
        } else if f < 0.9 {
            1
        } else {
            2
        };
        before += counts[i];
    }
    out
}

/// One scored target.
#[derive(Clone, Copy, Debug)]
pub struct Scored {
    pub id: u32,
    pub recall: u8,
    pub pos: u8,
    /// NLL under Ava and under T (nats)
    pub a: f32,
    pub t: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cell {
    pub n: u64,
    pub a: f64,
    pub t: f64,
}

impl Cell {
    fn add(&mut self, s: &Scored) {
        self.n += 1;
        self.a += s.a as f64;
        self.t += s.t as f64;
    }
    pub fn gap(&self) -> f64 {
        self.a - self.t
    }
}

/// The gap table: cells[split][class], and the total.
#[derive(Clone, Debug)]
pub struct GapTable {
    pub cells: Vec<Vec<Cell>>,
    pub total: Cell,
}

/// The windows' targets (`recall_classes` and `pos_class` per window), with each model's per-target NLL.
pub fn score_windows(windows: &[Vec<u32>], a_nll: &[Vec<f32>], t_nll: &[Vec<f32>]) -> Result<Vec<Scored>, String> {
    let mut out = Vec::new();
    for (k, w) in windows.iter().enumerate() {
        let (a, t) = (&a_nll[k], &t_nll[k]);
        let n = w.len().saturating_sub(1);
        if a.len() != n || t.len() != n {
            return Err(format!("window {k}: {n} targets but {} and {} scores", a.len(), t.len()));
        }
        for (p, r) in recall_classes(w).into_iter().enumerate() {
            out.push(Scored { id: w[p + 1], recall: r, pos: pos_class(p), a: a[p], t: t[p] });
        }
    }
    Ok(out)
}

/// Splits the scored targets (`kinds` from kind_table; the frequency classes are counted here).
pub fn tabulate(scored: &[Scored], kinds: &[u8]) -> GapTable {
    let mut counts = vec![0u64; kinds.len()];
    for s in scored {
        counts[s.id as usize] += 1;
    }
    let freq = freq_table(&counts);
    let mut cells: Vec<Vec<Cell>> = CLASSES.iter().map(|c| vec![Cell::default(); c.len()]).collect();
    let mut total = Cell::default();
    for s in scored {
        let i = s.id as usize;
        for (split, class) in [s.recall, freq[i], kinds[i], s.pos].into_iter().enumerate() {
            cells[split][class as usize].add(s);
        }
        total.add(s);
    }
    GapTable { cells, total }
}

/// The table as TSV: split, class, targets, share of targets, Ava and T nats per token, gap per token, share of the gap.
pub fn report(g: &GapTable) -> String {
    let mut s = String::from("split\tclass\ttargets\tshare_targets\tava_nats\tt_nats\tgap_nats\tshare_gap\n");
    let tot_gap = g.total.gap();
    let per = |c: &Cell, x: f64| if c.n == 0 { f64::NAN } else { x / c.n as f64 };
    for (si, split) in SPLITS.iter().enumerate() {
        for (ci, class) in CLASSES[si].iter().enumerate() {
            let c = &g.cells[si][ci];
            if c.n == 0 {
                s += &format!("{split}\t{class}\t0\t0.0000\t-\t-\t-\t+0.0000\n"); // e.g. positions past the window
                continue;
            }
            s += &format!(
                "{split}\t{class}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:+.4}\t{:+.4}\n",
                c.n,
                c.n as f64 / g.total.n.max(1) as f64,
                per(c, c.a),
                per(c, c.t),
                per(c, c.gap()),
                if tot_gap != 0.0 { c.gap() / tot_gap } else { f64::NAN }
            );
        }
    }
    s += &format!("all\tall\t{}\t1.0000\t{:.4}\t{:.4}\t{:+.4}\t{:+.4}\n", g.total.n, per(&g.total, g.total.a), per(&g.total, g.total.t), per(&g.total, tot_gap), 1.0);
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Kind;
    use crate::testkit::{byte_bpe, tiny};
    use candle_core::Device;

    #[test]
    fn recall_repeat_and_new() {
        // a b c a b x b
        let w = [1, 2, 3, 1, 2, 9, 2];
        // targets: b(new) c(new) a(repeat) b(recall: "a b" seen) x(new) b(repeat: "x b" never seen, b seen)
        assert_eq!(recall_classes(&w), vec![2, 2, 1, 0, 2, 1]);
        assert_eq!(recall_classes(&[5]), Vec::<u8>::new());
        assert_eq!((pos_class(0), pos_class(15), pos_class(16), pos_class(255), pos_class(256), pos_class(5000)), (0, 0, 1, 2, 3, 4));
    }

    #[test]
    fn kinds_and_frequency_classes() {
        let bpe = byte_bpe();
        let k = kind_table(&bpe, 258);
        assert_eq!(k[EOS as usize], 0);
        assert_eq!(k[b'7' as usize], 1);
        assert_eq!(k[b',' as usize], 2);
        assert_eq!(k[b' ' as usize], 2);
        assert_eq!(k[b'a' as usize], 4); // a lone byte is never a word start: no merged " a" token here
        // counts: id 0 is 60% of the targets, id 1 30%, ids 2 and 3 5% each
        let f = freq_table(&[60, 30, 5, 5, 0]);
        assert_eq!(f, vec![0, 1, 2, 2, 2]);
    }

    #[test]
    fn the_gap_shares_add_up_within_each_split() {
        let windows = vec![vec![1u32, 2, 3, 1, 2, 9, 2], vec![256u32, 7, 7, 7, 44, 32]];
        let a: Vec<Vec<f32>> = vec![vec![3.0, 2.0, 1.0, 0.5, 4.0, 1.0], vec![2.0, 1.0, 1.0, 3.0, 2.0]];
        let t: Vec<Vec<f32>> = vec![vec![2.5, 2.0, 1.2, 0.1, 3.0, 1.0], vec![2.0, 1.5, 0.5, 2.0, 2.0]];
        let sc = score_windows(&windows, &a, &t).unwrap();
        assert_eq!(sc.len(), 11);
        let g = tabulate(&sc, &kind_table(&byte_bpe(), 258));
        let tot = g.total.gap();
        assert!((tot - (20.5 - 17.8)).abs() < 1e-5, "{tot}");
        for (si, cells) in g.cells.iter().enumerate() {
            let n: u64 = cells.iter().map(|c| c.n).sum();
            assert_eq!(n, g.total.n, "split {}", SPLITS[si]);
            let share: f64 = cells.iter().map(|c| c.gap() / tot).sum();
            assert!((share - 1.0).abs() < 1e-9, "split {}: {share}", SPLITS[si]);
        }
        // the two recall hits ("a b ... a b" and "7 7 7") had gaps of 0.4 and 0.5
        assert_eq!(g.cells[0][0].n, 2);
        assert!((g.cells[0][0].gap() - 0.9).abs() < 1e-6);
        let r = report(&g);
        assert_eq!(r.lines().count(), 1 + 16 + 1);
        assert!(!r.contains("NaN"), "an empty class prints dashes:\n{r}");
        assert!(score_windows(&windows[..1], &[vec![1.0]], &t[..1]).is_err());
    }

    #[test]
    fn two_real_models_on_the_same_windows() {
        let (_, _, ava) = tiny(Kind::D, 5);
        let (_, _, yard) = tiny(Kind::T, 6);
        let dev = Device::Cpu;
        let text = b"the cat and the cat and 42 dogs. the cat sat.";
        let mut toks: Vec<u32> = vec![EOS];
        toks.extend(text.iter().map(|&b| b as u32));
        let windows: Vec<Vec<u32>> = toks.chunks(17).filter(|c| c.len() == 17).map(|c| c.to_vec()).collect();
        let from = vec![1; windows.len()];
        let a = crate::probe::token_nll(&ava, &dev, &windows, &from, 1 << 16).unwrap();
        let t = crate::probe::token_nll(&yard, &dev, &windows, &from, 1 << 16).unwrap();
        let g = tabulate(&score_windows(&windows, &a, &t).unwrap(), &kind_table(&byte_bpe(), 258));
        assert_eq!(g.total.n, windows.len() as u64 * 16);
        // each model's total equals its own sum of per-token scores
        let sa: f64 = a.iter().flatten().map(|&x| x as f64).sum();
        assert!((g.total.a - sa).abs() < 1e-6 * sa.max(1.0));
        assert!(g.cells[0][0].n > 0, "the repeated phrase gives recall hits");
    }
}
