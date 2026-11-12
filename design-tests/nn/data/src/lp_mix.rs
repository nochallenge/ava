// TESTING / DESIGN ONLY
//! v42 B6: curiosity as data choice. Training draws more from the sources where she is improving fastest
//! (learning progress, Oudeyer and Gottlieb), and keeps most of the natural mixture.
//!
//! - **Classes.** `train/tags.tsv` (written by v42_shards) gives every document's global start, length and
//!   source. A window (ctx + 1 tokens inside one shard, at the slots `Sampler` uses) belongs to the source
//!   that holds more than half of its tokens, else to the class "mixed". The stream interleaves documents, so
//!   short-document sources (sentences, dialogue lines, restated facts) mostly live in "mixed" windows. Every
//!   window is in exactly one class, so with the tilt at 0 the draw is uniform over windows: the natural mix.
//! - **Probes.** Each class with at least 100 x PROBE windows sets PROBE of them aside (seeded; at most 1%). They are never
//!   trained on; every EVERY steps the trainer measures her loss on them and calls `observe`.
//! - **Learning progress.** For a class, |mean of its last 2 probe losses − mean of the 2 before| (absolute,
//!   so a source being forgotten is visited again too). Before 4 probes the mix is the natural one.
//! - **The mix.** p = (1 − EPS) · base + EPS · B · q over the probed classes, where base is each class's share
//!   of the training windows, B is the probed classes' base share and q follows learning progress, capped at
//!   CAP · base / B per class (water-filling). Unprobed classes keep their base share. The floor (1 − EPS) ·
//!   base is the guard against v29's WP17 failure, where choosing by progress alone starved the times table.
//! - **Draws.** Step s draws each row's class from p with its own seed (data seed, s), then the class's next
//!   window from its own shuffled order (a new order every pass through that class). `state_line` holds the
//!   cursors and the probe history; p is recomputed from the history, so a resume continues exactly.

use crate::rng::{derive_seed, SplitMix64};
use crate::shards::{ShardSet, HEADER_LEN};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom};
use std::path::Path;

const PERM_LABEL: u64 = 0xB6_0001;
const PROBE_LABEL: u64 = 0xB6_0002;
const DRAW_LABEL: u64 = 0xB6_0003;
/// How many probe losses on each side of the learning-progress difference.
pub const LP_SPAN: usize = 2;

/// One document of `tags.tsv`: global start, length, source id.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Doc {
    pub start: u64,
    pub len: u64,
    pub source: u16,
}

/// The source names (from the `# sources:` line) and the documents, in written order.
pub fn read_tags(path: &Path) -> io::Result<(Vec<String>, Vec<Doc>)> {
    let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, format!("{}: {m}", path.display()));
    let f = BufReader::with_capacity(1 << 20, File::open(path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", path.display())))?);
    let (mut names, mut docs) = (Vec::new(), Vec::new());
    for line in f.lines() {
        let line = line?;
        if let Some(rest) = line.strip_prefix("# sources:") {
            for item in rest.split_whitespace() {
                let (id, name) = item.split_once('=').ok_or_else(|| bad(format!("bad source item {item:?}")))?;
                let id: usize = id.parse().map_err(|_| bad(format!("bad source id in {item:?}")))?;
                if id != names.len() {
                    return Err(bad(format!("source ids out of order at {item:?}")));
                }
                names.push(name.split('(').next().unwrap_or(name).to_string());
            }
            continue;
        }
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let mut it = line.split('\t');
        let mut num = |what: &str| -> io::Result<u64> { it.next().and_then(|v| v.parse().ok()).ok_or_else(|| bad(format!("bad {what} in {line:?}"))) };
        let (start, len, source) = (num("start")?, num("len")?, num("source")?);
        if source as usize >= names.len() {
            return Err(bad(format!("source {source} not in the sources line")));
        }
        docs.push(Doc { start, len, source: source as u16 });
    }
    if names.is_empty() {
        return Err(bad("no `# sources:` line".into()));
    }
    Ok((names, docs))
}

/// Every training window's start, by class (the sources, then "mixed" last).
pub struct SourceIndex {
    pub names: Vec<String>,
    pub windows: Vec<Vec<u64>>,
    /// Mean share of a class's windows held by the class's own source (1 for "mixed" means nothing).
    pub purity: Vec<f64>,
    pub ctx: usize,
}

impl SourceIndex {
    /// Label every slot of `set` (start k · ctx inside each shard, ctx + 1 tokens) by its majority source.
    pub fn build(set: &ShardSet, names: &[String], docs: &[Doc], ctx: usize) -> SourceIndex {
        let k = names.len();
        let mut windows = vec![Vec::new(); k + 1];
        let mut pure = vec![0f64; k + 1];
        let mut count = vec![0u64; k];
        let (mut off, mut d) = (0u64, 0usize);
        let span = ctx as u64 + 1;
        for sh in &set.shards {
            let n = sh.header.n_tokens;
            let slots = if n > ctx as u64 { (n - 1) / ctx as u64 } else { 0 };
            for j in 0..slots {
                let (s, e) = (off + j * ctx as u64, off + j * ctx as u64 + span);
                while d < docs.len() && docs[d].start + docs[d].len <= s {
                    d += 1;
                }
                count.iter_mut().for_each(|c| *c = 0);
                let mut i = d;
                while i < docs.len() && docs[i].start < e {
                    let (a, b) = (docs[i].start.max(s), (docs[i].start + docs[i].len).min(e));
                    if b > a {
                        count[docs[i].source as usize] += b - a;
                    }
                    i += 1;
                }
                let (best, &c) = count.iter().enumerate().max_by(|x, y| x.1.cmp(y.1).then(y.0.cmp(&x.0))).unwrap();
                let class = if 2 * c > span { best } else { k };
                windows[class].push(s);
                pure[class] += c as f64 / span as f64;
            }
            off += n;
        }
        let purity = pure.iter().zip(&windows).map(|(p, w)| if w.is_empty() { 0.0 } else { p / w.len() as f64 }).collect();
        let mut names = names.to_vec();
        names.push("mixed".into());
        SourceIndex { names, windows, purity, ctx }
    }

    pub fn total(&self) -> usize {
        self.windows.iter().map(|w| w.len()).sum()
    }
}

/// Random access to windows by global start (one open file per shard).
pub struct WindowReader {
    files: Vec<File>,
    offsets: Vec<u64>,
    lens: Vec<u64>,
    width: usize,
    buf: Vec<u8>,
}

impl WindowReader {
    pub fn open(set: &ShardSet) -> io::Result<WindowReader> {
        let mut off = 0;
        let (mut files, mut offsets, mut lens) = (Vec::new(), Vec::new(), Vec::new());
        for sh in &set.shards {
            files.push(File::open(&sh.path).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", sh.path.display())))?);
            offsets.push(off);
            lens.push(sh.header.n_tokens);
            off += sh.header.n_tokens;
        }
        let width = set.shards.first().map_or(2, |s| s.header.width as usize);
        Ok(WindowReader { files, offsets, lens, width, buf: Vec::new() })
    }

    /// `len` tokens from global position `start` (inside one shard).
    pub fn read(&mut self, start: u64, len: usize) -> io::Result<Vec<u32>> {
        let j = self.offsets.partition_point(|&o| o <= start).saturating_sub(1);
        let local = start - self.offsets[j];
        if local + len as u64 > self.lens[j] {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("window {start}+{len} leaves shard {j}")));
        }
        self.buf.resize(len * self.width, 0);
        let f = &mut self.files[j];
        f.seek(SeekFrom::Start(HEADER_LEN as u64 + local * self.width as u64))?;
        f.read_exact(&mut self.buf)?;
        Ok(match self.width {
            2 => self.buf.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).collect(),
            _ => self.buf.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        })
    }
}

/// `--lp-mix EPS[,EVERY[,CAP[,PROBE[,END]]]]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LpCfg {
    pub eps: f64,
    pub every: usize,
    pub cap: f64,
    pub probe: usize,
    /// the curriculum (v42 B6b): the tilt fades linearly to 0 at this fraction of training, then the mix is the natural
    /// one exactly; 0 = the tilt stays all run (B6)
    pub end: f64,
}

impl LpCfg {
    pub fn parse(s: &str) -> Result<LpCfg, String> {
        let v: Vec<&str> = s.split(',').collect();
        if v.is_empty() || v.len() > 5 {
            return Err(format!("--lp-mix EPS[,EVERY[,CAP[,PROBE[,END]]]], got {s:?}"));
        }
        let f = |i: usize, d: f64| -> Result<f64, String> { v.get(i).map_or(Ok(d), |x| x.parse().map_err(|_| format!("bad number {x:?} in --lp-mix"))) };
        let c = LpCfg { eps: f(0, 0.0)?, every: f(1, 250.0)? as usize, cap: f(2, 4.0)?, probe: f(3, 8.0)? as usize, end: f(4, 0.0)? };
        if !(0.0..=1.0).contains(&c.end) {
            return Err(format!("--lp-mix END is a fraction of training in [0, 1] (0 = no fade), got {}", c.end));
        }
        if !(0.0..=0.9).contains(&c.eps) {
            return Err(format!("--lp-mix EPS must be in [0, 0.9] (the natural mix keeps at least a tenth), got {}", c.eps));
        }
        if c.every == 0 || c.probe == 0 || c.cap < 1.0 {
            return Err("--lp-mix needs EVERY >= 1, PROBE >= 1 and CAP >= 1".into());
        }
        Ok(c)
    }
}

/// The learning-progress mixture and its draws (see the module docs).
pub struct LpMix {
    pub cfg: LpCfg,
    /// training done so far, 0..1 (set_frac); with cfg.end > 0 the tilt fades with it
    frac: f64,
    seed: u64,
    train: Vec<Vec<u64>>,
    pub probes: Vec<Vec<u64>>,
    pub base: Vec<f64>,
    pub p: Vec<f64>,
    pub hist: Vec<Vec<f64>>,
    cursor: Vec<(u64, usize)>,
    perm: Vec<Option<(u64, Vec<u32>)>>,
}

fn uniform(r: &mut SplitMix64) -> f64 {
    (r.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

impl LpMix {
    pub fn new(index: &SourceIndex, cfg: LpCfg, seed: u64) -> LpMix {
        let k = index.windows.len();
        let (mut train, mut probes) = (Vec::with_capacity(k), Vec::with_capacity(k));
        for (c, w) in index.windows.iter().enumerate() {
            if w.len() >= 100 * cfg.probe {
                let mut idx: Vec<usize> = (0..w.len()).collect();
                SplitMix64::new(derive_seed(seed, c as u64, PROBE_LABEL)).shuffle(&mut idx);
                let mut held: Vec<usize> = idx[..cfg.probe].to_vec();
                held.sort_unstable();
                probes.push(held.iter().map(|&i| w[i]).collect());
                let mut keep = Vec::with_capacity(w.len() - cfg.probe);
                let mut h = held.iter().peekable();
                for (i, &s) in w.iter().enumerate() {
                    if h.peek() == Some(&&i) {
                        h.next();
                    } else {
                        keep.push(s);
                    }
                }
                train.push(keep);
            } else {
                probes.push(Vec::new());
                train.push(w.clone());
            }
        }
        let total: usize = train.iter().map(|w| w.len()).sum();
        let base: Vec<f64> = train.iter().map(|w| w.len() as f64 / total.max(1) as f64).collect();
        LpMix { cfg, frac: 0.0, seed, p: base.clone(), base, train, probes, hist: vec![Vec::new(); k], cursor: vec![(0, 0); k], perm: vec![None; k] }
    }

    /// Each probed class's learning progress, once every probed class has 2 · LP_SPAN probe losses.
    pub fn progress(&self) -> Option<Vec<f64>> {
        let mut lp = vec![0.0; self.hist.len()];
        for (c, h) in self.hist.iter().enumerate() {
            if self.probes[c].is_empty() {
                continue;
            }
            if h.len() < 2 * LP_SPAN {
                return None;
            }
            let n = h.len();
            let now: f64 = h[n - LP_SPAN..].iter().sum::<f64>() / LP_SPAN as f64;
            let before: f64 = h[n - 2 * LP_SPAN..n - LP_SPAN].iter().sum::<f64>() / LP_SPAN as f64;
            lp[c] = (before - now).abs();
        }
        Some(lp)
    }

    /// B6b: how far training is (0..1); the next remix uses the faded tilt.
    pub fn set_frac(&mut self, f: f64) {
        self.frac = f;
    }

    /// The tilt now: EPS, or with END > 0 EPS · max(0, 1 − frac / END).
    pub fn eps(&self) -> f64 {
        if self.cfg.end > 0.0 {
            (self.cfg.eps * (1.0 - self.frac / self.cfg.end)).max(0.0)
        } else {
            self.cfg.eps
        }
    }

    fn remix(&mut self) {
        // the tilt off is the natural mix exactly (no renormalising rounding)
        if self.eps() == 0.0 {
            self.p = self.base.clone();
            return;
        }
        let Some(lp) = self.progress() else {
            self.p = self.base.clone();
            return;
        };
        let probed: Vec<usize> = (0..self.base.len()).filter(|&c| !self.probes[c].is_empty() && self.base[c] > 0.0).collect();
        let b: f64 = probed.iter().map(|&c| self.base[c]).sum();
        if probed.is_empty() || b <= 0.0 {
            self.p = self.base.clone();
            return;
        }
        // water-filling: q follows learning progress, at most cap · base / B per class
        let cap: Vec<f64> = probed.iter().map(|&c| self.cfg.cap * self.base[c] / b).collect();
        let mut q = vec![0.0; probed.len()];
        let mut fixed = vec![false; probed.len()];
        let mut left = 1.0;
        loop {
            let free: Vec<usize> = (0..probed.len()).filter(|&i| !fixed[i]).collect();
            if free.is_empty() || left <= 0.0 {
                break;
            }
            let w: f64 = free.iter().map(|&i| lp[probed[i]]).sum();
            let share = |i: usize| if w > 0.0 { lp[probed[i]] / w } else { self.base[probed[i]] / free.iter().map(|&j| self.base[probed[j]]).sum::<f64>() };
            let over: Vec<usize> = free.iter().copied().filter(|&i| left * share(i) > cap[i]).collect();
            if over.is_empty() {
                for &i in &free {
                    q[i] = left * share(i);
                }
                break;
            }
            for &i in &over {
                q[i] = cap[i];
                fixed[i] = true;
                left -= cap[i];
            }
        }
        let mut p = self.base.clone();
        for (i, &c) in probed.iter().enumerate() {
            p[c] = (1.0 - self.eps()) * self.base[c] + self.eps() * b * q[i];
        }
        let s: f64 = p.iter().sum();
        self.p = p.iter().map(|x| x / s).collect();
    }

    /// Record one probe loss per probed class (class, mean loss in nats per token); the mix moves.
    pub fn observe(&mut self, losses: &[(usize, f64)]) {
        for &(c, l) in losses {
            self.hist[c].push(l);
        }
        self.remix();
    }

    fn next_of(&mut self, c: usize) -> u64 {
        let n = self.train[c].len();
        let (epoch, pos) = self.cursor[c];
        if self.perm[c].as_ref().map_or(true, |(e, _)| *e != epoch) {
            let mut v: Vec<u32> = (0..n as u32).collect();
            SplitMix64::new(derive_seed(self.seed, c as u64 ^ (epoch << 8), PERM_LABEL)).shuffle(&mut v);
            self.perm[c] = Some((epoch, v));
        }
        let s = self.train[c][self.perm[c].as_ref().unwrap().1[pos] as usize];
        self.cursor[c] = if pos + 1 == n { (epoch + 1, 0) } else { (epoch, pos + 1) };
        s
    }

    /// Step `step`'s `rows` windows: (class, global start).
    pub fn draw(&mut self, step: u64, rows: usize) -> Vec<(usize, u64)> {
        let mut r = SplitMix64::new(derive_seed(self.seed, step, DRAW_LABEL));
        let mut out = Vec::with_capacity(rows);
        for _ in 0..rows {
            let u = uniform(&mut r);
            let mut acc = 0.0;
            let mut c = self.p.len() - 1;
            for (i, &pi) in self.p.iter().enumerate() {
                acc += pi;
                if u < acc && !self.train[i].is_empty() {
                    c = i;
                    break;
                }
            }
            while self.train[c].is_empty() {
                c -= 1;
            }
            out.push((c, self.next_of(c)));
        }
        out
    }

    /// Cursors and probe history, exactly (f64 bits in hex).
    pub fn state_line(&self) -> String {
        let cur: Vec<String> = self.cursor.iter().map(|(e, p)| format!("{e}:{p}")).collect();
        let hist: Vec<String> = self.hist.iter().map(|h| h.iter().map(|x| format!("{:016x}", x.to_bits())).collect::<Vec<_>>().join(";")).collect();
        format!("ava-lpmix v1 cursors={} hist={}", cur.join(","), hist.join("|"))
    }

    pub fn restore(&mut self, line: &str) -> Result<(), String> {
        let rest = line.strip_prefix("ava-lpmix v1 ").ok_or_else(|| format!("not an `ava-lpmix v1` line: {line:?}"))?;
        let (cur, hist) = rest.split_once(" hist=").ok_or("no hist= field")?;
        let cur = cur.strip_prefix("cursors=").ok_or("no cursors= field")?;
        let k = self.cursor.len();
        let cursor: Vec<(u64, usize)> = cur
            .split(',')
            .map(|x| {
                let (e, p) = x.split_once(':').ok_or_else(|| format!("bad cursor {x:?}"))?;
                Ok((e.parse().map_err(|_| format!("bad epoch {e:?}"))?, p.parse().map_err(|_| format!("bad pos {p:?}"))?))
            })
            .collect::<Result<_, String>>()?;
        let hist: Vec<Vec<f64>> = hist
            .split('|')
            .map(|h| h.split(';').filter(|x| !x.is_empty()).map(|x| u64::from_str_radix(x, 16).map(f64::from_bits).map_err(|_| format!("bad loss {x:?}"))).collect())
            .collect::<Result<_, String>>()?;
        if cursor.len() != k || hist.len() != k {
            return Err(format!("the state has {} classes, this index {k}", cursor.len()));
        }
        for (c, &(_, p)) in cursor.iter().enumerate() {
            if p >= self.train[c].len().max(1) {
                return Err(format!("cursor {p} beyond class {c}'s {} windows", self.train[c].len()));
            }
        }
        (self.cursor, self.hist) = (cursor, hist);
        self.remix();
        Ok(())
    }
}

#[cfg(test)]
mod tests;
