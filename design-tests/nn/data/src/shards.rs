//! TESTING / DESIGN ONLY — sharded token files, a streaming shuffled sampler, a resumable cursor.
//!
//! SHARD FILE (little-endian), named `shard_00000.tok`, `shard_00001.tok`, ...
//!   bytes  0..8   magic `AVATOK01`
//!   bytes  8..12  u32 version = 1
//!   bytes 12..16  u32 vocab_size
//!   bytes 16..24  u64 n_tokens
//!   byte  24      u8 width: 2 (u16 tokens) when vocab_size <= 65536, else 4 (u32 tokens)
//!   bytes 25..32  zero padding (header is 32 bytes)
//!   then n_tokens tokens of `width` bytes each.
//!
//! WRITING. `ShardWriter` streams tokens straight to disk (a `.tmp` file renamed into place
//! when the shard is complete, its header patched with the final count), so it holds no
//! shard in memory. A shard is cut exactly at `max_tokens_per_shard`: a document that does not
//! fit continues in the next shard, so every shard but the last is exactly full.
//! `TrainValWriter` / `make_shards` write `train/` and `val/` subdirectories: document number
//! `i` (0-based, in push order) goes to val when `mix64(seed ^ i) % 1000 < val_per_mille`
//! (`mix64` = first splitmix64 output; default 5 per mille = 0.5%).
//!
//! SAMPLING. `Sampler` yields windows of `ctx_len + 1` consecutive tokens (input + shifted
//! target). A shard of n tokens has `(n - 1) / ctx_len` window slots, slot k covering tokens
//! `k*ctx_len ..= k*ctx_len + ctx_len`; consecutive slots share one token, so every token after
//! the first is a target exactly once per epoch and no window crosses a shard boundary (a
//! shard shorter than ctx_len + 1 tokens has no slots and is skipped). Epoch e visits the
//! shards in the order `shuffle(0..n_shards)` seeded by `derive_seed(seed, e, u64::MAX)`, and
//! inside a shard visits a permutation of all its slots seeded by `derive_seed(seed, e,
//! shard_index)`. Only one shard is loaded at a time (the previous one is dropped first).
//!
//! CURSOR. `(seed, ctx_len, epoch, shard_pos, slot_pos)` always names the NEXT window: shard
//! `order[shard_pos]` of that epoch, slot `perm[slot_pos]`. It is one text line
//! (`ava-cursor v1 seed=.. ctx_len=.. epoch=.. shard_pos=.. slot_pos=..`), and
//! `Sampler::resume` continues with exactly the windows the original would have produced.
//!
//! EVALUATION. `ShardSet::iter_sequential(ctx_len)` reads the shards in file order as one
//! continuous stream and yields windows starting at 0, ctx_len, 2*ctx_len, ... (they share one
//! token, and may run across shard boundaries), so every token after the first is a target
//! exactly once. The last, partial window is PADDED with PAD (id 257) up to ctx_len + 1 tokens
//! (consumers must ignore PAD targets); a final remainder of a single token (nothing to
//! predict) is dropped.

use crate::bpe::PAD;
use crate::rng::{derive_seed, mix64, SplitMix64};
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const MAGIC: &[u8; 8] = b"AVATOK01";
pub const VERSION: u32 = 1;
pub const HEADER_LEN: usize = 32;
pub const DEFAULT_SHARD_TOKENS: usize = 50_000_000;
pub const DEFAULT_VAL_PER_MILLE: u32 = 5;
pub const DEFAULT_SPLIT_SEED: u64 = 0xA7A0_5EED;
const SHARD_ORDER_LABEL: u64 = u64::MAX;

fn invalid(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg)
}

fn bad_input(msg: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg)
}

/// 2 (u16 tokens) when every id fits in u16, else 4.
pub fn width_for_vocab(vocab_size: usize) -> u8 {
    if vocab_size <= 1 << 16 {
        2
    } else {
        4
    }
}

pub fn shard_name(index: usize) -> String {
    format!("shard_{index:05}.tok")
}

/// `shard_<digits>.tok` -> the number.
fn shard_index_of(name: &str) -> Option<u64> {
    let digits = name.strip_prefix("shard_")?.strip_suffix(".tok")?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// The shard files of `dir`, sorted by their number.
fn list_shards(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut found: Vec<(u64, String, PathBuf)> = Vec::new();
    let rd = fs::read_dir(dir).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", dir.display())))?;
    for entry in rd {
        let entry = entry?;
        if !entry.file_type()?.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if let Some(i) = shard_index_of(&name) {
            found.push((i, name, entry.path()));
        }
    }
    found.sort();
    Ok(found.into_iter().map(|(_, _, p)| p).collect())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShardHeader {
    pub vocab_size: u32,
    pub n_tokens: u64,
    pub width: u8,
}

impl ShardHeader {
    pub fn new(vocab_size: u32, n_tokens: u64) -> ShardHeader {
        ShardHeader { vocab_size, n_tokens, width: width_for_vocab(vocab_size as usize) }
    }

    pub fn to_bytes(&self) -> [u8; HEADER_LEN] {
        let mut b = [0u8; HEADER_LEN];
        b[0..8].copy_from_slice(MAGIC);
        b[8..12].copy_from_slice(&VERSION.to_le_bytes());
        b[12..16].copy_from_slice(&self.vocab_size.to_le_bytes());
        b[16..24].copy_from_slice(&self.n_tokens.to_le_bytes());
        b[24] = self.width;
        b
    }

    pub fn from_bytes(b: &[u8; HEADER_LEN]) -> io::Result<ShardHeader> {
        if &b[0..8] != MAGIC {
            return Err(invalid("not an AVATOK01 shard (bad magic)".into()));
        }
        let version = u32::from_le_bytes(b[8..12].try_into().unwrap());
        if version != VERSION {
            return Err(invalid(format!("unsupported shard version {version}")));
        }
        let vocab_size = u32::from_le_bytes(b[12..16].try_into().unwrap());
        let n_tokens = u64::from_le_bytes(b[16..24].try_into().unwrap());
        let width = b[24];
        if vocab_size == 0 {
            return Err(invalid("shard header has vocab_size 0".into()));
        }
        if width != width_for_vocab(vocab_size as usize) {
            return Err(invalid(format!("width {width} does not match vocab_size {vocab_size}")));
        }
        if b[25..32].iter().any(|&x| x != 0) {
            return Err(invalid("non-zero header padding".into()));
        }
        Ok(ShardHeader { vocab_size, n_tokens, width })
    }
}

// ---------------------------------------------------------------- writing

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WriterStats {
    pub shards: usize,
    pub tokens: u64,
    pub docs: u64,
}

/// Streams documents into `dir/shard_NNNNN.tok` files of at most `max_tokens_per_shard`
/// tokens each. Memory: one 1 MiB write buffer, never a whole shard.
pub struct ShardWriter {
    dir: PathBuf,
    vocab_size: u32,
    width: u8,
    max_tokens: u64,
    cur: Option<BufWriter<File>>,
    cur_tokens: u64,
    next_index: usize,
    stats: WriterStats,
    bytes: Vec<u8>,
}

impl ShardWriter {
    /// Creates `dir` if needed; refuses a directory that already holds shard files (so two
    /// runs are never silently mixed).
    pub fn new(dir: impl AsRef<Path>, vocab_size: usize, max_tokens_per_shard: usize) -> io::Result<ShardWriter> {
        let dir = dir.as_ref().to_path_buf();
        if max_tokens_per_shard == 0 {
            return Err(bad_input("max_tokens_per_shard must be > 0".into()));
        }
        if vocab_size == 0 || vocab_size > u32::MAX as usize {
            return Err(bad_input(format!("vocab_size {vocab_size} out of range")));
        }
        fs::create_dir_all(&dir)?;
        if !list_shards(&dir)?.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("{} already holds shard files; use an empty directory", dir.display()),
            ));
        }
        Ok(ShardWriter {
            dir,
            vocab_size: vocab_size as u32,
            width: width_for_vocab(vocab_size),
            max_tokens: max_tokens_per_shard as u64,
            cur: None,
            cur_tokens: 0,
            next_index: 0,
            stats: WriterStats::default(),
            bytes: Vec::new(),
        })
    }

    fn tmp_path(&self) -> PathBuf {
        self.dir.join(format!("{}.tmp", shard_name(self.next_index)))
    }

    fn open_next(&mut self) -> io::Result<()> {
        let mut w = BufWriter::with_capacity(1 << 20, File::create(self.tmp_path())?);
        w.write_all(&ShardHeader::new(self.vocab_size, 0).to_bytes())?;
        self.cur = Some(w);
        self.cur_tokens = 0;
        Ok(())
    }

    fn close_current(&mut self) -> io::Result<()> {
        let Some(mut w) = self.cur.take() else { return Ok(()) };
        w.flush()?;
        let mut f = w.into_inner().map_err(|e| e.into_error())?;
        f.seek(SeekFrom::Start(0))?;
        f.write_all(&ShardHeader::new(self.vocab_size, self.cur_tokens).to_bytes())?;
        f.sync_all()?;
        drop(f);
        fs::rename(self.tmp_path(), self.dir.join(shard_name(self.next_index)))?;
        self.next_index += 1;
        self.stats.shards += 1;
        self.cur_tokens = 0;
        Ok(())
    }

    fn write_tokens(&mut self, toks: &[u32]) -> io::Result<()> {
        self.bytes.clear();
        if self.width == 2 {
            for &t in toks {
                self.bytes.extend_from_slice(&(t as u16).to_le_bytes());
            }
        } else {
            for &t in toks {
                self.bytes.extend_from_slice(&t.to_le_bytes());
            }
        }
        self.cur.as_mut().expect("open shard").write_all(&self.bytes)
    }

    /// Append one document (it should already end in EOS). Ids must be < vocab_size.
    pub fn push_doc(&mut self, doc: &[u32]) -> io::Result<()> {
        if let Some(&bad) = doc.iter().find(|&&t| t >= self.vocab_size) {
            return Err(invalid(format!("token id {bad} >= vocab_size {}", self.vocab_size)));
        }
        let mut rest = doc;
        while !rest.is_empty() {
            if self.cur.is_none() {
                self.open_next()?;
            }
            let room = (self.max_tokens - self.cur_tokens) as usize;
            let take = room.min(rest.len());
            self.write_tokens(&rest[..take])?;
            self.cur_tokens += take as u64;
            self.stats.tokens += take as u64;
            rest = &rest[take..];
            if self.cur_tokens == self.max_tokens {
                self.close_current()?;
            }
        }
        self.stats.docs += 1;
        Ok(())
    }

    pub fn stats(&self) -> WriterStats {
        self.stats
    }

    /// Write out the last (partial) shard.
    pub fn finish(mut self) -> io::Result<WriterStats> {
        self.close_current()?;
        Ok(self.stats)
    }
}

/// The split rule: document `doc_index` goes to val when `mix64(seed ^ doc_index) % 1000 <
/// val_per_mille`.
pub fn is_val_doc(seed: u64, doc_index: u64, val_per_mille: u32) -> bool {
    mix64(seed ^ doc_index) % 1000 < val_per_mille as u64
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SplitStats {
    pub train: WriterStats,
    pub val: WriterStats,
}

/// Writes `out_dir/train` and `out_dir/val`, routing each document by `is_val_doc`.
pub struct TrainValWriter {
    train: ShardWriter,
    val: ShardWriter,
    seed: u64,
    val_per_mille: u32,
    next_doc_index: u64,
}

impl TrainValWriter {
    pub fn new(out_dir: impl AsRef<Path>, vocab_size: usize, max_tokens_per_shard: usize, val_per_mille: u32, seed: u64) -> io::Result<TrainValWriter> {
        if val_per_mille > 1000 {
            return Err(bad_input(format!("val_per_mille {val_per_mille} > 1000")));
        }
        let out = out_dir.as_ref();
        Ok(TrainValWriter {
            train: ShardWriter::new(out.join("train"), vocab_size, max_tokens_per_shard)?,
            val: ShardWriter::new(out.join("val"), vocab_size, max_tokens_per_shard)?,
            seed,
            val_per_mille,
            next_doc_index: 0,
        })
    }

    /// Route the next document; returns true when it went to val.
    pub fn push_doc(&mut self, doc: &[u32]) -> io::Result<bool> {
        let to_val = is_val_doc(self.seed, self.next_doc_index, self.val_per_mille);
        self.next_doc_index += 1;
        if to_val {
            self.val.push_doc(doc)?;
        } else {
            self.train.push_doc(doc)?;
        }
        Ok(to_val)
    }

    pub fn stats(&self) -> SplitStats {
        SplitStats { train: self.train.stats(), val: self.val.stats() }
    }

    pub fn finish(self) -> io::Result<SplitStats> {
        Ok(SplitStats { train: self.train.finish()?, val: self.val.finish()? })
    }
}

/// Write `docs` (each already ending in EOS) into `out_dir/train` and `out_dir/val`.
pub fn make_shards(
    out_dir: impl AsRef<Path>,
    vocab_size: usize,
    max_tokens_per_shard: usize,
    val_per_mille: u32,
    seed: u64,
    docs: impl IntoIterator<Item = Vec<u32>>,
) -> io::Result<SplitStats> {
    let mut w = TrainValWriter::new(out_dir, vocab_size, max_tokens_per_shard, val_per_mille, seed)?;
    for d in docs {
        w.push_doc(&d)?;
    }
    w.finish()
}

// ---------------------------------------------------------------- reading

#[derive(Clone, Debug)]
pub struct ShardInfo {
    pub path: PathBuf,
    pub header: ShardHeader,
}

#[derive(Clone, Debug)]
pub struct ShardSet {
    pub dir: PathBuf,
    /// Sorted by shard number.
    pub shards: Vec<ShardInfo>,
    /// 0 when the set is empty.
    pub vocab_size: u32,
    pub total_tokens: u64,
}

/// One loaded shard, kept at its on-disk width.
pub enum ShardTokens {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl ShardTokens {
    pub fn len(&self) -> usize {
        match self {
            ShardTokens::U16(v) => v.len(),
            ShardTokens::U32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, i: usize) -> u32 {
        match self {
            ShardTokens::U16(v) => v[i] as u32,
            ShardTokens::U32(v) => v[i],
        }
    }

    /// Tokens `start..start + len` as u32.
    pub fn window(&self, start: usize, len: usize) -> Vec<u32> {
        match self {
            ShardTokens::U16(v) => v[start..start + len].iter().map(|&t| t as u32).collect(),
            ShardTokens::U32(v) => v[start..start + len].to_vec(),
        }
    }

    pub fn to_vec(&self) -> Vec<u32> {
        self.window(0, self.len())
    }
}

impl ShardSet {
    /// List `dir/shard_*.tok` (sorted by number), validate every header and file length, and
    /// check that all shards share one vocab_size. Other files are ignored.
    pub fn open(dir: impl AsRef<Path>) -> io::Result<ShardSet> {
        let dir = dir.as_ref().to_path_buf();
        let mut shards = Vec::new();
        let mut vocab: Option<u32> = None;
        let mut total = 0u64;
        for path in list_shards(&dir)? {
            let ctx = |e: io::Error| io::Error::new(e.kind(), format!("{}: {e}", path.display()));
            let mut f = File::open(&path).map_err(ctx)?;
            let mut hb = [0u8; HEADER_LEN];
            f.read_exact(&mut hb).map_err(|_| invalid(format!("{}: shorter than a header", path.display())))?;
            let header = ShardHeader::from_bytes(&hb).map_err(ctx)?;
            let len = f.metadata().map_err(ctx)?.len();
            let want = HEADER_LEN as u64 + header.n_tokens * header.width as u64;
            if len != want {
                return Err(invalid(format!("{}: file is {len} bytes, header implies {want}", path.display())));
            }
            match vocab {
                None => vocab = Some(header.vocab_size),
                Some(v) if v != header.vocab_size => {
                    return Err(invalid(format!("{}: vocab_size {} differs from {v}", path.display(), header.vocab_size)));
                }
                _ => {}
            }
            total += header.n_tokens;
            shards.push(ShardInfo { path, header });
        }
        Ok(ShardSet { dir, shards, vocab_size: vocab.unwrap_or(0), total_tokens: total })
    }

    pub fn len(&self) -> usize {
        self.shards.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shards.is_empty()
    }

    /// Read shard `i` into memory (peak memory: the shard plus a 1 MiB read buffer).
    pub fn load_shard(&self, i: usize) -> io::Result<ShardTokens> {
        let info = &self.shards[i];
        let mut f = File::open(&info.path)?;
        let mut hb = [0u8; HEADER_LEN];
        f.read_exact(&mut hb)?;
        if ShardHeader::from_bytes(&hb)? != info.header {
            return Err(invalid(format!("{}: header changed since the set was opened", info.path.display())));
        }
        let n = info.header.n_tokens as usize;
        let w = info.header.width as usize;
        let mut raw = vec![0u8; (1 << 20) / w * w];
        let mut left = n;
        if w == 2 {
            let mut v: Vec<u16> = Vec::with_capacity(n);
            while left > 0 {
                let k = left.min(raw.len() / 2);
                f.read_exact(&mut raw[..k * 2])?;
                v.extend(raw[..k * 2].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])));
                left -= k;
            }
            Ok(ShardTokens::U16(v))
        } else {
            let mut v: Vec<u32> = Vec::with_capacity(n);
            while left > 0 {
                let k = left.min(raw.len() / 4);
                f.read_exact(&mut raw[..k * 4])?;
                v.extend(raw[..k * 4].chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])));
                left -= k;
            }
            Ok(ShardTokens::U32(v))
        }
    }

    /// Every token once, in file order, as windows of ctx_len + 1 (see module docs: stride
    /// ctx_len, across shard boundaries, last partial window PAD-padded). Panics on I/O errors.
    pub fn iter_sequential(&self, ctx_len: usize) -> SequentialIter {
        assert!(ctx_len >= 1, "ctx_len must be >= 1");
        SequentialIter { set: self.clone(), ctx_len, next_shard: 0, buf: Vec::new(), off: 0, done: false }
    }
}

pub struct SequentialIter {
    set: ShardSet,
    ctx_len: usize,
    next_shard: usize,
    buf: Vec<u32>,
    off: usize,
    done: bool,
}

impl Iterator for SequentialIter {
    type Item = Vec<u32>;

    fn next(&mut self) -> Option<Vec<u32>> {
        if self.done {
            return None;
        }
        let need = self.ctx_len + 1;
        while self.buf.len() - self.off < need && self.next_shard < self.set.len() {
            let toks = self
                .set
                .load_shard(self.next_shard)
                .unwrap_or_else(|e| panic!("ava_data: reading shard {} failed: {e}", self.next_shard));
            self.next_shard += 1;
            let mut nb = Vec::with_capacity(self.buf.len() - self.off + toks.len());
            nb.extend_from_slice(&self.buf[self.off..]);
            match &toks {
                ShardTokens::U16(v) => nb.extend(v.iter().map(|&t| t as u32)),
                ShardTokens::U32(v) => nb.extend_from_slice(v),
            }
            self.buf = nb;
            self.off = 0;
        }
        let avail = self.buf.len() - self.off;
        if avail >= need {
            let w = self.buf[self.off..self.off + need].to_vec();
            self.off += self.ctx_len;
            return Some(w);
        }
        self.done = true;
        if avail >= 2 {
            let mut w = self.buf[self.off..].to_vec();
            w.resize(need, PAD);
            return Some(w);
        }
        None
    }
}

// ---------------------------------------------------------------- sampler + cursor

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cursor {
    pub seed: u64,
    pub ctx_len: usize,
    pub epoch: u64,
    pub shard_pos: usize,
    pub slot_pos: usize,
}

impl Cursor {
    pub fn to_line(&self) -> String {
        format!(
            "ava-cursor v1 seed={} ctx_len={} epoch={} shard_pos={} slot_pos={}",
            self.seed, self.ctx_len, self.epoch, self.shard_pos, self.slot_pos
        )
    }

    pub fn parse(line: &str) -> Result<Cursor, String> {
        let mut it = line.split_whitespace();
        if it.next() != Some("ava-cursor") || it.next() != Some("v1") {
            return Err(format!("not an `ava-cursor v1` line: {line:?}"));
        }
        let mut vals: [Option<u64>; 5] = [None; 5];
        const KEYS: [&str; 5] = ["seed", "ctx_len", "epoch", "shard_pos", "slot_pos"];
        for kv in it {
            let (k, v) = kv.split_once('=').ok_or_else(|| format!("bad field {kv:?}"))?;
            let i = KEYS.iter().position(|&x| x == k).ok_or_else(|| format!("unknown field {k:?}"))?;
            if vals[i].is_some() {
                return Err(format!("duplicate field {k:?}"));
            }
            vals[i] = Some(v.parse().map_err(|_| format!("bad value in {kv:?}"))?);
        }
        let get = |i: usize| vals[i].ok_or_else(|| format!("missing field {:?}", KEYS[i]));
        Ok(Cursor {
            seed: get(0)?,
            ctx_len: get(1)? as usize,
            epoch: get(2)?,
            shard_pos: get(3)? as usize,
            slot_pos: get(4)? as usize,
        })
    }
}

fn n_slots(n_tokens: u64, ctx_len: usize) -> u64 {
    if n_tokens > ctx_len as u64 {
        (n_tokens - 1) / ctx_len as u64
    } else {
        0
    }
}

fn shard_order(seed: u64, epoch: u64, n: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    SplitMix64::new(derive_seed(seed, epoch, SHARD_ORDER_LABEL)).shuffle(&mut v);
    v
}

fn slot_perm(seed: u64, epoch: u64, shard: usize, n: u64) -> Vec<u32> {
    let mut v: Vec<u32> = (0..n as u32).collect();
    SplitMix64::new(derive_seed(seed, epoch, shard as u64)).shuffle(&mut v);
    v
}

struct Loaded {
    shard: usize,
    epoch: u64,
    tokens: ShardTokens,
    perm: Vec<u32>,
}

/// Shuffled windows over a `ShardSet`, one shard in memory at a time (see module docs).
pub struct Sampler {
    set: ShardSet,
    seed: u64,
    ctx_len: usize,
    slots: Vec<u64>,
    windows_per_epoch: u64,
    epoch: u64,
    shard_pos: usize,
    slot_pos: usize,
    order: Vec<usize>,
    loaded: Option<Loaded>,
    last_position: Option<(usize, usize)>,
}

impl Sampler {
    pub fn new(set: &ShardSet, ctx_len: usize, seed: u64) -> io::Result<Sampler> {
        Sampler::build(set, Cursor { seed, ctx_len, epoch: 0, shard_pos: 0, slot_pos: 0 })
    }

    /// Continue exactly where `cursor` was taken. Errors if the cursor cannot belong to this
    /// set (shard_pos or slot_pos out of range, ctx_len 0, or no window fits).
    pub fn resume(set: &ShardSet, cursor: Cursor) -> io::Result<Sampler> {
        Sampler::build(set, cursor)
    }

    fn build(set: &ShardSet, c: Cursor) -> io::Result<Sampler> {
        if c.ctx_len == 0 {
            return Err(bad_input("ctx_len must be >= 1".into()));
        }
        let slots: Vec<u64> = set.shards.iter().map(|s| n_slots(s.header.n_tokens, c.ctx_len)).collect();
        if slots.iter().any(|&s| s > u32::MAX as u64) {
            return Err(bad_input("a shard has more than 2^32 window slots; use larger ctx_len or smaller shards".into()));
        }
        let wpe: u64 = slots.iter().sum();
        if wpe == 0 {
            return Err(bad_input(format!("no shard in {} holds a window of {} tokens", set.dir.display(), c.ctx_len + 1)));
        }
        if c.shard_pos >= set.len() {
            return Err(bad_input(format!("cursor shard_pos {} but the set has {} shards", c.shard_pos, set.len())));
        }
        let order = shard_order(c.seed, c.epoch, set.len());
        if c.slot_pos as u64 > slots[order[c.shard_pos]] {
            return Err(bad_input(format!("cursor slot_pos {} beyond the shard's {} slots", c.slot_pos, slots[order[c.shard_pos]])));
        }
        let mut s = Sampler {
            set: set.clone(),
            seed: c.seed,
            ctx_len: c.ctx_len,
            slots,
            windows_per_epoch: wpe,
            epoch: c.epoch,
            shard_pos: c.shard_pos,
            slot_pos: c.slot_pos,
            order,
            loaded: None,
            last_position: None,
        };
        s.normalize();
        Ok(s)
    }

    /// Move (epoch, shard_pos, slot_pos) forward until it names an existing slot.
    fn normalize(&mut self) {
        while self.slot_pos as u64 >= self.slots[self.order[self.shard_pos]] {
            self.slot_pos = 0;
            self.shard_pos += 1;
            if self.shard_pos == self.order.len() {
                self.shard_pos = 0;
                self.epoch += 1;
                self.order = shard_order(self.seed, self.epoch, self.set.len());
            }
        }
    }

    /// Where the next window comes from.
    pub fn cursor(&self) -> Cursor {
        Cursor { seed: self.seed, ctx_len: self.ctx_len, epoch: self.epoch, shard_pos: self.shard_pos, slot_pos: self.slot_pos }
    }

    pub fn try_next_window(&mut self) -> io::Result<Vec<u32>> {
        let shard = self.order[self.shard_pos];
        if self.loaded.as_ref().map(|l| l.shard) != Some(shard) {
            self.loaded = None; // drop the old shard before reading the next one
            let tokens = self.set.load_shard(shard)?;
            let perm = slot_perm(self.seed, self.epoch, shard, self.slots[shard]);
            self.loaded = Some(Loaded { shard, epoch: self.epoch, tokens, perm });
        }
        let l = self.loaded.as_mut().unwrap();
        if l.epoch != self.epoch {
            // same shard again in a new epoch: new slot order, same tokens
            l.perm = slot_perm(self.seed, self.epoch, shard, self.slots[shard]);
            l.epoch = self.epoch;
        }
        let start = l.perm[self.slot_pos] as usize * self.ctx_len;
        let w = l.tokens.window(start, self.ctx_len + 1);
        self.last_position = Some((shard, start));
        self.slot_pos += 1;
        self.normalize();
        Ok(w)
    }

    /// `ctx_len + 1` consecutive tokens. Panics if a shard cannot be read (the files changed
    /// since the set was opened); use `try_next_window` to handle that.
    pub fn next_window(&mut self) -> Vec<u32> {
        self.try_next_window().unwrap_or_else(|e| panic!("ava_data sampler: {e}"))
    }

    pub fn batch(&mut self, n: usize) -> Vec<Vec<u32>> {
        (0..n).map(|_| self.next_window()).collect()
    }

    pub fn windows_per_epoch(&self) -> u64 {
        self.windows_per_epoch
    }

    pub fn ctx_len(&self) -> usize {
        self.ctx_len
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// (shard index in the set, token offset) of the window returned last.
    pub fn last_position(&self) -> Option<(usize, usize)> {
        self.last_position
    }
}
