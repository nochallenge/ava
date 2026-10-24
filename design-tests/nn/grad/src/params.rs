// TESTING / DESIGN ONLY — ava_grad::params
// The parameter store: named tensors, a gradient buffer per tensor, and a per-tensor weight-decay flag.
// Also the gradient bundle `Grads` returned by Graph::backward (kept separate so the graph can borrow
// the store read-only during the forward/backward pass and the caller adds the result in afterwards),
// and the AVAWTS01 checkpoint format.
//
// Checkpoint layout (all little-endian):
//   magic "AVAWTS01" (8 bytes), u32 n_tensors, then for each tensor:
//   u32 name_len, name bytes (UTF-8), u32 ndim, u64 dims[ndim], f32 data[prod(dims)].

use crate::tensor::{numel_of, Tensor};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

pub const CHECKPOINT_MAGIC: &[u8; 8] = b"AVAWTS01";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParamId(pub usize);

#[derive(Clone, Debug, Default)]
pub struct Params {
    names: Vec<String>,
    values: Vec<Tensor>,
    grads: Vec<Tensor>,
    decay: Vec<bool>,
    index: HashMap<String, usize>,
}

impl Params {
    pub fn new() -> Params {
        Params::default()
    }

    /// Register a parameter. `decay` = whether AdamW applies weight decay to it (usually true for
    /// weight matrices, false for gains, biases and LIF leak parameters). Panics on a duplicate name.
    pub fn add(&mut self, name: &str, value: Tensor, decay: bool) -> ParamId {
        assert!(!self.index.contains_key(name), "Params::add: duplicate parameter name {name:?}");
        let id = self.values.len();
        self.index.insert(name.to_string(), id);
        self.names.push(name.to_string());
        self.grads.push(Tensor::zeros(&value.shape));
        self.values.push(value);
        self.decay.push(decay);
        ParamId(id)
    }

    /// Number of parameter tensors.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn ids(&self) -> impl Iterator<Item = ParamId> {
        (0..self.values.len()).map(ParamId)
    }

    pub fn name(&self, id: ParamId) -> &str {
        &self.names[id.0]
    }

    pub fn decay(&self, id: ParamId) -> bool {
        self.decay[id.0]
    }

    pub fn find(&self, name: &str) -> Option<ParamId> {
        self.index.get(name).map(|&i| ParamId(i))
    }

    pub fn value(&self, id: ParamId) -> &Tensor {
        &self.values[id.0]
    }

    /// Mutable access to the values. Do not change the shape (the gradient buffer keeps the old one).
    pub fn value_mut(&mut self, id: ParamId) -> &mut Tensor {
        &mut self.values[id.0]
    }

    pub fn grad(&self, id: ParamId) -> &Tensor {
        &self.grads[id.0]
    }

    pub fn grad_mut(&mut self, id: ParamId) -> &mut Tensor {
        &mut self.grads[id.0]
    }

    /// Split borrow used by optimisers: the value mutably and its gradient immutably.
    pub fn value_grad_mut(&mut self, id: ParamId) -> (&mut Tensor, &Tensor) {
        (&mut self.values[id.0], &self.grads[id.0])
    }

    pub fn zero_grads(&mut self) {
        for g in &mut self.grads {
            g.data.iter_mut().for_each(|v| *v = 0.0);
        }
    }

    /// Add a backward pass's gradients into the stored gradient buffers (so several micro-batches
    /// can be accumulated before one optimiser step).
    pub fn accumulate(&mut self, grads: &Grads) {
        for (i, g) in grads.g.iter().enumerate() {
            if let Some(g) = g {
                let dst = &mut self.grads[i].data;
                assert_eq!(dst.len(), g.len(), "Params::accumulate: gradient size mismatch for {}", self.names[i]);
                for (d, s) in dst.iter_mut().zip(g) {
                    *d += *s;
                }
            }
        }
    }

    /// Total number of scalars.
    pub fn count(&self) -> usize {
        self.values.iter().map(|t| t.numel()).sum()
    }

    /// Number of scalars in the tensors whose name satisfies `f`.
    pub fn count_where<F: Fn(&str) -> bool>(&self, f: F) -> usize {
        self.names.iter().zip(&self.values).filter(|(n, _)| f(n)).map(|(_, t)| t.numel()).sum()
    }

    /// Global L2 norm of all stored gradients.
    pub fn grad_norm(&self) -> f32 {
        let s: f64 = self.grads.iter().flat_map(|g| g.data.iter()).map(|&v| v as f64 * v as f64).sum();
        s.sqrt() as f32
    }

    /// Write all values as an AVAWTS01 checkpoint (written to `<path>.tmp` then renamed into place).
    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let items: Vec<(&str, &Tensor)> = self.names.iter().map(|s| s.as_str()).zip(self.values.iter()).collect();
        write_checkpoint(path, &items)
    }

    /// Load values by name from a checkpoint into this (already built) store. The set of names must match
    /// exactly and every shape must agree; on any error nothing is modified.
    pub fn load<P: AsRef<Path>>(&mut self, path: P) -> io::Result<()> {
        let items = read_checkpoint(path)?;
        let bad = |msg: String| io::Error::new(io::ErrorKind::InvalidData, msg);
        if items.len() != self.values.len() {
            return Err(bad(format!("checkpoint has {} tensors, store has {}", items.len(), self.values.len())));
        }
        let mut seen = vec![false; self.values.len()];
        let mut plan = Vec::with_capacity(items.len());
        for (name, t) in items {
            let id = self.find(&name).ok_or_else(|| bad(format!("checkpoint tensor {name:?} not in the store")))?;
            if seen[id.0] {
                return Err(bad(format!("checkpoint repeats tensor {name:?}")));
            }
            seen[id.0] = true;
            if t.shape != self.values[id.0].shape {
                return Err(bad(format!("{name:?}: checkpoint shape {:?} vs store shape {:?}", t.shape, self.values[id.0].shape)));
            }
            plan.push((id, t));
        }
        for (id, t) in plan {
            self.values[id.0] = t;
        }
        Ok(())
    }
}

/// Write named tensors in the AVAWTS01 format.
pub fn write_checkpoint<P: AsRef<Path>>(path: P, items: &[(&str, &Tensor)]) -> io::Result<()> {
    let path = path.as_ref();
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp);
    {
        let mut w = BufWriter::new(File::create(&tmp)?);
        w.write_all(CHECKPOINT_MAGIC)?;
        w.write_all(&(items.len() as u32).to_le_bytes())?;
        for (name, t) in items {
            assert_eq!(numel_of(&t.shape), t.data.len());
            w.write_all(&(name.len() as u32).to_le_bytes())?;
            w.write_all(name.as_bytes())?;
            w.write_all(&(t.shape.len() as u32).to_le_bytes())?;
            for &d in &t.shape {
                w.write_all(&(d as u64).to_le_bytes())?;
            }
            let mut buf = Vec::with_capacity(t.data.len() * 4);
            for &v in &t.data {
                buf.extend_from_slice(&v.to_le_bytes());
            }
            w.write_all(&buf)?;
        }
        w.flush()?;
    }
    std::fs::rename(&tmp, path)
}

/// Read every (name, tensor) from an AVAWTS01 checkpoint, in file order. Rejects bad magic, truncation,
/// trailing bytes and absurd sizes.
pub fn read_checkpoint<P: AsRef<Path>>(path: P) -> io::Result<Vec<(String, Tensor)>> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut r = BufReader::new(file);
    let bad = |msg: &str| io::Error::new(io::ErrorKind::InvalidData, msg.to_string());
    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != CHECKPOINT_MAGIC {
        return Err(bad("not an AVAWTS01 checkpoint"));
    }
    let mut consumed: u64 = 8;
    let mut u32b = [0u8; 4];
    let mut u64b = [0u8; 8];
    r.read_exact(&mut u32b)?;
    consumed += 4;
    let n = u32::from_le_bytes(u32b) as usize;
    let mut out = Vec::with_capacity(n.min(1 << 16));
    for _ in 0..n {
        r.read_exact(&mut u32b)?;
        let name_len = u32::from_le_bytes(u32b) as usize;
        if name_len > 1 << 16 {
            return Err(bad("name too long"));
        }
        let mut name = vec![0u8; name_len];
        r.read_exact(&mut name)?;
        let name = String::from_utf8(name).map_err(|_| bad("name is not UTF-8"))?;
        r.read_exact(&mut u32b)?;
        let ndim = u32::from_le_bytes(u32b) as usize;
        if ndim > 16 {
            return Err(bad("too many dimensions"));
        }
        consumed += 4 + name_len as u64 + 4 + 8 * ndim as u64;
        let mut shape = Vec::with_capacity(ndim);
        let mut numel: u64 = 1;
        for _ in 0..ndim {
            r.read_exact(&mut u64b)?;
            let d = u64::from_le_bytes(u64b);
            numel = numel.checked_mul(d).ok_or_else(|| bad("shape overflows"))?;
            shape.push(d as usize);
        }
        let bytes = numel.checked_mul(4).ok_or_else(|| bad("shape overflows"))?;
        if consumed + bytes > file_len {
            return Err(bad("checkpoint is truncated"));
        }
        let mut raw = vec![0u8; bytes as usize];
        r.read_exact(&mut raw)?;
        consumed += bytes;
        let data = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        out.push((name, Tensor { shape, data }));
    }
    if consumed != file_len {
        return Err(bad("trailing bytes after the last tensor"));
    }
    Ok(out)
}

/// Gradients produced by one Graph::backward, indexed by ParamId. Only parameters that the loss actually
/// depends on are present.
#[derive(Clone, Debug, Default)]
pub struct Grads {
    pub(crate) g: Vec<Option<Vec<f32>>>,
}

impl Grads {
    pub(crate) fn new(n_params: usize) -> Grads {
        Grads { g: (0..n_params).map(|_| None).collect() }
    }

    pub(crate) fn add(&mut self, id: ParamId, data: Vec<f32>) {
        match &mut self.g[id.0] {
            Some(dst) => {
                for (d, s) in dst.iter_mut().zip(&data) {
                    *d += *s;
                }
            }
            slot @ None => *slot = Some(data),
        }
    }

    pub fn get(&self, id: ParamId) -> Option<&[f32]> {
        self.g.get(id.0).and_then(|o| o.as_deref())
    }

    /// Global L2 norm of these gradients.
    pub fn norm(&self) -> f32 {
        let s: f64 = self.g.iter().flatten().flat_map(|v| v.iter()).map(|&x| x as f64 * x as f64).sum();
        s.sqrt() as f32
    }
}
