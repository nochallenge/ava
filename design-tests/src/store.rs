//! Persistence. Until now the brain existed only while the process ran — every run
//! re-learned 652,477 facts from scratch and then forgot them on exit.
//!
//! What has to be saved is not obvious, and getting it wrong gives you a brain that
//! loads but cannot answer. THREE things, and all three are required:
//!
//!   1. the SYNAPSES        — the memory itself, `Unit.out`
//!   2. the ALLOCATOR       — which units each feature competes for, and the weights
//!                            that competition learned. Without this, the same word
//!                            encodes to a DIFFERENT code after reload and every
//!                            memory becomes unreachable even though it is all there
//!   3. the READOUT         — the op votes, or she recalls perfectly and says nothing
//!
//! Point 2 is the subtle one. The memory is addressed by content, so the addressing
//! function is part of the memory.

use crate::v4;
use crate::v6;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"AVABRAIN";
const VERSION: u32 = 1;

// ---------------------------------------------------------------- primitives

fn wu32<W: Write>(w: &mut W, v: u32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn wu16<W: Write>(w: &mut W, v: u16) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn wf32<W: Write>(w: &mut W, v: f32) -> io::Result<()> { w.write_all(&v.to_le_bytes()) }
fn wstr<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    wu32(w, s.len() as u32)?;
    w.write_all(s.as_bytes())
}

fn ru32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn ru16<R: Read>(r: &mut R) -> io::Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
fn rf32<R: Read>(r: &mut R) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}
fn rstr<R: Read>(r: &mut R) -> io::Result<String> {
    let n = ru32(r)? as usize;
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    String::from_utf8(b).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn bad(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
}

// ---------------------------------------------------------------- save

pub fn save(
    path: impl AsRef<Path>,
    brain: &v4::Brain,
    fac: &v6::Factored,
    readout: &v6::OpReadout,
) -> io::Result<u64> {
    let file = File::create(path.as_ref())?;
    let mut w = BufWriter::with_capacity(1 << 22, file);

    w.write_all(MAGIC)?;
    wu32(&mut w, VERSION)?;

    // ---- 1. synapses ----
    wu32(&mut w, brain.units.len() as u32)?;
    for u in &brain.units {
        wu32(&mut w, u.out.len() as u32)?;
        // targets are stored sorted, so write deltas: most fit in far fewer bits
        // than a raw u32 would need. kept as fixed u32 here for simplicity and
        // verified by round trip; varint would shrink the file further.
        for &(to, s) in &u.out {
            wu32(&mut w, to)?;
            wu16(&mut w, s)?;
        }
    }

    // ---- 2. allocator ----
    wu32(&mut w, fac.id.n)?;
    wu32(&mut w, fac.id.k as u32)?;
    wu32(&mut w, fac.gram.base)?;
    wu32(&mut w, fac.gram.span)?;
    wu32(&mut w, fac.gram.per_tag as u32)?;
    wu32(&mut w, fac.n_total)?;

    // features in id order, so cand[] and feat_id stay consistent on load
    let mut by_id: Vec<(&String, u32)> = fac.id.feat_id.iter().map(|(k, &v)| (k, v)).collect();
    by_id.sort_unstable_by_key(|&(_, v)| v);
    wu32(&mut w, by_id.len() as u32)?;
    for (name, id) in &by_id {
        wstr(&mut w, name)?;
        for &(unit, weight) in fac.id.cand[*id as usize].iter() {
            wu32(&mut w, unit)?;
            wf32(&mut w, weight)?;
        }
    }
    wu32(&mut w, fac.id.duty.len() as u32)?;
    for &d in &fac.id.duty {
        wf32(&mut w, d)?;
    }

    // ---- 3. readout ----
    wu32(&mut w, readout.n_learned)?;
    wu32(&mut w, readout.ops.len() as u32)?;
    for o in &readout.ops {
        wstr(&mut w, o)?;
    }
    wu32(&mut w, readout.votes.len() as u32)?;
    for (&unit, ops) in &readout.votes {
        wu32(&mut w, unit)?;
        wu32(&mut w, *readout.total.get(&unit).unwrap_or(&0))?;
        wu32(&mut w, ops.len() as u32)?;
        for (&op, &count) in ops {
            wu32(&mut w, op)?;
            // votes are SIGNED since the delta rule can drive them negative; the wire
            // format stays u32 and round-trips through a two's-complement cast.
            wu32(&mut w, count as u32)?;
        }
    }

    w.flush()?;
    Ok(std::fs::metadata(path.as_ref())?.len())
}

// ---------------------------------------------------------------- load

pub fn load(
    path: impl AsRef<Path>,
    params: v4::Params,
) -> io::Result<(v4::Brain, v6::Factored, v6::OpReadout)> {
    let file = File::open(path.as_ref())?;
    let mut r = BufReader::with_capacity(1 << 22, file);

    let mut magic = [0u8; 8];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(bad("not an ava brain file"));
    }
    let v = ru32(&mut r)?;
    if v != VERSION {
        return Err(bad(&format!("brain file version {v}, this build expects {VERSION}")));
    }

    // ---- 1. synapses ----
    let n_units = ru32(&mut r)? as usize;
    let mut brain = v4::Brain::new(n_units as u32, params);
    for i in 0..n_units {
        let n = ru32(&mut r)? as usize;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let to = ru32(&mut r)?;
            let s = ru16(&mut r)?;
            out.push((to, s));
        }
        brain.units[i].out = out;
    }

    // ---- 2. allocator ----
    let n_id = ru32(&mut r)?;
    let k_id = ru32(&mut r)? as usize;
    let g_base = ru32(&mut r)?;
    let g_span = ru32(&mut r)?;
    let per_tag = ru32(&mut r)? as usize;
    let n_total = ru32(&mut r)?;

    let mut fac = v6::Factored::new(n_id, g_span, k_id, per_tag);
    fac.gram.base = g_base;
    fac.n_total = n_total;

    let n_feat = ru32(&mut r)? as usize;
    fac.id.cand = Vec::with_capacity(n_feat);
    fac.id.feat_id = HashMap::with_capacity(n_feat);
    for id in 0..n_feat {
        let name = rstr(&mut r)?;
        let mut pool = [(0u32, 0.0f32); v6::CAND];
        for slot in pool.iter_mut() {
            slot.0 = ru32(&mut r)?;
            slot.1 = rf32(&mut r)?;
        }
        fac.id.cand.push(pool);
        fac.id.feat_id.insert(name, id as u32);
    }
    let n_duty = ru32(&mut r)? as usize;
    let mut duty = Vec::with_capacity(n_duty);
    for _ in 0..n_duty {
        duty.push(rf32(&mut r)?);
    }
    fac.id.duty = duty;

    // ---- 3. readout ----
    let mut readout = v6::OpReadout::default();
    readout.n_learned = ru32(&mut r)?;
    let n_ops = ru32(&mut r)? as usize;
    for i in 0..n_ops {
        let s = rstr(&mut r)?;
        readout.op_id.insert(s.clone(), i as u32);
        readout.ops.push(s);
    }
    let n_vu = ru32(&mut r)? as usize;
    readout.votes = HashMap::with_capacity(n_vu);
    readout.total = HashMap::with_capacity(n_vu);
    for _ in 0..n_vu {
        let unit = ru32(&mut r)?;
        let total = ru32(&mut r)?;
        let n = ru32(&mut r)? as usize;
        let mut m = HashMap::with_capacity(n);
        for _ in 0..n {
            let op = ru32(&mut r)?;
            let c = ru32(&mut r)? as i32;
            m.insert(op, c);
        }
        readout.votes.insert(unit, m);
        readout.total.insert(unit, total);
    }

    Ok((brain, fac, readout))
}
