// TESTING / DESIGN ONLY — ava_gpu::mem_gen: the general fast-weight memory as one fused op.
//
// arch::delta_rec / delta_chunk run the memory with P2 (per-channel gates, a separate erase key), P5 (momentum) or
// P1's head size 64 as many small candle ops, ~10x slower than D's fused delta kernel (fused::delta, head size 32,
// per-head gates only). `mem_gen` is the same rule as delta_rec in one kernel per direction (memg_* in src/lif.cu):
//     r = e^T F,  U = (b*k) v^T - (alpha*b*e) r^T,  S <- M S + U (P5),  F <- Diag(alpha) F + S (or + U),  y = q^T F
// for head sizes 32 and 64, per-head or per-channel gates, with or without momentum; no carried state (O1 keeps the
// candle path). Forward: one block of hs threads per head, column j of F in thread j's registers. Backward: by hand,
// with the forward's state checkpointed every 16 steps and recomputed chunk by chunk. The CPU path runs the same rule
// in plain loops (for the tests). Off switch: AVA_MEM_REF=1 goes back to delta_chunk.
// v45 W1 (--gates kda, `mem_gen_kda`): the same kernels with KDA's keys: read key alpha*e, erase key b*e (e = k),
//     r = (alpha*e)^T F,  U = (b*k) v^T - (b*e) r^T,  then as above (the fade first, then erase and write).

use candle_core::{CpuStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};

use crate::lif::cuda_slice;

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/lif.ptx"));

/// v52 fix_S: the bound on every element of the memory state, F <- clamp(Diag(alpha) F + S, -cap, cap) (lif.cu mg_cap;
/// arch::delta_rec and kda_rec, and the CPU loops here, do the same). Why: with momentum (P5, M = 0.5) and a write
/// strength b up to 2 the rule is not stable when the keys keep changing (the heavy-ball iteration grows from b ~ 0.6 at
/// M = 0.5; b < 2 is the stable range only without momentum). On a repeating text (an ASCII table, B-s52's step 1948)
/// one head's F grew ~1.4x per token past f32, and the NaN reached the hebb gates' gradients (runs/v52/fix_S.md). The
/// healthy heads stay far below the bound (block 2 there: the other 23 (row, head) pairs peak at 20 to 1110), so below it
/// every value is the old one; a runaway head stops at it, its gradient zero where the bound holds. Not in delta_chunk
/// or kda_chunk (closed forms per chunk; see delta_chunk).
pub const MEM_CAP: f32 = 1e5;

/// The bound new threads start with: MEM_CAP, or AVA_MEM_CAP when set to a positive number (experiments; "inf" = no bound,
/// the old rule's bits).
pub fn default_cap() -> f32 {
    static C: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *C.get_or_init(|| std::env::var("AVA_MEM_CAP").ok().and_then(|v| v.parse::<f32>().ok()).filter(|c| *c > 0.0).unwrap_or(MEM_CAP))
}

thread_local! {
    static CAP: std::cell::Cell<f32> = std::cell::Cell::new(default_cap());
}

/// The bound in force on this thread (default_cap() unless `with_cap` set another).
pub fn cap() -> f32 {
    CAP.with(|c| c.get())
}

/// Runs `f` with the bound `c` on this thread (the tests: a small bound makes the bounded paths run on ordinary inputs).
pub fn with_cap<R>(c: f32, f: impl FnOnce() -> R) -> R {
    let old = CAP.with(|x| x.replace(c));
    let r = f();
    CAP.with(|x| x.set(old));
    r
}

/// v52 fix_S: how many fused backward scans found the bound holding some element (fwin_ops::memg_bwd_k; those run on
/// memg_bwd4, which zeroes the gradient there). For the logs: 0 in normal training.
pub static SAT_EVENTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// The scan backward's checkpoint interval (C in lif.cu's memg_bwd; MG_C = 16 there when 0 is passed). v47: a shorter
/// chunk (8 at head size 64: +12% tok/s for WIN + SLOW at d384 L8, runs/v47/mgc.tsv) is faster there; AVA_MG_C
/// overrides it. The values do not depend on it (the saved states are exact copies).
pub fn ck_len(hs: usize) -> usize {
    static ENV: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    if let Some(c) = *ENV.get_or_init(|| std::env::var("AVA_MG_C").ok().and_then(|v| v.parse().ok()).filter(|&c| c >= 1)) {
        return c;
    }
    if hs == 64 { 8 } else { 16 }
}

/// v47 Amendment 3: the scan backward with each column's rows split over 4 threads (memg_bwd4, 4 HS threads per head)
/// at head size 64, by default only with momentum (where it wins: WIN + SLOW d384 57k against 53k tok/s; without
/// momentum it loses, runs/v47/bench3.tsv). AVA_MG4=0: never; AVA_MG4=1: at head size 64 with or without momentum.
pub fn bwd4(hs: usize, mom: bool) -> bool {
    static MODE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let mode = *MODE.get_or_init(|| std::env::var("AVA_MG4").ok().map(|v| v == "1"));
    hs == 64 && mode.unwrap_or(mom)
}

/// v49 (runs/v47 Amendment 6): the scan split in time (lif.cu memgk_*: memg_fwd's and memg_bwd4's bits, many blocks per
/// head) on the fused WIN step (fwin_ops). AVA_MGK=0: the v47 kernels; AVA_MGK=1: at head size 32 or 64; unset (default)
/// = head size 64 with channel gates at T >= 1024 (Amendment 6 results). mem_gen's own op (the autograd path, the G checks'
/// reference) keeps the v47 kernels.
pub fn mgk(hs: usize, t: usize, chan: bool) -> bool {
    static MODE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    let mode = *MODE.get_or_init(|| std::env::var("AVA_MGK").ok().map(|v| v != "0"));
    match mode {
        Some(true) => supports(hs),
        Some(false) => false,
        // v49 (Amendment 6 results, runs/v47/mgk.log + bench6.tsv): K1-K5 pass. Default on where it was checked
        // digit for digit AND is faster: head size 64, channel gates (Ava 0.4's recipe), T >= 1024 (ctx 2048 +18%,
        // ctx 8192 2.0x; ctx 256 is 5% slower). Head gates stay off: WIN alone ended step 29 at 6.6518 vs 6.6507.
        None => hs == 64 && chan && t >= 1024,
    }
}

/// v49: the item length of the split scan (steps per (head, item) pair; a multiple of ck_len). AVA_MGK_L overrides
/// (rounded up to a multiple of ck_len). Speed and memory only: the values do not depend on it.
pub fn mgk_len(hs: usize) -> usize {
    static ENV: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let c = ck_len(hs);
    match *ENV.get_or_init(|| std::env::var("AVA_MGK_L").ok().and_then(|v| v.parse().ok()).filter(|&l| l >= 1)) {
        Some(l) => l.div_ceil(c) * c,
        None => c,
    }
}

/// v49: blocks per head in the split scan's sequential walks (memgk_fwd, memgk_scan), each with hs / g columns (the
/// columns are independent recurrences: the same ops per column, so the same bits). AVA_MGK_G = 1, 2 or 4; default 4.
pub fn mgk_groups() -> usize {
    static ENV: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var("AVA_MGK_G").ok().and_then(|v| v.parse().ok()).filter(|g| [1, 2, 4].contains(g)).unwrap_or(4))
}

/// The fused memory is on unless AVA_MEM_REF=1.
pub fn on() -> bool {
    std::env::var("AVA_MEM_REF").map_or(true, |v| v != "1")
}

/// Whether mem_gen runs this shape: head size 32 or 64 (no carried state: the caller checks).
pub fn supports(hs: usize) -> bool {
    hs == 32 || hs == 64
}

/// The memory's read-out y [B, nh, T, hs] for q, k, e, v [B, nh, T, hs] (q, k, e unit length), log alpha and the
/// write strength b [B, nh, T, g] (g = 1 per head, hs per channel), momentum m (0: none). As delta_rec from a zero state.
pub fn mem_gen(q: &Tensor, k: &Tensor, e: &Tensor, v: &Tensor, la: &Tensor, b: &Tensor, m: f64) -> Result<Tensor> {
    let (bs, nh, t, hs) = q.dims4()?;
    let g = la.dim(3)?;
    if !supports(hs) || (g != 1 && g != hs) {
        candle_core::bail!("mem_gen: head size {hs} with {g} gates is not supported");
    }
    let bn = bs * nh;
    let qke = Tensor::cat(&[q, k, e], 3)?.reshape((bn, t, 3 * hs))?;
    let vg = Tensor::cat(&[v, la, b], 3)?.reshape((bn, t, hs + 2 * g))?;
    let y = qke.apply_op2(&vg, MemGen { k: MgKind { hs, pc: g > 1, m: m as f32, kda: false, cap: cap() } })?;
    y.reshape((bs, nh, t, hs))
}

/// v45 W1 (--gates kda): KDA's rule (arch::kda_rec) from a zero state: q, k, v [B, nh, T, hs] (q, k unit length), log
/// alpha [B, nh, T, hs] per channel, the write strength b [B, nh, T, 1] per head (broadcast over the channels for the
/// kernel; autograd sums its gradient back), momentum m (0: none).
pub fn mem_gen_kda(q: &Tensor, k: &Tensor, v: &Tensor, la: &Tensor, b: &Tensor, m: f64) -> Result<Tensor> {
    let (bs, nh, t, hs) = q.dims4()?;
    if !supports(hs) || la.dim(3)? != hs || b.dim(3)? != 1 {
        candle_core::bail!("mem_gen_kda: head size {hs}, {} fades and {} write strengths per head", la.dim(3)?, b.dim(3)?);
    }
    let bn = bs * nh;
    let bc = b.broadcast_as((bs, nh, t, hs))?.contiguous()?;
    let qke = Tensor::cat(&[q, k, k], 3)?.reshape((bn, t, 3 * hs))?;
    let vg = Tensor::cat(&[v, la, &bc], 3)?.reshape((bn, t, 3 * hs))?;
    let y = qke.apply_op2(&vg, MemGen { k: MgKind { hs, pc: true, m: m as f32, kda: true, cap: cap() } })?;
    y.reshape((bs, nh, t, hs))
}

#[derive(Clone, Copy, Debug)]
struct MgKind {
    hs: usize,
    /// per-channel gates (P2)
    pc: bool,
    /// momentum (P5; 0 = none)
    m: f32,
    /// v45 W1: KDA's keys (read alpha*e, erase b*e; per-channel gates)
    kda: bool,
    /// v52 fix_S: the bound on the state's elements (`cap()` when the op was made)
    cap: f32,
}

impl MgKind {
    fn g(&self) -> usize {
        if self.pc {
            self.hs
        } else {
            1
        }
    }

    fn kernel(&self, dir: &str) -> String {
        format!("memg_{dir}_{}{}{}", self.hs, if self.kda { "k" } else if self.pc { "c" } else { "h" }, (self.m != 0.0) as u8)
    }

    fn dims(&self, l1: &Layout, l2: &Layout) -> Result<(usize, usize)> {
        let (bn, t, w) = l1.shape().dims3()?;
        let (bn2, t2, w2) = l2.shape().dims3()?;
        if w != 3 * self.hs || (bn2, t2, w2) != (bn, t, self.hs + 2 * self.g()) || t == 0 {
            candle_core::bail!("mem_gen: shapes {:?} and {:?} do not fit head size {}", l1.shape(), l2.shape(), self.hs);
        }
        Ok((bn, t))
    }
}

fn cpu_slice<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
    let v = s.as_slice::<f32>()?;
    match l.contiguous_offsets() {
        Some((a, b)) => Ok(&v[a..b]),
        None => candle_core::bail!("mem_gen: input must be contiguous"),
    }
}

/// One step's inputs for head row-block `bh`, step t: (q, k, e, v, alpha, b), each of length hs (gates repeated).
fn step_in(k: &MgKind, qke: &[f32], vg: &[f32], t_len: usize, bh: usize, t: usize) -> [Vec<f32>; 6] {
    let (hs, g) = (k.hs, k.g());
    let (lq, lg) = (3 * hs, hs + 2 * g);
    let qr = &qke[(bh * t_len + t) * lq..][..lq];
    let gr = &vg[(bh * t_len + t) * lg..][..lg];
    let gate = |off: usize, f: &dyn Fn(f32) -> f32| (0..hs).map(|i| f(gr[off + if k.pc { i } else { 0 }])).collect::<Vec<f32>>();
    [
        qr[..hs].to_vec(),
        qr[hs..2 * hs].to_vec(),
        qr[2 * hs..].to_vec(),
        gr[..hs].to_vec(),
        gate(hs, &|x| x.exp()),
        gate(hs + g, &|x| x),
    ]
}

/// One step on F and S ([i·hs + j]); returns r (the recall for e). v52 fix_S: F is held in [-cap, cap]; `held` (if
/// given) gets which elements the bound held (their gradient is zero).
fn cpu_step(k: &MgKind, f: &mut [f32], s: &mut [f32], x: &[Vec<f32>; 6]) -> Vec<f32> {
    cpu_step_h(k, f, s, x, None)
}

fn cpu_step_h(k: &MgKind, f: &mut [f32], s: &mut [f32], x: &[Vec<f32>; 6], mut held: Option<&mut [bool]>) -> Vec<f32> {
    let hs = k.hs;
    let [_, kk, e, v, a, b] = x;
    // v45 W1 (KDA): the read key alpha*e, the erase key b*e
    let r: Vec<f32> = if k.kda {
        (0..hs).map(|j| (0..hs).map(|i| a[i] * e[i] * f[i * hs + j]).sum()).collect()
    } else {
        (0..hs).map(|j| (0..hs).map(|i| e[i] * f[i * hs + j]).sum()).collect()
    };
    for i in 0..hs {
        for j in 0..hs {
            let u = if k.kda { b[i] * kk[i] * v[j] - b[i] * e[i] * r[j] } else { b[i] * kk[i] * v[j] - a[i] * b[i] * e[i] * r[j] };
            let p = i * hs + j;
            let fp = if k.m != 0.0 {
                s[p] = k.m * s[p] + u;
                a[i] * f[p] + s[p]
            } else {
                a[i] * f[p] + u
            };
            if let Some(h) = held.as_deref_mut() {
                h[p] = !(fp.abs() < k.cap);
            }
            f[p] = fp.clamp(-k.cap, k.cap);
        }
    }
    r
}

struct MemGen {
    k: MgKind,
}

impl CustomOp2 for MemGen {
    fn name(&self) -> &'static str {
        "mem-gen"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> Result<(CpuStorage, Shape)> {
        let (bn, t) = self.k.dims(l1, l2)?;
        let (qke, vg) = (cpu_slice(s1, l1)?, cpu_slice(s2, l2)?);
        let hs = self.k.hs;
        let mut y = vec![0f32; bn * t * hs];
        for bh in 0..bn {
            let (mut f, mut s) = (vec![0f32; hs * hs], vec![0f32; hs * hs]);
            for ti in 0..t {
                let x = step_in(&self.k, qke, vg, t, bh, ti);
                cpu_step(&self.k, &mut f, &mut s, &x);
                for j in 0..hs {
                    y[(bh * t + ti) * hs + j] = (0..hs).map(|i| x[0][i] * f[i * hs + j]).sum();
                }
            }
        }
        Ok((CpuStorage::F32(y), Shape::from((bn, t, hs))))
    }

    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;
        let (bn, t) = self.k.dims(l1, l2)?;
        let hs = self.k.hs;
        let dev = s1.device().clone();
        let (qke, vg) = (cuda_slice(s1, l1)?, cuda_slice(s2, l2)?);
        let y = unsafe { dev.alloc::<f32>(bn * t * hs)? };
        let func = dev.get_or_load_custom_func(&self.k.kernel("fwd"), "ava_lif", PTX)?;
        let (bi, ti, m) = (bn as i32, t as i32, self.k.m);
        let mut lb = func.builder();
        let cap = self.k.cap;
        lb.arg(&qke).arg(&vg).arg(&y).arg(&bi).arg(&ti).arg(&m).arg(&cap);
        let cfg = LaunchConfig { grid_dim: (bn as u32, 1, 1), block_dim: (hs as u32, 1, 1), shared_mem_bytes: 0 };
        unsafe { lb.launch(cfg) }.w()?;
        Ok((candle_core::CudaStorage::wrap_cuda_slice(y, dev), Shape::from((bn, t, hs))))
    }

    fn bwd(&self, qke: &Tensor, vg: &Tensor, _y: &Tensor, gy: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> {
        let (bn, t, _) = qke.dims3()?;
        let (hs, g) = (self.k.hs, self.k.g());
        let packed = qke.detach().apply_op3_no_bwd(&vg.detach(), &gy.detach().contiguous()?, &MemGenBwd { k: self.k })?;
        let n1 = bn * t * 3 * hs;
        let dqke = packed.narrow(0, 0, n1)?.reshape((bn, t, 3 * hs))?;
        let dvg = packed.narrow(0, n1, bn * t * (hs + 2 * g))?.reshape((bn, t, hs + 2 * g))?;
        Ok((Some(dqke), Some(dvg)))
    }
}

/// (qke, vg, dy) -> flat [dqke | dvg].
struct MemGenBwd {
    k: MgKind,
}

impl CustomOp3 for MemGenBwd {
    fn name(&self) -> &'static str {
        "mem-gen-bwd"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (bn, t) = self.k.dims(l1, l2)?;
        let (qke, vg, dy) = (cpu_slice(s1, l1)?, cpu_slice(s2, l2)?, cpu_slice(s3, l3)?);
        let k = &self.k;
        let (hs, g) = (k.hs, k.g());
        let (lq, lg) = (3 * hs, hs + 2 * g);
        let mom = k.m != 0.0;
        let (n1, n2) = (bn * t * lq, bn * t * lg);
        let mut out = vec![0f32; n1 + n2];
        let sq = hs * hs;
        for bh in 0..bn {
            // the states before every step
            let (mut fh, mut sh) = (vec![0f32; t * sq], vec![0f32; t * sq]);
            let (mut f, mut s) = (vec![0f32; sq], vec![0f32; sq]);
            for ti in 0..t {
                fh[ti * sq..][..sq].copy_from_slice(&f);
                sh[ti * sq..][..sq].copy_from_slice(&s);
                cpu_step(k, &mut f, &mut s, &step_in(k, qke, vg, t, bh, ti));
            }
            let (mut df, mut ds) = (vec![0f32; sq], vec![0f32; sq]);
            for ti in (0..t).rev() {
                let x = step_in(k, qke, vg, t, bh, ti);
                let [q, kk, e, v, a, b] = &x;
                let f0 = &fh[ti * sq..][..sq];
                let (mut f1, mut s1) = (f0.to_vec(), sh[ti * sq..][..sq].to_vec());
                let mut held = vec![false; sq];
                let r = cpu_step_h(k, &mut f1, &mut s1, &x, Some(&mut held));
                let dyr = &dy[(bh * t + ti) * hs..][..hs];
                for i in 0..hs {
                    for j in 0..hs {
                        df[i * hs + j] += q[i] * dyr[j];
                        if held[i * hs + j] {
                            df[i * hs + j] = 0.0; // v52 fix_S: the bound holds F here
                        }
                    }
                }
                if mom {
                    for p in 0..sq {
                        ds[p] += df[p];
                    }
                }
                let du = if mom { ds.clone() } else { df.clone() };
                let (mut r1, mut r2, mut r3, mut r4, mut r5) = (vec![0f32; hs], vec![0f32; hs], vec![0f32; hs], vec![0f32; hs], vec![0f32; hs]);
                let mut dv = vec![0f32; hs];
                let mut dr = vec![0f32; hs];
                for j in 0..hs {
                    for i in 0..hs {
                        let p = i * hs + j;
                        dv[j] += du[p] * b[i] * kk[i];
                        let ae = if k.kda { b[i] * e[i] } else { a[i] * b[i] * e[i] };
                        dr[j] -= du[p] * ae;
                    }
                }
                for i in 0..hs {
                    for j in 0..hs {
                        let p = i * hs + j;
                        r1[i] += f1[p] * dyr[j];
                        r2[i] += df[p] * f0[p];
                        r3[i] += du[p] * v[j];
                        r4[i] += du[p] * r[j];
                        r5[i] += dr[j] * f0[p];
                    }
                }
                for i in 0..hs {
                    for j in 0..hs {
                        let p = i * hs + j;
                        let er = if k.kda { a[i] * e[i] } else { e[i] };
                        df[p] = a[i] * df[p] + er * dr[j];
                        if mom {
                            ds[p] *= k.m;
                        }
                    }
                }
                let dq = &mut out[(bh * t + ti) * lq..][..lq];
                for i in 0..hs {
                    dq[i] = r1[i];
                    dq[hs + i] = r3[i] * b[i];
                    dq[2 * hs + i] = if k.kda { r5[i] * a[i] - r4[i] * b[i] } else { r5[i] - r4[i] * a[i] * b[i] };
                }
                let dg = &mut out[n1 + (bh * t + ti) * lg..][..lg];
                let (mut sa, mut sb) = (0f32, 0f32);
                for i in 0..hs {
                    dg[i] = dv[i];
                    let (dla, db) = if k.kda {
                        ((r2[i] + r5[i] * e[i]) * a[i], r3[i] * kk[i] - r4[i] * e[i])
                    } else {
                        ((r2[i] - r4[i] * b[i] * e[i]) * a[i], r3[i] * kk[i] - r4[i] * a[i] * e[i])
                    };
                    if k.pc {
                        dg[hs + i] = dla;
                        dg[2 * hs + i] = db;
                    } else {
                        sa += dla;
                        sb += db;
                    }
                }
                if !k.pc {
                    dg[hs] = sa;
                    dg[hs + 1] = sb;
                }
            }
        }
        Ok((CpuStorage::F32(out), Shape::from(n1 + n2)))
    }

    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;
        let (bn, t) = self.k.dims(l1, l2)?;
        let (hs, g) = (self.k.hs, self.k.g());
        let dev = s1.device().clone();
        let (qke, vg, dy) = (cuda_slice(s1, l1)?, cuda_slice(s2, l2)?, cuda_slice(s3, l3)?);
        let (n1, n2) = (bn * t * 3 * hs, bn * t * (hs + 2 * g));
        let out = unsafe { dev.alloc::<f32>(n1 + n2)? };
        let ns = if self.k.m != 0.0 { 2 } else { 1 };
        let ckl = ck_len(hs);
        let nck = t.div_ceil(ckl);
        let ck = unsafe { dev.alloc::<f32>(bn * nck * ns * hs * hs)? };
        let scr = unsafe { dev.alloc::<f32>(bn * ckl * ns * hs * hs)? };
        let b4 = bwd4(hs, self.k.m != 0.0);
        let func = dev.get_or_load_custom_func(&self.k.kernel(if b4 { "bwd4" } else { "bwd" }), "ava_lif", PTX)?;
        let (bi, ti, m, ci) = (bn as i32, t as i32, self.k.m, ckl as i32);
        let (dq, dg) = (out.slice(0..n1), out.slice(n1..n1 + n2));
        let mut lb = func.builder();
        let cap = self.k.cap;
        lb.arg(&qke).arg(&vg).arg(&dy).arg(&dq).arg(&dg).arg(&ck).arg(&scr).arg(&bi).arg(&ti).arg(&m).arg(&ci).arg(&cap);
        let cfg = LaunchConfig { grid_dim: (bn as u32, 1, 1), block_dim: (if b4 { 4 * hs } else { hs } as u32, 1, 1), shared_mem_bytes: 0 };
        unsafe { lb.launch(cfg) }.w()?;
        drop((dq, dg));
        Ok((candle_core::CudaStorage::wrap_cuda_slice(out, dev), Shape::from(n1 + n2)))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::arch::delta_rec;
    use candle_core::{DType, Device, Var};

    fn maxabs(a: &Tensor, b: &Tensor) -> f32 {
        a.sub(b).unwrap().abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap()
    }

    fn scale(a: &Tensor) -> f32 {
        a.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap().max(1e-6)
    }

    fn unit(x: &Tensor) -> Tensor {
        x.broadcast_div(&x.sqr().unwrap().sum_keepdim(3).unwrap().sqrt().unwrap()).unwrap()
    }

    /// mem_gen = delta_rec (outputs, and gradients to q, k, e, v, log alpha and b) for head sizes 32 and 64, per-head
    /// and per-channel gates, with and without momentum, at a length that is not a multiple of the checkpoint chunk.
    fn compare(dev: &Device, tol: f32) {
        let (bs, nh, t) = (2, 2, 37);
        let mut cases = 0;
        for hs in [32usize, 64] {
            for pc in [false, true] {
                for m in [0.0f64, 0.5] {
                    cases += 1;
                    let g = if pc { hs } else { 1 };
                    let seed = cases as u64;
                    let r = |shape: (usize, usize, usize, usize), s: f32, salt: u64| {
                        let n = shape.0 * shape.1 * shape.2 * shape.3;
                        let v: Vec<f32> = (0..n)
                            .map(|j| {
                                let x = ((j as u64 * 2654435761 + seed * 97 + salt * 7919 + 13) % 10007) as f32 / 10007.0;
                                ((x * 6.2831853).sin() * 1.7 + (x * 51.0).cos()) * s
                            })
                            .collect();
                        Tensor::from_vec(v, n, dev).unwrap().reshape(shape).unwrap()
                    };
                    let sh = (bs, nh, t, hs);
                    let gs = (bs, nh, t, g);
                    let q = Var::from_tensor(&r(sh, 1.0, 1)).unwrap();
                    let k = Var::from_tensor(&r(sh, 1.0, 2)).unwrap();
                    let e = Var::from_tensor(&r(sh, 1.0, 3)).unwrap();
                    let v = Var::from_tensor(&r(sh, 1.0, 4)).unwrap();
                    // log alpha in about (-0.5, -0.02), b in about (0.2, 1.8)
                    let la = Var::from_tensor(&r(gs, 0.1, 5).affine(1.0, -0.26).unwrap()).unwrap();
                    let b = Var::from_tensor(&r(gs, 0.3, 6).affine(1.0, 1.0).unwrap()).unwrap();
                    let w = r(sh, 1.0, 7);
                    let run = |gen: bool| {
                        let (qn, kn, en) = (unit(q.as_tensor()), unit(k.as_tensor()), unit(e.as_tensor()));
                        let y = if gen {
                            mem_gen(&qn, &kn, &en, v.as_tensor(), la.as_tensor(), b.as_tensor(), m).unwrap()
                        } else {
                            delta_rec(&qn, &kn, &en, v.as_tensor(), la.as_tensor(), b.as_tensor(), m, None).unwrap().0
                        };
                        let loss = y.mul(&w).unwrap().sum_all().unwrap();
                        let grads = loss.backward().unwrap();
                        let gr = |x: &Var| grads.get(x.as_tensor()).cloned().unwrap();
                        (y, [gr(&q), gr(&k), gr(&e), gr(&v), gr(&la), gr(&b)])
                    };
                    let (y1, g1) = run(true);
                    let (y2, g2) = run(false);
                    let tag = format!("hs={hs} per-channel={pc} m={m}");
                    let ey = maxabs(&y1, &y2) / scale(&y2);
                    assert!(ey < tol, "{tag}: y rel err {ey:.2e}");
                    for (j, name) in ["q", "k", "e", "v", "log alpha", "b"].iter().enumerate() {
                        let er = maxabs(&g1[j], &g2[j]) / scale(&g2[j]);
                        assert!(er < tol, "{tag}: grad {name} rel err {er:.2e}");
                    }
                    assert_eq!(y1.dtype(), DType::F32);
                }
            }
        }
        eprintln!("mem_gen = delta_rec over {cases} cases on {dev:?}");
    }

    #[test]
    fn mem_gen_matches_delta_rec_cpu() {
        compare(&Device::Cpu, 1e-4);
    }

    /// v45 W1: mem_gen_kda = arch::kda_rec (outputs, and gradients to q, k, v, log alpha and the per-head b) for head
    /// sizes 32 and 64, with and without momentum, at a length that is not a multiple of the checkpoint chunk.
    fn compare_kda(dev: &Device, tol: f32) {
        let (bs, nh, t) = (2, 2, 37);
        let mut cases = 0;
        for hs in [32usize, 64] {
            for m in [0.0f64, 0.5] {
                cases += 1;
                let seed = 50 + cases as u64;
                let r = |shape: (usize, usize, usize, usize), s: f32, salt: u64| {
                    let n = shape.0 * shape.1 * shape.2 * shape.3;
                    let v: Vec<f32> = (0..n)
                        .map(|j| {
                            let x = ((j as u64 * 2654435761 + seed * 97 + salt * 7919 + 13) % 10007) as f32 / 10007.0;
                            ((x * 6.2831853).sin() * 1.7 + (x * 51.0).cos()) * s
                        })
                        .collect();
                    Tensor::from_vec(v, n, dev).unwrap().reshape(shape).unwrap()
                };
                let sh = (bs, nh, t, hs);
                let q = Var::from_tensor(&r(sh, 1.0, 1)).unwrap();
                let k = Var::from_tensor(&r(sh, 1.0, 2)).unwrap();
                let v = Var::from_tensor(&r(sh, 1.0, 4)).unwrap();
                let la = Var::from_tensor(&r((bs, nh, t, hs), 0.1, 5).affine(1.0, -0.26).unwrap()).unwrap();
                let b = Var::from_tensor(&r((bs, nh, t, 1), 0.3, 6).affine(1.0, 1.0).unwrap()).unwrap();
                let w = r(sh, 1.0, 7);
                let run = |gen: bool| {
                    let (qn, kn) = (unit(q.as_tensor()), unit(k.as_tensor()));
                    let y = if gen {
                        mem_gen_kda(&qn, &kn, v.as_tensor(), la.as_tensor(), b.as_tensor(), m).unwrap()
                    } else {
                        crate::arch::kda_rec(&qn, &kn, v.as_tensor(), la.as_tensor(), b.as_tensor(), m, None).unwrap().0
                    };
                    let loss = y.mul(&w).unwrap().sum_all().unwrap();
                    let grads = loss.backward().unwrap();
                    let gr = |x: &Var| grads.get(x.as_tensor()).cloned().unwrap();
                    (y, [gr(&q), gr(&k), gr(&v), gr(&la), gr(&b)])
                };
                let (y1, g1) = run(true);
                let (y2, g2) = run(false);
                let tag = format!("kda hs={hs} m={m}");
                let ey = maxabs(&y1, &y2) / scale(&y2);
                assert!(ey < tol, "{tag}: y rel err {ey:.2e}");
                for (j, name) in ["q", "k", "v", "log alpha", "b"].iter().enumerate() {
                    let er = maxabs(&g1[j], &g2[j]) / scale(&g2[j]);
                    assert!(er < tol, "{tag}: grad {name} rel err {er:.2e}");
                }
            }
        }
        eprintln!("mem_gen_kda = kda_rec over {cases} cases on {dev:?}");
    }

    #[test]
    fn mem_gen_kda_matches_kda_rec_cpu() {
        compare_kda(&Device::Cpu, 1e-4);
    }

    /// The KDA CUDA kernels against kda_rec on the GPU (tiny; run with --ignored).
    #[test]
    #[ignore]
    fn mem_gen_kda_matches_kda_rec_gpu() {
        let dev = Device::new_cuda(0).expect("no CUDA device");
        compare_kda(&dev, 1e-4);
    }

    /// The CUDA kernels against delta_rec on the GPU (run only when the GPU is free: --ignored).
    #[test]
    #[ignore]
    fn mem_gen_matches_delta_rec_gpu() {
        let dev = Device::new_cuda(0).expect("no CUDA device");
        compare(&dev, 1e-4);
    }

    // ---------------------------------------------------------------------------------------- v52 fix_S: the bound

    /// The share of (head, step, element) values the bound held, for the op's inputs (the CPU loops).
    fn held_share(k: &MgKind, qke: &[f32], vg: &[f32], bn: usize, t: usize) -> f64 {
        let sq = k.hs * k.hs;
        let mut n = 0usize;
        for bh in 0..bn {
            let (mut f, mut s, mut h) = (vec![0f32; sq], vec![0f32; sq], vec![false; sq]);
            for ti in 0..t {
                cpu_step_h(k, &mut f, &mut s, &step_in(k, qke, vg, t, bh, ti), Some(&mut h));
                n += h.iter().filter(|&&x| x).count();
            }
        }
        n as f64 / (bn * t * sq) as f64
    }

    /// v52 fix_S: with a bound small enough to hold a good share of the state (printed), mem_gen (its hand backward
    /// zeroes the gradient where the bound holds) still matches delta_rec / kda_rec (candle's autograd through clamp)
    /// at compare's bars; and the bound changes y there (so it was in force).
    fn compare_at_the_bound(dev: &Device) {
        let cap = 0.6f32;
        // how much the bound holds on inputs like compare's (per-channel gates, momentum 0.5)
        let (bs, nh, t, hs) = (2usize, 2usize, 37usize, 64usize);
        let r = |n: usize, s: f32, salt: u64| -> Vec<f32> {
            (0..n)
                .map(|j| {
                    let x = ((j as u64 * 2654435761 + 7 * 97 + salt * 7919 + 13) % 10007) as f32 / 10007.0;
                    ((x * 6.2831853).sin() * 1.7 + (x * 51.0).cos()) * s
                })
                .collect()
        };
        let n = bs * nh * t * hs;
        let sh = (bs, nh, t, hs);
        let cpu = Device::Cpu;
        let unitt = |v: Vec<f32>| unit(&Tensor::from_vec(v, sh, &cpu).unwrap());
        let (q, k, e) = (unitt(r(n, 1.0, 1)), unitt(r(n, 1.0, 2)), unitt(r(n, 1.0, 3)));
        let v = Tensor::from_vec(r(n, 1.0, 4), sh, &cpu).unwrap();
        let la = Tensor::from_vec(r(n, 0.1, 5), sh, &cpu).unwrap().affine(1.0, -0.26).unwrap();
        let b = Tensor::from_vec(r(n, 0.3, 6), sh, &cpu).unwrap().affine(1.0, 1.0).unwrap();
        let bn = bs * nh;
        let flat = |x: Tensor| x.reshape((bn, t, 3 * hs)).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let qke = flat(Tensor::cat(&[&q, &k, &e], 3).unwrap());
        let vg = flat(Tensor::cat(&[&v, &la, &b], 3).unwrap());
        let kind = MgKind { hs, pc: true, m: 0.5, kda: false, cap };
        let share = held_share(&kind, &qke, &vg, bn, t);
        eprintln!("bound {cap}: holds {:.1}% of the state's values", 100.0 * share);
        assert!(share > 0.05 && share < 0.95, "the bound should hold some of the values, not all: {share}");
        let y_inf = with_cap(f32::INFINITY, || mem_gen(&q, &k, &e, &v, &la, &b, 0.5).unwrap());
        let y_cap = with_cap(cap, || mem_gen(&q, &k, &e, &v, &la, &b, 0.5).unwrap());
        let dy = maxabs(&y_inf, &y_cap) / scale(&y_inf);
        eprintln!("bound {cap}: y moves by {dy:.3} (relative to max |y|)");
        assert!(dy > 1e-2, "the bound must change y here: {dy}");
        with_cap(cap, || compare(dev, 1e-4));
        with_cap(cap, || compare_kda(dev, 1e-4));
    }

    #[test]
    fn mem_gen_matches_delta_rec_at_the_bound_cpu() {
        compare_at_the_bound(&Device::Cpu);
    }

    /// The CUDA kernels at the bound (run when the GPU is free: --ignored).
    #[test]
    #[ignore]
    fn mem_gen_matches_delta_rec_at_the_bound_gpu() {
        compare_at_the_bound(&Device::new_cuda(0).expect("no CUDA device"));
    }

    /// v52 fix_S: a head that runs away, [1, 2, t, hs] each with per-channel gates: write strength 1.9 on every
    /// channel, alpha ~1, keys that cycle through 3 fixed directions with the erase key = the key (with momentum 0.5:
    /// mom_bound's heavy-ball iteration, which grows for b >= 0.6), random unit q and small v.
    pub(crate) fn runaway(dev: &Device, t: usize, hs: usize) -> [Tensor; 6] {
        let nh = 2;
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        let mut rnd = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let unitv = |v: Vec<f32>| {
            let n = v.iter().map(|a| a * a).sum::<f32>().sqrt();
            v.into_iter().map(|a| a / n).collect::<Vec<f32>>()
        };
        let (mut q, mut k, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..nh {
            let keys: Vec<Vec<f32>> = (0..3).map(|_| unitv((0..hs).map(|_| rnd()).collect())).collect();
            for ti in 0..t {
                q.extend(unitv((0..hs).map(|_| rnd()).collect()));
                k.extend(keys[ti % 3].iter().copied());
                v.extend((0..hs).map(|_| 0.5 * rnd()));
            }
        }
        let sh = (1, nh, t, hs);
        let mk = |d: Vec<f32>| Tensor::from_vec(d, sh, dev).unwrap();
        let k = mk(k);
        [mk(q), k.clone(), k, mk(v), Tensor::full(-1e-4f32, sh, dev).unwrap(), Tensor::full(1.9f32, sh, dev).unwrap()]
    }

    fn finite(x: &Tensor) -> bool {
        x.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().all(|v| v.is_finite())
    }

    /// v52 fix_S, the failing condition in small: without the bound (cap = inf: the old rule, the same bits) the
    /// runaway head overflows f32 in both mem_gen and delta_rec; with it every output and gradient is finite and
    /// |y| <= sqrt(hs) cap (|q| = 1, |F_ij| <= cap).
    fn runaway_check(dev: &Device) {
        let (t, hs) = (1024, 64);
        let [q, k, e, v, la, b] = runaway(dev, t, hs);
        for gen in [true, false] {
            let run = |c: f32| {
                with_cap(c, || {
                    let vs: Vec<Var> = [&q, &k, &e, &v, &la, &b].iter().map(|x| Var::from_tensor(x).unwrap()).collect();
                    let tt: Vec<&Tensor> = vs.iter().map(|x| x.as_tensor()).collect();
                    let y = if gen {
                        mem_gen(tt[0], tt[1], tt[2], tt[3], tt[4], tt[5], 0.5).unwrap()
                    } else {
                        delta_rec(tt[0], tt[1], tt[2], tt[3], tt[4], tt[5], 0.5, None).unwrap().0
                    };
                    let w = y.ones_like().unwrap().affine(1e-3, 0.0).unwrap();
                    let grads = y.mul(&w).unwrap().sum_all().unwrap().backward().unwrap();
                    let g: Vec<Tensor> = vs.iter().map(|x| grads.get(x.as_tensor()).cloned().unwrap()).collect();
                    (y, g)
                })
            };
            let who = if gen { "mem_gen" } else { "delta_rec" };
            let (y0, _) = run(f32::INFINITY);
            assert!(!finite(&y0), "{who}: without the bound the runaway head should overflow f32 (the failing condition)");
            let (y1, g1) = run(MEM_CAP);
            let my = y1.abs().unwrap().flatten_all().unwrap().max(0).unwrap().to_scalar::<f32>().unwrap();
            eprintln!("{who} on {dev:?}: no bound -> not finite; bound {MEM_CAP:e} -> max |y| {my:.3e}");
            assert!(finite(&y1), "{who}: y not finite with the bound");
            assert!(my <= (hs as f32).sqrt() * MEM_CAP * 1.0001, "{who}: |y| {my} above sqrt(hs) cap");
            for (j, g) in g1.iter().enumerate() {
                assert!(finite(g), "{who}: gradient {j} not finite with the bound");
            }
        }
    }

    #[test]
    fn memory_bound_keeps_a_runaway_head_finite_cpu() {
        runaway_check(&Device::Cpu);
    }

    /// The CUDA kernels (run when the GPU is free: --ignored).
    #[test]
    #[ignore]
    fn memory_bound_keeps_a_runaway_head_finite_gpu() {
        runaway_check(&Device::new_cuda(0).expect("no CUDA device"));
    }
}
