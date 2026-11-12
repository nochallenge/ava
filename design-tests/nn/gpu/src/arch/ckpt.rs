// TESTING / DESIGN ONLY — v42: block recompute (activation checkpointing) for forward_arch, the general training path.
//!
//! With AVA_CKPT=1 each block's forward keeps none of its inner values: the block runs on a detached copy of its input,
//! and a custom op stands in for it in the step's graph. The op's backward runs the block again from the same input
//! (so the same values) with gradients on, and takes the gradients there. Same math; one more block forward per step;
//! the memory of one block's inner values instead of every block's.
//!
//! A custom op's backward can only return the gradients of its inputs (the block input x and, under
//! --value-residual, the first block's values v0). The block's own weights get theirs in a side store, which `merge`
//! adds into the step's GradStore. Every backward that can follow forward_arch calls `merge`; a forward that finds
//! gradients nobody merged stops with an error rather than train without them.

use super::Stats;
use crate::model::Model;
use crate::wave5::ModSig;
use candle_core::backend::{BackendDevice, BackendStorage};
use candle_core::backprop::GradStore;
use candle_core::{CpuStorage, CustomOp1, CustomOp2, Layout, Result, Shape, Storage, Tensor, TensorId, Var};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

thread_local! {
    /// The block weights' gradients of the backwards since the last `merge`.
    static SIDE: RefCell<HashMap<TensorId, Tensor>> = RefCell::new(HashMap::new());
    /// Tests: Some overrides AVA_CKPT on this thread.
    static FORCE: Cell<Option<bool>> = const { Cell::new(None) };
    /// Tests: how many block recomputes ran on this thread.
    static RUNS: Cell<usize> = const { Cell::new(0) };
}

/// AVA_CKPT=1 turns the block recompute on (off by default: it costs one more block forward per step).
pub fn on() -> bool {
    FORCE.with(|f| f.get()).unwrap_or_else(|| std::env::var("AVA_CKPT").is_ok_and(|v| v == "1"))
}

#[cfg(test)]
pub(crate) fn force(v: Option<bool>) {
    FORCE.with(|f| f.set(v));
}

#[cfg(test)]
pub(crate) fn runs() -> usize {
    RUNS.with(|r| r.get())
}

/// Add the block weights' gradients of the last backward into `grads` (and empty the side store). Does nothing when
/// the recompute is off.
pub fn merge(grads: &mut GradStore) -> Result<()> {
    let side = SIDE.with(|c| std::mem::take(&mut *c.borrow_mut()));
    for (id, g) in side {
        let g = match grads.get_id(id) {
            Some(old) => old.add(&g)?,
            None => g,
        };
        grads.insert_id(id, g);
    }
    Ok(())
}

/// Stop if a backward left block gradients that no `merge` took (training on would silently skip the blocks).
pub(super) fn check_merged() -> Result<()> {
    let n = SIDE.with(|c| c.borrow().len());
    if n > 0 {
        candle_core::bail!("block recompute: {n} weight gradients of the last backward were never merged (call ava_gpu::arch::ckpt::merge after backward)");
    }
    Ok(())
}

fn add_side(id: TensorId, g: Tensor) -> Result<()> {
    SIDE.with(|c| -> Result<()> {
        let mut c = c.borrow_mut();
        let g = match c.remove(&id) {
            Some(old) => old.add(&g)?,
            None => g,
        };
        c.insert(id, g);
        Ok(())
    })
}

/// The model behind a recorded block. The op lives in one step's graph, which is built, run backward and dropped on
/// one thread while the model lives.
struct ModelPtr(*const Model);
// SAFETY: see above; the op never leaves the thread that made it while the graph is in use.
unsafe impl Send for ModelPtr {}
unsafe impl Sync for ModelPtr {}

/// One recorded block. `pre` is its output computed without a kept graph, packed: the block output [b·t·d], then
/// (when this block set it) v0, then (A7) its halting probabilities, then its extra losses (H12, F12; scalars).
struct BlockCk {
    m: ModelPtr,
    i: usize,
    b: usize,
    t: usize,
    msig: Option<ModSig>,
    /// the model's train_fwd during the forward (the backward runs after it is reset)
    train: bool,
    v0_out: bool,
    halt: bool,
    extra: usize,
    pre: Mutex<Option<Tensor>>,
}

/// A fresh contiguous copy of a storage's view.
fn copy_out<S: BackendStorage>(s: &S, l: &Layout) -> Result<(S, Shape)>
where
    S::Device: BackendDevice<Storage = S>,
{
    let shape = l.shape().clone();
    // SAFETY: copy_strided_src writes every element just below
    let mut dst = unsafe { s.device().alloc_uninit(&shape, s.dtype())? };
    s.copy_strided_src(&mut dst, 0, l)?;
    Ok((dst, shape))
}

fn msg(s: &str) -> candle_core::Error {
    candle_core::Error::Msg(format!("block recompute: {s}"))
}

impl BlockCk {
    fn take(&self) -> Result<Tensor> {
        self.pre.lock().map_err(|_| msg("lock"))?.take().ok_or_else(|| msg("the op ran twice"))
    }

    fn cpu(&self) -> Result<(CpuStorage, Shape)> {
        let pre = self.take()?;
        let (s, l) = pre.storage_and_layout();
        let r = match &*s {
            Storage::Cpu(c) => copy_out(c, l),
            _ => Err(msg("the output is not on the CPU")),
        };
        r
    }

    fn cuda(&self) -> Result<(candle_core::CudaStorage, Shape)> {
        let pre = self.take()?;
        let (s, l) = pre.storage_and_layout();
        let r = match &*s {
            Storage::Cuda(c) => copy_out(c, l),
            _ => Err(msg("the output is not on CUDA")),
        };
        r
    }

    /// The backward: run the block again with gradients on, return the input gradients, keep the weights' in SIDE.
    fn recompute(&self, x: &Tensor, v0: Option<&Tensor>, g: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> {
        RUNS.with(|r| r.set(r.get() + 1));
        debug_mem(&format!("block {} recompute start", self.i), x);
        // SAFETY: see ModelPtr
        let m = unsafe { &*self.m.0 };
        let i = self.i;
        let xv = Var::from_tensor(&x.detach())?;
        let v0v = v0.map(|v| Var::from_tensor(&v.detach())).transpose()?;
        let mut v0s = v0v.as_ref().map(|v| v.as_tensor().clone());
        let nb = m.blocks.len();
        let mut st = Stats { spikes: vec![None; nb], kwta: vec![None; nb], rates: vec![None; nb] };
        let w5 = m.ext.as_ref().map(|e| &e.mem.w5);
        let halt_len = || w5.map_or(0, |w| w.stash.borrow().halt.len());
        let h0 = halt_len();
        let extra_len = || w5.map_or(0, |w| w.stash.borrow().extra.len());
        let e0 = extra_len();
        // A9's read counts: the forward already counted this block's reads
        let pk = m.ext.as_ref().and_then(|e| e.mem.pkm.as_ref());
        let counts = pk.map(|p| p.counts.replace(None));
        let tf = m.train_fwd.replace(self.train);
        let tr = m.track.replace(false);
        let y = m.block_arch(i, &m.blocks[i], m.ext.as_ref().map(|e| &e.blocks[i]), xv.as_tensor(), self.b, self.t, &mut None, &mut None, &mut v0s, &mut st, self.msig.as_ref());
        m.train_fwd.set(tf);
        m.track.set(tr);
        if let (Some(p), Some(c)) = (pk, counts) {
            *p.counts.borrow_mut() = c;
        }
        let y = y?;
        let mut pieces = vec![y.flatten_all()?];
        if self.v0_out {
            pieces.push(v0s.as_ref().ok_or_else(|| msg("the block did not set v0 again"))?.flatten_all()?);
        }
        let pushed = halt_len() > h0;
        if pushed != self.halt {
            return Err(msg("the halting probabilities differ between the forward and the recompute"));
        }
        if pushed {
            let p = w5.and_then(|w| w.stash.borrow_mut().halt.pop()).ok_or_else(|| msg("halt stash"))?;
            pieces.push(p.flatten_all()?);
        }
        let ne = extra_len() - e0;
        if ne != self.extra {
            return Err(msg("the extra losses differ between the forward and the recompute"));
        }
        if let Some(w) = w5.filter(|_| ne > 0) {
            let mut s = w.stash.borrow_mut();
            let at = s.extra.len() - ne;
            pieces.extend(s.extra.drain(at..).map(|l| l.reshape(1)).collect::<Result<Vec<_>>>()?);
        }
        let packed = Tensor::cat(&pieces, 0)?;
        let mut grads = packed.mul(&g.detach())?.sum_all()?.backward()?;
        // detached: candle leaves the leaves' gradients linked to the values they were computed from, and a kept link
        // would keep this block's whole recompute alive until the step ends
        let dx = grads.remove(xv.as_tensor()).map(|g| g.detach());
        let dv0 = v0v.as_ref().and_then(|v| grads.remove(v.as_tensor())).map(|g| g.detach());
        let ids: Vec<TensorId> = grads.get_ids().copied().collect();
        for id in ids {
            if let Some(gr) = grads.get_id(id) {
                add_side(id, gr.detach())?;
            }
        }
        drop(grads);
        debug_mem(&format!("block {} recompute end", self.i), x);
        Ok((dx, dv0))
    }
}

impl CustomOp1 for BlockCk {
    fn name(&self) -> &'static str {
        "block-recompute"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        self.cpu()
    }
    fn cuda_fwd(&self, _: &candle_core::CudaStorage, _: &Layout) -> Result<(candle_core::CudaStorage, Shape)> {
        self.cuda()
    }
    fn bwd(&self, x: &Tensor, _: &Tensor, g: &Tensor) -> Result<Option<Tensor>> {
        Ok(self.recompute(x, None, g)?.0)
    }
}

impl CustomOp2 for BlockCk {
    fn name(&self) -> &'static str {
        "block-recompute"
    }
    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        self.cpu()
    }
    fn cuda_fwd(&self, _: &candle_core::CudaStorage, _: &Layout, _: &candle_core::CudaStorage, _: &Layout) -> Result<(candle_core::CudaStorage, Shape)> {
        self.cuda()
    }
    fn bwd(&self, x: &Tensor, v0: &Tensor, _: &Tensor, g: &Tensor) -> Result<(Option<Tensor>, Option<Tensor>)> {
        self.recompute(x, Some(v0), g)
    }
}

impl Model {
    /// Block i (no carried state) through the recompute op: the same output as block_arch, none of its inner values
    /// kept. Sets v0 and pushes A7's halting probabilities as block_arch does, both wired to the op.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn block_ck(&self, i: usize, x: &Tensor, b: usize, t: usize, v0: &mut Option<Tensor>, st: &mut Stats, msig: Option<&ModSig>) -> Result<Tensor> {
        let d = self.cfg.d;
        let w5 = self.ext.as_ref().map(|e| &e.mem.w5);
        let halt_len = || w5.map_or(0, |w| w.stash.borrow().halt.len());
        let h0 = halt_len();
        let extra_len = || w5.map_or(0, |w| w.stash.borrow().extra.len());
        let e0 = extra_len();
        debug_mem(&format!("block {i} fwd start"), x);
        let v0_in = v0.clone();
        let mut v0s = v0_in.as_ref().map(|v| v.detach());
        let bx = self.ext.as_ref().map(|e| &e.blocks[i]);
        let y = self.block_arch(i, &self.blocks[i], bx, &x.detach(), b, t, &mut None, &mut None, &mut v0s, st, msig)?;
        let mut pieces = vec![y.flatten_all()?];
        drop(y);
        let v0_dims = match (&v0_in, v0s) {
            (None, Some(v)) => {
                let dims = v.dims().to_vec();
                pieces.push(v.flatten_all()?);
                Some(dims)
            }
            _ => None,
        };
        let halt_dims = if halt_len() > h0 {
            let p = w5.and_then(|w| w.stash.borrow_mut().halt.pop()).ok_or_else(|| msg("halt stash"))?;
            let dims = p.dims().to_vec();
            pieces.push(p.flatten_all()?);
            Some(dims)
        } else {
            None
        };
        let ne = extra_len() - e0;
        if let Some(w) = w5.filter(|_| ne > 0) {
            let mut s = w.stash.borrow_mut();
            let at = s.extra.len() - ne;
            pieces.extend(s.extra.drain(at..).map(|l| l.reshape(1)).collect::<Result<Vec<_>>>()?);
        }
        let pre = Tensor::cat(&pieces, 0)?.detach();
        drop(pieces);
        debug_mem(&format!("block {i} fwd end"), x);
        let op = BlockCk {
            m: ModelPtr(self as *const Model),
            i,
            b,
            t,
            msig: msig.cloned(),
            train: self.train_fwd.get(),
            v0_out: v0_dims.is_some(),
            halt: halt_dims.is_some(),
            extra: ne,
            pre: Mutex::new(Some(pre)),
        };
        let packed = match &v0_in {
            Some(v) => {
                let op: Arc<Box<dyn CustomOp2 + Send + Sync>> = Arc::new(Box::new(op));
                x.apply_op2_arc(v, op)?
            }
            None => {
                let op: Arc<Box<dyn CustomOp1 + Send + Sync>> = Arc::new(Box::new(op));
                x.apply_op1_arc(op)?
            }
        };
        let nd = b * t * d;
        let out = packed.narrow(0, 0, nd)?.reshape((b * t, d))?;
        let mut off = nd;
        if let Some(dims) = v0_dims {
            let k: usize = dims.iter().product();
            *v0 = Some(packed.narrow(0, off, k)?.reshape(dims)?);
            off += k;
        }
        if let (Some(dims), Some(w)) = (halt_dims, w5) {
            let k: usize = dims.iter().product();
            w.stash.borrow_mut().halt.push(packed.narrow(0, off, k)?.reshape(dims)?);
            off += k;
        }
        if let Some(w) = w5.filter(|_| ne > 0) {
            for j in 0..ne {
                w.stash.borrow_mut().extra.push(packed.narrow(0, off + j, 1)?.squeeze(0)?);
            }
        }
        Ok(out)
    }
}

/// AVA_CKPT_DEBUG=1: the GPU memory in use (after the queued work finishes and the pool gives back what is free).
pub(crate) fn debug_mem(tag: &str, t: &Tensor) {
    if std::env::var("AVA_CKPT_DEBUG").is_err() {
        return;
    }
    if let candle_core::Device::Cuda(c) = t.device() {
        let s = c.cuda_stream();
        let _ = s.synchronize();
        if let Ok((free, total)) = s.context().mem_get_info() {
            eprintln!("ckpt {tag}: {} MiB in use", (total - free) >> 20);
        }
    }
}
