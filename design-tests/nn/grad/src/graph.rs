// TESTING / DESIGN ONLY — ava_grad::graph
// Define-by-run reverse-mode automatic differentiation.
//
// Usage pattern (borrow-friendly):
//   let mut g = Graph::new(&params);     // the graph borrows the parameter store read-only
//   let x = g.param(id); ...ops...       // every op runs immediately and records one tape node
//   let grads = g.backward(loss);        // gradients for every parameter the loss depends on
//   params.accumulate(&grads);           // after the graph's last use, mutate the store
//   opt.step(&mut params, lr); params.zero_grads();
//
// Notes:
// * Var is a Copy handle (an index into the tape). Nodes are created in topological order, so backward
//   walks the tape in reverse once. A Var may be used any number of times; contributions add up.
// * g.param(id) is memoised: asking for the same parameter twice returns the same Var, and the parameter's
//   value is read straight from the store (never copied into the tape).
// * Nodes that do not depend on any parameter (inputs and functions of inputs only) are marked as not
//   needing gradients and are skipped in backward.
// * backward(v) seeds dL/dv = 1 for every element of v, i.e. a non-scalar v behaves like sum(v).
// * Scalars (sum, mean, cross_entropy) have shape [1].
// * Every heavy kernel (matmul, elementwise forward/backward, norms, softmax, cross-entropy, kWTA, the LIF
//   scan, axis swaps) uses up to `threads` worker threads once the work is large enough; the split is by
//   rows or element ranges and reductions are combined in a fixed order, so results are bit-identical for
//   any thread count.

use crate::brain::{self, LifOpts, SpikeFn};
use crate::matmul::{self, axpy_par, gemm, gemm_batched, par_chunks, par_rows, par_rows2, ELEM_CHEAP, ELEM_COSTLY};
use crate::params::{Grads, ParamId, Params};
use crate::tensor::{numel_of, Tensor};

/// Target value that cross_entropy (and accuracy) skip, e.g. for padding or unpredictable positions.
pub const IGNORE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Var(usize);

impl Var {
    pub fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug)]
enum Mm {
    /// a [rows, k] (leading dims flattened) × b [k, n] (or b [n, k] transposed)
    Flat { rows: usize, k: usize, n: usize },
    /// a [batch, m, k] × b [batch, k, n] (or b [batch, n, k] transposed)
    Batched { batch: usize, m: usize, k: usize, n: usize },
}

enum Op {
    Input,
    Param(ParamId),
    MatMul { a: usize, b: usize, kind: Mm, bt: bool },
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
    Scale(usize, f32),
    AddBias(usize, usize),
    Relu(usize),
    Gelu(usize),
    Sigmoid(usize),
    Tanh(usize),
    Exp(usize),
    RmsNorm { x: usize, gain: usize, rinv: Vec<f32> },
    LayerNorm { x: usize, gain: usize, bias: usize, mean: Vec<f32>, rstd: Vec<f32> },
    Embedding { table: usize, ids: Vec<u32> },
    Reshape(usize),
    Transpose { x: usize, d1: usize, d2: usize },
    Softmax(usize),
    CrossEntropy { logits: usize, targets: Vec<u32>, lse: Vec<f32>, count: usize },
    Sum(usize),
    Mean(usize),
    Kwta { x: usize, mask: Vec<bool> },
    Lif { cur: usize, beta: usize, opts: LifOpts, v: Vec<f32> },
}

enum Val {
    Owned(Tensor),
    Param(ParamId),
}

struct Node {
    val: Val,
    op: Op,
    needs_grad: bool,
}

pub struct Graph<'p> {
    params: &'p Params,
    nodes: Vec<Node>,
    param_vars: Vec<Option<Var>>,
    threads: usize,
}

const GELU_C: f32 = 0.797_884_6; // sqrt(2/pi)
const GELU_A: f32 = 0.044_715;

#[inline]
fn gelu_fwd(x: f32) -> f32 {
    0.5 * x * (1.0 + (GELU_C * (x + GELU_A * x * x * x)).tanh())
}

#[inline]
fn gelu_grad(x: f32) -> f32 {
    let t = (GELU_C * (x + GELU_A * x * x * x)).tanh();
    0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * GELU_C * (1.0 + 3.0 * GELU_A * x * x)
}

fn axpy(dst: &mut [f32], alpha: f32, src: &[f32]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d += alpha * *s;
    }
}

/// `f(&mut dst[i], g[i], &x[i])` for every i, threaded over element ranges (each element is touched by
/// exactly one thread, so the result is independent of the thread count). The elementwise backward kernels.
fn for_each3_par<T: Sync>(dst: &mut [f32], g: &[f32], x: &[T], min_per_thread: usize, threads: usize, f: impl Fn(&mut f32, f32, &T) + Sync) {
    debug_assert_eq!(dst.len(), g.len());
    debug_assert_eq!(dst.len(), x.len());
    par_chunks(dst, min_per_thread, threads, |i0, chunk| {
        let n = chunk.len();
        for ((d, &gi), xi) in chunk.iter_mut().zip(&g[i0..i0 + n]).zip(&x[i0..i0 + n]) {
            f(d, gi, xi);
        }
    });
}

/// out[j] += Σ_{r < rows} term(r, j) for every column j. Threads split the columns; each column's sum
/// runs r = 0, 1, … on one thread, so the result is independent of the thread count.
fn col_reduce(out: &mut [f32], rows: usize, threads: usize, term: impl Fn(usize, usize) -> f32 + Sync) {
    par_rows(out, 1, rows, threads, |j0, cols| {
        for r in 0..rows {
            for (jj, o) in cols.iter_mut().enumerate() {
                *o += term(r, j0 + jj);
            }
        }
    });
}

/// Swap axes d1 and d2 of a row-major tensor with shape `shape`. Threads split the output rows
/// (one row = one (outer, j) pair, mid·n1·inner values); a pure copy, so the split never changes a bit.
fn swap_axes(x: &[f32], shape: &[usize], d1: usize, d2: usize, threads: usize) -> Vec<f32> {
    let (d1, d2) = (d1.min(d2), d1.max(d2));
    if d1 == d2 {
        return x.to_vec();
    }
    let n1 = shape[d1];
    let mid = numel_of(&shape[d1 + 1..d2]);
    let n2 = shape[d2];
    let inner = numel_of(&shape[d2 + 1..]);
    let mut out = vec![0.0f32; x.len()];
    let row_len = mid * n1 * inner;
    par_rows(&mut out, row_len, row_len, threads, |r0, chunk| {
        for (ri, orow) in chunk.chunks_mut(row_len).enumerate() {
            let r = r0 + ri;
            let (o, j) = (r / n2, r % n2);
            for m in 0..mid {
                for i in 0..n1 {
                    let src = (((o * n1 + i) * mid + m) * n2 + j) * inner;
                    let dst = (m * n1 + i) * inner;
                    orow[dst..dst + inner].copy_from_slice(&x[src..src + inner]);
                }
            }
        }
    });
    out
}

fn swapped_shape(shape: &[usize], d1: usize, d2: usize) -> Vec<usize> {
    let mut s = shape.to_vec();
    s.swap(d1, d2);
    s
}

impl<'p> Graph<'p> {
    pub fn new(params: &'p Params) -> Graph<'p> {
        Graph { params, nodes: Vec::new(), param_vars: vec![None; params.len()], threads: matmul::threads() }
    }

    /// Override the worker-thread count for this graph (clamped to 1..=32). Results do not change.
    pub fn set_threads(&mut self, n: usize) {
        self.threads = n.clamp(1, matmul::MAX_THREADS);
    }

    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Number of tape nodes so far.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn val(&self, i: usize) -> &Tensor {
        match &self.nodes[i].val {
            Val::Owned(t) => t,
            Val::Param(id) => self.params.value(*id),
        }
    }

    pub fn value(&self, v: Var) -> &Tensor {
        self.val(v.0)
    }

    pub fn shape(&self, v: Var) -> &[usize] {
        &self.val(v.0).shape
    }

    /// The single value of a one-element tensor.
    pub fn scalar(&self, v: Var) -> f32 {
        let t = self.val(v.0);
        assert_eq!(t.numel(), 1, "Graph::scalar: tensor has shape {:?}", t.shape);
        t.data[0]
    }

    fn push(&mut self, t: Tensor, op: Op, parents: &[usize]) -> Var {
        let needs_grad = parents.iter().any(|&p| self.nodes[p].needs_grad);
        self.nodes.push(Node { val: Val::Owned(t), op, needs_grad });
        Var(self.nodes.len() - 1)
    }

    /// A trainable parameter (memoised per graph).
    pub fn param(&mut self, id: ParamId) -> Var {
        assert!(id.0 < self.params.len(), "Graph::param: unknown ParamId {}", id.0);
        if let Some(v) = self.param_vars[id.0] {
            return v;
        }
        self.nodes.push(Node { val: Val::Param(id), op: Op::Param(id), needs_grad: true });
        let v = Var(self.nodes.len() - 1);
        self.param_vars[id.0] = Some(v);
        v
    }

    /// A constant input (no gradient).
    pub fn input(&mut self, t: Tensor) -> Var {
        assert_eq!(numel_of(&t.shape), t.data.len(), "Graph::input: shape/data mismatch");
        self.nodes.push(Node { val: Val::Owned(t), op: Op::Input, needs_grad: false });
        Var(self.nodes.len() - 1)
    }

    // ---------------------------------------------------------------- matmul

    /// a [.., m, k] × b [k, n] → [.., m, n] (leading dims of a flattened into rows), or batched
    /// a [B.., m, k] × b [B.., k, n] → [B.., m, n] when b has 3+ dims (same leading dims as a).
    pub fn matmul(&mut self, a: Var, b: Var) -> Var {
        self.mm(a, b, false)
    }

    /// a × bᵀ: a [.., m, k] × b [n, k] → [.., m, n], or batched a [B.., m, k] × b [B.., n, k] → [B.., m, n].
    pub fn matmul_t(&mut self, a: Var, b: Var) -> Var {
        self.mm(a, b, true)
    }

    fn mm(&mut self, a: Var, b: Var, bt: bool) -> Var {
        let name = if bt { "matmul_t" } else { "matmul" };
        let sa = self.shape(a).to_vec();
        let sb = self.shape(b).to_vec();
        assert!(!sa.is_empty() && sb.len() >= 2, "{name}: need a.ndim ≥ 1 and b.ndim ≥ 2, got {sa:?} and {sb:?}");
        let (kind, out_shape) = if sb.len() == 2 {
            let (k, n) = if bt { (sb[1], sb[0]) } else { (sb[0], sb[1]) };
            assert_eq!(*sa.last().unwrap(), k, "{name}: inner dims differ: {sa:?} vs {sb:?}");
            let rows = numel_of(&sa[..sa.len() - 1]);
            let mut os = sa[..sa.len() - 1].to_vec();
            os.push(n);
            (Mm::Flat { rows, k, n }, os)
        } else {
            let r = sa.len();
            assert_eq!(r, sb.len(), "{name}: batched operands need the same rank: {sa:?} vs {sb:?}");
            assert_eq!(sa[..r - 2], sb[..r - 2], "{name}: batch dims differ: {sa:?} vs {sb:?}");
            let batch = numel_of(&sa[..r - 2]);
            let (m, k) = (sa[r - 2], sa[r - 1]);
            let (kb, n) = if bt { (sb[r - 1], sb[r - 2]) } else { (sb[r - 2], sb[r - 1]) };
            assert_eq!(k, kb, "{name}: inner dims differ: {sa:?} vs {sb:?}");
            let mut os = sa[..r - 2].to_vec();
            os.push(m);
            os.push(n);
            (Mm::Batched { batch, m, k, n }, os)
        };
        let mut out = vec![0.0f32; numel_of(&out_shape)];
        {
            let av = &self.val(a.0).data;
            let bv = &self.val(b.0).data;
            match kind {
                Mm::Flat { rows, k, n } => gemm(rows, k, n, av, false, bv, bt, &mut out, self.threads),
                Mm::Batched { batch, m, k, n } => gemm_batched(batch, m, k, n, av, false, bv, bt, &mut out, self.threads),
            }
        }
        self.push(Tensor::from_vec(&out_shape, out), Op::MatMul { a: a.0, b: b.0, kind, bt }, &[a.0, b.0])
    }

    // ---------------------------------------------------------------- elementwise

    /// Elementwise y = f(x), threaded over element ranges once there are `min_per_thread` elements per thread.
    fn unary(&mut self, x: Var, min_per_thread: usize, f: impl Fn(f32) -> f32 + Sync, op: Op) -> Var {
        let t = {
            let xv = self.val(x.0);
            let xd = &xv.data;
            let mut out = vec![0.0f32; xd.len()];
            par_chunks(&mut out, min_per_thread, self.threads, |i0, chunk| {
                let n = chunk.len();
                for (o, &v) in chunk.iter_mut().zip(&xd[i0..i0 + n]) {
                    *o = f(v);
                }
            });
            Tensor { shape: xv.shape.clone(), data: out }
        };
        self.push(t, op, &[x.0])
    }

    fn binary(&mut self, a: Var, b: Var, name: &str, f: impl Fn(f32, f32) -> f32 + Sync, op: Op) -> Var {
        let t = {
            let (av, bv) = (self.val(a.0), self.val(b.0));
            assert_eq!(av.shape, bv.shape, "{name}: shapes differ");
            let (ad, bd) = (&av.data, &bv.data);
            let mut out = vec![0.0f32; ad.len()];
            par_chunks(&mut out, ELEM_CHEAP, self.threads, |i0, chunk| {
                let n = chunk.len();
                for ((o, &x), &y) in chunk.iter_mut().zip(&ad[i0..i0 + n]).zip(&bd[i0..i0 + n]) {
                    *o = f(x, y);
                }
            });
            Tensor { shape: av.shape.clone(), data: out }
        };
        self.push(t, op, &[a.0, b.0])
    }

    pub fn add(&mut self, a: Var, b: Var) -> Var {
        self.binary(a, b, "add", |x, y| x + y, Op::Add(a.0, b.0))
    }

    pub fn sub(&mut self, a: Var, b: Var) -> Var {
        self.binary(a, b, "sub", |x, y| x - y, Op::Sub(a.0, b.0))
    }

    pub fn mul(&mut self, a: Var, b: Var) -> Var {
        self.binary(a, b, "mul", |x, y| x * y, Op::Mul(a.0, b.0))
    }

    pub fn scale(&mut self, x: Var, s: f32) -> Var {
        self.unary(x, ELEM_CHEAP, |v| v * s, Op::Scale(x.0, s))
    }

    /// x [.., d] + b [d] (b broadcast over every row).
    pub fn add_bias(&mut self, x: Var, b: Var) -> Var {
        let t = {
            let (xv, bv) = (self.val(x.0), self.val(b.0));
            let d = xv.last_dim();
            assert_eq!(bv.shape, vec![d], "add_bias: bias shape {:?} vs last dim {d}", bv.shape);
            let (xd, bd) = (&xv.data, &bv.data);
            let mut data = vec![0.0f32; xd.len()];
            par_rows(&mut data, d, d, self.threads, |r0, rows| {
                for (ri, row) in rows.chunks_mut(d).enumerate() {
                    let xr = &xd[(r0 + ri) * d..(r0 + ri + 1) * d];
                    for ((o, &xx), &bb) in row.iter_mut().zip(xr).zip(bd) {
                        *o = xx + bb;
                    }
                }
            });
            Tensor { shape: xv.shape.clone(), data }
        };
        self.push(t, Op::AddBias(x.0, b.0), &[x.0, b.0])
    }

    pub fn relu(&mut self, x: Var) -> Var {
        self.unary(x, ELEM_CHEAP, |v| v.max(0.0), Op::Relu(x.0))
    }

    /// GELU, tanh approximation: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
    pub fn gelu(&mut self, x: Var) -> Var {
        self.unary(x, ELEM_COSTLY, gelu_fwd, Op::Gelu(x.0))
    }

    pub fn sigmoid(&mut self, x: Var) -> Var {
        self.unary(x, ELEM_COSTLY, brain::sigmoid, Op::Sigmoid(x.0))
    }

    pub fn tanh(&mut self, x: Var) -> Var {
        self.unary(x, ELEM_COSTLY, |v| v.tanh(), Op::Tanh(x.0))
    }

    pub fn exp(&mut self, x: Var) -> Var {
        self.unary(x, ELEM_COSTLY, |v| v.exp(), Op::Exp(x.0))
    }

    // ---------------------------------------------------------------- normalisation

    /// y = x / sqrt(mean(x²) + eps) · gain, over the last dim.
    pub fn rmsnorm(&mut self, x: Var, gain: Var, eps: f32) -> Var {
        let (t, rinv) = {
            let (xv, gv) = (self.val(x.0), self.val(gain.0));
            let d = xv.last_dim();
            assert_eq!(gv.shape, vec![d], "rmsnorm: gain shape {:?} vs last dim {d}", gv.shape);
            let rows = xv.numel() / d.max(1);
            let mut out = vec![0.0f32; xv.numel()];
            let mut rinv = vec![0.0f32; rows];
            let (xd, gd) = (&xv.data, &gv.data);
            par_rows2(&mut out, d, &mut rinv, 1, d * 4, self.threads, |r0, orows, rinvs| {
                for (i, (orow, rinv_r)) in orows.chunks_mut(d).zip(rinvs.iter_mut()).enumerate() {
                    let r = r0 + i;
                    let row = &xd[r * d..(r + 1) * d];
                    let ms = row.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / d as f64;
                    let ri = (1.0 / (ms + eps as f64).sqrt()) as f32;
                    *rinv_r = ri;
                    for j in 0..d {
                        orow[j] = row[j] * ri * gd[j];
                    }
                }
            });
            (Tensor { shape: xv.shape.clone(), data: out }, rinv)
        };
        self.push(t, Op::RmsNorm { x: x.0, gain: gain.0, rinv }, &[x.0, gain.0])
    }

    /// y = (x − mean) / sqrt(var + eps) · gain + bias, over the last dim.
    pub fn layernorm(&mut self, x: Var, gain: Var, bias: Var, eps: f32) -> Var {
        let (t, mean, rstd) = {
            let (xv, gv, bv) = (self.val(x.0), self.val(gain.0), self.val(bias.0));
            let d = xv.last_dim();
            assert_eq!(gv.shape, vec![d], "layernorm: gain shape");
            assert_eq!(bv.shape, vec![d], "layernorm: bias shape");
            let rows = xv.numel() / d.max(1);
            let mut out = vec![0.0f32; xv.numel()];
            let mut stats = vec![0.0f32; rows * 2]; // [mean, rstd] per row
            let (xd, gd, bd) = (&xv.data, &gv.data, &bv.data);
            par_rows2(&mut out, d, &mut stats, 2, d * 6, self.threads, |r0, orows, srows| {
                for (i, (orow, st)) in orows.chunks_mut(d).zip(srows.chunks_mut(2)).enumerate() {
                    let r = r0 + i;
                    let row = &xd[r * d..(r + 1) * d];
                    let mu = row.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
                    let var = row.iter().map(|&v| (v as f64 - mu) * (v as f64 - mu)).sum::<f64>() / d as f64;
                    let rs = 1.0 / (var + eps as f64).sqrt();
                    st[0] = mu as f32;
                    st[1] = rs as f32;
                    for j in 0..d {
                        orow[j] = ((row[j] as f64 - mu) * rs) as f32 * gd[j] + bd[j];
                    }
                }
            });
            let mean: Vec<f32> = stats.chunks(2).map(|s| s[0]).collect();
            let rstd: Vec<f32> = stats.chunks(2).map(|s| s[1]).collect();
            (Tensor { shape: xv.shape.clone(), data: out }, mean, rstd)
        };
        self.push(t, Op::LayerNorm { x: x.0, gain: gain.0, bias: bias.0, mean, rstd }, &[x.0, gain.0, bias.0])
    }

    // ---------------------------------------------------------------- gather / shape

    /// Rows of `table` [V, d] picked by `ids` → [ids.len(), d]. Backward scatter-adds into the table.
    pub fn embedding(&mut self, table: Var, ids: &[u32]) -> Var {
        let t = {
            let tv = self.val(table.0);
            assert_eq!(tv.ndim(), 2, "embedding: table must be [V, d], got {:?}", tv.shape);
            let (vocab, d) = (tv.shape[0], tv.shape[1]);
            let mut out = vec![0.0f32; ids.len() * d];
            for (r, &id) in ids.iter().enumerate() {
                let id = id as usize;
                assert!(id < vocab, "embedding: id {id} out of range for vocab {vocab}");
                out[r * d..(r + 1) * d].copy_from_slice(&tv.data[id * d..(id + 1) * d]);
            }
            Tensor::from_vec(&[ids.len(), d], out)
        };
        self.push(t, Op::Embedding { table: table.0, ids: ids.to_vec() }, &[table.0])
    }

    /// Same data, new shape (same number of elements).
    pub fn reshape(&mut self, x: Var, shape: &[usize]) -> Var {
        let t = {
            let xv = self.val(x.0);
            assert_eq!(numel_of(shape), xv.numel(), "reshape: {:?} → {:?} changes the element count", xv.shape, shape);
            Tensor { shape: shape.to_vec(), data: xv.data.clone() }
        };
        self.push(t, Op::Reshape(x.0), &[x.0])
    }

    /// Swap two axes (e.g. [B,T,H,D] ↔ [B,H,T,D] with d1 = 1, d2 = 2).
    pub fn transpose(&mut self, x: Var, d1: usize, d2: usize) -> Var {
        let t = {
            let xv = self.val(x.0);
            assert!(d1 < xv.ndim() && d2 < xv.ndim(), "transpose: axes {d1},{d2} out of range for {:?}", xv.shape);
            Tensor { shape: swapped_shape(&xv.shape, d1, d2), data: swap_axes(&xv.data, &xv.shape, d1, d2, self.threads) }
        };
        self.push(t, Op::Transpose { x: x.0, d1, d2 }, &[x.0])
    }

    // ---------------------------------------------------------------- softmax / loss

    /// Softmax over the last dim. With `causal`, x must be [.., Tq, Tk] with Tk ≥ Tq and entry (i, j) is
    /// masked (output 0) when j > i + (Tk − Tq); for square scores that is the usual j > i.
    pub fn softmax_last(&mut self, x: Var, causal: bool) -> Var {
        let t = {
            let xv = self.val(x.0);
            let tk = xv.last_dim();
            let (tq, offset) = if causal {
                assert!(xv.ndim() >= 2, "softmax_last(causal): need [.., Tq, Tk], got {:?}", xv.shape);
                let tq = xv.shape[xv.ndim() - 2];
                assert!(tk >= tq, "softmax_last(causal): Tk {tk} < Tq {tq}");
                (tq, tk - tq)
            } else {
                (1, 0)
            };
            let xd = &xv.data;
            let mut out = vec![0.0f32; xd.len()];
            par_rows(&mut out, tk, tk * 4, self.threads, |r0, chunk| {
                for (ri, orow) in chunk.chunks_mut(tk).enumerate() {
                    let r = r0 + ri;
                    let row = &xd[r * tk..(r + 1) * tk];
                    let allowed = if causal { (r % tq + offset + 1).min(tk) } else { tk };
                    let mx = row[..allowed].iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
                    let mut s = 0.0f64;
                    for j in 0..allowed {
                        let e = (row[j] - mx).exp();
                        orow[j] = e;
                        s += e as f64;
                    }
                    let inv = (1.0 / s) as f32;
                    for o in &mut orow[..allowed] {
                        *o *= inv;
                    }
                }
            });
            Tensor { shape: xv.shape.clone(), data: out }
        };
        self.push(t, Op::Softmax(x.0), &[x.0])
    }

    /// Mean cross-entropy of `logits` [n, V] (or [.., V], flattened to rows) against class ids; rows whose
    /// target is IGNORE are skipped (the loss is 0 if every row is ignored). Fused, stable log-softmax.
    pub fn cross_entropy(&mut self, logits: Var, targets: &[u32]) -> Var {
        let (loss, lse, count) = {
            let zv = self.val(logits.0);
            let v = zv.last_dim();
            let n = zv.numel() / v.max(1);
            assert_eq!(targets.len(), n, "cross_entropy: {} targets for {n} rows", targets.len());
            let z = &zv.data;
            let mut lse = vec![0.0f32; n];
            par_rows(&mut lse, 1, v * 2, self.threads, |r0, chunk| {
                for (i, o) in chunk.iter_mut().enumerate() {
                    let r = r0 + i;
                    if targets[r] == IGNORE {
                        continue;
                    }
                    let row = &z[r * v..(r + 1) * v];
                    let mx = row.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x));
                    let s: f64 = row.iter().map(|&x| (x - mx).exp() as f64).sum();
                    *o = mx + s.ln() as f32;
                }
            });
            let mut total = 0.0f64;
            let mut count = 0usize;
            for (r, &t) in targets.iter().enumerate() {
                if t == IGNORE {
                    continue;
                }
                assert!((t as usize) < v, "cross_entropy: target {t} out of range for {v} classes");
                total += (lse[r] - z[r * v + t as usize]) as f64;
                count += 1;
            }
            let loss = if count > 0 { total / count as f64 } else { 0.0 };
            (loss as f32, lse, count)
        };
        self.push(Tensor::scalar(loss), Op::CrossEntropy { logits: logits.0, targets: targets.to_vec(), lse, count }, &[logits.0])
    }

    pub fn sum(&mut self, x: Var) -> Var {
        let s: f64 = self.val(x.0).data.iter().map(|&v| v as f64).sum();
        self.push(Tensor::scalar(s as f32), Op::Sum(x.0), &[x.0])
    }

    pub fn mean(&mut self, x: Var) -> Var {
        let xv = self.val(x.0);
        let n = xv.numel().max(1);
        let s: f64 = xv.data.iter().map(|&v| v as f64).sum();
        self.push(Tensor::scalar((s / n as f64) as f32), Op::Mean(x.0), &[x.0])
    }

    // ---------------------------------------------------------------- brain ops

    /// k-winners-take-all over the last dim (ties → lower index). Gradient flows through winners only.
    pub fn kwta(&mut self, x: Var, k: usize) -> Var {
        let (t, mask) = {
            let xv = self.val(x.0);
            let (out, mask) = brain::kwta_forward_par(&xv.data, xv.last_dim(), k, self.threads);
            (Tensor { shape: xv.shape.clone(), data: out }, mask)
        };
        self.push(t, Op::Kwta { x: x.0, mask }, &[x.0])
    }

    /// Leaky integrate-and-fire scan over time: current [B, T, D], beta_param [D] → spikes [B, T, D].
    /// β = σ(beta_param), soft reset by subtraction, reset detached in backward (see brain.rs).
    pub fn lif_scan(&mut self, current: Var, beta_param: Var, threshold: f32, spike_fn: SpikeFn) -> Var {
        self.lif_scan_opts(current, beta_param, LifOpts::new(threshold, spike_fn))
    }

    /// lif_scan with every option explicit (e.g. detach_reset = false for the exact gradient).
    pub fn lif_scan_opts(&mut self, current: Var, beta_param: Var, opts: LifOpts) -> Var {
        let (t, v) = {
            let (cv, bv) = (self.val(current.0), self.val(beta_param.0));
            assert_eq!(cv.ndim(), 3, "lif_scan: current must be [B, T, D], got {:?}", cv.shape);
            let (b, tt, d) = (cv.shape[0], cv.shape[1], cv.shape[2]);
            assert_eq!(bv.shape, vec![d], "lif_scan: beta_param must be [{d}], got {:?}", bv.shape);
            let (spikes, v) = brain::lif_forward_par(&cv.data, &bv.data, b, tt, d, &opts, self.threads);
            (Tensor { shape: cv.shape.clone(), data: spikes }, v)
        };
        self.push(t, Op::Lif { cur: current.0, beta: beta_param.0, opts, v }, &[current.0, beta_param.0])
    }

    // ---------------------------------------------------------------- backward

    /// The gradient buffer of node `idx` (created as zeros on first use), or None if it needs no gradient.
    fn slot<'g>(&self, grads: &'g mut [Option<Vec<f32>>], idx: usize) -> Option<&'g mut Vec<f32>> {
        if !self.nodes[idx].needs_grad {
            return None;
        }
        let n = self.val(idx).numel();
        Some(grads[idx].get_or_insert_with(|| vec![0.0f32; n]))
    }

    /// Reverse pass from `loss` (seeded with ones). Returns the gradient of every parameter used.
    pub fn backward(&self, loss: Var) -> Grads {
        assert!(loss.0 < self.nodes.len(), "backward: Var not from this graph");
        let mut out = Grads::new(self.params.len());
        if !self.nodes[loss.0].needs_grad {
            return out;
        }
        let mut grads: Vec<Option<Vec<f32>>> = (0..=loss.0).map(|_| None).collect();
        grads[loss.0] = Some(vec![1.0f32; self.val(loss.0).numel()]);
        let th = self.threads;
        for i in (0..=loss.0).rev() {
            let Some(g) = grads[i].take() else { continue };
            let node = &self.nodes[i];
            if !node.needs_grad {
                continue;
            }
            match &node.op {
                Op::Input => {}
                Op::Param(id) => out.add(*id, g),
                Op::MatMul { a, b, kind, bt } => self.back_matmul(&mut grads, &g, *a, *b, *kind, *bt),
                Op::Add(a, b) => {
                    if let Some(ga) = self.slot(&mut grads, *a) {
                        axpy_par(ga, 1.0, &g, th);
                    }
                    if let Some(gb) = self.slot(&mut grads, *b) {
                        axpy_par(gb, 1.0, &g, th);
                    }
                }
                Op::Sub(a, b) => {
                    if let Some(ga) = self.slot(&mut grads, *a) {
                        axpy_par(ga, 1.0, &g, th);
                    }
                    if let Some(gb) = self.slot(&mut grads, *b) {
                        axpy_par(gb, -1.0, &g, th);
                    }
                }
                Op::Mul(a, b) => {
                    let (av, bv) = (&self.val(*a).data, &self.val(*b).data);
                    if let Some(ga) = self.slot(&mut grads, *a) {
                        for_each3_par(ga, &g, bv, ELEM_CHEAP, th, |d, gi, &y| *d += gi * y);
                    }
                    if let Some(gb) = self.slot(&mut grads, *b) {
                        for_each3_par(gb, &g, av, ELEM_CHEAP, th, |d, gi, &x| *d += gi * x);
                    }
                }
                Op::Scale(x, s) => {
                    if let Some(gx) = self.slot(&mut grads, *x) {
                        axpy_par(gx, *s, &g, th);
                    }
                }
                Op::AddBias(x, b) => {
                    if let Some(gx) = self.slot(&mut grads, *x) {
                        axpy_par(gx, 1.0, &g, th);
                    }
                    if let Some(gb) = self.slot(&mut grads, *b) {
                        let d = gb.len();
                        let rows = g.len() / d.max(1);
                        col_reduce(gb, rows, th, |r, j| g[r * d + j]);
                    }
                }
                Op::Relu(x) => {
                    let xv = &self.val(*x).data;
                    if let Some(gx) = self.slot(&mut grads, *x) {
                        for_each3_par(gx, &g, xv, ELEM_CHEAP, th, |d, gi, &v| {
                            if v > 0.0 {
                                *d += gi;
                            }
                        });
                    }
                }
                Op::Gelu(x) => {
                    let xv = &self.val(*x).data;
                    if let Some(gx) = self.slot(&mut grads, *x) {
                        for_each3_par(gx, &g, xv, ELEM_COSTLY, th, |d, gi, &v| *d += gi * gelu_grad(v));
                    }
                }
                Op::Sigmoid(x) => {
                    let yv = &self.val(i).data;
                    if let Some(gx) = self.slot(&mut grads, *x) {
                        for_each3_par(gx, &g, yv, ELEM_CHEAP, th, |d, gi, &y| *d += gi * y * (1.0 - y));
                    }
                }
                Op::Tanh(x) => {
                    let yv = &self.val(i).data;
                    if let Some(gx) = self.slot(&mut grads, *x) {
                        for_each3_par(gx, &g, yv, ELEM_CHEAP, th, |d, gi, &y| *d += gi * (1.0 - y * y));
                    }
                }
                Op::Exp(x) => {
                    let yv = &self.val(i).data;
                    if let Some(gx) = self.slot(&mut grads, *x) {
                        for_each3_par(gx, &g, yv, ELEM_CHEAP, th, |d, gi, &y| *d += gi * y);
                    }
                }
                Op::RmsNorm { x, gain, rinv } => {
                    let (xv, gv) = (&self.val(*x).data, &self.val(*gain).data);
                    let d = gv.len();
                    let rows = rinv.len();
                    if let Some(gg) = self.slot(&mut grads, *gain) {
                        col_reduce(gg, rows, th, |r, j| g[r * d + j] * xv[r * d + j] * rinv[r]);
                    }
                    if let Some(gx) = self.slot(&mut grads, *x) {
                        let g = &g;
                        par_rows(gx, d, d * 6, th, |r0, chunk| {
                            for (k, grow) in chunk.chunks_mut(d).enumerate() {
                                let r = r0 + k;
                                let ri = rinv[r];
                                let (xr, gr) = (&xv[r * d..(r + 1) * d], &g[r * d..(r + 1) * d]);
                                let mut dot = 0.0f64;
                                for j in 0..d {
                                    dot += (gr[j] * gv[j]) as f64 * (xr[j] * ri) as f64;
                                }
                                let m = (dot / d as f64) as f32;
                                for j in 0..d {
                                    let xh = xr[j] * ri;
                                    grow[j] += ri * (gr[j] * gv[j] - xh * m);
                                }
                            }
                        });
                    }
                }
                Op::LayerNorm { x, gain, bias, mean, rstd } => {
                    let (xv, gv) = (&self.val(*x).data, &self.val(*gain).data);
                    let d = gv.len();
                    let rows = mean.len();
                    if let Some(gb) = self.slot(&mut grads, *bias) {
                        col_reduce(gb, rows, th, |r, j| g[r * d + j]);
                    }
                    if let Some(gg) = self.slot(&mut grads, *gain) {
                        col_reduce(gg, rows, th, |r, j| g[r * d + j] * (xv[r * d + j] - mean[r]) * rstd[r]);
                    }
                    if let Some(gx) = self.slot(&mut grads, *x) {
                        let g = &g;
                        par_rows(gx, d, d * 8, th, |r0, chunk| {
                            for (k, grow) in chunk.chunks_mut(d).enumerate() {
                                let r = r0 + k;
                                let (mu, rs) = (mean[r], rstd[r]);
                                let (xr, gr) = (&xv[r * d..(r + 1) * d], &g[r * d..(r + 1) * d]);
                                let (mut s1, mut s2) = (0.0f64, 0.0f64);
                                for j in 0..d {
                                    let dxh = (gr[j] * gv[j]) as f64;
                                    s1 += dxh;
                                    s2 += dxh * ((xr[j] - mu) * rs) as f64;
                                }
                                let (m1, m2) = ((s1 / d as f64) as f32, (s2 / d as f64) as f32);
                                for j in 0..d {
                                    let xh = (xr[j] - mu) * rs;
                                    grow[j] += rs * (gr[j] * gv[j] - m1 - xh * m2);
                                }
                            }
                        });
                    }
                }
                Op::Embedding { table, ids } => {
                    if let Some(gt) = self.slot(&mut grads, *table) {
                        let d = if ids.is_empty() { 0 } else { g.len() / ids.len() };
                        for (r, &id) in ids.iter().enumerate() {
                            let id = id as usize;
                            axpy(&mut gt[id * d..(id + 1) * d], 1.0, &g[r * d..(r + 1) * d]);
                        }
                    }
                }
                Op::Reshape(x) => {
                    if let Some(gx) = self.slot(&mut grads, *x) {
                        axpy_par(gx, 1.0, &g, th);
                    }
                }
                Op::Transpose { x, d1, d2 } => {
                    let out_shape = &self.val(i).shape;
                    let back = swap_axes(&g, out_shape, *d1, *d2, th);
                    if let Some(gx) = self.slot(&mut grads, *x) {
                        axpy_par(gx, 1.0, &back, th);
                    }
                }
                Op::Softmax(x) => {
                    let yv = &self.val(i).data;
                    let tk = self.val(i).last_dim();
                    let g = &g;
                    let threads = self.threads;
                    if let Some(gx) = self.slot(&mut grads, *x) {
                        par_rows(gx, tk, tk * 3, threads, |r0, chunk| {
                            for (ri, grow) in chunk.chunks_mut(tk).enumerate() {
                                let r = r0 + ri;
                                let (yr, gr) = (&yv[r * tk..(r + 1) * tk], &g[r * tk..(r + 1) * tk]);
                                let dot: f64 = yr.iter().zip(gr).map(|(&y, &gg)| y as f64 * gg as f64).sum();
                                let dot = dot as f32;
                                for j in 0..tk {
                                    grow[j] += yr[j] * (gr[j] - dot);
                                }
                            }
                        });
                    }
                }
                Op::CrossEntropy { logits, targets, lse, count } => {
                    if *count == 0 {
                        continue;
                    }
                    let z = &self.val(*logits).data;
                    let v = self.val(*logits).last_dim();
                    let scale = g[0] / *count as f32;
                    let threads = self.threads;
                    if let Some(gz) = self.slot(&mut grads, *logits) {
                        par_rows(gz, v, v * 2, threads, |r0, chunk| {
                            for (ri, grow) in chunk.chunks_mut(v).enumerate() {
                                let r = r0 + ri;
                                let t = targets[r];
                                if t == IGNORE {
                                    continue;
                                }
                                let (zr, l) = (&z[r * v..(r + 1) * v], lse[r]);
                                for j in 0..v {
                                    grow[j] += scale * (zr[j] - l).exp();
                                }
                                grow[t as usize] -= scale;
                            }
                        });
                    }
                }
                Op::Sum(x) => {
                    if let Some(gx) = self.slot(&mut grads, *x) {
                        let s = g[0];
                        par_chunks(gx, ELEM_CHEAP, th, |_, chunk| chunk.iter_mut().for_each(|d| *d += s));
                    }
                }
                Op::Mean(x) => {
                    if let Some(gx) = self.slot(&mut grads, *x) {
                        let s = g[0] / gx.len().max(1) as f32;
                        par_chunks(gx, ELEM_CHEAP, th, |_, chunk| chunk.iter_mut().for_each(|d| *d += s));
                    }
                }
                Op::Kwta { x, mask } => {
                    if let Some(gx) = self.slot(&mut grads, *x) {
                        for_each3_par(gx, &g, mask, ELEM_CHEAP, th, |d, gi, &m| {
                            if m {
                                *d += gi;
                            }
                        });
                    }
                }
                Op::Lif { cur, beta, opts, v } => {
                    let s = &self.val(*cur).shape;
                    let (b, t, d) = (s[0], s[1], s[2]);
                    let bp = &self.val(*beta).data;
                    let (dcur, dbeta) = brain::lif_backward_par(&g, v, bp, b, t, d, opts, th);
                    if let Some(gc) = self.slot(&mut grads, *cur) {
                        axpy_par(gc, 1.0, &dcur, th);
                    }
                    if let Some(gb) = self.slot(&mut grads, *beta) {
                        axpy(gb, 1.0, &dbeta);
                    }
                }
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn back_matmul(&self, grads: &mut [Option<Vec<f32>>], g: &[f32], a: usize, b: usize, kind: Mm, bt: bool) {
        let av = &self.val(a).data;
        let bv = &self.val(b).data;
        let th = self.threads;
        match kind {
            Mm::Flat { rows, k, n } => {
                if let Some(ga) = self.slot(grads, a) {
                    // dA[rows,k] += dC[rows,n] · op(B)ᵀ
                    if bt {
                        gemm(rows, n, k, g, false, bv, false, ga, th); // B stored [n,k]
                    } else {
                        gemm(rows, n, k, g, false, bv, true, ga, th); // B stored [k,n]
                    }
                }
                if let Some(gb) = self.slot(grads, b) {
                    if bt {
                        // dB[n,k] += dCᵀ · A
                        gemm(n, rows, k, g, true, av, false, gb, th);
                    } else {
                        // dB[k,n] += Aᵀ · dC
                        gemm(k, rows, n, av, true, g, false, gb, th);
                    }
                }
            }
            Mm::Batched { batch, m, k, n } => {
                if let Some(ga) = self.slot(grads, a) {
                    gemm_batched(batch, m, n, k, g, false, bv, !bt, ga, th);
                }
                if let Some(gb) = self.slot(grads, b) {
                    if bt {
                        gemm_batched(batch, n, m, k, g, true, av, false, gb, th);
                    } else {
                        gemm_batched(batch, k, m, n, av, true, g, false, gb, th);
                    }
                }
            }
        }
    }
}

/// Fraction of rows whose argmax (first maximum on ties) equals the target; IGNORE rows are skipped
/// (returns 0 if every row is ignored). `logits` is [n, V] or [.., V].
pub fn accuracy(logits: &Tensor, targets: &[u32]) -> f32 {
    let v = logits.last_dim();
    let n = logits.numel() / v.max(1);
    assert_eq!(targets.len(), n, "accuracy: {} targets for {n} rows", targets.len());
    let (mut right, mut count) = (0usize, 0usize);
    for (r, &t) in targets.iter().enumerate() {
        if t == IGNORE {
            continue;
        }
        let row = &logits.data[r * v..(r + 1) * v];
        let mut best = 0;
        for j in 1..v {
            if row[j] > row[best] {
                best = j;
            }
        }
        count += 1;
        if best == t as usize {
            right += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        right as f32 / count as f32
    }
}
