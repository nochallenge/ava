// TESTING / DESIGN ONLY — ava_gpu::fast4::tape: a small CPU f32 tape that mirrors candle 0.11 for the e-prop session
// step (src/fast4/eprop.rs): the same nodes, the same values, the same backward, without candle's per-op costs.
//
// What it copies from candle, so its numbers are candle's bit for bit:
// * the graph: each call makes the node candle's call makes (or none where candle returns the same tensor: a full
//   narrow, a contiguous tensor's `contiguous`, a one-tensor cat, an empty unary or binary op), with candle's layout
//   (shape, strides, offset over a shared storage: views share it, every computing op writes a new contiguous one);
// * which nodes the backward walks and in what order (`Tensor::sorted_nodes`: a depth-first walk from the loss over
//   each op's arguments in order, a node after its arguments, reversed; a node counts when a variable is under it;
//   an affine with mul 0 is not walked). Nodes no variable is under are kept as values only;
// * each op's backward as backprop.rs writes it, every gradient summed into its argument's as candle sums them
//   (zeros + the first, then + each next, in the walk's order); gradients of arguments no variable is under are not
//   computed (candle computes and drops them);
// * the CPU kernels: elementwise ops as candle's (affine v·mul + add in f32, sigmoid pieces, relu v.max(0), silu
//   v / (1 + exp(-v))), sums one value after another from 0 in the source's logical order, max keeping the first
//   largest, and every matmul through candle's call of gemm 0.19 (`MatMul::f`: the strides it reads, its batch
//   folding) and gemm's dispatch on an AVX2 + FMA CPU: its transpose rule, the horizontal kernel (m·n <= 256), gevv
//   (k <= 2), the column- and row-major gemv. gemm's blocked kernel is not emulated: a product that would take it
//   fails (the step then refuses; the tests would show it).
// Checked through the e-prop session tests (fast4/eprop_tests.rs: logits, State, gradients, changes and traces bit
// for bit against the State path) and the structure check against candle's own `sorted_nodes`.

use candle_core::{bail, Result};
use std::rc::Rc;

pub(crate) type T = usize;

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Bin {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub(crate) enum Un {
    Exp,
    Neg,
    Recip,
    Log,
    Sqr,
    Relu,
    Abs,
    Silu,
}

#[derive(Clone, Debug)]
enum Op {
    Leaf,
    Var,
    Bin(Bin, T, T),
    Un(Un, T),
    Affine(T, f64),
    Powf(T, f64),
    Bcast(T),
    Reshape(T),
    Transpose(T, usize, usize),
    Copy(T),
    Narrow(T, usize, usize, usize),
    Cat(Vec<T>, usize),
    Sum(T),
    Max(T),
    Matmul(T, T),
    /// gather along the last dim of a 2-d tensor (one index per row)
    Gather(T, Vec<usize>),
    /// e-prop's RowMm (x [b, t, din] contiguous, the rows' changes dw [b, din, dout]): its backward is RowMmT
    RowMm(T, usize, [usize; 4]),
}

pub(crate) struct Node {
    s: Src,
    pub(crate) sh: Vec<usize>,
    st: Vec<usize>,
    off: usize,
    op: Op,
    pub(crate) tr: bool,
}

fn contig_strides(sh: &[usize]) -> Vec<usize> {
    let mut st = vec![0; sh.len()];
    let mut acc = 1;
    for i in (0..sh.len()).rev() {
        st[i] = acc;
        acc *= sh[i];
    }
    st
}

fn count(sh: &[usize]) -> usize {
    sh.iter().product()
}

/// candle's `Shape::is_contiguous`: size-1 dims are ignored.
fn is_contig(sh: &[usize], st: &[usize]) -> bool {
    let mut acc = 1;
    for (&s, &d) in st.iter().zip(sh).rev() {
        if d > 1 && s != acc {
            return false;
        }
        acc *= d;
    }
    true
}


/// candle's `Layout::outer_stride_for_dim(dim).is_some()`: dims dim.. contiguous (every stride exact, size-1 dims too)
/// and the dims before it contiguous among themselves.
fn outer_stride_ok(sh: &[usize], st: &[usize], dim: usize) -> bool {
    let mut expected = 1usize;
    for i in (dim..sh.len()).rev() {
        if st[i] != expected {
            return false;
        }
        expected *= sh[i];
    }
    if dim == 0 {
        return true;
    }
    let mut eo = st[dim - 1];
    for k in (0..dim - 1).rev() {
        eo *= sh[k + 1];
        if st[k] != eo {
            return false;
        }
    }
    true
}
/// The storage offsets of a layout's elements in logical (row-major) order.
fn offsets(sh: &[usize], st: &[usize], off: usize) -> Vec<usize> {
    let n = count(sh);
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return out;
    }
    let r = sh.len();
    let mut idx = vec![0usize; r];
    let mut o = off;
    for _ in 0..n {
        out.push(o);
        let mut d = r;
        while d > 0 {
            d -= 1;
            idx[d] += 1;
            o += st[d];
            if idx[d] < sh[d] {
                break;
            }
            o -= st[d] * sh[d];
            idx[d] = 0;
        }
    }
    out
}

/// sum_keepdim's shape over `dims`.
fn reduced(sh: &[usize], dims: &[usize]) -> Vec<usize> {
    let mut o = sh.to_vec();
    for &d in dims {
        o[d] = 1;
    }
    o
}

/// out[dst] += v over the source's logical order (candle's ReduceSum: each output summed from 0 in that order).
fn sum_vals(v: &[f32], sh: &[usize], dims: &[usize]) -> Vec<f32> {
    let osh = reduced(sh, dims);
    let ost = contig_strides(&osh);
    let mut out = vec![0f32; count(&osh)];
    let r = sh.len();
    let mut idx = vec![0usize; r];
    for &x in v.iter() {
        let mut o = 0;
        for d in 0..r {
            if osh[d] != 1 {
                o += idx[d] * ost[d];
            }
        }
        out[o] += x;
        let mut d = r;
        while d > 0 {
            d -= 1;
            idx[d] += 1;
            if idx[d] < sh[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

/// Values broadcast from shape `sh` to `to` (candle's broadcast_as: new leading dims and size-1 dims repeat).
fn bcast_vals(v: &[f32], sh: &[usize], to: &[usize]) -> Vec<f32> {
    let lead = to.len() - sh.len();
    let mut st = vec![0usize; to.len()];
    let cs = contig_strides(sh);
    for (i, (&d, &s)) in sh.iter().zip(&cs).enumerate() {
        st[lead + i] = if d == 1 && to[lead + i] != 1 { 0 } else { s };
    }
    offsets(to, &st, 0).into_iter().map(|o| v[o]).collect()
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
// SAFETY: threads write disjoint outputs through it (`par_ranges`)
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// gemm 0.19's horizontal kernel for one output: eight lanes of fma over 8-wide steps (the tail masked, its lanes
/// fma(0, 0, acc)), the lanes summed as its reduce_sum; k in blocks of 1024 (the next added onto the first).
fn hdot(l: &dyn Fn(usize) -> f32, r: &dyn Fn(usize) -> f32, k: usize) -> f32 {
    let mut dst = 0f32;
    let mut depth = 0;
    let mut first = true;
    while depth < k {
        let kb = (k - depth).min(1024);
        let mut acc = [0f32; 8];
        let mut j = 0;
        while j + 8 <= kb {
            for (lane, a) in acc.iter_mut().enumerate() {
                *a = l(depth + j + lane).mul_add(r(depth + j + lane), *a);
            }
            j += 8;
        }
        if j < kb {
            for (lane, a) in acc.iter_mut().enumerate() {
                let (x, y) = if j + lane < kb { (l(depth + j + lane), r(depth + j + lane)) } else { (0.0, 0.0) };
                *a = x.mul_add(y, *a);
            }
        }
        let s = ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]));
        dst = if first { 1.0 * s } else { 1f32.mul_add(s, dst) };
        first = false;
        depth += kb;
    }
    dst
}

/// One gemm call as candle makes it (dst [m, n] row-major, alpha 0, beta 1) on an AVX2 + FMA CPU, through the paths
/// gemm takes for it; `lhs`/`rhs` are the storages from their offsets.
#[allow(clippy::too_many_arguments)]
fn gemm(m: usize, n: usize, k: usize, lhs: &[f32], lhs_cs: usize, lhs_rs: usize, rhs: &[f32], rhs_cs: usize, rhs_rs: usize, th: usize) -> Result<Vec<f32>> {
    let mut dst = vec![0f32; m * n];
    if m == 0 || n == 0 || k == 0 {
        return Ok(dst);
    }
    let (dst_rs, dst_cs) = (n, 1usize);
    let tr = dst_cs < dst_rs;
    // after gemm's transpose: L [mm, k] (L_cs, L_rs), R [k, nn] (R_cs, R_rs), D(i, j) = dst at i·d_rs + j·d_cs
    let (mm, nn, d_cs, d_rs, lp, l_cs, l_rs, rp, r_cs, r_rs) =
        if tr { (n, m, dst_rs, dst_cs, rhs, rhs_rs, rhs_cs, lhs, lhs_rs, lhs_cs) } else { (m, n, dst_cs, dst_rs, lhs, lhs_cs, lhs_rs, rhs, rhs_cs, rhs_rs) };
    let lv = |i: usize, l: usize| lp[i * l_rs + l * l_cs];
    let rv = |l: usize, j: usize| rp[l * r_rs + j * r_cs];
    let dp = SendPtr(dst.as_mut_ptr());
    let put = move |i: usize, j: usize, v: f32| {
        let dp = dp;
        // SAFETY: (i, j) is in range and each output is written by one call
        unsafe { *dp.0.add(i * d_rs + j * d_cs) = v };
    };
    if l_cs == 1 && r_rs == 1 && mm * nn <= 256 {
        for j in 0..nn {
            for i in 0..mm {
                put(i, j, hdot(&|l| lv(i, l), &|l| rv(l, j), k));
            }
        }
        return Ok(dst);
    }
    if k <= 2 {
        for j in 0..nn {
            for i in 0..mm {
                let v = if k == 1 { lv(i, 0) * (1.0 * rv(0, j)) } else { lv(i, 1) * (1.0 * rv(1, j)) + lv(i, 0) * (1.0 * rv(0, j)) };
                put(i, j, v);
            }
        }
        return Ok(dst);
    }
    // column-major gemv: out = L(i, 0)·R(0) then fma over l (rows i in any order)
    let colmajor = |rows: usize, lv: &(dyn Fn(usize, usize) -> f32 + Sync), r: &(dyn Fn(usize) -> f32 + Sync), put: &(dyn Fn(usize, f32) + Sync)| {
        super::fmm::par_ranges(rows, 64, th, rows * k, |a, b| {
            for i in a..b {
                let mut acc = lv(i, 0) * r(0);
                for l in 1..k {
                    acc = lv(i, l).mul_add(r(l), acc);
                }
                put(i, acc);
            }
        });
    };
    let rowmajor = |rows: usize, src: &[f32], stride: usize, r: &[f32], put: &(dyn Fn(usize, f32) + Sync)| {
        super::fmm::par_ranges(rows, 16, th, rows * k, |a, b| {
            for i in a..b {
                put(i, super::fmm::dot_rm(&src[i * stride..i * stride + k], r));
            }
        });
    };
    if nn <= 1 && l_rs == 1 && d_rs == 1 {
        let rcol: Vec<f32> = (0..k).map(|l| rv(l, 0)).collect();
        if l_cs == mm && lp.len() >= k * mm {
            // L column-major with rows mm apart: one row-major [k, mm] matrix, the fused step's x·W kernel
            let o = super::fmm::xw(&rcol, &lp[..k * mm], mm, th);
            for (i, v) in o.into_iter().enumerate() {
                put(i, 0, v);
            }
        } else {
            colmajor(mm, &|i, l| lv(i, l), &|l| rcol[l], &|i, v| put(i, 0, v));
        }
        return Ok(dst);
    }
    if nn <= 1 && l_cs == 1 && r_rs == 1 {
        let rcol: Vec<f32> = (0..k).map(|l| rv(l, 0)).collect();
        rowmajor(mm, lp, l_rs, &rcol, &|i, v| put(i, 0, v));
        return Ok(dst);
    }
    if mm <= 1 && r_cs == 1 && d_cs == 1 {
        let lrow: Vec<f32> = (0..k).map(|l| lv(0, l)).collect();
        colmajor(nn, &|j, l| rv(l, j), &|l| lrow[l], &|j, v| put(0, j, v));
        return Ok(dst);
    }
    if mm <= 1 && r_rs == 1 && l_cs == 1 {
        let lrow: Vec<f32> = (0..k).map(|l| lv(0, l)).collect();
        rowmajor(nn, rp, r_cs, &lrow, &|j, v| put(0, j, v));
        return Ok(dst);
    }
    bail!("fast4 tape: a product gemm runs through its blocked kernel (m {mm}, n {nn}, k {k}), not emulated")
}

/// A node's storage: its own values, or a slice the caller lends (a weight, a session's changes), never copied.
#[derive(Clone)]
enum Src {
    Own(Rc<Vec<f32>>),
    Ext(usize),
}

pub(crate) struct Tape<'a> {
    pub(crate) ns: Vec<Node>,
    pub(crate) th: usize,
    ext: Vec<&'a [f32]>,
}

impl<'a> Tape<'a> {
    pub(crate) fn new(th: usize) -> Tape<'a> {
        Tape { ns: Vec::with_capacity(2048), th, ext: Vec::new() }
    }

    fn push(&mut self, s: Src, sh: Vec<usize>, st: Vec<usize>, off: usize, op: Op, tr: bool) -> T {
        let op = if tr { op } else { Op::Leaf };
        self.ns.push(Node { s, sh, st, off, op, tr });
        self.ns.len() - 1
    }

    fn fresh(&mut self, v: Vec<f32>, sh: Vec<usize>, op: Op, tr: bool) -> T {
        let st = contig_strides(&sh);
        self.push(Src::Own(Rc::new(v)), sh, st, 0, op, tr)
    }

    /// A constant (no variable under it) with contiguous layout.
    pub(crate) fn leaf(&mut self, v: Vec<f32>, sh: &[usize]) -> T {
        assert_eq!(v.len(), count(sh), "tape leaf {sh:?}");
        self.fresh(v, sh.to_vec(), Op::Leaf, false)
    }

    /// A constant over a lent slice (a weight), contiguous.
    pub(crate) fn leaf_ext(&mut self, s: &'a [f32], sh: &[usize]) -> T {
        assert_eq!(s.len(), count(sh), "tape leaf {sh:?}");
        self.ext.push(s);
        let st = contig_strides(sh);
        let k = self.ext.len() - 1;
        self.push(Src::Ext(k), sh.to_vec(), st, 0, Op::Leaf, false)
    }

    fn buf(&self, a: T) -> &[f32] {
        match &self.ns[a].s {
            Src::Own(v) => v,
            Src::Ext(k) => self.ext[*k],
        }
    }

    /// A variable of zeros (e-prop's probe).
    pub(crate) fn var(&mut self, sh: &[usize]) -> T {
        self.fresh(vec![0f32; count(sh)], sh.to_vec(), Op::Var, true)
    }

    pub(crate) fn shape(&self, a: T) -> &[usize] {
        &self.ns[a].sh
    }

    /// The values in logical order.
    pub(crate) fn vals(&self, a: T) -> Vec<f32> {
        let (n, s) = (&self.ns[a], self.buf(a));
        if is_contig(&n.sh, &n.st) {
            s[n.off..n.off + count(&n.sh)].to_vec()
        } else {
            offsets(&n.sh, &n.st, n.off).into_iter().map(|o| s[o]).collect()
        }
    }

    pub(crate) fn is_contiguous(&self, a: T) -> bool {
        is_contig(&self.ns[a].sh, &self.ns[a].st)
    }

    /// `detach`: the same values as a constant.
    pub(crate) fn detach(&mut self, a: T) -> T {
        let n = &self.ns[a];
        let (s, sh, st, off) = (n.s.clone(), n.sh.clone(), n.st.clone(), n.off);
        self.push(s, sh, st, off, Op::Leaf, false)
    }

    // ---------------------------------------------------------------------------------------------- computing ops

    pub(crate) fn bin(&mut self, k: Bin, a: T, b: T) -> Result<T> {
        if self.ns[a].sh != self.ns[b].sh {
            bail!("tape {k:?}: shapes {:?} {:?}", self.ns[a].sh, self.ns[b].sh);
        }
        if count(&self.ns[a].sh) == 0 {
            return Ok(a);
        }
        let (x, y) = (self.vals(a), self.vals(b));
        let v: Vec<f32> = match k {
            Bin::Add => x.iter().zip(&y).map(|(p, q)| p + q).collect(),
            Bin::Sub => x.iter().zip(&y).map(|(p, q)| p - q).collect(),
            Bin::Mul => x.iter().zip(&y).map(|(p, q)| p * q).collect(),
            Bin::Div => x.iter().zip(&y).map(|(p, q)| p / q).collect(),
        };
        let tr = self.ns[a].tr || self.ns[b].tr;
        let sh = self.ns[a].sh.clone();
        Ok(self.fresh(v, sh, Op::Bin(k, a, b), tr))
    }

    pub(crate) fn add(&mut self, a: T, b: T) -> Result<T> {
        self.bin(Bin::Add, a, b)
    }
    pub(crate) fn sub(&mut self, a: T, b: T) -> Result<T> {
        self.bin(Bin::Sub, a, b)
    }
    pub(crate) fn mul(&mut self, a: T, b: T) -> Result<T> {
        self.bin(Bin::Mul, a, b)
    }
    pub(crate) fn div(&mut self, a: T, b: T) -> Result<T> {
        self.bin(Bin::Div, a, b)
    }

    /// candle's broadcast_{add, sub, mul, div}: a broadcast_as node on each side whose shape differs.
    pub(crate) fn bbin(&mut self, k: Bin, a: T, b: T) -> Result<T> {
        let (sa, sb) = (self.ns[a].sh.clone(), self.ns[b].sh.clone());
        let r = sa.len().max(sb.len());
        let mut sh = vec![0usize; r];
        for i in 0..r {
            let da = if i + sa.len() >= r { sa[i + sa.len() - r] } else { 1 };
            let db = if i + sb.len() >= r { sb[i + sb.len() - r] } else { 1 };
            sh[i] = if da == db || db == 1 { da } else if da == 1 { db } else { bail!("tape broadcast {sa:?} {sb:?}") };
        }
        let a2 = if sh != sa { self.bcast(a, &sh)? } else { a };
        let b2 = if sh != sb { self.bcast(b, &sh)? } else { b };
        self.bin(k, a2, b2)
    }

    pub(crate) fn un(&mut self, k: Un, a: T) -> Result<T> {
        if count(&self.ns[a].sh) == 0 {
            return Ok(a);
        }
        let x = self.vals(a);
        let v: Vec<f32> = x
            .iter()
            .map(|&v| match k {
                Un::Exp => v.exp(),
                Un::Neg => -v,
                Un::Recip => v.recip(),
                Un::Log => v.ln(),
                Un::Sqr => v * v,
                Un::Relu => v.max(0f32),
                Un::Abs => v.abs(),
                Un::Silu => v / (1.0 + (-v).exp()),
            })
            .collect();
        let tr = self.ns[a].tr;
        let sh = self.ns[a].sh.clone();
        Ok(self.fresh(v, sh, Op::Un(k, a), tr))
    }

    pub(crate) fn affine(&mut self, a: T, mul: f64, add: f64) -> Result<T> {
        if count(&self.ns[a].sh) == 0 {
            return Ok(a);
        }
        let (m, d) = (mul as f32, add as f32);
        let v: Vec<f32> = self.vals(a).iter().map(|&v| v * m + d).collect();
        let tr = self.ns[a].tr && mul != 0.0;
        let sh = self.ns[a].sh.clone();
        Ok(self.fresh(v, sh, Op::Affine(a, mul), tr))
    }

    pub(crate) fn powf(&mut self, a: T, e: f64) -> Result<T> {
        if count(&self.ns[a].sh) == 0 {
            return Ok(a);
        }
        let v: Vec<f32> = self.vals(a).iter().map(|&v| v.powf(e as f32)).collect();
        let tr = self.ns[a].tr;
        let sh = self.ns[a].sh.clone();
        Ok(self.fresh(v, sh, Op::Powf(a, e), tr))
    }

    pub(crate) fn sum_keepdim(&mut self, a: T, dims: &[usize]) -> Result<T> {
        let x = self.vals(a);
        let sh = self.ns[a].sh.clone();
        let v = sum_vals(&x, &sh, dims);
        let tr = self.ns[a].tr;
        Ok(self.fresh(v, reduced(&sh, dims), Op::Sum(a), tr))
    }

    /// `sum` (keepdim false): sum_keepdim, then squeeze (one dim) or a reshape (several).
    pub(crate) fn sum(&mut self, a: T, dims: &[usize]) -> Result<T> {
        let s = self.sum_keepdim(a, dims)?;
        match dims {
            [] => Ok(s),
            [d] => self.squeeze(s, *d),
            _ => {
                let sh: Vec<usize> = self.ns[s].sh.iter().enumerate().filter(|(i, _)| !dims.contains(i)).map(|(_, &v)| v).collect();
                self.reshape(s, &sh)
            }
        }
    }

    /// `mean_keepdim`: sum_keepdim, then · (1 / n) as an affine.
    pub(crate) fn mean_keepdim(&mut self, a: T, dims: &[usize]) -> Result<T> {
        let n: usize = dims.iter().map(|&d| self.ns[a].sh[d]).product();
        let s = self.sum_keepdim(a, dims)?;
        self.affine(s, 1f64 / n as f64, 0.0)
    }

    /// `max_keepdim` over one dim: the first value, replaced by each larger one.
    pub(crate) fn max_keepdim(&mut self, a: T, dim: usize) -> Result<T> {
        let sh = self.ns[a].sh.clone();
        let x = self.vals(a);
        let osh = reduced(&sh, &[dim]);
        let (outer, len): (usize, usize) = (count(&sh[..dim]), sh[dim]);
        let inner = count(&sh[dim + 1..]);
        let mut v = vec![0f32; outer * inner];
        for o in 0..outer {
            for i in 0..inner {
                let mut val = x[o * len * inner + i];
                for j in 0..len {
                    let s = x[(o * len + j) * inner + i];
                    if val < s {
                        val = s;
                    }
                }
                v[o * inner + i] = val;
            }
        }
        let tr = self.ns[a].tr;
        Ok(self.fresh(v, osh, Op::Max(a), tr))
    }

    pub(crate) fn matmul(&mut self, a: T, b: T) -> Result<T> {
        let (sa, sb) = (self.ns[a].sh.clone(), self.ns[b].sh.clone());
        let r = sa.len();
        if r < 2 || sb.len() != r {
            bail!("tape matmul {sa:?} {sb:?}");
        }
        let (m, k, n) = (sa[r - 2], sa[r - 1], sb[r - 1]);
        let mut osh = sa[..r - 2].to_vec();
        osh.extend([m, n]);
        if count(&osh) == 0 || k == 0 {
            // candle: a fresh zeros tensor (no op)
            return Ok(self.leaf(vec![0f32; count(&osh)], &osh));
        }
        if sb[r - 2] != k || count(&sa[..r - 2]) != count(&sb[..r - 2]) {
            bail!("tape matmul {sa:?} {sb:?}");
        }
        let v = self.mm_vals(a, b)?;
        let tr = self.ns[a].tr || self.ns[b].tr;
        Ok(self.fresh(v, osh, Op::Matmul(a, b), tr))
    }

    /// candle's `MatMul::f` on the two layouts (its a_skip / b_skip, its batch folding), each step one gemm call.
    fn mm_vals(&self, a: T, b: T) -> Result<Vec<f32>> {
        let (na, nb) = (&self.ns[a], &self.ns[b]);
        mm_layouts(&self.buf(a)[na.off..], &na.sh, &na.st, &self.buf(b)[nb.off..], &nb.sh, &nb.st, self.th)
    }

    /// `broadcast_matmul`: broadcast_as + contiguous on the side whose shape differs.
    pub(crate) fn broadcast_matmul(&mut self, a: T, b: T) -> Result<T> {
        let (sa, sb) = (self.ns[a].sh.clone(), self.ns[b].sh.clone());
        let (ba, bb) = (&sa[..sa.len() - 2], &sb[..sb.len() - 2]);
        let r = ba.len().max(bb.len());
        let mut batch = vec![0usize; r];
        for i in 0..r {
            let da = if i + ba.len() >= r { ba[i + ba.len() - r] } else { 1 };
            let db = if i + bb.len() >= r { bb[i + bb.len() - r] } else { 1 };
            batch[i] = if da == db || db == 1 { da } else if da == 1 { db } else { bail!("tape broadcast_matmul {sa:?} {sb:?}") };
        }
        let ls: Vec<usize> = batch.iter().chain(&sa[sa.len() - 2..]).copied().collect();
        let rs: Vec<usize> = batch.iter().chain(&sb[sb.len() - 2..]).copied().collect();
        let a2 = if ls != sa {
            let t = self.bcast(a, &ls)?;
            self.contiguous(t)?
        } else {
            a
        };
        let b2 = if rs != sb {
            let t = self.bcast(b, &rs)?;
            self.contiguous(t)?
        } else {
            b
        };
        self.matmul(a2, b2)
    }

    /// `Tensor::cat`: one tensor is itself; else one Cat node when every part is contiguous or copies in 2-d blocks
    /// (`Layout::outer_stride_for_dim`), else along dim 0, else (another dim) each part transposed (0, dim), the parts
    /// joined along dim 0 and the result transposed back.
    pub(crate) fn cat(&mut self, args: &[T], dim: usize) -> Result<T> {
        if args.len() == 1 {
            return Ok(args[0]);
        }
        let compat = args.iter().all(|&a| {
            let n = &self.ns[a];
            is_contig(&n.sh, &n.st) || outer_stride_ok(&n.sh, &n.st, dim)
        });
        if compat || dim == 0 {
            return self.cat_node(args, dim);
        }
        let ts: Vec<T> = args.iter().map(|&a| self.transpose(a, 0, dim)).collect::<Result<_>>()?;
        let c = self.cat_node(&ts, 0)?;
        self.transpose(c, 0, dim)
    }

    fn cat_node(&mut self, args: &[T], dim: usize) -> Result<T> {
        let mut sh = self.ns[args[0]].sh.clone();
        sh[dim] = args.iter().map(|&a| self.ns[a].sh[dim]).sum();
        let outer = count(&sh[..dim]);
        let inner = count(&sh[dim + 1..]);
        let parts: Vec<Vec<f32>> = args.iter().map(|&a| self.vals(a)).collect();
        let mut v = Vec::with_capacity(count(&sh));
        for o in 0..outer {
            for (p, &a) in parts.iter().zip(args) {
                let len = self.ns[a].sh[dim] * inner;
                v.extend_from_slice(&p[o * len..(o + 1) * len]);
            }
        }
        let tr = args.iter().any(|&a| self.ns[a].tr);
        Ok(self.fresh(v, sh, Op::Cat(args.to_vec(), dim), tr))
    }

    /// `gather(idx, 1)` on a 2-d tensor with one index per row ([n, 1] out).
    pub(crate) fn gather_last(&mut self, a: T, idx: &[usize]) -> Result<T> {
        let sh = self.ns[a].sh.clone();
        if sh.len() != 2 || idx.len() != sh[0] {
            bail!("tape gather {sh:?}");
        }
        let x = self.vals(a);
        let v: Vec<f32> = idx.iter().enumerate().map(|(r, &j)| x[r * sh[1] + j]).collect();
        let tr = self.ns[a].tr;
        Ok(self.fresh(v, vec![sh[0], 1], Op::Gather(a, idx.to_vec()), tr))
    }

    /// e-prop's `RowMm` (x [b, t, din] contiguous, dw [b, din, dout]): y = 0 + sum_i x_i dw_i, the 0 inputs skipped.
    /// Each output column takes its terms in RowMm's order (i ascending, `o += x_i · w`), so the columns split over
    /// the threads give the same bits.
    pub(crate) fn rowmm(&mut self, x: T, dw: &'a [f32], b: usize, t: usize, din: usize, dout: usize) -> Result<T> {
        let xv = self.vals(x);
        if xv.len() != b * t * din || dw.len() != b * din * dout {
            bail!("tape rowmm sizes");
        }
        let mut y = vec![0f32; b * t * dout];
        for g in 0..b * t {
            let r = g / t;
            let yp = SendPtr(y[g * dout..].as_mut_ptr());
            let xr = &xv[g * din..(g + 1) * din];
            super::fmm::par_ranges(dout, 16, self.th, din * dout, |c0, c1| {
                let yp = yp;
                // SAFETY: this range writes only columns c0..c1 of row g
                let line = unsafe { std::slice::from_raw_parts_mut(yp.0.add(c0), c1 - c0) };
                for (i, &xi) in xr.iter().enumerate() {
                    if xi == 0.0 {
                        continue;
                    }
                    let base = (r * din + i) * dout;
                    for (o, &wj) in line.iter_mut().zip(&dw[base + c0..base + c1]) {
                        *o += xi * wj;
                    }
                }
            });
        }
        self.ext.push(dw);
        let k = self.ext.len() - 1;
        let tr = self.ns[x].tr;
        Ok(self.fresh(y, vec![b, t, dout], Op::RowMm(x, k, [b, t, din, dout]), tr))
    }

    // ------------------------------------------------------------------------------------------------- views

    /// `broadcast_as` (always a node in candle).
    pub(crate) fn bcast(&mut self, a: T, to: &[usize]) -> Result<T> {
        let n = &self.ns[a];
        if to.len() < n.sh.len() {
            bail!("tape bcast {:?} to {to:?}", n.sh);
        }
        let lead = to.len() - n.sh.len();
        let mut st = vec![0usize; to.len()];
        for i in 0..n.sh.len() {
            let (d, t) = (n.sh[i], to[lead + i]);
            st[lead + i] = if d == t {
                n.st[i]
            } else if d == 1 {
                0
            } else {
                bail!("tape bcast {:?} to {to:?}", n.sh)
            };
        }
        let (s, off, tr) = (n.s.clone(), n.off, n.tr);
        Ok(self.push(s, to.to_vec(), st, off, Op::Bcast(a), tr))
    }

    /// `reshape` (a node; a copy when not contiguous; contiguous strides either way).
    pub(crate) fn reshape(&mut self, a: T, sh: &[usize]) -> Result<T> {
        if count(sh) != count(&self.ns[a].sh) {
            bail!("tape reshape {:?} to {sh:?}", self.ns[a].sh);
        }
        let tr = self.ns[a].tr;
        if self.is_contiguous(a) {
            let (s, off) = (self.ns[a].s.clone(), self.ns[a].off);
            Ok(self.push(s, sh.to_vec(), contig_strides(sh), off, Op::Reshape(a), tr))
        } else {
            let v = self.vals(a);
            Ok(self.fresh(v, sh.to_vec(), Op::Reshape(a), tr))
        }
    }

    /// `squeeze`: a Reshape node keeping the other strides (the same tensor if the dim is not 1).
    pub(crate) fn squeeze(&mut self, a: T, dim: usize) -> Result<T> {
        let n = &self.ns[a];
        if n.sh[dim] != 1 {
            return Ok(a);
        }
        let (mut sh, mut st) = (n.sh.clone(), n.st.clone());
        sh.remove(dim);
        st.remove(dim);
        let (s, off, tr) = (n.s.clone(), n.off, n.tr);
        Ok(self.push(s, sh, st, off, Op::Reshape(a), tr))
    }

    /// `unsqueeze`: a Reshape node, the new dim's stride strides[dim] (1 past the end).
    pub(crate) fn unsqueeze(&mut self, a: T, dim: usize) -> Result<T> {
        let n = &self.ns[a];
        let (mut sh, mut st) = (n.sh.clone(), n.st.clone());
        let s1 = if dim < st.len() { st[dim] } else { 1 };
        sh.insert(dim, 1);
        st.insert(dim, s1);
        let (s, off, tr) = (n.s.clone(), n.off, n.tr);
        Ok(self.push(s, sh, st, off, Op::Reshape(a), tr))
    }

    pub(crate) fn transpose(&mut self, a: T, d1: usize, d2: usize) -> Result<T> {
        if d1 == d2 {
            return Ok(a);
        }
        let n = &self.ns[a];
        let (mut sh, mut st) = (n.sh.clone(), n.st.clone());
        sh.swap(d1, d2);
        st.swap(d1, d2);
        let (s, off, tr) = (n.s.clone(), n.off, n.tr);
        Ok(self.push(s, sh, st, off, Op::Transpose(a, d1, d2), tr))
    }

    /// `t()`: the last two dims swapped.
    pub(crate) fn t(&mut self, a: T) -> Result<T> {
        let r = self.ns[a].sh.len();
        self.transpose(a, r - 2, r - 1)
    }

    pub(crate) fn contiguous(&mut self, a: T) -> Result<T> {
        if self.is_contiguous(a) {
            return Ok(a);
        }
        let v = self.vals(a);
        let sh = self.ns[a].sh.clone();
        let tr = self.ns[a].tr;
        Ok(self.fresh(v, sh, Op::Copy(a), tr))
    }

    pub(crate) fn narrow(&mut self, a: T, dim: usize, start: usize, len: usize) -> Result<T> {
        let n = &self.ns[a];
        if start + len > n.sh[dim] {
            bail!("tape narrow");
        }
        if start == 0 && len == n.sh[dim] {
            return Ok(a);
        }
        let mut sh = n.sh.clone();
        sh[dim] = len;
        let (s, st, off, tr) = (n.s.clone(), n.st.clone(), n.off + start * n.st[dim], n.tr);
        Ok(self.push(s, sh, st, off, Op::Narrow(a, dim, start, len), tr))
    }

    // ---------------------------------------------------------------------------------------------- backward

    /// `Tensor::sorted_nodes`: the walk's order (the loss first).
    pub(crate) fn sorted(&self, loss: T) -> Vec<T> {
        let mut seen = vec![false; self.ns.len()];
        let mut out = Vec::new();
        // iterative post-order over tracked nodes, arguments in order
        let mut stack: Vec<(T, usize)> = vec![(loss, 0)];
        seen[loss] = true;
        while let Some(&mut (n, ref mut i)) = stack.last_mut() {
            let args = self.args(n);
            if *i < args.len() {
                let a = args[*i];
                *i += 1;
                if !seen[a] && self.ns[a].tr {
                    seen[a] = true;
                    stack.push((a, 0));
                }
            } else {
                out.push(n);
                stack.pop();
            }
        }
        out.reverse();
        out
    }

    fn args(&self, n: T) -> Vec<T> {
        match &self.ns[n].op {
            Op::Leaf | Op::Var => vec![],
            Op::Bin(_, a, b) | Op::Matmul(a, b) => vec![*a, *b],
            Op::Un(_, a) | Op::Affine(a, _) | Op::Powf(a, _) | Op::Bcast(a) | Op::Reshape(a) | Op::Transpose(a, _, _) | Op::Copy(a) | Op::Narrow(a, ..) | Op::Sum(a) | Op::Max(a) | Op::Gather(a, _) | Op::RowMm(a, ..) => vec![*a],
            Op::Cat(v, _) => v.clone(),
        }
    }

    /// The backward from `loss` (a one-element tensor): each variable's gradient (None for nodes that are not
    /// variables or got none), as `loss.backward()` gives them.
    pub(crate) fn backward(&self, loss: T) -> Result<Vec<Option<Vec<f32>>>> {
        let order = self.sorted(loss);
        let mut g: Vec<Option<Vec<f32>>> = vec![None; self.ns.len()];
        g[loss] = Some(vec![1f32; count(&self.ns[loss].sh)]);
        for &n in &order {
            if matches!(self.ns[n].op, Op::Var) {
                continue;
            }
            let gr = g[n].take().expect("fast4 tape: a gradient in the walk's order");
            self.back_one(n, &gr, &mut g)?;
        }
        Ok(g)
    }

    /// grads[a] = (grads[a] or zeros) + v (or - v), when a variable is under a.
    fn acc(&self, g: &mut [Option<Vec<f32>>], a: T, v: Vec<f32>, sub: bool) {
        if !self.ns[a].tr {
            return;
        }
        let cur = g[a].get_or_insert_with(|| vec![0f32; v.len()]);
        debug_assert_eq!(cur.len(), v.len());
        if sub {
            cur.iter_mut().zip(&v).for_each(|(c, x)| *c = *c - x);
        } else {
            cur.iter_mut().zip(&v).for_each(|(c, x)| *c = *c + x);
        }
    }

    fn back_one(&self, n: T, gr: &[f32], g: &mut [Option<Vec<f32>>]) -> Result<()> {
        let node = &self.ns[n];
        match &node.op {
            Op::Leaf | Op::Var => {}
            Op::Bin(k, a, b) => {
                let (a, b) = (*a, *b);
                match k {
                    Bin::Add => {
                        self.acc(g, a, gr.to_vec(), false);
                        self.acc(g, b, gr.to_vec(), false);
                    }
                    Bin::Sub => {
                        self.acc(g, a, gr.to_vec(), false);
                        self.acc(g, b, gr.to_vec(), true);
                    }
                    Bin::Mul => {
                        if self.ns[a].tr {
                            let y = self.vals(b);
                            self.acc(g, a, gr.iter().zip(&y).map(|(p, q)| p * q).collect(), false);
                        }
                        if self.ns[b].tr {
                            let x = self.vals(a);
                            self.acc(g, b, gr.iter().zip(&x).map(|(p, q)| p * q).collect(), false);
                        }
                    }
                    Bin::Div => {
                        let y = self.vals(b);
                        if self.ns[a].tr {
                            self.acc(g, a, gr.iter().zip(&y).map(|(p, q)| p / q).collect(), false);
                        }
                        if self.ns[b].tr {
                            let x = self.vals(a);
                            let v: Vec<f32> = gr.iter().zip(&x).zip(&y).map(|((p, q), r)| (p * q) / (r * r)).collect();
                            self.acc(g, b, v, true);
                        }
                    }
                }
            }
            Op::Un(k, a) => {
                let a = *a;
                let x = self.vals(a);
                match k {
                    Un::Exp => {
                        let y = self.vals(n);
                        self.acc(g, a, gr.iter().zip(&y).map(|(p, q)| p * q).collect(), false);
                    }
                    Un::Neg => self.acc(g, a, gr.to_vec(), true),
                    Un::Recip => self.acc(g, a, gr.iter().zip(&x).map(|(p, q)| p / (q * q)).collect(), true),
                    Un::Log => self.acc(g, a, gr.iter().zip(&x).map(|(p, q)| p / q).collect(), false),
                    Un::Sqr => self.acc(g, a, gr.iter().zip(&x).map(|(p, q)| (q * p) * 2f32 + 0f32).collect(), false),
                    Un::Relu => self.acc(g, a, gr.iter().zip(&x).map(|(p, &q)| p * if q >= 0.0 { 1.0 } else { 0.0 }).collect(), false),
                    Un::Abs => self.acc(g, a, gr.iter().zip(&x).map(|(p, &q)| p * if q >= 0.0 { 1.0 } else { -1.0 }).collect(), false),
                    Un::Silu => {
                        let y = self.vals(n);
                        let v = gr
                            .iter()
                            .zip(&x)
                            .zip(&y)
                            .map(|((p, &q), &yy)| {
                                let s = ((-q).exp() * 1.0 + 1.0).recip();
                                p * (s * (yy * -1.0 + 1.0) + yy)
                            })
                            .collect();
                        self.acc(g, a, v, false);
                    }
                }
            }
            Op::Affine(a, mul) => {
                let m = *mul as f32;
                self.acc(g, *a, gr.iter().map(|p| p * m + 0f32).collect(), false);
            }
            Op::Powf(a, e) => {
                let x = self.vals(*a);
                let (em1, ef) = ((e - 1.0) as f32, *e as f32);
                self.acc(g, *a, gr.iter().zip(&x).map(|(p, q)| (p * q.powf(em1)) * ef + 0f32).collect(), false);
            }
            Op::Bcast(a) => {
                let (ash, nsh) = (&self.ns[*a].sh, &node.sh);
                let left = nsh.len() - ash.len();
                let mut dims: Vec<usize> = (0..left).collect();
                for (i, (nd, ad)) in nsh[left..].iter().zip(ash).enumerate() {
                    if nd != ad {
                        dims.push(i + left);
                    }
                }
                let v = sum_vals(gr, nsh, &dims);
                self.acc(g, *a, v, false);
            }
            Op::Reshape(a) | Op::Copy(a) => self.acc(g, *a, gr.to_vec(), false),
            Op::Transpose(a, d1, d2) => {
                // grad.transpose(d1, d2), then added into the argument's (contiguous) gradient in logical order
                let mut sh = node.sh.clone();
                let mut st = contig_strides(&sh);
                sh.swap(*d1, *d2);
                st.swap(*d1, *d2);
                let v = offsets(&sh, &st, 0).into_iter().map(|o| gr[o]).collect();
                self.acc(g, *a, v, false);
            }
            Op::Narrow(a, dim, start, len) => {
                let ash = &self.ns[*a].sh;
                let outer = count(&ash[..*dim]);
                let inner = count(&ash[*dim + 1..]);
                let mut v = vec![0f32; count(ash)];
                for o in 0..outer {
                    for j in 0..*len {
                        let src = (o * len + j) * inner;
                        let dst = (o * ash[*dim] + start + j) * inner;
                        v[dst..dst + inner].copy_from_slice(&gr[src..src + inner]);
                    }
                }
                self.acc(g, *a, v, false);
            }
            Op::Cat(args, dim) => {
                let sh = &node.sh;
                let outer = count(&sh[..*dim]);
                let inner = count(&sh[*dim + 1..]);
                let mut start = 0;
                for &a in args {
                    let len = self.ns[a].sh[*dim];
                    let mut v = Vec::with_capacity(outer * len * inner);
                    for o in 0..outer {
                        let s = (o * sh[*dim] + start) * inner;
                        v.extend_from_slice(&gr[s..s + len * inner]);
                    }
                    self.acc(g, a, v, false);
                    start += len;
                }
            }
            Op::Sum(a) => {
                let v = bcast_vals(gr, &node.sh, &self.ns[*a].sh);
                self.acc(g, *a, v, false);
            }
            Op::Max(a) => {
                let ash = &self.ns[*a].sh;
                let (nb, gb) = (bcast_vals(&self.vals(n), &node.sh, ash), bcast_vals(gr, &node.sh, ash));
                let x = self.vals(*a);
                let v = nb.iter().zip(&x).zip(&gb).map(|((p, q), r)| if p == q { 1f32 } else { 0f32 } * r).collect();
                self.acc(g, *a, v, false);
            }
            Op::Matmul(a, b) => {
                let (a, b) = (*a, *b);
                let gsh = node.sh.clone();
                let gst = contig_strides(&gsh);
                if self.ns[a].tr {
                    // grad.matmul(rhs.t())
                    let nb = &self.ns[b];
                    let r = nb.sh.len();
                    let (mut sh, mut st) = (nb.sh.clone(), nb.st.clone());
                    sh.swap(r - 2, r - 1);
                    st.swap(r - 2, r - 1);
                    let v = mm_layouts(gr, &gsh, &gst, &self.buf(b)[nb.off..], &sh, &st, self.th)?;
                    self.acc(g, a, v, false);
                }
                if self.ns[b].tr {
                    // lhs.t().matmul(grad)
                    let na = &self.ns[a];
                    let r = na.sh.len();
                    let (mut sh, mut st) = (na.sh.clone(), na.st.clone());
                    sh.swap(r - 2, r - 1);
                    st.swap(r - 2, r - 1);
                    let v = mm_layouts(&self.buf(a)[na.off..], &sh, &st, gr, &gsh, &gst, self.th)?;
                    self.acc(g, b, v, false);
                }
            }
            Op::Gather(a, idx) => {
                let ash = &self.ns[*a].sh;
                let mut v = vec![0f32; count(ash)];
                for (r, &j) in idx.iter().enumerate() {
                    v[r * ash[1] + j] += gr[r];
                }
                self.acc(g, *a, v, false);
            }
            Op::RowMm(x, k, [b, t, din, dout]) => {
                // RowMmT: each input's sum as eprop.rs writes it (`.map(a · c).sum()`), the inputs split over threads
                let (t, din, dout, dw) = (*t, *din, *dout, self.ext[*k]);
                let mut gx = vec![0f32; b * t * din];
                for gi in 0..b * t {
                    let r = gi / t;
                    let gy = &gr[gi * dout..(gi + 1) * dout];
                    let gp = SendPtr(gx[gi * din..].as_mut_ptr());
                    super::fmm::par_ranges(din, 16, self.th, din * dout, |i0, i1| {
                        let gp = gp;
                        for i in i0..i1 {
                            let s: f32 = gy.iter().zip(&dw[(r * din + i) * dout..(r * din + i + 1) * dout]).map(|(a, c)| a * c).sum();
                            // SAFETY: this range writes only inputs i0..i1 of row gi
                            unsafe { *gp.0.add(i) = s };
                        }
                    });
                }
                self.acc(g, *x, gx, false);
            }
        }
        Ok(())
    }
}

/// candle's `MatMul::f` for f32 on the CPU over two (storage-from-offset, shape, strides) operands.
fn mm_layouts(ls: &[f32], ash: &[usize], ast: &[usize], rs: &[f32], bsh: &[usize], bst: &[usize], th: usize) -> Result<Vec<f32>> {
    let r = ash.len();
    let (m, k, n) = (ash[r - 2], ash[r - 1], bsh[r - 1]);
    let batch: usize = count(&ash[..r - 2]);
    let (lhs_cs, lhs_rs, rhs_cs, rhs_rs) = (ast[r - 1], ast[r - 2], bst[r - 1], bst[r - 2]);
    let skip = |st: &[usize], sh: &[usize], dflt: usize| -> Result<usize> {
        Ok(match st[..r - 2] {
            [s1, stride] if s1 == stride * sh[1] => stride,
            [_, stride] if sh[0] == 1 => stride,
            [stride, _] if sh[1] == 1 => stride,
            [stride] => stride,
            [] => dflt,
            _ => bail!("fast4 tape: a non-contiguous matmul operand"),
        })
    };
    let a_skip = skip(ast, ash, m * k)?;
    let b_skip = skip(bst, bsh, n * k)?;
    let (b, mm) = if b_skip == 0 && a_skip == m * k {
        (1, batch * m)
    } else if a_skip == 0 && b_skip == n * k {
        bail!("fast4 tape: a matmul with its lhs broadcast over the batch (not emulated)")
    } else {
        (batch, m)
    };
    let mut out = Vec::with_capacity(batch * m * n);
    for step in 0..b {
        let o = gemm(mm, n, k, &ls[step * a_skip..], lhs_cs, lhs_rs, &rs[step * b_skip..], rhs_cs, rhs_rs, th)?;
        out.extend(o);
    }
    Ok(out)
}
