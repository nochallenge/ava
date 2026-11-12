// TESTING / DESIGN ONLY
//! D2 as a layer: a holographic memory of role–filler bindings (Plate 1995). Spaun, Eliasmith's spiking brain model,
//! binds its semantic pointers the same way.
//!
//! Each token makes a role r, a filler f and a query q (db-vectors; r and q scaled to unit length). It binds r ⊛ f
//! (circular convolution) into one fading trace,
//!   T_t = λ·T_{t−1} + r_t ⊛ f_t,
//! and reads the trace of the tokens before it with its query's approximate inverse,
//!   u_t = T_{t−1} ⊛ q_t*,
//! which comes out close to the filler that was bound to a role like q_t. u_t·Wo joins the stream; Wo starts at 0, so
//! the layer starts as no change.
//!
//! In the frequency domain (a DFT written as fixed matrices) binding is an elementwise complex product and the inverse
//! is the complex conjugate, so the whole layer is matmuls and a decayed running sum. `binding::bind` and
//! `binding::unbind` are the reference it is tested against.

use crate::{linear, logit};
use candle_core::{Device, Result, Tensor, D};
use candle_nn::{Init, VarBuilder};

/// Tokens read per decay matrix.
pub const HOLO_CHUNK: usize = 256;

/// The DFT of length n as matrices: c[k, j] = cos(2πkj/n), s[k, j] = sin(2πkj/n) (both symmetric).
pub struct Dft {
    pub c: Tensor,
    pub s: Tensor,
    pub n: usize,
}

impl Dft {
    pub fn new(n: usize, dev: &Device) -> Result<Dft> {
        let mut c = Vec::with_capacity(n * n);
        let mut s = Vec::with_capacity(n * n);
        for k in 0..n {
            for j in 0..n {
                // (k·j mod n) keeps the angle small, so cos and sin stay exact to f32
                let a = 2.0 * std::f64::consts::PI * ((k * j) % n) as f64 / n as f64;
                c.push(a.cos() as f32);
                s.push(a.sin() as f32);
            }
        }
        Ok(Dft { c: Tensor::from_vec(c, (n, n), dev)?, s: Tensor::from_vec(s, (n, n), dev)?, n })
    }

    /// x [.., n] → (Re X, Im X), X_k = Σ_j x_j e^{−2πikj/n}.
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((linear(x, &self.c)?, linear(x, &self.s)?.neg()?))
    }

    /// The real part of the inverse: x_j = (1/n) Σ_k (Re_k cos − Im_k sin)(2πkj/n).
    pub fn inverse(&self, re: &Tensor, im: &Tensor) -> Result<Tensor> {
        (linear(re, &self.c)? - linear(im, &self.s)?)?.affine(1.0 / self.n as f64, 0.0)
    }
}

/// Complex product (a ⊙ b).
pub fn cmul(a: &(Tensor, Tensor), b: &(Tensor, Tensor)) -> Result<(Tensor, Tensor)> {
    Ok((((&a.0 * &b.0)? - (&a.1 * &b.1)?)?, ((&a.0 * &b.1)? + (&a.1 * &b.0)?)?))
}

/// a ⊙ conj(b): unbinding with b's involution.
pub fn cmul_conj(a: &(Tensor, Tensor), b: &(Tensor, Tensor)) -> Result<(Tensor, Tensor)> {
    Ok((((&a.0 * &b.0)? + (&a.1 * &b.1)?)?, ((&a.1 * &b.0)? - (&a.0 * &b.1)?)?))
}

pub struct HoloMem {
    pub wr: Tensor,
    pub wf: Tensor,
    pub wq: Tensor,
    pub wo: Tensor,
    /// [1]: λ = sigmoid(lam), starting at 0.95
    pub lam: Tensor,
    pub dft: Dft,
    pub db: usize,
}

/// The trace carried between pieces of a text: (Re T, Im T) [B, db].
pub type HoloState = (Tensor, Tensor);

fn unit(x: &Tensor) -> Result<Tensor> {
    x.broadcast_div(&(x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?)
}

impl HoloMem {
    pub fn new(vb: VarBuilder, d: usize, db: usize) -> Result<HoloMem> {
        let std = (1.0 / d as f64).sqrt();
        let r = |n: &str| vb.get_with_hints((d, db), n, Init::Randn { mean: 0.0, stdev: std });
        Ok(HoloMem {
            wr: r("wr")?,
            wf: r("wf")?,
            wq: r("wq")?,
            wo: vb.get_with_hints((db, d), "wo", Init::Const(0.0))?,
            lam: vb.get_with_hints(1, "lam", Init::Const(logit(0.95)))?,
            dft: Dft::new(db, vb.device())?,
            db,
        })
    }

    pub fn lambda(&self) -> Result<Tensor> {
        candle_nn::ops::sigmoid(&self.lam)
    }

    /// The traces before each token [B, T, db] (Re, Im) for bindings B_t, from `init` (zeros if None), and the trace
    /// after the last token. T_before_i = λ^i·T_init + Σ_{j<i} λ^(i−1−j)·B_j.
    pub fn traces(&self, b: &(Tensor, Tensor), init: Option<&HoloState>) -> Result<((Tensor, Tensor), HoloState)> {
        let (bs, t, db) = b.0.dims3()?;
        let dev = b.0.device();
        let ll = candle_nn::ops::sigmoid(&self.lam)?.log()?; // ln λ [1]
        // the decay matrix [T + 1, T]: rows 0..T are the traces before each token, row T the trace after the last
        let mut e = Vec::with_capacity((t + 1) * t);
        let mut m = Vec::with_capacity((t + 1) * t);
        for i in 0..=t {
            for j in 0..t {
                let on = j < i;
                e.push(if on { (i - 1 - j) as f32 } else { 0.0 });
                m.push(if on { 1.0f32 } else { 0.0 });
            }
        }
        let ex = Tensor::from_vec(e, (t + 1, t), dev)?;
        let mk = Tensor::from_vec(m, (t + 1, t), dev)?;
        let l = ex.broadcast_mul(&ll)?.exp()?.mul(&mk)?;
        let lb = l.unsqueeze(0)?.broadcast_as((bs, t + 1, t))?.contiguous()?;
        let mut tr = lb.matmul(&b.0.contiguous()?)?;
        let mut ti = lb.matmul(&b.1.contiguous()?)?;
        if let Some((r0, i0)) = init {
            let p: Vec<f32> = (0..=t).map(|i| i as f32).collect();
            let pw = Tensor::from_vec(p, (t + 1, 1), dev)?.broadcast_mul(&ll)?.exp()?; // λ^i [T + 1, 1]
            tr = tr.add(&pw.unsqueeze(0)?.broadcast_mul(&r0.unsqueeze(1)?)?)?;
            ti = ti.add(&pw.unsqueeze(0)?.broadcast_mul(&i0.unsqueeze(1)?)?)?;
        }
        let before = (tr.narrow(1, 0, t)?, ti.narrow(1, 0, t)?);
        let end = (tr.narrow(1, t, 1)?.squeeze(1)?, ti.narrow(1, t, 1)?.squeeze(1)?);
        debug_assert_eq!(db, self.db);
        Ok((before, end))
    }

    /// `traces` for long texts, in one pass of a fixed number of ops: the text is cut into chunks of c = HOLO_CHUNK
    /// tokens (zeros in front to fill the first; a zero binding adds nothing), each chunk's own traces come from a
    /// [c + 1, c] decay matrix (all chunks at once), and the trace carried into chunk k from the earlier chunks' ends E,
    /// S_k = Σ_{k'<k} λ^(c·(k−1−k'))·E_k', from a small [chunks, chunks] one. The cost is T·c, not T².
    pub fn traces_chunked(&self, b: &(Tensor, Tensor), init: Option<&HoloState>) -> Result<((Tensor, Tensor), HoloState)> {
        let (bs, t, db) = b.0.dims3()?;
        if t <= HOLO_CHUNK {
            return self.traces(b, init);
        }
        let dev = b.0.device();
        let c = HOLO_CHUNK;
        let nc = t.div_ceil(c);
        let pad = nc * c - t;
        let ll = candle_nn::ops::sigmoid(&self.lam)?.log()?; // ln λ [1]
        let host = |rows: usize, cols: usize, f: &dyn Fn(usize, usize) -> Option<f32>| -> Result<(Tensor, Tensor)> {
            let mut e = Vec::with_capacity(rows * cols);
            let mut m = Vec::with_capacity(rows * cols);
            for i in 0..rows {
                for j in 0..cols {
                    let v = f(i, j);
                    e.push(v.unwrap_or(0.0));
                    m.push(if v.is_some() { 1.0f32 } else { 0.0 });
                }
            }
            Ok((Tensor::from_vec(e, (rows, cols), dev)?, Tensor::from_vec(m, (rows, cols), dev)?))
        };
        let decay = |(ex, mk): (Tensor, Tensor)| ex.broadcast_mul(&ll)?.exp()?.mul(&mk);
        let lc = decay(host(c + 1, c, &|i, j| (j < i).then(|| (i - 1 - j) as f32))?)?;
        let mc = decay(host(nc, nc, &|k, j| (j < k).then(|| (c * (k - 1 - j)) as f32))?)?;
        let pw = decay(host(c, 1, &|i, _| Some(i as f32))?)?.reshape((1, 1, c, 1))?; // λ^i within a chunk
        let lam_c = (ll.affine(c as f64, 0.0)?).exp()?; // λ^c [1]
        let lb = lc.unsqueeze(0)?.broadcast_as((bs * nc, c + 1, c))?.contiguous()?;
        let mb = mc.unsqueeze(0)?.broadcast_as((bs, nc, nc))?.contiguous()?;
        let one = |x: &Tensor| -> Result<(Tensor, Tensor)> {
            let x = if pad > 0 { Tensor::cat(&[Tensor::zeros((bs, pad, db), x.dtype(), dev)?, x.clone()], 1)? } else { x.contiguous()? };
            let tr = lb.matmul(&x.reshape((bs * nc, c, db))?)?; // [bs·nc, c + 1, db]
            let ends = tr.narrow(1, c, 1)?.reshape((bs, nc, db))?;
            let s_in = mb.matmul(&ends)?; // [bs, nc, db]
            let before = tr.narrow(1, 0, c)?.reshape((bs, nc, c, db))?.add(&pw.broadcast_mul(&s_in.unsqueeze(2)?)?)?;
            let before = before.reshape((bs, nc * c, db))?.narrow(1, pad, t)?;
            let last = s_in.narrow(1, nc - 1, 1)?.squeeze(1)?.broadcast_mul(&lam_c)?.add(&ends.narrow(1, nc - 1, 1)?.squeeze(1)?)?;
            Ok((before, last))
        };
        let (mut br, mut er) = one(&b.0)?;
        let (mut bi, mut ei) = one(&b.1)?;
        if let Some((r0, i0)) = init {
            let p = decay(host(t + 1, 1, &|i, _| Some(i as f32))?)?; // λ^i [T + 1, 1]
            let pb = p.narrow(0, 0, t)?.unsqueeze(0)?;
            br = br.add(&pb.broadcast_mul(&r0.unsqueeze(1)?)?)?;
            bi = bi.add(&pb.broadcast_mul(&i0.unsqueeze(1)?)?)?;
            let pt = p.narrow(0, t, 1)?.reshape(1)?;
            er = er.add(&r0.broadcast_mul(&pt)?)?;
            ei = ei.add(&i0.broadcast_mul(&pt)?)?;
        }
        Ok(((br, bi), (er, ei)))
    }

    /// What each token reads from the bindings before it, [B, T, db]: r, f, q given directly (the core of forward).
    pub fn read_codes(&self, r: &Tensor, f: &Tensor, q: &Tensor, init: Option<&HoloState>) -> Result<(Tensor, HoloState)> {
        let bound = cmul(&self.dft.forward(r)?, &self.dft.forward(f)?)?;
        let (before, end) = self.traces_chunked(&bound, init)?;
        let u = cmul_conj(&before, &self.dft.forward(q)?)?;
        Ok((self.dft.inverse(&u.0, &u.1)?, end))
    }

    /// h [B, T, d] → (what joins the stream [B, T, d], the trace after the last token), from a carried trace.
    pub fn forward_from(&self, h: &Tensor, init: Option<&HoloState>) -> Result<(Tensor, HoloState)> {
        let r = unit(&linear(h, &self.wr)?)?;
        let f = linear(h, &self.wf)?;
        let q = unit(&linear(h, &self.wq)?)?;
        let (u, end) = self.read_codes(&r, &f, &q, init)?;
        Ok((linear(&u, &self.wo)?, end))
    }

    pub fn forward(&self, h: &Tensor) -> Result<Tensor> {
        Ok(self.forward_from(h, None)?.0)
    }

    /// A fresh trace for `b` rows.
    pub fn zero_state(&self, b: usize) -> Result<HoloState> {
        let z = Tensor::zeros((b, self.db), candle_core::DType::F32, self.wo.device())?;
        Ok((z.clone(), z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binding::{bind, cosine, unbind};
    use crate::testkit::*;

    fn vecs(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let v: Vec<f32> = randn(&[d], seed + i as u64).to_vec1().unwrap();
                let s = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter().map(|x| x / s).collect()
            })
            .collect()
    }

    fn row(v: &[Vec<f32>]) -> Tensor {
        let d = v[0].len();
        Tensor::from_vec(v.concat(), (1, v.len(), d), &Device::Cpu).unwrap()
    }

    /// The frequency-domain bind and unbind are binding::bind and binding::unbind.
    #[test]
    fn dft_binding_is_the_reference_binding() {
        let d = 16;
        let dft = Dft::new(d, &Device::Cpu).unwrap();
        let v = vecs(3, d, 3);
        let (a, b, c) = (row(&v[0..1]), row(&v[1..2]), row(&v[2..3]));
        let fa = dft.forward(&a).unwrap();
        let ab = cmul(&fa, &dft.forward(&b).unwrap()).unwrap();
        let got: Vec<f32> = dft.inverse(&ab.0, &ab.1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let want = bind(&v[0], &v[1]);
        assert!(got.iter().zip(&want).all(|(x, y)| (x - y).abs() < 1e-5), "{got:?} vs {want:?}");
        let un = cmul_conj(&dft.forward(&c).unwrap(), &fa).unwrap();
        let got: Vec<f32> = dft.inverse(&un.0, &un.1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let want = unbind(&v[2], &v[0]);
        assert!(got.iter().zip(&want).all(|(x, y)| (x - y).abs() < 1e-5));
    }

    /// Three role–filler pairs, then a query with the first role: the read is the first filler (by cosine), not the
    /// others; a decay of 1 keeps every pair; a token never reads its own binding.
    #[test]
    fn the_layer_recalls_a_filler_by_its_role() {
        let (_vm, vb) = vm();
        let db = 256;
        let hm = HoloMem::new(vb, 8, db).unwrap();
        let roles = vecs(4, db, 10);
        let fillers = vecs(4, db, 20);
        // tokens 0..3 bind (role i, filler i); token 3 queries role 0 and binds nothing useful (role 3, filler 3)
        let r = row(&roles);
        let f = row(&fillers);
        let q = row(&[roles[3].clone(), roles[3].clone(), roles[3].clone(), roles[0].clone()]);
        let lam_one = Tensor::new(&[30f32], &Device::Cpu).unwrap();
        let hm = HoloMem { lam: lam_one, ..hm };
        let (u, _) = hm.read_codes(&r, &f, &q, None).unwrap();
        let u3: Vec<f32> = u.narrow(1, 3, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let c: Vec<f32> = fillers.iter().map(|x| cosine(&u3, x)).collect();
        assert!(c[0] > 0.45 && c[1].abs() < 0.2 && c[2].abs() < 0.2 && c[3].abs() < 0.2, "{c:?}");
        let u0: Vec<f32> = u.narrow(1, 0, 1).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        assert!(u0.iter().all(|x| x.abs() < 1e-6), "the first token has nothing before it");
    }

    /// Reading a long text in chunks gives the same numbers as one decay matrix over all of it.
    #[test]
    fn chunked_read_is_the_whole_read() {
        let (_vm, vb) = vm();
        let hm = HoloMem::new(vb, 8, 16).unwrap();
        // λ = 0.995 so the carry across chunks is large enough to matter
        let hm = HoloMem { lam: Tensor::new(&[logit(0.995) as f32], &Device::Cpu).unwrap(), ..hm };
        let init = (randn(&[2, 16], 4), randn(&[2, 16], 5));
        for (t, with_init) in [(HOLO_CHUNK * 2 + 88, false), (HOLO_CHUNK * 2 + 88, true), (HOLO_CHUNK * 2, true), (HOLO_CHUNK + 1, true)] {
            let (r, f, q) = (randn(&[2, t, 16], 1), randn(&[2, t, 16], 2), randn(&[2, t, 16], 3));
            let st = with_init.then_some(&init);
            let (got, end) = hm.read_codes(&r, &f, &q, st).unwrap();
            let bound = cmul(&hm.dft.forward(&r).unwrap(), &hm.dft.forward(&f).unwrap()).unwrap();
            let (before, end1) = hm.traces(&bound, st).unwrap();
            let u = cmul_conj(&before, &hm.dft.forward(&q).unwrap()).unwrap();
            let want = hm.dft.inverse(&u.0, &u.1).unwrap();
            let scale = want.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            let es = end1.0.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
            assert!(max_diff(&got, &want) < 1e-5 * scale.max(1.0), "t {t}: {} (scale {scale})", max_diff(&got, &want));
            assert!(max_diff(&end.0, &end1.0) < 1e-5 * es.max(1.0) && max_diff(&end.1, &end1.1) < 1e-5 * es.max(1.0), "t {t} end");
        }
        // and the gradient through λ is the same
        let lam = candle_core::Var::new(&[logit(0.995) as f32], &Device::Cpu).unwrap();
        let hm = HoloMem { lam: lam.as_tensor().clone(), ..hm };
        let t = HOLO_CHUNK * 2 + 88;
        let (r, f) = (randn(&[2, t, 16], 6), randn(&[2, t, 16], 7));
        let bound = cmul(&hm.dft.forward(&r).unwrap(), &hm.dft.forward(&f).unwrap()).unwrap();
        let g = |chunked: bool| -> f32 {
            let (b, e) = if chunked { hm.traces_chunked(&bound, Some(&init)) } else { hm.traces(&bound, Some(&init)) }.unwrap();
            let loss = (b.0.sqr().unwrap().sum_all().unwrap() + b.1.sqr().unwrap().sum_all().unwrap() + e.0.sum_all().unwrap()).unwrap();
            loss.backward().unwrap().get(lam.as_tensor()).unwrap().to_vec1::<f32>().unwrap()[0]
        };
        let (gc, gw) = (g(true), g(false));
        assert!((gc - gw).abs() < 1e-4 * gw.abs().max(1.0), "dλ: {gc} vs {gw}");
    }

    /// Starts as no change; read in pieces with the carried trace equals one read; every weight learns.
    #[test]
    fn starts_silent_reads_in_pieces_and_learns() {
        let (vm, vb) = vm();
        let d = 12;
        let hm = HoloMem::new(vb, d, 16).unwrap();
        let h = randn(&[2, 20, d], 4);
        assert_eq!(hm.forward(&h).unwrap().abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap(), 0.0);
        let sd = (1.0 / d as f64).sqrt();
        seed(&vm, &[("wr", sd), ("wf", sd), ("wq", sd), ("wo", 1.0)], 5);
        let whole = hm.forward(&h).unwrap();
        let scale = whole.abs().unwrap().max_all().unwrap().to_scalar::<f32>().unwrap();
        let mut st = Some(hm.zero_state(2).unwrap());
        let mut outs = Vec::new();
        let mut p0 = 0;
        for p in [7usize, 1, 12] {
            let (o, s) = hm.forward_from(&h.narrow(1, p0, p).unwrap(), st.as_ref()).unwrap();
            outs.push(o);
            st = Some(s);
            p0 += p;
        }
        let cat = Tensor::cat(&outs, 1).unwrap();
        assert!(max_diff(&cat, &whole) < 1e-5 * scale.max(1.0), "{} (scale {scale})", max_diff(&cat, &whole));
        let loss = whole.sqr().unwrap().sum_all().unwrap();
        let g = grad_norms(&vm, &loss, &["wr", "wf", "wq", "wo", "lam"]);
        assert!(g.iter().all(|&x| x > 1e-6 && x.is_finite()), "{g:?}");
    }
}
