// TESTING / DESIGN ONLY — ava_gpu::model::fstep::prof: where one plain `Model::step` spends its time, by op class.
//
// `Model::step_profiled` is `Model::step` (today's block, the token-by-token stream) op for op, with each chain timed
// into its class and, in counting mode, the candle tensors it creates counted (every op makes one, a view too; the
// count is read from candle's tensor id counter, which itself costs one tensor per reading, taken off). Timing and
// counting run on separate steps, so the clock never sees the counting. The logits are the step's own bit for bit
// (checked by src/bin/fstep_bench.rs on every profiled step).

use super::super::{hebb_gates, head_norm, kwta_mask, l2_norm, out_gate, rmsnorm, Model, Stream, HEAD_SIZE, LIF_THRESHOLD};
use candle_core::{DType, Result, Tensor, D};
use std::time::Instant;

/// The op classes of one step.
pub const CLASSES: [&str; 10] =
    ["embed + glance + magnitude", "norms", "big matmuls (int8 or f32)", "LIF update", "gates", "fast weights", "attention + engram", "kWTA + relu", "residual adds", "reshapes"];

#[derive(Clone, Copy)]
#[repr(usize)]
enum C {
    Embed,
    Norm,
    Mm,
    Lif,
    Gates,
    Fast,
    Attn,
    Kwta,
    Resid,
    Shape,
}

/// Per class: seconds and tensors created, summed over the profiled steps.
#[derive(Default, Clone)]
pub struct StepProf {
    pub secs: [f64; 10],
    pub tensors: [u64; 10],
    pub steps: u64,
    /// counting mode (tensor counts, no times) or timing mode
    pub count: bool,
}

/// The id candle gives the next tensor (creates one).
pub fn next_tensor_id() -> u64 {
    let t = Tensor::zeros((), DType::F32, &candle_core::Device::Cpu).expect("a scalar");
    let s = format!("{:?}", t.id());
    s.trim_start_matches("TensorId(").trim_end_matches(')').parse().expect("TensorId(n)")
}

impl StepProf {
    fn sec<T>(&mut self, c: C, f: impl FnOnce() -> Result<T>) -> Result<T> {
        if self.count {
            let a = next_tensor_id();
            let r = f()?;
            self.tensors[c as usize] += next_tensor_id() - a - 1;
            Ok(r)
        } else {
            let t0 = Instant::now();
            let r = f()?;
            self.secs[c as usize] += t0.elapsed().as_secs_f64();
            Ok(r)
        }
    }
}

impl Model {
    /// `Model::step` (today's block, the token-by-token stream, fused step not used), with every chain timed or
    /// counted into `prof`. The same ops in the same order, so the same logits and state.
    pub fn step_profiled(&self, st: &mut Stream, tok: u32, prof: &mut StepProf) -> Result<Tensor> {
        let _q8 = self.int8_scope();
        let dev = self.tok.device();
        if st.st.is_some() {
            candle_core::bail!("step_profiled: today's token-by-token stream only (this one is State-backed)");
        }
        prof.steps += 1;
        let (place_off, mag_off, eng_off) = self.lesion.get();
        let mut x = prof.sec(C::Embed, || {
            let mut x = self.tok.embedding(&Tensor::new(&[tok], dev)?)?; // [1, d]
            let p = st.next_place(tok) as usize;
            if let (Some(place), false) = (&self.place, place_off) {
                x = x.add(&Self::place_table(place)?.narrow(0, p, 1)?)?;
            }
            st.hist.push(tok);
            if st.hist.len() > 17 {
                st.hist.remove(0);
            }
            if let (Some(mw), false) = (&self.mag_w, mag_off) {
                if let Some(m) = crate::mag::position(&st.hist, st.hist.len() - 1) {
                    let f = Tensor::from_vec(crate::mag::code(m).to_vec(), (1, crate::mag::MAG_F), dev)?;
                    x = x.add(&f.matmul(mw)?)?;
                }
            }
            Ok(x)
        })?;
        for (i, blk) in self.blocks.iter().enumerate() {
            let m = blk.mem.as_ref().expect("S block has SpikeMem");
            let h = prof.sec(C::Norm, || rmsnorm(&x, &m.norm))?;
            let cur = prof.sec(C::Mm, || crate::int8::mm(&h, &m.win))?;
            let (v, s) = prof.sec(C::Lif, || {
                let v = st.v[i].mul(&st.beta[i])?.add(&cur)?.sub(&st.s[i].affine(LIF_THRESHOLD, 0.0)?)?;
                let s = v.affine(1.0, -LIF_THRESHOLD)?.ge(0.0)?.to_dtype(DType::F32)?;
                Ok((v, s))
            })?;
            let o = prof.sec(C::Mm, || crate::int8::mm(&s, &m.wout))?;
            x = prof.sec(C::Resid, || x.add(&o))?;
            st.v[i] = v;
            st.s[i] = s;
            if let Some(hb) = &blk.hebb {
                let nh = self.cfg.heads();
                let h = prof.sec(C::Norm, || rmsnorm(&x, &hb.norm))?;
                let (q, k) = prof.sec(C::Mm, || Ok((crate::int8::mm(&h, &hb.wq)?, crate::int8::mm(&h, &hb.wk)?)))?;
                let (mut q, mut k) = prof.sec(C::Shape, || Ok((q.reshape((nh, 1, HEAD_SIZE))?, k.reshape((nh, 1, HEAD_SIZE))?)))?;
                if hb.qk_norm {
                    (q, k) = prof.sec(C::Norm, || Ok((l2_norm(&q)?, l2_norm(&k)?)))?;
                }
                let (k_row, k) = prof.sec(C::Shape, || Ok((k.clone(), k.reshape((nh, HEAD_SIZE, 1))?)))?;
                let v = prof.sec(C::Mm, || crate::int8::mm(&h, &hb.wv))?;
                let v = prof.sec(C::Shape, || v.reshape((nh, 1, HEAD_SIZE)))?;
                let imprint = prof.sec(C::Fast, || k.matmul(&v))?;
                let fw = match (&hb.gates, &st.gamma) {
                    (Some(g), _) if hb.delta => {
                        let (la, beta) = prof.sec(C::Gates, || hebb_gates(&h, g))?;
                        let (alpha, beta) = prof.sec(C::Gates, || Ok((la.exp()?.reshape((nh, 1, 1))?, beta.reshape((nh, 1, 1))?)))?;
                        prof.sec(C::Fast, || {
                            let pred = k_row.matmul(&st.fw[i])?;
                            let corr = k.matmul(&pred)?.broadcast_mul(&beta)?;
                            st.fw[i].sub(&corr)?.broadcast_mul(&alpha)?.add(&imprint.broadcast_mul(&beta)?)
                        })?
                    }
                    (Some(g), _) => {
                        let (la, beta) = prof.sec(C::Gates, || hebb_gates(&h, g))?;
                        prof.sec(C::Fast, || {
                            let alpha = la.exp()?.reshape((nh, 1, 1))?;
                            st.fw[i].broadcast_mul(&alpha)?.add(&imprint.broadcast_mul(&beta.reshape((nh, 1, 1))?)?)
                        })?
                    }
                    (None, Some(gamma)) => prof.sec(C::Fast, || st.fw[i].broadcast_mul(gamma)?.add(&imprint))?,
                    (None, None) => candle_core::bail!("step: fast weights without forgetting rates"),
                };
                let read = prof.sec(C::Fast, || {
                    let read = q.matmul(&fw)?;
                    if hb.delta {
                        Ok(read)
                    } else {
                        read.affine(1.0 / (HEAD_SIZE as f64).sqrt(), 0.0)
                    }
                })?;
                let y = prof.sec(C::Norm, || head_norm(&read))?;
                let y = prof.sec(C::Shape, || y.reshape((1, self.cfg.d)))?;
                // out_gate: the gate's matmul is a big one, its silu and product a gate op
                let gated = match &hb.wg {
                    Some(wg) => {
                        let g = prof.sec(C::Mm, || crate::int8::mm(&h, wg))?;
                        prof.sec(C::Gates, || y.mul(&candle_nn::ops::silu(&g)?))?
                    }
                    None => prof.sec(C::Gates, || out_gate(&y, &h, hb))?,
                };
                let o = prof.sec(C::Mm, || crate::int8::mm(&gated, &hb.wo))?;
                x = prof.sec(C::Resid, || x.add(&o))?;
                st.fw[i] = fw;
            }
            if let Some(a) = &blk.attn {
                let nh = self.cfg.heads();
                let h = prof.sec(C::Norm, || rmsnorm(&x, &a.norm))?;
                let (q, k, v) = prof.sec(C::Mm, || Ok((crate::int8::mm(&h, &a.wq)?, crate::int8::mm(&h, &a.wk)?, crate::int8::mm(&h, &a.wv)?)))?;
                let (q, k, v) = prof.sec(C::Shape, || Ok((q.reshape((nh, 1, HEAD_SIZE))?, k.reshape((nh, 1, HEAD_SIZE))?, v.reshape((nh, 1, HEAD_SIZE))?)))?;
                let (o, kc, vc) = prof.sec(C::Attn, || {
                    let (kc, vc) = match st.kv[i].take() {
                        None => (k, v),
                        Some((kp, vp)) => (Tensor::cat(&[&kp, &k], 1)?, Tensor::cat(&[&vp, &v], 1)?),
                    };
                    let sc = q.matmul(&kc.t()?.contiguous()?)?.affine(1.0 / (HEAD_SIZE as f64).sqrt(), 0.0)?;
                    let o = candle_nn::ops::softmax(&sc, D::Minus1)?.matmul(&vc)?;
                    Ok((o, kc, vc))
                })?;
                let o = prof.sec(C::Shape, || o.reshape((1, self.cfg.d)))?;
                let o = prof.sec(C::Mm, || crate::int8::mm(&o, &a.wo))?;
                x = prof.sec(C::Resid, || x.add(&o))?;
                st.kv[i] = Some((kc, vc));
            }
            if let Some(e) = &blk.eng {
                let nh = self.cfg.heads();
                let h = prof.sec(C::Norm, || rmsnorm(&x, &e.norm))?;
                let (q, k, v) = prof.sec(C::Mm, || Ok((crate::int8::mm(&h, &e.wq)?, crate::int8::mm(&h, &e.wk)?, crate::int8::mm(&h, &e.wv)?)))?;
                let (q, k, v) = prof.sec(C::Shape, || Ok((q.reshape((nh, 1, HEAD_SIZE))?, k.reshape((nh, 1, HEAD_SIZE))?, v.reshape((nh, 1, HEAD_SIZE))?)))?;
                let (kc, vc) = prof.sec(C::Attn, || {
                    Ok(match st.ekv.take() {
                        None => (k, v),
                        Some((kp, vp)) => (Tensor::cat(&[&kp, &k], 1)?, Tensor::cat(&[&vp, &v], 1)?),
                    })
                })?;
                if !eng_off {
                    let o = prof.sec(C::Attn, || {
                        let sc = q.matmul(&kc.t()?.contiguous()?)?.affine(1.0 / (HEAD_SIZE as f64).sqrt(), 0.0)?;
                        candle_nn::ops::softmax(&sc, D::Minus1)?.matmul(&vc)
                    })?;
                    let o = prof.sec(C::Shape, || o.reshape((1, self.cfg.d)))?;
                    let o = prof.sec(C::Mm, || crate::int8::mm(&o, &e.wo))?;
                    x = prof.sec(C::Resid, || x.add(&o))?;
                }
                st.ekv = Some((kc, vc));
            }
            let l = &blk.mlp;
            let h = prof.sec(C::Norm, || rmsnorm(&x, &l.norm))?;
            let u = prof.sec(C::Mm, || crate::int8::mm(&h, &l.w1))?;
            let a = prof.sec(C::Kwta, || u.mul(&kwta_mask(&u, self.cfg.kwta_k())?)?.relu())?;
            let o = prof.sec(C::Mm, || crate::int8::mm(&a, &l.w2))?;
            x = prof.sec(C::Resid, || x.add(&o))?;
        }
        st.seen += 1;
        let x = prof.sec(C::Norm, || rmsnorm(&x, &self.final_norm))?;
        let l = prof.sec(C::Mm, || crate::int8::head(&x, &self.tok))?;
        prof.sec(C::Shape, || l.squeeze(0))
    }
}
