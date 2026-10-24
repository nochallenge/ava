// TESTING / DESIGN ONLY — ava_brain::trainer: one optimisation step, shared by the train binary,
// the profiler and the tests.
//
// Step s (0-based): lr = cosine_lr(s, warmup, total_steps, lr_max, lr_max · min_lr_frac);
// forward → mean cross-entropy (IGNORE targets skipped) → backward → accumulate into the store →
// clip_grad_norm(clip) (the returned norm is the one before clipping) → AdamW(wd) step → zero grads.
// Everything is deterministic for a given seed and thread count (ava_grad's kernels are
// bit-identical for any thread count; the data order comes from the ava_data sampler seed).

use crate::model::{Config, Model, OpTimes};
use ava_grad::{clip_grad_norm, cosine_lr, AdamW, Graph, Params};
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct Hyper {
    pub lr: f32,
    pub warmup: usize,
    pub min_lr_frac: f32,
    pub wd: f32,
    pub clip: f32,
    /// Length of the LR schedule (steps = tokens / (batch · ctx)).
    pub total_steps: usize,
    pub threads: usize,
}

#[derive(Clone, Debug, Default)]
pub struct StepStats {
    pub loss: f32,
    pub lr: f32,
    pub grad_norm: f32,
    pub fwd_s: f64,
    pub bwd_s: f64,
    /// accumulate + clip + AdamW + zero_grads + freeing the tape.
    pub opt_s: f64,
    pub total_s: f64,
    pub fwd_ops: OpTimes,
}

pub struct Trainer {
    pub model: Model,
    pub params: Params,
    pub opt: AdamW,
    pub hyper: Hyper,
    pub step: usize,
}

impl Trainer {
    pub fn new(cfg: Config, seed: u64, hyper: Hyper) -> Trainer {
        let (model, params) = Model::new(cfg, seed);
        Trainer { model, params, opt: AdamW::new(hyper.wd), hyper, step: 0 }
    }

    pub fn lr_at(&self, step: usize) -> f32 {
        let h = &self.hyper;
        cosine_lr(step, h.warmup, h.total_steps, h.lr, h.lr * h.min_lr_frac)
    }

    /// One training step on `b` sequences of `t` tokens.
    pub fn train_step(&mut self, ids: &[u32], targets: &[u32], b: usize, t: usize) -> StepStats {
        let t0 = Instant::now();
        let lr = self.lr_at(self.step);
        let (loss, grads, fwd_s, bwd_s, fwd_ops, t_drop) = {
            let mut g = Graph::new(&self.params);
            g.set_threads(self.hyper.threads);
            let (l, f) = self.model.loss(&mut g, ids, targets, b, t);
            let fwd_s = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            let grads = g.backward(l);
            let bwd_s = t1.elapsed().as_secs_f64();
            let loss = g.scalar(l);
            let t_drop = Instant::now();
            drop(g);
            (loss, grads, fwd_s, bwd_s, f.times, t_drop)
        };
        self.params.accumulate(&grads);
        drop(grads);
        let grad_norm = clip_grad_norm(&mut self.params, self.hyper.clip);
        self.opt.step(&mut self.params, lr);
        self.params.zero_grads();
        self.step += 1;
        let opt_s = t_drop.elapsed().as_secs_f64();
        StepStats { loss, lr, grad_norm, fwd_s, bwd_s, opt_s, total_s: t0.elapsed().as_secs_f64(), fwd_ops }
    }
}
