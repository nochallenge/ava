// TESTING / DESIGN ONLY — end-to-end proof that ava_grad learns (target: well under 30 s total).
// Three tiny next-token models on synthetic sequences (vocab 16, 16 sequences × 32 positions per step):
//   (a) MLP:       embedding → rmsnorm → gelu MLP (residual) → rmsnorm → logits.
//                  Task "map": x_{i+1} = (5·x_i + 3) mod 16 (solvable from the current token alone).
//   (b) attention: token + position embeddings → rmsnorm → 2-head causal self-attention (residual)
//                  → rmsnorm → logits. Task "copy": a random 8-token pattern repeated; the first 7 targets
//                  are unpredictable and marked IGNORE, every later one needs a look back 8 positions.
//   (c) spiking:   embedding → rmsnorm → linear → kWTA (16 of 64) → LIF scan over time (hard spikes,
//                  surrogate gradient, learnable leak) → linear readout. Task "map".
// Each prints its loss at the start and the end (mean of the first / last 10 steps), held-out accuracy,
// and the wall time. Exits non-zero if any loss does not fall by at least half.
// Run: CARGO_BUILD_JOBS=4 cargo run --release --example learn_smoke

use ava_grad::*;
use std::time::Instant;

const V: usize = 16;
const T: usize = 32;
const B: usize = 16;
const P: usize = 8;

fn map_batch(rng: &mut Rng) -> (Vec<u32>, Vec<u32>) {
    let (mut ids, mut tg) = (Vec::with_capacity(B * T), Vec::with_capacity(B * T));
    for _ in 0..B {
        let mut x = rng.below(V);
        for _ in 0..T {
            let y = (5 * x + 3) % V;
            ids.push(x as u32);
            tg.push(y as u32);
            x = y;
        }
    }
    (ids, tg)
}

fn copy_batch(rng: &mut Rng) -> (Vec<u32>, Vec<u32>) {
    let (mut ids, mut tg) = (Vec::with_capacity(B * T), Vec::with_capacity(B * T));
    for _ in 0..B {
        let pat: Vec<u32> = (0..P).map(|_| rng.below(V) as u32).collect();
        let seq: Vec<u32> = (0..=T).map(|i| pat[i % P]).collect();
        for i in 0..T {
            ids.push(seq[i]);
            tg.push(if i + 1 < P { IGNORE } else { seq[i + 1] });
        }
    }
    (ids, tg)
}

struct Report {
    start: f32,
    end: f32,
    acc: f32,
    ms: u128,
    extra: String,
}

/// Generic loop: AdamW, warmup + cosine schedule, grad clipping at 1.0.
/// `model` returns (logits [B*T, V], optional Var whose activity fraction is reported).
fn train<D, M>(params: &mut Params, steps: usize, max_lr: f32, seed: u64, data: D, model: M) -> Report
where
    D: Fn(&mut Rng) -> (Vec<u32>, Vec<u32>),
    M: Fn(&mut Graph, &[u32]) -> (Var, Option<Var>),
{
    let t0 = Instant::now();
    let mut rng = Rng::new(seed);
    let mut opt = AdamW::new(0.01);
    let mut losses = Vec::with_capacity(steps);
    for step in 0..steps {
        let (ids, tg) = data(&mut rng);
        let (loss, grads) = {
            let mut g = Graph::new(params);
            let (logits, _) = model(&mut g, &ids);
            let l = g.cross_entropy(logits, &tg);
            (g.scalar(l), g.backward(l))
        };
        losses.push(loss);
        params.accumulate(&grads);
        clip_grad_norm(params, 1.0);
        opt.step(params, cosine_lr(step, steps / 15, steps, max_lr, max_lr * 0.1));
        params.zero_grads();
    }
    // held-out batch
    let (ids, tg) = data(&mut Rng::new(seed ^ 0xDEAD_BEEF));
    let mut g = Graph::new(params);
    let (logits, act) = model(&mut g, &ids);
    let acc = accuracy(g.value(logits), &tg);
    let extra = act.map(|a| format!("  spike activity {:.3}", fraction_active(g.value(a)))).unwrap_or_default();
    let k = 10.min(steps);
    Report {
        start: losses[..k].iter().sum::<f32>() / k as f32,
        end: losses[steps - k..].iter().sum::<f32>() / k as f32,
        acc,
        ms: t0.elapsed().as_millis(),
        extra,
    }
}

fn mlp() -> Report {
    let (d, h) = (32, 64);
    let mut rng = Rng::new(10);
    let mut p = Params::new();
    let emb = p.add("emb", Tensor::randn(&[V, d], 1.0, &mut rng), false);
    let g1 = p.add("g1", Tensor::full(&[d], 1.0), false);
    let w1 = p.add("w1", Tensor::randn(&[d, h], 1.0 / (d as f32).sqrt(), &mut rng), true);
    let b1 = p.add("b1", Tensor::zeros(&[h]), false);
    let w2 = p.add("w2", Tensor::randn(&[h, d], 1.0 / (h as f32).sqrt(), &mut rng), true);
    let gf = p.add("gf", Tensor::full(&[d], 1.0), false);
    let wout = p.add("wout", Tensor::randn(&[d, V], 1.0 / (d as f32).sqrt(), &mut rng), true);
    train(&mut p, 300, 1e-2, 1, map_batch, |g, ids| {
        let e = g.param(emb);
        let x = g.embedding(e, ids);
        let gg = g.param(g1);
        let hh = g.rmsnorm(x, gg, 1e-5);
        let (a, b, c) = (g.param(w1), g.param(b1), g.param(w2));
        let m = g.matmul(hh, a);
        let m = g.add_bias(m, b);
        let m = g.gelu(m);
        let m = g.matmul(m, c);
        let x = g.add(x, m);
        let gfv = g.param(gf);
        let x = g.rmsnorm(x, gfv, 1e-5);
        let wo = g.param(wout);
        (g.matmul(x, wo), None)
    })
}

fn attention() -> Report {
    let (d, heads) = (32, 2);
    let dh = d / heads;
    let s = 1.0 / (d as f32).sqrt();
    let mut rng = Rng::new(20);
    let mut p = Params::new();
    let tok = p.add("tok", Tensor::randn(&[V, d], 1.0, &mut rng), false);
    let pos = p.add("pos", Tensor::randn(&[T, d], 1.0, &mut rng), false);
    let g1 = p.add("g1", Tensor::full(&[d], 1.0), false);
    let wq = p.add("wq", Tensor::randn(&[d, d], s, &mut rng), true);
    let wk = p.add("wk", Tensor::randn(&[d, d], s, &mut rng), true);
    let wv = p.add("wv", Tensor::randn(&[d, d], s, &mut rng), true);
    let wo = p.add("wo", Tensor::randn(&[d, d], s, &mut rng), true);
    let gf = p.add("gf", Tensor::full(&[d], 1.0), false);
    let wout = p.add("wout", Tensor::randn(&[d, V], s, &mut rng), true);
    let pos_ids: Vec<u32> = (0..B * T).map(|i| (i % T) as u32).collect();
    train(&mut p, 600, 1e-2, 2, copy_batch, |g, ids| {
        let (te, pe) = (g.param(tok), g.param(pos));
        let xt = g.embedding(te, ids);
        let xp = g.embedding(pe, &pos_ids);
        let x = g.add(xt, xp);
        let gg = g.param(g1);
        let h = g.rmsnorm(x, gg, 1e-5);
        let split = |g: &mut Graph, w: ParamId| {
            let wv = g.param(w);
            let y = g.matmul(h, wv);
            let y = g.reshape(y, &[B, T, heads, dh]);
            g.transpose(y, 1, 2) // [B, heads, T, dh]
        };
        let (q, k, v) = (split(g, wq), split(g, wk), split(g, wv));
        let sc = g.matmul_t(q, k);
        let sc = g.scale(sc, 1.0 / (dh as f32).sqrt());
        let att = g.softmax_last(sc, true);
        let o = g.matmul(att, v);
        let o = g.transpose(o, 1, 2);
        let o = g.reshape(o, &[B * T, d]);
        let wov = g.param(wo);
        let o = g.matmul(o, wov);
        let x = g.add(x, o);
        let gfv = g.param(gf);
        let x = g.rmsnorm(x, gfv, 1e-5);
        let w = g.param(wout);
        (g.matmul(x, w), None)
    })
}

fn spiking() -> Report {
    let (d, hid, k) = (32, 64, 16);
    let mut rng = Rng::new(30);
    let mut p = Params::new();
    let emb = p.add("emb", Tensor::randn(&[V, d], 1.0, &mut rng), false);
    let g1 = p.add("g1", Tensor::full(&[d], 1.0), false);
    let win = p.add("win", Tensor::randn(&[d, hid], 1.5 / (d as f32).sqrt(), &mut rng), true);
    let beta = p.add("lif.beta", Tensor::zeros(&[hid]), false); // β = σ(0) = 0.5 to start
    let wout = p.add("wout", Tensor::randn(&[hid, V], 1.0 / (k as f32).sqrt(), &mut rng), true);
    let bout = p.add("bout", Tensor::zeros(&[V]), false);
    train(&mut p, 300, 1e-2, 3, map_batch, |g, ids| {
        let e = g.param(emb);
        let x = g.embedding(e, ids);
        let gg = g.param(g1);
        let x = g.rmsnorm(x, gg, 1e-5);
        let w = g.param(win);
        let c = g.matmul(x, w);
        let c = g.kwta(c, k);
        let c = g.reshape(c, &[B, T, hid]);
        let bp = g.param(beta);
        let s = g.lif_scan(c, bp, 1.0, SpikeFn::Surrogate { slope: 4.0 });
        let s2 = g.reshape(s, &[B * T, hid]);
        let (wo, bo) = (g.param(wout), g.param(bout));
        let z = g.matmul(s2, wo);
        (g.add_bias(z, bo), Some(s))
    })
}

fn main() {
    let total = Instant::now();
    println!("ava_grad learn_smoke (vocab {V}, {B} seqs x {T} positions per step; chance loss ln16 = {:.3})", (V as f32).ln());
    println!("{:<34} {:>10} {:>10} {:>9} {:>8}", "variant", "start", "end", "acc", "ms");
    let mut ok = true;
    for (name, r) in [
        ("(a) MLP, map task", mlp()),
        ("(b) causal attention, copy task", attention()),
        ("(c) kWTA + LIF scan, map task", spiking()),
    ] {
        println!("{:<34} {:>10.4} {:>10.4} {:>9.3} {:>8}{}", name, r.start, r.end, r.acc, r.ms, r.extra);
        ok &= r.end < 0.5 * r.start;
    }
    println!("total {:.2} s", total.elapsed().as_secs_f32());
    if !ok {
        eprintln!("FAIL: a loss did not fall by half");
        std::process::exit(1);
    }
}
