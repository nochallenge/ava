// TESTING / DESIGN ONLY — finite-difference checks of every differentiable op in ava_grad::Graph.
// Each op is checked alone on small random shapes, then two composite nets (a tiny pre-norm transformer
// block with causal attention, and a spiking block with kWTA + LIF) catch interaction bugs.
#![allow(clippy::needless_range_loop)]
mod common;

use ava_grad::*;
use common::{away_from_zero, fd_check, rt, weighted};

#[test]
fn fd_matmul_flat_3d_activations() {
    let mut rng = Rng::new(1);
    let mut p = Params::new();
    let a = p.add("a", rt(&[2, 3, 4], &mut rng), true);
    let b = p.add("b", rt(&[4, 5], &mut rng), true);
    let w = rt(&[2, 3, 5], &mut rng);
    fd_check(&mut p, &[a, b], |g| {
        let (av, bv) = (g.param(a), g.param(b));
        let y = g.matmul(av, bv);
        assert_eq!(g.shape(y), &[2, 3, 5]);
        weighted(g, y, &w)
    });
}

#[test]
fn fd_matmul_batched() {
    let mut rng = Rng::new(2);
    let mut p = Params::new();
    let a = p.add("a", rt(&[2, 3, 4], &mut rng), true);
    let b = p.add("b", rt(&[2, 4, 5], &mut rng), true);
    let w = rt(&[2, 3, 5], &mut rng);
    fd_check(&mut p, &[a, b], |g| {
        let (av, bv) = (g.param(a), g.param(b));
        let y = g.matmul(av, bv);
        assert_eq!(g.shape(y), &[2, 3, 5]);
        weighted(g, y, &w)
    });
}

#[test]
fn fd_matmul_batched_4d() {
    let mut rng = Rng::new(3);
    let mut p = Params::new();
    let a = p.add("a", rt(&[2, 2, 3, 4], &mut rng), true);
    let b = p.add("b", rt(&[2, 2, 4, 3], &mut rng), true);
    let w = rt(&[2, 2, 3, 3], &mut rng);
    fd_check(&mut p, &[a, b], |g| {
        let (av, bv) = (g.param(a), g.param(b));
        let y = g.matmul(av, bv);
        weighted(g, y, &w)
    });
}

#[test]
fn fd_matmul_t_flat() {
    let mut rng = Rng::new(4);
    let mut p = Params::new();
    let a = p.add("a", rt(&[5, 4], &mut rng), true);
    let b = p.add("b", rt(&[6, 4], &mut rng), true);
    let w = rt(&[5, 6], &mut rng);
    fd_check(&mut p, &[a, b], |g| {
        let (av, bv) = (g.param(a), g.param(b));
        let y = g.matmul_t(av, bv);
        assert_eq!(g.shape(y), &[5, 6]);
        weighted(g, y, &w)
    });
}

#[test]
fn fd_matmul_t_batched() {
    let mut rng = Rng::new(5);
    let mut p = Params::new();
    let a = p.add("a", rt(&[2, 3, 4], &mut rng), true);
    let b = p.add("b", rt(&[2, 5, 4], &mut rng), true);
    let w = rt(&[2, 3, 5], &mut rng);
    fd_check(&mut p, &[a, b], |g| {
        let (av, bv) = (g.param(a), g.param(b));
        let y = g.matmul_t(av, bv);
        assert_eq!(g.shape(y), &[2, 3, 5]);
        weighted(g, y, &w)
    });
}

#[test]
fn fd_add_sub_mul() {
    let mut rng = Rng::new(6);
    let mut p = Params::new();
    let a = p.add("a", rt(&[3, 4], &mut rng), true);
    let b = p.add("b", rt(&[3, 4], &mut rng), true);
    let c = p.add("c", rt(&[3, 4], &mut rng), true);
    let w = rt(&[3, 4], &mut rng);
    fd_check(&mut p, &[a, b, c], |g| {
        let (av, bv, cv) = (g.param(a), g.param(b), g.param(c));
        let s = g.add(av, bv);
        let d = g.sub(av, cv);
        let y = g.mul(s, d);
        weighted(g, y, &w)
    });
}

#[test]
fn fd_scale_and_bias() {
    let mut rng = Rng::new(7);
    let mut p = Params::new();
    let x = p.add("x", rt(&[2, 3, 4], &mut rng), true);
    let b = p.add("b", rt(&[4], &mut rng), false);
    let w = rt(&[2, 3, 4], &mut rng);
    fd_check(&mut p, &[x, b], |g| {
        let (xv, bv) = (g.param(x), g.param(b));
        let s = g.scale(xv, -0.7);
        let y = g.add_bias(s, bv);
        weighted(g, y, &w)
    });
}

#[test]
fn fd_relu() {
    let mut rng = Rng::new(8);
    let mut p = Params::new();
    let x = p.add("x", away_from_zero(&[4, 5], &mut rng), true);
    let w = rt(&[4, 5], &mut rng);
    fd_check(&mut p, &[x], |g| {
        let xv = g.param(x);
        let y = g.relu(xv);
        weighted(g, y, &w)
    });
}

#[test]
fn fd_gelu() {
    let mut rng = Rng::new(9);
    let mut p = Params::new();
    let x = p.add("x", Tensor::randn(&[4, 5], 2.0, &mut rng), true);
    let w = rt(&[4, 5], &mut rng);
    fd_check(&mut p, &[x], |g| {
        let xv = g.param(x);
        let y = g.gelu(xv);
        weighted(g, y, &w)
    });
}

#[test]
fn fd_sigmoid_tanh_exp() {
    let mut rng = Rng::new(10);
    let mut p = Params::new();
    let x = p.add("x", Tensor::randn(&[3, 5], 1.5, &mut rng), true);
    let w = rt(&[3, 5], &mut rng);
    fd_check(&mut p, &[x], |g| {
        let xv = g.param(x);
        let s = g.sigmoid(xv);
        let t = g.tanh(xv);
        let st = g.add(s, t);
        let e = g.exp(st);
        weighted(g, e, &w)
    });
}

#[test]
fn fd_rmsnorm() {
    let mut rng = Rng::new(11);
    let mut p = Params::new();
    let x = p.add("x", rt(&[2, 3, 6], &mut rng), true);
    let gain = p.add("gain", Tensor::randn(&[6], 0.5, &mut rng), false);
    for v in p.value_mut(gain).data.iter_mut() {
        *v += 1.0;
    }
    let w = rt(&[2, 3, 6], &mut rng);
    fd_check(&mut p, &[x, gain], |g| {
        let (xv, gv) = (g.param(x), g.param(gain));
        let y = g.rmsnorm(xv, gv, 1e-5);
        weighted(g, y, &w)
    });
}

#[test]
fn fd_layernorm() {
    let mut rng = Rng::new(12);
    let mut p = Params::new();
    let x = p.add("x", rt(&[4, 6], &mut rng), true);
    let gain = p.add("gain", Tensor::randn(&[6], 0.5, &mut rng), false);
    let bias = p.add("bias", rt(&[6], &mut rng), false);
    let w = rt(&[4, 6], &mut rng);
    fd_check(&mut p, &[x, gain, bias], |g| {
        let (xv, gv, bv) = (g.param(x), g.param(gain), g.param(bias));
        let y = g.layernorm(xv, gv, bv, 1e-5);
        weighted(g, y, &w)
    });
}

#[test]
fn fd_embedding_with_repeated_ids() {
    let mut rng = Rng::new(13);
    let mut p = Params::new();
    let table = p.add("table", rt(&[5, 3], &mut rng), false);
    let ids = [1u32, 3, 1, 0, 1];
    let w = rt(&[5, 3], &mut rng);
    fd_check(&mut p, &[table], |g| {
        let tv = g.param(table);
        let y = g.embedding(tv, &ids);
        assert_eq!(g.shape(y), &[5, 3]);
        weighted(g, y, &w)
    });
}

#[test]
fn fd_reshape_and_transpose() {
    let mut rng = Rng::new(14);
    let mut p = Params::new();
    let x = p.add("x", rt(&[2, 3, 4, 2], &mut rng), true);
    let w1 = rt(&[2, 4, 3, 2], &mut rng);
    let w2 = rt(&[6, 8], &mut rng);
    fd_check(&mut p, &[x], |g| {
        let xv = g.param(x);
        let t = g.transpose(xv, 1, 2); // [2,4,3,2]
        assert_eq!(g.shape(t), &[2, 4, 3, 2]);
        let l1 = weighted(g, t, &w1);
        let t2 = g.transpose(xv, 0, 3); // [2,3,4,2]
        let r = g.reshape(t2, &[6, 8]);
        let l2 = weighted(g, r, &w2);
        g.add(l1, l2)
    });
}

#[test]
fn fd_softmax_plain() {
    let mut rng = Rng::new(15);
    let mut p = Params::new();
    let x = p.add("x", Tensor::randn(&[3, 5], 2.0, &mut rng), true);
    let w = rt(&[3, 5], &mut rng);
    fd_check(&mut p, &[x], |g| {
        let xv = g.param(x);
        let y = g.softmax_last(xv, false);
        weighted(g, y, &w)
    });
}

#[test]
fn fd_softmax_causal() {
    let mut rng = Rng::new(16);
    let mut p = Params::new();
    let x = p.add("x", Tensor::randn(&[2, 4, 4], 2.0, &mut rng), true);
    let w = rt(&[2, 4, 4], &mut rng);
    // The masked (upper-triangle) inputs legitimately get zero gradient; the lower triangle must match.
    fd_check(&mut p, &[x], |g| {
        let xv = g.param(x);
        let y = g.softmax_last(xv, true);
        weighted(g, y, &w)
    });
}

#[test]
fn fd_cross_entropy_with_ignored_target() {
    let mut rng = Rng::new(17);
    let mut p = Params::new();
    let z = p.add("logits", Tensor::randn(&[5, 7], 2.0, &mut rng), true);
    let targets = [3u32, 0, IGNORE, 6, 3];
    fd_check(&mut p, &[z], |g| {
        let zv = g.param(z);
        g.cross_entropy(zv, &targets)
    });
}

#[test]
fn fd_sum_and_mean() {
    let mut rng = Rng::new(18);
    let mut p = Params::new();
    let x = p.add("x", rt(&[3, 4], &mut rng), true);
    let w = rt(&[3, 4], &mut rng);
    fd_check(&mut p, &[x], |g| {
        let xv = g.param(x);
        let wv = g.input(w.clone());
        let m = g.mul(xv, wv);
        let mean = g.mean(m);
        let sq = g.mul(xv, xv);
        let s = g.sum(sq);
        let s = g.scale(s, 0.3);
        g.add(mean, s)
    });
}

#[test]
fn fd_kwta_well_separated() {
    // Distinct values with gaps ≫ 2·H so a ±1e-3 nudge never changes who wins.
    let mut rng = Rng::new(19);
    let (rows, d) = (4usize, 9usize);
    let mut data = Vec::new();
    for _ in 0..rows {
        let mut vals: Vec<f32> = (0..d).map(|i| i as f32 * 0.1 - 0.4).collect();
        rng.shuffle(&mut vals);
        data.extend(vals);
    }
    let mut p = Params::new();
    let x = p.add("x", Tensor::from_vec(&[rows, d], data), true);
    let w = rt(&[rows, d], &mut rng);
    fd_check(&mut p, &[x], |g| {
        let xv = g.param(x);
        let y = g.kwta(xv, 3);
        weighted(g, y, &w)
    });
}

#[test]
fn fd_lif_sigmoid_full_bptt() {
    // Smooth spike + reset NOT detached: the analytic BPTT must equal the true derivative.
    let mut rng = Rng::new(20);
    let (b, t, d) = (2usize, 5usize, 3usize);
    let mut p = Params::new();
    let cur = p.add("cur", Tensor::randn(&[b, t, d], 1.0, &mut rng), false);
    let beta = p.add("beta", Tensor::randn(&[d], 1.0, &mut rng), false);
    let w = rt(&[b, t, d], &mut rng);
    let opts = LifOpts { threshold: 0.8, spike_fn: SpikeFn::Sigmoid { slope: 2.0 }, detach_reset: false };
    fd_check(&mut p, &[cur, beta], |g| {
        let (c, bp) = (g.param(cur), g.param(beta));
        let s = g.lif_scan_opts(c, bp, opts);
        weighted(g, s, &w)
    });
}

/// Reference LIF in f64 where the reset term uses FROZEN spikes (constants): its true gradient is
/// exactly what the detached-reset backward computes.
#[allow(clippy::too_many_arguments)]
fn lif_frozen_reset_loss(
    cur: &[f32],
    bp: &[f32],
    (b, t, d): (usize, usize, usize),
    th: f64,
    slope: f64,
    frozen: &[f32],
    w: &[f32],
) -> f64 {
    let mut loss = 0.0;
    for bi in 0..b {
        for di in 0..d {
            let beta = 1.0 / (1.0 + (-(bp[di] as f64)).exp());
            let mut v = 0.0f64;
            let mut s_prev = 0.0f64;
            for ti in 0..t {
                let idx = (bi * t + ti) * d + di;
                v = beta * v + cur[idx] as f64 - th * s_prev;
                let s = 1.0 / (1.0 + (-slope * (v - th)).exp());
                loss += w[idx] as f64 * s;
                s_prev = frozen[idx] as f64;
            }
        }
    }
    loss
}

#[test]
fn fd_lif_sigmoid_detached_reset_matches_frozen_reference() {
    let mut rng = Rng::new(21);
    let (b, t, d) = (2usize, 6usize, 3usize);
    let (th, slope) = (0.7f32, 3.0f32);
    let mut p = Params::new();
    let cur = p.add("cur", Tensor::randn(&[b, t, d], 1.0, &mut rng), false);
    let beta = p.add("beta", Tensor::randn(&[d], 1.0, &mut rng), false);
    let w = rt(&[b, t, d], &mut rng);
    let opts = LifOpts { threshold: th, spike_fn: SpikeFn::Sigmoid { slope }, detach_reset: true };

    let (spikes, loss_graph, grads) = {
        let mut g = Graph::new(&p);
        let (c, bp) = (g.param(cur), g.param(beta));
        let s = g.lif_scan_opts(c, bp, opts);
        let l = weighted(&mut g, s, &w);
        (g.value(s).data.clone(), g.scalar(l), g.backward(l))
    };
    let dims = (b, t, d);
    let lref =
        lif_frozen_reset_loss(&p.value(cur).data, &p.value(beta).data, dims, th as f64, slope as f64, &spikes, &w.data);
    assert!((lref - loss_graph as f64).abs() < 1e-4, "reference forward {lref} vs graph {loss_graph}");

    let h = 1e-3f32;
    for id in [cur, beta] {
        let an = grads.get(id).unwrap().to_vec();
        for i in 0..p.value(id).numel() {
            let orig = p.value(id).data[i];
            p.value_mut(id).data[i] = orig + h;
            let lp = lif_frozen_reset_loss(
                &p.value(cur).data,
                &p.value(beta).data,
                dims,
                th as f64,
                slope as f64,
                &spikes,
                &w.data,
            );
            p.value_mut(id).data[i] = orig - h;
            let lm = lif_frozen_reset_loss(
                &p.value(cur).data,
                &p.value(beta).data,
                dims,
                th as f64,
                slope as f64,
                &spikes,
                &w.data,
            );
            p.value_mut(id).data[i] = orig;
            let num = ((lp - lm) / (2.0 * h as f64)) as f32;
            let tol = 2e-2 * an[i].abs().max(num.abs()) + 1e-3;
            assert!((an[i] - num).abs() <= tol, "{}[{i}]: analytic {} vs frozen-reset numeric {num}", p.name(id), an[i]);
        }
    }
}

#[test]
fn fd_var_reused_many_times() {
    let mut rng = Rng::new(22);
    let mut p = Params::new();
    let x = p.add("x", rt(&[2, 3], &mut rng), true);
    let w = rt(&[2, 3], &mut rng);
    fd_check(&mut p, &[x], |g| {
        let xv = g.param(x);
        let xv2 = g.param(x); // same param asked for twice must alias the same node
        assert_eq!(xv, xv2);
        let sq = g.mul(xv, xv2);
        let y = g.add(sq, xv);
        let y = g.mul(y, xv);
        weighted(g, y, &w)
    });
}

#[test]
fn fd_composite_transformer_block() {
    // embedding → rmsnorm → 2-head causal self-attention (matmul_t + causal softmax) → residual
    // → rmsnorm → gelu MLP → residual → rmsnorm → tied logits (matmul_t with the embedding) → cross-entropy
    let mut rng = Rng::new(23);
    let (vocab, d, heads, b, t, hid) = (7usize, 8usize, 2usize, 2usize, 4usize, 12usize);
    let dh = d / heads;
    let s = 0.4;
    let mut p = Params::new();
    let emb = p.add("emb", Tensor::randn(&[vocab, d], 1.0, &mut rng), false);
    let g1 = p.add("g1", Tensor::full(&[d], 1.0), false);
    let wq = p.add("wq", Tensor::randn(&[d, d], s, &mut rng), true);
    let wk = p.add("wk", Tensor::randn(&[d, d], s, &mut rng), true);
    let wv = p.add("wv", Tensor::randn(&[d, d], s, &mut rng), true);
    let wo = p.add("wo", Tensor::randn(&[d, d], s, &mut rng), true);
    let g2 = p.add("g2", Tensor::full(&[d], 1.0), false);
    let w1 = p.add("w1", Tensor::randn(&[d, hid], s, &mut rng), true);
    let b1 = p.add("b1", Tensor::randn(&[hid], 0.1, &mut rng), false);
    let w2 = p.add("w2", Tensor::randn(&[hid, d], s, &mut rng), true);
    let g3 = p.add("g3", Tensor::full(&[d], 1.0), false);
    let ids: Vec<u32> = (0..b * t).map(|_| rng.below(vocab) as u32).collect();
    let targets: Vec<u32> = (0..b * t).map(|_| rng.below(vocab) as u32).collect();
    let all = [emb, g1, wq, wk, wv, wo, g2, w1, b1, w2, g3];
    fd_check(&mut p, &all, |g| {
        let e = g.param(emb);
        let x = g.embedding(e, &ids); // [b*t, d]
        let gg1 = g.param(g1);
        let h = g.rmsnorm(x, gg1, 1e-5);
        let heads_of = |g: &mut Graph, w: ParamId| {
            let wv = g.param(w);
            let y = g.matmul(h, wv);
            let y = g.reshape(y, &[b, t, heads, dh]);
            g.transpose(y, 1, 2) // [b, heads, t, dh]
        };
        let q = heads_of(g, wq);
        let k = heads_of(g, wk);
        let v = heads_of(g, wv);
        let sc = g.matmul_t(q, k); // [b, heads, t, t]
        let sc = g.scale(sc, 1.0 / (dh as f32).sqrt());
        let att = g.softmax_last(sc, true);
        let o = g.matmul(att, v); // [b, heads, t, dh]
        let o = g.transpose(o, 1, 2);
        let o = g.reshape(o, &[b * t, d]);
        let wov = g.param(wo);
        let o = g.matmul(o, wov);
        let x2 = g.add(x, o);
        let gg2 = g.param(g2);
        let h2 = g.rmsnorm(x2, gg2, 1e-5);
        let (w1v, b1v, w2v) = (g.param(w1), g.param(b1), g.param(w2));
        let m = g.matmul(h2, w1v);
        let m = g.add_bias(m, b1v);
        let m = g.gelu(m);
        let m = g.matmul(m, w2v);
        let x3 = g.add(x2, m);
        let gg3 = g.param(g3);
        let hf = g.rmsnorm(x3, gg3, 1e-5);
        let logits = g.matmul_t(hf, e); // tied output embedding
        g.cross_entropy(logits, &targets)
    });
}

#[test]
fn fd_composite_spiking_block() {
    // embedding → matmul → kWTA (sparse drive) → LIF scan over time (smooth spike, full BPTT)
    // → matmul + bias → layernorm → cross-entropy
    let mut rng = Rng::new(24);
    let (vocab, d, hid, b, t, k) = (6usize, 5usize, 8usize, 2usize, 4usize, 3usize);
    let mut p = Params::new();
    let emb = p.add("emb", Tensor::randn(&[vocab, d], 1.0, &mut rng), false);
    let win = p.add("win", Tensor::randn(&[d, hid], 0.8, &mut rng), true);
    let beta = p.add("beta", Tensor::randn(&[hid], 0.5, &mut rng), false);
    let wout = p.add("wout", Tensor::randn(&[hid, vocab], 0.8, &mut rng), true);
    let bout = p.add("bout", Tensor::randn(&[vocab], 0.1, &mut rng), false);
    let lg = p.add("ln_g", Tensor::full(&[vocab], 1.0), false);
    let lb = p.add("ln_b", Tensor::zeros(&[vocab]), false);
    let ids: Vec<u32> = (0..b * t).map(|_| rng.below(vocab) as u32).collect();
    let targets: Vec<u32> = (0..b * t).map(|_| rng.below(vocab) as u32).collect();
    let opts = LifOpts { threshold: 0.5, spike_fn: SpikeFn::Sigmoid { slope: 2.0 }, detach_reset: false };

    // The kWTA winner set must not flip under a ±1e-3 nudge: require a clear gap at the k-th boundary.
    {
        let mut g = Graph::new(&p);
        let e = g.param(emb);
        let x = g.embedding(e, &ids);
        let wv = g.param(win);
        let c = g.matmul(x, wv);
        for row in g.value(c).data.chunks(hid) {
            let mut s = row.to_vec();
            s.sort_by(|a, b| b.total_cmp(a));
            assert!(s[k - 1] - s[k] > 2e-2, "seed gives a near-tie at the kWTA boundary; pick another seed");
        }
    }

    let all = [emb, win, beta, wout, bout, lg, lb];
    fd_check(&mut p, &all, |g| {
        let e = g.param(emb);
        let x = g.embedding(e, &ids);
        let wv = g.param(win);
        let c = g.matmul(x, wv);
        let c = g.kwta(c, k);
        let c = g.reshape(c, &[b, t, hid]);
        let bp = g.param(beta);
        let s = g.lif_scan_opts(c, bp, opts);
        let s = g.reshape(s, &[b * t, hid]);
        let (wo, bo) = (g.param(wout), g.param(bout));
        let z = g.matmul(s, wo);
        let z = g.add_bias(z, bo);
        let (lgv, lbv) = (g.param(lg), g.param(lb));
        let z = g.layernorm(z, lgv, lbv, 1e-5);
        g.cross_entropy(z, &targets)
    });
}
