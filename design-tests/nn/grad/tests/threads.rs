// TESTING / DESIGN ONLY — every threaded op must be bit-for-bit identical to its 1-thread result.
// Random (seeded) shapes big enough that the element- and row-splitting helpers really split the work
// at 3, 8 and 17 threads (cheap elementwise kernels need 2^16 elements per thread, costly ones 2^14,
// the LIF scan 2^14 cells per thread). Forward values of every op and every parameter gradient are
// compared on their bit patterns (so a −0.0 vs 0.0 difference would be caught too).

mod common;

use ava_grad::*;
use common::weighted;

const THREADS: [usize; 3] = [3, 8, 17];

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Run `build` at every thread count and assert it returns the same (named) buffers as at 1 thread.
fn check<F>(name: &str, params: &Params, build: F)
where
    F: Fn(&mut Graph) -> Vec<(&'static str, Vec<f32>)>,
{
    let run = |t: usize| {
        let mut g = Graph::new(params);
        g.set_threads(t);
        assert_eq!(g.threads(), t, "set_threads({t}) was clamped");
        build(&mut g)
    };
    let base = run(1);
    assert!(!base.is_empty());
    for &t in &THREADS {
        let got = run(t);
        assert_eq!(base.len(), got.len());
        for ((n0, b), (n1, v)) in base.iter().zip(&got) {
            assert_eq!(n0, n1);
            assert_eq!(b.len(), v.len(), "{name}/{n0}: length differs at {t} threads");
            assert!(bits(b) == bits(v), "{name}/{n0}: {t} threads differ from 1 thread");
        }
    }
}

/// Every op output in `vars` plus every parameter gradient, as named buffers.
fn collect(g: &Graph, loss: Var, vars: &[(&'static str, Var)], params: &Params) -> Vec<(&'static str, Vec<f32>)> {
    let grads = g.backward(loss);
    let mut out: Vec<(&'static str, Vec<f32>)> = vars.iter().map(|&(n, v)| (n, g.value(v).data.clone())).collect();
    for id in params.ids() {
        out.push((Box::leak(format!("grad:{}", params.name(id)).into_boxed_str()), grads.get(id).map(|s| s.to_vec()).unwrap_or_default()));
    }
    out
}

#[test]
fn elementwise_ops_bit_identical_across_thread_counts() {
    let mut rng = Rng::new(2024);
    // ≥ 12·96·1024 = 1.18M elements: the cheap kernels split 17 ways, the costly ones many times over
    let (b, t, d) = (12 + rng.below(4), 96 + rng.below(32), 1024 + rng.below(256));
    let mut p = Params::new();
    let x = p.add("x", Tensor::randn(&[b, t, d], 1.0, &mut rng), true);
    let y = p.add("y", Tensor::randn(&[b, t, d], 1.0, &mut rng), true);
    let bias = p.add("bias", Tensor::randn(&[d], 0.5, &mut rng), true);
    let w = Tensor::randn(&[b, t, d], 1.0, &mut rng);
    check("elementwise", &p, |g| {
        let (xv, yv, bv) = (g.param(x), g.param(y), g.param(bias));
        let s = g.scale(xv, 0.7);
        let a = g.add(s, yv);
        let sb = g.sub(a, xv);
        let m = g.mul(sb, yv);
        let ab = g.add_bias(m, bv);
        let r = g.relu(ab);
        let ge = g.gelu(xv);
        let sg = g.sigmoid(yv);
        let th = g.tanh(ab);
        let sc = g.scale(xv, 0.1);
        let ex = g.exp(sc);
        let acc = g.add(r, ge);
        let acc = g.add(acc, sg);
        let acc = g.add(acc, th);
        let acc = g.add(acc, ex);
        let su = g.sum(acc);
        let me = g.mean(th);
        let l0 = weighted(g, acc, &w);
        let l1 = g.add(su, me);
        let l = g.add(l0, l1);
        let vars = [
            ("scale", s),
            ("add", a),
            ("sub", sb),
            ("mul", m),
            ("add_bias", ab),
            ("relu", r),
            ("gelu", ge),
            ("sigmoid", sg),
            ("tanh", th),
            ("exp", ex),
            ("sum", su),
            ("mean", me),
            ("loss", l),
        ];
        collect(g, l, &vars, &p)
    });
}

#[test]
fn norms_softmax_ce_kwta_embedding_bit_identical_across_thread_counts() {
    let mut rng = Rng::new(77);
    let (n, d, vocab) = (3000 + rng.below(1500), 192 + rng.below(128), 400 + rng.below(200));
    let k = 8 + rng.below(24);
    let mut p = Params::new();
    let table = p.add("table", Tensor::randn(&[vocab, d], 1.0, &mut rng), true);
    let gain = p.add("gain", Tensor::randn(&[d], 0.3, &mut rng), true);
    let gain2 = p.add("gain2", Tensor::randn(&[d], 0.3, &mut rng), true);
    let bias = p.add("bias", Tensor::randn(&[d], 0.3, &mut rng), true);
    let wo = p.add("wo", Tensor::randn(&[d, vocab], 0.05, &mut rng), true);
    let ids: Vec<u32> = (0..n).map(|_| rng.below(vocab) as u32).collect();
    let targets: Vec<u32> = (0..n).map(|i| if i % 11 == 0 { IGNORE } else { rng.below(vocab) as u32 }).collect();
    let w = Tensor::randn(&[n, d], 1.0, &mut rng);
    check("norms", &p, |g| {
        let (tv, gv, g2, bv, wov) = (g.param(table), g.param(gain), g.param(gain2), g.param(bias), g.param(wo));
        let e = g.embedding(tv, &ids);
        let rn = g.rmsnorm(e, gv, 1e-5);
        let ln = g.layernorm(rn, g2, bv, 1e-5);
        let kw = g.kwta(ln, k);
        let sm = g.softmax_last(kw, false);
        let logits = g.matmul(ln, wov);
        let ce = g.cross_entropy(logits, &targets);
        let side = g.add(sm, kw);
        let l0 = weighted(g, side, &w);
        let l = g.add(l0, ce);
        let vars = [("embedding", e), ("rmsnorm", rn), ("layernorm", ln), ("kwta", kw), ("softmax", sm), ("logits", logits), ("ce", ce), ("loss", l)];
        collect(g, l, &vars, &p)
    });
}

#[test]
fn attention_reshape_transpose_bit_identical_across_thread_counts() {
    let mut rng = Rng::new(5);
    let (b, t, h, dh) = (24 + rng.below(8), 96 + rng.below(32), 8, 32 + rng.below(16));
    let d = h * dh;
    let mut p = Params::new();
    let x = p.add("x", Tensor::randn(&[b, t, d], 1.0, &mut rng), true);
    let wq = p.add("wq", Tensor::randn(&[d, d], 0.05, &mut rng), true);
    let wk = p.add("wk", Tensor::randn(&[d, d], 0.05, &mut rng), true);
    let wv = p.add("wv", Tensor::randn(&[d, d], 0.05, &mut rng), true);
    let wo = p.add("wo", Tensor::randn(&[d, d], 0.05, &mut rng), true);
    let w = Tensor::randn(&[b, t, d], 1.0, &mut rng);
    check("attention", &p, |g| {
        let (xv, q, k, v, o) = (g.param(x), g.param(wq), g.param(wk), g.param(wv), g.param(wo));
        let xf = g.reshape(xv, &[b * t, d]);
        let heads = |g: &mut Graph, wm: Var| {
            let y = g.matmul(xf, wm);
            let y = g.reshape(y, &[b, t, h, dh]);
            g.transpose(y, 1, 2) // [b, h, t, dh]
        };
        let (qh, kh, vh) = (heads(g, q), heads(g, k), heads(g, v));
        let scores = g.matmul_t(qh, kh); // [b, h, t, t]
        let scores = g.scale(scores, 1.0 / (dh as f32).sqrt());
        let att = g.softmax_last(scores, true);
        let ctx = g.matmul(att, vh); // [b, h, t, dh]
        let ctx = g.transpose(ctx, 1, 2);
        let ctx = g.reshape(ctx, &[b * t, d]);
        let out = g.matmul(ctx, o);
        let out = g.reshape(out, &[b, t, d]);
        let res = g.add(out, xv);
        let l = weighted(g, res, &w);
        let vars = [("q", qh), ("scores", scores), ("att", att), ("ctx", ctx), ("out", out), ("loss", l)];
        collect(g, l, &vars, &p)
    });
}

#[test]
fn lif_scan_bit_identical_across_thread_counts() {
    let mut rng = Rng::new(31);
    // many batches (split across batch groups) and few batches with many channels (split across channel blocks)
    let shapes = [(16 + rng.below(8), 48 + rng.below(32), 256 + rng.below(128)), (2, 96 + rng.below(32), 3000 + rng.below(1000))];
    for (b, t, d) in shapes {
        let mut p = Params::new();
        let cur = p.add("cur", Tensor::randn(&[b, t, d], 1.0, &mut rng), true);
        let beta = p.add("beta", Tensor::randn(&[d], 1.0, &mut rng), true);
        let w = Tensor::randn(&[b, t, d], 1.0, &mut rng);
        for (fi, f) in [SpikeFn::Surrogate { slope: 3.0 }, SpikeFn::Sigmoid { slope: 4.0 }].into_iter().enumerate() {
            for detach in [true, false] {
                let name = Box::leak(format!("lif b={b} t={t} d={d} fn={fi} detach={detach}").into_boxed_str());
                check(name, &p, |g| {
                    let (c, bp) = (g.param(cur), g.param(beta));
                    let opts = LifOpts { threshold: 1.0, spike_fn: f, detach_reset: detach };
                    let s = g.lif_scan_opts(c, bp, opts);
                    let l = weighted(g, s, &w);
                    collect(g, l, &[("spikes", s), ("loss", l)], &p)
                });
            }
        }
    }
}
