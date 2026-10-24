// TESTING / DESIGN ONLY — per-op forward+backward wall time at the training shape (batch 32, ctx 128,
// d 128, hidden 512, 4 heads, vocab 256) with 1, 4 and 8 threads, to show what threading the kernels buys.
// Every op is run through the Graph (forward, then backward from a sum / cross-entropy), best of 5.
// Run: CARGO_BUILD_JOBS=2 cargo run --release --example bench_ops

use ava_grad::*;
use std::time::{Duration, Instant};

const B: usize = 32;
const T: usize = 128;
const D: usize = 128;
const HID: usize = 512;
const H: usize = 4;
const V: usize = 256;

type Build = Box<dyn Fn(&mut Graph) -> Var>;

fn time(params: &Params, threads: usize, build: &Build) -> Duration {
    let mut best = Duration::MAX;
    for _ in 0..5 {
        let t0 = Instant::now();
        let mut g = Graph::new(params);
        g.set_threads(threads);
        let l = build(&mut g);
        let grads = g.backward(l);
        std::hint::black_box(&grads);
        best = best.min(t0.elapsed());
    }
    best
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() {
    let mut rng = Rng::new(3);
    let mut p = Params::new();
    let x = p.add("x", Tensor::randn(&[B, T, D], 1.0, &mut rng), true);
    let xh = p.add("xh", Tensor::randn(&[B * T, HID], 1.0, &mut rng), true);
    let yh = p.add("yh", Tensor::randn(&[B * T, HID], 1.0, &mut rng), true);
    let gain = p.add("gain", Tensor::randn(&[D], 0.3, &mut rng), true);
    let bias = p.add("bias", Tensor::randn(&[D], 0.3, &mut rng), true);
    let beta = p.add("beta", Tensor::randn(&[D], 1.0, &mut rng), true);
    let w1 = p.add("w1", Tensor::randn(&[D, HID], 0.05, &mut rng), true);
    let wv = p.add("wv", Tensor::randn(&[D, V], 0.05, &mut rng), true);
    let sc = p.add("scores", Tensor::randn(&[B, H, T, T], 1.0, &mut rng), true);
    let targets: Vec<u32> = (0..B * T).map(|_| rng.below(V) as u32).collect();
    let cases: Vec<(&str, Build)> = vec![
        ("gelu [4096,512]", Box::new(move |g| {
            let v = g.param(xh);
            let v = g.gelu(v);
            g.sum(v)
        })),
        ("relu [4096,512]", Box::new(move |g| {
            let v = g.param(xh);
            let v = g.relu(v);
            g.sum(v)
        })),
        ("sigmoid+tanh+exp [4096,512]", Box::new(move |g| {
            let v = g.param(xh);
            let a = g.sigmoid(v);
            let b = g.tanh(v);
            let c = g.scale(v, 0.1);
            let c = g.exp(c);
            let s = g.add(a, b);
            let s = g.add(s, c);
            g.sum(s)
        })),
        ("add+sub+mul+scale [4096,512]", Box::new(move |g| {
            let a = g.param(xh);
            let b = g.param(yh);
            let s = g.add(a, b);
            let m = g.mul(s, b);
            let d = g.sub(m, a);
            let d = g.scale(d, 0.5);
            g.sum(d)
        })),
        ("rmsnorm [4096,128]", Box::new(move |g| {
            let v = g.param(x);
            let gv = g.param(gain);
            let n = g.rmsnorm(v, gv, 1e-5);
            g.sum(n)
        })),
        ("layernorm+add_bias [4096,128]", Box::new(move |g| {
            let v = g.param(x);
            let (gv, bv) = (g.param(gain), g.param(bias));
            let n = g.layernorm(v, gv, bv, 1e-5);
            let n = g.add_bias(n, bv);
            g.sum(n)
        })),
        ("kwta 16 of 128 [4096,128]", Box::new(move |g| {
            let v = g.param(x);
            let k = g.kwta(v, 16);
            g.sum(k)
        })),
        ("lif_scan [32,128,128]", Box::new(move |g| {
            let (c, b) = (g.param(x), g.param(beta));
            let s = g.lif_scan(c, b, 1.0, SpikeFn::Surrogate { slope: 5.0 });
            g.sum(s)
        })),
        ("softmax causal [32,4,128,128]", Box::new(move |g| {
            let s = g.param(sc);
            let a = g.softmax_last(s, true);
            g.sum(a)
        })),
        ("reshape+transpose [32,128,4,32]", Box::new(move |g| {
            let v = g.param(x);
            let v = g.reshape(v, &[B, T, H, D / H]);
            let t = g.transpose(v, 1, 2);
            let t = g.scale(t, 2.0);
            g.sum(t)
        })),
        ("matmul [4096,128]x[128,512]", Box::new(move |g| {
            let v = g.param(x);
            let v = g.reshape(v, &[B * T, D]);
            let w = g.param(w1);
            let h = g.matmul(v, w);
            g.sum(h)
        })),
        ("logits+cross_entropy [4096,256]", Box::new(move |g| {
            let v = g.param(x);
            let v = g.reshape(v, &[B * T, D]);
            let w = g.param(wv);
            let z = g.matmul(v, w);
            g.cross_entropy(z, &targets)
        })),
    ];
    println!("ava_grad bench_ops: forward+backward wall time, best of 5 (batch {B}, ctx {T}, d {D}, hidden {HID})");
    println!("{:<34} {:>10} {:>10} {:>10}", "op", "1 thread", "4 threads", "8 threads");
    let mut totals = [0.0f64; 3];
    for (name, build) in &cases {
        let ts: Vec<f64> = [1usize, 4, 8].iter().map(|&t| ms(time(&p, t, build))).collect();
        for (tot, t) in totals.iter_mut().zip(&ts) {
            *tot += t;
        }
        println!("{:<34} {:>8.2}ms {:>8.2}ms {:>8.2}ms", name, ts[0], ts[1], ts[2]);
    }
    println!("{:<34} {:>8.2}ms {:>8.2}ms {:>8.2}ms", "sum", totals[0], totals[1], totals[2]);
}
