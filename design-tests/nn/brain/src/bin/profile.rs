// TESTING / DESIGN ONLY — profile: where a training step's time goes, op by op.
// Every ava_grad op the models use is timed in isolation at the real shapes (inputs are parameters so
// the backward pass runs; backward is seeded with ones), forward and backward separately, best of
// --reps repetitions. The per-step estimate for T, S and H multiplies each op by how often the model
// uses it per step, and is compared with a measured full training step. Nothing is written.
// profile [--d 128] [--layers 4] [--batch 32] [--ctx 128] [--threads 4] [--reps 5]

use ava_brain::{Config, Hyper, Kind, Trainer};
use ava_data::input::parse_count;
use ava_grad::{Graph, Params, Rng, SpikeFn, Tensor, Var};
use std::time::Instant;

const V: usize = 4096;

struct Opts {
    d: usize,
    layers: usize,
    batch: usize,
    ctx: usize,
    threads: usize,
    reps: usize,
}

fn parse() -> Opts {
    let mut o = Opts { d: 128, layers: 4, batch: 32, ctx: 128, threads: 4, reps: 5 };
    let a: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i + 1 < a.len() {
        let v = parse_count(&a[i + 1]).unwrap_or_else(|e| panic!("{}: {e}", a[i])) as usize;
        match a[i].as_str() {
            "--d" => o.d = v,
            "--layers" => o.layers = v,
            "--batch" => o.batch = v,
            "--ctx" => o.ctx = v,
            "--threads" => o.threads = v,
            "--reps" => o.reps = v.max(1),
            k => panic!("unknown option {k}"),
        }
        i += 2;
    }
    o
}

/// (best forward seconds, best backward seconds) of `op` applied to parameters with the given shapes.
fn time_op<F>(shapes: &[&[usize]], reps: usize, threads: usize, op: F) -> (f64, f64)
where
    F: Fn(&mut Graph, &[Var]) -> Var,
{
    let mut rng = Rng::new(1);
    let mut p = Params::new();
    let ids: Vec<_> = shapes.iter().enumerate().map(|(i, s)| p.add(&format!("x{i}"), Tensor::randn(s, 1.0, &mut rng), true)).collect();
    let (mut bf, mut bb) = (f64::MAX, f64::MAX);
    for _ in 0..reps + 1 {
        let mut g = Graph::new(&p);
        g.set_threads(threads);
        let xs: Vec<Var> = ids.iter().map(|&id| g.param(id)).collect();
        let t0 = Instant::now();
        let y = op(&mut g, &xs);
        let f = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let gr = g.backward(y);
        let b = t1.elapsed().as_secs_f64();
        std::hint::black_box(&gr);
        bf = bf.min(f);
        bb = bb.min(b);
    }
    (bf, bb)
}

struct Row {
    name: String,
    fwd: f64,
    bwd: f64,
    /// uses per training step for T, S, H
    count: [usize; 3],
}

fn main() {
    let o = parse();
    let (b, t, d, l) = (o.batch, o.ctx, o.d, o.layers);
    let n = b * t;
    let hid = 4 * d;
    let nh = d / 32;
    let r = o.reps;
    let th = o.threads;
    println!("profile: d={d} layers={l} batch={b} ctx={t} threads={th} (best of {r}; ms per call)");
    let mut rows: Vec<Row> = Vec::new();
    let mut push = |name: String, (f, bw): (f64, f64), count: [usize; 3]| rows.push(Row { name, fwd: f, bwd: bw, count });

    // matmuls: activations [n, din] × weight [din, dout]
    push(format!("matmul [{n},{d}]x[{d},{d}]"), time_op(&[&[n, d], &[d, d]], r, th, |g, x| g.matmul(x[0], x[1])), [4 * l, 2 * l, 6 * l]);
    push(format!("matmul [{n},{d}]x[{d},{hid}]"), time_op(&[&[n, d], &[d, hid]], r, th, |g, x| g.matmul(x[0], x[1])), [l, l, l]);
    push(format!("matmul [{n},{hid}]x[{hid},{d}]"), time_op(&[&[n, hid], &[hid, d]], r, th, |g, x| g.matmul(x[0], x[1])), [l, l, l]);
    push(format!("logits matmul_t [{n},{d}]x[{V},{d}]T"), time_op(&[&[n, d], &[V, d]], r, th, |g, x| g.matmul_t(x[0], x[1])), [1, 1, 1]);
    let tg: Vec<u32> = (0..n).map(|i| (i * 7919 % V) as u32).collect();
    push(format!("cross_entropy [{n},{V}]"), time_op(&[&[n, V]], r, th, |g, x| g.cross_entropy(x[0], &tg)), [1, 1, 1]);
    // attention core
    if nh > 0 {
        push(
            format!("attn reshape+transpose [{n},{d}]->[{b},{nh},{t},32]"),
            time_op(&[&[n, d]], r, th, |g, x| {
                let y = g.reshape(x[0], &[b, t, nh, 32]);
                g.transpose(y, 1, 2)
            }),
            [4 * l, 0, 4 * l],
        );
        push(format!("attn scores matmul_t [{b},{nh},{t},32]"), time_op(&[&[b, nh, t, 32], &[b, nh, t, 32]], r, th, |g, x| g.matmul_t(x[0], x[1])), [l, 0, l]);
        push(format!("attn scale [{b},{nh},{t},{t}]"), time_op(&[&[b, nh, t, t]], r, th, |g, x| g.scale(x[0], 0.17)), [l, 0, l]);
        push(format!("softmax causal [{b},{nh},{t},{t}]"), time_op(&[&[b, nh, t, t]], r, th, |g, x| g.softmax_last(x[0], true)), [l, 0, l]);
        push(format!("attn att.v matmul [{b},{nh},{t},{t}]x[..,{t},32]"), time_op(&[&[b, nh, t, t], &[b, nh, t, 32]], r, th, |g, x| g.matmul(x[0], x[1])), [l, 0, l]);
    }
    // elementwise / norms
    push(format!("rmsnorm [{n},{d}]"), time_op(&[&[n, d], &[d]], r, th, |g, x| g.rmsnorm(x[0], x[1], 1e-5)), [2 * l + 1, 2 * l + 1, 3 * l + 1]);
    push(format!("add [{n},{d}]"), time_op(&[&[n, d], &[n, d]], r, th, |g, x| g.add(x[0], x[1])), [2 * l + 1, 2 * l, 3 * l + 1]);
    push(format!("gelu [{n},{hid}]"), time_op(&[&[n, hid]], r, th, |g, x| g.gelu(x[0])), [l, 0, 0]);
    push(format!("kwta k={} [{n},{hid}]", hid / 8), time_op(&[&[n, hid]], r, th, |g, x| g.kwta(x[0], hid / 8)), [0, l, l]);
    push(format!("relu [{n},{hid}]"), time_op(&[&[n, hid]], r, th, |g, x| g.relu(x[0])), [0, l, l]);
    push(
        format!("lif_scan [{b},{t},{d}] (+2 reshapes)"),
        time_op(&[&[n, d], &[d]], r, th, |g, x| {
            let c = g.reshape(x[0], &[b, t, d]);
            let s = g.lif_scan(c, x[1], 1.0, SpikeFn::Surrogate { slope: 4.0 });
            g.reshape(s, &[n, d])
        }),
        [0, l, l],
    );
    let ids: Vec<u32> = (0..n).map(|i| (i * 31 % V) as u32).collect();
    push(format!("embedding [{V},{d}] -> [{n},{d}]"), time_op(&[&[V, d]], r, th, |g, x| g.embedding(x[0], &ids)), [2, 1, 2]);

    println!("{:<52} {:>9} {:>9} {:>5} {:>5} {:>5}", "op", "fwd ms", "bwd ms", "T", "S", "H");
    for row in &rows {
        println!("{:<52} {:>9.3} {:>9.3} {:>5} {:>5} {:>5}", row.name, row.fwd * 1e3, row.bwd * 1e3, row.count[0], row.count[1], row.count[2]);
    }
    for (mi, kind) in [Kind::T, Kind::S, Kind::H].into_iter().enumerate() {
        if kind != Kind::S && nh == 0 {
            continue;
        }
        let (mut f, mut bw) = (0.0, 0.0);
        let mut parts: Vec<(f64, &str)> = Vec::new();
        for row in &rows {
            let c = row.count[mi] as f64;
            f += c * row.fwd;
            bw += c * row.bwd;
            if c > 0.0 {
                parts.push((c * (row.fwd + row.bwd), &row.name));
            }
        }
        parts.sort_by(|a, b| b.0.total_cmp(&a.0));
        // measured: 4 real steps on random tokens, best of the last 3
        let cfg = Config::new(kind, V, d, l, t);
        let hy = Hyper { lr: 1e-3, warmup: 1, min_lr_frac: 0.1, wd: 0.1, clip: 1.0, total_steps: 10, threads: th };
        let mut tr = Trainer::new(cfg, 1, hy);
        let mut rng = Rng::new(3);
        let mut best = (f64::MAX, 0.0, 0.0, 0.0);
        for i in 0..4 {
            let x: Vec<u32> = (0..n).map(|_| rng.below(V) as u32).collect();
            let y: Vec<u32> = (0..n).map(|_| rng.below(V) as u32).collect();
            let s = tr.train_step(&x, &y, b, t);
            if i > 0 && s.total_s < best.0 {
                best = (s.total_s, s.fwd_s, s.bwd_s, s.opt_s);
            }
        }
        println!(
            "\n{}: estimated fwd {:.1} ms + bwd {:.1} ms = {:.1} ms; measured step {:.1} ms (fwd {:.1}, bwd {:.1}, opt+free {:.1})",
            kind.name(),
            f * 1e3,
            bw * 1e3,
            (f + bw) * 1e3,
            best.0 * 1e3,
            best.1 * 1e3,
            best.2 * 1e3,
            best.3 * 1e3
        );
        let tot = f + bw;
        for (s, name) in parts.iter().take(8) {
            println!("    {:>5.1}%  {:>7.1} ms  {}", 100.0 * s / tot, s * 1e3, name);
        }
    }
}
