// TESTING / DESIGN ONLY — flops_table (v42): every measured run's real FLOPs per token, and its BPB against FLOPs
// instead of weights. CPU only (shapes and numbers; no model is loaded).
//
// flops_table --data FILE [--ctx 256] [--vocab 4096] [--mem chunk16|token] [--tokens-per-weight 20] [--a D] [--b T]
//
// FILE: TSV rows `family d layers bpb seed` (as scaling_fit). For each shape: weights (non-embedding and all), forward
// FLOPs per token by part (unembedding 2·d·V, dense matmuls, the look-back core: attention or fast weights, see
// src/flops.rs), training FLOPs per token (3 × forward), their ratio to the 6·N rule of thumb, tokens trained
// (tokens-per-weight × all weights, v34's budget) and the training compute in all. Then each family's log-linear
// slope of BPB on each axis, the ground change of A − B over A's span on each axis, and a plot of BPB against
// training FLOPs per token. For intervals and the power-law form on a FLOPs axis: scaling_fit --x flops (or compute).

use ava_gpu::flops::{self, Axis, MemForm};
use ava_gpu::scaling::{self, Row};

const USAGE: &str = "usage: flops_table --data FILE [--ctx 256] [--vocab 4096] [--mem chunk16|token] [--tokens-per-weight 20] [--a D] [--b T]";

fn main() {
    if let Err(e) = run() {
        eprintln!("flops_table: {e}");
        std::process::exit(1);
    }
}

fn sci(x: f64) -> String {
    format!("{x:.3e}")
}

fn run() -> Result<(), String> {
    let (mut data, mut ctx, mut vocab, mut mem, mut tpw) = (None::<String>, 256usize, 4096usize, flops::DEFAULT_MEM, 20.0f64);
    let (mut fa, mut fb) = ("D".to_string(), "T".to_string());
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.into());
        }
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<f64>().map_err(|_| format!("bad number {v:?}"));
        match key {
            "--data" => data = Some(v.into()),
            "--ctx" => ctx = num(v)? as usize,
            "--vocab" => vocab = num(v)? as usize,
            "--mem" => mem = MemForm::parse(v)?,
            "--tokens-per-weight" => tpw = num(v)?,
            "--a" => fa = v.into(),
            "--b" => fb = v.into(),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let data = data.ok_or_else(|| format!("--data is required\n{USAGE}"))?;
    let text = std::fs::read_to_string(&data).map_err(|e| format!("read {data}: {e}"))?;
    let runs = flops::read_runs(&text)?;

    // one line per shape (seeds averaged)
    let mut shapes: Vec<(String, usize, usize)> = Vec::new();
    for r in &runs {
        let k = (r.family.clone(), r.d, r.layers);
        if !shapes.contains(&k) {
            shapes.push(k);
        }
    }
    println!("ctx {ctx}, vocab {vocab}, D's memory core {mem:?}, {tpw} training tokens per weight; FLOPs per token unless said");
    println!(
        "{:<3} {:>4} {:>2} {:>11} {:>11} | {:>10} {:>10} {:>10} {:>10} | {:>10} {:>6} | {:>10} {:>8}",
        "fam", "d", "L", "N_nonembed", "N_all", "unembed", "dense", "core", "forward", "train", "/6N", "compute", "BPB"
    );
    for (fam, d, l) in &shapes {
        let r0 = runs.iter().find(|r| &r.family == fam && r.d == *d && r.layers == *l).expect("shape from runs");
        let c = flops::config_of(r0, vocab, ctx)?;
        let (total, nonembed) = c.weight_counts();
        let f = flops::per_token(&c, mem);
        let ys: Vec<f64> = runs.iter().filter(|r| &r.family == fam && r.d == *d && r.layers == *l).map(|r| r.bpb).collect();
        let bpb = ys.iter().sum::<f64>() / ys.len() as f64;
        println!(
            "{:<3} {:>4} {:>2} {:>11} {:>11} | {:>10} {:>10} {:>10} {:>10} | {:>10} {:>6.2} | {:>10} {:>8.4}",
            fam,
            d,
            l,
            nonembed,
            total,
            sci(f.unembed),
            sci(f.dense),
            sci(f.mixer),
            sci(f.forward()),
            sci(f.train()),
            f.train() / (6.0 * nonembed as f64),
            sci(f.train() * tpw * total as f64),
            bpb
        );
    }

    // slopes and the A − B ground change on each axis (log-linear, cell means)
    println!("\nlog-linear fits BPB = a + b ln x on each axis (b = BPB per e-fold), and the {fa} - {fb} ground change over {fa}'s span:");
    for axis in [Axis::NonEmbed, Axis::Total, Axis::Flops, Axis::Compute] {
        let mut rows = Vec::new();
        for r in &runs {
            let c = flops::config_of(r, vocab, ctx)?;
            rows.push(Row { family: r.family.clone(), x: flops::size_on(&c, axis, mem, tpw), bpb: r.bpb, seed: r.seed });
        }
        let mut fams: Vec<String> = rows.iter().map(|r| r.family.clone()).collect();
        fams.sort();
        fams.dedup();
        let mut parts = Vec::new();
        for f in &fams {
            let cs = scaling::cells(&rows, f);
            if cs.len() >= 2 {
                let xs: Vec<f64> = cs.iter().map(|c| c.x).collect();
                let ys: Vec<f64> = cs.iter().map(|c| c.mean()).collect();
                if let scaling::Curve::Lin { b, .. } = scaling::fit_loglinear(&xs, &ys) {
                    parts.push(format!("{f} {b:+.4}"));
                }
            }
        }
        let (ca, cb) = (scaling::cells(&rows, &fa), scaling::cells(&rows, &fb));
        let change = if ca.len() >= 2 && cb.len() >= 2 {
            let pa = scaling::fit_loglinear(&ca.iter().map(|c| c.x).collect::<Vec<_>>(), &ca.iter().map(|c| c.mean()).collect::<Vec<_>>());
            let pb = scaling::fit_loglinear(&cb.iter().map(|c| c.x).collect::<Vec<_>>(), &cb.iter().map(|c| c.mean()).collect::<Vec<_>>());
            let p = scaling::Pair { a: pa, b: pb };
            format!("{:+.4} (gap {:+.4} -> {:+.4})", p.change(ca[0].x, ca[ca.len() - 1].x), p.gap(ca[0].x), p.gap(ca[ca.len() - 1].x))
        } else {
            "-".into()
        };
        println!("  {:<26} slopes: {:<44} {fa}-{fb} change {change}", axis.name(), parts.join(", "));
    }

    // a plot: BPB against log10 of training FLOPs per token, one letter per family
    let pts: Vec<(f64, f64, char)> = runs
        .iter()
        .map(|r| {
            let c = flops::config_of(r, vocab, ctx).expect("checked above");
            (flops::per_token(&c, mem).train().log10(), r.bpb, r.family.chars().next().unwrap_or('?'))
        })
        .collect();
    let (x0, x1) = pts.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p.0), a.1.max(p.0)));
    let (y0, y1) = pts.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p.1), a.1.max(p.1)));
    let (w, h) = (64usize, 18usize);
    let mut grid = vec![vec![' '; w]; h];
    for &(x, y, ch) in &pts {
        let cx = (((x - x0) / (x1 - x0).max(1e-12)) * (w - 1) as f64).round() as usize;
        let cy = (((y1 - y) / (y1 - y0).max(1e-12)) * (h - 1) as f64).round() as usize;
        grid[cy][cx] = if grid[cy][cx] == ' ' || grid[cy][cx] == ch { ch } else { '*' };
    }
    println!("\nBPB (top {y1:.3}, bottom {y0:.3}) against log10 training FLOPs per token ({x0:.2} .. {x1:.2}); * = overlap");
    for row in grid {
        println!("  |{}", row.into_iter().collect::<String>());
    }
    println!("  +{}", "-".repeat(w));
    Ok(())
}
