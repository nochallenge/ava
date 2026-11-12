// TESTING / DESIGN ONLY — scaling_fit (v42): does family A lose ground to family B? Checked under two curve shapes,
// with bootstrap intervals. CPU only (no model is loaded; it reads numbers).
//
// scaling_fit --data FILE [--a D] [--b T] [--x nonembed|total|flops|compute] [--target N]... [--boot 4000] [--seed 1]
//             [--bar X] [--ctx 256] [--vocab 4096] [--mem chunk16|token] [--tokens-per-weight 20]
//
// FILE: TSV rows `family d layers bpb seed` (src/flops.rs read_runs; '#' comments). Each run's size on the chosen axis
// comes from its shape (Config::weight_counts or flops::per_token), so sizes are never typed by hand. Fits use the
// mean of each (family, size) cell. For each form (log-linear a + b·ln N per family; power law E + A·N^−α with one
// floor E shared by A and B) it prints the fitted curves, the ground change gap(N_hi) − gap(N_lo) over A's measured
// span (v34's G; gap = A − B, positive = A worse), the gap at each --target (default: A's largest size and 10× it),
// the crossing, and 95% intervals from a seed bootstrap and a residual bootstrap (src/scaling.rs). The bar is 2σ of the
// pooled seed spread of every multi-seed cell in FILE (v34's rule), unless --bar is given. A verdict says "loses
// ground" only when the change is above the bar and both intervals lie above 0.

use ava_gpu::flops::{self, Axis, MemForm};
use ava_gpu::scaling::{self, Cell, Curve, Form, Row};
use std::time::Instant;

const USAGE: &str = "usage: scaling_fit --data FILE [--a D] [--b T] [--x nonembed|total|flops|compute] [--target N]... [--boot 4000]\n\
    \x20                  [--seed 1] [--bar X] [--ctx 256] [--vocab 4096] [--mem chunk16|token] [--tokens-per-weight 20]";

fn main() {
    if let Err(e) = run() {
        eprintln!("scaling_fit: {e}");
        std::process::exit(1);
    }
}

fn big(x: f64) -> String {
    if x >= 1e12 {
        format!("{:.3e}", x)
    } else if x >= 1e6 {
        format!("{:.3}M", x / 1e6)
    } else if x >= 1e3 {
        format!("{:.1}k", x / 1e3)
    } else {
        format!("{x:.3}")
    }
}

fn curve(c: &Curve) -> String {
    match *c {
        Curve::Lin { a, b } => format!("{a:.4} {} {:.5} ln N", if b < 0.0 { "-" } else { "+" }, b.abs()),
        Curve::Pow { e, amp, alpha } => format!("{e:.4} + {amp:.4} N^-{alpha:.4}"),
    }
}

fn iv(c: (f64, f64)) -> String {
    format!("[{:+.4}, {:+.4}]", c.0, c.1)
}

fn run() -> Result<(), String> {
    let (mut data, mut fa, mut fb) = (None::<String>, "D".to_string(), "T".to_string());
    let (mut axis, mut mem) = (Axis::NonEmbed, flops::DEFAULT_MEM);
    let (mut ctx, mut vocab, mut tpw) = (256usize, 4096usize, 20.0f64);
    let (mut targets, mut boot, mut seed, mut bar) = (Vec::<f64>::new(), 4000usize, 1u64, None::<f64>);
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
            "--a" => fa = v.into(),
            "--b" => fb = v.into(),
            "--x" => axis = Axis::parse(v)?,
            "--mem" => mem = MemForm::parse(v)?,
            "--ctx" => ctx = num(v)? as usize,
            "--vocab" => vocab = num(v)? as usize,
            "--tokens-per-weight" => tpw = num(v)?,
            "--target" => targets.push(num(v)?),
            "--boot" => boot = num(v)?.max(10.0) as usize,
            "--seed" => seed = num(v)? as u64,
            "--bar" => bar = Some(num(v)?),
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let data = data.ok_or_else(|| format!("--data is required\n{USAGE}"))?;
    let text = std::fs::read_to_string(&data).map_err(|e| format!("read {data}: {e}"))?;
    let runs = flops::read_runs(&text)?;
    let mut rows = Vec::with_capacity(runs.len());
    for r in &runs {
        let c = flops::config_of(r, vocab, ctx)?;
        rows.push(Row { family: r.family.clone(), x: flops::size_on(&c, axis, mem, tpw), bpb: r.bpb, seed: r.seed });
    }
    let (ca, cb) = (scaling::cells(&rows, &fa), scaling::cells(&rows, &fb));
    if ca.is_empty() || cb.is_empty() {
        return Err(format!("no runs of {fa} or {fb} in {data}"));
    }
    if targets.is_empty() {
        let hi = ca[ca.len() - 1].x;
        targets = vec![hi, 10.0 * hi];
    }
    println!("{} runs from {data}; size = {} (ctx {ctx}, vocab {vocab}); gap = {fa} - {fb} (positive: {fa} worse)", rows.len(), axis.name());
    let show = |name: &str, cs: &[Cell]| {
        let s: Vec<String> = cs.iter().map(|c| format!("{} {:.4}{}", big(c.x), c.mean(), if c.ys.len() > 1 { format!(" ({} seeds)", c.ys.len()) } else { String::new() })).collect();
        println!("  {name}: {}", s.join(" | "));
    };
    show(&fa, &ca);
    show(&fb, &cb);
    let sd = scaling::seed_sd(&rows);
    let bar = match (bar, sd) {
        (Some(b), _) => b,
        (None, Some(s)) => 2.0 * s,
        (None, None) => return Err("no multi-seed cell for the 2σ bar: pass --bar".into()),
    };
    println!(
        "  seed spread: pooled sd {} over the multi-seed cells; bar = {bar:.4}; {boot} bootstrap replicates each, SplitMix64 seed {seed}",
        sd.map_or("-".to_string(), |s| format!("{s:.5}"))
    );
    let (x_lo, x_hi) = (ca[0].x, ca[ca.len() - 1].x);
    // the size's letter in the printout: N weights, F training FLOPs per token, C training FLOPs in all
    let n = match axis {
        Axis::NonEmbed | Axis::Total => "N",
        Axis::Flops => "F",
        Axis::Compute => "C",
    };
    for form in [Form::LogLinear, Form::PowerFloor] {
        let t0 = Instant::now();
        let r = match scaling::analyse(&rows, &fa, &fb, form, &targets, boot, seed) {
            Ok(r) => r,
            Err(e) => {
                println!("\n== {}: skipped ({e})", form.name());
                continue;
            }
        };
        let secs = t0.elapsed().as_secs_f64();
        let fits = 1 + boot * if r.seed_ci.is_some() { 2 } else { 1 };
        println!("\n== {}  ({fits} fits in {secs:.2}s, {:.0} fits/s)", form.name(), fits as f64 / secs.max(1e-9));
        println!("  {fa}: BPB = {}", curve(&r.pair.a));
        println!("  {fb}: BPB = {}", curve(&r.pair.b));
        println!(
            "  rms residual {:.4}; local slope per e-fold at {n}={}: {fa} {:+.4}, {fb} {:+.4}",
            r.rms,
            big(x_hi),
            r.pair.a.slope(x_hi),
            r.pair.b.slope(x_hi)
        );
        let mut cis = vec![r.resid_ci.0];
        if let Some(s) = &r.seed_ci {
            cis.push(s.0);
        }
        println!(
            "  ground change {fa}-{fb} from {n}={} to {}: {:+.4}   seeds {}   residuals {}   -> {}",
            big(x_lo),
            big(x_hi),
            r.change,
            r.seed_ci.as_ref().map_or("-".into(), |s| iv(s.0)),
            iv(r.resid_ci.0),
            scaling::verdict(r.change, bar, &cis)
        );
        for (k, &t) in targets.iter().enumerate() {
            println!(
                "  gap at {n}={}: {:+.4}   seeds {}   residuals {}",
                big(t),
                r.gaps[k],
                r.seed_ci.as_ref().map_or("-".into(), |s| iv(s.1[k])),
                iv(r.resid_ci.1[k])
            );
        }
        println!("  gap is zero at {n} = {}", r.crossing.map_or("none in range".into(), big));
    }
    Ok(())
}
