// TESTING / DESIGN ONLY — hebb_bench: D's fast-weight kernels, per-token (hebb_fwd_raw / hebb_bwd_raw with
// AVA_HEBB_CHUNK=0 semantics: fused::hebb_*) against the v38 chunk form (fused::hebc_*).
//
// hebb_bench [--reps 10] [--check-only]
// 1. Check: on random inputs at several shapes (including lengths that are not a multiple of 16), the two forms'
//    O, ck and dP (both backward passes fed the same ck), as the largest |difference| over the largest |value| of
//    each output and the cosine between them.
// 2. Speed: milliseconds per call for one layer at d 384 (12 heads of 32), 8,192 rows as 32x256, 8x1024, 2x4096.
// Inputs: P ~ N(0,1); cst = ba 0, bb 0, off = logit(1 - 2^-(5+h)) as Ava's init. The device is synchronised around
// every timed loop.

use ava_gpu::fused::{hebb_bwd_per_token, hebb_fwd_per_token, hebc_bwd_raw, hebc_fwd_raw};
use candle_core::{Device, Tensor};
use std::time::Instant;

fn time<F: FnMut() -> candle_core::Result<()>>(dev: &Device, reps: usize, mut f: F) -> f64 {
    f().unwrap();
    dev.synchronize().unwrap();
    let t = Instant::now();
    for _ in 0..reps {
        f().unwrap();
    }
    dev.synchronize().unwrap();
    t.elapsed().as_secs_f64() * 1000.0 / reps as f64
}

fn cst(nh: usize, dev: &Device) -> candle_core::Result<Tensor> {
    let mut c = vec![0f32; 3 * nh];
    for h in 0..nh {
        let g = 1.0 - 0.5f64.powi(5 + h as i32);
        c[2 * nh + h] = (g / (1.0 - g)).ln() as f32;
    }
    Tensor::from_vec(c, 3 * nh, dev)
}

/// (max |a - b| / max |b|, cosine(a, b))
fn cmp(a: &Tensor, b: &Tensor) -> candle_core::Result<(f32, f32)> {
    let (a, b) = (a.flatten_all()?.to_vec1::<f32>()?, b.flatten_all()?.to_vec1::<f32>()?);
    let (mut md, mut mb, mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(&b) {
        let (x, y) = (*x as f64, *y as f64);
        md = md.max((x - y).abs());
        mb = mb.max(y.abs());
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    Ok(((md / mb.max(1e-30)) as f32, (ab / (aa.sqrt() * bb.sqrt()).max(1e-30)) as f32))
}

fn main() -> candle_core::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let reps: usize = a.iter().position(|x| x == "--reps").map(|i| a[i + 1].parse().expect("number")).unwrap_or(10);
    let check_only = a.iter().any(|x| x == "--check-only");
    let dev = ava_gpu::device(false)?;
    let eps = 1e-5;
    let mut worst = 0f32;
    for (b, t, nh) in [(1usize, 16usize, 1usize), (2, 37, 2), (3, 100, 3), (2, 256, 12), (1, 1000, 12)] {
        let d = nh * 32;
        let ld = 4 * d + 2 * nh;
        let c = cst(nh, &dev)?;
        let p = Tensor::randn(0f32, 1.0, (b * t, ld), &dev)?;
        let d_o = Tensor::randn(0f32, 1.0, (b * t, d), &dev)?;
        let (o0, ck0) = hebb_fwd_per_token(&p, &c, b, t, nh, eps)?;
        let (o1, ck1) = hebc_fwd_raw(&p, &c, b, t, nh, eps)?;
        let g0 = hebb_bwd_per_token(&p, &c, &ck0, &d_o, b, t, nh, eps)?;
        let g1 = hebc_bwd_raw(&p, &c, &ck0, &d_o, b, t, nh, eps)?;
        let (eo, co) = cmp(&o1, &o0)?;
        let (ec, cc) = cmp(&ck1, &ck0)?;
        print!("check b {b} t {t} heads {nh}: O rel {eo:.2e} cos {co:.7}; ck rel {ec:.2e} cos {cc:.7}; dP");
        // dP by column group: q, k, v, g, za, zb
        let mut gw = eo.max(ec);
        for (name, c0, w) in [("q", 0, d), ("k", d, d), ("v", 2 * d, d), ("g", 3 * d, d), ("za", 4 * d, nh), ("zb", 4 * d + nh, nh)] {
            let (e, co) = cmp(&g1.narrow(1, c0, w)?, &g0.narrow(1, c0, w)?)?;
            print!(" {name} {e:.1e}/{co:.6}");
            gw = gw.max(e);
        }
        println!();
        worst = worst.max(gw);
    }
    println!("check: worst relative difference {worst:.2e}");
    if check_only {
        return Ok(());
    }
    let nh = 12usize;
    let d = nh * 32;
    let ld = 4 * d + 2 * nh;
    let c = cst(nh, &dev)?;
    for (b, t) in [(32usize, 256usize), (16, 512), (8, 1024), (4, 2048), (2, 4096)] {
        let p = Tensor::randn(0f32, 1.0, (b * t, ld), &dev)?;
        let d_o = Tensor::randn(0f32, 1.0, (b * t, d), &dev)?;
        let (_, ck) = hebb_fwd_per_token(&p, &c, b, t, nh, eps)?;
        let f0 = time(&dev, reps, || hebb_fwd_per_token(&p, &c, b, t, nh, eps).map(|_| ()));
        let b0 = time(&dev, reps, || hebb_bwd_per_token(&p, &c, &ck, &d_o, b, t, nh, eps).map(|_| ()));
        let f1 = time(&dev, reps, || hebc_fwd_raw(&p, &c, b, t, nh, eps).map(|_| ()));
        let b1 = time(&dev, reps, || hebc_bwd_raw(&p, &c, &ck, &d_o, b, t, nh, eps).map(|_| ()));
        println!(
            "speed {b} x {t}: per-token fwd {f0:.3} bwd {b0:.3} ms | chunk fwd {f1:.3} bwd {b1:.3} ms | both {:.2}x faster; x8 layers {:.1} -> {:.1} ms",
            (f0 + b0) / (f1 + b1),
            8.0 * (f0 + b0),
            8.0 * (f1 + b1)
        );
    }
    Ok(())
}
