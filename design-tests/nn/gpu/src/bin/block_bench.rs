// TESTING / DESIGN ONLY — block_bench (v40, a faster block): GPU milliseconds per call of the step's own kernels at
// D's two training shapes (d 384, 12 heads, 8,192 rows as 4 x 2048 and 32 x 256) and one long inference row.
//
// block_bench [--reps 20] [--only lif|hebb|rest|infer]  (infer: Model::forward of model D, d 384, 8 layers, --glance vocab,
// random init, set_inference(true), on one row of 16,384 tokens)
// Synchronises the device around every timed loop; inputs are random (P ~ N(0,1), Ava's gate init).

use ava_gpu::fused::{hebb_bwd_raw, hebb_fwd_raw, kwta_relu_b, lif_bwd_b, lif_fwd_b, relu_mask_bwd_b, rms_bwd_raw, rms_fwd_b, to_bf16};
use ava_gpu::model::{Config, Kind, Model};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
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

fn main() -> candle_core::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let reps: usize = a.iter().position(|x| x == "--reps").map(|i| a[i + 1].parse().expect("number")).unwrap_or(20);
    let only = a.iter().position(|x| x == "--only").map(|i| a[i + 1].clone()).unwrap_or_default();
    let on = |k: &str| only.is_empty() || only == k;
    let dev = ava_gpu::device(false)?;
    let (d, nh) = (384usize, 12usize);
    let ld = 4 * d + 2 * nh;
    let eps = 1e-5;
    let r = |name: String, ms: f64| println!("{name:<44} {ms:>8.3} ms");
    if on("infer") {
        let t = 16384usize;
        let cfg = Config::with_kind(Kind::D, 4112, d, 8, t).with_glance().map_err(candle_core::Error::Msg)?;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let model = Model::new(cfg, vb)?;
        model.set_inference(true);
        model.set_track_activity(false);
        // ordinary tokens (no length marks), a fixed pseudo-random sequence
        let ids: Vec<u32> = (0..t as u64).map(|i| ((i.wrapping_mul(2654435761) >> 7) % 4096) as u32).collect();
        let ids = Tensor::from_vec(ids, (1, t), &dev)?;
        let logits = model.forward(&ids)?;
        let finite = logits.flatten_all()?.to_vec1::<f32>()?.iter().all(|x| x.is_finite());
        let ms = time(&dev, reps.min(5), || model.forward(&ids).map(|_| ()));
        println!("Model::forward 1 x {t} (inference) {ms:.1} ms = {:.0} tok/s, logits {:?}, all finite {finite}", t as f64 * 1000.0 / ms, logits.dims());
        if only == "infer" {
            return Ok(());
        }
    }
    for (b, t) in [(4usize, 2048usize), (32, 256), (1, 16384)] {
        let n = b * t;
        let reps = if b == 1 { reps.min(5) } else { reps };
        println!("-- b {b} x t {t}");
        if on("lif") {
            let cur = Tensor::randn(0f32, 1.0, (b, t, d), &dev)?.affine(0.9, 0.35)?;
            let beta = Tensor::full(0.9f32, d, &dev)?;
            let gs = Tensor::randn(0f32, 1.0, (b, t, d), &dev)?;
            r(format!("lif_fwd_b"), time(&dev, reps, || lif_fwd_b(&cur, &beta).map(|_| ())));
            r(format!("lif_bwd_b"), time(&dev, reps, || lif_bwd_b(&cur, &beta, &gs).map(|_| ())));
        }
        if on("hebb") {
            let c = cst(nh, &dev)?;
            let p = Tensor::randn(0f32, 1.0, (n, ld), &dev)?;
            let d_o = Tensor::randn(0f32, 1.0, (n, d), &dev)?;
            let (_, ck) = hebb_fwd_raw(&p, &c, b, t, nh, eps)?;
            r(format!("hebb_fwd_raw"), time(&dev, reps, || hebb_fwd_raw(&p, &c, b, t, nh, eps).map(|_| ())));
            if b > 1 {
                r(format!("hebb_bwd_raw"), time(&dev, reps, || hebb_bwd_raw(&p, &c, &ck, &d_o, b, t, nh, eps).map(|_| ())));
            }
        }
        if on("rest") && b > 1 {
            let x = Tensor::randn(0f32, 1.0, (n, d), &dev)?;
            let g = Tensor::ones(d, candle_core::DType::F32, &dev)?;
            let u = Tensor::randn(0f32, 1.0, (n, 4 * d), &dev)?;
            let ga = Tensor::randn(0f32, 1.0, (n, 4 * d), &dev)?;
            let dp = Tensor::randn(0f32, 1.0, (n, ld), &dev)?;
            r(format!("rms_fwd_b"), time(&dev, reps, || rms_fwd_b(&x, &g, eps).map(|_| ())));
            r(format!("rms_bwd_raw"), time(&dev, reps, || rms_bwd_raw(&x, &g, &x, eps).map(|_| ())));
            r(format!("kwta_relu_b"), time(&dev, reps, || kwta_relu_b(&u, 4 * d / 8).map(|_| ())));
            let am = kwta_relu_b(&u, 4 * d / 8)?;
            r(format!("relu_mask_bwd_b"), time(&dev, reps, || relu_mask_bwd_b(&am, &ga).map(|_| ())));
            r(format!("to_bf16 dp [n, 4d+2H]"), time(&dev, reps, || to_bf16(&dp).map(|_| ())));
            r(format!("add [n, d]"), time(&dev, reps, || x.add(&x).map(|_| ())));
        }
    }
    Ok(())
}
