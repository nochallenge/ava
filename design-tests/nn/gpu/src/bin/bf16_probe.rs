// TESTING / DESIGN ONLY — bf16_probe: how far one training step's gradients move under TF32 and under bf16 matmuls,
// against plain f32, for S and T (fresh, or loaded from a trained model.safetensors).
//
// bf16_probe [--kind S|T] [--d 128] [--layers 2] [--ctx 64] [--batch 4] [--model PATH]
// Prints, per parameter, the cosine between gradients and the relative norm of their difference, then the mean.

use ava_gpu::model::{Config, Kind, Model, VOCAB};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};

fn arg(name: &str, default: &str) -> String {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).map(|i| a[i + 1].clone()).unwrap_or_else(|| default.to_string())
}

fn cos_rel(a: &Tensor, b: &Tensor) -> candle_core::Result<(f32, f32)> {
    let (a, b) = (a.flatten_all()?, b.flatten_all()?);
    let dot = a.mul(&b)?.sum_all()?.to_scalar::<f32>()?;
    let na = a.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let nb = b.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    let nd = a.sub(&b)?.sqr()?.sum_all()?.to_scalar::<f32>()?.sqrt();
    Ok((dot / (na * nb).max(1e-30), nd / na.max(1e-30)))
}

fn main() -> candle_core::Result<()> {
    let kind = Kind::parse(&arg("--kind", "S")).expect("kind");
    let (d, l, ctx, bsz): (usize, usize, usize, usize) =
        (arg("--d", "128").parse().unwrap(), arg("--layers", "2").parse().unwrap(), arg("--ctx", "64").parse().unwrap(), arg("--batch", "4").parse().unwrap());
    let dev = Device::new_cuda(0)?;
    let cfg = Config::with_kind(kind, VOCAB, d, l, ctx);
    let mut varmap = VarMap::new();
    let model = Model::new(cfg, VarBuilder::from_varmap(&varmap, DType::F32, &dev))?;
    let path = arg("--model", "");
    if !path.is_empty() {
        varmap.load(&path)?;
    }
    // a fixed pseudo-random batch from the common part of the vocabulary
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let toks: Vec<u32> = (0..bsz * (ctx + 1))
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x % 1024) as u32 + 4
        })
        .collect();
    let rows: Vec<&[u32]> = toks.chunks(ctx + 1).collect();
    let ids = Tensor::from_vec(rows.iter().flat_map(|r| r[..ctx].to_vec()).collect::<Vec<_>>(), (bsz, ctx), &dev)?;
    let tg = Tensor::from_vec(rows.iter().flat_map(|r| r[1..].to_vec()).collect::<Vec<_>>(), (bsz, ctx), &dev)?;
    let (l32, _, g32) = model.loss_and_grads_with(&ids, &tg, false)?.expect("manual path");
    candle_core::cuda::set_gemm_reduced_precision_f32(true);
    let (ltf, _, gtf) = model.loss_and_grads_with(&ids, &tg, false)?.expect("manual path");
    candle_core::cuda::set_gemm_reduced_precision_f32(false);
    let (l16, _, g16) = model.loss_and_grads_with(&ids, &tg, true)?.expect("manual path");
    println!("{kind:?} d{d} L{l}: loss f32 {:.6} tf32 {:.6} bf16 {:.6}", l32.to_scalar::<f32>()?, ltf.to_scalar::<f32>()?, l16.to_scalar::<f32>()?);
    let mut names: Vec<(String, candle_core::Var)> = varmap.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    names.sort_by(|a, b| a.0.cmp(&b.0));
    let (mut s_tf, mut s_16, mut cnt) = (0.0f32, 0.0f32, 0.0f32);
    for (name, v) in &names {
        let r = g32.get(v.as_tensor()).expect("grad");
        let (ct, rt) = cos_rel(r, gtf.get(v.as_tensor()).expect("grad"))?;
        let (cb, rb) = cos_rel(r, g16.get(v.as_tensor()).expect("grad"))?;
        println!("  {name:<16} tf32: cos {ct:.5} rel {rt:.4}   bf16: cos {cb:.5} rel {rb:.4}");
        s_tf += ct;
        s_16 += cb;
        cnt += 1.0;
    }
    println!("mean cos vs f32: tf32 {:.5}, bf16 {:.5}", s_tf / cnt, s_16 / cnt);
    Ok(())
}
