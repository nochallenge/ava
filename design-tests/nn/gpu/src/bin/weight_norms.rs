// TESTING / DESIGN ONLY
//! Prints the RMS and largest |value| of each saved weight whose name contains a pattern.
//! Usage: weight_norms <model.safetensors> [pattern]
use candle_core::Device;

fn main() -> candle_core::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let pat = a.get(2).map(String::as_str).unwrap_or("");
    let mut ts: Vec<_> = candle_core::safetensors::load(&a[1], &Device::Cpu)?.into_iter().filter(|(n, _)| n.contains(pat)).collect();
    ts.sort_by(|x, y| x.0.cmp(&y.0));
    for (n, t) in ts {
        let t = t.to_dtype(candle_core::DType::F32)?.flatten_all()?;
        let rms = t.sqr()?.mean_all()?.sqrt()?.to_scalar::<f32>()?;
        let mx = t.abs()?.max(0)?.to_scalar::<f32>()?;
        println!("{n:40} {:>8} rms {rms:.4} max {mx:.4}", t.elem_count());
    }
    Ok(())
}
