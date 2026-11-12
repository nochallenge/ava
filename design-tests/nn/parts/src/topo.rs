// TESTING / DESIGN ONLY
//! H12 topography. Cortex is laid out in maps: neighbouring neurons respond to similar things (orientation
//! maps, face patches, the tonotopic map). TDANN (Margalit et al., Neuron 2024) gets such maps in a network
//! by giving each unit a place on a 2-D sheet and adding a spatial loss. Units close together should
//! respond alike, and units far apart need not. That saves wiring and makes her parts easier to read.
//!
//! The loss over a batch of activations r [n, units] (units on a w×h grid, row by row):
//!   L = 1 − pearson(corr_ij, 1 / (1 + dist_ij))   over all pairs i < j
//! corr_ij is the correlation of units i and j over the batch.

use candle_core::{Result, Tensor};

/// 1 / (1 + distance) between every pair of units on a `w`-wide grid, as [units, units].
pub fn inverse_distance(units: usize, w: usize, dev: &candle_core::Device) -> Result<Tensor> {
    let mut v = Vec::with_capacity(units * units);
    for i in 0..units {
        for j in 0..units {
            let (dx, dy) = ((i % w) as f32 - (j % w) as f32, (i / w) as f32 - (j / w) as f32);
            v.push(1.0 / (1.0 + (dx * dx + dy * dy).sqrt()));
        }
    }
    Tensor::from_vec(v, (units, units), dev)
}

fn pearson_flat(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let ac = a.broadcast_sub(&a.mean_all()?)?;
    let bc = b.broadcast_sub(&b.mean_all()?)?;
    let num = (&ac * &bc)?.sum_all()?;
    let den = (ac.sqr()?.sum_all()? * bc.sqr()?.sum_all()?)?.sqrt()?;
    num / (den + 1e-12)?
}

/// The spatial loss for activations r [n, units] on a `w`-wide grid.
pub fn spatial_loss(r: &Tensor, w: usize) -> Result<Tensor> {
    let (_, u) = r.dims2()?;
    let rc = r.broadcast_sub(&r.mean_keepdim(0)?)?;
    let sd = (rc.sqr()?.sum_keepdim(0)? + 1e-8)?.sqrt()?;
    let z = rc.broadcast_div(&sd)?;
    let corr = z.t()?.matmul(&z)?;
    let inv = inverse_distance(u, w, r.device())?;
    // Pairs i < j only.
    let idx: Vec<u32> = (0..u * u).filter(|k| k / u < k % u).map(|k| k as u32).collect();
    let idx = Tensor::from_vec(idx.clone(), idx.len(), r.device())?;
    let c = corr.flatten_all()?.index_select(&idx, 0)?;
    let d = inv.flatten_all()?.index_select(&idx, 0)?;
    pearson_flat(&c, &d)?.affine(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::*;
    use candle_core::{Device, Var};

    #[test]
    fn smooth_maps_score_better_and_the_loss_has_a_gradient() {
        let (n, w) = (256, 6);
        let u = w * w;
        // A map: independent noise blurred across the sheet, so neighbours share most of their input.
        let mut blur = vec![0f32; u * u];
        for i in 0..u {
            for j in 0..u {
                let (dx, dy) = ((i % w) as f32 - (j % w) as f32, (i / w) as f32 - (j / w) as f32);
                blur[i * u + j] = (-(dx * dx + dy * dy) / (2.0 * 1.5 * 1.5)).exp();
            }
        }
        let blur = Tensor::from_vec(blur, (u, u), &Device::Cpu).unwrap();
        let smooth = randn(&[n, u], 41).matmul(&blur).unwrap();
        let noise = randn(&[n, u], 42);
        let (ls, ln) = (spatial_loss(&smooth, w).unwrap().to_scalar::<f32>().unwrap(), spatial_loss(&noise, w).unwrap().to_scalar::<f32>().unwrap());
        assert!(ls < 0.5 && ln > 0.8, "smooth {ls}, noise {ln}");
        let v = Var::from_tensor(&noise).unwrap();
        let g = spatial_loss(v.as_tensor(), w).unwrap().backward().unwrap();
        let gn = g.get(v.as_tensor()).unwrap().abs().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(gn > 1e-4);
    }
}
