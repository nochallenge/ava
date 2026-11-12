// TESTING / DESIGN ONLY
//! B10: structure apart from content (the Tolman-Eichenbaum Machine, Whittington et al. 2020). The
//! hippocampal system keeps two codes: where you are in a structure (grid cells: a code moved by your
//! own actions, the same in every place of that shape) and what is there (the senses). A memory binds
//! the two. In a new place with the same structure she needs to see each thing only once; she can then
//! infer what is at a spot she reaches by a new path, because the structure code tells her she is back.
//!
//! Here the structure code is grid-cell-like: K phases, moved by each action by learned angles,
//!   φ_t = φ_0 + Σ_{s<t} θ[a_s],   g_t = [cos φ_t, sin φ_t]/√K,
//! and memory recall is attention over earlier steps by structural similarity (a Hopfield-style read):
//!   x̂_t = Σ_{τ<t} softmax_τ(β·g_t·g_τ) · x_τ
//! Trained on many worlds with fresh contents, θ learns the structure (N undoes S, E undoes W, places
//! differ), and that transfers to worlds never seen. In Ava the actions can be relations ("parent of",
//! "next day") and the contents entities, so relations are learned apart from who fills them (D6).

use candle_core::{DType, Device, Result, Tensor, D};

/// A random walk on an L×L grid (no wrap). Actions: 0 N, 1 S, 2 E, 3 W. Returns (nodes, actions), with
/// actions[t] taken from nodes[t] to nodes[t + 1].
pub fn walk(l: usize, steps: usize, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let (mut x, mut y) = (l / 2, l / 2);
    let (mut nodes, mut acts) = (vec![y * l + x], Vec::new());
    while acts.len() < steps {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let a = (s >> 33) as usize % 4;
        let ok = match a {
            0 => y + 1 < l,
            1 => y > 0,
            2 => x + 1 < l,
            _ => x > 0,
        };
        if !ok {
            continue;
        }
        match a {
            0 => y += 1,
            1 => y -= 1,
            2 => x += 1,
            _ => x -= 1,
        }
        acts.push(a);
        nodes.push(y * l + x);
    }
    (nodes, acts)
}

/// Structure codes g [T+1, 2K] from θ [4, K] (actions as a list).
pub fn codes(theta: &Tensor, acts: &[usize]) -> Result<Tensor> {
    let dev = theta.device();
    let (n_a, k) = theta.dims2()?;
    let t = acts.len();
    let mut oh = vec![0f32; t * n_a];
    acts.iter().enumerate().for_each(|(i, &a)| oh[i * n_a + a] = 1.0);
    let steps = Tensor::from_vec(oh, (t, n_a), dev)?.matmul(theta)?;
    let phase = Tensor::cat(&[Tensor::zeros((1, k), DType::F32, dev)?, steps.cumsum(0)?], 0)?;
    Tensor::cat(&[phase.cos()?, phase.sin()?], D::Minus1)? / (k as f64).sqrt()
}

/// Recall x̂_t from earlier steps for every t (row 0 recalls nothing: zeros).
pub fn recall(g: &Tensor, x: &Tensor, beta: f64) -> Result<Tensor> {
    let t = g.dim(0)?;
    let sim = (g.matmul(&g.t()?)? * beta)?;
    // Only earlier steps: everything at or after t gets −1e9.
    let mut m = vec![0f32; t * t];
    for i in 0..t {
        for j in i..t {
            m[i * t + j] = -1e9;
        }
    }
    let att = candle_nn::ops::softmax(&sim.broadcast_add(&Tensor::from_vec(m, (t, t), g.device())?)?, D::Minus1)?;
    let mut keep = vec![1f32; t];
    keep[0] = 0.0;
    att.matmul(x)?.broadcast_mul(&Tensor::from_vec(keep, (t, 1), g.device())?)
}

/// The steps that return to a node seen before (where recall has something to find).
pub fn revisits(nodes: &[usize]) -> Vec<usize> {
    (1..nodes.len()).filter(|&t| nodes[..t].contains(&nodes[t])).collect()
}

/// Random unit contents for the nodes of a fresh world, and the content seen at each step.
pub fn world(nodes: &[usize], n_nodes: usize, d_x: usize, seed: u64, dev: &Device) -> Result<(Tensor, Tensor)> {
    let c = crate::randn_seeded(n_nodes * d_x, seed, dev)?.reshape((n_nodes, d_x))?;
    let c = c.broadcast_div(&c.sqr()?.sum_keepdim(1)?.sqrt()?)?;
    let idx = Tensor::from_vec(nodes.iter().map(|&n| n as u32).collect::<Vec<u32>>(), nodes.len(), dev)?;
    let x = c.index_select(&idx, 0)?;
    Ok((c, x))
}

/// 1 − cos(recall, truth), averaged over revisits.
pub fn loss(theta: &Tensor, nodes: &[usize], acts: &[usize], x: &Tensor, beta: f64) -> Result<Tensor> {
    let r = recall(&codes(theta, acts)?, x, beta)?;
    let rv = revisits(nodes);
    let idx = Tensor::from_vec(rv.iter().map(|&t| t as u32).collect::<Vec<u32>>(), rv.len(), x.device())?;
    let (r, x) = (r.index_select(&idx, 0)?, x.index_select(&idx, 0)?);
    let cos = (&r * &x)?.sum(D::Minus1)?.broadcast_div(&(r.sqr()?.sum(D::Minus1)? + 1e-9)?.sqrt()?)?;
    cos.affine(-1.0, 1.0)?.mean_all()
}

/// The share of revisits whose recall is closest to the right node's content.
pub fn accuracy(theta: &Tensor, nodes: &[usize], acts: &[usize], contents: &Tensor, x: &Tensor, beta: f64) -> Result<f32> {
    let r = recall(&codes(theta, acts)?, x, beta)?;
    let best = r.matmul(&contents.t()?)?.argmax(D::Minus1)?.to_vec1::<u32>()?;
    let rv = revisits(nodes);
    Ok(rv.iter().filter(|&&t| best[t] as usize == nodes[t]).count() as f32 / rv.len() as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Var;
    use candle_nn::{AdamW, Optimizer, ParamsAdamW};

    const L: usize = 5;
    const K: usize = 32;
    const DX: usize = 16;
    const BETA: f64 = 30.0;

    /// Angles that follow the grid: θ_N = k·(0, 1), θ_S = −θ_N, θ_E = k·(1, 0), θ_W = −θ_E.
    fn grid_theta() -> Tensor {
        let dev = Device::Cpu;
        let r = crate::randn_seeded(2 * K, 5, &dev).unwrap().to_vec1::<f32>().unwrap();
        let mut th = vec![0f32; 4 * K];
        for j in 0..K {
            let (kx, ky) = (r[2 * j], r[2 * j + 1]);
            th[j] = ky;
            th[K + j] = -ky;
            th[2 * K + j] = kx;
            th[3 * K + j] = -kx;
        }
        Tensor::from_vec(th, (4, K), &dev).unwrap()
    }

    #[test]
    fn with_the_structure_one_visit_is_enough_in_a_new_world() {
        let (nodes, acts) = walk(L, 80, 3);
        let (c, x) = world(&nodes, L * L, DX, 4, &Device::Cpu).unwrap();
        assert!(revisits(&nodes).len() > 30);
        let acc = accuracy(&grid_theta(), &nodes, &acts, &c, &x, BETA).unwrap();
        // Without structure (each step a fresh code) revisits cannot be recognised.
        let noise = (crate::randn_seeded(4 * K, 6, &Device::Cpu).unwrap().reshape((4, K)).unwrap() * 3.0).unwrap();
        let without = accuracy(&noise, &nodes, &acts, &c, &x, BETA).unwrap();
        eprintln!("revisit recall with the grid code {acc:.2}, with a code that ignores structure {without:.2}");
        assert!(acc > 0.95 && without < 0.5, "{acc} {without}");
    }

    #[test]
    fn the_structure_is_learned_from_many_worlds_and_transfers() {
        let dev = Device::Cpu;
        let theta = Var::from_tensor(&(crate::randn_seeded(4 * K, 7, &dev).unwrap().reshape((4, K)).unwrap() * 0.5).unwrap()).unwrap();
        let (tn, ta) = walk(L, 80, 999);
        let (tc, tx) = world(&tn, L * L, DX, 998, &dev).unwrap();
        let before = accuracy(theta.as_tensor(), &tn, &ta, &tc, &tx, BETA).unwrap();
        let mut opt = AdamW::new(vec![theta.clone()], ParamsAdamW { lr: 0.03, weight_decay: 0.0, ..Default::default() }).unwrap();
        for w in 0..300u64 {
            let (nodes, acts) = walk(L, 60, 10 + w);
            let (_, x) = world(&nodes, L * L, DX, 5000 + w, &dev).unwrap();
            opt.backward_step(&loss(theta.as_tensor(), &nodes, &acts, &x, BETA).unwrap()).unwrap();
        }
        let after = accuracy(theta.as_tensor(), &tn, &ta, &tc, &tx, BETA).unwrap();
        // N then S, and E then W, should now cancel.
        let th = theta.as_tensor().to_vec2::<f32>().unwrap();
        let undo = |a: usize, b: usize| (0..K).map(|j| (th[a][j] + th[b][j]).sin().abs()).sum::<f32>() / K as f32;
        eprintln!("new world, revisit recall: before {before:.2}, after {after:.2}; N+S {:.3}, E+W {:.3}", undo(0, 1), undo(2, 3));
        assert!(after > 0.9 && after > before + 0.3, "{before} -> {after}");
    }
}
