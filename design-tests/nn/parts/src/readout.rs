// TESTING / DESIGN ONLY
//! A learned linear readout (softmax over classes), the downstream area that reads a sense's code. Used to
//! test what the senses (`retina`, `cochlea`) make easy to learn. In Ava the code goes in as input tokens
//! instead, through a learned projection to her width d.

use candle_core::{DType, Device, Result, Tensor, Var, D};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

pub struct Readout {
    pub w: Var,
    pub b: Var,
}

impl Readout {
    /// Trains on rows `x` with labels `y` (full batch, AdamW).
    pub fn train(x: &[Vec<f32>], y: &[usize], classes: usize, steps: usize, lr: f64) -> Result<Readout> {
        let dev = Device::Cpu;
        let d = x[0].len();
        let xs = Tensor::from_vec(x.concat(), (x.len(), d), &dev)?;
        let ys = Tensor::from_vec(y.iter().map(|&c| c as u32).collect::<Vec<_>>(), y.len(), &dev)?;
        let w = Var::from_tensor(&Tensor::zeros((d, classes), DType::F32, &dev)?)?;
        let b = Var::from_tensor(&Tensor::zeros(classes, DType::F32, &dev)?)?;
        let mut opt = AdamW::new(vec![w.clone(), b.clone()], ParamsAdamW { lr, weight_decay: 1e-4, ..Default::default() })?;
        for _ in 0..steps {
            let logits = xs.matmul(w.as_tensor())?.broadcast_add(b.as_tensor())?;
            opt.backward_step(&candle_nn::loss::cross_entropy(&logits, &ys)?)?;
        }
        Ok(Readout { w, b })
    }

    pub fn predict(&self, x: &[Vec<f32>]) -> Result<Vec<usize>> {
        let xs = Tensor::from_vec(x.concat(), (x.len(), x[0].len()), &Device::Cpu)?;
        let logits = xs.matmul(self.w.as_tensor())?.broadcast_add(self.b.as_tensor())?;
        Ok(logits.argmax(D::Minus1)?.to_vec1::<u32>()?.into_iter().map(|c| c as usize).collect())
    }

    /// The class probabilities for each row.
    pub fn probs(&self, x: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let xs = Tensor::from_vec(x.concat(), (x.len(), x[0].len()), &Device::Cpu)?;
        let logits = xs.matmul(self.w.as_tensor())?.broadcast_add(self.b.as_tensor())?;
        candle_nn::ops::softmax(&logits, D::Minus1)?.to_vec2::<f32>()
    }

    pub fn accuracy(&self, x: &[Vec<f32>], y: &[usize]) -> Result<f32> {
        let p = self.predict(x)?;
        Ok(p.iter().zip(y).filter(|(a, b)| a == b).count() as f32 / y.len() as f32)
    }
}

