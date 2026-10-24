// TESTING / DESIGN ONLY — ava_grad::tensor
// The plain n-dimensional f32 array used everywhere: a shape plus row-major (last axis fastest) data.
// No views or strides: every op produces a fresh contiguous tensor, which keeps autodiff simple.

use crate::rng::Rng;

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// Number of scalars in a shape (the empty shape has 1).
pub fn numel_of(shape: &[usize]) -> usize {
    shape.iter().product()
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor { shape: shape.to_vec(), data: vec![0.0; numel_of(shape)] }
    }

    pub fn full(shape: &[usize], value: f32) -> Tensor {
        Tensor { shape: shape.to_vec(), data: vec![value; numel_of(shape)] }
    }

    /// Panics if `data.len()` does not match the shape.
    pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Tensor {
        assert_eq!(numel_of(shape), data.len(), "Tensor::from_vec: shape {:?} needs {} values, got {}", shape, numel_of(shape), data.len());
        Tensor { shape: shape.to_vec(), data }
    }

    /// A one-element tensor of shape [1] (the shape every scalar loss has in this crate).
    pub fn scalar(value: f32) -> Tensor {
        Tensor { shape: vec![1], data: vec![value] }
    }

    /// i.i.d. N(0, std²) values from the deterministic rng.
    pub fn randn(shape: &[usize], std: f32, rng: &mut Rng) -> Tensor {
        let n = numel_of(shape);
        let data = (0..n).map(|_| rng.normal() * std).collect();
        Tensor { shape: shape.to_vec(), data }
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Size of the last axis (1 for a 0-d tensor).
    pub fn last_dim(&self) -> usize {
        self.shape.last().copied().unwrap_or(1)
    }
}
