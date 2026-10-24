// TESTING / DESIGN ONLY — ava_grad: the gradient engine for the brain-like Ava language model.
// Tensors, a parameter store with checkpoints, define-by-run reverse-mode autodiff with the ops a spiking
// LM and a plain transformer need (matmul family, elementwise, norms, embedding, softmax, cross-entropy,
// k-winners-take-all, a leaky integrate-and-fire scan with surrogate gradients), and AdamW.
// Pure std, no dependencies.
//
// Modules: rng (splitmix64 + Box–Muller), tensor, params (store, Grads, AVAWTS01 checkpoints),
// matmul (threaded blocked GEMM), brain (kWTA + LIF kernels), graph (the tape), optim (AdamW, clipping,
// cosine schedule).

pub mod brain;
pub mod graph;
pub mod matmul;
pub mod optim;
pub mod params;
pub mod rng;
pub mod tensor;

pub use brain::{fraction_active, LifOpts, SpikeFn};
pub use graph::{accuracy, Graph, Var, IGNORE};
pub use matmul::threads;
pub use optim::{clip_grad_norm, cosine_lr, AdamW};
pub use params::{Grads, ParamId, Params};
pub use rng::Rng;
pub use tensor::Tensor;
