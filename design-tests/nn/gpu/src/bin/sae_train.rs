// TESTING / DESIGN ONLY — sae_train (A13): a TopK sparse autoencoder over one block's states of a saved Ava 0.4 model.
//
// sae_train --model FILE --val DIR [--block L|final[,..]] [--tokens 200000] [--latents 8d] [--k 32] [--steps N]
//           [--batch 1024] [--ctx 256] [--threads 8] [--out DIR] [--bpe FILE] [--by-source DIR] [--sources a,b]
//           [--source-tokens 16384] [--lr 0.001] [--k-aux d/2] [--aux-coef 0.03125] [--dead-after N] [--holdout 0.1]
//           [--seed 1]
//
// CPU only. Collects the states from the val windows once, trains, and writes sae.safetensors, report.md (FVU, dead
// share, L0, speed, the 40 most frequent latents' strongest contexts) and latents.tsv. See src/sae/run.rs.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = ava_gpu::sae::run::sae_train(&args) {
        eprintln!("sae_train: {e}");
        std::process::exit(1);
    }
}
