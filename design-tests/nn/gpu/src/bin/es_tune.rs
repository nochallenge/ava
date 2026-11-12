// TESTING / DESIGN ONLY — es_tune: M7 gradient-free tuning of a trained model (evolution strategies, low-rank noise), and H1.
//
// es_tune --model FILE --items TSV[,TSV...] --params PATTERNS|h1 --out FILE
//         [--fitness rmw|acc|nll] [--pop 8] [--sigma 0.01] [--lr 0.01] [--rank 1] [--gens 10] [--seed 1]
//         [--tune-share 0.5] [--split-seed 1] [--tau X] [--norm total|byte|pmi] [--arch D] [--limit N]
//         [--batch-tokens 32768] [--cpu] [--threads 8] [--bpe release/ava/bpe.txt]
//
// After training: the model file (its arch file beside it, as ckpt::load_auto reads them) is searched over the weights
// --params selects, by a fitness that needs no gradient, and the final θ is written to --out (with NAME.arch.txt beside
// it when the model has block options) only at the end. src/es.rs has the method (EGGROLL-style rank-r antithetic
// noise from derive_seed(seed, g, k), centred ranks, θ ← θ + lr/(pop·σ)·Σ_k (u_k⁺ − u_k⁻)/2·E_k with E_k regenerated
// from its seed, every evaluation restored bit for bit); src/es/tune.rs has this tool's half (the lib tests run it).
//
// * --items: dev item TSVs of the data crate (`id kind bin right prompt c0 c1 ...`, mc::read_items), e.g.
//   data/nn/evalsets_v42/*_dev.tsv and data/nn/evalsets_v42_reason/*_dev.tsv. Any path naming a sealed set is refused
//   (model, items, out, bpe), before anything is read.
// * The items are split once by a seeded hash of each item's id, kind and prompt (--split-seed, --tune-share): the tune
//   part is the fitness; the check part is scored for the current θ every generation and only reported. Nothing is
//   chosen by it: the output is the last θ.
// * --fitness, over the tune items, each graded as mc::grade does (one batched pass per evaluation, tokenized once):
//   rmw  right − wrong per item (the owner's rule: right +1, wrong −1, "I don't know" 0 when margin < --tau; with no
//        --tau, tau = 0: she always answers), the default;
//   acc  the share right;
//   nll  −mean score of the right choice (its total NLL, or per byte / PMI with --norm).
// * --params: comma list of name prefixes in dot-separated parts, `*` inside a part (`b*.parts.holo.lam,b3.mlp`);
//   `h1` = the write-rule weights below that the model has. A pattern matching nothing is an error.
// * --sigma is per entry: fine for gains, rates and biases; for d×d matrices use far less (a rank-1 E has spectral norm
//   about σ·d). The step is about lr/σ · 0.3·√(N/pop) for N tuned weights, so the default lr is large for big
//   selections: on the released Ava's 73,920 gate weights (h1) lr 0.01 at pop 2 stepped 224% of |θ| in a CPU trial;
//   lr 2e-4 at pop 8 is about 2%. Each generation prints the step relative to |θ| (flagged above 10%).
// * Per generation: the members' mean and best tune fitness, the centre's tune and check fitness, the step, seconds per
//   generation, evaluations per second and tokens per second. NAME.es_log.tsv beside --out keeps the same per generation
//   (rewritten every generation, so a stopped run still shows its curve). Last line: RESULT fitness=.. tune0=.. tune=..
//   check0=.. check=.. s_per_gen=.. evals_per_s=..
// * --threads N (default AVA_THREADS, else 8): candle's CPU threads. --cpu runs on the CPU (else CUDA 0). On the CPU,
//   --batch-tokens 4096 was the fastest in a trial on the released Ava (60 items: 4.5 s against 8–30 s at 32768; the
//   scores are the same at every batch size).
//
// H1, a learned learning rule: ES over the memory write-rule weights, scored on dev tasks. In this codebase they are
// (src/es.rs H1_PARAMS; `--params h1` keeps the ones the model has):
//   b*.hebb.wa, b*.hebb.ba   the fast-weight memory's forget gate, alpha_t = sigmoid(h·Wa + ba + logit γ_h) (model.rs
//                            Gates; every G, R, D and A block, so every released Ava has them)
//   b*.hebb.wb, b*.hebb.bb   its write strength, beta_t = 2·sigmoid(h·Wb + bb); in D's delta rule
//                            F_t = alpha_t (I − beta_t k kᵀ) F_{t−1} + beta_t k vᵀ
//   b*.hebb.wa1, .wa2, .wb1, .wb2, .we   --gates channel (arch.rs P2): per-channel forget and write (rank 32) and the
//                            erase key
//   b{L}.eng.wp, b{L}.eng.bp the engram's L3 BTSP write (memory.rs, --btsp W): plateau p_j = sigmoid(h_j·wp + bp)
//   b{L}.eng.btsp_g          how strongly a slot written without a plateau is hidden (score bias g_h · ln p_j)
//   b{L}.eng.btsp_mix        how far the written key is bound to the W tokens before it (the eligibility window)
//   b*.parts.holo.lam        the D2 holographic trace's fade, T_t = λ·T_{t−1} + r_t ⊛ f_t, λ = sigmoid(lam) (--holo)
//   b*.parts.tem.wth         B10 TEM's moves, path-integrated into the structural code each token is stored under (--tem)
//   b*.parts.wm.wg, .bg      B9 working memory's write gates (--wm)
//   b{L}.parts.ws.wg, .bg, .ws   B3 workspace: which slots win the write, and the overall write strength (--workspace)
//   b*.parts.mod.gains       B8 neuromodulation's gains of surprise on the write strength, the forget push and the
//                            input (--modulate)
//   b*.parts.stp.u_param, .f_param, .d_param   E9 short-term facilitation and depression (--stp)
// The read side (b*.hebb.wg the output gate, b{L}.eng.tau/.ss the read's sharpness) and the projections (wq, wk, wv,
// wo) are not the rule; select them by name if wanted.
//
// Example (H1 on the released Ava, one dev set of facts and one of binding, right − wrong at tau 0):
//   es_tune --model release/ava/model.safetensors \
//     --items data/nn/evalsets_v42/facts_dev.tsv,data/nn/evalsets_v42_reason/binding_dev.tsv \
//     --params h1 --fitness rmw --pop 8 --sigma 0.01 --lr 0.0002 --rank 1 --gens 10 --seed 1 --batch-tokens 4096 \
//     --out design-tests/nn/gpu/runs/h1/ava-h1.safetensors
// (the same with --params b*.hebb.wb,b*.hebb.bb tunes the write strength only; time a --limit 100 --gens 1 trial first).

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let a = match ava_gpu::es::tune::parse(&args) {
        Ok(a) => a,
        Err(e) if e.starts_with("usage") && args.iter().any(|x| x == "-h" || x == "--help") => {
            println!("{e}");
            return;
        }
        Err(e) => {
            eprintln!("es_tune: {e}");
            std::process::exit(2);
        }
    };
    // before the first tensor op
    ava_gpu::diary::set_threads(a.threads);
    if let Err(e) = ava_gpu::es::tune::run(&a) {
        eprintln!("es_tune: {e}");
        std::process::exit(1);
    }
}
