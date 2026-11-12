// TESTING / DESIGN ONLY — fluency_gen (v42 Gr): Ava's greedy continuation of every prompt in a file, for the fluency set.
//
// fluency_gen --model FILE --prompts P.tsv --out G.tsv [--arch D] [--ctx 256] [--tokens 64] [--prompt-col 1]
//             [--batch-tokens N] [--cpu] [--bpe FILE]
//
// P: rows `id<TAB>prompt` (escaped \\ \t \n; --prompt-col picks another column, e.g. 3 for the fluency items file
// itself; `fluency_score --items F --primed-prompts OUT` writes the primed form). G: `id<TAB>text` (escaped), one row
// per prompt in P's order, which evalsets' `fluency_score --items F --gen G [--primed]` scores. Greedy (argmax), at
// most --tokens tokens (64), ending at her EOS; each prompt is read as EOS + its tokens (length marks for a glance
// model; her own marks are not printed). See src/textgen.rs.
//
// The model's shape comes from its file and arch.txt (ckpt::load_auto; --arch gives the kind, D by default).
// Batched with carried state: --batch-tokens caps rows · (prompt + new tokens) per group (default 256 on the CPU,
// 8192 on the GPU). Files that look like test sets (test, sealed, spent, heldout in the name, or such a folder) are refused.
// Speed (prompts, new tokens per second) goes to stdout.

use ava_data::bpe::Bpe;
use ava_gpu::model::Kind;
use ava_gpu::textgen::{escape, greedy_texts, read_prompts};
use std::path::{Path, PathBuf};

const USAGE: &str = "usage: fluency_gen --model FILE --prompts P.tsv --out G.tsv [--arch D] [--ctx 256] [--tokens 64] [--prompt-col 1]\n\
    \x20                  [--batch-tokens N] [--cpu] [--bpe FILE]";

fn main() {
    if let Err(e) = run() {
        eprintln!("fluency_gen: {e}");
        std::process::exit(1);
    }
}

/// A file that looks like a test set: its name holds test, sealed, spent or heldout, or it sits in a folder named one of
/// them (grow_gpu's rule; a folder like design-tests is fine). Refused: dev tools never read test sets.
fn looks_like_a_test(p: &Path) -> bool {
    let bad = ["test", "sealed", "spent", "heldout"];
    let name = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    bad.iter().any(|b| name.contains(b)) || p.components().any(|c| bad.iter().any(|b| c.as_os_str().to_string_lossy().to_lowercase() == *b))
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut model, mut prompts, mut out) = (None::<PathBuf>, None::<PathBuf>, None::<PathBuf>);
    let (mut kind, mut ctx, mut tokens, mut col, mut batch_tokens, mut cpu) = (Kind::D, 256usize, 64usize, 1usize, None::<usize>, false);
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        if key == "-h" || key == "--help" {
            return Err(USAGE.into());
        }
        if key == "--cpu" {
            cpu = true;
            i += 1;
            continue;
        }
        let v = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?;
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?} for {key}"));
        match key {
            "--model" => model = Some(v.into()),
            "--prompts" => prompts = Some(v.into()),
            "--out" => out = Some(v.into()),
            "--arch" => kind = Kind::parse(v)?,
            "--ctx" => ctx = num(v)?,
            "--tokens" => tokens = num(v)?,
            "--prompt-col" => col = num(v)?,
            "--batch-tokens" => batch_tokens = Some(num(v)?.max(1)),
            "--bpe" => bpe_path = v.into(),
            _ => return Err(format!("unknown flag {key}\n{USAGE}")),
        }
        i += 2;
    }
    let (Some(model_path), Some(prompts), Some(out)) = (model, prompts, out) else { return Err(USAGE.into()) };
    if looks_like_a_test(&prompts) {
        return Err(format!("{}: looks like a test set; fluency_gen reads dev prompts only", prompts.display()));
    }
    let dev = ava_gpu::device(cpu).map_err(|e| format!("device: {e}"))?;
    let budget = batch_tokens.unwrap_or(if dev.is_cuda() { 8192 } else { 256 });
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    let (cfg, _varmap, m) = ava_gpu::ckpt::load_auto(&model_path, kind, ctx, &dev)?;
    m.set_inference(true);
    m.set_track_activity(false);
    let text = std::fs::read_to_string(&prompts).map_err(|e| format!("{}: {e}", prompts.display()))?;
    let rows = read_prompts(&text, col);
    if rows.is_empty() {
        return Err(format!("{}: no prompts in column {col}", prompts.display()));
    }
    let ps: Vec<String> = rows.iter().map(|r| r.1.clone()).collect();
    println!(
        "{}: {:?} d{} L{}{}; {} prompts, greedy, {tokens} tokens, batches of {budget} tokens on the {}",
        model_path.display(),
        cfg.kind,
        cfg.d,
        cfg.layers,
        cfg.arch.signature(),
        ps.len(),
        if dev.is_cuda() { "GPU" } else { "CPU" }
    );
    let (texts, st) = greedy_texts(&m, &dev, &bpe, &ps, tokens, budget).map_err(|e| format!("generate: {e}"))?;
    let body: String = rows.iter().zip(&texts).map(|((id, _), t)| format!("{id}\t{}\n", escape(t))).collect();
    if let Some(d) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(d).map_err(|e| format!("create {}: {e}", d.display()))?;
    }
    std::fs::write(&out, body).map_err(|e| format!("{}: {e}", out.display()))?;
    println!(
        "gen: {} prompts, {} tokens made in {:.1} s ({:.1} new tok/s, {}) -> {}",
        st.prompts,
        st.new_tokens,
        st.seconds,
        st.new_tokens as f64 / st.seconds.max(1e-9),
        if st.batched_state { "carried state" } else { "re-read per token" },
        out.display()
    );
    Ok(())
}
