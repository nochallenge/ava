// TESTING / DESIGN ONLY — generate: sample a continuation of a prompt from a trained T/S/H checkpoint.
//
// generate --prompt "text" [--model PATH.awts] [--bpe PATH.txt] [--arch S|T|H] [--d D --layers L --ctx C]
//          [--tokens 200] [--temperature 0.8] [--top-k 40] [--top-p 0.95] [--repetition-penalty 1.1]
//          [--seed 1] [--threads 4] [--raw]
//
// The architecture is read from the checkpoint (see src/generate.rs); flags may repeat it but not
// contradict it. ctx for S (no position table) is --ctx, default 256. Generation stops at EOS (id 256)
// unless --raw, which keeps going and prints EOS as <|eos|>. The prompt and the continuation go to
// stdout (streamed, whole UTF-8 characters only); arch, weights and tokens/s go to stderr.
// Deterministic for a seed (and for any thread count).

use ava_brain::generate::{describe, load_bpe, load_checkpoint, parse_cli, Cli, Generator, Utf8Stream, COMMON_FLAGS, SAMPLING_FLAGS};
use ava_data::bpe::{EOS, PAD};
use std::io::Write;
use std::time::Instant;

fn usage() -> ! {
    eprintln!("usage: generate --prompt \"text\" [--tokens 200] [--raw] {SAMPLING_FLAGS}\n\x20               {COMMON_FLAGS}");
    std::process::exit(2)
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cli = match parse_cli(&argv, Cli::default()) {
        Ok(Some(c)) => c,
        Ok(None) => usage(),
        Err(e) => {
            eprintln!("generate: {e}");
            usage()
        }
    };
    if let Err(e) = run(&cli) {
        eprintln!("generate: {e}");
        std::process::exit(1);
    }
}

fn run(cli: &Cli) -> Result<(), String> {
    let prompt = cli.prompt.as_deref().ok_or("--prompt is required")?;
    let t0 = Instant::now();
    let (model, params, cfg) = load_checkpoint(&cli.model, &cli.overrides)?;
    let bpe = load_bpe(&cli.bpe, cfg.vocab)?;
    eprintln!("model {}: {} (loaded in {:.2}s)", cli.model.display(), describe(&cfg, &params, cli.threads), t0.elapsed().as_secs_f64());
    eprintln!(
        "sampling: temperature {} top-k {} top-p {} repetition-penalty {} seed {} tokens {}{}",
        cli.sampling.temperature,
        cli.sampling.top_k,
        cli.sampling.top_p,
        cli.sampling.repetition_penalty,
        cli.seed,
        cli.tokens,
        if cli.raw { " raw" } else { "" }
    );

    let prompt_ids = bpe.encode(prompt);
    if prompt_ids.is_empty() {
        eprintln!("empty prompt: starting from EOS (a document start)");
    }
    if prompt_ids.len() > cfg.ctx {
        eprintln!("prompt is {} tokens; only the last {} fit the context", prompt_ids.len(), cfg.ctx);
    }
    let gen = Generator { model: &model, params: &params, threads: cli.threads };
    let out = std::io::stdout();
    let mut out = out.lock();
    write!(out, "{prompt}").map_err(|e| e.to_string())?;
    out.flush().ok();

    let mut stream = Utf8Stream::default();
    let raw = cli.raw;
    let t1 = Instant::now();
    let toks = gen.generate_with(&prompt_ids, cli.tokens, &cli.sampling, cli.seed, |t| !raw && t == EOS, |t| {
        let text = match t {
            EOS => stream.push(b"<|eos|>"),
            PAD => stream.push(b"<|pad|>"),
            _ => stream.push(bpe.token_bytes(t)),
        };
        if !text.is_empty() {
            write!(out, "{text}").ok();
            out.flush().ok();
        }
    });
    let n = toks.len();
    let secs = t1.elapsed().as_secs_f64();
    let tail = stream.finish();
    writeln!(out, "{tail}").map_err(|e| e.to_string())?;
    out.flush().ok();
    eprintln!(
        "generated {n} tokens in {secs:.2}s = {:.1} tokens/s (full forward over up to {} tokens per step)",
        n as f64 / secs.max(1e-9),
        cfg.ctx
    );
    Ok(())
}
