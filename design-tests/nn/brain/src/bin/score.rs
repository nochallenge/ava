// TESTING / DESIGN ONLY — score: bits per byte (and nats per token) of a text file under a checkpoint.
//
// score --text FILE [--model PATH.awts] [--bpe PATH.txt] [--arch S|T|H] [--d D --layers L --ctx C]
//       [--batch 8] [--threads 4]
//
// The file is tokenized as one document (Bpe::encode_doc: EOS appended, counted as 1 byte), cut into
// windows of ctx + 1 with stride ctx (last one PAD-padded) and scored with eval::evaluate, so the number
// is the same bits-per-byte train.rs reports for validation: Σ nats / ln 2 / Σ bytes of the scored
// targets, every token after the first scored once. ctx is the checkpoint's position table (T, H) or
// --ctx (default 256) for S; for T and H a --ctx that differs from the table is an error.

use ava_brain::byte_lengths;
use ava_brain::generate::{describe, load_bpe, load_checkpoint, parse_cli, score_tokens, windows_of, Cli, COMMON_FLAGS};
use std::time::Instant;

fn usage() -> ! {
    eprintln!("usage: score --text FILE [--batch 8]\n\x20            {COMMON_FLAGS}");
    std::process::exit(2)
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let cli = match parse_cli(&argv, Cli::default()) {
        Ok(Some(c)) => c,
        Ok(None) => usage(),
        Err(e) => {
            eprintln!("score: {e}");
            usage()
        }
    };
    if let Err(e) = run(&cli) {
        eprintln!("score: {e}");
        std::process::exit(1);
    }
}

fn run(cli: &Cli) -> Result<(), String> {
    let file = cli.text.as_ref().ok_or("--text FILE is required")?;
    let text = std::fs::read_to_string(file).map_err(|e| format!("read {}: {e}", file.display()))?;
    let (model, params, cfg) = load_checkpoint(&cli.model, &cli.overrides)?;
    let bpe = load_bpe(&cli.bpe, cfg.vocab)?;
    eprintln!("model {}: {}", cli.model.display(), describe(&cfg, &params, cli.threads));
    let byte_len = byte_lengths(&bpe);
    let tokens = bpe.encode_doc(&text);
    let ctx = cfg.ctx;
    let n_windows = windows_of(&tokens, ctx).len();
    let t0 = Instant::now();
    let acc = score_tokens(&model, &params, &tokens, ctx, &byte_len, cli.batch, cli.threads);
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "{}: {} bytes, {} tokens (+EOS); scored {} targets over {} bytes in {} windows of ctx {}: {:.4} nats/token, {:.4} bits/byte ({:.2}s, {:.0} tokens/s)",
        file.display(),
        text.len(),
        tokens.len() - 1,
        acc.tokens,
        acc.bytes,
        n_windows,
        ctx,
        acc.nats_per_token(),
        acc.bpb(),
        secs,
        acc.tokens as f64 / secs.max(1e-9)
    );
    Ok(())
}
