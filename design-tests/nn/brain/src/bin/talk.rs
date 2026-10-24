// TESTING / DESIGN ONLY — talk: a plain-stdin line REPL over a trained T/S/H checkpoint.
//
// talk [--model PATH.awts] [--bpe PATH.txt] [--arch S|T|H] [--d D --layers L --ctx C] [--tokens 60]
//      [--temperature 0.8] [--top-k 40] [--top-p 0.95] [--repetition-penalty 1.1] [--seed 1] [--threads 4]
//
// Each line typed is appended (with a newline) to a rolling conversation buffer; the model continues it
// for up to --tokens tokens, stopping at EOS or at the first token that contains a newline; the reply is
// printed (streamed) and appended to the buffer followed by a newline, and the buffer keeps only the
// last ctx tokens. The seed advances by one per turn, so the same transcript replays identically.
// `quit`, `exit` or end of input ends the session. No line editing (no rustyline): plain stdin.

use ava_brain::generate::{describe, load_bpe, load_checkpoint, parse_cli, Cli, Generator, Utf8Stream, COMMON_FLAGS, SAMPLING_FLAGS};
use ava_data::bpe::EOS;
use std::io::{BufRead, Write};
use std::time::Instant;

const NEWLINE_ID: u32 = b'\n' as u32;

fn usage() -> ! {
    eprintln!("usage: talk [--tokens 60] {SAMPLING_FLAGS}\n\x20           {COMMON_FLAGS}");
    std::process::exit(2)
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let defaults = Cli { tokens: 60, ..Cli::default() };
    let cli = match parse_cli(&argv, defaults) {
        Ok(Some(c)) => c,
        Ok(None) => usage(),
        Err(e) => {
            eprintln!("talk: {e}");
            usage()
        }
    };
    if let Err(e) = run(&cli) {
        eprintln!("talk: {e}");
        std::process::exit(1);
    }
}

fn run(cli: &Cli) -> Result<(), String> {
    let (model, params, cfg) = load_checkpoint(&cli.model, &cli.overrides)?;
    let bpe = load_bpe(&cli.bpe, cfg.vocab)?;
    eprintln!("model {}: {}", cli.model.display(), describe(&cfg, &params, cli.threads));
    eprintln!("type a line and press enter; `quit`, `exit` or end of input ends the session");
    let gen = Generator { model: &model, params: &params, threads: cli.threads };

    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let out = std::io::stdout();
    let mut out = out.lock();
    let mut buffer: Vec<u32> = Vec::new();
    let mut turn: u64 = 0;
    loop {
        write!(out, "> ").ok();
        out.flush().ok();
        let line = match lines.next() {
            Some(Ok(l)) => l,
            Some(Err(e)) => return Err(format!("stdin: {e}")),
            None => break,
        };
        let line = line.trim_end_matches(['\r', '\n']);
        let word = line.trim();
        if word == "quit" || word == "exit" {
            break;
        }
        if word.is_empty() {
            continue;
        }
        buffer.extend(bpe.encode(&format!("{line}\n")));
        if buffer.len() > cfg.ctx {
            buffer.drain(..buffer.len() - cfg.ctx);
        }

        let mut stream = Utf8Stream::default();
        let t0 = Instant::now();
        let stop = |t: u32| t == EOS || bpe.token_bytes(t).contains(&b'\n');
        let reply = gen.generate_with(&buffer, cli.tokens, &cli.sampling, cli.seed.wrapping_add(turn), stop, |t| {
            let text = stream.push(bpe.token_bytes(t));
            if !text.is_empty() {
                write!(out, "{text}").ok();
                out.flush().ok();
            }
        });
        let secs = t0.elapsed().as_secs_f64();
        let tail = stream.finish();
        writeln!(out, "{tail}").ok();
        out.flush().ok();
        eprintln!("[{} tokens, {:.1} tokens/s]", reply.len(), reply.len() as f64 / secs.max(1e-9));

        buffer.extend(reply);
        buffer.push(NEWLINE_ID);
        if buffer.len() > cfg.ctx {
            buffer.drain(..buffer.len() - cfg.ctx);
        }
        turn += 1;
    }
    Ok(())
}
