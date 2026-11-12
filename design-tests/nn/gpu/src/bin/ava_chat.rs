// TESTING / DESIGN ONLY — ava_chat: talk to Ava in the terminal; every turn goes into her diary at once.
//
// ava_chat --model FILE [--diary data/diary] [--cpu] [--temperature 0.7] [--top-k 40] [--tokens 80] [--memory 0]
//          [--bpe release/ava/bpe.txt] [--arch D] [--seed N] [--threads N] [--int8]
//
// --int8: talk with int8 weights on the CPU (Model::set_int8, src/int8.rs; implies --cpu): faster, with a small measured
// change in what she says. The diary does not record it yet.
//
// HER DIARY: THE DAILY LOOP. She keeps every conversation, never erased, so she can learn from her past and see her
// own growth. The hippocampus records the day (1), sleep files it into long-term memory (2, 3), and you can measure
// what she learned from you (4). Format and rules: src/diary.rs.
//   1. Talk:   ava_chat --model M --cpu
//              Each line you type is asked in her taught format ("Question: ...\nAnswer:", with the glance's length
//              marks if the model has them); she answers until her EOS, a new "Question:" or --tokens. Your line and
//              her reply are appended at once to data/diary/YYYY-MM-DD.tsv (local date), and how she made the reply
//              (model sha256, seed, sampler state, settings, prompt ids, drawn ids, log-probs) to
//              data/diary/YYYY-MM-DD.gen.tsv beside it (v42). Commands, each appended as
//              its own line naming the turn:  /correct THE RIGHT ANSWER   (for her last reply)
//                                             /good  /bad                (rate her last reply)
//                                             /quit  (or end of input; not recorded)   /help
//              --memory N puts the latest turns of today (as she should remember them: your correction where you
//              gave one, never a reply rated /bad) before the question, up to N tokens. 0 (the default) asks every
//              question from a fresh memory, as she was taught and as diary_growth measures.
//   2. Day:    diary_day --diary data/diary/2026-10-28.tsv --out data/diary/sleep/2026-10-28.diary-day.txt
//              (corrections and /good replies become question-answer episodes; your unrated words are kept as text
//              she heard; /bad replies are left out)
//   3. Sleep:  grow_gpu --phase sleep --init M --facts data/diary/sleep/2026-10-28.diary-day.txt --arch D
//                --bpe release/ava/bpe.txt --train data/nn/shards-v41-mix/train --val data/nn/shards-v41-mix/val
//                --anchor 100 --fisher-batches 16 --replay-rows 8 --replay-ctx 256 --replay-weight 1.0
//                --sleep-steps 300 --sleep-rows 16 --cycles 4 --surprise-alpha 1.0 --lr 3e-4 --warmup 30
//                --out design-tests/nn/gpu/runs/diary/2026-10-28
//              Tomorrow, talk to runs/diary/2026-10-28/model.safetensors.
//   4. Growth: diary_growth --model design-tests/nn/gpu/runs/diary/2026-10-28/model.safetensors --diary data/diary
//              (every correction in the whole diary: how likely the right answer is now, whether it beats her old
//              reply, and what she says now; one line per run in data/diary/growth.tsv: re-run after every sleep)
// PRIVATE: data/diary is git-ignored and must never be published. Day files and sleep outputs made from it carry
// "diary" in their paths, and a slept model's result.tsv names its --facts file, so release tooling can refuse it.
//
// v42 (S2) --conv ID [--engram-cap none|N|N:oldest|N:least-read]: she keeps this conversation's own state (LIF
// membranes, fast weights, engram store) between turns and between sessions, in conversations/ beside the brain file
// (src/state.rs ConvStore): each turn is read on top of everything said so far, in one batched pass, and the state is
// saved after her reply. A state saved by another brain is not used (the conversation starts afresh, with a note).
// --engram-cap keeps at most N engram slots (default: no cap). Not with --memory. Without --conv nothing changes.

use ava_data::bpe::Bpe;
use ava_data::rng::SplitMix64;
use ava_gpu::diary::{self, Event, Line};
use ava_gpu::model::{Kind, Model};
use ava_gpu::state::Loaded;
use std::io::{BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::time::Instant;

const USAGE: &str = "usage: ava_chat --model FILE [--diary data/diary] [--cpu] [--temperature 0.7] [--top-k 40] [--tokens 80]\n\
    \x20               [--memory 0] [--bpe FILE] [--arch D] [--seed N] [--threads N] [--conv ID [--engram-cap N]] [--int8]";

const HELP: &str = "Type a line to ask Ava. /correct THE RIGHT ANSWER fixes her last reply; /good or /bad rates it; /quit ends.\n\
    Every turn and command is saved in her diary at once; nothing is ever edited or erased.";

fn main() {
    if let Err(e) = run() {
        eprintln!("ava_chat: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut model_path: Option<PathBuf> = None;
    let mut dir = PathBuf::from(diary::DIARY_DIR);
    let mut kind = Kind::D;
    let (mut temperature, mut top_k, mut max_tokens, mut memory) = (0.7f32, 40usize, 80usize, 0usize);
    let mut cpu = false;
    let mut int8 = false;
    let mut seed: Option<u64> = None;
    // v42 (S2): --conv ID keeps this conversation's state between turns and sessions; --engram-cap caps its engram
    let (mut conv, mut cap) = (None::<String>, None::<ava_gpu::state::EngramCap>);
    let mut bpe_path = PathBuf::from(ava_gpu::DEFAULT_BPE);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        match key {
            "-h" | "--help" => return Err(USAGE.into()),
            "--cpu" => {
                cpu = true;
                i += 1;
                continue;
            }
            "--int8" => {
                // E10: the int8 talking path runs on the CPU only
                (int8, cpu) = (true, true);
                i += 1;
                continue;
            }
            _ => {}
        }
        let val = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?.as_str();
        let num = |v: &str| v.parse::<usize>().map_err(|_| format!("bad integer {v:?}"));
        match key {
            "--model" => model_path = Some(PathBuf::from(val)),
            "--diary" => dir = PathBuf::from(val),
            "--arch" => kind = Kind::parse(val)?,
            "--temperature" => temperature = val.parse().map_err(|_| format!("bad number {val:?}"))?,
            "--top-k" => top_k = num(val)?,
            "--tokens" => max_tokens = num(val)?,
            "--memory" => memory = num(val)?,
            "--seed" => seed = Some(val.parse().map_err(|_| format!("bad seed {val:?}"))?),
            "--bpe" => bpe_path = PathBuf::from(val),
            "--threads" => diary::set_threads(num(val)?),
            "--conv" => conv = Some(val.to_string()),
            "--engram-cap" => cap = ava_gpu::state::EngramCap::parse(val)?,
            _ => return Err(format!("unknown option {key}\n{USAGE}")),
        }
        i += 2;
    }
    let model_path = model_path.ok_or_else(|| format!("--model is required\n{USAGE}"))?;
    if !kind.is_ava() {
        return Err("ava_chat uses her carried state (an Ava kind, not T)".into());
    }
    let dev = ava_gpu::device(cpu).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&bpe_path).map_err(|e| format!("bpe {}: {e}", bpe_path.display()))?;
    let (cfg, _varmap, model) = ava_gpu::ckpt::load_auto(&model_path, kind, 4096, &dev)?;
    model.set_track_activity(false);
    model.set_inference(true);
    if int8 {
        model.set_int8(true).map_err(|e| format!("--int8: {e}"))?;
        // the fused one-token step (src/model/fstep.rs): the same bits, faster; it stays out by itself where it does not apply
        model.set_fused(true);
    }
    let name = diary::model_name(&model_path);
    let convs = match &conv {
        Some(_) if memory > 0 => return Err("--conv carries her own memory of the conversation; leave out --memory".into()),
        Some(id) => {
            model.state(1).map_err(|e| format!("--conv: {e}"))?;
            let s = ava_gpu::state::ConvStore::next_to(&model_path);
            eprintln!("conversation {id:?}: its state is kept in {}{}", s.path(id).display(), cap.map_or(String::new(), |c| format!(", engram capped at {} slots ({:?})", c.max, c.rule)));
            Some(s)
        }
        None => None,
    };
    let t_sha = Instant::now();
    let model_sha = diary::model_sha256(&model_path)?;
    eprintln!("model sha256 {}... ({:.2}s)", &model_sha[..16], t_sha.elapsed().as_secs_f64());
    let seed = seed.unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1));
    let mut rng = SplitMix64::new(seed);
    let top_k = top_k.max(1).min(cfg.vocab);
    let (today, _) = diary::local_now();
    eprintln!(
        "Ava ({name}: {:?} d={} L{}{}) on {}; diary {}; memory {memory} tokens; temperature {temperature}, top-k {top_k}, seed {seed}",
        kind,
        cfg.d,
        cfg.layers,
        if cfg.glance { ", glance" } else { "" },
        if dev.is_cuda() { "CUDA" } else { "CPU" },
        diary::day_path(&dir, &today).display()
    );
    eprintln!("{HELP}");

    // the turn the next command refers to: the last turn of this session, else the last of today's file
    let mut last: Option<(PathBuf, u64, String)> = {
        let p = diary::day_path(&dir, &today);
        let (lines, _) = diary::read_day(&p)?;
        lines.iter().rev().find_map(|l| match &l.event {
            Event::Turn { user, .. } => Some((p.clone(), l.turn, user.clone())),
            _ => None,
        })
    };
    let echo = !std::io::stdin().is_terminal();
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let (mut total_new, mut total_secs) = (0usize, 0f64);
    loop {
        if !echo {
            eprint!("You: ");
            std::io::stderr().flush().ok();
        }
        let Some(line) = lines.next() else { break };
        let line = line.map_err(|e| format!("stdin: {e}"))?;
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        if echo {
            println!("You: {text}");
        }
        if let Some(cmd) = text.strip_prefix('/') {
            let (word, rest) = cmd.split_once(char::is_whitespace).map_or((cmd, ""), |(w, r)| (w, r.trim()));
            let event = match word {
                "quit" | "exit" => break,
                "help" => {
                    eprintln!("{HELP}");
                    continue;
                }
                "correct" if rest.is_empty() => {
                    eprintln!("[usage: /correct THE RIGHT ANSWER]");
                    continue;
                }
                "correct" => Event::Correct(rest.to_string()),
                "good" => Event::Good,
                "bad" => Event::Bad,
                _ => {
                    eprintln!("[unknown command /{word}; not saved. {HELP}]");
                    continue;
                }
            };
            let Some((path, turn, user)) = &last else {
                eprintln!("[no reply of hers to {} yet; not saved]", if word == "correct" { "correct" } else { "rate" });
                continue;
            };
            let (_, time) = diary::local_now();
            diary::append_line(path, &Line { time, model: name.clone(), turn: *turn, event })?;
            eprintln!("[saved /{word} for turn {turn} ({:?})]", one_line_short(user));
            continue;
        }

        // a turn: ask her, then save your line and her reply at once
        let (date, time) = diary::local_now();
        let path = diary::day_path(&dir, &date);
        let mem: Vec<(String, String)> = if memory > 0 {
            let (ls, _) = diary::read_day(&path)?;
            diary::resolve(&date, &ls).0.iter().filter_map(diary::remembered).collect()
        } else {
            Vec::new()
        };
        let (ids, remembered) = diary::prompt_ids(&bpe, cfg.glance, &mem, text, memory);
        let t0 = Instant::now();
        let rng_at_start = rng.0;
        let (reply, new, trace) = match (&convs, &conv) {
            (Some(s), Some(id)) => {
                // v42 (S2): the conversation so far (her saved state), this turn read on top of it, the state saved again
                let mut st = match s.load(id, &model, &dev) {
                    Loaded::Resumed(state) => model.stream_from(state).map_err(|e| e.to_string())?,
                    Loaded::Fresh => model.stream_batched().map_err(|e| e.to_string())?,
                    Loaded::Stale(why) => {
                        eprintln!("[conversation {id:?}: its saved state does not fit this brain ({why}); starting it afresh]");
                        model.stream_batched().map_err(|e| e.to_string())?
                    }
                };
                if let Some(state) = Model::stream_state_mut(&mut st) {
                    state.cap = cap;
                }
                let r = diary::reply_traced_in(&model, &bpe, &mut st, &ids, max_tokens, temperature, top_k, &mut rng)?;
                let state = Model::stream_state(&st).ok_or("--conv: no state")?;
                s.save(id, &model, state)?;
                r
            }
            _ => diary::reply_traced(&model, &bpe, &ids, max_tokens, temperature, top_k, &mut rng)?,
        };
        let secs = t0.elapsed().as_secs_f64();
        let (ls, _) = diary::read_day(&path)?;
        let turn = diary::next_turn(&ls);
        diary::append_line(&path, &Line { time: time.clone(), model: name.clone(), turn, event: Event::Turn { user: text.to_string(), ava: reply.clone() } })?;
        let gen = diary::Gen {
            time,
            model: name.clone(),
            turn,
            model_sha256: model_sha.clone(),
            seed,
            rng: rng_at_start,
            temperature,
            top_k,
            memory,
            remembered,
            max_tokens,
            prompt_ids: ids.clone(),
            sampled_ids: trace.ids,
            logp: trace.logp,
            sample_logp: trace.sample_logp,
        };
        diary::append_gen(&diary::gen_path(&dir, &date), &gen)?;
        last = Some((path, turn, text.to_string()));
        println!("Ava: {reply}");
        std::io::stdout().flush().ok();
        total_new += new;
        total_secs += secs;
        eprintln!(
            "[turn {turn} saved; {} prompt tokens ({remembered} remembered turns) + {new} new in {secs:.2}s, {:.0} tokens/s]",
            ids.len(),
            (ids.len() + new) as f64 / secs.max(1e-9)
        );
    }
    if total_secs > 0.0 {
        eprintln!("[session: {total_new} tokens written in {total_secs:.1}s of answering]");
    }
    Ok(())
}

/// The first 40 characters of a line, for the console.
fn one_line_short(s: &str) -> String {
    let t = diary::one_line(s);
    if t.chars().count() > 40 {
        format!("{}...", t.chars().take(40).collect::<String>())
    } else {
        t
    }
}
