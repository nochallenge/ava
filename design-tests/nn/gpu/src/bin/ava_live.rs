// TESTING / DESIGN ONLY — ava_live: watch Ava's real neurons fire, live, while she reads and answers.
//
// ava_live [--model FILE] [--port 8077] [--threads 2] [--pace-ms 120] [--temperature 0.7] [--top-k 40] [--tokens 60]
//          [--seed 1] [--bpe FILE] [--diary DIR] [--arch D] [--cpu | --gpu] [--learner-root DIR [--watch-ms 2000]] [--int8]
//          [--memory 0] [--read-pace-ms N] [--chats DIR] [--library FILE] [--name TEXT] [--open] [--cb-online ETA]
//          [--learn e2,cb2|off]
//
// A tiny std-only web server on 127.0.0.1 (never another interface; default port 8077):
//   GET  /        the chat app: talk to her over the hologram brain (design-tests/nn/viz/ava_chat.html, built in)
//   GET  /brain_mesh.js  the brain (BodyParts3D, CC BY 4.0; design-tests/nn/viz/brain/, built in, so a download shows it)
//   GET  /neurons the neuron viewer (design-tests/nn/viz/ava_live.html, built in)
//   GET  /events  Server-Sent Events: one "hello" (the model's shape) on connect, then the live stream
//   POST /say     {"mode": "talk" | "continue", "text": "..."}  (Content-Type: application/json)
//   POST /stop    stop the current turn after the token being computed
// --memory N: a talk turn's prompt starts with her earlier talk turns of this run (newest first, up to N tokens, as
// ava_chat --memory does); 0 (the default) keeps every turn fresh. --read-pace-ms: the pace while she reads (default:
// --pace-ms), so a long remembered prompt need not be read at speaking pace.
// A turn starts from a fresh memory. Ava READS the text one token at a time (talk: her taught format
// "Question: ...\nAnswer:", after EOS; continue: EOS then the text as it is), then ANSWERS one token at a time
// (talk: until her EOS, the start of a new question, or --tokens; continue: until EOS or --tokens). Every token is one
// `Model::step_trace` (the exact arithmetic of `Model::step`), and its event goes out the moment it is computed:
// which LIF neurons spiked in each of her blocks, the kWTA winners of each MLP with their activity, the fast-weight
// memory per head (write = size of what it stored, read = size of what it recalled, keep and write gates), the engram's
// attention over every earlier token, the glance place code and magnitude code, and her top-5 guesses for the next
// token. --pace-ms makes each token take at least that long so a person can follow (0 = full speed); nothing is ever
// computed ahead. While idle she rests: no events. --diary DIR appends each talk turn to her diary (ava_gpu::diary;
// the default is no diary). The device is the CPU unless --gpu is given. Threads: --threads (default 2).
// --int8 (CPU only): she talks with int8 weights (Model::set_int8, src/int8.rs: E10 int8 matmuls and E2 event sums),
// faster at the cost of a small measured change in what she says; a swapped-in brain gets its int8 copies too.
//
// v42 (Wave 4, K6) HOT SWAP. With --learner-root DIR (the always-online learner's folder, design-tests/nn/online) she
// follows the learner's DIR/CURRENT: at start she serves the brain it names (else --model until it names one), and a
// watcher thread (src/hotswap.rs) reads it every --watch-ms. When it changes, the new brain is loaded and checked on that
// thread (it must load, be one of Ava's kinds with the same vocabulary, and give finite logits on a test prompt) while
// she keeps answering; it goes live between replies only (before the next turn, or while idle), never mid-reply, and
// requests are never dropped. Viewers get a new "hello" with the new brain's shape. A brain that fails to load or to pass
// is logged with the reason and the old one keeps serving. The diary names the brain that wrote each reply (its file and
// sha256). Without --learner-root nothing of this runs and she behaves exactly as before.
//
// v49 (Ava 0.4's chat app). DOUBLE-CLICK START: run with no arguments, she finds her files next to the program
// (model.safetensors with its arch.txt, bpe.txt, library.tsv, and name.txt for the name the page shows), talks with
// int8 weights and the fused step, saves chats in a `chats` folder beside it (no diary), serves on 127.0.0.1 port
// 8077 or the next free one, and opens the default browser at the page. A missing file is named in the console window,
// which then waits for Enter so the message can be read. Any argument gives the developer's flags as before.
// (AVA_NO_BROWSER=1: no browser is opened, for tests.)
// --name TEXT: the name the page shows (default: NAME.name.txt beside the model file, else name.txt in its folder,
// else "Ava"); the page shows no paths or settings. Block options (Ava 0.4) stream through a State and are traced on the
// step `step` runs (`Model::step_trace`; the fused step with its record on, the same bits). Each token's event also
// carries "io": the size of what entered her ("in"), how sure she was of the word she says ("conf", say only) and how
// much her next-word guess moved since the last token ("chg", total variation), and per block "h", the sum of its MLP
// units; spikes are the neurons whose spike is not 0 (int2 spikes can be negative).
//
// v51 (Part B's anatomy map). The hello's model carries "parts" (which brain parts she has: thalamus, cerebellum,
// nucleus, memory, engram, cb_online), and each token's event, for a brain with --thalamus, --cerebellum-head or
// --neuromod, carries "parts": "th" (per block, the mean residual and memory gate), "cb" (the size of the cerebellum's
// correction), "cbo" (its online share) and "nm" (NE, ACh, DA). They are read on the side and never feed back.
// --cb-online ETA: a cerebellum brain learns online from her error on each word she reads (never her own words).
//
// v52 LIVE LEARNING (--learn e2,cb2 | e2 | cb2 | off; the double-click start: cb2 with int8 (owner choice 11-07, speed first); the developer's flags: off). She
// keeps learning while she reads and talks, with v51 Part B's two learners as runs/v51/prereg_v51.md Result B-4 kept
// them and v51_online's `stream` ran them: E2, e-prop in her last two blocks (W_in, W_out, W1, W2; rate 0.001; the
// changes fade back to her saved weights with a half-life of 128 tokens), and CB2, the cerebellum's online table
// (12288 granule units, a zero read-out added to a brain without one, the same logits; rate 0.3 / mean |E|^2; half-life
// 256 tokens). She learns from everything she reads in a turn, the prompt (your words, her earlier turns that --memory
// puts in, the library's passages) and her own reply: each token learns to predict the next one, the text's own while
// she reads and the word she draws while she speaks (the turn's last token, when she stops at --tokens or is stopped,
// learns nothing); each turn starts from a fresh state, as each document of the stream did (the e-prop traces restart,
// the changes carry on and fade per token learned). Every token runs fast4's fused session step only
// (`Model::fast4_live_step`, src/fast4/live.rs: e-prop's tape with the cerebellum on it when both learn), never the
// State path: where fast4 cannot take this brain or setting the console says why and she talks without learning (and
// a token it refuses mid-turn ends the learning: she answers that turn again without it). E2 learns in f32 (fast4's
// e-prop step has no int8 path), so with e2 she talks in f32 even with --int8; cb2 alone keeps --int8 (the table learns
// in f32 beside the int8 weights: the fused step is the int8 State path's bits). With --chats each chat keeps its own
// changes in CHATS/ID.learn.safetensors beside ID.json (exact f32 values, the cerebellum's stored table and scale, the brain's
// sha256 and the settings), saved after each turn and read back when the chat is opened again; a new chat starts from
// her base brain, a deleted chat's changes are deleted with it, and changes made by another brain or other settings
// are not used (that chat starts again from the base brain). Without --chats the changes last for the run. Each token's
// event carries "learn": {"e2": the rms of e-prop's change, "cb": the rms of the table's change}, which the chat page
// shows (her planning area, where her last blocks light, and her cerebellum glow with them); the hello carries
// "learn". Not with --cb-online (the older rule) or --gpu.

use ava_data::bpe::{Bpe, EOS};
use ava_data::numbers::{is_mark, strip_marks, MARK_BASE};
use ava_data::rng::SplitMix64;
use ava_gpu::arch::Arch;
use ava_gpu::diary;
use ava_gpu::eprop::Site;
use ava_gpu::hotswap::{self, Current, WatchOpts, Watcher};
use ava_gpu::model::{Config, Kind, Model, StepTrace};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

const USAGE: &str = "usage: ava_live [--model FILE] [--port 8077] [--threads 2] [--pace-ms 120] [--temperature 0.7] [--top-k 40]\n\
    \x20               [--tokens 60] [--seed 1] [--bpe FILE] [--diary DIR] [--arch D] [--cpu | --gpu]\n\
    \x20               [--learner-root DIR [--watch-ms 2000]] [--int8] [--memory 0] [--read-pace-ms N] [--chats DIR] [--library FILE]\n\
    \x20               [--name TEXT] [--open] [--cb-online ETA] [--learn e2,cb2|e2|cb2|off]\n\
    with no arguments: the files next to this program, CPU int8, chats beside it, the next free port, the browser opened,\n\
    \x20 learning live (--learn cb2: her cerebellum's table learns in f32 beside the int8 weights; each chat keeps its own changes)";
const PAGE: &str = include_str!("../../../viz/ava_live.html");
// v51: the chat page carries the anatomy map of v51 Part B
const CHAT_PAGE: &str = include_str!("../../../viz/ava_chat.html");
const BRAIN_MESH: &str = include_str!("../../../viz/brain/brain_mesh.js");
const DEFAULT_MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/runs/v41/reward/model.safetensors");
const MAX_TEXT: usize = 2000;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Talk,
    Continue,
}

struct Job {
    mode: Mode,
    text: String,
    /// the saved chat this turn belongs to (with --chats)
    chat: Option<String>,
}

// ---------- saved chats (with --chats DIR): one JSON file per chat, on this computer only ----------

/// A chat id names a file, so only letters, digits and '-' (no path tricks).
fn chat_id_ok(id: &str) -> bool {
    !id.is_empty() && id.len() <= 48 && id.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
}

fn chat_path(dir: &Path, id: &str) -> PathBuf {
    dir.join(format!("{id}.json"))
}

fn read_chat(dir: &Path, id: &str) -> Option<Value> {
    serde_json::from_slice(&std::fs::read(chat_path(dir, id)).ok()?).ok()
}

/// Her earlier turns in this chat, oldest first, for the prompt (--memory).
fn chat_history(dir: &Path, id: &str) -> Vec<(String, String)> {
    let Some(c) = read_chat(dir, id) else { return Vec::new() };
    c["turns"]
        .as_array()
        .map(|t| {
            t.iter()
                .filter_map(|x| Some((x["user"].as_str()?.to_string(), x["ava"].as_str()?.to_string())))
                .filter(|(_, a)| !a.trim().is_empty())
                .collect()
        })
        .unwrap_or_default()
}

/// Add one turn to a chat (made on its first turn, titled by its first message); written whole, then renamed.
fn append_chat(dir: &Path, id: &str, user: &str, ava: &str) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("chats dir: {e}"))?;
    let (date, time) = diary::local_now();
    let now = format!("{date} {time}");
    let mut c = read_chat(dir, id).unwrap_or_else(|| {
        let title: String = user.chars().take(48).collect();
        json!({ "id": id, "title": title, "created": now, "turns": [] })
    });
    c["updated"] = json!(now);
    if let Some(t) = c["turns"].as_array_mut() {
        t.push(json!({ "user": user, "ava": ava, "time": now }));
    }
    let tmp = dir.join(format!("{id}.json.tmp"));
    std::fs::write(&tmp, c.to_string()).map_err(|e| format!("save chat: {e}"))?;
    std::fs::rename(&tmp, chat_path(dir, id)).map_err(|e| format!("save chat: {e}"))
}

/// Delete a saved chat. v52: her learned changes of the chat go with it, and the serving brain reads its changes again
/// before its next turn.
fn delete_chat(shared: &Shared, dir: &Path, id: &str) {
    let _ = std::fs::remove_file(chat_path(dir, id));
    let _ = std::fs::remove_file(learn_path(dir, id));
    shared.chats_epoch.fetch_add(1, Ordering::SeqCst);
}

/// Every saved chat (id, title, updated, turns), newest first.
fn list_chats(dir: &Path) -> Value {
    let mut v: Vec<Value> = std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| {
                    let name = e.file_name().into_string().ok()?;
                    let id = name.strip_suffix(".json")?;
                    let c = read_chat(dir, id)?;
                    Some(json!({ "id": id, "title": c["title"], "updated": c["updated"], "n": c["turns"].as_array().map_or(0, |t| t.len()) }))
                })
                .collect()
        })
        .unwrap_or_default();
    v.sort_by(|a, b| b["updated"].as_str().unwrap_or("").cmp(a["updated"].as_str().unwrap_or("")));
    Value::Array(v)
}

// ---------- v52: live learning (--learn): E2 (e-prop) and CB2 (the cerebellum's online table), kept per chat ----------

/// What learns while she reads and talks (`--learn e2,cb2`, any subset, or off).
#[derive(Clone, Copy, Default, PartialEq, Debug)]
struct LearnOpts {
    e2: bool,
    cb2: bool,
}

impl LearnOpts {
    fn parse(s: &str) -> Result<LearnOpts, String> {
        let mut o = LearnOpts::default();
        if s.trim() == "off" {
            return Ok(o);
        }
        for p in s.split(',') {
            match p.trim() {
                "e2" => o.e2 = true,
                "cb2" => o.cb2 = true,
                x => return Err(format!("--learn {s:?}: e2, cb2 (comma-separated) or off ({x:?} is neither)")),
            }
        }
        Ok(o)
    }

    fn any(self) -> bool {
        self.e2 || self.cb2
    }

    fn name(self) -> &'static str {
        match (self.e2, self.cb2) {
            (true, true) => "e2,cb2",
            (true, false) => "e2",
            (false, true) => "cb2",
            (false, false) => "off",
        }
    }
}

// v51 Part B's E2 and CB2 as v51_online `stream` ran them (runs/v51/prereg_v51.md: Amendments B-2, B-5, Result B-4)
const E2_LR: f64 = 0.001;
const E2_HALF: f64 = 128.0;
const CB2_UNITS: usize = 12288;
const CB2_ETA: f64 = 0.3;
const CB2_HALF: f64 = 256.0;

/// A chat's learned changes, beside its JSON.
fn learn_path(dir: &Path, id: &str) -> PathBuf {
    dir.join(format!("{id}.learn.safetensors"))
}

/// The live-learning session of the serving brain: what learns, with what, and whose changes it holds.
struct Learn {
    opts: LearnOpts,
    /// e-prop's plastic blocks (her last two, as v51_online's default)
    blocks: Vec<usize>,
    /// CB2's rate: eta / mean |E_v|^2 over her word embeddings, as v51_online computes it (f64, then f32)
    rate: f32,
    /// the brain the changes belong to (sha256 of its file) and the settings they were made with
    brain: String,
    settings: String,
    /// whose changes the brain holds now: None = read them before the next turn; Some(None) = this run's own (no chat)
    loaded: Option<Option<String>>,
    /// the chats folder's delete count when they were read (a delete reads them again)
    epoch: u64,
    /// tokens learned into the changes held now
    tokens: u64,
    /// tests: every token stepped while learning (token, the target it learned, the logits' bits)
    log: Option<Vec<(u32, Option<u32>, Vec<u32>)>>,
    /// fast4 refused a token of this session (why): learning stops after the turn, and she talks on without it
    refused: Option<String>,
}

/// mean |E_v|^2 over the word embeddings (v51_online's `emb_sq`, the cerebellum rate's scale)
fn emb_sq(vm: &VarMap) -> Result<f64, String> {
    let e = vm.data().lock().unwrap().get("tok_emb").ok_or("no tok_emb")?.as_tensor().clone();
    let v = e.dim(0).map_err(|e| e.to_string())?;
    Ok(e.sqr().and_then(|x| x.sum_all()).and_then(|x| x.to_scalar::<f32>()).map_err(|e| e.to_string())? as f64 / v as f64)
}

/// v51_online's `load_plus`: the brain plus a zero cerebellum head of `units` units (lean, the same logits), for CB2.
fn load_plus_cb(path: &Path, units: usize, dev: &Device) -> Result<(Config, VarMap, Model), String> {
    let t = ava_gpu::ckpt::read(path)?;
    let cfg0 = ava_gpu::ckpt::detect(&t, Kind::D, 4096)?;
    let (ar0, _) = ava_gpu::ckpt::arch_beside(path)?.ok_or("the brain needs its arch.txt beside it")?;
    let mut ar = Arch::from_flags(&format!("{} --cerebellum-head {units}", ar0.flags()))?;
    ar.top = t.contains_key("top_emb");
    let cfg = cfg0.with_arch(ar)?;
    drop(t);
    let vm = VarMap::new();
    let _build = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, dev)).map_err(|e| e.to_string())?;
    let (_copied, kept) = ava_gpu::ckpt::init_from(&vm, path)?;
    if kept != ["cbh.wo"] {
        return Err(format!("expected only the cerebellum read-out to be new, kept {kept:?}"));
    }
    let m = ava_gpu::ckpt::lean(cfg, &vm, dev)?;
    Ok((cfg, vm, m))
}

impl Learn {
    /// Start (or restart: every change 0) the sessions on `m`, with v51_online stream's calls for one row.
    fn start(&self, m: &Model) -> Result<(), String> {
        let e = |e: candle_core::Error| e.to_string();
        if self.opts.e2 {
            m.plastic_begin(1, &self.blocks, &Site::ALL).map_err(e)?;
            m.plastic_set_half_lives(&[E2_HALF]).map_err(e)?;
            m.plastic_probes(true);
        }
        if self.opts.cb2 {
            m.cb_online_begin(1, self.rate).map_err(e)?;
            m.cb_online_set_lr(0, self.rate);
            m.cb_online_set_half_life(0, CB2_HALF).map_err(e)?;
        }
        Ok(())
    }

    /// Stop both sessions (her base brain again).
    fn end(m: &Model) {
        m.plastic_end();
        m.cb_online_end();
    }

    /// Her base brain: every change 0.
    fn reset(&mut self, m: &Model) -> Result<(), String> {
        self.start(m)?;
        self.tokens = 0;
        Ok(())
    }

    /// The changes as they are now (exact values): e-prop's per site [1, in, out], the table's stored values and scale.
    fn tensors(&self, m: &Model) -> Result<HashMap<String, Tensor>, String> {
        let cpu = Device::Cpu;
        let e = |e: candle_core::Error| e.to_string();
        let mut ts = HashMap::new();
        if self.opts.e2 {
            for ((i, s), d) in m.plastic_deltas().map_err(e)? {
                ts.insert(format!("e2.{}", s.var_name(i)), d);
            }
        }
        if self.opts.cb2 {
            let (tab, sc, up) = m.cb_online_with(|o| (o.raw(0).0.to_vec(), o.raw(0).1, o.updates[0])).ok_or("no online table")?;
            if !tab.is_empty() {
                let n = tab.len();
                ts.insert("cb2.delta".into(), Tensor::from_vec(tab, n, &cpu).map_err(e)?);
            }
            ts.insert("cb2.scale".into(), Tensor::from_vec(vec![sc], 1, &cpu).map_err(e)?);
            ts.insert("cb2.updates".into(), Tensor::from_vec(vec![up as u64 as f64], 1, &cpu).map_err(e)?);
        }
        let text = |s: &str| Tensor::from_vec(s.as_bytes().to_vec(), s.len(), &cpu).map_err(e);
        ts.insert("meta.brain".into(), text(&self.brain)?);
        ts.insert("meta.settings".into(), text(&self.settings)?);
        ts.insert("meta.tokens".into(), Tensor::from_vec(vec![self.tokens as f64], 1, &cpu).map_err(e)?);
        Ok(ts)
    }

    /// Save the changes to `path` (written whole, then renamed).
    fn save(&self, m: &Model, path: &Path) -> Result<(), String> {
        let ts = self.tensors(m)?;
        let tmp = path.with_extension("safetensors.tmp");
        candle_core::safetensors::save(&ts, &tmp).map_err(|e| format!("save {}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path).map_err(|e| format!("save {}: {e}", path.display()))
    }

    /// Read a chat's changes into the brain: Ok(None) when they are back exactly, Ok(Some(why)) when the chat starts
    /// from her base brain instead (no file yet, or changes of another brain or other settings).
    fn load(&mut self, m: &Model, path: &Path) -> Result<Option<String>, String> {
        if !path.is_file() {
            self.reset(m)?;
            return Ok(Some("a new chat: her base brain".into()));
        }
        let ts = candle_core::safetensors::load(path, &Device::Cpu).map_err(|e| format!("{}: {e}", path.display()))?;
        let text = |k: &str| -> Option<String> { ts.get(k).and_then(|t| t.to_vec1::<u8>().ok()).and_then(|b| String::from_utf8(b).ok()) };
        if text("meta.brain").as_deref() != Some(self.brain.as_str()) || text("meta.settings").as_deref() != Some(self.settings.as_str()) {
            self.reset(m)?;
            return Ok(Some("its changes were made by another brain or with other settings: her base brain".into()));
        }
        self.start(m)?;
        let e = |e: candle_core::Error| e.to_string();
        let bad = |k: &str| format!("{}: no {k}", path.display());
        if self.opts.e2 {
            for &i in &self.blocks {
                for s in Site::ALL {
                    let k = format!("e2.{}", s.var_name(i));
                    m.plastic_set_delta(i, s, ts.get(&k).ok_or_else(|| bad(&k))?).map_err(e)?;
                }
            }
        }
        if self.opts.cb2 {
            let tab: Vec<f32> = match ts.get("cb2.delta") {
                Some(t) => t.to_vec1().map_err(e)?,
                None => Vec::new(),
            };
            let sc: f64 = ts.get("cb2.scale").ok_or_else(|| bad("cb2.scale"))?.to_vec1::<f64>().map_err(e)?[0];
            let up: f64 = ts.get("cb2.updates").ok_or_else(|| bad("cb2.updates"))?.to_vec1::<f64>().map_err(e)?[0];
            let gd = m.cb_weight().ok_or("no cerebellum")?.elem_count();
            if !tab.is_empty() && tab.len() != gd {
                return Err(format!("{}: a table of {} values for {gd}", path.display(), tab.len()));
            }
            m.cb_online_with(|o| {
                o.set_raw(0, tab, sc);
                o.updates[0] = up as usize;
            })
            .ok_or("no online table")?;
        }
        self.tokens = ts.get("meta.tokens").and_then(|t| t.to_vec1::<f64>().ok()).map_or(0, |v| v[0] as u64);
        Ok(None)
    }
}

/// v52: set up live learning on a freshly loaded brain (`vm` its weights, `sha` its file's sha256): the sessions
/// started and fast4's fused session step checked. Err(why): she talks without learning (the sessions are ended).
fn learn_setup(m: &Model, cfg: &Config, vm: &VarMap, dev: &Device, opts: LearnOpts, sha: &str) -> Result<Learn, String> {
    if !dev.is_cpu() {
        return Err("live learning runs on the CPU only (fast4)".into());
    }
    let rate = if opts.cb2 { (CB2_ETA * (1.0 / emb_sq(vm)?)) as f32 } else { 0.0 };
    let units = m.cb_weight().map_or(0, |w| w.dims()[0]);
    let blocks = vec![cfg.layers.saturating_sub(2), cfg.layers.saturating_sub(1)];
    let mut settings = String::new();
    if opts.e2 {
        settings += &format!("e2: e-prop, blocks {blocks:?}, W_in W_out W1 W2, lr {E2_LR}, half-life {E2_HALF}; ");
    }
    if opts.cb2 {
        settings += &format!("cb2: cerebellum online table, {units} units, eta {CB2_ETA}, rate {rate:e}, half-life {CB2_HALF}");
    }
    let l = Learn { opts, blocks, rate, brain: sha.to_string(), settings, loaded: None, epoch: 0, tokens: 0, log: None, refused: None };
    m.set_fast4(true);
    let r = l.start(m).and_then(|_| match m.fast4_live_blocker() {
        Some(why) => Err(format!("fast4 cannot take it: {why}")),
        None => Ok(()),
    });
    match r {
        Ok(()) => Ok(l),
        Err(why) => {
            Learn::end(m);
            Err(why)
        }
    }
}

struct Shared {
    clients: Mutex<Vec<TcpStream>>,
    /// v42: behind a lock, since a hot swap changes the brain it describes
    hello: Mutex<String>,
    busy: AtomicBool,
    stop: AtomicBool,
    port: u16,
    /// v43 --library: her library
    lib: Option<ava_gpu::lookup::Library>,
    /// --chats: where saved chats live
    chats: Option<PathBuf>,
    /// v52: chats deleted so far (the live-learning session reads its changes again after a delete)
    chats_epoch: AtomicU64,
}

/// v42: the brain that is serving, with what the diary says about it.
struct Live {
    model: Model,
    cfg: Config,
    path: PathBuf,
    /// sha256 of the file (with --diary or --learn)
    sha: String,
    /// the learner's generation (with --learner-root)
    gen: Option<u64>,
    _varmap: VarMap,
    /// v52: the live-learning session (None: she does not learn)
    learn: Option<Learn>,
    /// v52: why she does not learn although --learn asked for it
    learn_off: Option<String>,
}

/// Load a brain for serving: one of Ava's kinds (with `vocab`, when given) that passes hotswap::check_brain. v52
/// `learn`: the live-learning session (the lean brain, as v51_online's stream loads it; for CB2 a brain without a
/// cerebellum head gets a zero one, v51_online's `plus`: the same logits); where it cannot run, `learn_off` says why.
#[allow(clippy::too_many_arguments)]
fn load_live(path: &Path, kind: Kind, vocab: Option<usize>, dev: &Device, want_sha: bool, gen: Option<u64>, int8: bool, learn: LearnOpts) -> Result<Live, String> {
    let mut learn_off = None;
    let (cfg, varmap, model) = if !learn.any() {
        ava_gpu::ckpt::load_auto(path, kind, 4096, dev)?
    } else {
        let (c, v, m) = ava_gpu::ckpt::load_auto_lean(path, kind, 4096, dev)?;
        if learn.cb2 && c.kind.is_ava() && !m.part_flags()[1] {
            drop((m, v));
            match load_plus_cb(path, CB2_UNITS, dev) {
                Ok(x) => x,
                Err(why) => {
                    learn_off = Some(format!("cb2 needs a cerebellum head and this brain cannot take one: {why}"));
                    ava_gpu::ckpt::load_auto_lean(path, kind, 4096, dev)?
                }
            }
        } else {
            (c, v, m)
        }
    };
    if !cfg.kind.is_ava() {
        return Err("the live viewer needs one of Ava's kinds (T has no spikes)".into());
    }
    if vocab.is_some() {
        // a swapped-in brain is checked before it serves (the first one keeps the old start-up behaviour)
        hotswap::check_brain(&model, vocab)?;
    }
    model.set_track_activity(false);
    // v51: keep the parts' signals of each token for the anatomy map (read on the side, never fed back)
    model.set_part_signals(true);
    let sha = if want_sha || learn.any() { diary::model_sha256(path)? } else { String::new() };
    let mut session = None;
    if learn.any() && learn_off.is_none() {
        if int8 && !learn.e2 {
            // CB2 alone: the table learns in f32 beside the int8 weights (the fused step is the int8 State path's bits)
            model.set_int8(true).map_err(|e| format!("--int8: {e}"))?;
        }
        match learn_setup(&model, &cfg, &varmap, dev, learn, &sha) {
            Ok(l) => {
                model.set_fused(true);
                session = Some(l);
            }
            Err(why) => learn_off = Some(why),
        }
    }
    if int8 && session.is_none() && !model.int8() {
        // E10: the int8 copies, built once per brain from its weights as loaded
        model.set_int8(true).map_err(|e| format!("--int8: {e}"))?;
    }
    if int8 || session.is_some() {
        // the fused one-token step (src/model/fstep.rs): the same bits, faster; it stays out by itself where it does not apply
        model.set_fused(true);
    } else if !dev.is_cuda() {
        // v52: f32 without learning talks through fast4's fused step too (the State path's bits, runs/v51 Disclosure D-3;
        // ~220 vs ~75 tok/s at d384, runs/v52/chat_learn.md), so no talking runs on the slow path
        model.set_fast4(true);
        model.set_fused(true);
    }
    Ok(Live { model, cfg, path: path.to_path_buf(), sha, gen, _varmap: varmap, learn: session, learn_off })
}

/// The "hello" event: the serving brain's shape and the server's settings.
fn hello_of(live: &Live, o: &Opts, dev: &Device) -> Value {
    let cfg = &live.cfg;
    let mut h = json!({
        "type": "hello",
        "model": {
            "name": diary::model_name(&live.path), "title": display_name(o, &live.path), "kind": cfg.kind.name(), "d": cfg.d, "layers": cfg.layers, "hidden": cfg.hidden(), "k": cfg.kwta_k(),
            "heads": cfg.heads(), "engram": cfg.engram, "glance": cfg.glance, "mag": cfg.mag, "params": cfg.weight_counts().0,
        },
        "pace_ms": o.pace_ms, "threads": o.threads, "device": if dev.is_cuda() { "cuda" } else { "cpu" }, "int8": live.model.int8(),
        "tokens": o.tokens, "temperature": o.temperature, "diary": o.diary.is_some(),
    });
    if let Some(g) = live.gen {
        h["model"]["gen"] = json!(g);
    }
    // v51: which brain parts this brain has (the anatomy map lights an area only for a part she has)
    let [th, cb, nm, mem, eng] = live.model.part_flags();
    let cb2 = live.learn.as_ref().is_some_and(|l| l.opts.cb2);
    h["model"]["parts"] = json!({ "thalamus": th, "cerebellum": cb, "nucleus": nm, "memory": mem, "engram": eng, "cb_online": cb && (o.cb_online.is_some() || cb2) });
    // v52: what learns live (e2: e-prop in her last blocks; cb2: the cerebellum's table)
    let lo = live.learn.as_ref().map_or(LearnOpts::default(), |l| l.opts);
    h["learn"] = json!({ "e2": lo.e2, "cb2": lo.cb2 });
    h
}

/// v51: this token's part signals for the anatomy map (TH gates per block, CB's correction and its online share,
/// v50's NE / ACh / DA); None when she has none of these parts.
fn parts_event(model: &Model) -> Option<Value> {
    let [th, cb, nm, ..] = model.part_flags();
    if !(th || cb || nm) {
        return None;
    }
    let s = model.part_signals();
    let mut v = json!({});
    if th {
        v["th"] = json!(s.th.iter().map(|g| json!([r4(g[0]), r4(g[1])])).collect::<Vec<_>>());
    }
    if let Some(x) = s.cb {
        v["cb"] = json!(r4(x));
    }
    if let Some(x) = s.cb_online {
        v["cbo"] = json!(r4(x));
    }
    if let Some(n) = s.nm {
        v["nm"] = json!([r4(n[0]), r4(n[1]), r4(n[2])]);
    }
    Some(v)
}

/// A log line with the local time.
fn note(line: &str) {
    let (date, time) = diary::local_now();
    eprintln!("{date}T{time} ava_live: {line}");
}

impl Shared {
    /// Send one event to every viewer; a viewer whose connection fails is dropped.
    fn broadcast(&self, v: &Value) {
        let msg = format!("data: {v}\n\n");
        let mut cs = self.clients.lock().unwrap_or_else(|e| e.into_inner());
        cs.retain_mut(|c| c.write_all(msg.as_bytes()).and_then(|_| c.flush()).is_ok());
    }
}

struct Opts {
    model: PathBuf,
    port: u16,
    threads: usize,
    pace_ms: u64,
    temperature: f32,
    top_k: usize,
    tokens: usize,
    seed: u64,
    bpe: PathBuf,
    diary: Option<PathBuf>,
    kind: Kind,
    gpu: bool,
    /// v42 (Wave 4): follow the learner's ROOT/CURRENT, read every watch_ms
    learner_root: Option<PathBuf>,
    watch_ms: u64,
    /// E10: talk with int8 weights (CPU only)
    int8: bool,
    /// tokens of earlier talk turns put before each new question (0 = every turn fresh)
    memory: usize,
    /// pace while reading (None = pace_ms)
    read_pace_ms: Option<u64>,
    /// v43: her library (lookup::Library), searched before each talk turn
    library: Option<PathBuf>,
    /// where saved chats live (None = no saving, no sidebar)
    chats: Option<PathBuf>,
    /// v49: the name the page shows (None: the name file beside the model, else "Ava")
    name: Option<String>,
    /// v49: the double-click start (no arguments): the next free port if 8077 is taken, the browser opened
    package: bool,
    /// v49: open the default browser at the page once she is ready
    open: bool,
    /// v51 (--cerebellum-head brains only): her cerebellum learns online from her error on each word she reads, at
    /// rate ETA / mean |E|^2 (the online test's rule); the changes last while this brain serves
    cb_online: Option<f64>,
    /// v52: what learns live while she reads and talks (--learn; off unless asked, on in the double-click start)
    learn: LearnOpts,
}

fn main() {
    let package = std::env::args().len() <= 1;
    if let Err(e) = run() {
        if package {
            eprintln!("{e}");
        } else {
            eprintln!("ava_live: {e}");
        }
        if package {
            // a double-clicked window would close before the message could be read
            eprintln!("\nPress Enter to close this window.");
            let _ = std::io::stdin().read_line(&mut String::new());
        }
        std::process::exit(1);
    }
}

fn parse() -> Result<Opts, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return package_opts();
    }
    parse_args(&args)
}

/// v49: the folder this program is in (where a download keeps her files).
fn exe_dir() -> Result<PathBuf, String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot find where this program is: {e}"))?;
    exe.parent().map(Path::to_path_buf).ok_or_else(|| "cannot find the folder of this program".to_string())
}

/// v49: the double-click start: her files next to the program, the settings she talks with, every file checked first.
fn package_opts() -> Result<Opts, String> {
    let dir = exe_dir()?;
    let mut o = parse_args(&[])?;
    o.model = dir.join("model.safetensors");
    o.bpe = dir.join("bpe.txt");
    let lib = dir.join("library.tsv");
    let mut missing = Vec::new();
    for (p, what) in [(&o.model, "her brain"), (&o.bpe, "her vocabulary")] {
        if !p.is_file() {
            missing.push(format!("  {}  ({what})", p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()));
        }
    }
    if !missing.is_empty() {
        return Err(format!(
            "Ava cannot start: these files are missing from the folder this program is in ({}):\n{}\nKeep every file of the download together in one folder.",
            dir.display(),
            missing.join("\n")
        ));
    }
    if lib.is_file() {
        o.library = Some(lib);
    } else {
        println!("note: library.tsv is not next to this program, so Ava answers without her library.");
    }
    o.chats = Some(dir.join("chats"));
    o.int8 = true;
    o.temperature = 0.0;
    o.threads = std::thread::available_parallelism().map_or(2, |n| (n.get() / 2).clamp(1, 4));
    o.pace_ms = 90;
    o.read_pace_ms = Some(20);
    o.memory = 200;
    o.tokens = 80;
    o.package = true;
    o.open = true;
    // v52: she keeps learning from each chat (each chat its own changes, beside it). Owner choice 11-07: CB2 only, so her
    // frozen weights stay int8 (~350-580 tok/s on a CPU); E2 would make her talk in f32 (~90 tok/s).
    o.learn = LearnOpts { e2: false, cb2: true };
    Ok(o)
}

/// v49: the name the page shows: --name, else NAME.name.txt beside the model file, else name.txt in its folder, else "Ava".
fn display_name(o: &Opts, model: &Path) -> String {
    if let Some(n) = &o.name {
        return n.clone();
    }
    let own = model.with_extension("name.txt");
    let folder = model.parent().map(|p| p.join("name.txt"));
    for p in std::iter::once(own).chain(folder) {
        if let Ok(t) = std::fs::read_to_string(&p) {
            let t: String = t.lines().next().unwrap_or("").trim().chars().take(40).collect();
            if !t.is_empty() {
                return t;
            }
        }
    }
    "Ava".into()
}

/// v49: open the user's default browser at `url` (nothing happens where no opener exists; the console names the page).
fn open_browser(url: &str) {
    #[cfg(windows)]
    let r = std::process::Command::new("rundll32").args(["url.dll,FileProtocolHandler", url]).spawn();
    #[cfg(target_os = "macos")]
    let r = std::process::Command::new("open").arg(url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let r = std::process::Command::new("xdg-open").arg(url).spawn();
    if let Err(e) = r {
        println!("(could not open the browser: {e}; open {url} yourself)");
    }
}

fn parse_args(args: &[String]) -> Result<Opts, String> {
    let mut o = Opts {
        model: PathBuf::from(DEFAULT_MODEL),
        port: 8077,
        threads: 2,
        pace_ms: 120,
        temperature: 0.7,
        top_k: 40,
        tokens: 60,
        seed: 1,
        bpe: PathBuf::from(ava_gpu::DEFAULT_BPE),
        diary: None,
        kind: Kind::D,
        gpu: false,
        learner_root: None,
        watch_ms: 2000,
        int8: false,
        memory: 0,
        read_pace_ms: None,
        chats: None,
        library: None,
        name: None,
        package: false,
        open: false,
        cb_online: None,
        learn: LearnOpts::default(),
    };
    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        match key {
            "-h" | "--help" => return Err(USAGE.into()),
            "--cpu" => o.gpu = false,
            "--gpu" => o.gpu = true,
            "--int8" => o.int8 = true,
            "--open" => o.open = true,
            _ => {
                let val = args.get(i + 1).ok_or_else(|| format!("{key} needs a value\n{USAGE}"))?;
                let num = |v: &str| v.parse::<u64>().map_err(|_| format!("bad integer {v:?}"));
                match key {
                    "--model" => o.model = PathBuf::from(val),
                    "--port" => o.port = u16::try_from(num(val)?).map_err(|_| "bad port".to_string())?,
                    "--threads" => o.threads = num(val)?.max(1) as usize,
                    "--pace-ms" => o.pace_ms = num(val)?,
                    "--temperature" => o.temperature = val.parse().map_err(|_| format!("bad number {val:?}"))?,
                    "--top-k" => o.top_k = num(val)?.max(1) as usize,
                    "--tokens" => o.tokens = num(val)? as usize,
                    "--seed" => o.seed = num(val)?,
                    "--bpe" => o.bpe = PathBuf::from(val),
                    "--diary" => o.diary = Some(PathBuf::from(val)),
                    "--arch" => o.kind = Kind::parse(val)?,
                    "--learner-root" => o.learner_root = Some(PathBuf::from(val)),
                    "--watch-ms" => o.watch_ms = num(val)?.max(50),
                    "--memory" => o.memory = num(val)? as usize,
                    "--read-pace-ms" => o.read_pace_ms = Some(num(val)?),
                    "--chats" => o.chats = Some(PathBuf::from(val)),
                    "--library" => o.library = Some(PathBuf::from(val)),
                    "--name" => o.name = Some(val.chars().take(40).collect()),
                    "--cb-online" => o.cb_online = Some(val.parse().map_err(|_| format!("bad number {val:?}"))?),
                    "--learn" => o.learn = LearnOpts::parse(val)?,
                    _ => return Err(format!("unknown option {key}\n{USAGE}")),
                }
                i += 1;
            }
        }
        i += 1;
    }
    if o.learn.any() && o.cb_online.is_some() {
        return Err("--cb-online (the older rule) and --learn: use one".into());
    }
    Ok(o)
}

/// v52: say on the console whether she learns live, with what, and why not where --learn asked and she cannot.
fn learn_note(live: &Live, o: &Opts) {
    match (&live.learn, &live.learn_off) {
        (Some(_), _) if o.package => {
            println!("She keeps learning as she reads and talks; each chat keeps what she learned in it.");
        }
        (Some(l), _) => {
            println!("learning live ({}): {}; every token through fast4's fused session step; talking in {}", l.opts.name(), l.settings.trim_end_matches("; "), if live.model.int8() { "int8 (the table in f32)" } else { "f32" });
            if o.int8 && !live.model.int8() {
                println!("  (--int8 is off while e-prop learns: its fused step learns in f32 only)");
            }
        }
        (None, Some(why)) => println!("not learning (--learn {}): {why}; she talks without learning", o.learn.name()),
        (None, None) => {}
    }
}

/// Swap a brain that loaded and passed into service (called between replies only) and tell the viewers.
fn swap_in(live: &mut Live, ready: hotswap::Ready<Live>, shared: &Shared, o: &Opts, dev: &Device) {
    let was = format!("{}{}", diary::model_name(&live.path), live.gen.map(|g| format!(", generation {g}")).unwrap_or_default());
    *live = ready.brain; // the old brain is freed here
    let hello = hello_of(live, o, dev);
    *shared.hello.lock().unwrap_or_else(|e| e.into_inner()) = hello.to_string();
    let mut ev = hello;
    ev["busy"] = json!(shared.busy.load(Ordering::SeqCst));
    ev["swapped"] = json!(true);
    shared.broadcast(&ev);
    note(&format!(
        "now serving {} (generation {}, d {}, {} layers; loaded and checked in {:.2}s); was {was}",
        diary::model_name(&live.path),
        ready.cur.gen,
        live.cfg.d,
        live.cfg.layers,
        ready.secs
    ));
    learn_note(live, o);
}

fn run() -> Result<(), String> {
    let o = parse()?;
    if o.package {
        println!("Starting {} (loading her brain and her library takes a few seconds)...", display_name(&o, &o.model));
    }
    diary::set_threads(o.threads); // before the first tensor op
    let dev = ava_gpu::device(!o.gpu).map_err(|e| format!("device: {e}"))?;
    let bpe = Bpe::load(&o.bpe).map_err(|e| format!("bpe {}: {e}", o.bpe.display()))?;
    let want_sha = o.diary.is_some();
    // v42: with --learner-root, start on the brain CURRENT names (checked like a swapped-in one); else, or when it fails,
    // on --model. `served`: CURRENT's text whose brain is live, so the watcher does not load it twice.
    let mut served: Option<String> = None;
    let mut first: Option<Live> = None;
    if let Some(root) = &o.learner_root {
        match hotswap::read_current(root) {
            Err(e) => note(&format!("{e}; starting on --model")),
            Ok((text, None)) => {
                note(&format!("{} names no brain yet; starting on --model", root.join("CURRENT").display()));
                served = Some(text);
            }
            Ok((text, Some(c))) => match load_live(&c.path, o.kind, None, &dev, want_sha, Some(c.gen), o.int8, o.learn).and_then(|l| hotswap::check_brain(&l.model, None).map(|_| l)) {
                Ok(l) => {
                    note(&format!("starting on the learner's brain {} ({})", c.gen, c.path.display()));
                    served = Some(text);
                    first = Some(l);
                }
                Err(why) => note(&format!("the learner's brain {} ({}) failed: {why}; starting on --model", c.gen, c.path.display())),
            },
        }
    }
    let mut live = match first {
        Some(l) => l,
        None => load_live(&o.model, o.kind, None, &dev, want_sha, None, o.int8, o.learn)?,
    };
    let hello = hello_of(&live, &o, &dev).to_string();
    let (name, cfg) = (diary::model_name(&live.path), live.cfg);

    let lib = match &o.library {
        Some(p) => {
            let t = Instant::now();
            let l = ava_gpu::lookup::Library::load(p)?;
            if o.package {
                println!("library: {} passages ({:.1}s)", l.len(), t.elapsed().as_secs_f64());
            } else {
                println!("library: {} passages from {} in {:.1}s", l.len(), p.display(), t.elapsed().as_secs_f64());
            }
            Some(l)
        }
        None => None,
    };
    // 127.0.0.1 only: the viewer is for this machine. The double-click start takes the next free port if 8077 is taken.
    let tries = if o.package { 20 } else { 1 };
    let (listener, port) = (0..tries)
        .find_map(|k| o.port.checked_add(k).and_then(|p| TcpListener::bind(("127.0.0.1", p)).ok().map(|l| (l, p))))
        .ok_or_else(|| if tries > 1 { format!("ports {} to {} on 127.0.0.1 are all taken; close another program using them, or start with --port N", o.port, o.port + tries - 1) } else { format!("bind 127.0.0.1:{}: the port is taken (try --port N)", o.port) })?;
    let shared = Arc::new(Shared { clients: Mutex::new(Vec::new()), hello: Mutex::new(hello), busy: AtomicBool::new(false), stop: AtomicBool::new(false), port, chats: o.chats.clone(), chats_epoch: AtomicU64::new(0), lib });
    let (tx, rx) = mpsc::channel::<Job>();
    {
        let shared = shared.clone();
        std::thread::spawn(move || {
            for conn in listener.incoming().flatten() {
                let (shared, tx) = (shared.clone(), tx.clone());
                std::thread::spawn(move || {
                    let _ = handle(conn, &shared, &tx);
                });
            }
        });
    }
    let url = format!("http://127.0.0.1:{port}/");
    if o.package {
        // the double-click start: only what a person needs to read
        println!(
            "\n{} is ready. Your browser should open by itself; if it does not, open {url}\nShe runs entirely on this computer. Close this window to stop her.\n",
            display_name(&o, &live.path)
        );
    } else {
        println!(
            "Ava is live at {url}  ({}, {} {}, d {}, {} layers; {} threads; pace {} ms)",
            name,
            cfg.kind.name(),
            if dev.is_cuda() { "on cuda" } else if live.model.int8() { "on cpu, int8" } else { "on cpu" },
            cfg.d,
            cfg.layers,
            o.threads,
            o.pace_ms
        );
    }
    learn_note(&live, &o);
    if o.open && std::env::var_os("AVA_NO_BROWSER").is_none() {
        open_browser(&url);
    }

    // the brain: idle (blocked, no events) until a turn arrives
    let mut rng = SplitMix64::new(o.seed);
    let mut turn_no = 0u64;
    // her earlier talk turns of this run (with --memory), oldest first
    let mut history: Vec<(String, String)> = Vec::new();
    let mut serve = |live: &mut Live, job: Job| {
        turn_no += 1;
        serve_turn(live, &bpe, &o, &shared, job, &mut history, turn_no, &mut rng, &dev);
    };
    let Some(root) = o.learner_root.clone() else {
        while let Ok(job) = rx.recv() {
            serve(&mut live, job);
        }
        return Ok(());
    };
    // v42 hot swap: a watcher thread loads and checks each new brain CURRENT names; it goes live here, between replies
    let vocab = live.cfg.vocab;
    let (kind, ldev, int8) = (o.kind, dev.clone(), o.int8);
    let watch = WatchOpts { poll: Duration::from_millis(o.watch_ms), ..WatchOpts::default() };
    let lopts = o.learn;
    let watcher: Watcher<Live> = Watcher::spawn(root.clone(), served, watch, move |c: &Current| load_live(&c.path, kind, Some(vocab), &ldev, want_sha, Some(c.gen), int8, lopts), note);
    note(&format!("following {} (read every {} ms)", root.join("CURRENT").display(), o.watch_ms));
    let idle_check = Duration::from_millis(o.watch_ms.min(250));
    loop {
        if let Some(ready) = watcher.take() {
            swap_in(&mut live, ready, &shared, &o, &dev);
        }
        match rx.recv_timeout(idle_check) {
            Ok(job) => {
                // a brain that became ready while this request waited goes live first: before the reply, never during it
                if let Some(ready) = watcher.take() {
                    swap_in(&mut live, ready, &shared, &o, &dev);
                }
                serve(&mut live, job);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    Ok(())
}

/// One request: her turn, then the chat saved (talk mode). v52: with live learning, the chat's own changes are in
/// her brain before she reads (a new chat: her base brain) and are saved beside the chat after the turn.
#[allow(clippy::too_many_arguments)]
fn serve_turn(live: &mut Live, bpe: &Bpe, o: &Opts, shared: &Shared, job: Job, history: &mut Vec<(String, String)>, turn_no: u64, rng: &mut SplitMix64, dev: &Device) {
    shared.stop.store(false, Ordering::SeqCst);
    // a saved chat brings back its own earlier turns; without one, this run's turns
    let saved = match (&o.chats, &job.chat) {
        (Some(dir), Some(id)) => Some((dir.clone(), id.clone())),
        _ => None,
    };
    let hist = match &saved {
        Some((dir, id)) => chat_history(dir, id),
        None => history.clone(),
    };
    if let Err(e) = learn_before(live, shared, saved.as_ref()) {
        let e = format!("her learned changes of this chat could not be read ({e}); she starts from her base brain for it");
        eprintln!("turn {turn_no}: {e}");
        shared.broadcast(&json!({ "type": "error", "message": e }));
    }
    let mut r = turn(live, bpe, o, shared, &job, &hist, turn_no, rng);
    // v52: a token fast4 refused ends the learning (never the State path): she answers this turn again without it
    if let Some(why) = live.learn.as_ref().and_then(|l| l.refused.clone()) {
        Learn::end(&live.model);
        live.learn = None;
        live.learn_off = Some(why.clone());
        note(&format!("learning stopped: {why}; she talks on without learning"));
        let hello = hello_of(live, o, dev);
        *shared.hello.lock().unwrap_or_else(|e| e.into_inner()) = hello.to_string();
        shared.broadcast(&hello);
        r = turn(live, bpe, o, shared, &job, &hist, turn_no, rng);
    }
    match r {
        Ok(reply) => {
            if let (Some((dir, id)), Mode::Talk) = (&saved, job.mode) {
                if let Err(e) = append_chat(dir, id, &job.text, &reply) {
                    eprintln!("turn {turn_no}: {e}");
                    shared.broadcast(&json!({ "type": "error", "message": e }));
                }
                shared.broadcast(&json!({ "type": "saved", "chat": id }));
            } else if o.memory > 0 && job.mode == Mode::Talk && !reply.trim().is_empty() {
                history.push((job.text.clone(), reply));
                if history.len() > 64 {
                    history.remove(0);
                }
            }
            if let Err(e) = learn_after(live, saved.as_ref()) {
                eprintln!("turn {turn_no}: {e}");
                shared.broadcast(&json!({ "type": "error", "message": e }));
            }
        }
        Err(e) => {
            eprintln!("turn {turn_no}: {e}");
            shared.broadcast(&json!({ "type": "error", "message": e }));
            if let Some(l) = live.learn.as_mut() {
                l.loaded = None; // a turn that failed keeps no changes: the chat's saved ones are read again
            }
        }
    }
    shared.busy.store(false, Ordering::SeqCst);
    shared.broadcast(&json!({ "type": "idle" }));
}

/// v52: before a turn, her brain holds the changes of the chat it belongs to: read from beside the chat when it is
/// another chat than the last turn's (or a chat was deleted since); a new chat (no file) starts from her base brain.
/// Without a chat, this run's own changes go on.
fn learn_before(live: &mut Live, shared: &Shared, saved: Option<&(PathBuf, String)>) -> Result<(), String> {
    let Live { model, learn, .. } = live;
    let Some(l) = learn.as_mut() else { return Ok(()) };
    let want = saved.map(|(_, id)| id.clone());
    let epoch = shared.chats_epoch.load(Ordering::SeqCst);
    if l.loaded.as_ref() == Some(&want) && (want.is_none() || l.epoch == epoch) {
        return Ok(());
    }
    l.loaded = Some(want.clone());
    l.epoch = epoch;
    let r = match saved {
        Some((dir, id)) => l.load(model, &learn_path(dir, id)).map(|why| {
            if let Some(why) = why.filter(|w| !w.starts_with("a new chat")) {
                note(&format!("chat {id}: {why}"));
            }
        }),
        None => l.reset(model),
    };
    if r.is_err() {
        l.reset(model)?;
    }
    r
}

/// v52: after a turn, the chat's changes are saved beside it (once the chat itself is saved).
fn learn_after(live: &Live, saved: Option<&(PathBuf, String)>) -> Result<(), String> {
    let (Some(l), Some((dir, id))) = (&live.learn, saved) else { return Ok(()) };
    if l.refused.is_some() || l.loaded.as_ref() != Some(&Some(id.clone())) || !chat_path(dir, id).is_file() {
        return Ok(());
    }
    l.save(&live.model, &learn_path(dir, id))
}

fn tok_text(bpe: &Bpe, id: u32) -> String {
    if id == EOS {
        "\u{b6}".into()
    } else if is_mark(id) {
        let l = id - MARK_BASE + 1;
        if l >= 16 { "\u{27e8}16+\u{27e9}".into() } else { format!("\u{27e8}{l}\u{27e9}") }
    } else {
        bpe.decode(&[id])
    }
}

fn tok_kind(id: u32) -> u8 {
    if id == EOS {
        1
    } else if is_mark(id) {
        2
    } else {
        0
    }
}

/// v52: four significant digits (for the learning steps' sizes, which are small)
fn sig4(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let m = 10f64.powi(3 - x.abs().log10().floor() as i32);
    (x * m).round() / m
}

fn r4(x: f32) -> f64 {
    ((x as f64) * 1e4).round() / 1e4
}

fn host(t: &Tensor) -> Result<Vec<f32>, String> {
    t.to_dtype(DType::F32).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>()).map_err(|e| format!("to host: {e}"))
}

/// One token's event: everything the viewer draws for it, from the trace and the logits of this very step. `prev`:
/// the previous step's next-token probabilities in this turn (replaced by this step's), for "chg".
fn token_event(bpe: &Bpe, phase: &str, pos: usize, id: u32, tr: &StepTrace, logits: &Tensor, prev: &mut Option<Vec<f32>>) -> Result<Value, String> {
    let mut layers = Vec::with_capacity(tr.layers.len());
    let mut eng = Value::Null;
    for (li, l) in tr.layers.iter().enumerate() {
        // v49: a spike is any value not 0 (int2 spikes are -2..2)
        let s: Vec<u32> = host(&l.spikes)?.iter().enumerate().filter(|(_, &v)| v.abs() > 0.5).map(|(j, _)| j as u32).collect();
        let wm = host(&l.winners)?;
        let hid = host(&l.hidden)?;
        let w: Vec<u32> = wm.iter().enumerate().filter(|(_, &v)| v > 0.5).map(|(j, _)| j as u32).collect();
        let amax = w.iter().map(|&j| hid[j as usize]).fold(0f32, f32::max);
        let a: String = w
            .iter()
            .map(|&j| {
                let v = hid[j as usize];
                let q = if v > 0.0 && amax > 0.0 { (1.0 + (8.0 * v / amax).round()).min(9.0) as u8 } else { 0 };
                (b'0' + q) as char
            })
            .collect();
        // v49: "h", the sum of the MLP units this token (what its winners pass on to W2)
        let hsum: f32 = w.iter().map(|&j| hid[j as usize]).sum();
        let mut o = json!({ "s": s, "w": w, "a": a, "h": r4(hsum) });
        for (key, t) in [("mw", &l.mem_write), ("mr", &l.mem_read), ("mk", &l.mem_keep), ("mb", &l.mem_gate)] {
            if let Some(t) = t {
                o[key] = json!(host(t)?.iter().map(|&x| r4(x)).collect::<Vec<_>>());
            }
        }
        layers.push(o);
        if let (Some(at), Some(rd)) = (&l.engram_attn, &l.engram_read) {
            let (nh, seen) = at.dims2().map_err(|e| e.to_string())?;
            let at = host(at)?;
            let mut mean = vec![0f32; seen];
            for h in 0..nh {
                for (j, m) in mean.iter_mut().enumerate() {
                    *m += at[h * seen + j] / nh as f32;
                }
            }
            let mut earlier: Vec<(usize, f32)> = mean[..seen - 1].iter().copied().enumerate().collect();
            earlier.sort_by(|x, y| y.1.total_cmp(&x.1));
            earlier.truncate(5);
            let rd = host(rd)?;
            let rms = (rd.iter().map(|v| v * v).sum::<f32>() / rd.len().max(1) as f32).sqrt();
            eng = json!({ "layer": li, "top": earlier.iter().map(|&(j, w)| json!([j, r4(w)])).collect::<Vec<_>>(), "self": r4(mean[seen - 1]), "rms": r4(rms) });
        }
    }
    let row = host(logits)?;
    let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let z: f64 = row.iter().map(|&x| ((x - mx) as f64).exp()).sum();
    let mut idx: Vec<usize> = (0..row.len()).collect();
    idx.select_nth_unstable_by(4.min(row.len() - 1), |&x, &y| row[y].total_cmp(&row[x]));
    idx.truncate(5);
    idx.sort_by(|&x, &y| row[y].total_cmp(&row[x]));
    let p: Vec<Value> = idx.iter().map(|&j| json!([tok_text(bpe, j as u32), r4((((row[j] - mx) as f64).exp() / z) as f32), j])).collect();
    // v49: how much her next-token guess moved since the previous token: total variation 0.5 sum |p - p_prev|, 0..1
    let probs: Vec<f32> = row.iter().map(|&x| (((x - mx) as f64).exp() / z) as f32).collect();
    let chg = prev.as_ref().filter(|q| q.len() == probs.len()).map(|q| 0.5 * q.iter().zip(&probs).map(|(a, b)| (a - b).abs() as f64).sum::<f64>());
    *prev = Some(probs);
    Ok(json!({
        "type": "tok", "phase": phase, "i": pos, "id": id, "text": tok_text(bpe, id), "kind": tok_kind(id),
        "place": if tr.place < 16 { json!(tr.place) } else { Value::Null },
        "mag": tr.mag.map(|m| json!(((m * 100.0).round()) / 100.0)).unwrap_or(Value::Null),
        "L": layers, "e": eng, "p": p,
        "io": { "in": r4(tr.input_rms), "chg": chg.map(|c| r4(c as f32)) },
    }))
}

/// One turn; returns her reply (cut at a new question in talk mode).
#[allow(clippy::too_many_arguments)]
fn turn(
    live: &mut Live,
    bpe: &Bpe,
    o: &Opts,
    shared: &Shared,
    job: &Job,
    history: &[(String, String)],
    turn_no: u64,
    rng: &mut SplitMix64,
) -> Result<String, String> {
    let (model, glance): (&Model, bool) = (&live.model, live.cfg.glance);
    let e = |e: candle_core::Error| e.to_string();
    // v43: with a library, the question first cues the passages that best match it (her context)
    let (ctx, titles): (String, Vec<String>) = match (&shared.lib, job.mode) {
        (Some(l), Mode::Talk) => {
            let hits = l.search(&job.text, 2);
            (hits.iter().map(|(_, t, x)| format!("{t}: {x}")).collect::<Vec<_>>().join(" "), hits.iter().map(|h| h.1.to_string()).collect())
        }
        _ => (String::new(), Vec::new()),
    };
    if shared.lib.is_some() && job.mode == Mode::Talk {
        shared.broadcast(&json!({ "type": "lookup", "turn": turn_no, "chat": job.chat, "titles": titles, "context": ctx }));
    }
    let (ids, remembered): (Vec<u32>, usize) = match job.mode {
        Mode::Talk => diary::prompt_ids_ctx(bpe, glance, history, &job.text, &ctx, o.memory),
        Mode::Continue => {
            let mut v = vec![EOS];
            v.extend(diary::encode(bpe, glance, &job.text));
            (v, 0)
        }
    };
    let mode = if job.mode == Mode::Talk { "talk" } else { "continue" };
    shared.broadcast(&json!({ "type": "turn", "turn": turn_no, "mode": mode, "text": job.text, "prompt_tokens": ids.len(), "remembered": remembered, "chat": job.chat }));
    let pace = Duration::from_millis(o.pace_ms);
    let read_pace = Duration::from_millis(o.read_pace_ms.unwrap_or(o.pace_ms));
    let (mut ms_sum, mut steps) = (0f64, 0usize);
    let mut prev_p: Option<Vec<f32>> = None;
    let mut reason = "limit";
    let mut out: Vec<u32> = Vec::new();
    // v42: every drawn token and its log-probs, for the diary's provenance file (pick_traced draws what pick draws)
    let rng_at_start = rng.0;
    let mut trace = diary::Trace::default();
    // v52: tokens learned this turn (None: she does not learn)
    let mut learned: Option<usize> = None;
    if let Some(l) = live.learn.as_mut() {
        // v52 live learning, through fast4's fused session step only: each token learns the word after it, chosen
        // from its own logits (the text's next word while she reads, the word she draws while she speaks); the last
        // token (at the token limit or a stop) learns nothing. A turn is one of v51_online stream's documents: a fresh
        // state, e-prop's traces restarted, the changes carried on.
        if ids.is_empty() {
            return Err("empty prompt".into());
        }
        if l.opts.e2 {
            model.plastic_new_documents().map_err(e)?;
        }
        let mut st = model.state(1).map_err(e)?;
        let (mut cur, mut p, mut n_learned) = (ids[0], 0usize, 0usize);
        let mut said: Option<(String, f32)> = None;
        loop {
            if shared.stop.load(Ordering::SeqCst) {
                reason = "stopped";
                break;
            }
            let phase = if p < ids.len() { "read" } else { "say" };
            let t0 = Instant::now();
            let pos = st.seen[0];
            let mut drawn: Option<diary::Draw> = None;
            let n_out = out.len();
            let step = {
                let mut target = |lg: &[f32]| -> candle_core::Result<Option<u32>> {
                    if p + 1 < ids.len() {
                        return Ok(Some(ids[p + 1]));
                    }
                    if shared.stop.load(Ordering::SeqCst) || n_out >= o.tokens {
                        return Ok(None);
                    }
                    let d = diary::pick_traced(&Tensor::from_slice(lg, lg.len(), &Device::Cpu)?, o.temperature, o.top_k, rng)?;
                    drawn = Some(d);
                    Ok(Some(d.id))
                };
                model.fast4_live_step(&mut st, cur, E2_LR, true, &mut target).map_err(e)?
            };
            let Some(s) = step else {
                let why = model.fast4_live_blocker().unwrap_or("its fused step did not take this token");
                let why = format!("fast4 refused token {p} of turn {turn_no}: {why}");
                l.refused = Some(why.clone());
                return Err(why);
            };
            if let Some(log) = l.log.as_mut() {
                log.push((cur, s.next, s.logits.to_vec1::<f32>().map_err(e)?.iter().map(|x| x.to_bits()).collect()));
            }
            if s.next.is_some() {
                n_learned += 1;
                l.tokens += 1;
            }
            let tr = s.trace.ok_or("no trace of the token")?;
            let mut ev = token_event(bpe, phase, pos, cur, &tr, &s.logits, &mut prev_p)?;
            if let Some(pe) = parts_event(model) {
                ev["parts"] = pe;
            }
            // the size of each learning step (for the brain view): e-prop's change, the cerebellum table's
            ev["learn"] = json!({ "e2": s.eprop_rms.map(sig4), "cb": s.cb.map(|c| sig4(c.1)) });
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            ms_sum += ms;
            steps += 1;
            ev["ms"] = json!((ms * 100.0).round() / 100.0);
            if let Some((reply, conf)) = said.take() {
                ev["reply"] = json!(reply);
                ev["io"]["conf"] = json!(r4(conf));
            }
            shared.broadcast(&ev); // the moment it is computed
            if let Some(rest) = (if phase == "read" { read_pace } else { pace }).checked_sub(t0.elapsed()) {
                std::thread::sleep(rest);
            }
            p += 1;
            if p < ids.len() {
                cur = ids[p];
                continue;
            }
            let Some(d) = drawn else {
                if shared.stop.load(Ordering::SeqCst) {
                    reason = "stopped";
                }
                break;
            };
            trace.push(d);
            if d.id == EOS {
                reason = "eos";
                break;
            }
            out.push(d.id);
            let s = bpe.decode(&strip_marks(&out));
            if job.mode == Mode::Talk && diary::cut_reply(&s).1 {
                reason = "new question";
                break;
            }
            said = Some((s, d.logp.exp()));
            cur = d.id;
        }
        learned = Some(n_learned);
    } else {
        let mut st = model.stream().map_err(e)?;
        // v51 --cb-online: the table starts with the first turn of each brain and lasts while it serves
        let cb_on = match o.cb_online {
            Some(eta) if model.part_flags()[1] => {
                if !model.cb_online_on() {
                    model.cb_online_begin(1, model.cb_rate(eta).map_err(e)?).map_err(e)?;
                }
                true
            }
            _ => false,
        };
        let mut prev_lg: Option<Tensor> = None;
        // `said`: in say phase, her reply so far and the probability (temperature 1) she gave the word she says now
        let mut step = |tok: u32, phase: &str, said: Option<(String, f32)>| -> Result<Tensor, String> {
            let t0 = Instant::now();
            let pos = st.seen;
            // v51 --cb-online: before she reads the next word, her cerebellum learns from her error on it (words she
            // reads only, never her own)
            if let (true, "read", Some(lg)) = (cb_on, phase, prev_lg.take()) {
                model.cb_learn(&lg.reshape((1, ())).map_err(e)?, &[tok]).map_err(e)?;
            }
            let (logits, tr) = model.step_trace(&mut st, tok).map_err(e)?;
            if cb_on {
                prev_lg = Some(logits.clone());
            }
            let mut ev = token_event(bpe, phase, pos, tok, &tr, &logits, &mut prev_p)?;
            if let Some(p) = parts_event(model) {
                ev["parts"] = p;
            }
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            ms_sum += ms;
            steps += 1;
            ev["ms"] = json!((ms * 100.0).round() / 100.0);
            if let Some((reply, conf)) = said {
                ev["reply"] = json!(reply);
                ev["io"]["conf"] = json!(r4(conf));
            }
            shared.broadcast(&ev); // the moment it is computed
            if let Some(rest) = (if phase == "read" { read_pace } else { pace }).checked_sub(t0.elapsed()) {
                std::thread::sleep(rest);
            }
            Ok(logits)
        };
        let mut logits = None;
        for &t in &ids {
            if shared.stop.load(Ordering::SeqCst) {
                reason = "stopped";
                break;
            }
            logits = Some(step(t, "read", None)?);
        }
        if reason != "stopped" {
            let mut next = logits.ok_or("empty prompt")?;
            for _ in 0..o.tokens {
                if shared.stop.load(Ordering::SeqCst) {
                    reason = "stopped";
                    break;
                }
                let draw = diary::pick_traced(&next, o.temperature, o.top_k, rng).map_err(e)?;
                trace.push(draw);
                let tok = draw.id;
                if tok == EOS {
                    reason = "eos";
                    break;
                }
                out.push(tok);
                let said = bpe.decode(&strip_marks(&out));
                if job.mode == Mode::Talk && diary::cut_reply(&said).1 {
                    reason = "new question";
                    break;
                }
                next = step(tok, "say", Some((said, draw.logp.exp())))?;
            }
        }
    }
    let said = bpe.decode(&strip_marks(&out));
    let reply = if job.mode == Mode::Talk { diary::cut_reply(&said).0 } else { said };
    let ms_per_token = ms_sum / steps.max(1) as f64;
    let learn_txt = learned.map(|n| format!(", learned from {n}")).unwrap_or_default();
    println!("turn {turn_no} ({mode}): read {} tokens, said {} ({reason}){learn_txt}; {ms_per_token:.1} ms per token computed ({:.0} tok/s)", ids.len(), out.len(), 1e3 / ms_per_token.max(1e-9));
    shared.broadcast(&json!({
        "type": "done", "turn": turn_no, "reply": reply, "read": ids.len(), "said": out.len(), "reason": reason,
        "ms_per_token": (ms_per_token * 100.0).round() / 100.0, "learned": learned,
    }));
    if let (Some(dir), Mode::Talk) = (&o.diary, job.mode) {
        let (date, time) = diary::local_now();
        let path = diary::day_path(dir, &date);
        let (lines, _) = diary::read_day(&path)?;
        let turn = diary::next_turn(&lines);
        // the brain that wrote this reply (v42: it can change between replies with --learner-root)
        let line = diary::Line { time: time.clone(), model: diary::model_name(&live.path), turn, event: diary::Event::Turn { user: job.text.clone(), ava: reply.clone() } };
        diary::append_line(&path, &line)?;
        let gen = diary::Gen {
            time,
            model: diary::model_name(&live.path),
            turn,
            model_sha256: live.sha.clone(),
            seed: o.seed,
            rng: rng_at_start,
            temperature: o.temperature,
            top_k: o.top_k,
            memory: o.memory,
            remembered,
            max_tokens: o.tokens,
            prompt_ids: ids.clone(),
            sampled_ids: trace.ids,
            logp: trace.logp,
            sample_logp: trace.sample_logp,
        };
        diary::append_gen(&diary::gen_path(dir, &date), &gen)?;
    }
    Ok(reply)
}

// ------------------------------------------------------------------------------------------------ HTTP

fn respond(c: &mut TcpStream, status: &str, ctype: &str, body: &[u8]) -> std::io::Result<()> {
    let head = format!("HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n", body.len());
    c.write_all(head.as_bytes())?;
    c.write_all(body)?;
    c.flush()
}

/// Only this machine's own names for the server are allowed as Host.
fn host_ok(host: Option<&str>, port: u16) -> bool {
    match host {
        Some(h) => h == format!("127.0.0.1:{port}") || h == format!("localhost:{port}"),
        None => false,
    }
}

#[cfg(test)]
mod host_tests {
    use super::host_ok;

    #[test]
    fn only_our_own_host_names_pass() {
        assert!(host_ok(Some("127.0.0.1:8077"), 8077));
        assert!(host_ok(Some("localhost:8077"), 8077));
        assert!(!host_ok(Some("evil.example:8077"), 8077));
        assert!(!host_ok(Some("127.0.0.1:9999"), 8077));
        assert!(!host_ok(Some("127.0.0.1"), 8077));
        assert!(!host_ok(None, 8077));
    }

    #[test]
    fn without_a_learner_root_nothing_new_runs() {
        let args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let o = super::parse_args(&args(&["--model", "m.safetensors"])).unwrap();
        assert!(o.learner_root.is_none());
        assert_eq!(o.watch_ms, 2000);
        let o = super::parse_args(&args(&["--learner-root", "learner", "--watch-ms", "10"])).unwrap();
        assert_eq!(o.learner_root.as_deref(), Some(std::path::Path::new("learner")));
        assert_eq!(o.watch_ms, 50, "the watch period has a floor");
    }

    #[test]
    fn memory_and_read_pace_are_off_unless_asked() {
        let args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let o = super::parse_args(&args(&[])).unwrap();
        assert_eq!((o.memory, o.read_pace_ms), (0, None), "every turn fresh, reading at speaking pace, as before");
        let o = super::parse_args(&args(&["--memory", "200", "--read-pace-ms", "5"])).unwrap();
        assert_eq!((o.memory, o.read_pace_ms), (200, Some(5)));
    }

    #[test]
    fn a_saved_chat_comes_back_with_its_turns() {
        assert!(super::chat_id_ok("c-20261102-0012-ab3f"));
        for bad in ["", "../x", "a/b", "a.json", "a b", &"x".repeat(49)] {
            assert!(!super::chat_id_ok(bad), "{bad:?}");
        }
        let dir = std::env::temp_dir().join(format!("ava_chats_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        super::append_chat(&dir, "c1", "hey ava", "hello").unwrap();
        super::append_chat(&dir, "c1", "what are you?", "").unwrap();
        super::append_chat(&dir, "c2", "second chat", "ok").unwrap();
        let c = super::read_chat(&dir, "c1").unwrap();
        assert_eq!(c["title"], "hey ava");
        assert_eq!(c["turns"].as_array().unwrap().len(), 2, "every turn is kept, an empty answer too");
        assert_eq!(super::chat_history(&dir, "c1"), vec![("hey ava".to_string(), "hello".to_string())], "her memory skips empty answers");
        let list = super::list_chats(&dir);
        assert_eq!(list.as_array().unwrap().len(), 2);
        assert!(super::chat_history(&dir, "nope").is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_page_shows_a_name_not_a_path() {
        let dir = std::env::temp_dir().join(format!("ava_name_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model.safetensors");
        let args = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        let o = super::parse_args(&args(&[])).unwrap();
        assert!(!o.package && !o.open, "flags: the developer's start, as before");
        assert_eq!(super::display_name(&o, &model), "Ava");
        std::fs::write(dir.join("name.txt"), "Ava 0.4\n").unwrap();
        assert_eq!(super::display_name(&o, &model), "Ava 0.4");
        std::fs::write(dir.join("model.name.txt"), "Ava test\n").unwrap();
        assert_eq!(super::display_name(&o, &model), "Ava test", "the model's own name file first");
        let o = super::parse_args(&args(&["--name", "Ava X"])).unwrap();
        assert_eq!(super::display_name(&o, &model), "Ava X");
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(!super::CHAT_PAGE.contains("on this computer ·"), "no paths or settings on the page");
    }

    #[test]
    fn the_chat_page_and_its_brain_are_built_in() {
        assert!(super::CHAT_PAGE.starts_with("<!-- TESTING / DESIGN ONLY"));
        assert!(super::CHAT_PAGE.contains("<script src=\"brain_mesh.js\">"));
        assert!(super::BRAIN_MESH.contains("BodyParts3D") && super::BRAIN_MESH.len() > 1_000_000);
    }
}

// v52: live learning through the app's own turn path (`serve_turn`), checked against v51_online stream --fused's own
// calls on the same tokens (its `load_plus`, `stream` set-up and `read_round` for one row, seg 1): the same logits and
// the same kept changes, bit for bit; the changes saved with each chat, read back when it is reopened, and deleted with
// it. Run with RAYON_NUM_THREADS=1 (tests never change the thread settings: src/thread_env.rs).
#[cfg(test)]
mod learn_tests {
    use super::*;

    type Log = Vec<(u32, Option<u32>, Vec<u32>)>;

    fn args(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// The answer-trained d384 brain (Ava 0.4b teach3) and the release BPE; None (the test is skipped) without them.
    fn files() -> Option<(PathBuf, PathBuf)> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let model = ["runs/v49/teach3/model.safetensors", "../gpu/runs/v49/teach3/model.safetensors"].iter().map(|r| root.join(r)).find(|p| p.is_file())?;
        let bpe = root.join("../../../release/ava/bpe.txt");
        bpe.is_file().then_some((model, bpe))
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits()).collect()
    }

    /// A tensor's exact values (any of the dtypes a changes file holds).
    fn tbits(t: &Tensor) -> Vec<u64> {
        let t = t.flatten_all().unwrap();
        match t.dtype() {
            DType::F64 => t.to_vec1::<f64>().unwrap().iter().map(|x| x.to_bits()).collect(),
            DType::U8 => t.to_vec1::<u8>().unwrap().iter().map(|&x| x as u64).collect(),
            _ => t.to_vec1::<f32>().unwrap().iter().map(|x| x.to_bits() as u64).collect(),
        }
    }

    fn sorted(ts: &HashMap<String, Tensor>, pre: &[&str]) -> Vec<(String, Vec<u64>)> {
        let mut v: Vec<(String, Vec<u64>)> = ts.iter().filter(|(k, _)| pre.iter().any(|p| k.starts_with(p))).map(|(k, t)| (k.clone(), tbits(t))).collect();
        v.sort();
        v
    }

    /// The chat app as `run` sets it up, without the web server (its events go to no viewer).
    struct App {
        live: Live,
        o: Opts,
        shared: Shared,
        bpe: Bpe,
        history: Vec<(String, String)>,
        rng: SplitMix64,
        dev: Device,
        dir: PathBuf,
        turns: u64,
    }

    impl App {
        fn new(model: &Path, bpe: &Path, learn: &str, dir: &Path) -> App {
            App::with(model, bpe, learn, dir, &[])
        }

        fn with(model: &Path, bpe: &Path, learn: &str, dir: &Path, more: &[&str]) -> App {
            match App::try_with(model, bpe, learn, dir, more) {
                Ok(app) => app,
                Err(why) => panic!("learning is on ({why:?})"),
            }
        }

        /// As `with`, but learning that is asked for and refused is an Err with the refusal (v52 Amendment S1-G: the
        /// speed bench reports it and goes on to its next mode).
        fn try_with(model: &Path, bpe: &Path, learn: &str, dir: &Path, more: &[&str]) -> Result<App, Option<String>> {
            let _ = std::fs::remove_dir_all(dir);
            let mut flags = vec!["--model", model.to_str().unwrap(), "--bpe", bpe.to_str().unwrap(), "--learn", learn, "--temperature", "0", "--pace-ms", "0", "--memory", "200", "--tokens", "24", "--chats", dir.to_str().unwrap()];
            flags.extend_from_slice(more);
            let o = parse_args(&args(&flags)).unwrap();
            let dev = Device::Cpu;
            let live = load_live(&o.model, o.kind, None, &dev, false, None, o.int8, o.learn).unwrap();
            if !(learn == "off" || live.learn.is_some()) {
                return Err(live.learn_off);
            }
            let shared = Shared { clients: Mutex::new(Vec::new()), hello: Mutex::new(String::new()), busy: AtomicBool::new(false), stop: AtomicBool::new(false), port: 0, lib: None, chats: o.chats.clone(), chats_epoch: AtomicU64::new(0) };
            let bpe = Bpe::load(&o.bpe).unwrap();
            let rng = SplitMix64::new(o.seed);
            Ok(App { live, o, shared, bpe, history: Vec::new(), rng, dev, dir: dir.to_path_buf(), turns: 0 })
        }

        /// One talk turn of chat `id` through `serve_turn`: every token it stepped (token, target, the logits' bits).
        fn say(&mut self, id: &str, text: &str) -> Log {
            self.live.learn.as_mut().unwrap().log = Some(Vec::new());
            let before = read_chat(&self.dir, id).map_or(0, |c| c["turns"].as_array().unwrap().len());
            self.turns += 1;
            let job = Job { mode: Mode::Talk, text: text.into(), chat: Some(id.into()) };
            serve_turn(&mut self.live, &self.bpe, &self.o, &self.shared, job, &mut self.history, self.turns, &mut self.rng, &self.dev);
            let l = self.live.learn.as_mut().expect("still learning");
            assert!(l.refused.is_none(), "{:?}", l.refused);
            let log = l.log.take().unwrap();
            let n = log.len();
            assert!(n > 0 && log[..n - 1].iter().all(|x| x.1.is_some()), "only a turn's last token may learn nothing");
            assert_eq!(read_chat(&self.dir, id).unwrap()["turns"].as_array().unwrap().len(), before + 1, "the turn is saved");
            log
        }

        /// The changes saved beside chat `id`.
        fn saved(&self, id: &str) -> HashMap<String, Tensor> {
            candle_core::safetensors::load(learn_path(&self.dir, id), &Device::Cpu).unwrap()
        }
    }

    /// v51_online stream --fused for one row: its model (`load_plus`), its session (`stream`'s begin calls) and its
    /// documents (`read_round`).
    struct V51 {
        m: Model,
        _vm: VarMap,
        e2: bool,
        rate: f32,
        blocks: Vec<usize>,
    }

    impl V51 {
        fn new(model: &Path, e2: bool) -> V51 {
            let dev = Device::Cpu;
            let (cfg, vm, m) = if e2 { ava_gpu::ckpt::load_auto_lean(model, Kind::D, 4096, &dev).unwrap() } else { load_plus_cb(model, CB2_UNITS, &dev).unwrap() };
            m.set_track_activity(false);
            let rate = if e2 { 0.0 } else { (CB2_ETA * (1.0 / emb_sq(&vm).unwrap())) as f32 };
            let r = V51 { m, _vm: vm, e2, rate, blocks: vec![cfg.layers - 2, cfg.layers - 1] };
            r.begin();
            r
        }

        fn begin(&self) {
            let m = &self.m;
            if self.e2 {
                m.plastic_begin(1, &self.blocks, &Site::ALL).unwrap();
                m.plastic_set_half_lives(&[E2_HALF]).unwrap();
            } else {
                m.cb_online_begin(1, self.rate).unwrap();
                m.cb_online_set_lr(0, self.rate);
                m.cb_online_set_half_life(0, CB2_HALF).unwrap();
            }
            m.set_fast4(true);
            if self.e2 {
                m.plastic_probes(true);
                assert_eq!(m.fast4_eprop_blocker(), None);
            } else {
                assert_eq!(m.fused_blocker(), None);
            }
        }

        /// One document: the tokens that learned, each from the word after it; their logits' bits.
        fn read(&self, log: &Log) -> Vec<Vec<u32>> {
            let m = &self.m;
            if self.e2 {
                m.plastic_new_documents().unwrap();
                m.plastic_probes(true);
            }
            let mut st = m.state(1).unwrap();
            let mut out = Vec::new();
            for &(tok, next, _) in log {
                let Some(next) = next else { continue };
                let lg = if self.e2 {
                    m.fast4_eprop_step(&mut st, tok, next, E2_LR).unwrap().expect("the fused session step").0
                } else {
                    let lg = m.fast4_state_step(&mut st, tok).unwrap().expect("the fused step");
                    m.fast4_cb_learn(&lg.flatten_all().unwrap().to_vec1::<f32>().unwrap(), next).unwrap();
                    lg
                };
                out.push(bits(&lg));
            }
            out
        }

        /// The kept changes, named as the app saves them.
        fn kept(&self) -> Vec<(String, Vec<u64>)> {
            let mut ts = HashMap::new();
            if self.e2 {
                for ((i, s), d) in self.m.plastic_deltas().unwrap() {
                    ts.insert(format!("e2.{}", s.var_name(i)), d);
                }
            } else {
                let (tab, sc, up) = self.m.cb_online_with(|o| (o.raw(0).0.to_vec(), o.raw(0).1, o.updates[0])).unwrap();
                if !tab.is_empty() {
                    let n = tab.len();
                    ts.insert("cb2.delta".to_string(), Tensor::from_vec(tab, n, &Device::Cpu).unwrap());
                }
                ts.insert("cb2.scale".to_string(), Tensor::from_vec(vec![sc], 1, &Device::Cpu).unwrap());
                ts.insert("cb2.updates".to_string(), Tensor::from_vec(vec![up as f64], 1, &Device::Cpu).unwrap());
            }
            sorted(&ts, &["e2.", "cb2."])
        }
    }

    /// The app's turn against v51's document: the same logits at every token that learned, and the same changes
    /// kept (as saved beside the chat).
    fn same(app: &App, id: &str, log: &Log, r: &V51, what: &str) {
        let want = r.read(log);
        let got: Vec<&Vec<u32>> = log.iter().filter(|x| x.1.is_some()).map(|x| &x.2).collect();
        assert_eq!(got.len(), want.len(), "{what}");
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(**g == *w, "{what}: token {k}'s logits differ from v51_online stream --fused");
        }
        let kept = sorted(&app.saved(id), &["e2.", "cb2."]);
        assert!(!kept.is_empty());
        assert!(kept == r.kept(), "{what}: the changes saved with the chat differ from v51_online's");
    }

    /// The chat's saved changes read into a freshly loaded brain: the same values (every key), and the same logits and
    /// changes after one more learning step on both.
    fn round_trip(app: &App, model: &Path, id: &str) {
        let mut fresh = load_live(model, app.o.kind, None, &app.dev, false, None, app.o.int8, app.o.learn).unwrap();
        let l = fresh.learn.as_mut().unwrap();
        assert_eq!(l.load(&fresh.model, &learn_path(&app.dir, id)).unwrap(), None, "the changes are read back");
        let a = app.live.learn.as_ref().unwrap();
        let (ta, tb) = (a.tensors(&app.live.model).unwrap(), l.tensors(&fresh.model).unwrap());
        assert!(sorted(&ta, &[""]) == sorted(&tb, &[""]), "the reopened chat holds exactly the changes it was saved with");
        let e2 = a.opts.e2;
        let step = |m: &Model| -> Vec<u32> {
            if e2 {
                m.plastic_new_documents().unwrap();
            }
            let mut st = m.state(1).unwrap();
            let s = m.fast4_live_step(&mut st, 300, E2_LR, false, &mut |_| Ok(Some(301))).unwrap().unwrap();
            bits(&s.logits)
        };
        assert!(step(&app.live.model) == step(&fresh.model), "the same logits after the round trip");
        let (ta, tb) = (a.tensors(&app.live.model).unwrap(), l.tensors(&fresh.model).unwrap());
        assert!(sorted(&ta, &["e2.", "cb2."]) == sorted(&tb, &["e2.", "cb2."]), "and the same changes after one more step");
    }

    fn chat_as_v51(learn: &str, e2: bool) {
        let Some((model, bpe)) = files() else {
            eprintln!("skipped: runs/v49/teach3/model.safetensors or release/ava/bpe.txt is missing");
            return;
        };
        let dir = std::env::temp_dir().join(format!("ava_learn_test_{learn}_{}", std::process::id()));
        let mut app = App::new(&model, &bpe, learn, &dir);
        let (r1, r2) = (V51::new(&model, e2), V51::new(&model, e2));
        let t1 = app.say("c1", "Hello Ava. What is the sun?");
        same(&app, "c1", &t1, &r1, "c1, turn 1");
        app.o.tokens = 3; // this turn stops at the token limit: its last token learns nothing (v51 never steps it)
        let t2 = app.say("c1", "And what is the moon?");
        app.o.tokens = 24;
        assert!(t2.last().unwrap().1.is_none(), "the limit's last token learns nothing");
        same(&app, "c1", &t2, &r1, "c1, turn 2");
        let t3 = app.say("c2", "Tell me about cats.");
        same(&app, "c2", &t3, &r2, "c2 (a new chat: her base brain)");
        let t4 = app.say("c1", "What did I ask you first?");
        same(&app, "c1", &t4, &r1, "c1 reopened (its changes read back from beside it)");
        let learned = [&t1, &t2, &t4].iter().map(|t| t.iter().filter(|x| x.1.is_some()).count()).sum::<usize>();
        assert_eq!(tbits(&app.saved("c1")["meta.tokens"]), vec![(learned as f64).to_bits()]);
        round_trip(&app, &model, "c1");
        // a deleted chat's changes go with it: the chat starts again from her base brain
        drop(r1);
        delete_chat(&app.shared, &dir, "c1");
        assert!(!learn_path(&dir, "c1").exists() && !chat_path(&dir, "c1").exists());
        let r3 = V51::new(&model, e2);
        let t5 = app.say("c1", "Hello again.");
        same(&app, "c1", &t5, &r3, "c1 after its delete");
        println!("{learn}: turns of {} {} {} {} {} tokens, all bit for bit with v51_online stream --fused", t1.len(), t2.len(), t3.len(), t4.len(), t5.len());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// v52: with learning off and no --int8, she talks through fast4's fused f32 step, and says exactly what the State
    /// path says (greedy, three turns with memory).
    #[test]
    fn f32_without_learning_talks_on_the_fused_step_and_says_the_same() {
        let Some((model, bpe)) = files() else { return };
        let tmp = std::env::temp_dir();
        let mut fast = App::new(&model, &bpe, "off", &tmp.join("ava_live_test_f32_fast"));
        assert!(fast.live.model.fast4() && fast.live.model.fused() && !fast.live.model.int8(), "f32 talking is on the fused step");
        let mut slow = App::new(&model, &bpe, "off", &tmp.join("ava_live_test_f32_slow"));
        slow.live.model.set_fast4(false);
        slow.live.model.set_fused(false);
        for q in ["What is the sun?", "Why do birds sing?", "Tell me about rivers."] {
            for a in [&mut fast, &mut slow] {
                a.turns += 1;
                let job = Job { mode: Mode::Talk, text: q.into(), chat: Some("c".into()) };
                serve_turn(&mut a.live, &a.bpe, &a.o, &a.shared, job, &mut a.history, a.turns, &mut a.rng, &a.dev);
            }
        }
        let replies = |a: &App| -> Vec<String> { read_chat(&a.dir, "c").unwrap()["turns"].as_array().unwrap().iter().map(|t| t["ava"].as_str().unwrap().to_string()).collect() };
        let (f, s) = (replies(&fast), replies(&slow));
        assert_eq!(f.len(), 3);
        assert!(f.iter().any(|r| !r.trim().is_empty()), "she said something");
        assert_eq!(f, s, "the fused f32 step says what the State path says");
        for a in [&fast, &slow] {
            let _ = std::fs::remove_dir_all(&a.dir);
        }
    }

    #[test]
    fn e2_in_a_chat_is_v51_online_streams_bit_for_bit() {
        chat_as_v51("e2", true);
    }

    #[test]
    fn cb2_in_a_chat_is_v51_online_streams_bit_for_bit() {
        chat_as_v51("cb2", false);
    }

    /// Both together (v51_online has no such stream): a chat that is left and reopened goes on exactly as one that is
    /// never left, and its changes come back exactly.
    #[test]
    fn e2_and_cb2_in_a_chat_come_back_exactly() {
        let Some((model, bpe)) = files() else {
            eprintln!("skipped: runs/v49/teach3/model.safetensors or release/ava/bpe.txt is missing");
            return;
        };
        let base = std::env::temp_dir().join(format!("ava_learn_test_both_{}", std::process::id()));
        let mut a = App::new(&model, &bpe, "e2,cb2", &base.join("a"));
        let mut b = App::new(&model, &bpe, "e2,cb2", &base.join("b"));
        let qs = ["Hello Ava. What is the sun?", "And what is the moon?", "What did I ask you first?"];
        let a1 = a.say("c1", qs[0]);
        a.o.tokens = 3;
        let a2 = a.say("c1", qs[1]);
        a.o.tokens = 24;
        let _ = a.say("c2", "Tell me about cats.");
        let a3 = a.say("c1", qs[2]);
        let b1 = b.say("c1", qs[0]);
        b.o.tokens = 3;
        let b2 = b.say("c1", qs[1]);
        b.o.tokens = 24;
        let b3 = b.say("c1", qs[2]);
        assert!(a1 == b1 && a2 == b2 && a3 == b3, "the reopened chat goes on as if it had never been left");
        assert!(a.live.learn.as_ref().unwrap().tensors(&a.live.model).unwrap().contains_key("cb2.delta"));
        let (sa, sb) = (sorted(&a.saved("c1"), &["e2.", "cb2."]), sorted(&b.saved("c1"), &["e2.", "cb2."]));
        assert!(sa.len() == 11 && sa == sb, "both learners' changes saved, the same ({} values)", sa.len());
        round_trip(&a, &model, "c1");
        std::fs::remove_dir_all(&base).unwrap();
    }

    /// The speed of a chat, learning off and on (ignored: run with --ignored --nocapture, RAYON_NUM_THREADS=8). Each
    /// mode: one warm-up turn, then three turns; the console's per-turn lines are the app's own speed (ms per token
    /// computed, the viewer's event included), and the save of the changes is timed on its own. AVA_BENCH_MODELS: the
    /// brains (paths, comma-separated; default Ava 0.4b's base and answer-trained brains). A mode whose learning the
    /// brain refuses prints "refused: <mode>: <reason>" and the bench goes on to the next mode (v52 Amendment S1-G).
    #[test]
    #[ignore]
    fn bench_chat_speed_learning_off_and_on() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let Some((_, bpe)) = files() else { return };
        let models: Vec<PathBuf> = match std::env::var("AVA_BENCH_MODELS") {
            Ok(s) => s.split(',').map(PathBuf::from).collect(),
            Err(_) => ["v49/ava04b/grow", "v49/teach3"].iter().map(|m| [root.join("runs"), root.join("../gpu/runs")].iter().map(|r| r.join(m).join("model.safetensors")).find(|p| p.is_file()).unwrap()).collect(),
        };
        let qs = ["Hello Ava. What is the sun?", "And what is the moon made of?", "Tell me about cats and dogs.", "What did I ask you first?"];
        let dir = std::env::temp_dir().join(format!("ava_learn_bench_{}", std::process::id()));
        println!("{} threads", candle_core::utils::get_num_threads());
        for model in &models {
            for (name, learn, more, fast) in [
                ("off, int8 (fused)", "off", &["--int8"][..], false),
                ("off, f32 (as the app runs it without --int8)", "off", &[][..], false),
                ("off, f32 fused (fast4)", "off", &[][..], true),
                ("e2 (f32)", "e2", &[][..], false),
                ("cb2, f32", "cb2", &[][..], false),
                ("cb2, int8", "cb2", &["--int8"][..], false),
                ("e2,cb2 (f32)", "e2,cb2", &[][..], false),
            ] {
                let mut flags = vec!["--tokens", "64"];
                flags.extend_from_slice(more);
                let mut app = match App::try_with(model, &bpe, learn, &dir, &flags) {
                    Ok(app) => app,
                    Err(why) => {
                        println!("refused: {name}: {}", why.unwrap_or_else(|| "no reason given".into()));
                        continue;
                    }
                };
                if fast {
                    app.live.model.set_fast4(true);
                    app.live.model.set_fused(true);
                }
                println!("== {} | {name} | int8 {}", model.display(), app.live.model.int8());
                for q in qs {
                    app.turns += 1;
                    let job = Job { mode: Mode::Talk, text: q.into(), chat: Some("b1".into()) };
                    serve_turn(&mut app.live, &app.bpe, &app.o, &app.shared, job, &mut app.history, app.turns, &mut app.rng, &app.dev);
                }
                if let Some(l) = &app.live.learn {
                    let t = Instant::now();
                    l.save(&app.live.model, &learn_path(&dir, "b1")).unwrap();
                    let mb = std::fs::metadata(learn_path(&dir, "b1")).unwrap().len() as f64 / 1e6;
                    println!("save of the changes: {:.1} ms, {mb:.1} MB", t.elapsed().as_secs_f64() * 1e3);
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn learning_is_asked_for_by_name() {
        let o = parse_args(&args(&[])).unwrap();
        assert_eq!(o.learn, LearnOpts::default(), "off unless asked (the double-click start turns CB2 on)");
        for (s, e2, cb2) in [("e2", true, false), ("cb2", false, true), ("e2,cb2", true, true), ("cb2, e2", true, true), ("off", false, false)] {
            assert_eq!(parse_args(&args(&["--learn", s])).unwrap().learn, LearnOpts { e2, cb2 }, "{s}");
        }
        assert!(parse_args(&args(&["--learn", "e3"])).is_err());
        assert!(parse_args(&args(&["--learn", "cb2", "--cb-online", "0.3"])).is_err(), "one cerebellum learner at a time");
    }
}

fn handle(mut c: TcpStream, shared: &Shared, tx: &mpsc::Sender<Job>) -> std::io::Result<()> {
    c.set_read_timeout(Some(Duration::from_secs(10)))?;
    let mut rd = BufReader::new(c.try_clone()?);
    let mut line = String::new();
    rd.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let (method, path) = (parts.next().unwrap_or("").to_string(), parts.next().unwrap_or("").to_string());
    let (mut len, mut ctype, mut origin) = (0usize, String::new(), None::<String>);
    let mut host = None::<String>;
    let mut header_bytes = 0usize;
    loop {
        let mut h = String::new();
        if rd.read_line(&mut h)? == 0 {
            break;
        }
        header_bytes += h.len();
        if header_bytes > 16 * 1024 {
            return respond(&mut c, "431 Request Header Fields Too Large", "text/plain", b"headers too large");
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((k, v)) = h.split_once(':') {
            match k.trim().to_ascii_lowercase().as_str() {
                "content-length" => len = v.trim().parse().unwrap_or(0),
                "content-type" => ctype = v.trim().to_ascii_lowercase(),
                "origin" => origin = Some(v.trim().to_string()),
                "host" => host = Some(v.trim().to_ascii_lowercase()),
                _ => {}
            }
        }
    }
    // DNS rebinding guard (item G3): a web page on another name that resolves to 127.0.0.1 still sends
    // its own name as Host, so every route, including the page and the live stream, needs our own host.
    if !host_ok(host.as_deref(), shared.port) {
        return respond(&mut c, "403 Forbidden", "text/plain", b"wrong host");
    }
    let path = path.split('?').next().unwrap_or("").to_string();
    match (method.as_str(), path.as_str()) {
        ("GET", "/") | ("GET", "/index.html") => respond(&mut c, "200 OK", "text/html; charset=utf-8", CHAT_PAGE.as_bytes()),
        ("GET", "/brain_mesh.js") => respond(&mut c, "200 OK", "text/javascript; charset=utf-8", BRAIN_MESH.as_bytes()),
        ("GET", "/neurons") => respond(&mut c, "200 OK", "text/html; charset=utf-8", PAGE.as_bytes()),
        // saved chats: list, one chat, delete (only with --chats; delete only from this machine's own page)
        ("GET", "/chats") => match &shared.chats {
            Some(dir) => respond(&mut c, "200 OK", "application/json", list_chats(dir).to_string().as_bytes()),
            None => respond(&mut c, "404 Not Found", "text/plain", b"chats are off"),
        },
        ("GET", p) | ("POST", p) if p.starts_with("/chats/") => {
            let rest = &p["/chats/".len()..];
            let (id, delete) = match rest.strip_suffix("/delete") {
                Some(id) => (id, true),
                None => (rest, false),
            };
            let Some(dir) = shared.chats.as_ref().filter(|_| chat_id_ok(id) && (method == "POST") == delete) else {
                return respond(&mut c, "404 Not Found", "text/plain", b"not found");
            };
            if delete {
                let ok_origin = origin.as_deref().map_or(true, |o| o == format!("http://127.0.0.1:{}", shared.port) || o == format!("http://localhost:{}", shared.port));
                if !ok_origin {
                    return respond(&mut c, "403 Forbidden", "text/plain", b"wrong origin");
                }
                delete_chat(shared, dir, id);
                return respond(&mut c, "200 OK", "application/json", b"{\"ok\":true}");
            }
            match read_chat(dir, id) {
                Some(v) => respond(&mut c, "200 OK", "application/json", v.to_string().as_bytes()),
                None => respond(&mut c, "404 Not Found", "text/plain", b"no such chat"),
            }
        }
        ("GET", "/events") => {
            let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: keep-alive\r\n\r\n";
            c.write_all(head.as_bytes())?;
            let busy = shared.busy.load(Ordering::SeqCst);
            let text = shared.hello.lock().unwrap_or_else(|e| e.into_inner()).clone();
            let mut hello: Value = serde_json::from_str(&text).unwrap_or(Value::Null);
            hello["busy"] = json!(busy);
            c.write_all(format!("data: {hello}\n\n").as_bytes())?;
            c.flush()?;
            c.set_write_timeout(Some(Duration::from_secs(2)))?;
            shared.clients.lock().unwrap_or_else(|e| e.into_inner()).push(c);
            Ok(())
        }
        ("POST", "/say") | ("POST", "/stop") => {
            // only this machine's own page (or a tool without an Origin, like curl) may drive her
            let ok_origin = origin.as_deref().map_or(true, |o| o == format!("http://127.0.0.1:{}", shared.port) || o == format!("http://localhost:{}", shared.port));
            if !ok_origin {
                return respond(&mut c, "403 Forbidden", "text/plain", b"wrong origin");
            }
            if path == "/stop" {
                shared.stop.store(true, Ordering::SeqCst);
                return respond(&mut c, "200 OK", "application/json", b"{\"ok\":true}");
            }
            if !ctype.starts_with("application/json") {
                return respond(&mut c, "415 Unsupported Media Type", "text/plain", b"send application/json");
            }
            if len > 64 * 1024 {
                return respond(&mut c, "413 Payload Too Large", "text/plain", b"too long");
            }
            let mut body = vec![0u8; len];
            rd.read_exact(&mut body)?;
            let v: Value = match serde_json::from_slice(&body) {
                Ok(v) => v,
                Err(_) => return respond(&mut c, "400 Bad Request", "text/plain", b"bad json"),
            };
            let mode = match v.get("mode").and_then(Value::as_str).unwrap_or("talk") {
                "talk" => Mode::Talk,
                "continue" => Mode::Continue,
                _ => return respond(&mut c, "400 Bad Request", "text/plain", b"mode is talk or continue"),
            };
            let text = v.get("text").and_then(Value::as_str).unwrap_or("").trim().to_string();
            if text.is_empty() || text.chars().count() > MAX_TEXT {
                return respond(&mut c, "400 Bad Request", "text/plain", b"text: 1 to 2000 characters");
            }
            let chat = match v.get("chat").and_then(Value::as_str) {
                Some(id) if !chat_id_ok(id) => return respond(&mut c, "400 Bad Request", "text/plain", b"bad chat id"),
                Some(id) if shared.chats.is_some() => Some(id.to_string()),
                _ => None,
            };
            if shared.busy.swap(true, Ordering::SeqCst) {
                return respond(&mut c, "409 Conflict", "application/json", b"{\"ok\":false,\"busy\":true}");
            }
            if tx.send(Job { mode, text, chat }).is_err() {
                shared.busy.store(false, Ordering::SeqCst);
                return respond(&mut c, "503 Service Unavailable", "text/plain", b"brain stopped");
            }
            respond(&mut c, "202 Accepted", "application/json", b"{\"ok\":true}")
        }
        _ => respond(&mut c, "404 Not Found", "text/plain", b"not found"),
    }
}
