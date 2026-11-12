// TESTING / DESIGN ONLY
//! ava_learner: Ava's always-online learner (Wave 4 of v42). It runs beside the live server and never
//! stops her talking: it learns in the background and swaps a checked brain in when one is ready.
//!
//! Usage (ROOT is the learner's private folder; it never goes to GitHub):
//!   ava_learner --root ROOT import BRAIN.safetensors   the first live brain
//!   ava_learner --root ROOT run [options]              the loop; safe to stop at any moment
//!   ava_learner --root ROOT status
//!   ava_learner --root ROOT rollback GEN
//!   ava_learner --root ROOT erase-episode ID
//!   ava_learner --root ROOT erase-person NAME [--before YYYY-MM-DDTHH:MM:SS]
//!   ava_learner --root ROOT teach FILE.tsv [PERSON]    lessons (question<TAB>answer) as taught episodes (J8)
//!   ava_learner --root ROOT reflect [YYYY-MM-DD]       the day's moments worth thinking back on (F8)
//!   ava_learner --root ROOT life                       the facts of her life so far (F11)
//!   ava_learner --root ROOT [--person P] quiz [N]      self-questions, the fading ones first (F14)
//!   ava_learner --root ROOT person NAME                what she knows about talking with NAME (B13)
//!   ava_learner --root ROOT goal add OWNER|- "TEXT" "CUE WORDS" [YYYY-MM-DD|-] [VALUE]
//!                                                      a standing goal ("-" = her own) (F5)
//!   ava_learner --root ROOT goal progress|done|drop ID [0..1]
//!   ava_learner --root ROOT [--person P] goals [WHAT WAS SAID]   open goals by pull, cued ones first
//!   ava_learner --root ROOT care NAME                  how much NAME leans on her, and the care weight (J9)
//!
//! run options:
//!   --diary DIR              the chat's diary folder (default data/diary)
//!   --person NAME            whose turns the diary holds (default owner)
//!   --train-cmd "CMD"        trains a candidate; placeholders {model} {out} {episodes}.
//!                            Without it, a dry run: the candidate is a copy of the live brain.
//!   --gate-cmd "CMD"         scores the live brain and the candidate; placeholders {old} {new} {out}.
//!                            It writes rows `skill<TAB>name<TAB>old<TAB>new` (one per probe item) and
//!                            `self<TAB>index<TAB>old answer<TAB>new answer` (one per self-question).
//!   --no-gates               promote without gates (dry runs only, never on her real brain)
//!   --dream-cmd "CMD"        samples dreams; placeholders {model} {out} (blank-line separated texts)
//!   --surprise-cmd "CMD"     before each rehearsal, measures how much unmeasured episodes surprise the
//!                            live brain; placeholders {model} {episodes} {ids} {out}. It writes rows
//!                            `id<TAB>nats_per_token...` (gpu crate: episode_surprise --ids {ids} --out {out}).
//!   --every SECS             seconds between cycles (default 300)
//!   --once                   one cycle, then exit
//!   --gpu-check              train only while the GPU is free (below 20% busy while the learner is idle)
//!   --pause-if-running LIST  never train while these programs run (comma separated, e.g. game.exe);
//!                            a running job is stopped and thrown away if one starts
//!   --quiet N                quiet readings in a row before training resumes (default 3)
//!   --batch N                replay batch size (default 64)
//!   --text-budget-mb N       episode text kept word for word before gists (default 256)
//!   --migration              allow a shape change (growth) at the next promotion
//!
//! Every step is logged first (registry.log, episodes.log, learner.log), so a stop or a crash at any
//! point loses nothing, and a half-made candidate is never used.

use ava_online::ckpt::Registry;
use ava_online::clock::{format_time, local_now, parse_time};
use ava_online::diary::{day_episodes, one_line, read_turns};
use ava_online::drives::{Activity, Competence, Drives, Rates};
use ava_online::episodes::{first_sentence_gist, Episode, Origin, Store, Tier, TrustPolicy};
use ava_online::gates::{identity_gate, regression_gate, IdentityBudget, SkillCheck};
use ava_online::gpu_share::{running_of, running_programs, sample_util, Pauser};
use ava_online::log::{atomic_write, Log};
use ava_online::care::reliance;
use ava_online::goals::{arbitrate, Goals};
use ava_online::mind;
use ava_online::replay::{need, pick_batch, strength_after, Cand, MixPolicy};
use ava_online::{fields, rec, unesc};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

struct Opts {
    root: PathBuf,
    diary: PathBuf,
    person: String,
    train_cmd: Option<String>,
    gate_cmd: Option<String>,
    dream_cmd: Option<String>,
    surprise_cmd: Option<String>,
    no_gates: bool,
    every: u64,
    once: bool,
    gpu_check: bool,
    watch: Vec<String>,
    quiet: usize,
    batch: usize,
    budget: usize,
    migration: bool,
    before: Option<i64>,
    rest: Vec<String>,
}

fn parse_opts() -> Result<Opts, String> {
    let mut o = Opts {
        root: PathBuf::new(),
        diary: PathBuf::from("data/diary"),
        person: "owner".into(),
        train_cmd: None,
        gate_cmd: None,
        dream_cmd: None,
        surprise_cmd: None,
        no_gates: false,
        every: 300,
        once: false,
        gpu_check: false,
        watch: Vec::new(),
        quiet: 3,
        batch: 64,
        budget: 256 << 20,
        migration: false,
        before: None,
        rest: Vec::new(),
    };
    let mut a = std::env::args().skip(1);
    while let Some(x) = a.next() {
        let mut v = || a.next().ok_or(format!("{x} needs a value"));
        match x.as_str() {
            "--root" => o.root = PathBuf::from(v()?),
            "--diary" => o.diary = PathBuf::from(v()?),
            "--person" => o.person = v()?,
            "--train-cmd" => o.train_cmd = Some(v()?),
            "--gate-cmd" => o.gate_cmd = Some(v()?),
            "--dream-cmd" => o.dream_cmd = Some(v()?),
            "--surprise-cmd" => o.surprise_cmd = Some(v()?),
            "--no-gates" => o.no_gates = true,
            "--every" => o.every = v()?.parse().map_err(|_| "--every needs seconds")?,
            "--once" => o.once = true,
            "--gpu-check" => o.gpu_check = true,
            "--pause-if-running" => o.watch = v()?.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
            "--quiet" => o.quiet = v()?.parse().map_err(|_| "--quiet needs a number")?,
            "--batch" => o.batch = v()?.parse().map_err(|_| "--batch needs a number")?,
            "--text-budget-mb" => o.budget = v()?.parse::<usize>().map_err(|_| "--text-budget-mb needs a number")? << 20,
            "--migration" => o.migration = true,
            "--before" => o.before = Some(parse_time(&v()?).ok_or("--before needs YYYY-MM-DDTHH:MM:SS")?),
            _ if x.starts_with("--") => return Err(format!("unknown option {x}")),
            _ => o.rest.push(x),
        }
    }
    Ok(o)
}

/// Splits a command template into arguments (spaces separate; double quotes group) and fills in
/// `{name}` placeholders in each argument.
fn command_args(template: &str, vals: &[(&str, &str)]) -> Vec<String> {
    let (mut args, mut cur, mut quoted, mut any) = (Vec::new(), String::new(), false, false);
    for c in template.chars() {
        match c {
            '"' => {
                quoted = !quoted;
                any = true;
            }
            c if c.is_whitespace() && !quoted => {
                if any || !cur.is_empty() {
                    args.push(std::mem::take(&mut cur));
                }
                any = false;
            }
            c => cur.push(c),
        }
    }
    if any || !cur.is_empty() {
        args.push(cur);
    }
    args.into_iter().map(|mut s| {
        for (k, v) in vals {
            s = s.replace(&format!("{{{k}}}"), v);
        }
        s
    }).collect()
}

/// Runs a command. While it runs, stops it if a watched program starts. Ok(false) = stopped.
fn run_cmd(args: &[String], watch: &[String]) -> Result<bool, String> {
    let (exe, rest) = args.split_first().ok_or("empty command")?;
    let mut child = Command::new(exe).args(rest).spawn().map_err(|e| format!("{exe}: {e}"))?;
    let mut last_check = Instant::now();
    loop {
        if let Some(s) = child.try_wait().map_err(|e| e.to_string())? {
            return if s.success() { Ok(true) } else { Err(format!("{exe} failed ({s})")) };
        }
        if !watch.is_empty() && last_check.elapsed() >= Duration::from_secs(5) {
            last_check = Instant::now();
            if running_programs().is_some_and(|names| !running_of(&names, watch).is_empty()) {
                let _ = child.kill();
                let _ = child.wait();
                return Ok(false);
            }
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

struct Learner {
    reg: Registry,
    store: Store,
    log: Log,
    drives: Drives,
    comp: Competence,
    anchor: BTreeMap<usize, String>,
    /// (person, day) -> episodes learned into the weights that day.
    learned: BTreeMap<(String, String), usize>,
    /// day file -> size when last read.
    seen: BTreeMap<String, u64>,
    last_tick: i64,
}

impl Learner {
    fn open(root: &Path) -> Result<Learner, String> {
        std::fs::create_dir_all(root.join("work")).map_err(|e| e.to_string())?;
        let reg = Registry::open(root)?;
        let store = Store::open(&root.join("episodes.log"))?;
        let o = Log::open(&root.join("learner.log"))?;
        let mut l = Learner {
            reg,
            store,
            log: o.log,
            drives: Drives::default(),
            comp: Competence::default(),
            anchor: BTreeMap::new(),
            learned: BTreeMap::new(),
            seen: BTreeMap::new(),
            last_tick: 0,
        };
        for r in &o.records {
            let f = fields(r);
            l.apply(&f.iter().map(String::as_str).collect::<Vec<_>>());
        }
        let cut = l.reg.repaired_bytes + l.store.repaired_bytes + o.cut_bytes;
        if cut > 0 {
            eprintln!("ava_learner: repaired {cut} bytes of torn log ends (a crash cut a write short)");
        }
        Ok(l)
    }

    fn apply(&mut self, f: &[&str]) {
        let n = |i: usize| f.get(i).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
        match (f.first().copied(), f.len()) {
            (Some("anchor"), 3) => {
                self.anchor.insert(n(1) as usize, f[2].to_string());
            }
            (Some("skill"), 4) => self.comp.note(f[1], n(2) as i64, n(3)),
            (Some("drives"), 5) => {
                self.last_tick = n(1) as i64;
                self.drives = Drives { memory: n(2), curiosity: n(3), fatigue: n(4) };
            }
            (Some("learned"), 4) => {
                self.learned.insert((f[1].to_string(), f[2].to_string()), n(3) as usize);
            }
            (Some("seen"), 3) => {
                self.seen.insert(f[1].to_string(), n(2) as u64);
            }
            _ => eprintln!("ava_learner: skipped an unknown learner record"),
        }
    }

    fn put(&mut self, f: &[&str]) -> Result<(), String> {
        self.log.append(&rec(f))?;
        self.apply(f);
        Ok(())
    }

    /// Reads new or grown diary days. Returns the number of new episodes.
    fn ingest(&mut self, o: &Opts) -> Result<usize, String> {
        let Ok(rd) = std::fs::read_dir(&o.diary) else { return Ok(0) };
        let mut days: Vec<(String, PathBuf)> = rd
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                let day = name.strip_suffix(".tsv")?.to_string();
                let ok = day.len() == 10 && day.chars().enumerate().all(|(i, c)| if i == 4 || i == 7 { c == '-' } else { c.is_ascii_digit() });
                ok.then(|| (day, e.path()))
            })
            .collect();
        days.sort();
        let mut new = 0;
        for (day, path) in days {
            let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            if self.seen.get(&day) == Some(&size) {
                continue;
            }
            let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let (turns, _bad) = read_turns(&bytes);
            for e in day_episodes(&day, &turns, &o.person) {
                let (id, trust) = (e.id.clone(), e.trust);
                if self.store.add(e)? {
                    new += 1;
                } else {
                    // A rating, correction or reaction that came later changes how far her reply is trusted.
                    self.store.set_trust(&id, trust)?;
                }
            }
            self.put(&["seen", &day, &bytes.len().to_string()])?;
        }
        Ok(new)
    }

    fn today(now: i64) -> String {
        format_time(now)[..10].to_string()
    }

    /// One consolidation: pick a replay batch, train a candidate, gate it, swap it in or not.
    /// L1: measures against the live brain how much each not-yet-measured episode surprises her (at most
    /// 256 a cycle, in id order), so replay priority and "what surprised her" use real values. Needs
    /// --surprise-cmd; without it nothing changes. A blank line inside a text would split it in the
    /// blank-line format, so blank lines are closed up in the copy that is measured.
    fn measure_surprise(&mut self, o: &Opts) -> Result<usize, String> {
        let (Some(cmd), Some(live)) = (&o.surprise_cmd, self.reg.live) else { return Ok(0) };
        let mut ids: Vec<String> = self.store.eps.values().filter(|e| e.surprise <= 0.0 && e.has_text()).map(|e| e.id.clone()).collect();
        ids.sort();
        ids.truncate(256);
        if ids.is_empty() {
            return Ok(0);
        }
        let mut texts: Vec<String> = ids.iter().map(|id| self.store.eps[id].text.lines().filter(|l| !l.trim().is_empty()).collect::<Vec<_>>().join("\n")).collect();
        if texts.len() == 1 {
            texts.push(texts[0].clone());
        }
        let work = o.root.join("work");
        let (ep, idp, out) = (work.join("surprise_in.txt"), work.join("surprise_ids.txt"), work.join("surprise_out.tsv"));
        atomic_write(&ep, (texts.join("\n\n") + "\n").as_bytes())?;
        atomic_write(&idp, (ids.join("\n") + "\n").as_bytes())?;
        let _ = std::fs::remove_file(&out);
        let s = |p: &Path| p.to_string_lossy().to_string();
        let lp = s(&self.reg.path(live));
        if !run_cmd(&command_args(cmd, &[("model", &lp), ("episodes", &s(&ep)), ("ids", &s(&idp)), ("out", &s(&out))]), &o.watch)? {
            return Ok(0);
        }
        let text = std::fs::read_to_string(&out).map_err(|e| format!("the surprise command wrote nothing: {e}"))?;
        let mut n = 0;
        for l in text.lines() {
            let mut f = l.split('\t');
            if let (Some(id), Some(Ok(v))) = (f.next(), f.next().map(str::parse::<f32>)) {
                if self.store.eps.contains_key(id) && v.is_finite() && v > 0.0 {
                    self.store.set_surprise(id, v)?;
                    n += 1;
                }
            }
        }
        Ok(n)
    }

    fn consolidate(&mut self, o: &Opts, now: i64) -> Result<String, String> {
        let live = self.reg.live.ok_or("no live brain yet: run `import` first")?;
        let today = Self::today(now);
        let policy = TrustPolicy::default();
        let mut cands: Vec<Cand> = Vec::new();
        for e in self.store.eps.values() {
            let done_today = *self.learned.get(&(e.user.clone(), today.clone())).unwrap_or(&0);
            let ok = self.store.may_learn(e, &policy, done_today).is_ok() || (e.has_text() && matches!(e.origin, Origin::Dreamt | Origin::Imagined));
            if !ok {
                continue;
            }
            let mut uses = e.uses.clone();
            uses.push(e.t1);
            cands.push(Cand {
                id: e.id.clone(),
                origin: e.origin,
                gain: if e.surprise > 0.0 { e.surprise as f64 } else { 1.0 },
                need: need(&uses, now),
                strength_days: e.strength_days,
                since: (e.last_replay > 0).then(|| now - e.last_replay),
                replays: e.replays,
            });
        }
        let mix = MixPolicy { batch: o.batch, ..MixPolicy::default() };
        let mut per_user: BTreeMap<String, usize> = BTreeMap::new();
        let mut ids: Vec<String> = Vec::new();
        for i in pick_batch(&cands, &mix) {
            let e = &self.store.eps[&cands[i].id];
            let used = per_user.entry(e.user.clone()).or_insert(*self.learned.get(&(e.user.clone(), today.clone())).unwrap_or(&0));
            if !e.user.is_empty() && *used >= policy.per_user_daily_cap {
                continue;
            }
            *used += 1;
            ids.push(e.id.clone());
        }
        if ids.is_empty() {
            return Ok("nothing is due for rehearsal".into());
        }
        // The batch, in the format grow_gpu reads: episodes separated by blank lines (a lone one twice).
        let work = o.root.join("work");
        let mut texts: Vec<&str> = ids.iter().map(|id| self.store.eps[id].text.as_str()).collect();
        if texts.len() == 1 {
            texts.push(texts[0]);
        }
        let batch_path = work.join("batch.txt");
        atomic_write(&batch_path, (texts.join("\n\n") + "\n").as_bytes())?;
        atomic_write(&work.join("batch_ids.txt"), (ids.join("\n") + "\n").as_bytes())?;
        let out = work.join("candidate.safetensors");
        let _ = std::fs::remove_file(&out);
        let live_path = self.reg.path(live);
        let (lp, op, bp) = (live_path.to_string_lossy().to_string(), out.to_string_lossy().to_string(), batch_path.to_string_lossy().to_string());
        match &o.train_cmd {
            Some(t) => {
                if !run_cmd(&command_args(t, &[("model", &lp), ("out", &op), ("episodes", &bp)]), &o.watch)? {
                    return Ok("training stopped: a watched program started; the half-made candidate is thrown away".into());
                }
            }
            None => {
                std::fs::copy(&live_path, &out).map_err(|e| e.to_string())?;
            }
        }
        if !out.exists() {
            return Err("the training command made no candidate".into());
        }
        let gen = self.reg.add_candidate(&out, Some(live), &format!("rehearsed {} episodes", ids.len()), &ids)?;
        let verdict = self.gate(o, live, gen, now)?;
        if let Some(why) = verdict {
            self.reg.reject(gen, &why)?;
            return Ok(format!("brain {gen} rejected: {why}"));
        }
        if let Err(why) = self.reg.promote(gen, o.migration) {
            self.reg.reject(gen, &why)?;
            return Ok(format!("brain {gen} rejected: {why}"));
        }
        for id in &ids {
            let e = &self.store.eps[id];
            let s = strength_after(e.strength_days, 0.5, true);
            let (user, first) = (e.user.clone(), e.tier != Tier::Weights);
            self.store.note_replay(id, now, s)?;
            if first {
                self.store.mark_learned(id)?;
                let k = (user.clone(), today.clone());
                let n = self.learned.get(&k).copied().unwrap_or(0) + 1;
                self.put(&["learned", &user, &today, &n.to_string()])?;
            }
        }
        Ok(format!("brain {gen} is live (rehearsed {} episodes)", ids.len()))
    }

    /// Runs the gates. Ok(None) = passed, Ok(Some(why)) = blocked.
    fn gate(&mut self, o: &Opts, live: u64, gen: u64, now: i64) -> Result<Option<String>, String> {
        let Some(g) = &o.gate_cmd else {
            return Ok(if o.no_gates { None } else { Some("no gate command (use --no-gates only for dry runs)".into()) });
        };
        let out = o.root.join("work").join("gate.tsv");
        let _ = std::fs::remove_file(&out);
        let (a, b, c) = (self.reg.path(live), self.reg.path(gen), out.clone());
        let vals = [("old", a.to_string_lossy().to_string()), ("new", b.to_string_lossy().to_string()), ("out", c.to_string_lossy().to_string())];
        let vals: Vec<(&str, &str)> = vals.iter().map(|(k, v)| (*k, v.as_str())).collect();
        if !run_cmd(&command_args(g, &vals), &o.watch)? {
            return Ok(Some("gate stopped: a watched program started".into()));
        }
        let text = std::fs::read_to_string(&out).map_err(|e| format!("gate output {}: {e}", out.display()))?;
        let mut skills: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
        let mut selfq: BTreeMap<usize, (String, String)> = BTreeMap::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split('\t').collect();
            match (f.first().copied(), f.len()) {
                (Some("skill"), 4) => {
                    let (x, y) = (f[2].parse::<f64>(), f[3].parse::<f64>());
                    if let (Ok(x), Ok(y)) = (x, y) {
                        skills.entry(f[1].to_string()).or_default().push((x, y));
                    }
                }
                (Some("self"), 4) => {
                    if let Ok(i) = f[1].parse::<usize>() {
                        selfq.insert(i, (unesc(f[2]), unesc(f[3])));
                    }
                }
                _ => return Ok(Some(format!("gate output has a bad row: {}", one_line(line)))),
            }
        }
        if skills.is_empty() {
            return Ok(Some("gate output has no skill rows".into()));
        }
        let checks: Vec<SkillCheck> = skills.into_iter().map(|(skill, pairs)| SkillCheck { skill, pairs, margin: 0.02 }).collect();
        let reports = regression_gate(&checks, 1.645);
        let blocked: Vec<String> = reports.iter().filter(|r| r.blocks).map(|r| format!("{} {:+.3} (z {:.1})", r.skill, r.diff, r.z)).collect();
        for r in &reports {
            let score = if blocked.is_empty() { r.new } else { r.old };
            self.put(&["skill", &r.skill, &now.to_string(), &score.to_string()])?;
        }
        if !blocked.is_empty() {
            return Ok(Some(format!("skills dropped: {}", blocked.join(", "))));
        }
        if !selfq.is_empty() {
            if self.anchor.is_empty() {
                for (i, (old, _)) in &selfq {
                    self.put(&["anchor", &i.to_string(), old])?;
                }
            }
            let keys: Vec<usize> = selfq.keys().copied().collect();
            let pick = |m: &dyn Fn(&usize) -> String| keys.iter().map(m).collect::<Vec<String>>();
            let anchor = pick(&|i| self.anchor.get(i).cloned().unwrap_or_default());
            let old = pick(&|i| selfq[i].0.clone());
            let new = pick(&|i| selfq[i].1.clone());
            let r = identity_gate(&anchor, &old, &new, &IdentityBudget::default());
            if let Some(why) = r.blocks {
                return Ok(Some(why));
            }
        }
        Ok(None)
    }

    /// B11: dreams from her own model, kept as dreamt (never facts), rehearsed only within their cap.
    fn dream(&mut self, o: &Opts, now: i64) -> Result<String, String> {
        let (Some(cmd), Some(live)) = (&o.dream_cmd, self.reg.live) else { return Ok("no dream command".into()) };
        let out = o.root.join("work").join("dreams.txt");
        let _ = std::fs::remove_file(&out);
        let (m, p) = (self.reg.path(live).to_string_lossy().to_string(), out.to_string_lossy().to_string());
        if !run_cmd(&command_args(cmd, &[("model", &m), ("out", &p)]), &o.watch)? {
            return Ok("dreaming stopped: a watched program started".into());
        }
        let text = std::fs::read_to_string(&out).unwrap_or_default();
        let mut n = 0;
        for (k, d) in text.split("\n\n").map(str::trim).filter(|d| !d.is_empty()).enumerate() {
            let e = Episode::new(&format!("dream/{}#{k}", format_time(now)), "", Origin::Dreamt, &format!("a dream, {}", format_time(now)), now, now, d, 0.0);
            n += usize::from(self.store.add(e)?);
        }
        Ok(format!("{n} dreams kept (tagged as dreamt)"))
    }
}

fn cmd_run(o: &Opts) -> Result<(), String> {
    let mut l = Learner::open(&o.root)?;
    let rates = Rates { per_episode: 1.0 / o.batch.max(1) as f64, ..Rates::default() };
    let mut pauser = Pauser::new(20.0, o.quiet);
    loop {
        let now = local_now();
        let new = l.ingest(o)?;
        let hours = if l.last_tick > 0 { ((now - l.last_tick) as f64 / 3600.0).max(0.0) } else { 0.0 };
        l.drives.tick(hours, new, &rates);
        let gpu_free = if o.gpu_check || !o.watch.is_empty() {
            let games = if o.watch.is_empty() { Vec::new() } else { running_programs().map(|n| running_of(&n, &o.watch)).unwrap_or_else(|| vec!["(cannot list programs)".into()]) };
            let util = if o.gpu_check { sample_util() } else { Some(0.0) };
            pauser.observe(util, &games)
        } else {
            true
        };
        let fresh = l.store.eps.values().any(|e| e.origin == Origin::Heard && now - e.t0 < 86_400 && e.uses.is_empty());
        let act = l.drives.choose(gpu_free, fresh);
        let what = match act {
            Activity::Consolidate => {
                let m = l.measure_surprise(o)?;
                let s = l.consolidate(o, now)?;
                if m > 0 {
                    format!("{s} (surprise measured for {m} episodes)")
                } else {
                    s
                }
            }
            Activity::Dream | Activity::Think => l.dream(o, now)?,
            Activity::Rest if !gpu_free => format!("resting: {}", pauser.why),
            Activity::Rest => "resting".into(),
        };
        l.drives.done(act, &rates);
        let forgot = l.store.plan_forgetting(o.budget, now, &first_sentence_gist)?;
        let d = l.drives.clone();
        l.put(&["drives", &now.to_string(), &d.memory.to_string(), &d.curiosity.to_string(), &d.fatigue.to_string()])?;
        println!(
            "{}  +{new} episodes  {:?}: {what}{}",
            format_time(now),
            act,
            if forgot.is_empty() { String::new() } else { format!("  ({} texts turned into gists)", forgot.len()) }
        );
        if o.once {
            return Ok(());
        }
        std::thread::sleep(Duration::from_secs(o.every));
    }
}

fn cmd_status(o: &Opts) -> Result<(), String> {
    let l = Learner::open(&o.root)?;
    let r = &l.reg;
    println!("live brain: {}", r.live.map_or("none".to_string(), |g| format!("{g} (lineage {:?})", r.lineage(g))));
    let count = |f: &dyn Fn(&ava_online::ckpt::Ckpt) -> bool| r.ckpts.values().filter(|c| f(c)).count();
    println!(
        "brains: {} made, {} rejected, {} tainted (erased data)",
        r.ckpts.len(),
        count(&|c| c.rejected.is_some()),
        count(&|c| c.tainted.is_some())
    );
    let mut by: BTreeMap<String, usize> = BTreeMap::new();
    for e in l.store.eps.values() {
        *by.entry(format!("{}/{:?}", e.origin.name(), e.tier)).or_default() += 1;
    }
    let bytes: usize = l.store.eps.values().map(|e| e.text.len()).sum();
    println!("episodes: {} ({:.1} MB of text kept): {by:?}", l.store.eps.len(), bytes as f64 / 1e6);
    println!("erased: {} episodes, {} people", l.store.erased_ids.len(), l.store.erased_users.len());
    println!("drives: memory {:.2}, curiosity {:.2}, fatigue {:.2}", l.drives.memory, l.drives.curiosity, l.drives.fatigue);
    println!("self-questions anchored: {}", l.anchor.len());
    for line in l.comp.lines() {
        println!("  {line}");
    }
    Ok(())
}

fn erase_and_taint(l: &mut Learner, ids: Vec<String>) -> Result<(), String> {
    let set: BTreeSet<String> = ids.iter().cloned().collect();
    let tainted = l.reg.taint_by_episodes(&set)?;
    println!("erased {} episodes; {} brains that learned from them are deleted; live brain: {:?}", ids.len(), tainted.len(), l.reg.live);
    println!("the diary files are not touched (append-only by design); remove them yourself if you want them gone too");
    Ok(())
}

fn main() {
    let r = parse_opts().and_then(|o| {
        if o.root.as_os_str().is_empty() && !matches!(o.rest.first().map(String::as_str), Some("fake-gate" | "fake-surprise")) {
            return Err("--root DIR is required".into());
        }
        match o.rest.first().map(String::as_str) {
            Some("run") => cmd_run(&o),
            Some("status") => cmd_status(&o),
            Some("import") => {
                let src = o.rest.get(1).ok_or("import BRAIN.safetensors")?;
                let mut reg = Registry::open(&o.root)?;
                let gen = reg.add_candidate(Path::new(src), None, "import", &[])?;
                reg.promote(gen, true)?;
                println!("brain {gen} is live");
                Ok(())
            }
            Some("rollback") => {
                let g = o.rest.get(1).and_then(|x| x.parse().ok()).ok_or("rollback GEN")?;
                Registry::open(&o.root)?.rollback(g)?;
                println!("brain {g} is live again");
                Ok(())
            }
            Some("erase-episode") => {
                let id = o.rest.get(1).ok_or("erase-episode ID")?.clone();
                let mut l = Learner::open(&o.root)?;
                l.store.erase(&id)?;
                erase_and_taint(&mut l, vec![id])
            }
            Some("erase-person") => {
                let who = o.rest.get(1).ok_or("erase-person NAME")?.clone();
                let mut l = Learner::open(&o.root)?;
                let ids = l.store.erase_user(&who, o.before.unwrap_or_else(local_now))?;
                erase_and_taint(&mut l, ids)
            }
            // A stand-in gate for tests: `fake-gate OUT same|drop`.
            Some("fake-gate") => {
                let out = o.rest.get(1).ok_or("fake-gate OUT same|drop")?;
                let drop = o.rest.get(2).map(String::as_str) == Some("drop");
                let mut s = String::new();
                for i in 0..100 {
                    let new = if drop && i < 20 { 0 } else { 1 };
                    s.push_str(&format!("skill\tfacts\t1\t{new}\n"));
                }
                s.push_str("self\t0\tMy name is Ava.\tMy name is Ava.\n");
                std::fs::write(out, s).map_err(|e| e.to_string())
            }
            // A stand-in surprise command for tests: `fake-surprise IDS OUT` (2.5 nats for every id).
            Some("fake-surprise") => {
                let (ids, out) = (o.rest.get(1).ok_or("fake-surprise IDS OUT")?, o.rest.get(2).ok_or("fake-surprise IDS OUT")?);
                let text = std::fs::read_to_string(ids).map_err(|e| format!("{ids}: {e}"))?;
                let s: String = std::iter::once("id\tnats_per_token\ttokens\n".to_string()).chain(text.lines().filter(|l| !l.is_empty()).map(|id| format!("{id}\t2.5\t10\n"))).collect();
                std::fs::write(out, s).map_err(|e| e.to_string())
            }
            Some("teach") => {
                let f = o.rest.get(1).ok_or("teach FILE.tsv [PERSON]")?;
                let text = std::fs::read_to_string(f).map_err(|e| format!("{f}: {e}"))?;
                let name = Path::new(f).file_stem().and_then(|s| s.to_str()).unwrap_or("lessons");
                let who = o.rest.get(2).map(String::as_str).unwrap_or("");
                let (eps, bad) = mind::lessons(name, &text, who, local_now());
                let mut l = Learner::open(&o.root)?;
                let mut added = 0;
                for e in eps {
                    added += l.store.add(e)? as usize;
                }
                println!("taught {added} new lessons ({bad} bad lines){}", if who.is_empty() { ", shared".to_string() } else { format!(", for {who} only") });
                Ok(())
            }
            Some("reflect") => {
                let l = Learner::open(&o.root)?;
                let day = match o.rest.get(1) {
                    Some(d) => parse_time(&format!("{d}T00:00:00")).ok_or("reflect [YYYY-MM-DD]")?,
                    None => local_now().div_euclid(86_400) * 86_400,
                };
                for (id, why) in mind::reflect_pick(&l.store, day, day + 86_400, 10) {
                    let e = &l.store.eps[&id];
                    println!("{why:?}\t{}\t{}", e.source, one_line(e.shown()));
                }
                Ok(())
            }
            Some("life") => {
                let l = Learner::open(&o.root)?;
                mind::autobiography(&l.store, &l.comp, l.reg.live_history.len()).iter().for_each(|x| println!("{x}"));
                Ok(())
            }
            Some("quiz") => {
                let n = o.rest.get(1).and_then(|x| x.parse().ok()).unwrap_or(20);
                let l = Learner::open(&o.root)?;
                println!("id\trecall\tquestion\tanswer");
                for q in mind::self_questions(&l.store, &o.person, local_now(), n) {
                    println!("{}\t{:.3}\t{}\t{}", ava_online::esc(&q.id), q.recall, ava_online::esc(&q.question), ava_online::esc(&q.answer));
                }
                Ok(())
            }
            Some("person") => {
                let who = o.rest.get(1).ok_or("person NAME")?;
                let l = Learner::open(&o.root)?;
                mind::person_model(&l.store, who).lines().iter().for_each(|x| println!("{x}"));
                Ok(())
            }
            Some("goal") => {
                let mut g = Goals::open(&o.root.join("goals.log"))?;
                let arg = |i: usize, what: &str| o.rest.get(i).cloned().ok_or_else(|| what.to_string());
                match o.rest.get(1).map(String::as_str) {
                    Some("add") => {
                        let use_ = "goal add OWNER|- \"TEXT\" \"CUE WORDS\" [YYYY-MM-DD|-] [VALUE]";
                        let owner = arg(2, use_)?;
                        let owner = if owner == "-" { String::new() } else { owner };
                        let due = match o.rest.get(5).map(String::as_str) {
                            None | Some("-") => None,
                            Some(d) => Some(parse_time(&format!("{d}T23:59:59")).ok_or(use_)?),
                        };
                        let value = o.rest.get(6).and_then(|v| v.parse().ok()).unwrap_or(0.5);
                        let id = g.add(&owner, &arg(3, use_)?, o.rest.get(4).map_or("", String::as_str), due, value, local_now())?;
                        println!("goal {id} added");
                    }
                    Some("progress") => {
                        let p = arg(3, "goal progress ID 0..1")?.parse().map_err(|_| "goal progress ID 0..1")?;
                        g.progress(&arg(2, "goal progress ID 0..1")?, p, local_now())?;
                    }
                    Some("done") => g.done(&arg(2, "goal done ID")?, local_now())?,
                    Some("drop") => g.drop_goal(&arg(2, "goal drop ID")?)?,
                    _ => return Err("goal add|progress|done|drop ...".into()),
                }
                Ok(())
            }
            Some("goals") => {
                let g = Goals::open(&o.root.join("goals.log"))?;
                let heard = o.rest[1..].join(" ");
                println!("id\tpull\towner\tprogress\tgoal");
                for (x, p) in arbitrate(&g, &o.person, local_now(), &heard) {
                    let owner = if x.owner.is_empty() { "her" } else { &x.owner };
                    println!("{}\t{p:.3}\t{owner}\t{:.2}\t{}", x.id, x.progress, ava_online::esc(&x.text));
                }
                Ok(())
            }
            Some("care") => {
                let who = o.rest.get(1).ok_or("care NAME")?;
                let l = Learner::open(&o.root)?;
                reliance(&l.store, who, local_now()).lines().iter().for_each(|x| println!("{x}"));
                Ok(())
            }
            _ => Err("give a command: import, run, status, rollback, erase-episode, erase-person, teach, reflect, life, quiz, person, goal, goals, care".into()),
        }
    });
    if let Err(e) = r {
        eprintln!("ava_learner: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::command_args;

    #[test]
    fn command_templates_split_and_fill() {
        let a = command_args(r#"grow --load {model} --out "{out}" --note "two words" """#, &[("model", "a b.st"), ("out", "c.st")]);
        assert_eq!(a, vec!["grow", "--load", "a b.st", "--out", "c.st", "--note", "two words", ""]);
    }
}
