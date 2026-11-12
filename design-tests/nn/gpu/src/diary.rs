// TESTING / DESIGN ONLY — diary: Ava's diary, shared by ava_chat, diary_day and diary_growth. She keeps every
// conversation (append-only, never edited or erased), sleeps on each day to learn from it, and can be shown her own
// growth. Brain model: the hippocampus records the day, sleep files it into long-term memory, and a correction teaches
// her the right answer you gave, not her own wrong one.
//
// THE DIARY. One file per local day, DIR/YYYY-MM-DD.tsv (DIR is data/diary in the repo root by default, git-ignored:
// the diary is private and never published). A header line, then one line per event, appended the moment it happens
// (one write per line, then synced) and never rewritten:
//   time                 model                         turn  user              ava
//   2026-10-28T14:03:05  v41/reward/model.safetensors  3     What is 2 and 3?  (her reply)
//   2026-10-28T14:03:20  v41/reward/model.safetensors  3     /correct          5
//   2026-10-28T14:03:31  v41/reward/model.safetensors  4     /good
// A turn line holds what you said and her reply. A command line (user field exactly /correct, /good or /bad) refers
// to the turn with that id. Text fields escape backslash, tab, newline and CR as \\ \t \n \r (mc::unescape reads the
// first three the same way); a turn's text that starts with '/' is written as \/ so it can never read as a command.
// Turn ids count up within a day's file. `model` is the model file's path after "runs/" (else its folder and file
// name), never a full path.
//
// TORN LINES (v42). A crash can cut the last write short, leaving a line with no newline, and it can cut a character
// in half. The reader reads bytes, drops a last piece with no newline, decodes each line on its own and counts a bad
// line instead of failing the file. The next append closes a torn line with TORN_MARK (five tabs, so it can never
// split into the 5 fields of a good line) before its newline, so a torn "/correct\tPar" never teaches "Par".
//
// PROVENANCE (v42). Next to each day file, DIR/YYYY-MM-DD.gen.tsv records how each reply was made (model file sha256,
// seed and sampler state, temperature, top-k, memory, prompt ids, every drawn id and its log-probs), so later learning
// can correct for replies an older model wrote. The day file's format is unchanged, so old readers still work.
//
// RESOLUTION. A turn's correction is the last /correct for it, its rating the last /good or /bad. A command applies
// to the latest turn line with its id above it (two sessions on one day share the counter).
//
// THE DAY (diary_day). Each turn gives at most one sleep episode, in her taught question-answer format:
//   corrected            -> "Question: {you}\nAnswer: {your correction}"   (she learns your answer, not her own)
//   rated /good          -> "Question: {you}\nAnswer: {her reply}"
//   rated /bad, no fix   -> left out (she does not rehearse her mistakes)
//   unrated (Heard)      -> your words alone, as plain text she heard; her unconfirmed reply is never rehearsed.
//                           Unrated::Drop leaves them out.
// Text is put on one line (each whitespace run becomes one space), so an episode's only newline is the format's.
// Episodes are separated by blank lines, the rule of grow_gpu's `episodes()` (copied here as `grow_episodes`, and
// checked against it in the tests). A file holding one two-line episode would be read one line per episode, so a
// lone episode is written twice (identical copies are drawn exactly like one).

use ava_data::bpe::{Bpe, EOS};
use ava_data::numbers::{insert_marks, strip_marks};
use ava_data::rng::SplitMix64;
use candle_core::{Tensor, D};
use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::model::{Model, Stream};

/// Where the diary lives by default (git-ignored, never published).
pub const DIARY_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/diary");
pub const HEADER: &str = "time\tmodel\tturn\tuser\tava";

/// Backslash, tab, newline and CR as `\\`, `\t`, `\n`, `\r`.
pub fn escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '\\' => o.push_str("\\\\"),
            '\t' => o.push_str("\\t"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            _ => o.push(c),
        }
    }
    o
}

/// A turn's text: `escape`, and a leading '/' as `\/` (so it never reads as a command).
fn escape_text(s: &str) -> String {
    let e = escape(s);
    if e.starts_with('/') {
        format!("\\{e}")
    } else {
        e
    }
}

/// The inverse of `escape` (and of `\/`); an unknown escape is kept as it is.
pub fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('\\') => out.push('\\'),
            Some('/') => out.push('/'),
            Some(o) => {
                out.push('\\');
                out.push(o);
            }
            None => out.push('\\'),
        }
    }
    out
}

/// What one diary line records.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Turn { user: String, ava: String },
    Correct(String),
    Good,
    Bad,
}

/// One diary line.
#[derive(Clone, Debug, PartialEq)]
pub struct Line {
    pub time: String,
    pub model: String,
    pub turn: u64,
    pub event: Event,
}

impl Line {
    /// The TSV line (no newline).
    pub fn to_tsv(&self) -> String {
        let (a, b) = match &self.event {
            Event::Turn { user, ava } => (escape_text(user), escape(ava)),
            Event::Correct(t) => ("/correct".to_string(), escape(t)),
            Event::Good => ("/good".to_string(), String::new()),
            Event::Bad => ("/bad".to_string(), String::new()),
        };
        format!("{}\t{}\t{}\t{a}\t{b}", escape(&self.time), escape(&self.model), self.turn)
    }

    pub fn parse(s: &str) -> Result<Line, String> {
        let f: Vec<&str> = s.split('\t').collect();
        if f.len() != 5 {
            return Err(format!("{} fields, want 5", f.len()));
        }
        let turn = f[2].parse::<u64>().map_err(|_| format!("bad turn id {:?}", f[2]))?;
        let event = match f[3] {
            "/correct" => Event::Correct(unescape(f[4])),
            "/good" => Event::Good,
            "/bad" => Event::Bad,
            u if u.starts_with('/') => return Err(format!("unknown command {u:?}")),
            u => Event::Turn { user: unescape(u), ava: unescape(f[4]) },
        };
        Ok(Line { time: unescape(f[0]), model: unescape(f[1]), turn, event })
    }
}

/// Written after a torn last line, before the newline that closes it. A good line has exactly 4 tabs (escape never
/// writes a raw tab), so a line with these 5 more splits into 6+ fields and can never parse, however it was cut.
pub const TORN_MARK: &str = "\t\t\t\t\t#torn";

/// Append one record to a TSV file: `header` first if the file is new; if a crash left the last line torn, TORN_MARK
/// and a newline first, to close it as a bad line; then the record in one write, synced before returning. Nothing
/// already in the file is touched.
pub(crate) fn append_record(path: &Path, header: &str, record: &str) -> Result<(), String> {
    let ctx = |e: std::io::Error| format!("diary {}: {e}", path.display());
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir).map_err(ctx)?;
    }
    let mut f = OpenOptions::new().create(true).append(true).read(true).open(path).map_err(ctx)?;
    let len = f.metadata().map_err(ctx)?.len();
    let mut buf = String::new();
    if len == 0 {
        buf.push_str(header);
        buf.push('\n');
    } else {
        let mut last = [0u8; 1];
        f.seek(SeekFrom::End(-1)).map_err(ctx)?;
        f.read_exact(&mut last).map_err(ctx)?;
        if last[0] != b'\n' {
            buf.push_str(TORN_MARK);
            buf.push('\n');
        }
    }
    buf.push_str(record);
    buf.push('\n');
    f.write_all(buf.as_bytes()).map_err(ctx)?;
    f.sync_data().map_err(ctx)
}

/// Append one line to a day file (see `append_record`).
pub fn append_line(path: &Path, line: &Line) -> Result<(), String> {
    append_record(path, HEADER, &line.to_tsv())
}

/// The complete lines of a file's bytes, split at '\n' with a CR before it removed. A last piece with no '\n' (a write
/// a crash cut short) is not a line: it is dropped, and the second value says whether there was one.
pub fn complete_lines(bytes: &[u8]) -> (Vec<&[u8]>, bool) {
    let mut lines: Vec<&[u8]> = bytes.split(|&b| b == b'\n').collect();
    // split always yields a last piece: empty when the file ends with '\n', else the torn tail
    let tail = lines.pop().unwrap_or(&[]);
    let lines = lines.into_iter().map(|l| l.strip_suffix(b"\r").unwrap_or(l)).collect();
    (lines, !tail.is_empty())
}

/// The lines of a TSV file as text, each decoded on its own: (good text lines, count of bad ones: a torn tail, a line
/// that is not UTF-8 (a character cut by a crash), or one closed with TORN_MARK). Empty lines and `header` are skipped.
/// A missing file is empty.
pub(crate) fn read_text_lines(path: &Path, header: &str) -> Result<(Vec<String>, usize), String> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), 0)),
        Err(e) => return Err(format!("read {}: {e}", path.display())),
    };
    let (lines, torn) = complete_lines(&bytes);
    let (mut out, mut bad) = (Vec::new(), usize::from(torn));
    for raw in lines {
        match std::str::from_utf8(raw) {
            Err(_) => bad += 1,
            Ok(s) if s.contains(TORN_MARK) => bad += 1,
            Ok(s) if s.is_empty() || s == header => {}
            Ok(s) => out.push(s.to_string()),
        }
    }
    Ok((out, bad))
}

/// Every well-formed line of a day file, and the number of lines skipped (torn, not UTF-8, or malformed). A missing
/// file is empty. Only an I/O error fails.
pub fn read_day(path: &Path) -> Result<(Vec<Line>, usize), String> {
    let (lines, mut bad) = read_text_lines(path, HEADER)?;
    let mut out = Vec::with_capacity(lines.len());
    for line in &lines {
        match Line::parse(line) {
            Ok(l) => out.push(l),
            Err(_) => bad += 1,
        }
    }
    Ok((out, bad))
}

// ------------------------------------------------------------------------------------------------ provenance

/// The provenance file's header (one line per reply, next to the day file).
pub const GEN_HEADER: &str = "time\tmodel\tturn\tmodel_sha256\tseed\trng\ttemperature\ttop_k\tmemory\tremembered\tmax_tokens\tprompt_ids\tsampled_ids\tlogp\tsample_logp";

/// How one reply was made. Joined to its day-file turn by (time, turn).
#[derive(Clone, Debug, PartialEq)]
pub struct Gen {
    pub time: String,
    /// the model's name as in the day file
    pub model: String,
    pub turn: u64,
    /// sha256 of the model file (hex)
    pub model_sha256: String,
    /// the session's seed
    pub seed: u64,
    /// the sampler's state when this reply began: SplitMix64(rng) draws exactly her tokens again
    pub rng: u64,
    pub temperature: f32,
    pub top_k: usize,
    /// the --memory budget in tokens, and how many earlier turns fitted in it
    pub memory: usize,
    pub remembered: usize,
    pub max_tokens: usize,
    /// the exact prompt she read
    pub prompt_ids: Vec<u32>,
    /// every token she drew, in order (the EOS that ended her reply too, and any tokens past a cut)
    pub sampled_ids: Vec<u32>,
    /// ln p(token) under the model itself (temperature 1, every token allowed)
    pub logp: Vec<f32>,
    /// ln p(token) under the sampler she used (temperature and top-k); 0 when the pick was greedy
    pub sample_logp: Vec<f32>,
}

fn join_nums<T: std::fmt::Display>(v: &[T]) -> String {
    v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
}

fn parse_nums<T: std::str::FromStr>(s: &str, what: &str) -> Result<Vec<T>, String> {
    s.split_whitespace().map(|x| x.parse::<T>().map_err(|_| format!("bad {what} {x:?}"))).collect()
}

impl Gen {
    /// The TSV line (no newline). Floats are written in Rust's shortest form, which reads back to the same bits.
    pub fn to_tsv(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            escape(&self.time),
            escape(&self.model),
            self.turn,
            self.model_sha256,
            self.seed,
            self.rng,
            self.temperature,
            self.top_k,
            self.memory,
            self.remembered,
            self.max_tokens,
            join_nums(&self.prompt_ids),
            join_nums(&self.sampled_ids),
            join_nums(&self.logp),
            join_nums(&self.sample_logp)
        )
    }

    pub fn parse(s: &str) -> Result<Gen, String> {
        let f: Vec<&str> = s.split('\t').collect();
        if f.len() != 15 {
            return Err(format!("{} fields, want 15", f.len()));
        }
        let int = |x: &str, what: &str| x.parse::<u64>().map_err(|_| format!("bad {what} {x:?}"));
        let g = Gen {
            time: unescape(f[0]),
            model: unescape(f[1]),
            turn: int(f[2], "turn")?,
            model_sha256: f[3].to_string(),
            seed: int(f[4], "seed")?,
            rng: int(f[5], "rng")?,
            temperature: f[6].parse().map_err(|_| format!("bad temperature {:?}", f[6]))?,
            top_k: int(f[7], "top_k")? as usize,
            memory: int(f[8], "memory")? as usize,
            remembered: int(f[9], "remembered")? as usize,
            max_tokens: int(f[10], "max_tokens")? as usize,
            prompt_ids: parse_nums(f[11], "prompt id")?,
            sampled_ids: parse_nums(f[12], "sampled id")?,
            logp: parse_nums(f[13], "logp")?,
            sample_logp: parse_nums(f[14], "sample_logp")?,
        };
        if g.logp.len() != g.sampled_ids.len() || g.sample_logp.len() != g.sampled_ids.len() {
            return Err(format!("{} sampled ids but {} logp and {} sample_logp", g.sampled_ids.len(), g.logp.len(), g.sample_logp.len()));
        }
        Ok(g)
    }
}

/// The provenance file of a day: DIR/YYYY-MM-DD.gen.tsv (not a day file: `is_day_name` is false for it).
pub fn gen_path(dir: &Path, date: &str) -> PathBuf {
    dir.join(format!("{date}.gen.tsv"))
}

/// Append one reply's provenance (the same torn-line rules as the day file).
pub fn append_gen(path: &Path, g: &Gen) -> Result<(), String> {
    append_record(path, GEN_HEADER, &g.to_tsv())
}

/// Every well-formed provenance line, and the number of bad ones. A missing file is empty.
pub fn read_gen(path: &Path) -> Result<(Vec<Gen>, usize), String> {
    let (lines, mut bad) = read_text_lines(path, GEN_HEADER)?;
    let mut out = Vec::with_capacity(lines.len());
    for line in &lines {
        match Gen::parse(line) {
            Ok(g) => out.push(g),
            Err(_) => bad += 1,
        }
    }
    Ok((out, bad))
}

/// The sha256 (hex) of a model file, for `Gen::model_sha256`.
pub fn model_sha256(path: &Path) -> Result<String, String> {
    ava_data::sha256::sha256_file(path).map_err(|e| format!("sha256 {}: {e}", path.display()))
}

/// The id for the next turn of a day: one past the largest so far (the first is 1).
pub fn next_turn(lines: &[Line]) -> u64 {
    lines.iter().map(|l| l.turn).max().unwrap_or(0) + 1
}

/// A turn with its correction and rating resolved.
#[derive(Clone, Debug, PartialEq)]
pub struct Turn {
    pub date: String,
    pub time: String,
    pub model: String,
    pub turn: u64,
    pub user: String,
    pub ava: String,
    pub correction: Option<String>,
    /// Some(true) = /good, Some(false) = /bad (the last one given)
    pub rating: Option<bool>,
}

/// Turns in file order with their last correction and last rating; also the number of commands that name no turn above
/// them (orphans, ignored).
pub fn resolve(date: &str, lines: &[Line]) -> (Vec<Turn>, usize) {
    let (mut turns, mut orphans) = (Vec::<Turn>::new(), 0usize);
    let mut latest = std::collections::HashMap::<u64, usize>::new();
    for l in lines {
        match &l.event {
            Event::Turn { user, ava } => {
                latest.insert(l.turn, turns.len());
                turns.push(Turn {
                    date: date.to_string(),
                    time: l.time.clone(),
                    model: l.model.clone(),
                    turn: l.turn,
                    user: user.clone(),
                    ava: ava.clone(),
                    correction: None,
                    rating: None,
                });
            }
            ev => match latest.get(&l.turn) {
                None => orphans += 1,
                Some(&i) => match ev {
                    Event::Correct(t) => turns[i].correction = Some(t.clone()),
                    Event::Good => turns[i].rating = Some(true),
                    Event::Bad => turns[i].rating = Some(false),
                    Event::Turn { .. } => unreachable!(),
                },
            },
        }
    }
    (turns, orphans)
}

/// What she should remember of a turn (short-term memory in ava_chat): your words and the answer that stands (your
/// correction, else her reply), or None for a reply rated /bad and never corrected.
pub fn remembered(t: &Turn) -> Option<(String, String)> {
    match (&t.correction, t.rating) {
        (Some(c), _) => Some((t.user.clone(), c.clone())),
        (None, Some(false)) => None,
        (None, _) => Some((t.user.clone(), t.ava.clone())),
    }
}

/// Whitespace runs to one space, trimmed.
pub fn one_line(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// One sleep episode in her taught format: "Question: q\nAnswer: a".
pub fn qa_episode(question: &str, answer: &str) -> String {
    format!("{} {}", crate::qa::prompt_text("", &one_line(question)), one_line(answer))
}

/// What happens to unrated turns in a day file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unrated {
    /// your words as plain text she heard (her unconfirmed reply left out)
    Heard,
    /// left out
    Drop,
}

/// The counts of a day.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DayCounts {
    pub turns: usize,
    pub corrected: usize,
    pub confirmed: usize,
    pub bad_dropped: usize,
    pub heard: usize,
    pub unrated_dropped: usize,
    /// turns left out because the text to rehearse was empty
    pub empty: usize,
}

/// The sleep episodes of a day's turns (the rule at the top of this file), in diary order.
pub fn day_episodes(turns: &[Turn], unrated: Unrated) -> (Vec<String>, DayCounts) {
    let mut c = DayCounts { turns: turns.len(), ..Default::default() };
    let mut eps = Vec::new();
    for t in turns {
        let q = one_line(&t.user);
        match (&t.correction, t.rating) {
            (Some(fix), _) => {
                if q.is_empty() || one_line(fix).is_empty() {
                    c.empty += 1;
                } else {
                    eps.push(qa_episode(&q, fix));
                    c.corrected += 1;
                }
            }
            (None, Some(true)) => {
                if q.is_empty() || one_line(&t.ava).is_empty() {
                    c.empty += 1;
                } else {
                    eps.push(qa_episode(&q, &t.ava));
                    c.confirmed += 1;
                }
            }
            (None, Some(false)) => c.bad_dropped += 1,
            (None, None) => match unrated {
                Unrated::Drop => c.unrated_dropped += 1,
                Unrated::Heard if q.is_empty() => c.empty += 1,
                Unrated::Heard => {
                    eps.push(q);
                    c.heard += 1;
                }
            },
        }
    }
    (eps, c)
}

/// The day file's text: episodes separated by blank lines; a lone episode is written twice (see the top of the file).
pub fn day_text(eps: &[String]) -> String {
    let mut v: Vec<&str> = eps.iter().map(String::as_str).collect();
    if v.len() == 1 {
        v.push(v[0]);
    }
    let mut s = v.join("\n\n");
    if !s.is_empty() {
        s.push('\n');
    }
    s
}

/// A copy of grow_gpu's `episodes()` (blocks separated by blank lines; a file without blank lines has one episode per
/// line), so the day file can be checked against what sleep will read.
pub fn grow_episodes(text: &str) -> Vec<String> {
    let text = text.replace("\r\n", "\n");
    let blocks: Vec<String> = text.split("\n\n").map(|b| b.trim().to_string()).filter(|b| !b.is_empty()).collect();
    if blocks.len() > 1 {
        return blocks;
    }
    text.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()
}

/// Her reply cut at the first sign of a new question ("Question:", "Context:", or "Answer:" on a new line), trimmed;
/// and whether it was cut.
pub fn cut_reply(text: &str) -> (String, bool) {
    let cut = ["Question:", "Context:", "\nAnswer:"].iter().filter_map(|p| text.find(p)).min();
    match cut {
        Some(i) => (text[..i].trim().to_string(), true),
        None => (text.trim().to_string(), false),
    }
}

/// Lowercase, letters and digits kept, everything else a space, runs collapsed.
pub fn norm_answer(s: &str) -> String {
    let t: String = s.chars().map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { ' ' }).collect();
    one_line(&t)
}

/// Does `reply` say `right`: equal after `norm_answer`, or starts with it as whole words.
pub fn says(reply: &str, right: &str) -> bool {
    let (f, r) = (norm_answer(reply), norm_answer(right));
    !r.is_empty() && (f == r || f.starts_with(&format!("{r} ")))
}

/// The model's name for the diary: its path after a "runs" folder, else its folder and file name (never a full path).
pub fn model_name(path: &Path) -> String {
    let parts: Vec<String> = path.components().map(|c| c.as_os_str().to_string_lossy().to_string()).collect();
    if let Some(i) = parts.iter().rposition(|p| p == "runs") {
        if i + 1 < parts.len() {
            return parts[i + 1..].join("/");
        }
    }
    let n = parts.len();
    parts[n.saturating_sub(2)..].join("/")
}

/// The day file for a date.
pub fn day_path(dir: &Path, date: &str) -> PathBuf {
    dir.join(format!("{date}.tsv"))
}

/// Is `name` a day file's name (YYYY-MM-DD.tsv)?
pub fn is_day_name(name: &str) -> bool {
    let b = name.as_bytes();
    b.len() == 14
        && name.ends_with(".tsv")
        && b[..10].iter().enumerate().all(|(i, &c)| if i == 4 || i == 7 { c == b'-' } else { c.is_ascii_digit() })
}

/// The date of a day file (its name without .tsv), if it is one.
pub fn day_of(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    is_day_name(name).then(|| name[..10].to_string())
}

/// Every day file in `dir` (not its subfolders), oldest first.
pub fn day_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("read {}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && day_of(p).is_some())
        .collect();
    v.sort();
    Ok(v)
}

/// (date "YYYY-MM-DD", time "YYYY-MM-DDTHH:MM:SS") of a civil date and time.
fn stamp(y: i64, mo: u32, d: u32, h: u32, mi: u32, s: u32) -> (String, String) {
    let date = format!("{y:04}-{mo:02}-{d:02}");
    let time = format!("{date}T{h:02}:{mi:02}:{s:02}");
    (date, time)
}

/// (year, month, day) of a day count since 1970-01-01 (Howard Hinnant's civil_from_days).
pub fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

#[cfg(windows)]
#[repr(C)]
#[derive(Default)]
struct WinSystemTime {
    year: u16,
    month: u16,
    day_of_week: u16,
    day: u16,
    hour: u16,
    minute: u16,
    second: u16,
    millis: u16,
}

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn GetLocalTime(st: *mut WinSystemTime);
}

/// The local date and time now (Windows: GetLocalTime; elsewhere UTC).
pub fn local_now() -> (String, String) {
    #[cfg(windows)]
    {
        let mut st = WinSystemTime::default();
        // SAFETY: GetLocalTime fills the SYSTEMTIME it is given (8 u16 fields, laid out as WinSystemTime).
        unsafe { GetLocalTime(&mut st) };
        stamp(st.year as i64, st.month as u32, st.day as u32, st.hour as u32, st.minute as u32, st.second as u32)
    }
    #[cfg(not(windows))]
    {
        let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
        let (y, m, d) = civil_from_days(secs.div_euclid(86_400));
        let r = secs.rem_euclid(86_400) as u32;
        stamp(y, m, d, r / 3600, r / 60 % 60, r % 60)
    }
}

/// `--threads N`: the CPU thread count candle uses (set before the first tensor op).
pub fn set_threads(n: usize) {
    if n > 0 {
        std::env::set_var("RAYON_NUM_THREADS", n.to_string());
        std::env::set_var("CANDLE_NUM_THREADS", n.to_string());
    }
}

// ------------------------------------------------------------------------------------------------ asking her

/// Tokens of `text`, with the glance's length marks when the model has them.
pub fn encode(bpe: &Bpe, glance: bool, text: &str) -> Vec<u32> {
    let t = bpe.encode(text);
    if glance {
        insert_marks(&t)
    } else {
        t
    }
}

/// The prompt she is asked with: [EOS], then as many of the most recent remembered turns as fit in `budget` tokens
/// (each "Question: ..\nAnswer: .." [EOS], the shape of a sleep episode), then "Question: q\nAnswer:" in her taught
/// format. Returns the tokens and how many turns were remembered.
pub fn prompt_ids(bpe: &Bpe, glance: bool, memory: &[(String, String)], question: &str, budget: usize) -> (Vec<u32>, usize) {
    prompt_ids_ctx(bpe, glance, memory, question, "", budget)
}

/// v43: `prompt_ids` with a context (her library's passages) in the last question (qa::prompt_text's Context line).
pub fn prompt_ids_ctx(bpe: &Bpe, glance: bool, memory: &[(String, String)], question: &str, context: &str, budget: usize) -> (Vec<u32>, usize) {
    let mut kept: Vec<Vec<u32>> = Vec::new();
    let mut used = 0usize;
    for (q, a) in memory.iter().rev() {
        let mut t = encode(bpe, glance, &qa_episode(q, a));
        t.push(EOS);
        if used + t.len() > budget {
            break;
        }
        used += t.len();
        kept.push(t);
    }
    let mut ids = vec![EOS];
    for t in kept.iter().rev() {
        ids.extend_from_slice(t);
    }
    ids.extend(encode(bpe, glance, &crate::qa::prompt_text(context, &one_line(question))));
    (ids, kept.len())
}

/// One token from `logits` [V]: the most likely when `temperature` <= 0, else logits / temperature, the top-k kept,
/// softmax, one draw (generate_gpu's sampler).
pub fn pick(logits: &Tensor, temperature: f32, top_k: usize, rng: &mut SplitMix64) -> candle_core::Result<u32> {
    if temperature <= 0.0 {
        return logits.argmax(0)?.to_scalar::<u32>();
    }
    let probs = sampling_probs(logits, temperature)?;
    Ok(draw_top_k(&probs, top_k, rng).0 as u32)
}

/// softmax(logits / temperature) on the logits' device, copied out (the sampler's distribution before top-k).
fn sampling_probs(logits: &Tensor, temperature: f32) -> candle_core::Result<Vec<f32>> {
    let last = logits.affine(1.0 / temperature as f64, 0.0)?;
    candle_nn::ops::softmax(&last, D::Minus1)?.to_vec1()
}

/// One draw from the `top_k` most likely of `probs` (one rng step): (the index drawn, the kept probability mass).
fn draw_top_k(probs: &[f32], top_k: usize, rng: &mut SplitMix64) -> (usize, f64) {
    let mut order: Vec<usize> = (0..probs.len()).collect();
    order.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    order.truncate(top_k.max(1));
    let total: f64 = order.iter().map(|&j| probs[j] as f64).sum();
    let mut r = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * total;
    let mut out = order[order.len() - 1];
    for &j in &order {
        r -= probs[j] as f64;
        if r <= 0.0 {
            out = j;
            break;
        }
    }
    (out, total)
}

/// One token drawn by `pick_traced`, with its log-probabilities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Draw {
    pub id: u32,
    /// ln p(id) under the model (temperature 1, every token allowed)
    pub logp: f32,
    /// ln p(id) under the sampler used (temperature, then top-k renormalised); 0 for a greedy pick
    pub sample_logp: f32,
}

/// `pick` with the drawn token's log-probabilities. It draws exactly the token `pick` draws from the same rng state
/// (the same device ops and the same draw), and advances the rng the same way.
pub fn pick_traced(logits: &Tensor, temperature: f32, top_k: usize, rng: &mut SplitMix64) -> candle_core::Result<Draw> {
    let row: Vec<f32> = logits.to_dtype(candle_core::DType::F32)?.to_vec1()?;
    let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = mx + row.iter().map(|&x| (x as f64 - mx).exp()).sum::<f64>().ln();
    let model_logp = |j: usize| (row[j] as f64 - lse) as f32;
    if temperature <= 0.0 {
        let id = logits.argmax(0)?.to_scalar::<u32>()?;
        return Ok(Draw { id, logp: model_logp(id as usize), sample_logp: 0.0 });
    }
    let probs = sampling_probs(logits, temperature)?;
    let (j, total) = draw_top_k(&probs, top_k, rng);
    Ok(Draw { id: j as u32, logp: model_logp(j), sample_logp: ((probs[j] as f64) / total).ln() as f32 })
}

/// Every token drawn for one reply, with its log-probabilities (for `Gen`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trace {
    pub ids: Vec<u32>,
    pub logp: Vec<f32>,
    pub sample_logp: Vec<f32>,
}

impl Trace {
    pub fn push(&mut self, d: Draw) {
        self.ids.push(d.id);
        self.logp.push(d.logp);
        self.sample_logp.push(d.sample_logp);
    }
}

/// Her reply to the prompt `ids`, from a fresh carried state (Model::step): tokens until her EOS, a new question
/// (`cut_reply`) or `max_tokens`. Returns the reply (marks stripped) and the number of tokens she wrote.
pub fn reply(model: &Model, bpe: &Bpe, ids: &[u32], max_tokens: usize, temperature: f32, top_k: usize, rng: &mut SplitMix64) -> Result<(String, usize), String> {
    reply_core(model, bpe, ids, max_tokens, temperature, top_k, rng, None)
}

/// `reply`, also returning every token drawn and its log-probabilities (the same tokens `reply` draws).
pub fn reply_traced(model: &Model, bpe: &Bpe, ids: &[u32], max_tokens: usize, temperature: f32, top_k: usize, rng: &mut SplitMix64) -> Result<(String, usize, Trace), String> {
    let mut tr = Trace::default();
    let (r, n) = reply_core(model, bpe, ids, max_tokens, temperature, top_k, rng, Some(&mut tr))?;
    Ok((r, n, tr))
}

/// v42 (S2): `reply_traced` from a stream she already has (a conversation's saved state, `Model::stream_from`): the
/// prompt is read in one batched pass (`Model::feed`), and the stream keeps everything she read and wrote, to be saved.
#[allow(clippy::too_many_arguments)]
pub fn reply_traced_in(model: &Model, bpe: &Bpe, st: &mut Stream, ids: &[u32], max_tokens: usize, temperature: f32, top_k: usize, rng: &mut SplitMix64) -> Result<(String, usize, Trace), String> {
    let mut tr = Trace::default();
    let (r, n) = reply_from(model, bpe, st, ids, true, max_tokens, temperature, top_k, rng, Some(&mut tr))?;
    Ok((r, n, tr))
}

#[allow(clippy::too_many_arguments)]
fn reply_core(
    model: &Model,
    bpe: &Bpe,
    ids: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    rng: &mut SplitMix64,
    trace: Option<&mut Trace>,
) -> Result<(String, usize), String> {
    let mut st = model.stream().map_err(|e| e.to_string())?;
    reply_from(model, bpe, &mut st, ids, false, max_tokens, temperature, top_k, rng, trace)
}

/// The reply from `st`: read `ids` (token by token as before, or with `batched` in one `Model::feed`), then write.
#[allow(clippy::too_many_arguments)]
fn reply_from(
    model: &Model,
    bpe: &Bpe,
    st: &mut Stream,
    ids: &[u32],
    batched: bool,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    rng: &mut SplitMix64,
    mut trace: Option<&mut Trace>,
) -> Result<(String, usize), String> {
    let e = |e: candle_core::Error| e.to_string();
    let mut next = None;
    if batched && !ids.is_empty() {
        next = Some(model.feed(st, ids).map_err(e)?);
    } else {
        for &t in ids {
            next = Some(model.step(st, t).map_err(e)?);
        }
    }
    let mut next = next.ok_or("empty prompt")?;
    let mut out = Vec::new();
    for n in 0..max_tokens {
        // without a trace, `pick` exactly as before (no extra copy of the logits)
        let tok = match trace.as_deref_mut() {
            Some(tr) => {
                let d = pick_traced(&next, temperature, top_k, rng).map_err(e)?;
                tr.push(d);
                d.id
            }
            None => pick(&next, temperature, top_k, rng).map_err(e)?,
        };
        if tok == EOS {
            break;
        }
        out.push(tok);
        if cut_reply(&bpe.decode(&strip_marks(&out))).1 {
            break;
        }
        if n + 1 < max_tokens {
            next = model.step(st, tok).map_err(e)?;
        }
    }
    Ok((cut_reply(&bpe.decode(&strip_marks(&out))).0, out.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("ava-diary-test-{}-{name}", std::process::id()));
        let _ = fs::remove_file(&p);
        p
    }

    fn line(turn: u64, event: Event) -> Line {
        Line { time: "2026-10-28T10:00:00".into(), model: "v41/reward/model.safetensors".into(), turn, event }
    }

    fn turn(user: &str, ava: &str) -> Event {
        Event::Turn { user: user.into(), ava: ava.into() }
    }

    #[test]
    fn escaping_round_trips() {
        for s in ["plain", "a\tb\nc\\d\re", "\\t is not a tab", "trailing\\", "/good", "\\/x", "", "naïve café ☕"] {
            assert_eq!(unescape(&escape(s)), s, "{s:?}");
            assert_eq!(unescape(&escape_text(s)), s, "{s:?}");
            assert!(!escape(s).contains(['\t', '\n', '\r']));
        }
        // mc::unescape reads the three escapes it knows the same way
        let s = "a\tb\nc\\d";
        assert_eq!(crate::mc::unescape(&escape(s)), s);
    }

    #[test]
    fn lines_round_trip_and_turn_text_never_reads_as_a_command() {
        let ls = vec![
            line(1, turn("What is 2 and 3?", "it is\tfour\nmaybe")),
            line(1, Event::Correct("5".into())),
            line(2, Event::Good),
            line(3, Event::Bad),
            line(4, turn("/good", "/bad")),
            line(5, turn("", "")),
        ];
        for l in &ls {
            let t = l.to_tsv();
            assert_eq!(t.split('\t').count(), 5, "{t:?}");
            assert_eq!(Line::parse(&t).unwrap(), *l);
        }
        assert!(Line::parse("a\tb\tc").is_err());
        assert!(Line::parse("t\tm\tx\tq\ta").is_err());
        assert!(Line::parse("t\tm\t1\t/quit\t").is_err());
    }

    #[test]
    fn appending_never_changes_what_is_there() {
        let p = tmp("append.tsv");
        let mut before = Vec::new();
        for (i, l) in [line(1, turn("hi", "hello")), line(1, Event::Good), line(2, turn("x\ty", "z"))].iter().enumerate() {
            append_line(&p, l).unwrap();
            let now = fs::read(&p).unwrap();
            assert!(now.starts_with(&before), "line {i} changed earlier bytes");
            assert!(now.ends_with(b"\n"));
            before = now;
        }
        let text = fs::read_to_string(&p).unwrap();
        assert_eq!(text.lines().next(), Some(HEADER));
        assert_eq!(text.matches(HEADER).count(), 1);
        let (ls, bad) = read_day(&p).unwrap();
        assert_eq!((ls.len(), bad), (3, 0));
        assert_eq!(next_turn(&ls), 3);
        // a crash tore the last line: the next append closes it first, the reader skips it, nothing is lost
        OpenOptions::new().append(true).open(&p).unwrap().write_all(b"2026-10-28T10:00:00\tm\t3\tWhat").unwrap();
        let torn = fs::read(&p).unwrap();
        append_line(&p, &line(3, turn("again", "ok"))).unwrap();
        assert!(fs::read(&p).unwrap().starts_with(&torn));
        let (ls, bad) = read_day(&p).unwrap();
        assert_eq!((ls.len(), bad), (4, 1));
        assert_eq!(ls[3], line(3, turn("again", "ok")));
        fs::remove_file(&p).unwrap();
        assert_eq!(read_day(&p).unwrap(), (Vec::new(), 0)); // a missing day is empty
    }

    #[test]
    fn a_torn_five_field_line_never_teaches() {
        let p = tmp("torn5.tsv");
        append_line(&p, &line(1, turn("Capital of France?", "Lyon"))).unwrap();
        // a crash cut "/correct\tParis" after "Par": the torn tail has all 5 fields but no newline
        let torn = b"2026-10-28T10:00:01\tm\t1\t/correct\tPar";
        assert!(Line::parse(std::str::from_utf8(torn).unwrap()).is_ok()); // why the old reader taught "Par"
        OpenOptions::new().append(true).open(&p).unwrap().write_all(torn).unwrap();
        let (ls, bad) = read_day(&p).unwrap();
        assert_eq!((ls.len(), bad), (1, 1));
        assert_eq!(resolve("d", &ls).0[0].correction, None);
        // the next append closes it with the mark: still one bad line, never "Par"
        append_line(&p, &line(1, Event::Correct("Paris".into()))).unwrap();
        let (ls, bad) = read_day(&p).unwrap();
        assert_eq!((ls.len(), bad), (2, 1));
        assert_eq!(resolve("d", &ls).0[0].correction.as_deref(), Some("Paris"));
        // a second crash in the middle of writing the mark itself: the line stays bad
        OpenOptions::new().append(true).open(&p).unwrap().write_all(b"2026-10-28T10:00:02\tm\t1\t/correct\tLy\t\t").unwrap();
        append_line(&p, &line(2, turn("and Spain?", "Madrid"))).unwrap();
        let (ls, bad) = read_day(&p).unwrap();
        assert_eq!((ls.len(), bad), (3, 2));
        assert_eq!(resolve("d", &ls).0[0].correction.as_deref(), Some("Paris"));
        let text = fs::read_to_string(&p).unwrap();
        assert_eq!(text.matches(TORN_MARK).count(), 2);
        assert!(text.ends_with('\n'));
        fs::remove_file(&p).unwrap();
    }

    #[test]
    fn a_cut_character_is_one_bad_line_not_a_failed_file() {
        let p = tmp("utf8.tsv");
        append_line(&p, &line(1, turn("naïve café", "☕ ok"))).unwrap();
        // a crash cut "☕" (3 bytes) after its first byte: no newline, and not UTF-8
        let full = line(2, turn("tea?", "☕")).to_tsv();
        OpenOptions::new().append(true).open(&p).unwrap().write_all(&full.as_bytes()[..full.len() - 2]).unwrap();
        assert!(fs::read_to_string(&p).is_err()); // the old reader failed the whole file here
        let (ls, bad) = read_day(&p).unwrap();
        assert_eq!((ls.len(), bad), (1, 1));
        assert_eq!(ls[0], line(1, turn("naïve café", "☕ ok")));
        append_line(&p, &line(3, turn("again", "fine"))).unwrap();
        let (ls, bad) = read_day(&p).unwrap();
        assert_eq!((ls.len(), bad), (2, 1));
        // a cut character inside a closed line (as an old binary would have left it): one bad line, the rest reads
        let mut bytes = fs::read(&p).unwrap();
        bytes.extend_from_slice(b"2026-10-28T10:00:00\tm\t4\tq\t\xe2\x98\n");
        bytes.extend_from_slice(format!("{}\r\n", line(5, turn("x", "y")).to_tsv()).as_bytes());
        fs::write(&p, bytes).unwrap();
        let (ls, bad) = read_day(&p).unwrap();
        assert_eq!((ls.len(), bad), (3, 2));
        assert_eq!(ls[2], line(5, turn("x", "y")));
        fs::remove_file(&p).unwrap();
        // the split itself
        assert_eq!(complete_lines(b"a\nb\r\nc"), (vec![&b"a"[..], &b"b"[..]], true));
        assert_eq!(complete_lines(b"a\n"), (vec![&b"a"[..]], false));
        assert_eq!(complete_lines(b""), (Vec::<&[u8]>::new(), false));
    }

    fn sample_gen(turn: u64) -> Gen {
        Gen {
            time: "2026-10-28T10:00:00".into(),
            model: "v41/reward/model.safetensors".into(),
            turn,
            model_sha256: "ab".repeat(32),
            seed: u64::MAX - 7,
            rng: 0x1234_5678_9abc_def0,
            temperature: 0.7,
            top_k: 40,
            memory: 512,
            remembered: 2,
            max_tokens: 80,
            prompt_ids: vec![256, 81, 117, 101, 4095],
            sampled_ids: vec![32, 53, 256],
            logp: vec![-0.123_456_79, -1.0e-7, -17.25],
            sample_logp: vec![-0.1, 0.0, -3.333_333_3],
        }
    }

    #[test]
    fn provenance_round_trips_next_to_the_day_file() {
        let dir = std::env::temp_dir().join(format!("ava-diary-gen-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let p = gen_path(&dir, "2026-10-28");
        assert_eq!(p.file_name().unwrap(), "2026-10-28.gen.tsv");
        let (a, b) = (sample_gen(1), Gen { prompt_ids: vec![], sampled_ids: vec![], logp: vec![], sample_logp: vec![], ..sample_gen(2) });
        append_gen(&p, &a).unwrap();
        append_gen(&p, &b).unwrap();
        append_line(&day_path(&dir, "2026-10-28"), &line(1, turn("q", "a"))).unwrap();
        let (gs, bad) = read_gen(&p).unwrap();
        assert_eq!((gs.clone(), bad), (vec![a.clone(), b], 0));
        // floats come back bit for bit
        assert!(gs[0].logp.iter().zip(&a.logp).all(|(x, y)| x.to_bits() == y.to_bits()));
        assert_eq!(gs[0].temperature.to_bits(), 0.7f32.to_bits());
        // the provenance file is not a day file, and the day file still reads as before
        assert_eq!(day_files(&dir).unwrap(), vec![day_path(&dir, "2026-10-28")]);
        assert_eq!(read_day(&day_path(&dir, "2026-10-28")).unwrap().0.len(), 1);
        // a torn provenance line is dropped like a torn diary line
        let t = sample_gen(3).to_tsv();
        OpenOptions::new().append(true).open(&p).unwrap().write_all(&t.as_bytes()[..t.len() - 3]).unwrap();
        assert_eq!(read_gen(&p).unwrap().0.len(), 2);
        append_gen(&p, &sample_gen(4)).unwrap();
        let (gs, bad) = read_gen(&p).unwrap();
        assert_eq!((gs.len(), bad, gs[2].turn), (3, 1, 4));
        let short = a.to_tsv().replace(" -17.25\t", "\t");
        assert_ne!(short, a.to_tsv());
        assert!(Gen::parse(&short).is_err()); // as many log-probs as drawn ids
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn traced_picks_draw_what_pick_draws() {
        let dev = candle_core::Device::Cpu;
        let mut src = SplitMix64::new(7);
        for _ in 0..40 {
            let v: Vec<f32> = (0..300).map(|_| (src.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 8.0 - 4.0).collect();
            let logits = Tensor::from_vec(v.clone(), 300, &dev).unwrap();
            let mx = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let lse = mx + v.iter().map(|&x| (x as f64 - mx).exp()).sum::<f64>().ln();
            for (temp, k) in [(0.0f32, 1usize), (0.7, 40), (1.0, 300), (1.3, 1)] {
                let seed = src.next_u64();
                let (mut a, mut b) = (SplitMix64::new(seed), SplitMix64::new(seed));
                let id = pick(&logits, temp, k, &mut a).unwrap();
                let d = pick_traced(&logits, temp, k, &mut b).unwrap();
                assert_eq!((id, a.0), (d.id, b.0), "temp {temp} k {k}");
                assert!((d.logp as f64 - (v[id as usize] as f64 - lse)).abs() < 1e-5);
                if temp <= 0.0 || k == 1 {
                    assert_eq!(d.sample_logp, 0.0);
                } else {
                    let p: Vec<f64> = v.iter().map(|&x| (x as f64 / temp as f64 - mx / temp as f64).exp()).collect();
                    let mut sorted = p.clone();
                    sorted.sort_by(|x, y| y.total_cmp(x));
                    let kept: f64 = sorted[..k].iter().sum();
                    assert!((d.sample_logp as f64 - (p[id as usize] / kept).ln()).abs() < 1e-4, "{} vs {}", d.sample_logp, (p[id as usize] / kept).ln());
                }
            }
        }
    }

    #[test]
    fn traced_replies_are_the_same_replies() {
        use candle_nn::{VarBuilder, VarMap};
        let dev = candle_core::Device::Cpu;
        let cfg = crate::model::Config::with_kind(crate::model::Kind::D, 258, 32, 1, 64);
        let vm = VarMap::new();
        let model = Model::new(cfg, VarBuilder::from_varmap(&vm, candle_core::DType::F32, &dev)).unwrap();
        model.set_track_activity(false);
        model.set_inference(true);
        let bpe = Bpe::from_merges(Vec::new()).unwrap();
        let ids = prompt_ids(&bpe, false, &[], "What is 2 and 3?", 0).0;
        for (temp, k) in [(0.0f32, 1usize), (0.9, 50), (1.0, 258)] {
            let (r1, n1) = reply(&model, &bpe, &ids, 12, temp, k, &mut SplitMix64::new(3)).unwrap();
            let (r2, n2, tr) = reply_traced(&model, &bpe, &ids, 12, temp, k, &mut SplitMix64::new(3)).unwrap();
            assert_eq!((&r1, n1), (&r2, n2));
            // every drawn token: her n tokens, then possibly the EOS that ended the reply
            assert_eq!(tr.ids.len(), tr.logp.len());
            assert!(tr.ids.len() == n1 || (tr.ids.len() == n1 + 1 && tr.ids[n1] == EOS));
            assert!(tr.logp.iter().chain(&tr.sample_logp).all(|&x| x <= 1e-6 && x.is_finite()));
        }
    }

    #[test]
    fn corrections_and_ratings_resolve_to_the_last_one() {
        let ls = vec![
            line(9, Event::Good), // names no turn above it
            line(1, turn("q1", "a1")),
            line(1, Event::Bad),
            line(1, Event::Correct("first".into())),
            line(1, Event::Correct("second".into())),
            line(2, turn("q2", "a2")),
            line(2, Event::Bad),
            line(2, Event::Good),
            line(3, turn("q3", "a3")),
            line(3, Event::Good),
            line(3, Event::Bad),
            line(2, turn("q2 again", "b2")), // a second session reused id 2
            line(2, Event::Correct("fix".into())),
            line(4, turn("q4", "a4")),
        ];
        let (ts, orphans) = resolve("2026-10-28", &ls);
        assert_eq!(orphans, 1);
        assert_eq!(ts.len(), 5);
        assert_eq!((ts[0].correction.as_deref(), ts[0].rating), (Some("second"), Some(false)));
        assert_eq!((ts[1].correction.as_deref(), ts[1].rating), (None, Some(true)));
        assert_eq!((ts[2].correction.as_deref(), ts[2].rating), (None, Some(false)));
        assert_eq!((ts[3].user.as_str(), ts[3].correction.as_deref(), ts[3].rating), ("q2 again", Some("fix"), None));
        assert_eq!((ts[4].correction.as_deref(), ts[4].rating), (None, None));
        assert_eq!(remembered(&ts[0]), Some(("q1".into(), "second".into())));
        assert_eq!(remembered(&ts[1]), Some(("q2".into(), "a2".into())));
        assert_eq!(remembered(&ts[2]), None);
        assert_eq!(remembered(&ts[4]), Some(("q4".into(), "a4".into())));
    }

    #[test]
    fn the_day_keeps_corrections_confirmations_and_heard_words_only() {
        let ls = vec![
            line(1, turn("What is 2 and 3?", "seven")),
            line(1, Event::Correct("5".into())),
            line(2, turn("Why is the  sky\nblue?", "Because of\tlight.")),
            line(2, Event::Good),
            line(3, turn("Name a fish.", "a tree")),
            line(3, Event::Bad),
            line(4, turn("My dog is called Rex.", "blah")),
            line(5, turn("Say nothing", "")),
            line(5, Event::Good),
        ];
        let (ts, _) = resolve("d", &ls);
        let (eps, c) = day_episodes(&ts, Unrated::Heard);
        assert_eq!(
            eps,
            vec![
                "Question: What is 2 and 3?\nAnswer: 5".to_string(),
                "Question: Why is the sky blue?\nAnswer: Because of light.".to_string(),
                "My dog is called Rex.".to_string(),
            ]
        );
        assert_eq!(c, DayCounts { turns: 5, corrected: 1, confirmed: 1, bad_dropped: 1, heard: 1, unrated_dropped: 0, empty: 1 });
        let (eps, c) = day_episodes(&ts, Unrated::Drop);
        assert_eq!((eps.len(), c.heard, c.unrated_dropped), (2, 0, 1));
        // her own wrong replies never reach the day
        assert!(eps.iter().all(|e| !e.contains("seven") && !e.contains("tree") && !e.contains("blah")));
    }

    /// grow_gpu's `episodes()` (design-tests/nn/gpu/src/bin/grow_gpu.rs), verbatim, to check `grow_episodes` copies it.
    fn grow_gpu_episodes(text: &str) -> Vec<String> {
        let text = text.replace("\r\n", "\n");
        let blocks: Vec<String> = text.split("\n\n").map(|b| b.trim().to_string()).filter(|b| !b.is_empty()).collect();
        if blocks.len() > 1 {
            return blocks;
        }
        text.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect()
    }

    #[test]
    fn the_day_file_splits_as_grow_gpu_reads_it() {
        // the rule itself, on sample texts: blocks when there are 2+, else one per line
        for s in ["a\nb\n\nc\n", "a\r\nb\r\n\r\nc", "one\ntwo\nthree\n", "only\n", "\n\n x \n\n\n y \n", ""] {
            assert_eq!(grow_episodes(s), grow_gpu_episodes(s), "{s:?}");
        }
        assert_eq!(grow_gpu_episodes("Q: a\nA: b\n"), vec!["Q: a", "A: b"]); // why a lone episode is doubled
        let qa = |q: &str, a: &str| qa_episode(q, a);
        let cases: Vec<Vec<String>> = vec![
            vec![qa("What is 2 and 3?", "5"), qa("Why  is\nthe sky blue?", "Light\n\nscatters."), "My dog is Rex.".into()],
            vec![qa("Only one?", "yes")],
            vec!["heard one".into()],
            vec!["heard one".into(), "heard two".into()],
            vec![],
        ];
        for eps in cases {
            let text = day_text(&eps);
            let got = grow_gpu_episodes(&text);
            if eps.len() == 1 {
                assert_eq!(got, vec![eps[0].clone(), eps[0].clone()], "{text:?}");
            } else {
                assert_eq!(got, eps, "{text:?}");
            }
            assert!(!text.contains("\n\n\n"));
        }
        assert_eq!(qa("Why  is\nthe sky blue?", " Light\n\nscatters. "), "Question: Why is the sky blue?\nAnswer: Light scatters.");
    }

    #[test]
    fn replies_are_cut_at_a_new_question_and_judged_by_words() {
        assert_eq!(cut_reply(" 5\nQuestion: what is 6?"), ("5".to_string(), true));
        assert_eq!(cut_reply("blue Context: x"), ("blue".to_string(), true));
        assert_eq!(cut_reply("It is blue.\nAnswer: no"), ("It is blue.".to_string(), true));
        assert_eq!(cut_reply("Answer: fine"), ("Answer: fine".to_string(), false));
        assert_eq!(cut_reply("Question: x"), (String::new(), true));
        assert!(says("5.", "5") && says("Five apples", "five") && says("The Sky", "the sky"));
        assert!(!says("50", "5") && !says("fivefold", "five") && !says("anything", "") && !says("", "x"));
    }

    #[test]
    fn names_dates_and_files() {
        assert_eq!(model_name(Path::new("design-tests/nn/gpu/runs/v41/reward/model.safetensors")), "v41/reward/model.safetensors");
        assert_eq!(model_name(Path::new("C:\\Users\\x\\release\\ava\\model.safetensors")), "ava/model.safetensors");
        assert_eq!(model_name(Path::new("model.safetensors")), "model.safetensors");
        assert!(is_day_name("2026-10-28.tsv") && !is_day_name("growth.tsv") && !is_day_name("2026-10-28.tsv.bak") && !is_day_name("2026_10_28.tsv"));
        assert_eq!(day_of(Path::new("data/diary/2026-10-28.tsv")).as_deref(), Some("2026-10-28"));
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_754), (2026, 10, 28));
        assert_eq!(civil_from_days(11_016), (2000, 2, 29));
        let (d, t) = local_now();
        assert!(is_day_name(&format!("{d}.tsv")) && t.starts_with(&d) && t.len() == 19);
    }

    /// v42 (S2): two turns of a conversation with its state saved and loaded between them give exactly the replies of
    /// one unbroken stream; a fresh stream with the same seed gives the first reply again.
    #[test]
    fn a_conversation_resumes_from_its_saved_state() {
        use crate::model::Kind;
        use crate::state::{ConvStore, Loaded};
        let (_, _, m) = crate::testkit::tiny(Kind::D, 6);
        let bpe = crate::testkit::byte_bpe();
        let dir = crate::testkit::scratch("conv_resume");
        let store = ConvStore::next_to(&dir.join("brain.safetensors"));
        let (q1, q2) = (prompt_ids(&bpe, false, &[], "Hello there", 0).0, prompt_ids(&bpe, false, &[], "And again", 0).0);
        // unbroken
        let mut rng = SplitMix64::new(9);
        let mut st = m.stream_batched().unwrap();
        let a1 = reply_traced_in(&m, &bpe, &mut st, &q1, 6, 0.7, 5, &mut rng).unwrap();
        let a2 = reply_traced_in(&m, &bpe, &mut st, &q2, 6, 0.7, 5, &mut rng).unwrap();
        // saved and resumed
        let mut rng = SplitMix64::new(9);
        assert!(matches!(store.load("chat-1", &m, &candle_core::Device::Cpu), Loaded::Fresh));
        let mut st = m.stream_batched().unwrap();
        let b1 = reply_traced_in(&m, &bpe, &mut st, &q1, 6, 0.7, 5, &mut rng).unwrap();
        store.save("chat-1", &m, Model::stream_state(&st).unwrap()).unwrap();
        let Loaded::Resumed(state) = store.load("chat-1", &m, &candle_core::Device::Cpu) else { panic!("not resumed") };
        let mut st = m.stream_from(state).unwrap();
        let b2 = reply_traced_in(&m, &bpe, &mut st, &q2, 6, 0.7, 5, &mut rng).unwrap();
        assert_eq!((a1.0.clone(), a1.2.ids.clone()), (b1.0, b1.2.ids));
        assert_eq!((a2.0, a2.2.ids), (b2.0, b2.2.ids));
        let _ = fs::remove_dir_all(&dir);
    }
}
