// TESTING / DESIGN ONLY — hotswap (v42 Wave 4, K6): the live server follows the learner's CURRENT file.
//
// The learner (design-tests/nn/online, `ava_learner`) writes ROOT/CURRENT atomically after every promotion or rollback:
// one line, `gen<TAB>ckpt/NNNNNN.safetensors`, the brain's path relative to ROOT (an empty file: no live brain yet).
// `Watcher` reads it every `poll` on a thread of its own. When the text changes it loads the named brain there and
// checks it (the caller's `load`), so the server keeps answering meanwhile. A brain that loaded and passed waits in a
// slot; the server takes it only between replies (`Watcher::take`), so a reply never mixes two brains, and no request
// is dropped (requests queue as before; the swap itself is one move). A brain that fails to load or to pass is logged
// with the reason and the old one keeps serving; the same CURRENT is tried again after `retry` (at most `tries` times in
// all, in case the file was busy) and at once when CURRENT changes again. A rollback is just another change of CURRENT.
// A CURRENT whose path is absolute or leaves ROOT is refused (the learner never writes one).

use crate::model::Model;
use ava_data::bpe::EOS;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// What CURRENT names: the brain's generation and its file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Current {
    pub gen: u64,
    pub path: PathBuf,
}

/// CURRENT's text read: Ok(None) when there is no live brain yet (an empty file).
pub fn parse_current(root: &Path, text: &str) -> Result<Option<Current>, String> {
    let t = text.trim_end_matches(['\n', '\r']);
    if t.trim().is_empty() {
        return Ok(None);
    }
    let (g, p) = t.split_once('\t').ok_or_else(|| format!("CURRENT has no tab: {t:?}"))?;
    let gen = g.trim().parse::<u64>().map_err(|_| format!("CURRENT: bad generation {g:?}"))?;
    let rel = Path::new(p);
    if p.is_empty() || rel.is_absolute() || !rel.components().all(|c| matches!(c, Component::Normal(_))) {
        return Err(format!("CURRENT: the path {p:?} must be relative and stay inside the learner's folder"));
    }
    Ok(Some(Current { gen, path: root.join(rel) }))
}

/// ROOT/CURRENT: (its text, what it names). A missing file reads as empty.
pub fn read_current(root: &Path) -> Result<(String, Option<Current>), String> {
    let p = root.join("CURRENT");
    let text = match std::fs::read_to_string(&p) {
        Ok(s) => s,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(e) => return Err(format!("read {}: {e}", p.display())),
    };
    let cur = parse_current(root, &text)?;
    Ok((text, cur))
}

/// A loaded brain is fit to serve when it has Ava's token-by-token form (`Model::stream`: not T, no block options the
/// stream does not know), its vocabulary is `vocab` (the tokenizer and the glance stay the same), and reading a few
/// tokens gives finite logits.
pub fn check_brain(model: &Model, vocab: Option<usize>) -> Result<(), String> {
    if let Some(v) = vocab {
        if model.cfg.vocab != v {
            return Err(format!("its vocabulary is {}, the live brain's {v} (a new tokenizer needs a restart)", model.cfg.vocab));
        }
    }
    let mut st = model.stream().map_err(|e| format!("it cannot talk token by token: {e}"))?;
    let mut last = None;
    for t in [EOS, b'H' as u32, b'i' as u32, b'.' as u32] {
        last = Some(model.step(&mut st, t).map_err(|e| format!("reading a test prompt: {e}"))?);
    }
    let row: Vec<f32> = last.expect("four tokens").to_dtype(candle_core::DType::F32).and_then(|t| t.to_vec1()).map_err(|e| e.to_string())?;
    if row.iter().any(|x| !x.is_finite()) {
        return Err("its logits are not finite".into());
    }
    Ok(())
}

/// A brain that loaded and passed, waiting to be swapped in.
pub struct Ready<B> {
    pub cur: Current,
    pub brain: B,
    /// seconds the load and the check took
    pub secs: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct WatchOpts {
    pub poll: Duration,
    pub retry: Duration,
    pub tries: usize,
}

impl Default for WatchOpts {
    fn default() -> Self {
        WatchOpts { poll: Duration::from_secs(2), retry: Duration::from_secs(30), tries: 3 }
    }
}

/// Watches ROOT/CURRENT and loads each new brain in the background (see the top of the file).
pub struct Watcher<B> {
    rx: mpsc::Receiver<Ready<B>>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl<B: Send + 'static> Watcher<B> {
    /// `served`: CURRENT's text whose brain the server already holds (it loaded it at start), so it is not loaded twice;
    /// None makes the first reading of CURRENT count as a change. `load` loads and checks a brain; `log` gets one line
    /// per load, failure and bad CURRENT.
    pub fn spawn<L, G>(root: PathBuf, served: Option<String>, o: WatchOpts, load: L, log: G) -> Watcher<B>
    where
        L: Fn(&Current) -> Result<B, String> + Send + 'static,
        G: Fn(&str) + Send + 'static,
    {
        let (tx, rx) = mpsc::channel::<Ready<B>>();
        let stop = Arc::new(AtomicBool::new(false));
        let halt = stop.clone();
        let handle = std::thread::spawn(move || {
            let mut handled = served;
            // the text that failed, how many tries so far, and when the next is due
            let mut failed: Option<(String, usize, Instant)> = None;
            let mut bad_text: Option<String> = None;
            while !halt.load(Ordering::SeqCst) {
                match read_current(&root) {
                    Err(e) => {
                        let text = std::fs::read_to_string(root.join("CURRENT")).unwrap_or_default();
                        if bad_text.as_ref() != Some(&text) {
                            log(&format!("kept the live brain: {e}"));
                            bad_text = Some(text);
                        }
                    }
                    Ok((text, cur)) => {
                        bad_text = None;
                        let due = match &failed {
                            Some((t, n, at)) if *t == text => *n < o.tries && Instant::now() >= *at,
                            _ => handled.as_ref() != Some(&text),
                        };
                        if due {
                            match cur {
                                None => handled = Some(text),
                                Some(c) => {
                                    let t0 = Instant::now();
                                    match load(&c) {
                                        Ok(brain) => {
                                            let secs = t0.elapsed().as_secs_f64();
                                            log(&format!("brain {} ({}) loaded and checked in {secs:.2}s; it goes live before the next reply", c.gen, c.path.display()));
                                            handled = Some(text);
                                            failed = None;
                                            if tx.send(Ready { cur: c, brain, secs }).is_err() {
                                                return; // the server is gone
                                            }
                                        }
                                        Err(why) => {
                                            let n = match &failed {
                                                Some((t, n, _)) if *t == text => n + 1,
                                                _ => 1,
                                            };
                                            let more = n < o.tries;
                                            log(&format!(
                                                "kept the live brain: brain {} ({}) failed, try {n} of {}: {why}{}",
                                                c.gen,
                                                c.path.display(),
                                                o.tries,
                                                if more { format!("; trying again in {:.0}s", o.retry.as_secs_f64()) } else { "; waiting for CURRENT to change".into() }
                                            ));
                                            if !more {
                                                handled = Some(text.clone());
                                            }
                                            failed = Some((text, n, Instant::now() + o.retry));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // sleep in short slices so a stop is quick
                let until = Instant::now() + o.poll;
                while !halt.load(Ordering::SeqCst) && Instant::now() < until {
                    std::thread::sleep(o.poll.min(Duration::from_millis(50)));
                }
            }
        });
        Watcher { rx, stop, handle: Some(handle) }
    }

    /// Call only between replies: the newest brain that is ready, if any (older ready ones are dropped).
    pub fn take(&self) -> Option<Ready<B>> {
        let mut last = None;
        while let Ok(r) = self.rx.try_recv() {
            last = Some(r);
        }
        last
    }
}

impl<B> Drop for Watcher<B> {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Kind;
    use crate::testkit::{save, scratch, tiny, tiny_with};
    use candle_core::Device;
    use std::sync::Mutex;

    #[test]
    fn current_is_read_as_the_learner_writes_it() {
        let root = Path::new("learner");
        assert_eq!(parse_current(root, "5\tckpt/000005.safetensors\n").unwrap(), Some(Current { gen: 5, path: root.join("ckpt/000005.safetensors") }));
        assert_eq!(parse_current(root, "").unwrap(), None);
        assert_eq!(parse_current(root, "\n").unwrap(), None);
        assert!(parse_current(root, "junk").is_err());
        assert!(parse_current(root, "x\tckpt/a.safetensors").is_err());
        assert!(parse_current(root, "3\t../elsewhere/a.safetensors").is_err());
        assert!(parse_current(root, "3\t/abs/a.safetensors").is_err());
        let d = scratch("hotswap_read");
        assert_eq!(read_current(&d).unwrap(), (String::new(), None));
    }

    /// The fingerprint of a brain: its logits after reading a fixed prompt.
    fn answer(m: &Model) -> Vec<f32> {
        let mut st = m.stream().unwrap();
        let mut last = None;
        for t in [EOS, 72, 105] {
            last = Some(m.step(&mut st, t).unwrap());
        }
        last.unwrap().to_vec1().unwrap()
    }

    fn write_current(root: &Path, gen: u64) {
        // as the learner does: a temporary file, then a rename
        let tmp = root.join("CURRENT.tmp");
        std::fs::write(&tmp, format!("{gen}\tckpt/{gen:06}.safetensors\n")).unwrap();
        std::fs::rename(&tmp, root.join("CURRENT")).unwrap();
    }

    fn wait_for<B: Send + 'static>(w: &Watcher<B>, secs: f64) -> Option<Ready<B>> {
        let t0 = Instant::now();
        while t0.elapsed().as_secs_f64() < secs {
            if let Some(r) = w.take() {
                return Some(r);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        None
    }

    fn wait_log(log: &Mutex<Vec<String>>, needle: &str, secs: f64) -> bool {
        let t0 = Instant::now();
        while t0.elapsed().as_secs_f64() < secs {
            if log.lock().unwrap().iter().any(|l| l.contains(needle)) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        false
    }

    #[test]
    fn swaps_between_replies_only_and_keeps_the_old_brain_on_a_bad_file() {
        let root = scratch("hotswap_live");
        std::fs::create_dir_all(root.join("ckpt")).unwrap();
        let (_c1, vm1, m1) = tiny(Kind::D, 1);
        let (_c2, vm2, m2) = tiny(Kind::D, 2);
        save(&vm1, &root.join("ckpt/000001.safetensors"));
        save(&vm2, &root.join("ckpt/000002.safetensors"));
        let (a1, a2) = (answer(&m1), answer(&m2));
        assert_ne!(a1, a2);
        write_current(&root, 1);

        // the server starts on CURRENT's brain, as ava_live does
        let (text, cur) = read_current(&root).unwrap();
        let cur = cur.unwrap();
        let (_, _, first) = crate::ckpt::load_auto(&cur.path, Kind::D, 64, &Device::Cpu).unwrap();
        let mut live = (cur.gen, first);
        let log = Arc::new(Mutex::new(Vec::<String>::new()));
        let lg = log.clone();
        let o = WatchOpts { poll: Duration::from_millis(20), retry: Duration::from_millis(100), tries: 2 };
        let w: Watcher<Model> = Watcher::spawn(
            root.clone(),
            Some(text),
            o,
            |c: &Current| {
                let (_, _, m) = crate::ckpt::load_auto(&c.path, Kind::D, 64, &Device::Cpu)?;
                check_brain(&m, Some(258))?;
                Ok(m)
            },
            move |l: &str| lg.lock().unwrap().push(l.to_string()),
        );
        std::thread::sleep(Duration::from_millis(100));
        assert!(w.take().is_none(), "the brain already served is not loaded again");

        // a reply is under way when the learner promotes brain 2: the whole reply stays on brain 1
        write_current(&root, 2);
        let mut reply = Vec::new();
        for _ in 0..5 {
            reply.push(answer(&live.1));
            std::thread::sleep(Duration::from_millis(40)); // the watcher loads meanwhile
        }
        assert!(reply.iter().all(|r| *r == a1));
        // between replies: the swap
        let r = wait_for(&w, 10.0).expect("brain 2 becomes ready");
        assert_eq!(r.cur.gen, 2);
        live = (r.cur.gen, r.brain);
        assert_eq!(answer(&live.1), a2);

        // a broken brain 3: logged, retried, and brain 2 keeps serving
        std::fs::write(root.join("ckpt/000003.safetensors"), b"not a brain").unwrap();
        write_current(&root, 3);
        assert!(wait_log(&log, "try 2 of 2", 10.0), "{:?}", log.lock().unwrap());
        assert!(w.take().is_none());
        assert_eq!(live.0, 2);
        assert_eq!(answer(&live.1), a2);
        assert!(log.lock().unwrap().iter().any(|l| l.contains("kept the live brain: brain 3") && l.contains("waiting for CURRENT to change")));

        // a brain that loads but cannot serve (another vocabulary) is refused the same way
        let (_c4, vm4, _m4) = tiny_with(Kind::D, 300, 32, 1, 4);
        save(&vm4, &root.join("ckpt/000004.safetensors"));
        write_current(&root, 4);
        assert!(wait_log(&log, "kept the live brain: brain 4", 10.0));
        assert!(w.take().is_none());

        // a rollback to brain 1 is one more change of CURRENT
        write_current(&root, 1);
        let r = wait_for(&w, 10.0).expect("rollback to brain 1");
        assert_eq!(r.cur.gen, 1);
        assert_eq!(answer(&r.brain), a1);

        // a bad CURRENT is logged once and changes nothing
        std::fs::write(root.join("CURRENT"), "9\t../../outside.safetensors\n").unwrap();
        assert!(wait_log(&log, "stay inside", 10.0));
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(log.lock().unwrap().iter().filter(|l| l.contains("stay inside")).count(), 1);
        assert!(w.take().is_none());
    }

    #[test]
    fn several_changes_while_busy_give_the_newest_brain() {
        let root = scratch("hotswap_newest");
        std::fs::create_dir_all(root.join("ckpt")).unwrap();
        let mut answers = Vec::new();
        for g in 1..=3u64 {
            let (_c, vm, m) = tiny(Kind::D, 10 + g);
            save(&vm, &root.join(format!("ckpt/{g:06}.safetensors")));
            answers.push(answer(&m));
        }
        let o = WatchOpts { poll: Duration::from_millis(10), retry: Duration::from_secs(60), tries: 3 };
        let w: Watcher<Model> = Watcher::spawn(root.clone(), None, o, |c: &Current| Ok(crate::ckpt::load_auto(&c.path, Kind::D, 64, &Device::Cpu)?.2), |_: &str| {});
        for g in 1..=3u64 {
            write_current(&root, g);
            std::thread::sleep(Duration::from_millis(150));
        }
        std::thread::sleep(Duration::from_millis(300));
        // the server was busy all along: it takes only the newest
        let r = w.take().expect("a brain is ready");
        assert_eq!(r.cur.gen, 3);
        assert_eq!(answer(&r.brain), answers[2]);
        assert!(w.take().is_none());
    }
}
