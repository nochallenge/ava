// TESTING / DESIGN ONLY — the process's CPU thread settings (RAYON_NUM_THREADS, CANDLE_NUM_THREADS), changed only for
// a while and put back. candle reads RAYON_NUM_THREADS at every CPU op (candle_core::utils::get_num_threads), so a
// change is seen by every thread of the process at once. In a tool that is what --threads means; in the test suite
// (many tests in one process) a change used to leak into other tests: an exact comparison of two runs could see the
// count change between them (fast4's e-prop session test, arch::tests::rwkv's round trip, 2026-11-06).
// * `scoped` / `threads`: set the variables, put back what was there (or remove them) when the guard drops, also on a
//   panic; while any change is in force it holds the write side of one lock.
// * `steady`: the read side, for code (tests) whose numbers must see one thread count from start to end.
// The same thread may nest changes (the SAE tools set --threads, then collect_par sets 1 inside); a thread that holds
// `steady` and asks for a change gets a panic with a message, not a deadlock.
// * In the test build the thread settings themselves are never changed (`THREAD_VARS`): the suite runs every test in
//   one process at the count it was started with (RAYON_NUM_THREADS=1 by convention). A change for a while was not
//   enough: some exact checks (merge_ident's and v51 golden's gradient hashes, 2026-11-06) also broke when the count
//   changed and changed back during their run. The lock, the nesting and the putting back still run (and are tested
//   on other variable names); the tools (non-test build) change the settings as before.

use std::cell::Cell;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

static LOCK: RwLock<()> = RwLock::new(());

/// The process's thread settings: candle reads RAYON_NUM_THREADS at every CPU op, Ava's int8 kernels CANDLE_NUM_THREADS.
const THREAD_VARS: [&str; 2] = ["RAYON_NUM_THREADS", "CANDLE_NUM_THREADS"];

/// Whether a change to `k` is made: always in the tools; never to the thread settings in the test build.
fn applies(k: &str) -> bool {
    !(cfg!(test) && THREAD_VARS.contains(&k))
}

thread_local! {
    /// How many `scoped` guards this thread holds (the write lock is taken by the first).
    static WRITES: Cell<usize> = const { Cell::new(0) };
    /// How many `steady` guards this thread holds.
    static READS: Cell<usize> = const { Cell::new(0) };
}

/// A change to the thread settings in force; dropping it puts back what was there.
pub struct Scoped {
    prev: Vec<(&'static str, Option<String>)>,
    _lock: Option<RwLockWriteGuard<'static, ()>>,
}

/// Sets each (name, value) until the guard drops. Waits while any `steady` reader or another thread's change is in force.
pub fn scoped(vars: &[(&'static str, String)]) -> Scoped {
    assert!(
        READS.get() == 0,
        "thread_env::scoped inside thread_env::steady on the same thread: the thread settings would change under a reader that asked for them not to"
    );
    let lock = if WRITES.get() == 0 { Some(LOCK.write().unwrap_or_else(|e| e.into_inner())) } else { None };
    WRITES.set(WRITES.get() + 1);
    let vars: Vec<&(&'static str, String)> = vars.iter().filter(|(k, _)| applies(k)).collect();
    let prev = vars.iter().map(|(k, _)| (*k, std::env::var(k).ok())).collect();
    for (k, v) in vars {
        std::env::set_var(k, v);
    }
    Scoped { prev, _lock: lock }
}

/// `--threads N` for as long as the guard lives: RAYON_NUM_THREADS and CANDLE_NUM_THREADS = N (N = 0 changes nothing,
/// as diary::set_threads).
pub fn threads(n: usize) -> Scoped {
    if n == 0 {
        return scoped(&[]);
    }
    scoped(&[("RAYON_NUM_THREADS", n.to_string()), ("CANDLE_NUM_THREADS", n.to_string())])
}

impl Drop for Scoped {
    fn drop(&mut self) {
        for (k, v) in self.prev.iter().rev() {
            match v {
                Some(v) => std::env::set_var(k, v),
                None => std::env::remove_var(k),
            }
        }
        WRITES.set(WRITES.get() - 1);
        // the write lock (if this guard took it) is released after the variables are back
    }
}

/// While this lives, no `scoped` change can start on another thread (one already in force finishes first).
pub struct Steady {
    _lock: Option<RwLockReadGuard<'static, ()>>,
}

/// The read side: the thread settings stay as they are until the guard drops. On a thread that holds a change itself,
/// nothing is taken (its own change is the one in force).
pub fn steady() -> Steady {
    let lock = if WRITES.get() == 0 { Some(LOCK.read().unwrap_or_else(|e| e.into_inner())) } else { None };
    READS.set(READS.get() + 1);
    Steady { _lock: lock }
}

impl Drop for Steady {
    fn drop(&mut self) {
        READS.set(READS.get() - 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const K: &str = "AVA_THREAD_ENV_TEST_VAR";

    #[test]
    fn a_change_is_put_back_nested_and_on_a_panic() {
        let _s = steady(); // nobody else changes the settings meanwhile (this test's own changes are on this thread)
        drop(_s);
        assert!(std::env::var(K).is_err());
        {
            let _a = scoped(&[(K, "1".into())]);
            assert_eq!(std::env::var(K).unwrap(), "1");
            {
                let _b = scoped(&[(K, "2".into())]);
                assert_eq!(std::env::var(K).unwrap(), "2");
            }
            assert_eq!(std::env::var(K).unwrap(), "1", "the inner change put back the outer one");
        }
        assert!(std::env::var(K).is_err(), "the outer change removed what was not there");
        let r = std::panic::catch_unwind(|| {
            let _a = scoped(&[(K, "3".into())]);
            panic!("inside a change");
        });
        assert!(r.is_err());
        assert!(std::env::var(K).is_err(), "a panic still puts the settings back");
        assert_eq!(WRITES.get(), 0);
    }

    #[test]
    fn a_change_waits_for_a_steady_reader_on_another_thread() {
        let s = steady();
        let started = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (st, dn) = (started.clone(), done.clone());
        let h = std::thread::spawn(move || {
            st.store(true, std::sync::atomic::Ordering::SeqCst);
            let _c = scoped(&[("AVA_THREAD_ENV_TEST_VAR2", "x".into())]);
            dn.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        while !started.load(std::sync::atomic::Ordering::SeqCst) {
            std::thread::yield_now();
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(!done.load(std::sync::atomic::Ordering::SeqCst), "the change started while a reader held steady");
        drop(s);
        h.join().unwrap();
        assert!(done.load(std::sync::atomic::Ordering::SeqCst));
        assert!(std::env::var("AVA_THREAD_ENV_TEST_VAR2").is_err());
    }

    #[test]
    fn the_test_build_never_changes_the_thread_settings() {
        let before = (std::env::var("RAYON_NUM_THREADS").ok(), std::env::var("CANDLE_NUM_THREADS").ok());
        {
            let _t = threads(if before.0.as_deref() == Some("3") { 5 } else { 3 });
            let _o = scoped(&[("RAYON_NUM_THREADS", "7".into()), ("AVA_THREAD_ENV_TEST_VAR4", "z".into())]);
            assert_eq!((std::env::var("RAYON_NUM_THREADS").ok(), std::env::var("CANDLE_NUM_THREADS").ok()), before);
            assert_eq!(std::env::var("AVA_THREAD_ENV_TEST_VAR4").unwrap(), "z", "other variables still change");
        }
        assert!(std::env::var("AVA_THREAD_ENV_TEST_VAR4").is_err());
    }

    #[test]
    fn steady_inside_a_change_on_the_same_thread_does_not_deadlock() {
        let _c = scoped(&[("AVA_THREAD_ENV_TEST_VAR3", "y".into())]);
        let _s = steady();
        assert_eq!(std::env::var("AVA_THREAD_ENV_TEST_VAR3").unwrap(), "y");
    }
}
