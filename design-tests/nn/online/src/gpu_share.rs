// TESTING / DESIGN ONLY
//! G5: sharing the PC with games. The learner only trains when the GPU is free.
//!
//! Two signals, read while the learner itself is NOT using the GPU:
//! - a named program is running (the owner lists games, e.g. `--pause-if-running game.exe`), read
//!   from `tasklist`;
//! - the GPU is busy anyway (utilization from `nvidia-smi`), so something else is using it.
//! A reading that cannot be made counts as busy: the learner never uses a GPU it cannot see.
//! `Pauser` adds hysteresis: one busy reading pauses at once; it resumes only after a run of quiet
//! readings, so a game's loading screen does not let training sneak in. While a training job runs,
//! the learner checks only the program list (its own load would look like "busy") and stops the
//! job if a listed program starts; the half-trained candidate is thrown away, never used.

use std::process::Command;

/// GPU utilization in percent from `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits`
/// output (the highest over all GPUs).
pub fn parse_util(out: &str) -> Option<f32> {
    out.lines().filter_map(|l| l.trim().trim_end_matches('%').trim().parse::<f32>().ok()).reduce(f32::max)
}

/// Lowercase program names from `tasklist /FO CSV /NH` output.
pub fn parse_tasklist(out: &str) -> Vec<String> {
    out.lines()
        .filter_map(|l| {
            let l = l.trim();
            let rest = l.strip_prefix('"')?;
            Some(rest[..rest.find('"')?].to_lowercase())
        })
        .collect()
}

/// The listed programs that are running.
pub fn running_of(names: &[String], watch: &[String]) -> Vec<String> {
    watch.iter().filter(|w| names.iter().any(|n| n == &w.to_lowercase())).cloned().collect()
}

pub fn sample_util() -> Option<f32> {
    let o = Command::new("nvidia-smi").args(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).output().ok()?;
    if !o.status.success() {
        return None;
    }
    parse_util(&String::from_utf8_lossy(&o.stdout))
}

pub fn running_programs() -> Option<Vec<String>> {
    let o = Command::new("tasklist").args(["/FO", "CSV", "/NH"]).output().ok()?;
    if !o.status.success() {
        return None;
    }
    Some(parse_tasklist(&String::from_utf8_lossy(&o.stdout)))
}

#[derive(Clone, Debug)]
pub struct Pauser {
    /// Utilization at or above this (while the learner is idle) means someone else uses the GPU.
    pub busy_util: f32,
    /// Quiet readings in a row needed to resume.
    pub quiet_needed: usize,
    quiet_run: usize,
    pub paused: bool,
    pub why: String,
}

impl Pauser {
    pub fn new(busy_util: f32, quiet_needed: usize) -> Pauser {
        Pauser { busy_util, quiet_needed, quiet_run: 0, paused: true, why: "no reading yet".into() }
    }

    /// One reading taken while the learner is idle. Returns true when training may run.
    pub fn observe(&mut self, util: Option<f32>, games: &[String]) -> bool {
        let busy = if !games.is_empty() {
            Some(format!("{} is running", games.join(", ")))
        } else {
            match util {
                None => Some("cannot read the GPU".to_string()),
                Some(u) if u >= self.busy_util => Some(format!("GPU {u:.0}% busy")),
                _ => None,
            }
        };
        match busy {
            Some(w) => {
                self.paused = true;
                self.quiet_run = 0;
                self.why = w;
            }
            None => {
                self.quiet_run += 1;
                if self.quiet_run >= self.quiet_needed {
                    self.paused = false;
                    self.why.clear();
                }
            }
        }
        !self.paused
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tool_output() {
        assert_eq!(parse_util("37\n"), Some(37.0));
        assert_eq!(parse_util("5 %\n91 %\n"), Some(91.0));
        assert_eq!(parse_util("[N/A]\n"), None);
        let t = "\"System Idle Process\",\"0\",\"Services\",\"0\",\"8 K\"\r\n\"Game.exe\",\"4242\",\"Console\",\"1\",\"3,000 K\"\r\n";
        assert_eq!(parse_tasklist(t), vec!["system idle process", "game.exe"]);
        assert_eq!(running_of(&parse_tasklist(t), &["GAME.exe".into(), "other.exe".into()]), vec!["GAME.exe"]);
    }

    #[test]
    fn pauses_at_once_and_resumes_only_after_a_quiet_run() {
        let mut p = Pauser::new(20.0, 3);
        assert!(!p.observe(Some(2.0), &[]));
        assert!(!p.observe(Some(2.0), &[]));
        assert!(p.observe(Some(2.0), &[]));
        assert!(!p.observe(Some(2.0), &["game.exe".into()]));
        assert!(p.why.contains("game.exe"));
        assert!(!p.observe(Some(3.0), &[]));
        assert!(!p.observe(Some(95.0), &[]));
        assert!(!p.observe(None, &[]));
        assert_eq!(p.why, "cannot read the GPU");
        for _ in 0..2 {
            assert!(!p.observe(Some(1.0), &[]));
        }
        assert!(p.observe(Some(1.0), &[]));
    }
}
