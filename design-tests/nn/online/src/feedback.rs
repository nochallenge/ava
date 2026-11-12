// TESTING / DESIGN ONLY
//! C3: learning from people's natural feedback in the diary, not only from /good, /bad and /correct.
//!
//! What a person says right after her reply tells how good it was:
//! - thanks or agreement ("thanks", "great", "yes, right") -> her reply is trusted more (0.7, below an
//!   explicit /good at 0.9);
//! - disagreement ("no", "wrong", "that's not right") -> her reply is trusted less (0.1);
//! - disagreement with an answer ("no, it's 5", "wrong, the answer is Paris") -> also a taught
//!   episode with that answer (trust 0.8, below an explicit /correct at 1.0);
//! - asking (nearly) the same question again soon -> her reply probably missed (0.15).
//! Explicit commands always win over these. They are graded trust signals, never bans: nothing she
//! says is blocked, only how much each reply counts when she rehearses.

use crate::content_words;
use crate::diary::Turn;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq)]
pub enum Signal {
    Thanks,
    Disagree,
    /// Disagreement with the answer they gave.
    Corrects(String),
    ReAsk,
}

fn norm(s: &str) -> String {
    s.trim().to_lowercase().replace('’', "'")
}

const THANKS: [&str; 12] = ["thanks", "thank you", "great", "perfect", "correct", "that's right", "yes, right", "well done", "good job", "nice", "exactly", "awesome"];
const NO: [&str; 9] = ["no,", "no.", "no ", "nope", "wrong", "that's wrong", "that's not right", "that is not right", "incorrect"];
const LEADS: [&str; 8] = ["the right answer is ", "the answer is ", "it should be ", "it's ", "it is ", "actually ", "that's ", "it was "];

/// The answer in "no, it's 5" style replies, if there is one (one short phrase).
fn answer_after_no(s: &str) -> Option<String> {
    let t = norm(s);
    let start = NO.iter().filter(|n| t.starts_with(*n)).map(|n| n.len()).max()?;
    let orig: String = s.trim().chars().skip(t[..start].chars().count()).collect();
    let mut rest = orig.replace('’', "'").trim_start_matches([',', '.', '!', ' ']).to_string();
    loop {
        let low = rest.to_lowercase().replace('’', "'");
        match LEADS.iter().find(|l| low.starts_with(*l)) {
            Some(l) => rest = rest[l.len()..].to_string(),
            None => break,
        }
    }
    let rest = rest.trim().trim_end_matches(['.', '!']).trim().to_string();
    let words = rest.split_whitespace().count();
    (words >= 1 && words <= 12 && !rest.ends_with('?')).then_some(rest)
}

/// What the person's next words say about her reply.
pub fn signal(reply_q: &str, next: &str, gap_secs: i64) -> Option<Signal> {
    let t = norm(next);
    if NO.iter().any(|n| t.starts_with(n)) || t == "no" {
        return Some(answer_after_no(next).map_or(Signal::Disagree, Signal::Corrects));
    }
    if THANKS.iter().any(|w| t.starts_with(w)) {
        return Some(Signal::Thanks);
    }
    let (a, b): (BTreeSet<String>, BTreeSet<String>) = (content_words(reply_q).into_iter().collect(), content_words(next).into_iter().collect());
    // A question with other numbers in it is a new question, however alike the words are.
    let nums = |s: &BTreeSet<String>| s.iter().filter(|w| w.chars().any(|c| c.is_ascii_digit())).cloned().collect::<BTreeSet<String>>();
    if gap_secs <= 600 && a.len() >= 2 && nums(&a) == nums(&b) {
        let j = a.intersection(&b).count() as f64 / a.union(&b).count() as f64;
        if j >= 0.6 {
            return Some(Signal::ReAsk);
        }
    }
    None
}

/// For each turn, the signal from the turn after it (same day, in file order).
pub fn signals(turns: &[Turn]) -> Vec<Option<Signal>> {
    let time = |t: &Turn| crate::clock::parse_time(&t.time).unwrap_or(0);
    (0..turns.len())
        .map(|i| turns.get(i + 1).and_then(|n| signal(&turns[i].user, &n.user, time(n) - time(&turns[i]))))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_thanks_disagreement_corrections_and_re_asks() {
        assert_eq!(signal("What is 2 and 3?", "Thanks!", 10), Some(Signal::Thanks));
        assert_eq!(signal("What is 2 and 3?", "No, it's 5.", 10), Some(Signal::Corrects("5".into())));
        assert_eq!(signal("Capital of France?", "Wrong, the answer is Paris", 10), Some(Signal::Corrects("Paris".into())));
        assert_eq!(signal("Capital of France?", "no", 10), Some(Signal::Disagree));
        assert_eq!(signal("Capital of France?", "No. Why did you say that?", 10), Some(Signal::Disagree));
        assert_eq!(signal("What is the capital of France?", "what is the capital of france", 60), Some(Signal::ReAsk));
        assert_eq!(signal("What is the capital of France?", "what is the capital of france", 4000), None);
        assert_eq!(signal("What is 2 and 3?", "Tell me about cats.", 10), None);
        assert_eq!(signal("What is 1 plus 1?", "What is 2 plus 1?", 10), None);
        assert_eq!(signal("What is 1 plus 1?", "what is 1 plus 1", 10), Some(Signal::ReAsk));
        // "Nothing" and "Notice" are not "no".
        assert_eq!(signal("x y", "Nothing else today.", 10), None);
    }
}
