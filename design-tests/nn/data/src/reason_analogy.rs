// TESTING / DESIGN ONLY
//! reason_analogy (D6): relational analogies over letter strings and word pairs, each with the link named.
//!
//! LETTERS: eleven string changes (drop the last letter, double the first, swap the ends, first letter to
//! the end, last letter to the front, the first letter one back in the alphabet, sort the letters, say it
//! twice, add the mirror image, double every letter, drop the vowels) on strings of 3 to 6 letters, shown
//! as "Before: ... After: ..." with one or two examples, then the new string and the change in words.
//! WORDS: fourteen relations (plural, past tense, "more", made of, the sound it makes, a group of them,
//! the female, where they work, whose tool it is, what measures it, a stronger word, the same meaning,
//! what holds it, the person who does it), as pairs, "goes with ... the way ... goes with", a choice
//! among three, or "what links these pairs?".
//!
//! AWAY FROM THE TEST. The D6 test uses five changes (next letter at the end, reverse, double the last,
//! drop the first, upper case) on 4-letter strings with "If X changes to Y, what does Z change to?", and
//! seven word relations (opposite, young, home, part, use, next, kind) with "A is to B as C is to". None
//! of those changes, relations, words (unit test) or frames appear here; a string puzzle that any test
//! change would also explain is dropped.
//! Random letter strings can spell one of the tests' made-up words by chance ("dax", "zup", "nefa"); the frame
//! guard then drops the whole document (4 of 24,000 in the v42 build). With AVOID_TEST_WORDS on (v42_reason
//! --avoid-test-words, v43 on) such a puzzle is drawn again instead. Off by default, so the default output is
//! byte-for-byte the v42 build.

use crate::reason::{pick, range, sample_k};
use crate::reason_guard::{test_change, TEST_CHANGES, WORDS};
use crate::rng::SplitMix64;
use std::sync::atomic::{AtomicBool, Ordering};

/// Redraw a string puzzle that would spell a test's made-up word (see the module note). Off by default.
pub static AVOID_TEST_WORDS: AtomicBool = AtomicBool::new(false);

/// True when `s` is one of the tests' made-up single words (reason_guard::WORDS).
pub fn is_test_word(s: &str) -> bool {
    WORDS.iter().any(|(_, w)| !w.contains(' ') && *w == s)
}

pub const CHANGES: [&str; 11] = ["drop-last", "double-first", "swap-ends", "first-to-end", "last-to-front", "prev-first", "sort", "say-twice", "mirror", "double-each", "drop-vowels"];

/// Apply a change (None: it does not apply, or changes nothing).
pub fn change(c: &str, s: &str) -> Option<String> {
    let v: Vec<char> = s.chars().collect();
    let n = v.len();
    let out: String = match c {
        "drop-last" => v[..n - 1].iter().collect(),
        "double-first" => std::iter::once(v[0]).chain(v.iter().copied()).collect(),
        "swap-ends" => {
            if n < 4 {
                return None;
            }
            let mut w = v.clone();
            w.swap(0, n - 1);
            w.iter().collect()
        }
        "first-to-end" => v[1..].iter().chain(std::iter::once(&v[0])).collect(),
        "last-to-front" => std::iter::once(&v[n - 1]).chain(v[..n - 1].iter()).collect(),
        "prev-first" => {
            if v[0] == 'a' {
                return None;
            }
            std::iter::once((v[0] as u8 - 1) as char).chain(v[1..].iter().copied()).collect()
        }
        "sort" => {
            let mut w = v.clone();
            w.sort_unstable();
            w.iter().collect()
        }
        "say-twice" => {
            if n > 4 {
                return None;
            }
            format!("{s}{s}")
        }
        "mirror" => {
            if n > 4 {
                return None;
            }
            v.iter().chain(v.iter().rev()).collect()
        }
        "double-each" => {
            if n > 4 {
                return None;
            }
            v.iter().flat_map(|&c| [c, c]).collect()
        }
        "drop-vowels" => v.iter().filter(|c| !"aeiou".contains(**c)).collect(),
        _ => return None,
    };
    (out != s && !out.is_empty()).then_some(out)
}

pub fn describe_change(c: &str) -> &'static str {
    match c {
        "drop-last" => "the last letter is dropped",
        "double-first" => "the first letter is written twice",
        "swap-ends" => "the first and last letters trade places",
        "first-to-end" => "the first letter moves to the end",
        "last-to-front" => "the last letter moves to the front",
        "prev-first" => "the first letter becomes the letter before it in the alphabet",
        "sort" => "the letters are put in alphabetical order",
        "say-twice" => "the whole string is written twice",
        "mirror" => "the string is followed by its mirror image",
        "double-each" => "every letter is written twice",
        _ => "the vowels are taken out",
    }
}

/// (relation, the link in words, pairs).
pub const RELATIONS: [(&str, &str, &[(&str, &str)]); 14] = [
    (
        "plural",
        "the second word is the plural of the first",
        &[
            ("mouse", "mice"), ("goose", "geese"), ("child", "children"), ("tooth", "teeth"), ("man", "men"), ("woman", "women"), ("wolf", "wolves"),
            ("box", "boxes"), ("city", "cities"), ("baby", "babies"), ("person", "people"), ("ox", "oxen"), ("cactus", "cacti"), ("half", "halves"),
            ("shelf", "shelves"), ("loaf", "loaves"), ("berry", "berries"), ("glass", "glasses"), ("bus", "buses"), ("dish", "dishes"),
            ("church", "churches"), ("potato", "potatoes"), ("hero", "heroes"), ("thief", "thieves"),
        ],
    ),
    (
        "past",
        "the second word is the past tense of the first: today one, yesterday the other",
        &[
            ("run", "ran"), ("swim", "swam"), ("eat", "ate"), ("go", "went"), ("sing", "sang"), ("think", "thought"), ("catch", "caught"),
            ("bring", "brought"), ("fly", "flew"), ("drink", "drank"), ("ride", "rode"), ("speak", "spoke"), ("take", "took"), ("draw", "drew"),
            ("freeze", "froze"), ("wake", "woke"), ("teach", "taught"), ("buy", "bought"), ("sleep", "slept"), ("begin", "began"),
            ("throw", "threw"), ("grow", "grew"), ("forget", "forgot"), ("give", "gave"),
        ],
    ),
    (
        "more",
        "the second word means more of the first",
        &[
            ("tall", "taller"), ("good", "better"), ("bad", "worse"), ("far", "farther"), ("warm", "warmer"), ("quiet", "quieter"), ("heavy", "heavier"),
            ("early", "earlier"), ("thin", "thinner"), ("wide", "wider"), ("strong", "stronger"), ("deep", "deeper"), ("bright", "brighter"),
            ("soft", "softer"), ("loud", "louder"), ("busy", "busier"), ("funny", "funnier"), ("old", "older"),
        ],
    ),
    (
        "material",
        "the first thing is often made of the second",
        &[
            ("table", "wood"), ("window", "glass"), ("coin", "metal"), ("sweater", "wool"), ("tire", "rubber"), ("candle", "wax"), ("shirt", "cotton"),
            ("brick", "clay"), ("statue", "stone"), ("sail", "canvas"), ("nail", "iron"), ("ring", "gold"), ("scarf", "silk"), ("boot", "leather"),
            ("basket", "straw"), ("bottle", "glass"),
        ],
    ),
    (
        "sound",
        "the second word is the sound the first one makes",
        &[
            ("lion", "roar"), ("duck", "quack"), ("snake", "hiss"), ("owl", "hoot"), ("wolf", "howl"), ("mouse", "squeak"), ("donkey", "bray"),
            ("crow", "caw"), ("goat", "bleat"), ("cricket", "chirp"), ("pigeon", "coo"), ("turkey", "gobble"), ("elephant", "trumpet"),
            ("bull", "bellow"), ("monkey", "chatter"),
        ],
    ),
    (
        "group",
        "a group of the first is called the second",
        &[
            ("wolf", "pack"), ("lion", "pride"), ("goose", "gaggle"), ("deer", "herd"), ("ship", "fleet"), ("singer", "choir"), ("player", "team"),
            ("grape", "bunch"), ("sailor", "crew"), ("actor", "cast"), ("soldier", "army"), ("ant", "colony"), ("musician", "band"), ("whale", "pod"),
        ],
    ),
    (
        "female",
        "the second word is the female of the first",
        &[
            ("king", "queen"), ("prince", "princess"), ("actor", "actress"), ("uncle", "aunt"), ("brother", "sister"), ("nephew", "niece"),
            ("lion", "lioness"), ("waiter", "waitress"), ("husband", "wife"), ("son", "daughter"), ("grandfather", "grandmother"), ("duke", "duchess"),
            ("stallion", "mare"), ("boy", "girl"), ("father", "mother"), ("emperor", "empress"), ("tiger", "tigress"),
        ],
    ),
    (
        "workplace",
        "the first one works in the second",
        &[
            ("teacher", "school"), ("doctor", "hospital"), ("chef", "kitchen"), ("farmer", "farm"), ("pilot", "cockpit"), ("judge", "court"),
            ("librarian", "library"), ("baker", "bakery"), ("miner", "mine"), ("monk", "monastery"), ("scientist", "laboratory"), ("clerk", "office"),
            ("lifeguard", "beach"), ("nurse", "clinic"), ("waiter", "restaurant"),
        ],
    ),
    (
        "tool",
        "the first is a tool of the second",
        &[
            ("scalpel", "surgeon"), ("brush", "painter"), ("wrench", "plumber"), ("saw", "carpenter"), ("stethoscope", "doctor"), ("baton", "conductor"),
            ("camera", "photographer"), ("microscope", "scientist"), ("plough", "farmer"), ("chalk", "teacher"), ("whistle", "referee"),
            ("telescope", "astronomer"), ("anvil", "blacksmith"), ("scissors", "barber"),
        ],
    ),
    (
        "measure",
        "the second is what measures the first",
        &[
            ("time", "clock"), ("weight", "scale"), ("temperature", "thermometer"), ("length", "ruler"), ("speed", "speedometer"), ("angle", "protractor"),
            ("rain", "rain gauge"), ("air pressure", "barometer"),
        ],
    ),
    (
        "stronger",
        "the second word is a stronger form of the first",
        &[
            ("damp", "soaked"), ("tired", "exhausted"), ("glad", "overjoyed"), ("hungry", "starving"), ("breeze", "gale"), ("chilly", "freezing"),
            ("pleased", "delighted"), ("scared", "terrified"), ("annoyed", "furious"), ("pretty", "beautiful"), ("smart", "brilliant"), ("sip", "gulp"),
            ("stroll", "sprint"), ("drizzle", "downpour"), ("whisper", "shout"), ("tap", "bang"),
        ],
    ),
    (
        "same",
        "the two words mean nearly the same",
        &[
            ("begin", "start"), ("end", "finish"), ("quick", "rapid"), ("shut", "close"), ("gift", "present"), ("sick", "ill"), ("rock", "stone"),
            ("under", "below"), ("jump", "leap"), ("talk", "speak"), ("rich", "wealthy"), ("choose", "pick"), ("fix", "repair"), ("huge", "enormous"),
            ("simple", "easy"), ("silent", "quiet"),
        ],
    ),
    (
        "holder",
        "the first is kept in the second",
        &[
            ("letter", "envelope"), ("money", "wallet"), ("soup", "bowl"), ("milk", "jug"), ("arrow", "quiver"), ("coin", "purse"), ("clothes", "wardrobe"),
            ("water", "bucket"), ("tea", "teapot"), ("jewels", "chest"), ("photo", "album"), ("seeds", "packet"), ("paint", "tin"), ("rice", "sack"),
        ],
    ),
    (
        "doer",
        "the second is a person who does the first",
        &[
            ("sing", "singer"), ("teach", "teacher"), ("paint", "painter"), ("run", "runner"), ("swim", "swimmer"), ("drive", "driver"), ("dance", "dancer"),
            ("climb", "climber"), ("build", "builder"), ("garden", "gardener"), ("bake", "baker"), ("read", "reader"), ("skate", "skater"), ("visit", "visitor"),
            ("act", "actor"), ("sail", "sailor"),
        ],
    ),
];

fn letters(r: &mut SplitMix64) -> String {
    let n = range(r, 3, 6) as usize;
    if r.below(2) == 0 {
        let start = r.below(26 - n as u64) as u8;
        (0..n).map(|k| (b'a' + start + k as u8) as char).collect()
    } else {
        (0..n).map(|_| (b'a' + r.below(26) as u8) as char).collect()
    }
}

/// A string puzzle: examples (before, after), the query, its answer, the change. None: try again.
fn string_puzzle(r: &mut SplitMix64) -> Option<(Vec<(String, String)>, String, String, &'static str)> {
    string_puzzle_with(r, AVOID_TEST_WORDS.load(Ordering::Relaxed))
}

/// string_puzzle with the test-word redraw given explicitly (the tests use this, never the global switch).
fn string_puzzle_with(r: &mut SplitMix64, avoid_test_words: bool) -> Option<(Vec<(String, String)>, String, String, &'static str)> {
    let c = *pick(r, &CHANGES);
    let n_ex = 1 + r.below(2) as usize;
    let mut ex = Vec::new();
    for _ in 0..n_ex {
        let s = letters(r);
        ex.push((s.clone(), change(c, &s)?));
    }
    let q = letters(r);
    let a = change(c, &q)?;
    if ex.iter().any(|(s, _)| *s == q) {
        return None;
    }
    // the examples must single out this change among ours...
    if CHANGES.iter().any(|&o| o != c && ex.iter().all(|(s, t)| change(o, s).as_ref() == Some(t))) {
        return None;
    }
    // ...and no test change may explain the puzzle
    if TEST_CHANGES.iter().any(|&t| ex.iter().all(|(s, tt)| test_change(t, s).as_ref() == Some(tt)) || test_change(t, &q).as_ref() == Some(&a)) {
        return None;
    }
    if avoid_test_words && (is_test_word(&q) || is_test_word(&a) || ex.iter().any(|(s, t)| is_test_word(s) || is_test_word(t))) {
        return None;
    }
    Some((ex, q, a, c))
}

fn string_text(r: &mut SplitMix64) -> String {
    let (ex, q, a, c) = loop {
        if let Some(p) = string_puzzle(r) {
            break p;
        }
    };
    match r.below(3) {
        0 => {
            let mut s: Vec<String> = ex.iter().map(|(b, t)| format!("Before: {b}. After: {t}.")).collect();
            s.push(format!("Now before: {q}. After: {a}. The rule: {}.", describe_change(c)));
            s.join("\n")
        }
        1 => {
            let shown: Vec<String> = ex.iter().map(|(b, t)| format!("{b} turned into {t}")).collect();
            format!("{}. Do the same to {q}: {a}, because {}.", crate::reason::cap(&shown.join(", and ")), describe_change(c))
        }
        _ => {
            let (b, t) = &ex[0];
            format!("What happened between {b} and {t}? {}. So {q} turns into {a}.", crate::reason::cap(describe_change(c)))
        }
    }
}

fn word_text(r: &mut SplitMix64) -> String {
    let ri = r.below(RELATIONS.len() as u64) as usize;
    let (_, link, pairs) = RELATIONS[ri];
    let k = sample_k(r, pairs.len(), 3);
    let (p1, p2, p3) = (pairs[k[0]], pairs[k[1]], pairs[k[2]]);
    match r.below(4) {
        0 => format!("Pair: {}, {}. Pair: {}, ? Answer: {}. Link: {link}.", p1.0, p1.1, p2.0, p2.1),
        1 => format!("{} goes with {} the way {} goes with {}, since {link}.", crate::reason::cap(p1.0), p1.1, p2.0, p2.1),
        2 => {
            // two wrong choices: second words of other relations (and of the same first word when there is one)
            let mut wrong: Vec<&str> = Vec::new();
            for (j, (_, _, ps)) in RELATIONS.iter().enumerate() {
                if j != ri {
                    if let Some(p) = ps.iter().find(|p| p.0 == p2.0) {
                        wrong.push(p.1);
                    }
                }
            }
            while wrong.len() < 2 {
                let (_, _, ps) = RELATIONS[(ri + 1 + r.below(RELATIONS.len() as u64 - 1) as usize) % RELATIONS.len()];
                let w = pick(r, ps).1;
                if w != p2.1 && !wrong.contains(&w) {
                    wrong.push(w);
                }
            }
            wrong.truncate(2);
            let mut ch = vec![p2.1, wrong[0], wrong[1]];
            r.shuffle(&mut ch);
            format!("{} goes with {}. Which word goes with {} in the same way: {}? It is {}, because {link}.", crate::reason::cap(p1.0), p1.1, p2.0, ch.join(", "), p2.1)
        }
        _ => format!("What links {} with {}, {} with {}, and {} with {}? In each pair, {link}.", p1.0, p1.1, p2.0, p2.1, p3.0, p3.1),
    }
}

pub fn doc(r: &mut SplitMix64) -> String {
    let n = range(r, 3, 6);
    let mut out = vec![pick(r, &["Analogies.", "Same link, new pair.", "Find the link and use it again.", "Patterns between pairs."]).to_string()];
    for _ in 0..n {
        out.push(if r.below(5) < 3 { string_text(r) } else { word_text(r) });
    }
    out.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reason_guard::{TEST_ANALOGY_WORDS, TEST_RELATIONS};

    #[test]
    fn changes_work() {
        assert_eq!(change("drop-last", "abcd").as_deref(), Some("abc"));
        assert_eq!(change("swap-ends", "abcd").as_deref(), Some("dbca"));
        assert_eq!(change("swap-ends", "abc"), None, "on 3 letters it would be a reversal");
        assert_eq!(change("first-to-end", "abcd").as_deref(), Some("bcda"));
        assert_eq!(change("prev-first", "bcd").as_deref(), Some("acd"));
        assert_eq!(change("prev-first", "abc"), None);
        assert_eq!(change("sort", "abc"), None);
        assert_eq!(change("mirror", "ab c".replace(' ', "").as_str()).as_deref(), Some("abccba"));
        assert_eq!(change("drop-vowels", "bcd"), None);
    }

    /// No change here does what a test change does, on the test's own strings (4 letters from a to y).
    #[test]
    fn no_change_equals_a_test_change() {
        let mut r = SplitMix64::new(11);
        let strs: Vec<String> = (0..300).map(|_| (0..4).map(|_| (b'a' + r.below(25) as u8) as char).collect()).collect();
        for c in CHANGES {
            for t in TEST_CHANGES {
                let same = strs.iter().all(|s| change(c, s) == test_change(t, s));
                assert!(!same, "{c} acts like the test change {t}");
            }
        }
    }

    #[test]
    fn the_redraw_keeps_test_words_out_and_is_off_by_default() {
        assert!(is_test_word("dax") && is_test_word("nefa") && !is_test_word("the bell") && !is_test_word("abc"));
        assert!(!AVOID_TEST_WORDS.load(Ordering::Relaxed));
        let (mut on, mut off) = (SplitMix64::new(7), SplitMix64::new(7));
        let (mut hits_off, mut n_on) = (0, 0);
        for _ in 0..200_000 {
            if let Some((ex, q, a, _)) = string_puzzle_with(&mut on, true) {
                n_on += 1;
                assert!(!is_test_word(&q) && !is_test_word(&a) && ex.iter().all(|(s, t)| !is_test_word(s) && !is_test_word(t)));
            }
            if let Some((ex, q, a, _)) = string_puzzle_with(&mut off, false) {
                hits_off += (is_test_word(&q) || is_test_word(&a) || ex.iter().any(|(s, t)| is_test_word(s) || is_test_word(t))) as usize;
            }
        }
        assert!(n_on > 10_000);
        assert!(hits_off > 0, "the off path should still meet a test word by chance in 200k puzzles");
    }

    #[test]
    fn relations_and_words_avoid_the_test() {
        for (rel, _, pairs) in RELATIONS {
            assert!(!TEST_RELATIONS.contains(&rel));
            for (a, b) in pairs {
                for w in a.split(' ').chain(b.split(' ')) {
                    assert!(!TEST_ANALOGY_WORDS.contains(&w.to_lowercase().as_str()), "{w} is a test word");
                }
            }
        }
    }

    #[test]
    fn puzzles_are_never_explained_by_a_test_change() {
        let mut r = SplitMix64::new(4);
        let mut n = 0;
        for _ in 0..3000 {
            if let Some((ex, q, a, _)) = string_puzzle(&mut r) {
                n += 1;
                for t in TEST_CHANGES {
                    assert!(!(ex.iter().all(|(s, tt)| test_change(t, s).as_ref() == Some(tt))));
                    assert_ne!(test_change(t, &q).as_ref(), Some(&a));
                }
            }
        }
        assert!(n > 500, "{n}");
    }
}
