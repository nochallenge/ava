// TESTING / DESIGN ONLY
//! tags: graded source and toxicity tags for v42's documents (V6 + C2). Tags only, never a ban.
//!
//! ORIGIN. Where a document's text came from: book, dialogue, sentence, qa, practice, diary, web, wiki,
//! speech (transcribed talk). `KNOWN_SOURCES` gives each v42 source (BabyLM's six parts by their own names) its origin. The
//! model may later read it as a conditional prefix; for now it is stored next to the shards.
//!
//! TOXICITY. A small built-in lexicon of strong profanity and slurs, each with a severity:
//!   1 coarse word, 2 strong profanity, 3 slur.
//! The words are stored ROT13-encoded below so this file never prints them; `ToxLexicon::builtin`
//! decodes them once at start. Words that are also ordinary words in science or geography text
//! (an embankment, a verb for slowing, pricking a finger, a crack or a clink, ...) are left out on
//! purpose, so the science corpus and old books are not flagged for their normal vocabulary; the
//! v42 audit found the crack/clink word in 41 of the 337 training books (old literature, where the
//! ordinary sense is the usual one).
//!
//! MATCHING. Text is cut into runs of ASCII letters, lowercased. A word matches when it is a lexicon
//! word, or (severity 1-2 only) when it is one after removing one ending (-ers -er -ing -in -ed -es
//! -s -y, and a doubled last letter: "shitty" -> "shitt" -> "shit"). A slur (severity 3) matches only
//! as itself or with a plural -s, so ordinary longer words that start with the same letters do not.
//!
//! SCORE (soft, per document): `level` = the highest severity found (0 = none); `weight` = the sum of
//! 1 / 2 / 4 per hit of severity 1 / 2 / 3; `per_mille` = 1000 * weight / words, capped at 999 (a
//! density, so one coarse word in a long book chapter scores near 0 and a short insult scores high).

use crate::hash::{new_map, FastMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Origin {
    Book,
    Dialogue,
    Sentence,
    Qa,
    Practice,
    Diary,
    /// Web pages (FineWeb-Edu).
    Web,
    /// Encyclopedia articles (Simple English Wikipedia).
    Wiki,
    /// Transcribed speech (CHILDES, the BNC's spoken part, Switchboard).
    Speech,
}

/// Every origin; an origin's id is its place here (new origins are only ever added at the end).
pub const ORIGINS: [Origin; 9] =
    [Origin::Book, Origin::Dialogue, Origin::Sentence, Origin::Qa, Origin::Practice, Origin::Diary, Origin::Web, Origin::Wiki, Origin::Speech];

/// v42's sources by name, with their origin. BabyLM's six parts keep BabyLM's own names.
pub const KNOWN_SOURCES: [(&str, Origin); 15] = [
    ("fineweb", Origin::Web),
    ("simplewiki", Origin::Wiki),
    ("arc", Origin::Sentence),
    ("tatoeba", Origin::Sentence),
    ("school", Origin::Dialogue),
    ("cornell", Origin::Dialogue),
    ("dailydialog", Origin::Dialogue),
    ("books", Origin::Book),
    ("facts", Origin::Practice),
    ("childes", Origin::Speech),
    ("bnc_spoken", Origin::Speech),
    ("switchboard", Origin::Speech),
    ("open_subtitles", Origin::Dialogue),
    ("gutenberg", Origin::Book),
    ("simple_wiki", Origin::Wiki),
];

/// The origin of a known source name.
pub fn known_origin(name: &str) -> Option<Origin> {
    KNOWN_SOURCES.iter().find(|(n, _)| *n == name).map(|&(_, o)| o)
}

impl Origin {
    pub fn name(self) -> &'static str {
        match self {
            Origin::Book => "book",
            Origin::Dialogue => "dialogue",
            Origin::Sentence => "sentence",
            Origin::Qa => "qa",
            Origin::Practice => "practice",
            Origin::Diary => "diary",
            Origin::Web => "web",
            Origin::Wiki => "wiki",
            Origin::Speech => "speech",
        }
    }

    pub fn id(self) -> u8 {
        ORIGINS.iter().position(|&o| o == self).unwrap() as u8
    }

    pub fn parse(s: &str) -> Result<Origin, String> {
        ORIGINS.iter().copied().find(|o| o.name() == s).ok_or_else(|| format!("unknown origin {s:?} (book, dialogue, sentence, qa, practice, diary, web, wiki, speech)"))
    }
}

/// (ROT13 word, severity). Decode with `rot13` to audit.
const LEXICON_ROT13: [(&str, u8); 39] = [
    ("shpx", 2),
    ("zbgureshpx", 2),
    ("zbgureshpxre", 2),
    ("phag", 2),
    ("pbpxfhpxre", 2),
    ("nffubyr", 2),
    ("juber", 2),
    ("fyhg", 2),
    ("gjng", 2),
    ("jnaxre", 2),
    ("qvpxurnq", 2),
    ("fxnax", 2),
    ("ovgpu", 2),
    ("fuvg", 1),
    ("ohyyfuvg", 1),
    ("onfgneq", 1),
    ("cvff", 1),
    ("tbqqnza", 1),
    ("wnpxnff", 1),
    ("qbhpuront", 1),
    ("penc", 1),
    ("avttre", 3),
    ("avttn", 3),
    ("snttbg", 3),
    ("snt", 3),
    ("xvxr", 3),
    ("fcvp", 3),
    ("tbbx", 3),
    ("jrgonpx", 3),
    ("genaal", 3),
    ("cnxv", 3),
    ("enturnq", 3),
    ("gbjryurnq", 3),
    ("ornare", 3),
    ("wnc", 3),
    ("ubaxl", 3),
    ("qnexvr", 3),
    ("fnzob", 3),
    ("furznyr", 3),
];

/// Endings removed (one at a time) when matching severity 1-2 words.
const ENDINGS: [&str; 8] = ["ers", "er", "ing", "in", "ed", "es", "s", "y"];

pub fn rot13(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'a'..='m' | 'A'..='M' => (c as u8 + 13) as char,
            'n'..='z' | 'N'..='Z' => (c as u8 - 13) as char,
            _ => c,
        })
        .collect()
}

fn word_key(w: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in w {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    crate::rng::mix64(h)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ToxScore {
    pub words: u32,
    pub hits: u32,
    pub weight: u32,
    pub level: u8,
}

impl ToxScore {
    /// Weighted hits per 1000 words, capped at 999.
    pub fn per_mille(&self) -> u32 {
        if self.weight == 0 {
            return 0;
        }
        ((1000.0 * self.weight as f64 / self.words.max(1) as f64).round() as u32).min(999)
    }
}

pub struct ToxLexicon {
    map: FastMap<u64, u8>,
}

impl ToxLexicon {
    /// The built-in lexicon (decoded from ROT13).
    pub fn builtin() -> ToxLexicon {
        let words: Vec<(String, u8)> = LEXICON_ROT13.iter().map(|&(w, s)| (rot13(w), s)).collect();
        ToxLexicon::from_words(words.iter().map(|(w, s)| (w.as_str(), *s)))
    }

    /// A lexicon from plain (lowercase) words, for tests.
    pub fn from_words<'a>(words: impl IntoIterator<Item = (&'a str, u8)>) -> ToxLexicon {
        let mut map = new_map();
        for (w, s) in words {
            map.insert(word_key(w.as_bytes()), s);
        }
        ToxLexicon { map }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    fn exact(&self, w: &[u8]) -> u8 {
        self.map.get(&word_key(w)).copied().unwrap_or(0)
    }

    /// Severity of one lowercase ASCII word (0 = not in the lexicon).
    pub fn severity(&self, w: &[u8]) -> u8 {
        let s = self.exact(w);
        if s > 0 {
            return s;
        }
        // a slur with a plural -s
        if w.len() > 3 && w.ends_with(b"s") {
            let s = self.exact(&w[..w.len() - 1]);
            if s == 3 {
                return 3;
            }
        }
        for end in ENDINGS {
            let e = end.as_bytes();
            if w.len() > e.len() + 2 && w.ends_with(e) {
                let stem = &w[..w.len() - e.len()];
                let s = self.exact(stem);
                if s > 0 && s < 3 {
                    return s;
                }
                let n = stem.len();
                if n >= 2 && stem[n - 1] == stem[n - 2] {
                    let s = self.exact(&stem[..n - 1]);
                    if s > 0 && s < 3 {
                        return s;
                    }
                }
            }
        }
        0
    }

    /// The soft score of a text.
    pub fn score(&self, text: &str) -> ToxScore {
        let mut sc = ToxScore::default();
        let mut w: Vec<u8> = Vec::with_capacity(32);
        let flush = |w: &mut Vec<u8>, sc: &mut ToxScore| {
            if w.is_empty() {
                return;
            }
            sc.words += 1;
            let s = self.severity(w);
            if s > 0 {
                sc.hits += 1;
                sc.weight += [0, 1, 2, 4][s as usize];
                sc.level = sc.level.max(s);
            }
            w.clear();
        };
        for &b in text.as_bytes() {
            if b.is_ascii_alphabetic() {
                w.push(b.to_ascii_lowercase());
            } else {
                flush(&mut w, &mut sc);
            }
        }
        flush(&mut w, &mut sc);
        sc
    }
}

/// Per-source tag counts.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TagStats {
    pub docs: u64,
    pub words: u64,
    pub hits: u64,
    /// Documents by level 0..=3.
    pub level_docs: [u64; 4],
    pub per_mille_sum: u64,
}

impl TagStats {
    pub fn add(&mut self, s: &ToxScore) {
        self.docs += 1;
        self.words += s.words as u64;
        self.hits += s.hits as u64;
        self.level_docs[s.level as usize] += 1;
        self.per_mille_sum += s.per_mille() as u64;
    }

    /// Share of documents with level >= `l`, in percent.
    pub fn share_at_least(&self, l: u8) -> f64 {
        let n: u64 = self.level_docs[l as usize..].iter().sum();
        100.0 * n as f64 / self.docs.max(1) as f64
    }

    pub fn line(&self) -> String {
        format!(
            "{} docs  any {:.3}%  strong+ {:.3}%  slur {:.3}%  hits per 10k words {:.2}  mean per-mille {:.2}",
            self.docs,
            self.share_at_least(1),
            self.share_at_least(2),
            self.share_at_least(3),
            1e4 * self.hits as f64 / self.words.max(1) as f64,
            self.per_mille_sum as f64 / self.docs.max(1) as f64
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex() -> ToxLexicon {
        // stand-in words: the mechanism is the same for the real list
        ToxLexicon::from_words([("zonk", 2), ("blip", 1), ("grut", 3)])
    }

    #[test]
    fn origins_round_trip() {
        for o in ORIGINS {
            assert_eq!(Origin::parse(o.name()).unwrap(), o);
            assert_eq!(ORIGINS[o.id() as usize], o);
        }
        assert!(Origin::parse("radio").is_err());
        // the first six keep their old ids (tag files written before v42's new origins still read the same)
        assert_eq!((Origin::Book.id(), Origin::Diary.id(), Origin::Web.id(), Origin::Speech.id()), (0, 5, 6, 8));
    }

    #[test]
    fn known_sources_are_unique_and_named() {
        let mut names: Vec<&str> = KNOWN_SOURCES.iter().map(|s| s.0).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), KNOWN_SOURCES.len());
        assert_eq!(known_origin("childes"), Some(Origin::Speech));
        assert_eq!(known_origin("fineweb"), Some(Origin::Web));
        assert_eq!(known_origin("gutenberg"), Some(Origin::Book));
        assert_eq!(known_origin("nope"), None);
    }

    #[test]
    fn rot13_is_its_own_inverse() {
        assert_eq!(rot13("Hello, World"), "Uryyb, Jbeyq");
        assert_eq!(rot13(&rot13("abcxyz")), "abcxyz");
    }

    #[test]
    fn endings_and_doubling() {
        let l = lex();
        assert_eq!(l.severity(b"zonk"), 2);
        assert_eq!(l.severity(b"zonking"), 2);
        assert_eq!(l.severity(b"zonkers"), 2);
        assert_eq!(l.severity(b"blippy"), 1);
        assert_eq!(l.severity(b"blipped"), 1);
        assert_eq!(l.severity(b"grut"), 3);
        assert_eq!(l.severity(b"gruts"), 3);
        // slurs never match through other endings
        assert_eq!(l.severity(b"gruty"), 0);
        assert_eq!(l.severity(b"gruter"), 0);
        assert_eq!(l.severity(b"zon"), 0);
        assert_eq!(l.severity(b"apple"), 0);
    }

    #[test]
    fn score_is_a_density_and_level_is_the_max() {
        let l = lex();
        let s = l.score("Zonk, you BLIP! A grut.");
        assert_eq!(s.words, 5);
        assert_eq!(s.hits, 3);
        assert_eq!(s.weight, 2 + 1 + 4);
        assert_eq!(s.level, 3);
        assert_eq!(s.per_mille(), 999);
        let long = format!("{} blip", "word ".repeat(999));
        let s = l.score(&long);
        assert_eq!((s.words, s.level, s.per_mille()), (1000, 1, 1));
        assert_eq!(l.score("nothing here at all").per_mille(), 0);
    }

    #[test]
    fn builtin_lexicon_loads_and_matches() {
        let l = ToxLexicon::builtin();
        assert_eq!(l.len(), LEXICON_ROT13.len());
        // one strong word, given in ROT13 so this file does not print it
        let w = rot13("shpxvat");
        assert_eq!(l.severity(w.as_bytes()), 2);
        // ordinary science words stay clean
        for w in ["spicy", "dykes", "retarded", "pricked", "cockpit", "assume", "scrap", "therapist", "japan", "classic"] {
            assert_eq!(l.severity(w.as_bytes()), 0, "{w}");
        }
    }

    #[test]
    fn stats_rates() {
        let l = lex();
        let mut st = TagStats::default();
        for t in ["clean", "a zonk", "grut", "fine"] {
            st.add(&l.score(t));
        }
        assert_eq!(st.level_docs, [2, 0, 1, 1]);
        assert!((st.share_at_least(1) - 50.0).abs() < 1e-9);
        assert!((st.share_at_least(3) - 25.0).abs() < 1e-9);
    }
}
