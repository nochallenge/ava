//! TESTING / DESIGN ONLY — recall: v41's recall practice (made-up notes planted in her reading, asked about later).
//!
//! THE ROW. `recall_row(rng, bpe, filler, len)` takes a window of real reading text (marked tokens, ids < 4112,
//! at least `len + 1` of them) and returns EXACTLY `len + 1` marked tokens: the filler with notes, cues and
//! (half the time) one re-read inserted at word boundaries, cut to `len + 1`. The trainer reads it as an ordinary
//! row (next-token loss everywhere), so the lesson is in the text: a note appears, ordinary text goes on, later
//! the cue appears and the answer follows; predicting the answer needs looking back.
//!
//!   note : " {lead}{body} {value}{tail}."     e.g. " By the way, the password for the attic is cloavin."
//!   cue  : " {lead}{body}" + " {value}{tail}." e.g. " Remember, the password for the attic is" + " cloavin."
//!
//! Per row: 1–4 notes (distinct subjects; a note repeats the previous note's frame with another subject 35% of
//! the time, so she must bind subject to value, not copy the last value), each cue `g` tokens after its note,
//! `g` log-uniform in [MIN_GAP = 32, 0.9·len] (capped by the room left); and with probability 1/2 a RE-READ: an
//! exact copy of a 32–128-token span of the earlier filler (whole words) inserted later (copying / induction).
//! Inserted text goes before a token that starts a sentence (a word start after `.`, `!` or `?`) when one is
//! within 128 tokens (a cue: within max(16, g/2), so short gaps stay short), else before any word start within 64, else at the next place that splits no number from its
//! length mark. No insertion falls inside the re-read's source span, so the copy is exact in the output too.
//! All insertions sit at filler positions <= F = len + 1 − (inserted tokens), so every note, cue and answer is
//! wholly inside the row. Numbers in notes are glance-marked (`numbers::insert_marks` after encoding).
//!
//! FRAMES (12): favourite, room, kept, lives, password, farm, leaves, ticket, called, hid, meet, age — see
//! `make_fact`. None uses the v41 test frames: never "The code for the …" (fact test), never "… was born in …" or
//! "… worked as …" (sleep test); a unit test checks the decoded insertions.
//!
//! NAMES. Every made-up name word, town and password, and every first name used, is rejected when sleep_facts'
//! syllable grammar could produce it (`sleep_producible`: a superset of sleep's name grammar — any number of
//! onset+vowel+coda syllables — and of its town grammar — syllables + one of its town endings). So no sleep
//! population, of any seed (v40's dev and sealed, v41's sealed), can share a name word or town with recall text.
//!
//! DEV ITEMS. `dev_items` builds 4-choice items the same way inside book text (PG-19 validation for
//! data/v41/recall_dev.tsv): prompt = A (48 tokens) + note + B (the gap, G tokens) + cue; choices = the answer
//! and three other values of the same kind, none of which occurs in the book text. Bins g256 / g1024 / g1900;
//! every scored sequence ([EOS] prompt choice [EOS]) fits in 2048 tokens.

use crate::bpe::Bpe;
use crate::fact_items::tsv_line;
use crate::numbers::{insert_marks, is_digit, is_mark, strip_marks};
use crate::rng::{derive_seed, SplitMix64};
use crate::sleep_facts::{NAME_CODAS, NAME_ONSETS, NAME_VOWELS, TOWN_CODAS, TOWN_ENDS, TOWN_ONSETS, TOWN_VOWELS};
use std::collections::HashSet;

/// The shortest gap (tokens between a note's end and its cue's start).
pub const MIN_GAP: usize = 32;
/// The longest gap drawn, as a fraction of the row length.
pub const MAX_GAP_FRAC: f64 = 0.9;
pub const MAX_NOTES: usize = 4;
pub const REREAD_MIN: usize = 32;
pub const REREAD_MAX: usize = 128;
/// Percent chance that a row also gets a re-read.
pub const REREAD_PCT: u64 = 50;
/// Percent chance that a note repeats the previous note's frame (with another subject).
pub const SAME_FRAME_PCT: u64 = 35;
const SENT_SLACK: usize = 128;
const WORD_SLACK: usize = 64;

pub const FRAMES: [&str; 12] =
    ["favourite", "room", "kept", "lives", "password", "farm", "leaves", "ticket", "called", "hid", "meet", "age"];

const NOTE_LEADS: [&str; 8] = ["", "", "", "Note: ", "By the way, ", "For the record, ", "Mind you, ", "Remember this: "];
const CUE_LEADS: [&str; 9] = ["", "", "", "Remember, ", "As noted, ", "Recall that ", "Again, ", "We know that ", "As we said, "];

const COLOURS: [&str; 24] = [
    "red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "grey", "black", "white", "teal", "violet",
    "crimson", "amber", "indigo", "scarlet", "silver", "gold", "olive", "maroon", "navy", "turquoise", "lilac",
];
const FRUITS: [&str; 22] = [
    "apple", "pear", "plum", "cherry", "peach", "apricot", "grape", "lemon", "melon", "mango", "banana", "fig", "lime",
    "quince", "damson", "raspberry", "strawberry", "gooseberry", "blackberry", "blueberry", "pineapple", "coconut",
];
const ANIMALS: [&str; 22] = [
    "otter", "badger", "fox", "hare", "rabbit", "deer", "wolf", "bear", "horse", "goat", "tiger", "lion", "zebra",
    "giraffe", "elephant", "hedgehog", "squirrel", "beaver", "seal", "whale", "dolphin", "camel",
];
const FLOWERS: [&str; 20] = [
    "rose", "tulip", "daisy", "lily", "poppy", "iris", "orchid", "daffodil", "bluebell", "lavender", "primrose",
    "foxglove", "peony", "marigold", "sunflower", "snowdrop", "crocus", "heather", "jasmine", "cornflower",
];
const TREES: [&str; 19] = [
    "oak", "ash", "elm", "birch", "beech", "willow", "pine", "maple", "cedar", "yew", "hazel", "holly", "rowan", "alder",
    "larch", "chestnut", "sycamore", "poplar", "spruce",
];
const BIRDS: [&str; 20] = [
    "robin", "wren", "sparrow", "blackbird", "thrush", "swallow", "magpie", "crow", "raven", "owl", "heron",
    "kingfisher", "finch", "starling", "lark", "swift", "pigeon", "gull", "eagle", "falcon",
];
const INSTRUMENTS: [&str; 16] = [
    "piano", "violin", "cello", "flute", "harp", "trumpet", "guitar", "drum", "clarinet", "oboe", "organ", "banjo",
    "trombone", "harmonica", "accordion", "viola",
];
const FAV_THINGS: [(&str, &[&str]); 7] = [
    ("colour", &COLOURS),
    ("fruit", &FRUITS),
    ("animal", &ANIMALS),
    ("flower", &FLOWERS),
    ("tree", &TREES),
    ("bird", &BIRDS),
    ("instrument", &INSTRUMENTS),
];
const KEPT_ADJS: [&str; 20] = [
    "old", "small", "heavy", "wooden", "broken", "little", "round", "tall", "painted", "ancient", "rusty", "tiny",
    "square", "empty", "carved", "iron", "leather", "glass", "brass", "velvet",
];
const KEPT_NOUNS: [&str; 24] = [
    "lamp", "map", "letter", "clock", "mirror", "kettle", "basket", "ladder", "fiddle", "compass", "lantern", "blanket",
    "painting", "telescope", "rope", "hammer", "saddle", "bell", "jar", "diary", "umbrella", "candlestick",
    "chessboard", "globe",
];
const PLACES: [&str; 26] = [
    "cellar", "attic", "barn", "kitchen", "study", "library", "stable", "pantry", "chapel", "workshop", "tower",
    "hallway", "nursery", "greenhouse", "boathouse", "larder", "loft", "porch", "scullery", "parlour", "granary",
    "dairy", "forge", "bakery", "gatehouse", "wardrobe",
];
const PW_NOUNS: [&str; 20] = [
    "tower", "library", "archive", "attic", "workshop", "garden", "boathouse", "laboratory", "clubhouse", "treehouse",
    "cave", "bridge", "tunnel", "cabin", "lodge", "mill", "observatory", "armoury", "chapel", "greenhouse",
];
const FARM_ANIMALS: [&str; 15] = [
    "sheep", "goats", "cows", "pigs", "hens", "ducks", "geese", "horses", "rabbits", "donkeys", "turkeys", "llamas",
    "alpacas", "ponies", "oxen",
];
const VEHICLES: [&str; 6] = ["train", "coach", "boat", "ferry", "bus", "tram"];
const RELATIONS: [&str; 12] =
    ["sister", "brother", "cousin", "aunt", "uncle", "friend", "dog", "cat", "horse", "parrot", "neighbour", "teacher"];
const HIDE_NOUNS: [&str; 16] = [
    "key", "letter", "ring", "coin", "map", "knife", "watch", "book", "purse", "locket", "medal", "photograph",
    "candle", "whistle", "spoon", "ribbon",
];
const UNDER: [&str; 19] = [
    "bed", "stairs", "floorboards", "bridge", "rug", "table", "stone", "hedge", "cushion", "sink", "carpet", "wagon",
    "piano", "mattress", "doormat", "bench", "desk", "step", "sofa",
];
const WEEKDAYS: [&str; 7] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
/// Real first names; the ones sleep's grammar could produce are skipped when drawn (see `first_name`).
const FIRST_NAMES: [&str; 64] = [
    "Alice", "Ada", "Agnes", "Arthur", "Clara", "Charles", "Cecil", "Edith", "Eleanor", "Emily", "Ivy", "Isaac", "Jack",
    "Jane", "Jasper", "Joan", "Jacob", "Josephine", "Lucy", "Grace", "Oliver", "Owen", "Ursula", "Rachel", "Catherine",
    "Vincent", "Beatrice", "Felix", "Oscar", "Iris", "Esther", "Edgar", "Alfred", "Albert", "Abigail", "Amelia", "Ella",
    "Evelyn", "Hugo", "Ian", "Isla", "Juliet", "June", "Jude", "Celia", "Cyril", "Cora", "Clive", "Connor", "Quentin",
    "Yvonne", "Olive", "Otto", "Ezra", "Eliza", "Imogen", "Archie", "Alec", "Chloe", "Colin", "Ignatius", "Unity",
    "Jocelyn", "Cuthbert",
];
// The recall word style (surnames, towns, passwords): onset, vowel, middle consonants, vowel, final; then the
// sleep-grammar filter.
const R_ONSETS: [&str; 24] =
    ["b", "c", "ch", "cl", "cr", "d", "f", "g", "gl", "j", "k", "l", "m", "n", "p", "ph", "qu", "r", "s", "sc", "t", "v", "w", "y"];
const R_VOWELS: [&str; 9] = ["a", "e", "i", "o", "u", "a", "e", "o", "ai"];
const R_MIDS: [&str; 24] =
    ["b", "c", "ck", "d", "f", "g", "j", "l", "ll", "m", "n", "nn", "p", "r", "rr", "s", "t", "tt", "v", "nd", "rd", "ld", "nc", "sc"];
const R_FINALS: [&str; 10] = ["", "", "", "n", "r", "l", "ck", "s", "t", "x"];
/// Surname endings (they start with a vowel and follow a middle consonant).
const R_SURNAME_ENDS: [&str; 10] = ["ett", "ick", "ard", "ow", "in", "on", "er", "ey", "ock", "is"];
const R_TOWN_ENDS: [&str; 20] = [
    "port", "gate", "bridge", "cross", "minster", "kirk", "wold", "brook", "hythe", "chester", "fell", "garth", "stow",
    "worth", "burgh", "mouth", "head", "wich", "caster", "ness",
];

// ---------------------------------------------------------------- the sleep-grammar filter

/// Positions reachable after one or more syllables (onset + vowel + coda) from position 0.
fn syllable_reach(w: &[u8], onsets: &[&str], vowels: &[&str], codas: &[&str]) -> Vec<bool> {
    let n = w.len();
    let mut start = vec![false; n + 1]; // a syllable may start here
    let mut done = vec![false; n + 1]; // >= 1 syllable ends here
    start[0] = true;
    for i in 0..=n {
        if !start[i] {
            continue;
        }
        for o in onsets {
            let i1 = i + o.len();
            if i1 > n || &w[i..i1] != o.as_bytes() {
                continue;
            }
            for v in vowels {
                let i2 = i1 + v.len();
                if i2 > n || &w[i1..i2] != v.as_bytes() {
                    continue;
                }
                for c in codas {
                    let i3 = i2 + c.len();
                    if i3 <= n && &w[i2..i3] == c.as_bytes() {
                        done[i3] = true;
                        start[i3] = true;
                    }
                }
            }
        }
    }
    done
}

/// True when sleep_facts' generators could produce `word` (case-insensitive) as a name word or a town, under a
/// superset of their grammar (any number of syllables, any length). Recall never uses such a word.
pub fn sleep_producible(word: &str) -> bool {
    let w = word.to_lowercase();
    let w = w.as_bytes();
    if w.is_empty() {
        return false;
    }
    if syllable_reach(w, &NAME_ONSETS, &NAME_VOWELS, &NAME_CODAS)[w.len()] {
        return true;
    }
    let reach = syllable_reach(w, &TOWN_ONSETS, &TOWN_VOWELS, &TOWN_CODAS);
    TOWN_ENDS.iter().any(|e| w.len() > e.len() && w.ends_with(e.as_bytes()) && reach[w.len() - e.len()])
}

// ---------------------------------------------------------------- made-up values

fn pick<'a>(r: &mut SplitMix64, v: &[&'a str]) -> &'a str {
    v[r.below(v.len() as u64) as usize]
}

fn uniform(r: &mut SplitMix64, lo: u64, hi: u64) -> u64 {
    lo + r.below(hi - lo + 1)
}

fn capitalised(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c).collect(),
        None => String::new(),
    }
}

/// onset vowel (middle vowel) x `n_more` final, e.g. "callor", "jondabex".
fn stem(r: &mut SplitMix64, n_more: u64) -> String {
    let mut s = String::new();
    s.push_str(pick(r, &R_ONSETS));
    s.push_str(pick(r, &R_VOWELS));
    for _ in 0..n_more {
        s.push_str(pick(r, &R_MIDS));
        s.push_str(pick(r, &R_VOWELS));
    }
    s.push_str(pick(r, &R_FINALS));
    s
}

/// Draw from `gen` until the word is at least `min` letters and not sleep-producible.
fn filtered(r: &mut SplitMix64, min: usize, gen: impl Fn(&mut SplitMix64) -> String) -> String {
    loop {
        let w = gen(r);
        if w.len() >= min && !sleep_producible(&w) {
            return w;
        }
    }
}

pub fn first_name(r: &mut SplitMix64) -> String {
    filtered(r, 1, |r| pick(r, &FIRST_NAMES).to_string())
}

pub fn surname(r: &mut SplitMix64) -> String {
    filtered(r, 5, |r| {
        if r.below(2) == 0 {
            capitalised(&stem(r, 1))
        } else {
            let mut s = String::new();
            s.push_str(pick(r, &R_ONSETS));
            s.push_str(pick(r, &R_VOWELS));
            s.push_str(pick(r, &R_MIDS));
            s.push_str(pick(r, &R_SURNAME_ENDS));
            capitalised(&s)
        }
    })
}

pub fn full_name(r: &mut SplitMix64) -> String {
    format!("{} {}", first_name(r), surname(r))
}

pub fn town(r: &mut SplitMix64) -> String {
    filtered(r, 6, |r| {
        let n = r.below(2);
        let mut s = stem(r, n);
        s.push_str(pick(r, &R_TOWN_ENDS));
        capitalised(&s)
    })
}

/// A made-up lowercase word (a password).
pub fn word(r: &mut SplitMix64) -> String {
    filtered(r, 5, |r| {
        let n = 1 + r.below(2);
        stem(r, n)
    })
}

/// What kind of value a frame asks for (distractors are drawn from the same kind).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    List(&'static [&'static str]),
    FullName,
    FirstName,
    Town,
    Word,
    Num(u64, u64),
    Time,
}

/// A value of `kind`.
pub fn value_of(kind: Kind, r: &mut SplitMix64) -> String {
    match kind {
        Kind::List(v) => pick(r, v).to_string(),
        Kind::FullName => full_name(r),
        Kind::FirstName => first_name(r),
        Kind::Town => town(r),
        Kind::Word => word(r),
        Kind::Num(lo, hi) => uniform(r, lo, hi).to_string(),
        Kind::Time => format!("{}:{:02}", uniform(r, 1, 12), 5 * r.below(12)),
    }
}

/// One made-up fact: the note sentence, the cue text, and the answer that follows the cue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fact {
    pub frame: &'static str,
    /// The subject (unique inside one row).
    pub key: String,
    pub kind: Kind,
    pub value: String,
    /// Text after the value, before the full stop (e.g. " years old").
    pub tail: &'static str,
    /// The note, lead-in included: "{lead}{body} {value}{tail}."
    pub note: String,
    /// The cue, lead-in included: "{lead}{body}".
    pub cue: String,
}

impl Fact {
    /// What follows the cue: " {value}{tail}."
    pub fn answer(&self) -> String {
        answer_text(&self.value, self.tail)
    }
}

pub fn answer_text(value: &str, tail: &str) -> String {
    format!(" {value}{tail}.")
}

/// `lead` + `body`, the body's first letter lowered after a lead-in unless it starts a name.
fn with_lead(lead: &str, body: &str, starts_with_name: bool) -> String {
    if lead.is_empty() || starts_with_name {
        format!("{lead}{body}")
    } else {
        let mut c = body.chars();
        let f = c.next().map(|f| f.to_lowercase().collect::<String>()).unwrap_or_default();
        format!("{lead}{f}{}", c.as_str())
    }
}

/// A fact of frame `frame` (index into `FRAMES`), with random lead-ins.
pub fn make_fact(frame: usize, r: &mut SplitMix64) -> Fact {
    let (body, key, kind, tail, by_name): (String, String, Kind, &'static str, bool) = match frame {
        0 => {
            let (thing, pool) = FAV_THINGS[r.below(FAV_THINGS.len() as u64) as usize];
            let n = full_name(r);
            (format!("{n}'s favourite {thing} is"), format!("fav:{n}:{thing}"), Kind::List(pool), "", true)
        }
        1 => {
            let room = uniform(r, 100, 999);
            (format!("Room {room} belongs to"), format!("room:{room}"), Kind::FullName, "", false)
        }
        2 => {
            let (a, n) = (pick(r, &KEPT_ADJS), pick(r, &KEPT_NOUNS));
            (format!("The {a} {n} is kept in the"), format!("kept:{a} {n}"), Kind::List(&PLACES), "", false)
        }
        3 => {
            let n = full_name(r);
            (format!("{n} lives in"), format!("lives:{n}"), Kind::Town, "", true)
        }
        4 => {
            let n = pick(r, &PW_NOUNS);
            (format!("The password for the {n} is"), format!("pw:{n}"), Kind::Word, "", false)
        }
        5 => {
            let (a, n) = (pick(r, &FARM_ANIMALS), full_name(r));
            (format!("The number of {a} on {n}'s farm is"), format!("farm:{n}:{a}"), Kind::Num(2, 999), "", false)
        }
        6 => {
            let (v, t) = (pick(r, &VEHICLES), town(r));
            (format!("The {v} to {t} leaves at"), format!("leaves:{v}:{t}"), Kind::Time, "", false)
        }
        7 => {
            let d = 3 + r.below(4) as u32;
            let n = full_name(r);
            (format!("{n}'s ticket number is"), format!("ticket:{n}"), Kind::Num(10u64.pow(d - 1), 10u64.pow(d) - 1), "", true)
        }
        8 => {
            let (rel, n) = (pick(r, &RELATIONS), full_name(r));
            (format!("{n}'s {rel} is called"), format!("called:{n}:{rel}"), Kind::FirstName, "", true)
        }
        9 => {
            let (n, thing) = (full_name(r), pick(r, &HIDE_NOUNS));
            (format!("{n} hid the {thing} under the"), format!("hid:{n}:{thing}"), Kind::List(&UNDER), "", true)
        }
        10 => {
            let (a, b) = (full_name(r), full_name(r));
            (format!("{a} will meet {b} on"), format!("meet:{a}:{b}"), Kind::List(&WEEKDAYS), "", true)
        }
        _ => {
            let n = full_name(r);
            (format!("{n} is"), format!("age:{n}"), Kind::Num(3, 99), " years old", true)
        }
    };
    let value = value_of(kind, r);
    let note = format!("{}{}", with_lead(pick(r, &NOTE_LEADS), &body, by_name), answer_text(&value, tail));
    let cue = with_lead(pick(r, &CUE_LEADS), &body, by_name);
    Fact { frame: FRAMES[frame.min(FRAMES.len() - 1)], key, kind, value, tail, note, cue }
}

// ---------------------------------------------------------------- word and sentence boundaries

fn marked(bpe: &Bpe, text: &str) -> Vec<u32> {
    insert_marks(&bpe.encode(text))
}

fn tok_bytes(bpe: &Bpe, t: u32) -> &[u8] {
    if is_mark(t) || t as usize >= bpe.vocab_size() {
        &[]
    } else {
        bpe.token_bytes(t)
    }
}

/// A token whose text starts with a space or a newline.
fn word_start(bpe: &Bpe, t: u32) -> bool {
    matches!(tok_bytes(bpe, t).first(), Some(b' ' | b'\n'))
}

/// Does the text up to token `i` (inclusive) end a sentence (`.`, `!`, `?`, maybe followed by quotes/brackets)?
fn ends_sentence(bpe: &Bpe, f: &[u32], i: usize) -> bool {
    let b = tok_bytes(bpe, f[i]);
    let core: &[u8] = {
        let mut e = b.len();
        while e > 0 && matches!(b[e - 1], b'"' | b'\'' | b')') {
            e -= 1;
        }
        &b[..e]
    };
    match core.last() {
        Some(c) => matches!(c, b'.' | b'!' | b'?'),
        None if !b.is_empty() && i > 0 => matches!(tok_bytes(bpe, f[i - 1]).last(), Some(b'.' | b'!' | b'?')),
        None => false,
    }
}

/// Inserting before `f[i]` starts a sentence.
fn sent_boundary(bpe: &Bpe, f: &[u32], i: usize) -> bool {
    i > 0 && i < f.len() && word_start(bpe, f[i]) && ends_sentence(bpe, f, i - 1)
}

/// Inserting before `f[i]` splits no digit run and no mark from its digits.
fn safe_raw(f: &[u32], i: usize) -> bool {
    i == 0 || i >= f.len() || (!is_mark(f[i - 1]) && !(is_digit(f[i]) && is_digit(f[i - 1])))
}

struct Bounds {
    word: Vec<bool>,
    sent: Vec<bool>,
}

fn bounds(bpe: &Bpe, f: &[u32]) -> Bounds {
    let word: Vec<bool> = (0..f.len()).map(|i| i > 0 && word_start(bpe, f[i])).collect();
    let sent = (0..f.len()).map(|i| word[i] && ends_sentence(bpe, f, i - 1)).collect();
    Bounds { word, sent }
}

/// The first sentence start in [x, x+sent_slack], else word start in [x, x+64], else safe place in [x, hi], else x.
fn snap(b: &Bounds, f: &[u32], x: usize, hi: usize, sent_slack: usize) -> usize {
    if x >= hi {
        return x.min(hi);
    }
    let end = hi.min(f.len() - 1);
    for i in x..=end.min(x + sent_slack) {
        if b.sent[i] {
            return i;
        }
    }
    for i in x..=end.min(x + WORD_SLACK) {
        if b.word[i] {
            return i;
        }
    }
    (x..=end).find(|&i| safe_raw(f, i)).unwrap_or(x)
}

fn log_uniform(r: &mut SplitMix64, lo: usize, hi: usize) -> usize {
    if hi <= lo {
        return lo;
    }
    let u = (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    let (a, b) = ((lo as f64).ln(), ((hi + 1) as f64).ln());
    ((a + u * (b - a)).exp().floor() as usize).clamp(lo, hi)
}

// ---------------------------------------------------------------- the row

/// Where a note and its cue landed in the row (output token positions).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Planted {
    pub fact: Fact,
    pub note_at: usize,
    pub note_len: usize,
    pub cue_at: usize,
    /// Cue tokens before the answer.
    pub cue_len: usize,
    pub answer_len: usize,
}

impl Planted {
    /// Tokens between the note's end and the cue's start.
    pub fn gap(&self) -> usize {
        self.cue_at - (self.note_at + self.note_len)
    }
}

/// The re-read: `len` tokens at `src_at` copied to `copy_at` (output positions).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reread {
    pub src_at: usize,
    pub copy_at: usize,
    pub len: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RowPlan {
    pub notes: Vec<Planted>,
    pub reread: Option<Reread>,
}

/// v41's recall practice row: `filler` (>= len + 1 marked tokens) with notes, cues and maybe a re-read inserted;
/// exactly `len + 1` marked tokens. See the module docs.
pub fn recall_row(rng: &mut SplitMix64, bpe: &Bpe, filler: &[u32], len: usize) -> Vec<u32> {
    recall_row_plan(rng, bpe, filler, len).0
}

/// `recall_row` and where everything went.
pub fn recall_row_plan(rng: &mut SplitMix64, bpe: &Bpe, filler: &[u32], len: usize) -> (Vec<u32>, RowPlan) {
    let total = len + 1;
    assert!(filler.len() >= total, "recall_row: filler has {} tokens, need len + 1 = {total}", filler.len());
    let f = &filler[..total];
    let n_want = 1 + rng.below(MAX_NOTES as u64) as usize;
    let reread_want = rng.below(100) < REREAD_PCT;
    let mut facts: Vec<Fact> = Vec::with_capacity(n_want);
    let mut keys = HashSet::new();
    let mut prev: Option<usize> = None;
    while facts.len() < n_want {
        let frame = match prev {
            Some(p) if rng.below(100) < SAME_FRAME_PCT => p,
            _ => rng.below(FRAMES.len() as u64) as usize,
        };
        let fact = make_fact(frame, rng);
        if keys.insert(fact.key.clone()) {
            prev = Some(frame);
            facts.push(fact);
        }
    }
    let mut enc: Vec<[Vec<u32>; 3]> = facts
        .iter()
        .map(|x| [marked(bpe, &format!(" {}", x.note)), marked(bpe, &format!(" {}", x.cue)), marked(bpe, &x.answer())])
        .collect();
    // keep as many notes as fit with at least MIN_GAP filler tokens
    let cost = |e: &[Vec<u32>; 3]| e[0].len() + e[1].len() + e[2].len();
    while !enc.is_empty() && total < enc.iter().map(cost).sum::<usize>() + MIN_GAP + 1 {
        enc.pop();
        facts.pop();
    }
    if enc.is_empty() {
        return (f.to_vec(), RowPlan::default());
    }
    let notes_t: usize = enc.iter().map(cost).sum();
    let b = bounds(bpe, f);

    // the re-read's source span [s, e) of the filler (whole words, no EOS when possible)
    let mut span: Option<(usize, usize)> = None;
    if reread_want {
        let r = REREAD_MIN + rng.below((REREAD_MAX - REREAD_MIN + 1) as u64) as usize;
        for attempt in 0..4 {
            // room: s + 2·len' + notes_t + MIN_GAP <= total
            let Some(room) = total.checked_sub(notes_t + MIN_GAP + 2 * r) else { break };
            let s0 = rng.below(room as u64 + 1) as usize;
            let s = (s0..=(s0 + WORD_SLACK).min(room)).find(|&i| b.word[i]).unwrap_or(s0);
            let e = (s + REREAD_MIN..=(s + REREAD_MAX).min(total - 1))
                .filter(|&i| b.word[i])
                .min_by_key(|&i| (i as i64 - (s + r) as i64).unsigned_abs() * 2 + (i < s + r) as u64)
                .unwrap_or(s + r);
            let l = e - s;
            if s + 2 * l + notes_t + MIN_GAP > total {
                continue;
            }
            let has_eos = f[s..e].iter().any(|t| tok_bytes(bpe, *t).is_empty() && !is_mark(*t));
            if has_eos && attempt < 3 {
                continue;
            }
            span = Some((s, e));
            break;
        }
    }
    let r_len = span.map_or(0, |(s, e)| e - s);
    let fmax = total - notes_t - r_len; // every insertion at a filler position <= fmax fits wholly

    // (filler position, order, what): 0 = note j, 1 = cue j, 2 = copy
    let mut ins: Vec<(usize, u8, usize)> = Vec::with_capacity(2 * enc.len() + 1);
    let gmax_all = ((len as f64 * MAX_GAP_FRAC) as usize).max(MIN_GAP);
    for j in 0..enc.len() {
        let gmax = gmax_all.min(fmax);
        let g = log_uniform(rng, MIN_GAP, gmax);
        let a0 = rng.below((fmax - g + 1) as u64) as usize;
        let mut a = snap(&b, f, a0, fmax - g, SENT_SLACK);
        if let Some((s, e)) = span {
            if a > s && a < e {
                a = s;
            }
        }
        let mut c = snap(&b, f, a + g, fmax, (g / 2).clamp(16, SENT_SLACK));
        if let Some((s, e)) = span {
            if c > s && c < e {
                c = e;
            }
        }
        ins.push((a, 0, j));
        ins.push((c, 1, j));
    }
    if let Some((_, e)) = span {
        let q0 = e + rng.below((fmax - e + 1) as u64) as usize;
        ins.push((snap(&b, f, q0, fmax, SENT_SLACK), 2, 0));
    }
    ins.sort();

    let mut out: Vec<u32> = Vec::with_capacity(total + notes_t + r_len);
    let mut at = vec![[0usize; 2]; enc.len()];
    let mut copy_at = 0;
    let mut fi = 0;
    for &(pos, what, j) in &ins {
        out.extend_from_slice(&f[fi..pos]);
        fi = pos;
        match what {
            0 => {
                at[j][0] = out.len();
                out.extend_from_slice(&enc[j][0]);
            }
            1 => {
                at[j][1] = out.len();
                out.extend_from_slice(&enc[j][1]);
                out.extend_from_slice(&enc[j][2]);
            }
            _ => {
                copy_at = out.len();
                let (s, e) = span.unwrap();
                out.extend_from_slice(&f[s..e]);
            }
        }
    }
    out.extend_from_slice(&f[fi..]);
    out.truncate(total);
    let reread = span.map(|(s, e)| {
        let before: usize = ins
            .iter()
            .filter(|x| x.0 <= s)
            .map(|&(_, w, j)| match w {
                0 => enc[j][0].len(),
                1 => enc[j][1].len() + enc[j][2].len(),
                _ => e - s,
            })
            .sum();
        Reread { src_at: s + before, copy_at, len: e - s }
    });
    let notes = facts
        .into_iter()
        .zip(enc.iter())
        .zip(at)
        .map(|((fact, e), [na, ca])| Planted { fact, note_at: na, note_len: e[0].len(), cue_at: ca, cue_len: e[1].len(), answer_len: e[2].len() })
        .collect();
    (out, RowPlan { notes, reread })
}

// ---------------------------------------------------------------- dev items (data/v41/recall_dev.tsv)

pub const DEV_KIND: &str = "recall";
pub const DEV_GAPS: [usize; 3] = [256, 1024, 1900];
pub const DEV_BINS: [&str; 3] = ["g256", "g1024", "g1900"];
/// Book tokens before the note.
pub const DEV_A_TOKENS: usize = 48;
/// The most tokens a span may grow to reach a sentence start.
pub const DEV_SLACK: usize = 24;
/// Every scored sequence [EOS] prompt choice [EOS] fits in this many tokens.
pub const DEV_MAX_SEQ: usize = 2048;
const DEV_LABEL: u64 = 0x7265_6361_6c6c_6476;
const DEV_MAX_DRAWS: usize = 200_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DevItem {
    pub id: u32,
    pub bin: &'static str,
    pub gap: usize,
    pub frame: &'static str,
    pub right: usize,
    pub prompt: String,
    pub choices: Vec<String>,
    pub book: usize,
    /// Marked tokens of the prompt, of the gap text, and of the longest scored sequence.
    pub prompt_tokens: usize,
    pub gap_tokens: usize,
    pub max_seq: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DevStats {
    /// Per bin: draws that hit a short book, found no sentence start within the slack, made a sequence over the
    /// limit, or met a value in the book text.
    pub short_books: Vec<u64>,
    pub no_sentence_start: Vec<u64>,
    pub too_long: Vec<u64>,
    pub value_in_text: Vec<u64>,
}

/// Does `needle` occur in `hay` with no letter or digit right before or after it?
pub fn contains_whole(hay: &str, needle: &str) -> bool {
    let h = hay.as_bytes();
    let mut from = 0;
    while let Some(p) = hay[from..].find(needle) {
        let (s, e) = (from + p, from + p + needle.len());
        let before = s == 0 || !h[s - 1].is_ascii_alphanumeric();
        let after = e >= h.len() || !h[e].is_ascii_alphanumeric();
        if before && after {
            return true;
        }
        from = s + 1;
        while !hay.is_char_boundary(from) {
            from += 1;
        }
    }
    false
}

fn next_sent(bpe: &Bpe, toks: &[u32], x: usize, slack: usize) -> Option<usize> {
    (x..=(x + slack).min(toks.len().saturating_sub(1))).find(|&i| sent_boundary(bpe, toks, i))
}

/// `n_per_bin` recall items per gap bin (`DEV_GAPS`) from `books` (marked tokens), seeded.
pub fn dev_items(bpe: &Bpe, books: &[Vec<u32>], n_per_bin: usize, seed: u64) -> Result<(Vec<DevItem>, DevStats), String> {
    dev_items_with(bpe, books, n_per_bin, seed, &DEV_GAPS, &DEV_BINS, DEV_MAX_SEQ)
}

/// `dev_items` with other gap bins and another sequence limit.
pub fn dev_items_with(
    bpe: &Bpe,
    books: &[Vec<u32>],
    n_per_bin: usize,
    seed: u64,
    gaps: &[usize],
    bins: &[&'static str],
    max_seq_limit: usize,
) -> Result<(Vec<DevItem>, DevStats), String> {
    if books.is_empty() && n_per_bin > 0 {
        return Err("no books".into());
    }
    let mut out = Vec::with_capacity(gaps.len() * n_per_bin);
    let z = vec![0u64; gaps.len()];
    let mut st = DevStats { short_books: z.clone(), no_sentence_start: z.clone(), too_long: z.clone(), value_in_text: z };
    for (bi, (&gap, &bin)) in gaps.iter().zip(bins.iter()).enumerate() {
        let mut r = SplitMix64::new(derive_seed(seed, DEV_LABEL, bi as u64));
        let mut pos: Vec<usize> = (0..n_per_bin).map(|i| i % 4).collect();
        r.shuffle(&mut pos);
        let need = DEV_A_TOKENS + gap + 3 * DEV_SLACK + 2;
        for &right in &pos {
            let mut draws = 0;
            let item = loop {
                draws += 1;
                if draws > DEV_MAX_DRAWS {
                    return Err(format!("bin {bin}: no item found in {DEV_MAX_DRAWS} draws ({st:?})"));
                }
                let book = r.below(books.len() as u64) as usize;
                let toks = &books[book];
                if toks.len() < need {
                    st.short_books[bi] += 1;
                    continue;
                }
                let s0 = r.below((toks.len() - need) as u64) as usize;
                let spans = next_sent(bpe, toks, s0.max(1), DEV_SLACK).and_then(|s| {
                    let a = next_sent(bpe, toks, s + DEV_A_TOKENS, DEV_SLACK)?;
                    let e = next_sent(bpe, toks, a + gap, DEV_SLACK)?;
                    Some((s, a, e))
                });
                let Some((s, a_end, b_end)) = spans else {
                    st.no_sentence_start[bi] += 1;
                    continue;
                };
                let a_text = bpe.decode(&strip_marks(&toks[s..a_end]));
                let a_text = a_text.trim_start();
                let b_text = bpe.decode(&strip_marks(&toks[a_end..b_end]));
                let frame = r.below(FRAMES.len() as u64) as usize;
                let fact = make_fact(frame, &mut r);
                let book_text = format!("{a_text}\n{b_text}");
                if book_text.contains('\r') || contains_whole(&book_text, &fact.value) {
                    st.value_in_text[bi] += 1;
                    continue;
                }
                let prompt = format!("{a_text} {}{b_text} {}", fact.note, fact.cue);
                let mut values = vec![fact.value.clone()];
                let mut tries = 0;
                while values.len() < 4 && tries < 10_000 {
                    tries += 1;
                    let v = value_of(fact.kind, &mut r);
                    if !values.contains(&v) && !contains_whole(&prompt, &v) {
                        values.push(v);
                    }
                }
                if values.len() < 4 {
                    st.value_in_text[bi] += 1;
                    continue;
                }
                let mut choices: Vec<String> = values[1..].iter().map(|v| answer_text(v, fact.tail)).collect();
                choices.insert(right, fact.answer());
                let prompt_tokens = marked(bpe, &prompt).len();
                let max_seq = 2 + prompt_tokens + choices.iter().map(|c| marked(bpe, &format!(" {}", c.trim())).len()).max().unwrap();
                if max_seq > max_seq_limit {
                    st.too_long[bi] += 1;
                    continue;
                }
                break DevItem {
                    id: out.len() as u32,
                    bin,
                    gap,
                    frame: fact.frame,
                    right,
                    gap_tokens: marked(bpe, &b_text).len(),
                    prompt,
                    choices,
                    book,
                    prompt_tokens,
                    max_seq,
                };
            };
            out.push(item);
        }
    }
    Ok((out, st))
}

/// One TSV line (`id kind bin right prompt c0..c3`, fields escaped as in fact_items).
pub fn dev_line(it: &DevItem) -> String {
    tsv_line(it.id, DEV_KIND, it.bin, it.right, &it.prompt, &it.choices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::books::encode_book;
    use crate::bpe::EOS;
    use crate::numbers::VOCAB_GLANCE;
    use crate::sleep_facts::{people, split_people};

    const WORDS: [&str; 14] =
        ["the", "old", "house", "stood", "by", "a", "river", "where", "children", "played", "in", "summer", "and", "sang"];

    fn text(seed: u64, words: usize) -> String {
        let mut r = SplitMix64::new(seed);
        let mut s = String::new();
        for i in 0..words {
            if i > 0 {
                s.push(if r.below(12) == 0 { '\n' } else { ' ' });
            }
            if r.below(30) == 0 {
                s.push_str(&r.below(5000).to_string());
            } else {
                s.push_str(WORDS[r.below(WORDS.len() as u64) as usize]);
            }
            if r.below(8) == 0 {
                s.push(if r.below(3) == 0 { ',' } else { '.' });
            }
        }
        s
    }

    fn filler(bpe: &Bpe, seed: u64, min: usize) -> Vec<u32> {
        let mut f = Vec::new();
        let mut k = 0;
        while f.len() < min {
            f.extend(encode_book(bpe, &text(seed * 1000 + k, 400)));
            k += 1;
        }
        f
    }

    #[test]
    fn rows_exact_length_valid_ids_answers_after_cues_and_gaps_in_range() {
        let bpe = Bpe::from_merges(vec![]).unwrap();
        let mut gaps = Vec::new();
        let (mut n_rows, mut n_reread, mut notes_seen) = (0, 0, [0usize; 5]);
        for (k, &len) in [256usize, 1024, 2048, 4096].iter().enumerate() {
            for t in 0..40u64 {
                let f = filler(&bpe, 100 * k as u64 + t, len + 1 + (t as usize % 7) * 50);
                let mut r = SplitMix64::new(t ^ (k as u64) << 32);
                let (row, plan) = recall_row_plan(&mut r, &bpe, &f, len);
                n_rows += 1;
                assert_eq!(row.len(), len + 1);
                assert!(row.iter().all(|&x| (x as usize) < VOCAB_GLANCE));
                assert!(!plan.notes.is_empty());
                notes_seen[plan.notes.len()] += 1;
                for p in &plan.notes {
                    let note = marked(&bpe, &format!(" {}", p.fact.note));
                    let cue = marked(&bpe, &format!(" {}", p.fact.cue));
                    let ans = marked(&bpe, &p.fact.answer());
                    assert_eq!(&row[p.note_at..p.note_at + p.note_len], note.as_slice());
                    assert_eq!(&row[p.cue_at..p.cue_at + p.cue_len], cue.as_slice());
                    let a0 = p.cue_at + p.cue_len;
                    assert_eq!(&row[a0..a0 + p.answer_len], ans.as_slice(), "the answer follows its cue");
                    assert!(a0 + p.answer_len <= len + 1, "the answer is inside the row");
                    assert!(p.cue_at > p.note_at + p.note_len);
                    // the note holds the answer: note = lead + body + answer
                    assert!(note.ends_with(&ans));
                    assert!(p.gap() >= MIN_GAP && p.gap() <= len, "gap {} of len {len}", p.gap());
                    gaps.push((p.gap(), len));
                }
                if let Some(rr) = &plan.reread {
                    n_reread += 1;
                    assert!((REREAD_MIN..=REREAD_MAX).contains(&rr.len));
                    assert!(rr.src_at + rr.len <= rr.copy_at && rr.copy_at + rr.len <= len + 1);
                    assert_eq!(row[rr.src_at..rr.src_at + rr.len], row[rr.copy_at..rr.copy_at + rr.len]);
                }
            }
        }
        assert!(n_reread > n_rows / 4 && n_reread < 3 * n_rows / 4, "{n_reread} of {n_rows}");
        assert!(notes_seen[1..].iter().all(|&c| c > 10), "{notes_seen:?}");
        // gaps cover short to very long
        assert!(gaps.iter().any(|&(g, _)| g < 64));
        assert!(gaps.iter().any(|&(g, l)| g > l / 2));
        assert!(gaps.iter().filter(|&&(g, l)| l == 4096 && g > 2048).count() > 3);
    }

    #[test]
    fn deterministic_and_seeded() {
        let bpe = Bpe::from_merges(vec![]).unwrap();
        let f = filler(&bpe, 9, 2049);
        let a = recall_row_plan(&mut SplitMix64::new(5), &bpe, &f, 2048);
        let b = recall_row_plan(&mut SplitMix64::new(5), &bpe, &f, 2048);
        assert_eq!(a, b);
        assert_ne!(recall_row(&mut SplitMix64::new(6), &bpe, &f, 2048), a.0);
        // recall_row is the row of recall_row_plan, and consumes the same draws
        let mut r1 = SplitMix64::new(5);
        let mut r2 = SplitMix64::new(5);
        assert_eq!(recall_row(&mut r1, &bpe, &f, 2048), recall_row_plan(&mut r2, &bpe, &f, 2048).0);
        assert_eq!(r1, r2);
    }

    #[test]
    fn tiny_rows_and_marks_kept_whole() {
        let bpe = Bpe::from_merges(vec![]).unwrap();
        let f = filler(&bpe, 3, 400);
        // too short for any note: the filler itself
        for len in [0usize, 8, 40] {
            let row = recall_row(&mut SplitMix64::new(1), &bpe, &f, len);
            assert_eq!(row.len(), len + 1);
        }
        for len in [64usize, 100, 200, 399] {
            for s in 0..30 {
                let row = recall_row(&mut SplitMix64::new(s), &bpe, &f, len);
                assert_eq!(row.len(), len + 1);
                // every mark is followed by exactly its digits (or the row end)
                for (i, &t) in row.iter().enumerate() {
                    if is_mark(t) {
                        let l = (t - crate::numbers::MARK_BASE + 1) as usize;
                        let run = row[i + 1..].iter().take_while(|&&x| is_digit(x)).count();
                        assert!(run == l || i + 1 + run == row.len(), "mark {l} then {run} digits");
                    }
                }
            }
        }
    }

    #[test]
    fn no_test_frames_and_varied_frames() {
        let bpe = Bpe::from_merges(vec![]).unwrap();
        let f = filler(&bpe, 4, 1025);
        let mut frames = HashSet::new();
        for s in 0..400u64 {
            let (_, plan) = recall_row_plan(&mut SplitMix64::new(s), &bpe, &f, 1024);
            for p in &plan.notes {
                frames.insert(p.fact.frame);
                for t in [&p.fact.note, &p.fact.cue] {
                    let l = t.to_lowercase();
                    assert!(!l.contains("code for the") && !l.contains(" born in") && !l.contains("worked as"), "{t}");
                }
                assert!(p.fact.note.ends_with('.'));
            }
        }
        assert_eq!(frames.len(), FRAMES.len());
    }

    #[test]
    fn sleep_grammar_filter() {
        // the parser accepts everything sleep's generators make (v40 dev + sealed, v41 sealed, other seeds); the
        // messages name no word, as some of these populations are sealed
        let (d, t) = split_people(40, 150, 150);
        let v41 = crate::sleep_facts::people_excluding(41, 150, &crate::sleep_facts::words_of(&[d.clone(), t.clone()].concat()));
        for ps in [d.clone(), t.clone(), v41.clone(), people(7, 300)] {
            for p in ps {
                assert!(p.name.split(' ').all(sleep_producible), "a sleep name word the parser rejects");
                assert!(sleep_producible(&p.town), "a sleep town the parser rejects");
            }
        }
        assert!(sleep_producible("Molly") && !sleep_producible("Alice") && !sleep_producible("Quorrin"));
        // recall's words never are; and none meets a sleep population word
        let mut sleep_words = HashSet::new();
        for p in d.iter().chain(&t).chain(&v41) {
            sleep_words.extend(p.name.split(' ').map(|w| w.to_lowercase()));
            sleep_words.insert(p.town.to_lowercase());
        }
        let mut r = SplitMix64::new(77);
        let ok_first = FIRST_NAMES.iter().filter(|n| !sleep_producible(n)).count();
        assert!(ok_first >= 40, "{ok_first} first names usable");
        let mut distinct = HashSet::new();
        for _ in 0..3000 {
            for w in [first_name(&mut r), surname(&mut r), town(&mut r), word(&mut r)] {
                assert!(!sleep_producible(&w), "{w}");
                assert!(!sleep_words.contains(&w.to_lowercase()), "a recall word met a sleep word");
                distinct.insert(w);
            }
        }
        assert!(distinct.len() > 5000, "{}", distinct.len());
    }

    #[test]
    fn dev_items_well_formed() {
        let bpe = Bpe::from_merges(vec![]).unwrap();
        // byte-level tokens: about 5 a word; books of 3000 words reach the g1900 bin
        let books: Vec<Vec<u32>> = (0..4).map(|i| encode_book(&bpe, &text(50 + i, if i == 1 { 100 } else { 3000 }))).collect();
        // byte-level notes are about four times longer than real ones, so the bins here are smaller
        const G: [usize; 3] = [64, 256, 1024];
        const B: [&str; 3] = ["g64", "g256", "g1024"];
        let (items, st) = dev_items_with(&bpe, &books, 8, 1, &G, &B, DEV_MAX_SEQ).unwrap();
        assert_eq!(dev_items_with(&bpe, &books, 8, 1, &G, &B, DEV_MAX_SEQ).unwrap(), (items.clone(), st.clone()));
        assert_ne!(dev_items_with(&bpe, &books, 8, 2, &G, &B, DEV_MAX_SEQ).unwrap().0, items);
        assert_eq!(items.len(), 24);
        assert!(st.short_books.iter().sum::<u64>() > 0);
        for (bi, bin) in B.iter().enumerate() {
            let of: Vec<&DevItem> = items.iter().filter(|x| x.bin == *bin).collect();
            let mut pos = [0; 4];
            for it in &of {
                pos[it.right] += 1;
                assert_ne!(it.book, 1);
                assert!(it.max_seq <= DEV_MAX_SEQ);
                assert!(it.gap_tokens + 8 >= G[bi] && it.gap_tokens <= G[bi] + DEV_SLACK + 4, "{}", it.gap_tokens);
            }
            assert_eq!(pos, [2, 2, 2, 2]);
        }
        for it in &items {
            assert_eq!(it.choices.len(), 4);
            let mut c = it.choices.clone();
            c.sort();
            c.dedup();
            assert_eq!(c.len(), 4);
            // the note holds the right answer; the prompt ends with the cue; no wrong value is in the prompt
            let right = &it.choices[it.right];
            assert!(it.prompt.contains(&format!("{}", right.trim_end_matches('.'))));
            assert!(!it.prompt.ends_with(' ') && !it.prompt.starts_with([' ', '\n']));
            for (j, ch) in it.choices.iter().enumerate() {
                assert!(ch.starts_with(' ') && ch.ends_with('.'));
                if j != it.right {
                    let core = ch.trim().trim_end_matches('.').trim_end_matches(" years old");
                    assert!(!contains_whole(&it.prompt, core), "{ch:?} in prompt");
                }
            }
            let line = dev_line(it);
            let f: Vec<&str> = line.split('\t').collect();
            assert_eq!(f.len(), 9);
            assert_eq!(crate::fact_items::unescape_field(f[4]), it.prompt);
        }
        let _ = EOS;
    }

    #[test]
    fn contains_whole_words_only() {
        assert!(contains_whole("in 1845 and 45.", "45"));
        assert!(!contains_whole("in 1845 and 450.", "45"));
        assert!(contains_whole("Plum, then", "Plum"));
        assert!(!contains_whole("Plumber", "Plum"));
        assert!(contains_whole("at 7:45.", "7:45"));
    }
}
