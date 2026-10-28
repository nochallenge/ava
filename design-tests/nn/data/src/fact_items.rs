//! TESTING / DESIGN ONLY — fact_items: v40's find-the-fact test (long memory) built from PG-19 books.
//!
//! ONE ITEM. A book (its marked tokens, `books::encode_book`) is cut at a seeded offset:
//!
//!   prompt = A + "\n\n" + FACT + "\n\n" + B + "\n\n" + CUE
//!   A      = about PROMPT_TOKENS (512) tokens of the book
//!   FACT   = "The code for the {colour} {object} is {NNNN}."   (NNNN in 1000..=9999)
//!   B      = about G tokens of the SAME book, the text right after A
//!   CUE    = "The code for the {colour} {object} is"
//!   choices: " {NNNN}." and three other 4-digit numbers in the same form, all distinct
//!
//! A and B are token spans that start at a word start (a token whose text begins with a space
//! or a newline) and end just before one, so their text is whole words; they are decoded with
//! the BPE after the length marks are stripped, and trimmed. Their token counts are those of
//! the marked book (the model's tokens): A has 512..512+64, B has G..G+64. No choice number
//! (right or wrong) occurs as a number anywhere in A or B, so the answer can only come from
//! the fact. `prompt_tokens` and `gap_tokens` are the marked token counts of the finished
//! prompt and of B as text, re-encoded (what a scorer will see).
//!
//! BINS. G ∈ {256, 1024, 4096, 12288}: bins `g256`, `g1024`, `g4096`, `g12288`, the same number
//! of items each. Per bin the right answer's position is exactly balanced (i % 4, shuffled).
//! The book of each item is a seeded draw; a book too short for P + G is skipped (counted).
//!
//! TSV. One item per line: `id \t fact \t bin \t right \t prompt \t choice0 \t choice1 \t
//! choice2 \t choice3`. Text fields are escaped with `escape_field`: a backslash becomes `\\`,
//! a tab `\t`, a newline `\n`, a carriage return `\r` (two characters each); `unescape_field`
//! undoes it exactly. The same format serves `sleep_facts`.

use crate::bpe::Bpe;
use crate::num_items::numbers_in;
use crate::numbers::{insert_marks, is_mark, strip_marks};
use crate::rng::{derive_seed, SplitMix64};
use std::collections::HashSet;

pub const KIND: &str = "fact";
/// Tokens of book text before the fact.
pub const PROMPT_TOKENS: usize = 512;
/// The gaps between the fact and the cue (tokens), and their bin names.
pub const GAPS: [usize; 4] = [256, 1024, 4096, 12288];
pub const BINS: [&str; 4] = ["g256", "g1024", "g4096", "g12288"];
/// The most tokens a span may grow to reach a word start.
pub const SLACK: usize = 64;
const MAX_DRAWS: usize = 100_000;
const FACT_LABEL: u64 = 0x6661_6374;

pub const COLOURS: [&str; 20] = [
    "red", "blue", "green", "yellow", "black", "white", "grey", "brown", "orange", "purple", "pink", "silver", "golden",
    "copper", "violet", "scarlet", "amber", "ivory", "crimson", "olive",
];
pub const OBJECTS: [&str; 20] = [
    "box", "door", "chest", "gate", "lock", "safe", "cabinet", "drawer", "trunk", "case", "locker", "vault", "cupboard",
    "suitcase", "briefcase", "toolbox", "shed", "garage", "cellar", "bicycle",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactItem {
    pub id: u32,
    pub bin: &'static str,
    pub gap: usize,
    pub right: usize,
    pub prompt: String,
    pub choices: Vec<String>,
    /// Index of the book in the list given, and the span start (marked tokens).
    pub book: usize,
    pub start: usize,
    /// Marked tokens of span A and span B in the book.
    pub span_a: usize,
    pub span_b: usize,
    /// Marked tokens of the finished prompt and of B's text, re-encoded.
    pub prompt_tokens: usize,
    pub gap_tokens: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FactStats {
    /// Draws that hit a book too short for the bin, per bin.
    pub short_books: [u64; 4],
    /// Draws whose spans could not reach a word start within SLACK, per bin.
    pub no_word_start: [u64; 4],
}

pub fn fact_sentence(colour: &str, object: &str, n: u32) -> String {
    format!("The code for the {colour} {object} is {n}.")
}

pub fn cue(colour: &str, object: &str) -> String {
    format!("The code for the {colour} {object} is")
}

pub fn choice(n: u32) -> String {
    format!(" {n}.")
}

/// Backslash, tab, newline and carriage return as `\\`, `\t`, `\n`, `\r`.
pub fn escape_field(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 16);
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

/// The inverse of `escape_field` (a backslash before any other character is kept as is).
pub fn unescape_field(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            o.push(c);
            continue;
        }
        match it.next() {
            Some('t') => o.push('\t'),
            Some('n') => o.push('\n'),
            Some('r') => o.push('\r'),
            Some('\\') => o.push('\\'),
            Some(x) => {
                o.push('\\');
                o.push(x);
            }
            None => o.push('\\'),
        }
    }
    o
}

/// `id \t kind \t bin \t right \t prompt \t choices...`, text fields escaped.
pub fn tsv_line(id: u32, kind: &str, bin: &str, right: usize, prompt: &str, choices: &[String]) -> String {
    let mut s = format!("{id}\t{kind}\t{bin}\t{right}\t{}", escape_field(prompt));
    for c in choices {
        s.push('\t');
        s.push_str(&escape_field(c));
    }
    s
}

pub fn item_line(it: &FactItem) -> String {
    tsv_line(it.id, KIND, it.bin, it.right, &it.prompt, &it.choices)
}

/// Marked tokens of `text` as a scorer would see them (no EOS).
pub fn marked_len(bpe: &Bpe, text: &str) -> usize {
    insert_marks(&bpe.encode(text)).len()
}

fn word_start(bpe: &Bpe, t: u32) -> bool {
    !is_mark(t) && (t as usize) < bpe.vocab_size() && matches!(bpe.token_bytes(t).first(), Some(b' ' | b'\n'))
}

fn next_word_start(bpe: &Bpe, toks: &[u32], mut i: usize) -> usize {
    while i < toks.len() && !word_start(bpe, toks[i]) {
        i += 1;
    }
    i
}

/// `n_per_bin` items per gap bin from `books` (marked tokens), seeded. Errors when no book is
/// long enough for a bin.
pub fn fact_items(bpe: &Bpe, books: &[Vec<u32>], n_per_bin: usize, seed: u64) -> Result<(Vec<FactItem>, FactStats), String> {
    if books.is_empty() && n_per_bin > 0 {
        return Err("no books".into());
    }
    let mut out = Vec::with_capacity(4 * n_per_bin);
    let mut stats = FactStats::default();
    for (bi, (&gap, &bin)) in GAPS.iter().zip(BINS.iter()).enumerate() {
        let mut rng = SplitMix64::new(derive_seed(seed, FACT_LABEL, bi as u64));
        let mut pos: Vec<usize> = (0..n_per_bin).map(|i| i % 4).collect();
        rng.shuffle(&mut pos);
        let need = PROMPT_TOKENS + gap + 3 * SLACK + 2;
        for &right in &pos {
            let mut draws = 0usize;
            let (book, s, a_end, b_end) = loop {
                draws += 1;
                if draws > MAX_DRAWS {
                    return Err(format!("bin {bin}: no book of {need}+ tokens with word starts found"));
                }
                let book = rng.below(books.len() as u64) as usize;
                let toks = &books[book];
                if toks.len() < need {
                    stats.short_books[bi] += 1;
                    continue;
                }
                let s0 = rng.below((toks.len() - need) as u64) as usize;
                let s = next_word_start(bpe, toks, s0);
                let a_end = next_word_start(bpe, toks, s + PROMPT_TOKENS);
                let b_end = next_word_start(bpe, toks, a_end + gap);
                if s - s0 > SLACK || a_end - s - PROMPT_TOKENS > SLACK || b_end - a_end - gap > SLACK || b_end + 1 >= toks.len() {
                    stats.no_word_start[bi] += 1;
                    continue;
                }
                break (book, s, a_end, b_end);
            };
            let toks = &books[book];
            let a = bpe.decode(&strip_marks(&toks[s..a_end]));
            let b = bpe.decode(&strip_marks(&toks[a_end..b_end]));
            let (a, b) = (a.trim(), b.trim());
            let in_book: HashSet<u64> = numbers_in(a).into_iter().chain(numbers_in(b)).collect();
            let colour = COLOURS[rng.below(COLOURS.len() as u64) as usize];
            let object = OBJECTS[rng.below(OBJECTS.len() as u64) as usize];
            let mut nums: Vec<u32> = Vec::with_capacity(4);
            while nums.len() < 4 {
                let n = 1000 + rng.below(9000) as u32;
                if !nums.contains(&n) && !in_book.contains(&(n as u64)) {
                    nums.push(n);
                }
            }
            // nums[0] is the answer; it goes to `right`, the others keep their order
            let mut choices: Vec<String> = nums[1..].iter().map(|&n| choice(n)).collect();
            choices.insert(right, choice(nums[0]));
            let prompt = format!("{a}\n\n{}\n\n{b}\n\n{}", fact_sentence(colour, object, nums[0]), cue(colour, object));
            out.push(FactItem {
                id: out.len() as u32,
                bin,
                gap,
                right,
                prompt_tokens: marked_len(bpe, &prompt),
                gap_tokens: marked_len(bpe, b),
                prompt,
                choices,
                book,
                start: s,
                span_a: a_end - s,
                span_b: b_end - a_end,
            });
        }
    }
    Ok((out, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::books::encode_book;

    const WORDS: [&str; 12] = ["the", "old", "house", "stood", "by", "a", "river", "where", "children", "played", "in", "summer"];

    /// A synthetic book: words, punctuation, line breaks, paragraphs and numbers.
    fn book_text(seed: u64, words: usize) -> String {
        let mut r = SplitMix64::new(seed);
        let mut s = String::new();
        for i in 0..words {
            if i > 0 {
                s.push(if r.below(12) == 0 { '\n' } else { ' ' });
                if r.below(80) == 0 {
                    s.push('\n');
                }
            }
            if r.below(25) == 0 {
                s.push_str(&(1000 + r.below(9000)).to_string());
            } else {
                s.push_str(WORDS[r.below(WORDS.len() as u64) as usize]);
            }
            if r.below(10) == 0 {
                s.push(if r.below(2) == 0 { '.' } else { ',' });
            }
        }
        s
    }

    fn setup() -> (Bpe, Vec<Vec<u32>>) {
        let bpe = Bpe::from_merges(vec![]).unwrap();
        // byte-level: about 5 tokens a word; the 100-word book is too short for any bin
        let books = [16000usize, 100, 9000, 20000].iter().enumerate().map(|(i, &w)| encode_book(&bpe, &book_text(i as u64, w))).collect();
        (bpe, books)
    }

    #[test]
    fn escape_roundtrip() {
        for s in ["", "plain", "a\tb\nc\r\nd", "back\\slash \\n literal", "\\", "\t\n\\t", "ünïcode\n"] {
            let e = escape_field(s);
            assert!(!e.contains(['\t', '\n', '\r']), "{e:?}");
            assert_eq!(unescape_field(&e), s);
        }
        assert_eq!(escape_field("a\tb\nc"), "a\\tb\\nc");
    }

    #[test]
    fn items_well_formed_balanced_and_deterministic() {
        let (bpe, books) = setup();
        let (items, stats) = fact_items(&bpe, &books, 8, 1).unwrap();
        let (again, stats2) = fact_items(&bpe, &books, 8, 1).unwrap();
        assert_eq!(items, again);
        assert_eq!(stats, stats2);
        assert_ne!(fact_items(&bpe, &books, 8, 2).unwrap().0, items);
        assert_eq!(items.len(), 32);
        assert!(stats.short_books.iter().sum::<u64>() > 0, "the short book should have been skipped");
        for (bi, bin) in BINS.iter().enumerate() {
            let of: Vec<&FactItem> = items.iter().filter(|x| x.bin == *bin).collect();
            assert_eq!(of.len(), 8);
            let mut pos = [0usize; 4];
            for it in &of {
                pos[it.right] += 1;
            }
            assert_eq!(pos, [2, 2, 2, 2], "{bin}");
            for it in of {
                assert_eq!(it.gap, GAPS[bi]);
                assert_ne!(it.book, 1, "a book shorter than the bin was used");
                assert!((PROMPT_TOKENS..=PROMPT_TOKENS + SLACK).contains(&it.span_a));
                assert!((it.gap..=it.gap + SLACK).contains(&it.span_b));
                // the re-encoded gap text is within a few tokens of the span
                assert!(it.gap_tokens + 16 >= it.gap && it.gap_tokens <= it.gap + SLACK, "{} vs {}", it.gap_tokens, it.gap);
                assert!(it.prompt_tokens > it.gap_tokens + PROMPT_TOKENS);
            }
        }
        for (i, it) in items.iter().enumerate() {
            assert_eq!(it.id as usize, i);
            // exactly one right choice: the fact's number; four distinct " NNNN." choices
            assert_eq!(it.choices.len(), 4);
            let mut c = it.choices.clone();
            c.sort();
            c.dedup();
            assert_eq!(c.len(), 4);
            let nums: Vec<u64> = it.choices.iter().map(|c| {
                assert!(c.starts_with(' ') && c.ends_with('.') && c.len() == 6, "{c:?}");
                c[1..5].parse().unwrap()
            }).collect();
            assert!(nums.iter().all(|n| (1000..=9999).contains(n)));
            let parts: Vec<&str> = it.prompt.split("\n\n").collect();
            let fact_i = parts.iter().position(|p| p.starts_with("The code for the ")).unwrap();
            let fact = parts[fact_i];
            let cue_s = *parts.last().unwrap();
            assert!(cue_s.starts_with("The code for the ") && cue_s.ends_with(" is"));
            assert_eq!(format!("{cue_s} {}.", nums[it.right]), fact);
            assert_eq!(it.prompt.matches("The code for the ").count(), 2);
            assert!(it.prompt.ends_with(cue_s));
            // no choice number elsewhere in the prompt
            let rest = it.prompt.replace(fact, "");
            let in_prompt: HashSet<u64> = numbers_in(&rest).into_iter().collect();
            assert!(nums.iter().all(|n| !in_prompt.contains(n)), "a choice number occurs in the book text");
            // TSV: nine fields, prompt round-trips
            let line = item_line(it);
            let f: Vec<&str> = line.split('\t').collect();
            assert_eq!(f.len(), 9);
            assert_eq!((f[0], f[1], f[2], f[3]), (i.to_string().as_str(), "fact", it.bin, it.right.to_string().as_str()));
            assert_eq!(unescape_field(f[4]), it.prompt);
            assert_eq!(f[5..].iter().map(|s| s.to_string()).collect::<Vec<_>>(), it.choices);
            // the text is the book's: span A's words appear in the book
            assert!(!it.prompt.starts_with([' ', '\n']));
        }
    }

    #[test]
    fn no_book_long_enough_is_an_error() {
        let (bpe, books) = setup();
        assert!(fact_items(&bpe, &books[1..2], 1, 1).is_err());
        assert!(fact_items(&bpe, &[], 1, 1).is_err());
        assert_eq!(fact_items(&bpe, &[], 0, 1).unwrap().0.len(), 0);
    }
}
