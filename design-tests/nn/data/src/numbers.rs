//! TESTING / DESIGN ONLY — numbers: v39's glance (length marks, place codes) and number words.
//!
//! THE GLANCE. Ava's BPE never merges digits, so a number is a run of single-digit tokens (ids
//! 48..=57). v39 lets her see a number's length before she reads it: `insert_marks` puts a
//! length mark before every maximal digit run (ids 4096..=4111; mark `MARK_BASE + L - 1` for a
//! run of L digits, the last mark meaning "16 or more"), so the vocabulary grows to 4112.
//! The model then adds a place embedding to every digit: `place_ids` reads each digit's place
//! (0 = units, 1 = tens, ...) off the mark in front of its run, or gives `PLACE_NONE` (16) when
//! that mark is not in the row (a window that starts inside a number), the run is 16 or more
//! digits long, or the digits overrun what the mark said. It only looks backwards, so it is
//! causal. The GPU side implements the identical rule in CUDA and tests against `place_ids`.
//!
//! NUMBER WORDS. `number_words(n)` spells 0..=999 the British way: "twenty-one",
//! "one hundred and five".

/// The first length mark: mark id `MARK_BASE + L - 1` stands for a digit run of length L.
pub const MARK_BASE: u32 = 4096;
/// Number of length marks (lengths 1..=15, and 16 meaning "16 or more").
pub const N_MARKS: u32 = 16;
/// Vocabulary size with the marks: 4096 BPE ids + 16 marks.
pub const VOCAB_GLANCE: usize = 4112;
/// The place code of every token that has no known place (non-digits, unmarked digits).
pub const PLACE_NONE: u32 = 16;

/// Longest run a mark can name exactly; a mark of this many or more is "16 or more".
const MAX_RUN: usize = N_MARKS as usize;

/// True for the ten digit tokens (their byte ids, 48..=57).
#[inline]
pub fn is_digit(id: u32) -> bool {
    (48..=57).contains(&id)
}

/// True for the 16 length marks (ids `MARK_BASE..MARK_BASE + N_MARKS`).
#[inline]
pub fn is_mark(id: u32) -> bool {
    (MARK_BASE..MARK_BASE + N_MARKS).contains(&id)
}

/// Append `ids` to `out` with a length mark before each maximal digit run (see
/// `insert_marks`). Returns the number of marks inserted.
pub fn insert_marks_into(ids: &[u32], out: &mut Vec<u32>) -> u64 {
    let mut marks = 0u64;
    let mut i = 0;
    while i < ids.len() {
        if is_digit(ids[i]) {
            let mut j = i + 1;
            while j < ids.len() && is_digit(ids[j]) {
                j += 1;
            }
            out.push(MARK_BASE + (j - i).min(MAX_RUN) as u32 - 1);
            out.extend_from_slice(&ids[i..j]);
            marks += 1;
            i = j;
        } else {
            out.push(ids[i]);
            i += 1;
        }
    }
    marks
}

/// Before each maximal run of digit tokens of length L insert `MARK_BASE + min(L, 16) - 1`.
/// Nothing else changes ("1,024" → mark1 1 , mark3 0 2 4).
pub fn insert_marks(ids: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(ids.len() + ids.len() / 8 + 1);
    insert_marks_into(ids, &mut out);
    out
}

/// Drop every length mark (the inverse of `insert_marks`).
pub fn strip_marks(ids: &[u32]) -> Vec<u32> {
    ids.iter().copied().filter(|&t| !is_mark(t)).collect()
}

/// The place code of every token of `row` (same length). For a digit at index i:
///   k = the number of digit tokens directly before i (counting back while the token is a
///       digit, at most 16, never past the row start);
///   if i > k and `row[i - k - 1]` is a mark of length L <= 15 and k < L: place = L - 1 - k;
///   otherwise `PLACE_NONE`.
/// Non-digits get `PLACE_NONE`. So a run whose mark was cut off by the row start, a run under
/// mark 4111 ("16 or more"), and digits beyond the length the mark said all get `PLACE_NONE`.
pub fn place_ids(row: &[u32]) -> Vec<u32> {
    let mut out = vec![PLACE_NONE; row.len()];
    for i in 0..row.len() {
        if !is_digit(row[i]) {
            continue;
        }
        let mut k = 0usize;
        while k < MAX_RUN && i > k && is_digit(row[i - k - 1]) {
            k += 1;
        }
        if i == k {
            continue; // the run reaches the row start: its mark is not in this row
        }
        let p = row[i - k - 1];
        if is_mark(p) {
            let len = (p - MARK_BASE + 1) as usize;
            if len < MAX_RUN && k < len {
                out[i] = (len - 1 - k) as u32;
            }
        }
    }
    out
}

const ONES: [&str; 20] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
];
const TENS: [&str; 10] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];

/// English words for 0..=999, British style ("one hundred and five", hyphenated tens:
/// "twenty-one"). Panics above 999.
pub fn number_words(n: u32) -> String {
    assert!(n <= 999, "number_words: {n} is above 999");
    let n = n as usize;
    if n < 20 {
        ONES[n].to_string()
    } else if n < 100 {
        if n % 10 == 0 {
            TENS[n / 10].to_string()
        } else {
            format!("{}-{}", TENS[n / 10], ONES[n % 10])
        }
    } else if n % 100 == 0 {
        format!("{} hundred", ONES[n / 100])
    } else {
        format!("{} hundred and {}", ONES[n / 100], number_words((n % 100) as u32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    #[test]
    fn number_words_cases() {
        assert_eq!(number_words(0), "zero");
        assert_eq!(number_words(13), "thirteen");
        assert_eq!(number_words(20), "twenty");
        assert_eq!(number_words(21), "twenty-one");
        assert_eq!(number_words(100), "one hundred");
        assert_eq!(number_words(105), "one hundred and five");
        assert_eq!(number_words(340), "three hundred and forty");
        assert_eq!(number_words(999), "nine hundred and ninety-nine");
        // all distinct
        let mut all: Vec<String> = (0..=999).map(number_words).collect();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 1000);
    }

    #[test]
    #[should_panic]
    fn number_words_above_999_panics() {
        number_words(1000);
    }

    #[test]
    fn insert_marks_one_comma_024() {
        // "1,024": 1 , 0 2 4
        assert_eq!(insert_marks(&[49, 44, 48, 50, 52]), vec![4096, 49, 44, 4098, 48, 50, 52]);
        // "3.14": each run its own mark
        assert_eq!(insert_marks(&[51, 46, 49, 52]), vec![4096, 51, 46, 4097, 49, 52]);
        assert_eq!(insert_marks(&[]), Vec::<u32>::new());
        assert_eq!(insert_marks(&[32, 256]), vec![32, 256]);
    }

    fn random_row(r: &mut SplitMix64, len: usize) -> Vec<u32> {
        let mut row = Vec::with_capacity(len);
        while row.len() < len {
            match r.below(4) {
                0 => {
                    // a digit run, sometimes very long
                    let l = if r.below(8) == 0 { 1 + r.below(25) } else { 1 + r.below(6) };
                    for _ in 0..l {
                        row.push(48 + r.below(10) as u32);
                    }
                }
                1 => row.push(r.below(4096) as u32),
                _ => row.push([32u32, 44, 46, 256, 300, 4095][r.below(6) as usize]),
            }
        }
        row.truncate(len);
        row
    }

    #[test]
    fn strip_inverts_insert_on_random_rows() {
        let mut r = SplitMix64::new(39);
        for _ in 0..2000 {
            let len = r.below(80) as usize;
            let row = random_row(&mut r, len);
            let marked = insert_marks(&row);
            assert_eq!(strip_marks(&marked), row);
            // every mark is followed by exactly the run it names
            for (i, &t) in marked.iter().enumerate() {
                if is_mark(t) {
                    let run = marked[i + 1..].iter().take_while(|&&x| is_digit(x)).count();
                    assert!(run >= 1);
                    assert_eq!(t, MARK_BASE + run.min(16) as u32 - 1);
                    assert!(i == 0 || !is_digit(marked[i - 1]));
                }
            }
            // every digit run is preceded by a mark
            for i in 0..marked.len() {
                if is_digit(marked[i]) && (i == 0 || !is_digit(marked[i - 1])) {
                    assert!(i > 0 && is_mark(marked[i - 1]));
                }
            }
        }
    }

    #[test]
    fn place_ids_cases() {
        assert_eq!(place_ids(&[4098, 49, 50, 51]), vec![16, 2, 1, 0]);
        // window starts mid-run: no mark in the row
        assert_eq!(place_ids(&[50, 51, 32]), vec![16, 16, 16]);
        // mark 2 then three digits: the overrun digit has no place
        assert_eq!(place_ids(&[4097, 49, 50, 51]), vec![16, 1, 0, 16]);
        // "1,024" with marks
        assert_eq!(place_ids(&[4096, 49, 44, 4098, 48, 50, 52]), vec![16, 0, 16, 16, 2, 1, 0]);
        // a mark followed by a non-digit, and a mark at the end
        assert_eq!(place_ids(&[4097, 32, 49, 4096]), vec![16, 16, 16, 16]);
        assert_eq!(place_ids(&[]), Vec::<u32>::new());
    }

    #[test]
    fn twenty_digit_run_gets_mark_4111_and_no_places() {
        let run: Vec<u32> = (0..20).map(|i| 48 + (i % 10)).collect();
        let marked = insert_marks(&run);
        assert_eq!(marked[0], 4111);
        assert_eq!(marked.len(), 21);
        assert!(place_ids(&marked).iter().all(|&p| p == PLACE_NONE));
        // exactly 15 digits is still named exactly; 16 is "16 or more"
        let r15: Vec<u32> = vec![49; 15];
        let m15 = insert_marks(&r15);
        assert_eq!(m15[0], 4110);
        let p15 = place_ids(&m15);
        assert_eq!(p15[1], 14);
        assert_eq!(p15[15], 0);
        let m16 = insert_marks(&[49; 16]);
        assert_eq!(m16[0], 4111);
        assert!(place_ids(&m16).iter().all(|&p| p == PLACE_NONE));
    }

    #[test]
    fn place_ids_on_marked_rows_and_their_windows() {
        let mut r = SplitMix64::new(7);
        for _ in 0..500 {
            let len = r.below(60) as usize;
            let row = insert_marks(&random_row(&mut r, len));
            let full = place_ids(&row);
            // on a fully marked row, every digit of a run of L <= 15 has place L - 1 - k
            let mut i = 0;
            while i < row.len() {
                if is_digit(row[i]) {
                    let mut j = i;
                    while j < row.len() && is_digit(row[j]) {
                        j += 1;
                    }
                    let l = j - i;
                    for k in 0..l {
                        let want = if l <= 15 { (l - 1 - k) as u32 } else { PLACE_NONE };
                        assert_eq!(full[i + k], want);
                    }
                    i = j;
                } else {
                    assert_eq!(full[i], PLACE_NONE);
                    i += 1;
                }
            }
            // any window (suffix) agrees with the full row or says PLACE_NONE
            for s in 0..row.len() {
                let w = place_ids(&row[s..]);
                for (j, &p) in w.iter().enumerate() {
                    assert!(p == full[s + j] || p == PLACE_NONE, "window {s} pos {j}");
                }
            }
        }
    }
}
