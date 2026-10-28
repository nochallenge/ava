// TESTING / DESIGN ONLY — v40: the magnitude code ("how much"), the reference rule on the CPU.
//
// People read a number's size on a compressed (log) mental number line, and know it roughly before they have
// read every digit: "4,8.." is already "about five thousand". Ava gets that as an input code beside v39's
// glance (length mark + place). For every token of a digit run whose length mark is in the row:
//   * the mark itself (length L = 1..15): m = L - 0.5, the middle of the run's decade range on the log line;
//   * the k-th digit (k = 0..L-1, value V of the digits read so far, r = L-1-k digits still to come):
//     m = log10(V + 1) when r = 0 (the whole number is known), else log10((V + 0.5) * 10^r + 1)
//     (the middle of what the number can still be).
// Every other token (and a run whose mark is missing, cut off by the row start, or 16+) gets no code (all 0).
// The code is MAG_F numbers: [1, m/16, sin(2 pi m / P), cos(2 pi m / P) for P = 1, 2, 4, 8, 16, 32]. A fixed
// input code, like the place code: what it means is learned (Model's mag_w [MAG_F, d], which starts at 0).
// The CUDA kernel `mag_feats` (lif.cu) computes the same numbers in double precision, one thread per token.

use ava_data::numbers::{is_digit, is_mark, MARK_BASE};

/// Numbers per token in the magnitude code.
pub const MAG_F: usize = 14;
/// Periods of the sine/cosine pairs on the log10 line.
pub const MAG_PERIODS: [f64; 6] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
/// Longest run (and mark) with a code; a mark of 16+ digits says too little.
const MAX_LEN: usize = 15;

/// The code of one position on the log line.
pub fn code(m: f64) -> [f32; MAG_F] {
    let mut f = [0f32; MAG_F];
    f[0] = 1.0;
    f[1] = (m / 16.0) as f32;
    for (j, p) in MAG_PERIODS.iter().enumerate() {
        let a = 2.0 * std::f64::consts::PI * m / p;
        f[2 + 2 * j] = a.sin() as f32;
        f[3 + 2 * j] = a.cos() as f32;
    }
    f
}

/// The log-line position of token i of `row`, or None (no code). Causal: reads only row[..=i].
pub fn position(row: &[u32], i: usize) -> Option<f64> {
    let tok = row[i];
    if is_mark(tok) {
        let len = (tok - MARK_BASE + 1) as usize;
        return (len <= MAX_LEN).then(|| len as f64 - 0.5);
    }
    if !is_digit(tok) {
        return None;
    }
    let mut k = 0usize;
    while k < 16 && i > k && is_digit(row[i - k - 1]) {
        k += 1;
    }
    if i == k || k >= 16 {
        return None; // the run reaches the row start (its mark is not in this row), or is too long
    }
    let mk = row[i - k - 1];
    if !is_mark(mk) {
        return None;
    }
    let len = (mk - MARK_BASE + 1) as usize;
    if len > MAX_LEN || k >= len {
        return None;
    }
    let mut v = 0f64;
    for &d in &row[i - k..=i] {
        v = v * 10.0 + (d - 48) as f64;
    }
    let r = (len - 1 - k) as i32;
    Some(if r == 0 { (v + 1.0).log10() } else { ((v + 0.5) * 10f64.powi(r) + 1.0).log10() })
}

/// The code of every token of one row, [row.len() * MAG_F] row-major.
pub fn feats_row(row: &[u32]) -> Vec<f32> {
    let mut out = vec![0f32; row.len() * MAG_F];
    for i in 0..row.len() {
        if let Some(m) = position(row, i) {
            out[i * MAG_F..(i + 1) * MAG_F].copy_from_slice(&code(m));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_follow_the_number() {
        // [mark 3] 4 8 2 -> 3-digit run "482"
        let row = [32u32, MARK_BASE + 2, 52, 56, 50, 32];
        assert_eq!(position(&row, 0), None);
        assert!((position(&row, 1).unwrap() - 2.5).abs() < 1e-12);
        assert!((position(&row, 2).unwrap() - (450.0f64 + 1.0).log10()).abs() < 1e-12); // "4.." -> 450
        assert!((position(&row, 3).unwrap() - (485.0f64 + 1.0).log10()).abs() < 1e-12); // "48." -> 485
        assert!((position(&row, 4).unwrap() - 483.0f64.log10()).abs() < 1e-12); // 482 + 1
        assert_eq!(position(&row, 5), None);
        // a run without its mark (cut by the row start) has no code
        assert_eq!(position(&[52u32, 56], 1), None);
        // the 16+ mark has none
        assert_eq!(position(&[MARK_BASE + 15], 0), None);
        let f = feats_row(&row);
        assert_eq!(f.len(), 6 * MAG_F);
        assert!(f[..MAG_F].iter().all(|&x| x == 0.0) && f[MAG_F] == 1.0);
    }
}
