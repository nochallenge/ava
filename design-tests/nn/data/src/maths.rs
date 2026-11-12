// TESTING / DESIGN ONLY
//! maths: C11's worked word problems (GSM8K train, MIT) as reading, and our own checker for numeric answers.
//!
//! NOTHING DOWNLOADED IS EVER RUN. "Checked by execution" here means only this: the calculator notes in a
//! worked solution (`<<48/2=24>>`) are re-computed by `eval_arith`, a small arithmetic evaluator written
//! here (numbers, + - * /, parentheses, unary minus), and a final answer is compared as a number by
//! `answer_matches`. Ava's own answers can be checked the same way later.
//!
//! INPUT. JSON lines `{"question": "...", "answer": "...steps...\n#### 72"}` (read by `json_strings`, a
//! flat-object reader for string fields; other value kinds are skipped).
//!
//! DOC. One problem per document, in her taught QA shape:
//!   "Question: {question}\nAnswer: {steps, calculator notes removed}\nThe answer is {final}."
//! A problem is kept only when its final answer is a number and every calculator note re-computes to its
//! stated result (within the rounding the result is written with).

use std::collections::BTreeMap;

/// The string fields of one flat JSON object (other value kinds are skipped). None when it is not one.
pub fn json_strings(s: &str) -> Option<BTreeMap<String, String>> {
    let b: Vec<char> = s.trim().chars().collect();
    let mut i = 0;
    let ws = |i: &mut usize| {
        while *i < b.len() && b[*i].is_whitespace() {
            *i += 1;
        }
    };
    let string = |i: &mut usize| -> Option<String> {
        if b.get(*i) != Some(&'"') {
            return None;
        }
        *i += 1;
        let mut out = String::new();
        while *i < b.len() {
            let c = b[*i];
            *i += 1;
            match c {
                '"' => return Some(out),
                '\\' => {
                    let e = *b.get(*i)?;
                    *i += 1;
                    match e {
                        'n' => out.push('\n'),
                        't' => out.push('\t'),
                        'r' => out.push('\r'),
                        'b' => out.push('\u{8}'),
                        'f' => out.push('\u{c}'),
                        'u' => {
                            let hex: String = b.get(*i..*i + 4)?.iter().collect();
                            *i += 4;
                            let mut cp = u32::from_str_radix(&hex, 16).ok()?;
                            if (0xD800..0xDC00).contains(&cp) && b.get(*i) == Some(&'\\') && b.get(*i + 1) == Some(&'u') {
                                let lo: String = b.get(*i + 2..*i + 6)?.iter().collect();
                                let lo = u32::from_str_radix(&lo, 16).ok()?;
                                if (0xDC00..0xE000).contains(&lo) {
                                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                    *i += 6;
                                }
                            }
                            out.push(char::from_u32(cp).unwrap_or('\u{fffd}'));
                        }
                        o => out.push(o),
                    }
                }
                o => out.push(o),
            }
        }
        None
    };
    ws(&mut i);
    if b.get(i) != Some(&'{') {
        return None;
    }
    i += 1;
    let mut out = BTreeMap::new();
    loop {
        ws(&mut i);
        if b.get(i) == Some(&'}') {
            return Some(out);
        }
        let k = string(&mut i)?;
        ws(&mut i);
        if b.get(i) != Some(&':') {
            return None;
        }
        i += 1;
        ws(&mut i);
        if b.get(i) == Some(&'"') {
            out.insert(k, string(&mut i)?);
        } else {
            // a number, true/false/null: skip to the next comma or the end (nested values are not expected)
            while i < b.len() && b[i] != ',' && b[i] != '}' {
                if matches!(b[i], '{' | '[') {
                    return None;
                }
                i += 1;
            }
        }
        ws(&mut i);
        match b.get(i) {
            Some(',') => i += 1,
            Some('}') => return Some(out),
            _ => return None,
        }
    }
}

/// Evaluates arithmetic: numbers (`3`, `2.5`, `.5`), + - * /, parentheses, unary minus. None on anything else
/// or on division by zero.
pub fn eval_arith(s: &str) -> Option<f64> {
    let t: Vec<char> = s.chars().filter(|c| !c.is_whitespace() && *c != ',').collect();
    let mut i = 0;
    let v = expr(&t, &mut i)?;
    (i == t.len() && v.is_finite()).then_some(v)
}

fn expr(t: &[char], i: &mut usize) -> Option<f64> {
    let mut v = term(t, i)?;
    while let Some(&c) = t.get(*i) {
        match c {
            '+' => {
                *i += 1;
                v += term(t, i)?;
            }
            '-' => {
                *i += 1;
                v -= term(t, i)?;
            }
            _ => break,
        }
    }
    Some(v)
}

fn term(t: &[char], i: &mut usize) -> Option<f64> {
    let mut v = factor(t, i)?;
    while let Some(&c) = t.get(*i) {
        match c {
            '*' | 'x' | '×' => {
                *i += 1;
                v *= factor(t, i)?;
            }
            '/' | '÷' => {
                *i += 1;
                let d = factor(t, i)?;
                if d == 0.0 {
                    return None;
                }
                v /= d;
            }
            _ => break,
        }
    }
    Some(v)
}

fn factor(t: &[char], i: &mut usize) -> Option<f64> {
    match t.get(*i)? {
        '-' => {
            *i += 1;
            Some(-factor(t, i)?)
        }
        '+' => {
            *i += 1;
            factor(t, i)
        }
        '(' => {
            *i += 1;
            let v = expr(t, i)?;
            if t.get(*i) != Some(&')') {
                return None;
            }
            *i += 1;
            Some(v)
        }
        _ => {
            let st = *i;
            while *i < t.len() && (t[*i].is_ascii_digit() || t[*i] == '.') {
                *i += 1;
            }
            let num: String = t[st..*i].iter().collect();
            if num.is_empty() {
                return None;
            }
            num.parse().ok()
        }
    }
}

/// A number as written: commas, a leading `$` and a trailing `%` or `.` dropped.
pub fn parse_number(s: &str) -> Option<f64> {
    let c: String = s.trim().trim_start_matches('$').trim_end_matches(['%', '.']).chars().filter(|&c| c != ',').collect();
    c.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// Digits after the decimal point in a written number.
fn decimals(s: &str) -> i32 {
    s.trim().split_once('.').map(|(_, d)| d.chars().take_while(|c| c.is_ascii_digit()).count() as i32).unwrap_or(0)
}

/// Does `value` round to the written result `shown`?
pub fn matches_shown(value: f64, shown: &str) -> bool {
    let Some(r) = parse_number(shown) else { return false };
    (value - r).abs() <= 0.5 * 10f64.powi(-decimals(shown)) + 1e-9 * r.abs().max(1.0)
}

/// The calculator notes `<<expr=result>>` of a solution: (expr, result) pairs.
pub fn notes(sol: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut rest = sol;
    while let Some(a) = rest.find("<<") {
        let Some(b) = rest[a..].find(">>") else { break };
        let inner = &rest[a + 2..a + b];
        if let Some((e, r)) = inner.rsplit_once('=') {
            out.push((e.to_string(), r.to_string()));
        } else {
            out.push((inner.to_string(), String::new()));
        }
        rest = &rest[a + b + 2..];
    }
    out
}

/// The solution with its calculator notes removed.
pub fn strip_notes(sol: &str) -> String {
    let mut out = String::with_capacity(sol.len());
    let mut rest = sol;
    while let Some(a) = rest.find("<<") {
        let Some(b) = rest[a..].find(">>") else { break };
        out.push_str(&rest[..a]);
        rest = &rest[a + b + 2..];
    }
    out.push_str(rest);
    out
}

/// The last number written in a text (for checking a free answer), as written.
pub fn last_number(text: &str) -> Option<String> {
    let mut best = None;
    let c: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < c.len() {
        let neg = c[i] == '-' && c.get(i + 1).is_some_and(|d| d.is_ascii_digit());
        if c[i].is_ascii_digit() || neg {
            let st = i;
            i += 1;
            while i < c.len() && (c[i].is_ascii_digit() || ((c[i] == ',' || c[i] == '.') && c.get(i + 1).is_some_and(|d| d.is_ascii_digit()))) {
                i += 1;
            }
            best = Some(c[st..i].iter().collect());
        } else {
            i += 1;
        }
    }
    best
}

/// Is the last number of `answer` equal to `gold`?
pub fn answer_matches(answer: &str, gold: &str) -> bool {
    match (last_number(answer).and_then(|s| parse_number(&s)), parse_number(gold)) {
        (Some(a), Some(g)) => (a - g).abs() <= 1e-6 * g.abs().max(1.0),
        _ => false,
    }
}

/// Why a problem was left out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reject {
    NotJson,
    NoFinal,
    BadNote,
    WrongNote,
}

/// One GSM8K line into (doc, number of notes checked), or why it was left out.
pub fn gsm8k_doc(line: &str) -> Result<(String, usize), Reject> {
    let o = json_strings(line).ok_or(Reject::NotJson)?;
    let (q, a) = (o.get("question").ok_or(Reject::NotJson)?, o.get("answer").ok_or(Reject::NotJson)?);
    let (steps, fin) = a.rsplit_once("####").ok_or(Reject::NoFinal)?;
    let fin = fin.trim();
    parse_number(fin).ok_or(Reject::NoFinal)?;
    let ns = notes(steps);
    for (e, r) in &ns {
        let v = eval_arith(e).ok_or(Reject::BadNote)?;
        if !matches_shown(v, r) {
            return Err(Reject::WrongNote);
        }
    }
    let steps: Vec<String> = strip_notes(steps).lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect();
    let q = q.split_whitespace().collect::<Vec<_>>().join(" ");
    Ok((format!("Question: {q}\nAnswer: {}\nThe answer is {fin}.", steps.join("\n")), ns.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic() {
        assert_eq!(eval_arith("48/2"), Some(24.0));
        assert_eq!(eval_arith("2+3*4"), Some(14.0));
        assert_eq!(eval_arith("(2+3)*4"), Some(20.0));
        assert_eq!(eval_arith("-3+5"), Some(2.0));
        assert_eq!(eval_arith("100*.2"), Some(20.0));
        assert_eq!(eval_arith("1,000*2"), Some(2000.0));
        assert_eq!(eval_arith("1/0"), None);
        assert_eq!(eval_arith("2^3"), None);
        assert_eq!(eval_arith("3*"), None);
        assert!(matches_shown(10.0 / 3.0, "3.33"));
        assert!(matches_shown(10.0 / 3.0, "3.3333333"));
        assert!(!matches_shown(10.0 / 3.0, "3.34"));
        // an integer result is read as rounded to a whole number
        assert!(matches_shown(10.0 / 3.0, "3"));
        assert!(!matches_shown(10.0 / 3.0, "4"));
        assert!(matches_shown(0.1, ".1"));
    }

    #[test]
    fn json_and_docs() {
        let l = r#"{"question": "Tom has 3 applesé and \"buys\" 2.", "answer": "He has 3+2=<<3+2=5>>5.\n#### 5", "n": 3}"#;
        let o = json_strings(l).unwrap();
        assert_eq!(o["question"], "Tom has 3 applesé and \"buys\" 2.");
        let (d, n) = gsm8k_doc(l).unwrap();
        assert_eq!(n, 1);
        assert_eq!(d, "Question: Tom has 3 applesé and \"buys\" 2.\nAnswer: He has 3+2=5.\nThe answer is 5.");
        assert_eq!(gsm8k_doc(r#"{"question": "q", "answer": "x <<3+2=6>>6\n#### 6"}"#), Err(Reject::WrongNote));
        assert_eq!(gsm8k_doc(r#"{"question": "q", "answer": "no final"}"#), Err(Reject::NoFinal));
        assert_eq!(json_strings(r#"{"a": [1]}"#), None);
        assert_eq!(json_strings("{\"a\": \"\\ud83d\\ude00\"}").unwrap()["a"], "\u{1F600}");
    }

    #[test]
    fn free_answers() {
        assert!(answer_matches("So she pays $1,250 in total.", "1250"));
        assert!(answer_matches("The answer is -4.", "-4"));
        assert!(answer_matches("It is 2.50 dollars", "2.5"));
        assert!(!answer_matches("I think 12, no, 13", "12"));
        assert!(!answer_matches("no number", "3"));
        assert_eq!(last_number("from 3 to 4.5."), Some("4.5".into()));
    }
}
