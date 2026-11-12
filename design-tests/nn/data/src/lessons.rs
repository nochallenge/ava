// TESTING / DESIGN ONLY
//! lessons: the owner's early lessons (question, answer) as teaching text in her taught QA format.
//!
//! INPUT. TSV files with the header `question<TAB>answer`, one lesson per line, exactly two fields. The
//! two-character escapes `\t`, `\n` and `\\` are read as a tab, a newline and a backslash (the escaping of
//! Ava's item TSVs), so a lesson may hold them; a line with any other field count is an error.
//!
//! TEXT. One lesson is one document, written exactly as the gpu crate's sleep episodes and QA teaching
//! (`qa::prompt_text` + `diary::qa_episode`): "Question: {q}\nAnswer: {a}", each side with its whitespace
//! runs joined to one space and trimmed. So what she reads here has the same shape as what she is asked.

use std::io;
use std::path::{Path, PathBuf};

/// The header every lessons file starts with.
pub const HEADER: &str = "question\tanswer";

/// Undo the TSV escapes `\t`, `\n`, `\\` (any other backslash stays as it is).
pub fn unescape_field(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('\\') => out.push('\\'),
            Some(o) => {
                out.push('\\');
                out.push(o);
            }
            None => out.push('\\'),
        }
    }
    out
}

/// Whitespace runs to one space, trimmed (the gpu crate's `diary::one_line`).
pub fn one_line(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// One lesson in her taught format: "Question: q\nAnswer: a".
pub fn lesson_text(question: &str, answer: &str) -> String {
    format!("Question: {}\nAnswer: {}", one_line(question), one_line(answer))
}

/// Parse one lessons file's text. `name` is used in error messages.
pub fn parse_lessons(text: &str, name: &str) -> Result<Vec<(String, String)>, String> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut lines = text.lines();
    match lines.next().map(|h| h.trim_end_matches('\r')) {
        Some(HEADER) => {}
        other => return Err(format!("{name}: header must be {HEADER:?}, found {other:?}")),
    }
    let mut out = Vec::new();
    for (i, line) in lines.enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 2 {
            return Err(format!("{name} line {}: want 2 tab-separated fields, found {}", i + 2, f.len()));
        }
        let (q, a) = (unescape_field(f[0]), unescape_field(f[1]));
        if q.trim().is_empty() || a.trim().is_empty() {
            return Err(format!("{name} line {}: empty question or answer", i + 2));
        }
        out.push((q, a));
    }
    Ok(out)
}

/// The lesson files in a folder: `lessons_*.tsv`, sorted by name.
pub fn lesson_files(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            let n = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            n.starts_with("lessons_") && n.ends_with(".tsv")
        })
        .collect();
    v.sort();
    Ok(v)
}

/// Every lesson of every file, in file order, as documents.
pub fn lesson_docs(files: &[PathBuf]) -> io::Result<Vec<String>> {
    let mut docs = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(f).map_err(|e| io::Error::new(e.kind(), format!("{}: {e}", f.display())))?;
        let name = f.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        for (q, a) in parse_lessons(&text, name).map_err(io::Error::other)? {
            docs.push(lesson_text(&q, &a));
        }
    }
    Ok(docs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_taught_format() {
        // the gpu crate's qa_episode("Why  is\nthe sky blue?", " Light\n\nscatters. ") gives exactly this
        assert_eq!(lesson_text("Why  is\nthe sky blue?", " Light\n\nscatters. "), "Question: Why is the sky blue?\nAnswer: Light scatters.");
    }

    #[test]
    fn parses_and_unescapes() {
        let t = "question\tanswer\r\nWhat is 2 and 3?\t5\r\n\r\nSay a\\tb\tline one\\nline two \\\\ end\n";
        let v = parse_lessons(t, "x").unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0], ("What is 2 and 3?".to_string(), "5".to_string()));
        assert_eq!(v[1].0, "Say a\tb");
        assert_eq!(v[1].1, "line one\nline two \\ end");
        assert_eq!(lesson_text(&v[1].0, &v[1].1), "Question: Say a b\nAnswer: line one line two \\ end");
        assert_eq!(unescape_field("a\\qb\\"), "a\\qb\\");
    }

    #[test]
    fn rejects_bad_files() {
        assert!(parse_lessons("q\ta\nx\ty\n", "x").is_err());
        assert!(parse_lessons("question\tanswer\nonly one field\n", "x").is_err());
        assert!(parse_lessons("question\tanswer\na\tb\tc\n", "x").is_err());
        assert!(parse_lessons("question\tanswer\n \tb\n", "x").is_err());
        assert_eq!(parse_lessons("\u{feff}question\tanswer\n", "x").unwrap().len(), 0);
    }
}
