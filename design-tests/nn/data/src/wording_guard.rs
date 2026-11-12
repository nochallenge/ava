// TESTING / DESIGN ONLY
//! wording_guard: keeps the exact v42 test wordings out of her training text.
//!
//! TEMPLATES. One per line, `{}` marks where the test puts a name, and a newline is written as the two
//! characters `\n`. The CORE of a template is its question without the answer frame: a leading "Q: " is
//! removed and the text is cut at the first newline. So "Q: What is the capital of {}?\nA:" has the core
//! "What is the capital of {}?". A template without "Q: " and without a newline is its own core.
//!
//! NAMES. Every name the test could put into a template: countries, capitals, continents, elements and
//! symbols (Wikidata), and every form of the hand-written facts. A template is FILLED with each name.
//!
//! HITS. Matching ignores the case of ASCII letters. A CORE HIT is a filled core found anywhere in a text.
//! A FULL HIT is a core hit of a "Q: " template that has "Q:" right before it (spaces or newlines may sit
//! between): that is the test's own prompt. The filled strings are never built: each template's fixed part
//! before `{}` (or the part after it, when nothing comes before) is searched for, and only at those places
//! are the names tried, so a text costs one fast search per template.

/// The longest run of consecutive words that are a single 0 or 1, with words as the 13-gram contamination
/// check cuts them (runs of ASCII letters and digits; a run of non-ASCII bytes is a word too; anything else
/// separates). The v42 parity items are strings of 15-16 random bits, so any training text with a run of
/// 13 or more such words (a number table like "1.0 1.1 -1.0 ...", a bit vector) can share a 13-word gram
/// with them by chance; the shard builder drops those documents.
pub fn longest_bit_run(text: &str) -> usize {
    let (mut best, mut run) = (0usize, 0usize);
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c.is_ascii_alphanumeric() || c >= 0x80 {
            let ascii = c < 0x80;
            let start = i;
            while i < b.len() && (if ascii { b[i].is_ascii_alphanumeric() } else { b[i] >= 0x80 }) {
                i += 1;
            }
            if i - start == 1 && (c == b'0' || c == b'1') {
                run += 1;
                best = best.max(run);
            } else {
                run = 0;
            }
        } else {
            i += 1;
        }
    }
    best
}

/// One test template.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Template {
    /// The line as written in the file.
    pub line: String,
    /// The core's text before `{}`, lowercase.
    pub prefix: String,
    /// The core's text after `{}`, lowercase.
    pub suffix: String,
    /// The template starts with "Q: ".
    pub qa: bool,
}

/// Turn the two-character escapes `\n` and `\t` into the characters.
pub fn unescape(s: &str) -> String {
    s.replace("\\n", "\n").replace("\\t", "\t")
}

/// Read templates (one per non-empty line).
pub fn parse_templates(text: &str) -> Result<Vec<Template>, String> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let l = line.trim_end_matches('\r');
        if l.trim().is_empty() {
            continue;
        }
        let raw = unescape(l);
        let (qa, body) = match raw.strip_prefix("Q: ") {
            Some(b) => (true, b),
            None => (false, raw.as_str()),
        };
        let core = body.split('\n').next().unwrap_or("");
        if core.matches("{}").count() != 1 {
            return Err(format!("template line {}: want exactly one {{}} in {core:?}", i + 1));
        }
        let (p, s) = core.split_once("{}").unwrap();
        if p.is_empty() && s.is_empty() {
            return Err(format!("template line {}: nothing but {{}}", i + 1));
        }
        out.push(Template { line: l.to_string(), prefix: p.to_ascii_lowercase(), suffix: s.to_ascii_lowercase(), qa });
    }
    Ok(out)
}

/// One filled template found in a text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Hit {
    pub template: usize,
    pub name: usize,
    /// Preceded by "Q:" (the test's own prompt).
    pub full: bool,
    /// Byte offset where the filled core starts.
    pub at: usize,
}

pub struct WordingGuard {
    pub templates: Vec<Template>,
    /// Names as given (unique ignoring ASCII case).
    shown: Vec<String>,
    /// The same names, lowercase.
    names: Vec<String>,
}

fn q_before(b: &[u8], pos: usize) -> bool {
    let mut i = pos;
    while i > 0 && b[i - 1].is_ascii_whitespace() {
        i -= 1;
    }
    i >= 2 && &b[i - 2..i] == b"q:"
}

impl WordingGuard {
    pub fn new(templates: Vec<Template>, names: impl IntoIterator<Item = String>) -> WordingGuard {
        let mut seen = std::collections::HashSet::new();
        let (mut shown, mut low) = (Vec::new(), Vec::new());
        for n in names {
            let n = n.trim().to_string();
            if n.is_empty() {
                continue;
            }
            let l = n.to_ascii_lowercase();
            if seen.insert(l.clone()) {
                shown.push(n);
                low.push(l);
            }
        }
        WordingGuard { templates, shown, names: low }
    }

    /// Read the templates file and take the names.
    pub fn load(templates_path: impl AsRef<std::path::Path>, names: impl IntoIterator<Item = String>) -> std::io::Result<WordingGuard> {
        let p = templates_path.as_ref();
        let text = std::fs::read_to_string(p).map_err(|e| std::io::Error::new(e.kind(), format!("{}: {e}", p.display())))?;
        let t = parse_templates(&text).map_err(std::io::Error::other)?;
        Ok(WordingGuard::new(t, names))
    }

    pub fn n_names(&self) -> usize {
        self.names.len()
    }

    pub fn name(&self, i: usize) -> &str {
        &self.shown[i]
    }

    /// A template's core filled with a name, as the test would write it (for messages and tests).
    pub fn filled_core(&self, t: usize, name: &str) -> String {
        let raw = unescape(&self.templates[t].line);
        let body = raw.strip_prefix("Q: ").unwrap_or(&raw);
        body.split('\n').next().unwrap_or("").replacen("{}", name, 1)
    }

    /// Every hit in a text.
    pub fn hits(&self, text: &str) -> Vec<Hit> {
        let mut out = Vec::new();
        let low = text.to_ascii_lowercase();
        let b = low.as_bytes();
        for (ti, t) in self.templates.iter().enumerate() {
            let (pre, suf) = (t.prefix.as_bytes(), t.suffix.as_bytes());
            if !pre.is_empty() {
                for (pos, _) in low.match_indices(t.prefix.as_str()) {
                    let after = pos + pre.len();
                    for (ni, n) in self.names.iter().enumerate() {
                        let end = after + n.len();
                        if end + suf.len() <= b.len() && &b[after..end] == n.as_bytes() && &b[end..end + suf.len()] == suf {
                            out.push(Hit { template: ti, name: ni, full: t.qa && q_before(b, pos), at: pos });
                        }
                    }
                }
            } else {
                for (pos, _) in low.match_indices(t.suffix.as_str()) {
                    for (ni, n) in self.names.iter().enumerate() {
                        if pos >= n.len() && &b[pos - n.len()..pos] == n.as_bytes() {
                            let start = pos - n.len();
                            out.push(Hit { template: ti, name: ni, full: t.qa && q_before(b, start), at: start });
                        }
                    }
                }
            }
        }
        out
    }

    /// (core hits, full hits) in a text.
    pub fn count(&self, text: &str) -> (u32, u32) {
        let h = self.hits(text);
        (h.len() as u32, h.iter().filter(|x| x.full).count() as u32)
    }

    /// A readable description of a hit.
    pub fn describe(&self, h: &Hit) -> String {
        format!("template {} filled with {:?}{}", h.template + 1, self.shown[h.name], if h.full { " (with Q:)" } else { "" })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guard() -> WordingGuard {
        let t = parse_templates("Q: What is the capital of {}?\\nA:\nQ: {} is the capital of which country?\\nA:\nIf you flew to {} you would land in\n\n").unwrap();
        WordingGuard::new(t, ["France".to_string(), "Paris".to_string(), "Niger".to_string(), "france".to_string()])
    }

    #[test]
    fn bit_runs_count_words_as_the_gram_check_does() {
        assert_eq!(longest_bit_run("Count the ones in 1 0 0 1 0 1. The number"), 6);
        assert_eq!(longest_bit_run("1.0 1.1 -1.0 0.1"), 8);
        assert_eq!(longest_bit_run("0, 0, 1 x 1 0"), 3);
        assert_eq!(longest_bit_run("10 01 101 2 0"), 1);
        assert_eq!(longest_bit_run("1 é 0 1"), 2, "a non-ASCII word breaks the run");
        assert_eq!(longest_bit_run("no digits here"), 0);
    }

    #[test]
    fn templates_parse_to_cores() {
        let g = guard();
        assert_eq!(g.templates.len(), 3);
        assert_eq!((g.templates[0].prefix.as_str(), g.templates[0].suffix.as_str(), g.templates[0].qa), ("what is the capital of ", "?", true));
        assert_eq!((g.templates[1].prefix.as_str(), g.templates[1].suffix.as_str()), ("", " is the capital of which country?"));
        assert!(!g.templates[2].qa);
        assert_eq!(g.n_names(), 3, "names are unique ignoring case");
        assert_eq!(g.filled_core(0, "Peru"), "What is the capital of Peru?");
        assert!(parse_templates("no slot here").is_err());
        assert!(parse_templates("{} and {}").is_err());
    }

    #[test]
    fn finds_core_and_full_hits_ignoring_case() {
        let g = guard();
        assert_eq!(g.count("Quiz time. what is the capital of FRANCE? Paris."), (1, 0));
        assert_eq!(g.count("Q: What is the capital of France?\nA: Paris"), (1, 1));
        assert_eq!(g.count("Q:\n What is the capital of France?"), (1, 1));
        assert_eq!(g.count("Q: Paris is the capital of which country?\nA:"), (1, 1));
        assert_eq!(g.count("Then: Paris is the capital of which country? France."), (1, 0));
        assert_eq!(g.count("if you flew to niger you would land in"), (1, 0));
        // near misses are not hits
        assert_eq!(g.count("What is the capital of Nigeria?"), (0, 0));
        assert_eq!(g.count("What is the capital city of France?"), (0, 0));
        assert_eq!(g.count("The capital of France is Paris. Paris is the capital of France."), (0, 0));
        assert_eq!(g.count("Do you know the capital of France? It is Paris."), (0, 0));
        let h = g.hits("x. What is the capital of France?")[0];
        assert_eq!((h.at, g.name(h.name)), (3, "France"));
    }
}
