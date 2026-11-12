// TESTING / DESIGN ONLY — lookup (v43): her library, a fact store she searches before she answers.
//
// Like the hippocampus fetching what the cortex needs: a question's words cue the few passages that best match,
// and those become her context ("Context: ...\nQuestion: ...\nAnswer:", the format she was taught in v37).
//
// LIBRARY FILE (`build` writes it, `Library::load` reads it): UTF-8, one passage per line, `title \t text`.
// * The Simple English Wikipedia (`data/nn/docs/simplewiki.docs`, CC BY-SA 4.0): each article's title and its
//   opening text (headings skipped), cut at sentence ends into chunks of at most `CHUNK_WORDS` words, at most
//   `CHUNKS_PER_ARTICLE` per article. Lists, disambiguation pages and near-empty pages are left out.
// * Facts written for her (`topic \t fact` lines, the header skipped): one passage each, the topic as its title.
//
// SEARCH: BM25 (k1 1.2, b 0.75) over the passage words, plus a title bonus (each question word that is also a
// title word adds TITLE_BONUS · its idf, scaled by (share of the title the question covers)², so a short title the
// question names, like "France" or "Sky", counts fully and "NASCAR on Sky" hardly), so "capital of france" finds France.
// Words: lower case, letters and digits, a short list of question/function words dropped, "'s" and a plural "s"
// trimmed. A question made only of dropped words ("hi ava", "how are you") cues nothing: no context.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

pub const CHUNK_WORDS: usize = 60;
pub const CHUNKS_PER_ARTICLE: usize = 2;
const K1: f32 = 1.2;
const B: f32 = 0.75;
const TITLE_BONUS: f32 = 1.5;

const STOP: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am", "of", "in", "on", "at", "to", "for", "from",
    "by", "with", "and", "or", "but", "not", "no", "do", "does", "did", "doe", "what", "whats", "which", "who", "whom",
    "whose", "why", "how", "when", "where", "that", "this", "these", "those", "it", "its", "i", "you", "your", "u", "me",
    "my", "we", "our", "they", "them", "their", "he", "him", "his", "she", "her", "as", "if", "so", "than", "then",
    "there", "here", "can", "could", "would", "should", "will", "shall", "may", "might", "must", "have", "has", "had",
    "about", "into", "also", "very", "just", "ava", "tell", "please", "hey", "hi", "hello", "much", "many", "some", "any",
    "all", "know", "think", "like", "get", "got", "one", "thing", "things", "kind", "called", "call", "name", "mean",
    "means", "really", "ok", "okay", "yo", "s", "t", "re", "ll", "d", "m", "ve", "doesnt", "dont", "isnt", "arent", "im",
    "youre", "thats", "whos", "wheres", "hows", "whys",
];

/// The search words of a text, in order (with repeats).
pub fn words(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in s.split(|c: char| !c.is_alphanumeric() && c != '\'') {
        let w = raw.trim_matches('\'').to_lowercase();
        let w = w.strip_suffix("'s").unwrap_or(&w).replace('\'', "");
        if w.is_empty() || STOP.contains(&w.as_str()) {
            continue;
        }
        out.push(stem(&w));
    }
    out
}

fn stem(w: &str) -> String {
    let n = w.chars().count();
    if n > 4 && w.ends_with("ies") {
        format!("{}y", &w[..w.len() - 3])
    } else if n > 4 && (w.ends_with("lves") || w.ends_with("eaves")) {
        // leaves → leaf, wolves → wolf, halves → half, shelves → shelf
        format!("{}f", &w[..w.len() - 3])
    } else if n > 3 && w.ends_with('s') && !w.ends_with("ss") && !w.ends_with("us") && !w.ends_with("is") {
        w[..w.len() - 1].to_string()
    } else {
        w.to_string()
    }
}

/// Split a paragraph into sentences (at ". ", "! ", "? ", keeping the mark).
fn sentences(p: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let b = p.as_bytes();
    let mut start = 0;
    for i in 0..b.len() {
        if matches!(b[i], b'.' | b'!' | b'?') && (i + 1 == b.len() || b[i + 1] == b' ') {
            let s = p[start..=i].trim();
            if !s.is_empty() {
                out.push(s);
            }
            start = i + 1;
        }
    }
    let s = p[start..].trim();
    if !s.is_empty() {
        out.push(s);
    }
    out
}

/// One article's passages: (title, chunk) for its opening text. Empty for pages that are not articles.
pub fn article_passages(doc: &str) -> Vec<(String, String)> {
    let mut lines = doc.lines();
    let title = lines.next().unwrap_or("").trim().to_string();
    if title.is_empty() || title.starts_with("List of") || title.contains("(disambiguation)") {
        return Vec::new();
    }
    // paragraphs of real sentences (headings are short lines without an end mark)
    let mut paras: Vec<String> = Vec::new();
    for l in lines {
        let l = l.trim();
        if l.is_empty() {
            continue;
        }
        let ends = l.ends_with('.') || l.ends_with('!') || l.ends_with('?');
        if !ends && l.split_whitespace().count() < 8 {
            continue;
        }
        paras.push(l.to_string());
        if paras.len() >= 2 {
            break;
        }
    }
    let text = paras.join(" ");
    if text.split_whitespace().count() < 5 || text.contains("may refer to") || text.contains("can mean:") {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut cur: Vec<&str> = Vec::new();
    let mut n = 0;
    for s in sentences(&text) {
        let k = s.split_whitespace().count();
        if n > 0 && n + k > CHUNK_WORDS {
            out.push((title.clone(), cur.join(" ")));
            cur.clear();
            n = 0;
            if out.len() >= CHUNKS_PER_ARTICLE {
                return out;
            }
        }
        if k > CHUNK_WORDS {
            // one very long sentence: its first CHUNK_WORDS words
            let w: Vec<&str> = s.split_whitespace().take(CHUNK_WORDS).collect();
            out.push((title.clone(), w.join(" ")));
            if out.len() >= CHUNKS_PER_ARTICLE {
                return out;
            }
            continue;
        }
        cur.push(s);
        n += k;
    }
    if n > 0 && out.len() < CHUNKS_PER_ARTICLE {
        out.push((title, cur.join(" ")));
    }
    out
}

fn clean(s: &str) -> String {
    s.replace(['\t', '\n', '\r'], " ")
}

/// Write the library file from a docs file and (optionally) a facts TSV. Returns (articles used, passages).
pub fn build(docs: &Path, facts: Option<&Path>, out: &Path) -> Result<(usize, usize), String> {
    let e = |x: std::io::Error| x.to_string();
    let tmp = out.with_extension("tsv.tmp");
    let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp).map_err(e)?);
    let (mut arts, mut n) = (0usize, 0usize);
    if let Some(f) = facts {
        let rd = BufReader::new(std::fs::File::open(f).map_err(|x| format!("{}: {x}", f.display()))?);
        for (i, line) in rd.lines().enumerate() {
            let line = line.map_err(e)?;
            if i == 0 && line.starts_with("topic\t") {
                continue;
            }
            let Some((topic, fact)) = line.split_once('\t') else { continue };
            if fact.trim().is_empty() {
                continue;
            }
            writeln!(w, "{}\t{}", clean(topic.trim()), clean(fact.trim())).map_err(e)?;
            n += 1;
        }
    }
    let mut rd = ava_data::docs::open_docs(docs).map_err(|x| format!("{}: {x}", docs.display()))?;
    while let Some(doc) = rd.next_doc().map_err(e)? {
        let ps = article_passages(&doc);
        if ps.is_empty() {
            continue;
        }
        arts += 1;
        for (t, x) in ps {
            writeln!(w, "{}\t{}", clean(&t), clean(&x)).map_err(e)?;
            n += 1;
        }
    }
    w.flush().map_err(e)?;
    drop(w);
    std::fs::rename(&tmp, out).map_err(e)?;
    Ok((arts, n))
}

/// The library in memory: passages and their word index.
pub struct Library {
    titles: Vec<String>,
    texts: Vec<String>,
    lens: Vec<u16>,
    avg_len: f32,
    ids: HashMap<String, u32>,
    /// per word: (passage, count in its text)
    post: Vec<Vec<(u32, u16)>>,
    /// per word: passages whose title holds it
    tpost: Vec<Vec<u32>>,
    /// title words per passage
    tlens: Vec<u8>,
}

impl Library {
    pub fn load(path: &Path) -> Result<Library, String> {
        let rd = BufReader::new(std::fs::File::open(path).map_err(|x| format!("{}: {x}", path.display()))?);
        let mut pairs = Vec::new();
        for line in rd.lines() {
            let line = line.map_err(|x| x.to_string())?;
            if let Some((t, x)) = line.split_once('\t') {
                pairs.push((t.to_string(), x.to_string()));
            }
        }
        Ok(Library::from_pairs(pairs))
    }

    pub fn from_pairs(pairs: Vec<(String, String)>) -> Library {
        let mut lib = Library { titles: Vec::new(), texts: Vec::new(), lens: Vec::new(), avg_len: 1.0, ids: HashMap::new(), post: Vec::new(), tpost: Vec::new(), tlens: Vec::new() };
        let mut total = 0f64;
        let mut counts: HashMap<u32, u16> = HashMap::new();
        for (d, (t, x)) in pairs.into_iter().enumerate() {
            let d = d as u32;
            counts.clear();
            let ws = words(&x);
            for w in &ws {
                let id = lib.id_of(w);
                let c = counts.entry(id).or_insert(0);
                *c = c.saturating_add(1);
            }
            for (&id, &c) in &counts {
                lib.post[id as usize].push((d, c));
            }
            let mut seen: Vec<u32> = Vec::new();
            for w in words(&t) {
                let id = lib.id_of(&w);
                if !seen.contains(&id) {
                    seen.push(id);
                    lib.tpost[id as usize].push(d);
                }
            }
            lib.tlens.push(seen.len().clamp(1, 255) as u8);
            lib.lens.push(ws.len().min(u16::MAX as usize) as u16);
            total += ws.len() as f64;
            lib.titles.push(t);
            lib.texts.push(x);
        }
        lib.avg_len = (total / lib.texts.len().max(1) as f64).max(1.0) as f32;
        lib
    }

    fn id_of(&mut self, w: &str) -> u32 {
        if let Some(&id) = self.ids.get(w) {
            return id;
        }
        let id = self.post.len() as u32;
        self.ids.insert(w.to_string(), id);
        self.post.push(Vec::new());
        self.tpost.push(Vec::new());
        id
    }

    pub fn len(&self) -> usize {
        self.texts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }

    /// The best `k` passages for a question: (score, title, text), best first. Nothing if no word cues anything.
    pub fn search(&self, q: &str, k: usize) -> Vec<(f32, &str, &str)> {
        let n = self.texts.len() as f32;
        let mut qs = words(q);
        qs.sort();
        qs.dedup();
        let mut acc: HashMap<u32, f32> = HashMap::new();
        // per passage: (idf of the question words in its title, how many)
        let mut tacc: HashMap<u32, (f32, u32)> = HashMap::new();
        for w in &qs {
            let Some(&id) = self.ids.get(w) else { continue };
            let p = &self.post[id as usize];
            let df = p.len() as f32;
            if df == 0.0 {
                continue;
            }
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &(d, c) in p {
                let tf = c as f32;
                let len = self.lens[d as usize] as f32;
                *acc.entry(d).or_insert(0.0) += idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * len / self.avg_len));
            }
            for &d in &self.tpost[id as usize] {
                let e = tacc.entry(d).or_insert((0.0, 0));
                e.0 += idf;
                e.1 += 1;
            }
        }
        for (d, (idf_sum, hit)) in tacc {
            let cover = hit as f32 / self.tlens[d as usize] as f32;
            *acc.entry(d).or_insert(0.0) += TITLE_BONUS * idf_sum * cover * cover;
        }
        let mut v: Vec<(f32, u32)> = acc.into_iter().map(|(d, s)| (s, d)).collect();
        let k = k.min(v.len());
        if k == 0 {
            return Vec::new();
        }
        v.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        v.truncate(k);
        v.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        v.into_iter().map(|(s, d)| (s, self.titles[d as usize].as_str(), self.texts[d as usize].as_str())).collect()
    }

    /// Her context for a question: the best `k` passages as "Title: text", joined; empty if nothing is cued.
    pub fn context(&self, q: &str, k: usize) -> String {
        self.search(q, k).iter().map(|(_, t, x)| format!("{t}: {x}")).collect::<Vec<_>>().join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lib() -> Library {
        let p = |t: &str, x: &str| (t.to_string(), x.to_string());
        Library::from_pairs(vec![
            p("France", "France is a country in Western Europe. Its capital city is Paris."),
            p("Paris", "Paris is the capital and largest city of France."),
            p("Spider", "Spiders are animals with eight legs. Most spiders make webs."),
            p("animals", "A dog has four legs."),
            p("Sky", "The sky looks blue because air scatters blue sunlight more than red sunlight."),
        ])
    }

    #[test]
    fn words_drop_question_words_and_trim_plurals() {
        assert_eq!(words("How many legs does a spider have?"), vec!["leg", "spider"]);
        assert_eq!(words("whats France's capital"), vec!["france", "capital"]);
        assert_eq!(words("ponies and countries"), vec!["pony", "country"]);
        assert_eq!(words("leaves wolves"), vec!["leaf", "wolf"]);
        assert!(words("hey ava, how are you").is_empty());
    }

    #[test]
    fn the_right_passage_comes_first() {
        let l = lib();
        assert_eq!(l.search("how many legs does a spider have", 2)[0].1, "Spider");
        assert_eq!(l.search("what is the capital of france", 2)[0].1, "France");
        assert_eq!(l.search("why is the sky blue", 1)[0].1, "Sky");
        assert!(l.search("hi ava how are you", 2).is_empty(), "a greeting cues nothing");
        assert!(l.context("hello", 2).is_empty());
        assert!(l.context("dog legs", 1).starts_with("animals: A dog has four legs."));
    }

    #[test]
    fn an_article_gives_its_opening_in_chunks() {
        let doc = "April\n\nApril (Apr.) is the fourth month of the year. It has 30 days.\n\nThe Month\n\nApril comes between March and May.";
        let ps = article_passages(doc);
        assert_eq!(ps.len(), 1);
        assert_eq!(ps[0].0, "April");
        assert_eq!(ps[0].1, "April (Apr.) is the fourth month of the year. It has 30 days. April comes between March and May.");
        assert!(article_passages("Mercury\n\nMercury may refer to:\n\nA planet.").is_empty());
        assert!(article_passages("List of birds\n\nThese are birds of the world and more birds.").is_empty());
        let long = format!("Long\n\n{}", (0..30).map(|i| format!("Sentence number {i} is here.")).collect::<Vec<_>>().join(" "));
        let ps = article_passages(&long);
        assert_eq!(ps.len(), CHUNKS_PER_ARTICLE);
        assert!(ps.iter().all(|(_, x)| x.split_whitespace().count() <= CHUNK_WORDS));
    }
}

// ---------- grading her answers (v43 pre-registration: runs/v43/prereg_answers.md) ----------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grade {
    Right,
    Abstain,
    Wrong,
}

/// Lower case, every run of non-letters/digits one space, padded with a space each side.
fn norm(s: &str) -> String {
    let mut out = String::from(" ");
    for c in s.chars().flat_map(char::to_lowercase) {
        if c.is_alphanumeric() {
            out.push(c);
        } else if !out.ends_with(' ') {
            out.push(' ');
        }
    }
    if !out.ends_with(' ') {
        out.push(' ');
    }
    out
}

/// Right if her answer holds an accepted answer as whole words; else Abstain if she says she doesn't know; else Wrong.
pub fn grade(answer: &str, accept: &str) -> Grade {
    let a = norm(answer);
    if accept.split('|').map(norm).any(|x| x.trim() != "" && a.contains(&x)) {
        return Grade::Right;
    }
    if [" don t know ", " dont know ", " do not know ", " not sure ", " no idea "].iter().any(|m| a.contains(m)) {
        return Grade::Abstain;
    }
    Grade::Wrong
}

#[cfg(test)]
mod grade_tests {
    use super::*;

    #[test]
    fn grading_follows_the_preregistered_rules() {
        assert_eq!(grade("Grass is green.", "green"), Grade::Right);
        assert_eq!(grade("A spider has 8 legs.", "8|eight"), Grade::Right);
        assert_eq!(grade("It is greenish", "green"), Grade::Wrong, "whole words only");
        assert_eq!(grade("I don't know.", "paris"), Grade::Abstain);
        assert_eq!(grade("I'm not sure, maybe Lyon.", "paris"), Grade::Abstain);
        assert_eq!(grade("Paris, I think, but I'm not sure", "paris"), Grade::Right, "an accepted answer counts first");
        assert_eq!(grade("", "paris"), Grade::Wrong);
        assert_eq!(grade("The colour is BLUE!", "blue|color"), Grade::Right);
        assert_eq!(grade("chlorophyll fades", "chlorophyll"), Grade::Right);
    }
}
