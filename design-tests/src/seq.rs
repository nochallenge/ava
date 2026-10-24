//! v8 — SEQUENCE. The mechanism she did not have.
//!
//! Everything up to v7 answers one shape of question: (lemma, tags) -> edit operation.
//! That is recall. It cannot represent "and then", so she can produce a word but never
//! two words in a row, and never a sentence.
//!
//! The claim being tested here is that the substrate is general: that the SAME
//! neurons, the SAME competitive identity allocation, the SAME shared category units
//! and the SAME specificity-weighted vote can hold a completely different relation,
//! and that only the encoder and readout have to change.
//!
//! The relation is now:
//!
//!     the words just said  ->  the word that comes next
//!
//! and the dual route falls out again, in a way that is almost too neat:
//!
//!   IDENTITY features = the exact context, "C3:i want to". Near-private, memorises
//!       this specific continuation. This is the n-gram.
//!   GRAMMAR features  = backed-off suffixes, "B2:want to", "B1:to", and the
//!       positional words. Shared by hundreds of thousands of contexts, so they
//!       carry what USUALLY follows.
//!
//! Specificity weighting then interpolates between them automatically: a context seen
//! often enough gets its memorised continuation, an unseen one falls back to the
//! shorter suffix, exactly as it falls back from "ate" to "+ed". The same mechanism
//! that makes an irregular verb an exception makes a memorised phrase an exception.
//! That is backoff smoothing, arrived at from the other direction.

use crate::v6::NeuronId;

pub const BOS: &str = "<s>";
pub const EOS: &str = "</s>";

/// how many scrolled-out words stay in working memory
pub const USED_MAX: usize = 5;

/// Split a raw sentence into tokens. Punctuation becomes its own token so she can
/// learn where it goes; everything is lowercased so "The" and "the" are one word.
pub fn tokenize(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if c.is_alphanumeric() || c == '\'' {
            cur.extend(c.to_lowercase());
        } else {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            if !c.is_whitespace() {
                out.push(c.to_string());
            }
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// The features of "here is the context, what comes next".
///
/// `ctx` is the preceding words, oldest first, most recent LAST.
pub fn feats(ctx: &[&str]) -> (Vec<String>, Vec<String>) {
    feats_opt(ctx, false)
}

/// `backoff` gives the IDENTITY route a ladder instead of one all-or-nothing key.
///
/// v9 found that a single exact-match identity key is brittle: when the exact context
/// was never seen, identity contributed ZERO units and the answer fell entirely to the
/// shared general features, which are hubs. Giving identity graded keys — most
/// specific first, competition takes whichever rungs exist — turned degenerate replies
/// into real ones. The sequence model has the same shape of key, so the same cure
/// should transfer. Whether it does is a measurement, not an argument.
pub fn feats_opt(ctx: &[&str], backoff: bool) -> (Vec<String>, Vec<String>) {
    // ---- identity: the exact context, one symbol. This is what memorises.
    let mut id = vec![format!("C{}:{}", ctx.len(), ctx.join(" "))];
    if backoff {
        // shorter suffixes as IDENTITY too — competitively allocated and near-private,
        // unlike the B: versions below which are shared hub units.
        for n in (1..ctx.len()).rev() {
            id.push(format!("C{n}:{}", ctx[ctx.len() - n..].join(" ")));
        }
    }

    // ---- general: every shorter suffix, plus each word at its own distance.
    // A suffix is shared by every context that ends the same way, so it holds the
    // usual continuation. The positional features let a word vote even when the
    // exact word order was never seen.
    let mut g: Vec<String> = Vec::with_capacity(ctx.len() * 2);
    for n in 1..ctx.len() {
        g.push(format!("B{}:{}", n, ctx[ctx.len() - n..].join(" ")));
    }
    for (i, w) in ctx.iter().enumerate() {
        g.push(format!("P{}:{}", ctx.len() - i, w)); // P1 = most recent word
    }
    (id, g)
}

/// The same, plus WORKING MEMORY: what has already been said in this sentence,
/// beyond the sliding window.
///
/// A window model cannot compose. "i want to be a ." happens because by the time she
/// reaches "a" the word "be" has scrolled out of view, and nothing records that she
/// has opened a noun phrase and owes a noun. That is not a size problem — a bigger
/// window only moves the boundary. It is a missing mechanism: sustained state.
///
/// So two more feature families, both GENERAL (shared, no competition), because they
/// describe the situation rather than identify the fact:
///
///   L:n   how many words in we are. This is what lets her learn to STOP; without it
///         the only signal for EOS is the local window, and sentences run on.
///   U:w   a word already used in this sentence. Persists for the whole sentence
///         however far back it was said, which is exactly what the window cannot do.
///
/// This is the closest thing here to prefrontal sustained activity: not a memory that
/// was stored and recalled, but a trace of the current episode held active while the
/// rest of the network works.
pub fn feats_state(ctx: &[&str], pos: usize, used: &[&str]) -> (Vec<String>, Vec<String>) {
    let (id, mut g) = feats(ctx);
    g.push(format!("L:{}", pos.min(20)));
    for w in used {
        g.push(format!("U:{w}"));
    }
    (id, g)
}

/// One training position: the context and the word that actually followed.
pub struct Step<'a> {
    pub ctx: Vec<&'a str>,
    pub next: &'a str,
    /// how many words into the sentence we are — needed for the L: feature
    pub pos: usize,
    /// words that have already scrolled OUT of the window, most recent last
    pub used: Vec<&'a str>,
}

/// Slide a window over one sentence, padded with BOS so she learns how to START and
/// EOS so she learns how to STOP. Without those two she can neither begin nor end.
pub fn steps<'a>(toks: &'a [String], order: usize, buf: &mut Vec<Step<'a>>) {
    buf.clear();
    if toks.is_empty() {
        return;
    }
    let mut ctx: Vec<&str> = vec![BOS; order];
    for (i, t) in toks.iter().enumerate() {
        let scrolled = i.saturating_sub(order);
        let used: Vec<&str> = toks[scrolled.saturating_sub(USED_MAX)..scrolled]
            .iter().map(|s| s.as_str()).collect();
        buf.push(Step { ctx: ctx.clone(), next: t.as_str(), pos: i, used });
        ctx.remove(0);
        ctx.push(t.as_str());
    }
    let i = toks.len();
    let scrolled = i.saturating_sub(order);
    let used: Vec<&str> = toks[scrolled.saturating_sub(USED_MAX)..scrolled]
        .iter().map(|s| s.as_str()).collect();
    buf.push(Step { ctx, next: EOS, pos: i, used });
}

pub fn code_of(idc: Vec<NeuronId>, gc: &[NeuronId]) -> Vec<NeuronId> {
    let mut c = idc;
    c.extend_from_slice(gc);
    c
}
