//! v9 — REPLY. Rung 4: answer a person instead of continuing a sentence.
//!
//! v8 learned `the words just said -> the word that comes next`. That completes a
//! sentence, which is not the same thing as answering one. Continuation has one
//! source of context; a reply has two, and they play different roles:
//!
//!     what THEY said   +   what I have said of my reply so far   ->   my next word
//!
//! The prompt is a standing condition on the whole reply — it must still be exerting
//! influence at the last word — while the reply-so-far is a local sequence exactly as
//! in v8. So the two get encoded differently on purpose:
//!
//!   PROMPT  -> only as GENERAL features, as a bag of content words. It is shared by
//!              thousands of exchanges ("anyone who said 'thanks'"), and it must not
//!              scroll out of view, so it cannot be part of the local window.
//!   REPLY   -> the v8 treatment. exact context as IDENTITY (memorises what was
//!              actually said back), backed-off suffixes as GENERAL.
//!
//! One thing learned the hard way in v8 and applied here from the start: an
//! UNCROSSED feature is not neutral. `U:word` failed because "this word appeared"
//! says nothing about what should follow, so it votes for the globally most common
//! continuation. Every prompt feature here is therefore crossed with the position in
//! the reply, so a prompt word can only ever speak about where the reply currently is.

use crate::seq::{BOS, EOS};

/// Words worth conditioning on. Function words appear in nearly every prompt, so
/// they identify nothing and only add hub units — the v8 lesson about uncrossed
/// features, applied before rather than after measuring.
pub fn is_content(w: &str) -> bool {
    !matches!(
        w,
        "the" | "a" | "an" | "of" | "to" | "and" | "is" | "are" | "was" | "were"
            | "in" | "on" | "at" | "it" | "this" | "that" | "i" | "you" | "he"
            | "she" | "we" | "they" | "be" | "do" | "does" | "did" | "have" | "has"
            | "," | "." | "?" | "!" | "'" | "s" | "t"
    ) && w.len() > 1
}

/// How far into the reply we are, bucketed. Crossing the prompt with this is what
/// keeps a prompt word from voting for the same thing at every position.
fn slot(pos: usize) -> &'static str {
    match pos {
        0 => "0",
        1 => "1",
        2..=3 => "2",
        4..=7 => "3",
        _ => "4",
    }
}

/// Features for "they said P, I have said R so far, what do I say next?"
pub fn feats(prompt: &[String], ctx: &[&str], pos: usize) -> (Vec<String>, Vec<String>) {
    feats_hist(prompt, &[], ctx, pos)
}

/// The same, plus CONVERSATIONAL MEMORY: the turn before this one.
///
/// Without it every reply is independent. "i have a dog" then "what is its name ?"
/// and she has no idea what "its" refers to, because the first turn is simply gone.
/// That is the difference between replying and conversing.
///
/// `hist` is the previous turn, encoded as GENERAL features only and crossed with the
/// reply slot — the same discipline as the prompt, for the same reason. History words
/// uncrossed would be pure hubs; that mistake has been made three times in this
/// project already and does not need making a fourth.
///
/// History is deliberately WEAKER than the current prompt: only its content words, no
/// exact-sequence key, and marked `H:` so it can never be confused with `Q:`. What is
/// being asked right now should dominate what was asked a moment ago.
pub fn feats_hist(prompt: &[String], hist: &[String], ctx: &[&str], pos: usize)
    -> (Vec<String>, Vec<String>)
{
    feats_mask(prompt, hist, ctx, pos, 0xF)
}

/// The same, plus DISTRIBUTIONAL NEIGHBOUR KEYS for the word being responded to.
///
/// v11 measured the ceiling as generalisation: 98.66% on training positions, 35.5%
/// on held-out ones, and every change that ever helped this project made stored
/// knowledge easier to REACH rather than adding machinery. The deepest limit on
/// reaching it is that every address is a literal word — `QW:dog` and `QW:puppy` are
/// unrelated symbols.
///
/// `nbrs` maps a word to its distributional neighbours, learned by counting which
/// words share contexts (see `wordclass`). Emitting them as extra identity keys means
/// a reply filed under "tuesday" is reachable by "monday". No new mechanism: the same
/// key format, just more doors into the same room. Identity keys are free (v9 §3e).
pub fn feats_near(
    prompt: &[String], ctx: &[&str], pos: usize,
    nbrs: &std::collections::HashMap<String, Vec<String>>,
) -> (Vec<String>, Vec<String>) {
    let (mut id, g) = feats_mask(prompt, &[], ctx, pos, 0xF);
    let rctx = ctx.join(" ");
    if let Some(last) = prompt.iter().rev().find(|w| is_content(w)) {
        if let Some(ns) = nbrs.get(last) {
            for n in ns {
                id.push(format!("QW:{n}|R:{rctx}"));
            }
        }
    }
    (id, g)
}

/// ABLATION. `mask` selects which GENERAL feature families are emitted:
///   bit 0  B:  backed-off suffixes of my reply so far
///   bit 1  P:  each word of my reply context at its own distance
///   bit 2  Q:  their prompt's content words, crossed with the reply slot
///   bit 3  QL: their prompt's LAST content word, crossed with the slot
///
/// The measured rule from §3e is that general features are TAXED — each one adds
/// units to the code and dilutes the specificity-weighted vote — so some of the ones
/// already here may be net negative. Deleting `S:slot|L:len` was worth +12.6 points.
/// This finds out which of the rest are carrying their weight.
pub fn feats_mask(
    prompt: &[String], hist: &[String], ctx: &[&str], pos: usize, mask: u32,
) -> (Vec<String>, Vec<String>) {
    // ---- IDENTITY: what makes this exact situation this one.
    // The full prompt plus my exact reply-context. Near-private: it memorises what
    // was actually said back to this exact opener. This is the route that produces a
    // real remembered reply rather than a generic one.
    // MEASURED. This used to be a single all-or-nothing key: the entire prompt string
    // joined to the entire reply context. Splitting the vote by route showed exactly
    // what that costs:
    //
    //   "do you like music ?"  id[6u] well 65.0   -> "well , it depends ."   (verbatim
    //                                                 in the corpus, identity fires)
    //   "i love you"           id[0u] -           -> "i . . ."   (not verbatim, zero
    //                                                 identity units, hubs decide)
    //
    // She could only converse about sentences she had seen word for word. Everything
    // else fell through to two general routes that are hubs, which is precisely the
    // degenerate filler.
    //
    // This is v6's addressing bug one level up. There the rule was "whatever addresses
    // a memory at write time must be available at read time"; here the address IS
    // available and simply never recurs. So identity gets the backoff ladder that
    // every other route in this system already has — most specific first, and the
    // competition takes whichever rungs actually exist:
    //
    //   the exact prompt          — verbatim repeats, strongest
    //   its content words, sorted — paraphrases and reorderings
    //   its last content word     — the thing being responded to; nearly always recurs
    let key: Vec<&str> = prompt.iter().map(|s| s.as_str()).collect();
    let rctx = ctx.join(" ");
    let mut id = vec![format!("Q:{}|R:{rctx}", key.join(" "))];

    let mut content: Vec<&str> = prompt.iter().map(|s| s.as_str())
        .filter(|w| is_content(w)).collect();
    let last_q = content.last().copied();
    if !content.is_empty() {
        let last = *content.last().unwrap();
        content.sort_unstable();
        content.dedup();
        id.push(format!("QS:{}|R:{rctx}", content.join(",")));
        id.push(format!("QW:{last}|R:{rctx}"));
    }

    // ---- CONVERSATIONAL MEMORY, and it belongs HERE, in identity.
    //
    // The first attempt put the previous turn in the general route. Measured, it made
    // her worse everywhere and 6x worse on the referring words it was built for:
    // 24.51% -> 23.69% overall, 3.37% -> 0.56% on pronouns.
    //
    // Reading the allocator afterwards, the asymmetry is structural:
    //
    //     self.id.allocate(id, k_id)   ALWAYS returns exactly k_id units
    //     self.gram.code(gram)         returns per_tag units PER FEATURE
    //
    // identity features are FREE — the competition picks a fixed number of winners
    // however many keys you supply, and a better key just wins. every general feature
    // adds two more units to the code, diluting the specificity-weighted vote and
    // costing memory. three failed additions in this project were general features;
    // the one that worked was identity.
    if let (Some(h), Some(q)) = (hist.iter().rev().find(|w| is_content(w)), last_q) {
        // the pair that carries reference: "dog" then "name" means something neither
        // does alone. an identity key, so it competes rather than dilutes.
        id.push(format!("HQ:{h}>{q}|R:{rctx}"));
    }

    let mut g: Vec<String> = Vec::with_capacity(prompt.len() + ctx.len() * 2 + 2);

    // ---- GENERAL, from my reply so far: the v8 backoff ladder.
    //
    // MEASURED, and it refuted my own rule. I removed the all-BOS features on the
    // argument that at position 0 the context is `<s> <s> <s>` for every reply in the
    // corpus, so they carry no information and must be voting for the mode. That cost
    // **11.8 points of top-1 accuracy** (38.0% -> 26.2%) and did not fix generation.
    //
    // The rule as I stated it -- "a feature that takes the same value in every
    // training example carries no information" -- is too strong. These features are
    // not constant across examples; they are constant across examples AT THE SAME
    // POSITION, and position is strongly predictive: they encode the prior over how
    // replies OPEN. That prior is exactly what a prompt with weak content words needs
    // to fall back on.
    //
    // The distinction that actually separates the four bugs: a hub is harmful when it
    // is constant across the thing you are trying to DISCRIMINATE. `U:word` and
    // `S:slot|L:len` were constant across which WORD comes next, which is the
    // decision. The BOS window is constant across replies but varies with position,
    // and position changes the decision.
    if mask & 1 != 0 {
        for n in 1..ctx.len() {
            g.push(format!("B{}:{}", n, ctx[ctx.len() - n..].join(" ")));
        }
    }
    if mask & 2 != 0 {
        for (i, w) in ctx.iter().enumerate() {
            g.push(format!("P{}:{}", ctx.len() - i, w));
        }
    }

    // ---- GENERAL, from their prompt: a bag of content words, EACH CROSSED with how
    // far into the reply we are. Uncrossed, "they said dinner" would vote the same
    // way at word 1 and word 9 and would just pull toward the modal word. Crossed, it
    // says "when someone mentions dinner, replies tend to open like this".
    let s = slot(pos);
    if mask & 4 != 0 {
        for w in prompt.iter().filter(|w| is_content(w)) {
            g.push(format!("Q:{w}|S:{s}"));
        }
    }
    // the last content word of the prompt is the one most often being responded to
    if mask & 8 != 0 {
        if let Some(last) = prompt.iter().rev().find(|w| is_content(w)) {
            g.push(format!("QL:{last}|S:{s}"));
        }
    }
    // History goes in the IDENTITY route, not here. See `feats_hist` — putting it in
    // the general route was measured and it cost 24.51% -> 23.69% overall and
    // 3.37% -> 0.56% on exactly the referring words it was built for.

    // NO bare position/length feature here.
    //
    // It was `S:{slot}|L:{prompt_len}` — position in the reply crossed with how long
    // their sentence was, and NO content whatsoever. Every exchange with those two
    // numbers shares it, so it is a pure hub that votes for the globally modal token
    // at that slot, which is "." . Measured: prompts whose content words were rare
    // collapsed to `i . . .` and `. . . .` while prompts with strong content words
    // answered fine.
    //
    // That is the THIRD time this exact bug has appeared here — uncrossed shape
    // features in v6, `U:word` in v8, and this. Every prompt feature above is crossed
    // with the slot precisely to avoid it, and then this line put a bare hub back
    // underneath them.
    //
    //   an unconditioned feature is not neutral. it is a vote for the most common
    //   answer, and it gets one vote per occurrence.

    (id, g)
}

/// Every training position inside one reply: BOS-padded context, EOS-terminated.
pub struct Turn<'a> {
    pub ctx: Vec<&'a str>,
    pub next: &'a str,
    pub pos: usize,
}

pub fn turns<'a>(reply: &'a [String], order: usize, buf: &mut Vec<Turn<'a>>) {
    buf.clear();
    let mut ctx: Vec<&str> = vec![BOS; order];
    for (i, t) in reply.iter().enumerate() {
        buf.push(Turn { ctx: ctx.clone(), next: t.as_str(), pos: i });
        ctx.remove(0);
        ctx.push(t.as_str());
    }
    buf.push(Turn { ctx, next: EOS, pos: reply.len() });
}

/// ORDERED neighbour backoff — the fix v12 named and did not build.
///
/// v12 emitted distributional neighbours as extra identity keys at both write and read
/// time, and measured +0.35 top-1 while several common replies got WORSE
/// (`hello` -> `yes , i have .`, `i love you` -> `i'm sorry .`). Reading the allocator
/// afterwards explained it: `IdAllocator::compete` sums candidate weights over ALL the
/// supplied keys and returns a FIXED k winners. So three neighbour keys do not sit
/// BELOW the exact key, they sit BESIDE it and take slots from it. When a word's
/// neighbours mislead, the memory addressed precisely loses to memories addressed
/// approximately.
///
/// That is exactly what separates it from the identity ladder in v9, which worked:
///
///     the ladder is ORDERED. that was FLAT.
///
/// So make the neighbours a genuine rung: they are consulted ONLY when no exact
/// address exists, and they are consulted only at READ time. Writing stays literal —
/// every memory is still filed under its own spelling — so this cannot disturb
/// anything the literal route already gets right. It can only add coverage where the
/// literal route currently returns nothing:
///
///     exact address interned?  -> use it alone, behaviour identical to the baseline
///     no exact address at all  -> try the addresses of distributionally similar words
///
/// `known` is the allocator's interned-feature table (`Factored::id.feat_id`). Needing
/// it is the point: backoff is a decision about what the memory ALREADY CONTAINS, so
/// the encoder has to be able to see that, which a pure feature function cannot.
pub fn feats_backoff(
    prompt: &[String], ctx: &[&str], pos: usize,
    nbrs: &std::collections::HashMap<String, Vec<String>>,
    known: &std::collections::HashMap<String, u32>,
) -> (Vec<String>, Vec<String>) {
    let (mut id, g) = feats_mask(prompt, &[], ctx, pos, 0xF);
    // any rung of the literal ladder present means the memory is reachable exactly.
    if id.iter().any(|k| known.contains_key(k)) {
        return (id, g);
    }
    let rctx = ctx.join(" ");
    if let Some(last) = prompt.iter().rev().find(|w| is_content(w)) {
        if let Some(ns) = nbrs.get(last) {
            for n in ns {
                id.push(format!("QW:{n}|R:{rctx}"));
            }
        }
    }
    (id, g)
}

/// v13 — BACK OFF THE OTHER AXIS.
///
/// MEASURED, and it reframes the ceiling. Splitting held-out positions by how many of
/// the three identity rungs actually exist in the allocator:
///
///     0 of 3 interned : 70.7% of positions   top-1 27.66%
///     1 of 3          : 16.6%                top-1 29.25%
///     3 of 3          : 11.8%                top-1 89.93%
///
/// So reachability is not merely still the bottleneck, it is nearly the whole of it.
/// When a memory has an address she is ~90% right; the 35.5% overall is an average
/// dominated by the 70.7% of positions that address nothing and fall through to the
/// general route.
///
/// And that is also why v12's neighbour keys bought nothing, in either the flat or the
/// ordered form. Look at what every identity key is:
///
///     Q:{whole prompt}|R:{whole reply context}
///     QS:{content words}|R:{whole reply context}
///     QW:{last content word}|R:{whole reply context}
///
/// The ladder backs off along the PROMPT axis and never along the REPLY axis. All
/// three rungs demand the exact three-word reply prefix. Swapping `monday` for
/// `tuesday` in the prompt half cannot rescue a key whose reply half was never seen,
/// which is the actual reason 70.7% of positions miss.
///
/// v8 already solved this for sequences — its identity ladder walks shorter and
/// shorter suffixes of the context and was worth 21% fewer synapses at equal accuracy.
/// The reply encoder simply never got the same treatment. So: give it the second axis.
///
///   bit 0  QW:{last}|R1:{last reply word}   — one word of reply context, not three
///   bit 1  QS:{content}|R1:{last reply word}
///   bit 2  Q:{prompt}|R1:{last reply word}
///   bit 3  QW:{last}|S:{slot}               — no reply context at all, position only
///
/// Every rung stays CROSSED with something from the prompt. An uncrossed rung would be
/// the hub bug for the fourth time; `R1:` alone would just vote for the modal word.
pub fn feats_rungs(prompt: &[String], ctx: &[&str], pos: usize, rungs: u32)
    -> (Vec<String>, Vec<String>)
{
    let (mut id, g) = feats_mask(prompt, &[], ctx, pos, 0xF);
    push_rungs(&mut id, prompt, ctx, pos, rungs);
    (id, g)
}

fn push_rungs(id: &mut Vec<String>, prompt: &[String], ctx: &[&str], pos: usize, rungs: u32) {
    if rungs == 0 { return; }
    let r1 = ctx.last().copied().unwrap_or(BOS);
    let s = slot(pos);
    let mut content: Vec<&str> = prompt.iter().map(|w| w.as_str())
        .filter(|w| is_content(w)).collect();
    let Some(&last) = content.last() else { return };
    if rungs & 1 != 0 { id.push(format!("QW:{last}|R1:{r1}")); }
    if rungs & 2 != 0 {
        content.sort_unstable();
        content.dedup();
        id.push(format!("QS:{}|R1:{r1}", content.join(",")));
    }
    if rungs & 4 != 0 {
        let key: Vec<&str> = prompt.iter().map(|w| w.as_str()).collect();
        id.push(format!("Q:{}|R1:{r1}", key.join(" ")));
    }
    if rungs & 8 != 0 { id.push(format!("QW:{last}|S:{s}")); }
}

/// v13 — WRITE EVERY RUNG, READ ONLY THE FINEST ONE THAT EXISTS.
///
/// The two halves of this were each measured separately and each came out flat:
///
///   * coarse rungs raise coverage (0-address positions 70.7% -> 7.0%) and LOWER
///     accuracy, because a rung that fires everywhere is a hub;
///   * neighbour keys give a coarse rung a better spelling, which does not help when
///     the problem is that it fires everywhere.
///
/// Both fail for the same reason and it is not the features, it is the reading. A
/// coarse address should be consulted when there is nothing better, not IN ADDITION to
/// something better. So split the two directions, which nothing in this project has
/// done before:
///
///     WRITE  every rung, so the coarse memories physically exist. With per-key units
///            (v13, `IdAllocator::allocate_each_learn`) this costs the coarse rungs
///            their own neurons and costs the exact rung nothing.
///     READ   the exact rungs. Only if NONE of them was ever interned — the 70.7%
///            case, where today she has no address at all and scores 27.66% — fall
///            through to the coarse ones and to distributional neighbours.
///
/// This is the identity ladder from v9 finally made ordered rather than flat, and it
/// is only expressible because the allocator no longer makes keys share a budget.
pub fn feats_ladder(
    prompt: &[String], ctx: &[&str], pos: usize, rungs: u32,
    nbrs: Option<&std::collections::HashMap<String, Vec<String>>>,
    known: &std::collections::HashMap<String, u32>,
) -> (Vec<String>, Vec<String>) {
    let (mut id, g) = feats_mask(prompt, &[], ctx, pos, 0xF);
    if id.iter().any(|k| known.contains_key(k)) {
        return (id, g);
    }
    push_rungs(&mut id, prompt, ctx, pos, rungs);
    if let Some(nb) = nbrs {
        let rctx = ctx.join(" ");
        if let Some(last) = prompt.iter().rev().find(|w| is_content(w)) {
            if let Some(ns) = nb.get(last) {
                for n in ns { id.push(format!("QW:{n}|R:{rctx}")); }
            }
        }
    }
    (id, g)
}

/// v15 — class-crossed keys, for the positions that have no address at all.
///
/// `cls` maps a word to its distributional class (see `wordclass::exemplar_classes`).
/// A word too rare to profile has none, and then the key is simply not emitted — the
/// same discipline as every other backoff here: a rung that cannot be built is skipped,
/// not faked.
///
/// bits, and they are an ablation on purpose because the routes behave differently:
///   1  identity  CQ:{class of prompt word}|R1:{literal last reply word}   coarse prompt
///   2  identity  QW:{literal prompt word}|C1:{class of last reply word}   coarse reply
///   4  identity  CQ:{class}|C1:{class}                                    coarse both
///   8  GENERAL   CG:{class}|C1:{class}                                    same, taxed route
///
/// The last two carry identical information and differ only in which route they take.
/// v9 measured that route matters more than content (`+12.6` points for deleting one
/// general feature; history helped as identity and hurt as general), and v13 measured
/// that the reason was a shared budget which no longer exists. So this is also the
/// clean re-test of that rule under the new allocator.
pub fn push_class_keys(
    id: &mut Vec<String>, g: &mut Vec<String>,
    prompt: &[String], ctx: &[&str], bits: u32,
    cls: &std::collections::HashMap<String, u16>,
) {
    if bits == 0 { return; }
    let r1 = ctx.last().copied().unwrap_or(BOS);
    let Some(last) = prompt.iter().rev().find(|w| is_content(w)) else { return };
    let cq = cls.get(last.as_str());
    let cr = cls.get(r1);
    if bits & 1 != 0 { if let Some(c) = cq { id.push(format!("CQ:{c}|R1:{r1}")); } }
    if bits & 2 != 0 { if let Some(c) = cr { id.push(format!("QW:{last}|C1:{c}")); } }
    if bits & 4 != 0 {
        if let (Some(a), Some(b)) = (cq, cr) { id.push(format!("CQ:{a}|C1:{b}")); }
    }
    if bits & 8 != 0 {
        if let (Some(a), Some(b)) = (cq, cr) { g.push(format!("CG:{a}|C1:{b}")); }
    }
}

/// `feats_rungs` plus the class-crossed keys.
pub fn feats_class(
    prompt: &[String], ctx: &[&str], pos: usize, rungs: u32, bits: u32,
    cls: &std::collections::HashMap<String, u16>,
) -> (Vec<String>, Vec<String>) {
    let (mut id, mut g) = feats_mask(prompt, &[], ctx, pos, 0xF);
    push_rungs(&mut id, prompt, ctx, pos, rungs);
    push_class_keys(&mut id, &mut g, prompt, ctx, bits, cls);
    (id, g)
}
