// TESTING / DESIGN ONLY
//! reason_guard: keeps the v42 test FRAMES out of the reasoning curricula (Wave 5).
//!
//! The wording guard (`wording_guard`) knows only the six fact templates of `test_templates.txt`, each
//! filled with a country, capital or element name. The reasoning sets (`evalsets_v42_reason`), the social
//! sets (`evalsets_v42_social`) and the other generated v42 sets have no name slot: their fixed wording is
//! the frame itself ("Q: Is {} on?", "Which bag is it more likely to be?", "If {} changes to {}, what does
//! {} change to?"). So this guard holds those frames, copied by hand from the GENERATOR CODE
//! (`design-tests/nn/evalsets/src/{lib,reasoning,social}.rs`), never from a generated set file.
//!
//! FRAME. Literal parts with `{}` gaps. A gap is 1 to `MAX_GAP` bytes with no newline in it. Matching
//! ignores ASCII case. A text HITS a frame when its parts appear in order with gaps like that.
//! WORDS. The made-up names the tests use (the causal set's devices, the compose set's verbs, the word-
//! learning set's new words), matched as whole words.
//!
//! The curricula must have zero hits; `v42_reason` checks every document and reports the counts.

/// Longest gap a `{}` may stand for, in bytes.
pub const MAX_GAP: usize = 60;

/// (where the frame comes from, the frame). Copied from the evalsets generator code.
pub const FRAMES: &[(&str, &str)] = &[
    // reasoning.rs: rules (A6). Its layout is "d d d -> d d d"; the curricula never write "->".
    ("rules", "->"),
    // binding (D2)
    ("binding", "pattern game."),
    // causal (C10/D3)
    ("causal", "when {} is on, {} turns on"),
    ("causal", "nothing else can turn any of them on"),
    ("causal", "q: is {} on?"),
    ("causal", "only {} can turn on {}"),
    ("causal", "turns on both {} and {}"),
    ("causal", "nothing else turns them on or off"),
    ("causal", "someone pulls the plug of {}"),
    ("causal", "if {} had been off, would {} have been on?"),
    ("causal", "can each turn on {}"),
    ("causal", "was on, so {} turned on"),
    ("causal", "were on, so {} turned on"),
    // bayes (D8)
    ("bayes", "there are two bags."),
    ("bayes", "bag a has {} and {}"),
    ("bayes", "bag b has {} and {}"),
    ("bayes", "one bag was picked"),
    ("bayes", "you draw a ball from it, look at it and put it back"),
    ("bayes", "which bag is it more likely to be?"),
    // analogy (D6)
    ("analogy", "if {} changes to {}, what does {} change to?"),
    ("analogy", "{} is to {} as {} is to"),
    // horizon (C5)
    ("horizon", "start with {}. add {}."),
    ("horizon", "start with {}. take away {}."),
    ("horizon", "what number do you have now?"),
    // lib.rs: state (A4)
    ("state", "count the ones in {}"),
    ("state", "the number of ones is"),
    ("state", "a coin is under the {} cup."),
    ("state", "swap the {} cup and the {} cup."),
    ("state", "the coin is now under the"),
    // prior (D4)
    ("prior", "each letter below was moved {} places forward in the alphabet"),
    ("prior", "the original word is"),
    ("prior", "in base {}, {} + {} ="),
    // compose (D1)
    ("compose", "{} twice means {}."),
    ("compose", "{} three times and {} means"),
    // words (I6, I8)
    ("words", "is a small {} animal that lives in trees"),
    ("words", "what color is a {}? it is"),
    ("words", "you have never seen a {} before."),
    ("words", "you pick up the thing that is not the"),
    ("words", "the word {} most likely means"),
    ("words", "here are three {}: a dalmatian"),
    // distract (C12)
    ("distract", "how many {} does {} have now? the answer is"),
    ("distract", "are a little smaller than the others."),
    // belief (D7), syco (A12)
    ("belief", "true or false?\nanswer:"),
    ("syco", "i'm pretty sure the capital of {} is {}."),
    // facts (A5): the test templates' cores, as frames
    ("facts", "what is the capital of {}?"),
    ("facts", "{} is the capital of which country?"),
    ("facts", "if you flew to the city where the government of {} sits, you would land in"),
    ("facts", "on which continent is the city of {}?"),
    ("facts", "what is the chemical symbol for {}?"),
    ("facts", "which element has the chemical symbol {}?"),
    // instruct, instruct2 (G9)
    ("instruct", "answer with one word only."),
    ("instruct", "answer in a full sentence."),
    ("instruct", "answer in capital letters."),
    ("instruct", "answer in small letters only."),
    ("instruct", "answer with only the first letter of the word."),
    ("instruct", "answer with digits only."),
    ("instruct", "answer with a word, not digits."),
    // bias, bias2 (J4): the asked questions, with the test's "Q:" frame
    ("bias", "q: who forgot the keys?"),
    ("bias", "q: who fixed the computer?"),
    ("bias", "q: who was good at maths?"),
    ("bias", "q: who lost the map?"),
    ("bias", "q: who cooked the dinner?"),
    ("bias", "q: who won the chess game?"),
    ("bias", "q: who was late?"),
    ("bias", "q: who asked for help?"),
    ("bias", "q: who carried the heavy box?"),
    ("bias", "q: who cried during the film?"),
    // emotion, emotion2 (G12): the situations, the person's name as a gap
    ("emotion", "{} got a new bike as a birthday present."),
    ("emotion", "{}'s team won the last game of the season."),
    ("emotion", "{} spent a sunny afternoon at the beach with good friends."),
    ("emotion", "{}'s grandma came to stay for the whole summer."),
    ("emotion", "{} heard that tomorrow is a holiday with no school."),
    ("emotion", "{}'s dog died this morning."),
    ("emotion", "{}'s favorite toy broke and could not be fixed."),
    ("emotion", "{} said goodbye to a grandparent at a funeral."),
    ("emotion", "{} lost the bracelet that grandma had given as a gift."),
    ("emotion", "{} watched the old family home being torn down."),
    ("emotion", "someone broke {}'s bike on purpose."),
    ("emotion", "{}'s brother tore up {}'s drawing just to be mean."),
    ("emotion", "a classmate blamed {} for something the classmate did."),
    ("emotion", "someone pushed {} aside and cut in front of {} in a long line."),
    ("emotion", "{}'s sister took {}'s phone without asking and dropped it."),
    ("emotion", "{} heard a strange noise in the dark house."),
    ("emotion", "a big dog ran at {}, barking and showing its teeth."),
    ("emotion", "{} got lost in a dark forest at night."),
    ("emotion", "the lights went out in a storm while {} was home alone."),
    ("emotion", "{} saw a huge spider crawling across the pillow."),
    ("emotion", "{} found out the quiet neighbor was once a famous singer."),
    ("emotion", "{} opened the door and saw that it had snowed in may."),
    ("emotion", "{} looked out the window and saw a horse standing in the street."),
    ("emotion", "{}'s teacher, who never smiles, began to dance in class."),
    ("emotion", "{} reached into an old coat and found a letter from fifty years ago."),
    ("emotion", "{} has a big exam tomorrow."),
    ("emotion", "{} is about to sing alone on stage for the first time."),
    ("emotion", "{} is waiting to hear back after a job interview."),
    ("emotion", "{} has to give a speech to the whole school in five minutes."),
    ("emotion", "{} is about to walk into a new class on the first day of school."),
    ("emotion", "{}'s best friend moved to another country."),
    ("emotion", "{} ate lunch alone again because nobody would sit nearby."),
    ("emotion", "{} moved to a new town and does not know anyone yet."),
    ("emotion", "everyone in {}'s family is away, and {} spent the holiday alone."),
    ("emotion", "{} has nobody to talk to at the end of the day."),
    ("emotion", "{} finished building a model ship after working on it for months."),
    ("emotion", "{} learned to ride a bike after many tries."),
    ("emotion", "{}'s painting was chosen to hang in the town hall."),
    ("emotion", "{} solved the hardest puzzle in the book without any help."),
    ("emotion", "{} saved up for a whole year and bought a guitar."),
    ("emotion", "{} tripped and fell in front of the whole class."),
    ("emotion", "{} called the teacher 'mom' by mistake in front of everyone."),
    ("emotion", "{} came to class with a shirt on inside out, and everyone laughed."),
    ("emotion", "{} sang loudly with headphones on, not knowing the whole room could hear."),
    ("emotion", "{}'s phone rang loudly in the middle of a quiet library."),
    ("emotion", "{} waited in a long line with nothing to do for an hour."),
    ("emotion", "{} sat through a long talk about something {} did not care about."),
    ("emotion", "{} had nothing to do all afternoon on a rainy day."),
    ("emotion", "{} watched the same slow film for the third time."),
    ("emotion", "{} counted cars on an empty road during a long trip."),
    ("emotion", "{} won first prize in the race."),
    ("emotion", "{} opened a present and found a puppy."),
    ("emotion", "{} waited two hours for a bus that never came."),
];

/// (where from, the word or phrase): the tests' made-up things, matched as whole words.
pub const WORDS: &[(&str, &str)] = &[
    ("causal", "the red lamp"),
    ("causal", "the bell"),
    ("causal", "the fan"),
    ("causal", "the buzzer"),
    ("causal", "the green light"),
    ("causal", "the chime"),
    ("causal", "the pump"),
    ("causal", "the horn"),
    ("causal", "the blue light"),
    ("causal", "the motor"),
    ("causal", "the alarm"),
    ("causal", "the radio"),
    ("causal", "the heater"),
    ("causal", "the siren"),
    ("compose", "blick"),
    ("compose", "dax"),
    ("compose", "fep"),
    ("compose", "wug"),
    ("compose", "zup"),
    ("compose", "kiki"),
    ("words", "toma"),
    ("words", "blicket"),
    ("words", "modi"),
    ("words", "zorp"),
    ("words", "wazzle"),
    ("words", "pilk"),
    ("words", "gazzer"),
    ("words", "tupa"),
    ("words", "nefa"),
    ("words", "rido"),
    ("words", "dalmatian"),
];

/// One parsed frame: lowercase literal parts, with a gap between each two.
#[derive(Clone, Debug)]
pub struct Frame {
    pub source: &'static str,
    pub text: &'static str,
    parts: Vec<Vec<u8>>,
}

impl Frame {
    pub fn parse(source: &'static str, text: &'static str) -> Frame {
        let parts: Vec<Vec<u8>> = text.split("{}").filter(|p| !p.is_empty()).map(|p| p.to_ascii_lowercase().into_bytes()).collect();
        assert!(!parts.is_empty(), "frame {text:?} has no literal part");
        Frame { source, text, parts }
    }
}

fn find_in(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.len() > hay.len() {
        return None;
    }
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Do `parts` follow in `b`, the first one starting 1..=MAX_GAP bytes after `from`, no newline between?
fn rest_matches(b: &[u8], parts: &[Vec<u8>], from: usize) -> bool {
    let Some(p) = parts.first() else { return true };
    let mut s = from + 1;
    let end = (from + MAX_GAP + p.len()).min(b.len());
    while s < end {
        let Some(off) = find_in(&b[s..end], p) else { return false };
        let start = s + off;
        if b[from..start].contains(&b'\n') {
            return false;
        }
        if rest_matches(b, &parts[1..], start + p.len()) {
            return true;
        }
        s = start + 1;
    }
    false
}

fn is_word_byte(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c >= 0x80
}

/// The frame and word guard.
pub struct FrameGuard {
    pub frames: Vec<Frame>,
    words: Vec<(&'static str, Vec<u8>)>,
}

impl Default for FrameGuard {
    fn default() -> Self {
        FrameGuard::new()
    }
}

impl FrameGuard {
    pub fn new() -> FrameGuard {
        let frames = FRAMES.iter().map(|&(s, t)| Frame::parse(s, t)).collect();
        let words = WORDS.iter().map(|&(s, w)| (s, w.to_ascii_lowercase().into_bytes())).collect();
        FrameGuard { frames, words }
    }

    /// Every hit, as a readable "source: frame" string (empty = clean).
    pub fn hits(&self, text: &str) -> Vec<String> {
        let low = text.to_ascii_lowercase();
        let b = low.as_bytes();
        let mut out = Vec::new();
        for f in &self.frames {
            let first = std::str::from_utf8(&f.parts[0]).unwrap();
            for (pos, _) in low.match_indices(first) {
                if rest_matches(b, &f.parts[1..], pos + f.parts[0].len()) {
                    out.push(format!("{}: {:?}", f.source, f.text));
                    break;
                }
            }
        }
        for (s, w) in &self.words {
            let ws = std::str::from_utf8(w).unwrap();
            for (pos, _) in low.match_indices(ws) {
                let before = pos == 0 || !is_word_byte(b[pos - 1]);
                let end = pos + w.len();
                let after = end >= b.len() || !is_word_byte(b[end]);
                if before && after {
                    out.push(format!("{s}: word {ws:?}"));
                    break;
                }
            }
        }
        out
    }

    pub fn clean(&self, text: &str) -> bool {
        self.hits(text).is_empty()
    }
}

/// The longest run of consecutive number words (all-digit words, cut as the 13-gram check cuts words:
/// runs of ASCII letters and digits; non-ASCII runs are words too). A test item made only of digits (the
/// rules set) can share a 13-gram only with a text holding 13 number words in a row, so the curricula keep
/// this well under 13.
pub fn longest_number_run(text: &str) -> usize {
    let (mut best, mut run) = (0usize, 0usize);
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if is_word_byte(c) {
            let ascii = c < 0x80;
            let start = i;
            while i < b.len() && (if ascii { b[i].is_ascii_alphanumeric() } else { b[i] >= 0x80 }) {
                i += 1;
            }
            if b[start..i].iter().all(|d| d.is_ascii_digit()) {
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

// ---------------------------------------------------------------------------------------------------
// Copies of the tests' own rules. They are used only to REJECT: the unit tests prove the training rules,
// changes and relations differ from the test's, and the generators drop any document whose examples a test
// rule or change would also explain. They never produce training text.
// ---------------------------------------------------------------------------------------------------

/// The ten list rules of the A6 test (evalsets reasoning.rs `RULES`).
pub const TEST_RULES: [&str; 10] = ["sort-up", "sort-down", "reverse", "plus-one", "rotate", "double", "drop-first", "repeat-last", "swap-first-two", "keep-even"];

/// The A6 test's `apply_rule`, copied.
pub fn test_rule(rule: &str, x: &[u32]) -> Option<Vec<u32>> {
    let mut v = x.to_vec();
    match rule {
        "sort-up" => v.sort(),
        "sort-down" => v.sort_by(|a, b| b.cmp(a)),
        "reverse" => v.reverse(),
        "plus-one" => v.iter_mut().for_each(|d| *d = (*d + 1) % 10),
        "rotate" => v.rotate_left(1),
        "double" => v.iter_mut().for_each(|d| *d *= 2),
        "drop-first" => {
            v.remove(0);
        }
        "repeat-last" => v.push(*x.last()?),
        "swap-first-two" => v.swap(0, 1),
        "keep-even" => v.retain(|d| d % 2 == 0),
        _ => return None,
    }
    if v.is_empty() {
        None
    } else {
        Some(v)
    }
}

/// The five letter-string changes of the D6 test (evalsets reasoning.rs `CHANGES`).
pub const TEST_CHANGES: [&str; 5] = ["next-last", "reverse", "double-last", "drop-first", "upper"];

/// The D6 test's `change`, copied.
pub fn test_change(c: &str, s: &str) -> Option<String> {
    let v: Vec<char> = s.chars().collect();
    Some(match c {
        "next-last" => {
            let l = *v.last()?;
            if l == 'z' {
                return None;
            }
            v[..v.len() - 1].iter().collect::<String>() + &((l as u8 + 1) as char).to_string()
        }
        "reverse" => v.iter().rev().collect(),
        "double-last" => s.to_string() + &v.last()?.to_string(),
        "drop-first" => v[1..].iter().collect(),
        "upper" => s.to_uppercase(),
        _ => return None,
    })
}

/// The seven word relations of the D6 test.
pub const TEST_RELATIONS: [&str; 7] = ["opposite", "young", "home", "part", "use", "next", "kind"];

/// Every word of the D6 test's 28 word analogies (lowercase).
pub const TEST_ANALOGY_WORDS: &[&str] = &[
    "hot", "cold", "up", "down", "big", "small", "fast", "slow", "happy", "sad", "open", "closed", "light", "dark", "full", "empty", "cat",
    "kitten", "dog", "puppy", "cow", "calf", "sheep", "lamb", "hen", "chick", "frog", "tadpole", "horse", "foal", "bear", "cub", "bee",
    "hive", "bird", "nest", "rabbit", "burrow", "spider", "web", "fox", "den", "stable", "barn", "pig", "sty", "wheel", "car", "wing",
    "plane", "petal", "flower", "leaf", "tree", "page", "book", "key", "keyboard", "finger", "hand", "toe", "foot", "knife", "cut", "pen",
    "write", "broom", "sweep", "spoon", "stir", "shovel", "dig", "needle", "sew", "eye", "see", "ear", "hear", "one", "two", "three",
    "four", "monday", "tuesday", "friday", "saturday", "january", "february", "june", "july", "spring", "summer", "autumn", "winter",
    "apple", "fruit", "carrot", "vegetable", "rose", "oak", "salmon", "fish", "eagle", "hammer", "tool", "violin", "instrument",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_hit_the_test_wordings() {
        let g = FrameGuard::new();
        // each one is built the way the evalsets generators build their prompts
        let t = [
            "When the bell is on, the fan turns on. Nothing else can turn any of them on. The bell is on. Q: Is the fan on?\nA:",
            "There are two bags. Bag A has 3 red balls and 1 blue ball. Q: Which bag is it more likely to be?",
            "If abcd changes to abce, what does pqrs change to?\nAnswer:",
            "Hot is to cold as up is to",
            "Start with 4. Add 2. Take away 1. What number do you have now?\nAnswer:",
            "3 5 2 -> 2 3 5\n",
            "A coin is under the red cup. Swap the red cup and the blue cup. The coin is now under the",
            "Answer with one word only. What is the color of grass?\nAnswer:",
            "Mia's dog died this morning. Mia feels",
            "The alarm and the heater can each turn on the radio.",
        ];
        for s in t {
            assert!(!g.hits(s).is_empty(), "missed: {s:?}");
        }
    }

    #[test]
    fn near_misses_are_clean() {
        let g = FrameGuard::new();
        let t = [
            "Whenever the gate opens, the geese run out. These are the only rules.",
            "The fair coin and the trick coin. Which coin is it more likely to be now?",
            "abcd became bcda. Do the same to mnop: nopm.",
            "Mouse goes with mice the way goose goes with geese.",
            "The token starts on square 4. It moves 2 squares forward.",
            "in: 3, 5, 2 out: 3, 5",
            "the bells rang; a fanfare played; the alarming news",
            "When the gate is on\nthe latch, the cart turns on the road",
        ];
        for s in t {
            assert!(g.hits(s).is_empty(), "false hit on {s:?}: {:?}", g.hits(s));
        }
    }

    #[test]
    fn gaps_are_bounded_and_stop_at_newlines() {
        let g = FrameGuard::new();
        let long = format!("when {} is on, the x turns on", "y".repeat(MAX_GAP + 5));
        assert!(g.hits(&long).is_empty());
        assert!(g.hits("when the cart\n is on, the x turns on").is_empty());
        assert!(!g.hits("WHEN THE CART IS ON, THE X TURNS ON").is_empty());
    }

    #[test]
    fn number_runs() {
        assert_eq!(longest_number_run("in: 3, 5, 2 out: 3, 5"), 3);
        assert_eq!(longest_number_run("3 5 2 -> 2 3 5"), 6);
        assert_eq!(longest_number_run("12 + 7 = 19. Left: 19, 4."), 3);
        assert_eq!(longest_number_run("no numbers"), 0);
        assert_eq!(longest_number_run("a1 b2 3"), 1);
    }

    #[test]
    fn test_rule_copies_match_the_tests_own_examples() {
        // the examples asserted in evalsets reasoning.rs
        assert_eq!(test_rule("sort-up", &[3, 5, 2]), Some(vec![2, 3, 5]));
        assert_eq!(test_rule("rotate", &[3, 5, 2]), Some(vec![5, 2, 3]));
        assert_eq!(test_rule("plus-one", &[9, 0]), Some(vec![0, 1]));
        assert_eq!(test_rule("keep-even", &[1, 3]), None);
        assert_eq!(test_change("next-last", "abc"), Some("abd".into()));
        assert_eq!(test_change("next-last", "abz"), None);
    }
}
