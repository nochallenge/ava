//! TESTING / DESIGN ONLY — sleep_facts: made-up people (facts she cannot have seen) for v40's sleep test.
//!
//! A PERSON has a made-up name (two capitalised words from a syllable generator: onset, vowel,
//! coda, two or three syllables, at least five letters), a birth year in 1700..=1999, a made-up
//! town (another syllable style plus an English-looking ending such as -holt, -mere, -wick) and
//! a job from a fixed list of 40 common jobs. Every name word and every town is unique across
//! the whole population, so a dev set and a sealed set cut from it share no name and no town.
//!
//! THE DIARY. One document per person:
//!   "{Name} was born in {year} in {Town}. {Name} worked as a/an {job}."
//!
//! QUESTIONS (three per person, scored by the likelihood of each choice after the prompt):
//!   year: "{Name} was born in"            choices " {year}"   (right + 3 other years of the range)
//!   town: "{Name} was born in {year} in"  choices " {Town}."  (right + 3 other towns of the same set)
//!   job:  "{Name} worked as"              choices " a/an {job}." (right + 3 other jobs)
//! All four choices are distinct; per kind the right position is balanced (i % 4, shuffled).
//! TSV as in `fact_items` (`id \t kind \t bin \t right \t prompt \t choices`, bin = kind).

use crate::fact_items::tsv_line;
use crate::rng::{derive_seed, SplitMix64};
use std::collections::HashSet;

pub const KINDS: [&str; 3] = ["year", "town", "job"];
pub const YEAR_MIN: u32 = 1700;
pub const YEAR_MAX: u32 = 1999;
pub const DEFAULT_SEED: u64 = 40;

pub const JOBS: [&str; 40] = [
    "baker", "farmer", "teacher", "nurse", "carpenter", "blacksmith", "tailor", "miller", "weaver", "fisherman", "sailor",
    "soldier", "doctor", "lawyer", "clerk", "shoemaker", "butcher", "potter", "mason", "painter", "printer", "gardener",
    "shepherd", "brewer", "cook", "engineer", "architect", "merchant", "banker", "judge", "musician", "actor", "writer",
    "librarian", "postman", "miner", "innkeeper", "watchmaker", "chemist", "optician",
];

pub(crate) const NAME_ONSETS: [&str; 25] = [
    "b", "br", "d", "dr", "f", "g", "gr", "h", "k", "kr", "l", "m", "n", "p", "pr", "r", "s", "st", "t", "tr", "v", "z",
    "th", "sh", "w",
];
pub(crate) const NAME_VOWELS: [&str; 11] = ["a", "e", "i", "o", "u", "ae", "ai", "ei", "oa", "ou", "y"];
pub(crate) const NAME_CODAS: [&str; 14] = ["", "", "", "n", "r", "l", "s", "m", "th", "nd", "rk", "ld", "x", "st"];
pub(crate) const TOWN_ONSETS: [&str; 22] = [
    "", "b", "c", "d", "f", "g", "h", "l", "m", "n", "p", "r", "s", "t", "w", "ch", "cr", "gl", "pl", "sk", "sw", "wr",
];
pub(crate) const TOWN_VOWELS: [&str; 8] = ["a", "e", "i", "o", "u", "ea", "oo", "ay"];
pub(crate) const TOWN_CODAS: [&str; 10] = ["", "l", "n", "r", "s", "ck", "mb", "nt", "sh", "lt"];
pub(crate) const TOWN_ENDS: [&str; 18] = [
    "mere", "holt", "wick", "thorpe", "stead", "combe", "dale", "ford", "bury", "moor", "well", "by", "ham", "ton",
    "field", "ley", "cote", "haven",
];
const PEOPLE_LABEL: u64 = 0x7065_6f70_6c65;
const Q_LABEL: u64 = 0x736c_6565_70;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Person {
    pub name: String,
    pub year: u32,
    pub town: String,
    pub job: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QItem {
    pub id: u32,
    pub kind: &'static str,
    pub right: usize,
    pub prompt: String,
    pub choices: Vec<String>,
}

fn pick<'a>(r: &mut SplitMix64, v: &[&'a str]) -> &'a str {
    v[r.below(v.len() as u64) as usize]
}

fn capitalised(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c).collect(),
        None => String::new(),
    }
}

fn name_word(r: &mut SplitMix64) -> String {
    let syl = if r.below(5) < 3 { 2 } else { 3 };
    let mut s = String::new();
    for _ in 0..syl {
        s.push_str(pick(r, &NAME_ONSETS));
        s.push_str(pick(r, &NAME_VOWELS));
        s.push_str(pick(r, &NAME_CODAS));
    }
    capitalised(&s)
}

fn town_name(r: &mut SplitMix64) -> String {
    let syl = 1 + r.below(2);
    let mut s = String::new();
    for _ in 0..syl {
        s.push_str(pick(r, &TOWN_ONSETS));
        s.push_str(pick(r, &TOWN_VOWELS));
        s.push_str(pick(r, &TOWN_CODAS));
    }
    s.push_str(pick(r, &TOWN_ENDS));
    capitalised(&s)
}

/// A word from `gen` at least `min` letters long and not used yet (case-insensitive).
fn fresh(r: &mut SplitMix64, used: &mut HashSet<String>, min: usize, gen: fn(&mut SplitMix64) -> String) -> String {
    loop {
        let w = gen(r);
        if w.len() >= min && used.insert(w.to_lowercase()) {
            return w;
        }
    }
}

/// `n` people, seeded; all name words and towns unique.
pub fn people(seed: u64, n: usize) -> Vec<Person> {
    people_excluding(seed, n, &HashSet::new())
}

/// v41's sealed population: seed 41, clear of every name word and town of v40's population (seed 40, 150 dev +
/// 150 sealed people, regenerated in memory; no file is read).
pub const V41_SEED: u64 = 41;

/// Every name word and town of `people`, lowercase.
pub fn words_of(people: &[Person]) -> HashSet<String> {
    let mut w = HashSet::new();
    for p in people {
        w.extend(p.name.split(' ').map(|x| x.to_lowercase()));
        w.insert(p.town.to_lowercase());
    }
    w
}

/// `people(seed, n)` with the lowercase words of `exclude` counted as used (a draw that hits one is drawn
/// again), so no name word or town of the result is in `exclude`. With an empty `exclude` this is `people`.
pub fn people_excluding(seed: u64, n: usize, exclude: &HashSet<String>) -> Vec<Person> {
    let mut r = SplitMix64::new(derive_seed(seed, PEOPLE_LABEL, 0));
    let mut used = exclude.clone();
    (0..n)
        .map(|_| {
            let first = fresh(&mut r, &mut used, 5, name_word);
            let last = fresh(&mut r, &mut used, 5, name_word);
            let town = fresh(&mut r, &mut used, 6, town_name);
            let year = YEAR_MIN + r.below((YEAR_MAX - YEAR_MIN + 1) as u64) as u32;
            let job = JOBS[r.below(JOBS.len() as u64) as usize];
            Person { name: format!("{first} {last}"), year, town, job }
        })
        .collect()
}

/// The dev people (the first `n_dev`) and the sealed people (the next `n_test`) of one
/// population: no name word or town in common.
pub fn split_people(seed: u64, n_dev: usize, n_test: usize) -> (Vec<Person>, Vec<Person>) {
    let mut all = people(seed, n_dev + n_test);
    let test = all.split_off(n_dev);
    (all, test)
}

pub fn article(job: &str) -> &'static str {
    if job.starts_with(['a', 'e', 'i', 'o', 'u']) {
        "an"
    } else {
        "a"
    }
}

/// The person's diary document (one line).
pub fn diary(p: &Person) -> String {
    format!("{} was born in {} in {}. {} worked as {} {}.", p.name, p.year, p.town, p.name, article(p.job), p.job)
}

fn job_choice(job: &str) -> String {
    format!(" {} {job}.", article(job))
}

/// Three questions per person (year, town, job), ids from 0 in person order. Town distractors
/// come from the other people of `people`.
pub fn questions(people: &[Person], seed: u64) -> Vec<QItem> {
    let mut r = SplitMix64::new(derive_seed(seed, Q_LABEL, 0));
    let n = people.len();
    let mut pos: Vec<Vec<usize>> = (0..3)
        .map(|_| {
            let mut v: Vec<usize> = (0..n).map(|i| i % 4).collect();
            r.shuffle(&mut v);
            v
        })
        .collect();
    let mut out = Vec::with_capacity(3 * n);
    for (i, p) in people.iter().enumerate() {
        for (k, &kind) in KINDS.iter().enumerate() {
            let right = pos[k][i];
            let (prompt, truth) = match kind {
                "year" => (format!("{} was born in", p.name), format!(" {}", p.year)),
                "town" => (format!("{} was born in {} in", p.name, p.year), format!(" {}.", p.town)),
                _ => (format!("{} worked as", p.name), job_choice(p.job)),
            };
            let mut choices = vec![truth];
            while choices.len() < 4 {
                let c = match kind {
                    "year" => format!(" {}", YEAR_MIN + r.below((YEAR_MAX - YEAR_MIN + 1) as u64) as u32),
                    "town" => format!(" {}.", people[r.below(n as u64) as usize].town),
                    _ => job_choice(JOBS[r.below(JOBS.len() as u64) as usize]),
                };
                if !choices.contains(&c) {
                    choices.push(c);
                }
            }
            let t = choices.remove(0);
            choices.insert(right, t);
            out.push(QItem { id: out.len() as u32, kind, right, prompt, choices });
        }
    }
    pos.clear();
    out
}

pub fn q_line(q: &QItem) -> String {
    tsv_line(q.id, q.kind, q.kind, q.right, &q.prompt, &q.choices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fact_items::unescape_field;

    #[test]
    fn v41_people_share_nothing_with_v40() {
        let (d, t) = split_people(DEFAULT_SEED, 150, 150);
        let old = words_of(&d.iter().chain(&t).cloned().collect::<Vec<_>>());
        assert_eq!(old.len(), 900);
        let new = people_excluding(V41_SEED, 150, &old);
        assert!(new == people_excluding(V41_SEED, 150, &old), "not deterministic"); // no item in a message: sealed
        let w = words_of(&new);
        assert_eq!(w.len(), 450, "all new words distinct");
        assert!(w.is_disjoint(&old));
        // without the exclusion the population is people(41, n) itself
        assert!(people_excluding(V41_SEED, 150, &HashSet::new()) == people(V41_SEED, 150));
    }

    #[test]
    fn people_unique_disjoint_and_deterministic() {
        let (dev, test) = split_people(DEFAULT_SEED, 150, 150);
        assert_eq!((dev.len(), test.len()), (150, 150));
        assert_eq!(split_people(DEFAULT_SEED, 150, 150), (dev.clone(), test.clone()));
        assert_ne!(people(DEFAULT_SEED + 1, 150), dev);
        let words = |ps: &[Person]| -> Vec<String> {
            ps.iter().flat_map(|p| p.name.split(' ').chain(std::iter::once(p.town.as_str())).map(|w| w.to_lowercase()).collect::<Vec<_>>()).collect()
        };
        let (wd, wt) = (words(&dev), words(&test));
        let all: HashSet<&String> = wd.iter().chain(&wt).collect();
        assert_eq!(all.len(), 3 * 300, "every name word and town is unique");
        let sd: HashSet<&String> = wd.iter().collect();
        assert!(wt.iter().all(|w| !sd.contains(w)), "dev and sealed share a word");
        for p in dev.iter().chain(&test) {
            let parts: Vec<&str> = p.name.split(' ').collect();
            assert_eq!(parts.len(), 2);
            for w in parts.iter().chain(std::iter::once(&p.town.as_str())) {
                assert!(w.len() >= 5 && w.chars().next().unwrap().is_ascii_uppercase(), "{w}");
                assert!(w.chars().skip(1).all(|c| c.is_ascii_lowercase()), "{w}");
            }
            assert!((YEAR_MIN..=YEAR_MAX).contains(&p.year));
            assert!(JOBS.contains(&p.job));
        }
        // the population varies
        let years: HashSet<u32> = dev.iter().map(|p| p.year).collect();
        let jobs: HashSet<&str> = dev.iter().map(|p| p.job).collect();
        assert!(years.len() > 100 && jobs.len() > 30);
    }

    #[test]
    fn diary_and_articles() {
        let p = Person { name: "Draelith Kouvarn".into(), year: 1823, town: "Brelholt".into(), job: "engineer" };
        assert_eq!(diary(&p), "Draelith Kouvarn was born in 1823 in Brelholt. Draelith Kouvarn worked as an engineer.");
        assert_eq!(article("baker"), "a");
        assert_eq!(article("optician"), "an");
        assert_eq!(JOBS.iter().filter(|j| article(j) == "an").count(), 5);
        let mut j = JOBS.to_vec();
        j.sort();
        j.dedup();
        assert_eq!(j.len(), 40);
    }

    #[test]
    fn questions_have_one_true_choice_and_balanced_positions() {
        let (dev, _) = split_people(DEFAULT_SEED, 150, 150);
        let qs = questions(&dev, 1);
        assert_eq!(qs, questions(&dev, 1));
        assert_ne!(qs, questions(&dev, 2));
        assert_eq!(qs.len(), 450);
        let towns: HashSet<String> = dev.iter().map(|p| format!(" {}.", p.town)).collect();
        let mut pos = [[0usize; 4]; 3];
        for (i, q) in qs.iter().enumerate() {
            assert_eq!(q.id as usize, i);
            let p = &dev[i / 3];
            let k = i % 3;
            assert_eq!(q.kind, KINDS[k]);
            pos[k][q.right] += 1;
            assert_eq!(q.choices.len(), 4);
            let mut c = q.choices.clone();
            c.sort();
            c.dedup();
            assert_eq!(c.len(), 4, "{q:?}");
            // the right choice completes the diary; no wrong choice does
            let doc = diary(p);
            for (j, ch) in q.choices.iter().enumerate() {
                let completes = doc.starts_with(&format!("{}{ch}", q.prompt)) || doc.contains(&format!(". {}{ch}", q.prompt));
                assert_eq!(completes, j == q.right, "{q:?}");
                match q.kind {
                    "year" => {
                        assert!(ch.len() == 5 && ch.starts_with(' '));
                        let y: u32 = ch[1..].parse().unwrap();
                        assert!((YEAR_MIN..=YEAR_MAX).contains(&y));
                    }
                    "town" => assert!(towns.contains(ch), "{ch}"),
                    _ => {
                        let job = ch.trim_start_matches(" a ").trim_start_matches(" an ").trim_end_matches('.');
                        assert!(JOBS.contains(&job), "{ch}");
                        assert_eq!(ch, &format!(" {} {job}.", article(job)));
                    }
                }
            }
            let line = q_line(q);
            let f: Vec<&str> = line.split('\t').collect();
            assert_eq!(f.len(), 9);
            assert_eq!((f[1], f[2]), (q.kind, q.kind));
            assert_eq!(unescape_field(f[4]), q.prompt);
        }
        for k in 0..3 {
            assert!(pos[k].iter().all(|&c| c == 37 || c == 38), "{:?}", pos[k]);
        }
    }
}
