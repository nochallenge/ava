// TESTING / DESIGN ONLY
//! fact_ledger: how often each of 200 hand-written facts appears in her text, and templated
//! restatements of the facts training may restate (V4 + U6), reconciled with the v42 test keys.
//!
//! THE FACTS (written here by hand): 100 country -> capital, 50 element -> symbol, 42 animal ->
//! class, 8 planet -> order from the Sun. A name may carry match-only aliases after `|`
//! ("Kyiv|Kiev"); the first form is the one used in restatements.
//!
//! OWN SPLIT. A fact's id is `rel:first-form-of-a` (e.g. `capital:France`). It is a DEV fact when
//! `mix64(derive_seed(seed, SPLIT_LABEL, fnv(id))) % 100 < dev_percent` (default 20%), else a train
//! fact. Since v42's fresh tests, this own split is used only for facts the tests do not ask about
//! (animals, planets); see RECONCILIATION.
//!
//! RECONCILIATION (v42). The fresh test sets ask capitals and element symbols. Their keys
//! (`country:France`, `element:helium`) come in two lists: `restate_fact_keys.txt` (training MAY
//! restate these) and `control_fact_keys.txt` (training must NEVER restate these). Only those two
//! files and `test_templates.txt` are read; no test item is. Then:
//!   - a capital or element fact is restated only when its key is in the restate list and it has no
//!     key in the control list. A hand-written fact goes by every key its names give (`country:` or
//!     `element:` + each form) and by the Wikidata country with the same capital (so `China` is also
//!     `People's Republic of China`) or the Wikidata element with the same symbol;
//!   - a capital or element fact in neither list is not restated (status `unlisted`);
//!   - animals and planets keep the own split above (status `train` or `dev`);
//!   - every country and element key of the restate list is restated once, with the Wikidata names
//!     first and the hand-written forms as aliases. Continents are never restated: the two-hop test
//!     needs the continent link to come from her ordinary reading;
//!   - no restatement may hold a test template filled with any name (`wording_guard`), and no
//!     restatement may mention a control fact (both its sides in one sentence, by the ledger below).
//!     The builder refuses to write when either check fails.
//!
//! RESTATEMENTS. Six or seven phrasings per relation, both directions and a question form ("The capital
//! of France is Paris." / "Paris is the capital of France." / "Which country has Paris as its capital?
//! France."). The test's own question wordings are not among them. One round holds every phrasing of
//! every restated fact once; rounds are shuffled (seeded) and cut into documents of `per_doc`
//! sentences joined by a space.
//!
//! THE LEDGER. Text is cut into sentences (after . ! ? followed by a space, and at newlines of a
//! line source), words are runs of letters (any script) or digits. For each fact and sentence:
//!   mention = both sides appear as whole words or phrases (case-insensitive, except element symbols
//!             and planet names, which must match their capitalised form exactly, so "he" is not
//!             helium and "earth" (soil) is not the planet)
//!   stated  = a mention that also carries the relation: the word "capital" (capitals), "symbol"
//!             (elements), "planet" and "sun" (planets), or for animals the class word within 6
//!             words after the animal word ("whales are mammals").

use crate::hash::{new_map, FastMap};
use crate::rng::{derive_seed, mix64, SplitMix64};
use crate::wikidata::Wikidata;
use std::collections::BTreeSet;
use std::io;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rel {
    Capital,
    Symbol,
    Class,
    Planet,
}

impl Rel {
    pub fn name(self) -> &'static str {
        match self {
            Rel::Capital => "capital",
            Rel::Symbol => "symbol",
            Rel::Class => "class",
            Rel::Planet => "planet",
        }
    }
}

/// (country, capital).
pub const CAPITALS: [(&str, &str); 100] = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Italy", "Rome"), ("Spain", "Madrid"), ("Portugal", "Lisbon"),
    ("United Kingdom|Britain", "London"), ("Ireland", "Dublin"), ("Netherlands|Holland", "Amsterdam"), ("Belgium", "Brussels"),
    ("Switzerland", "Bern|Berne"), ("Austria", "Vienna"), ("Poland", "Warsaw"), ("Czech Republic|Czechia", "Prague"),
    ("Hungary", "Budapest"), ("Greece", "Athens"), ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Denmark", "Copenhagen"),
    ("Finland", "Helsinki"), ("Iceland", "Reykjavik|Reykjavík"), ("Russia", "Moscow"), ("Ukraine", "Kyiv|Kiev"),
    ("Romania", "Bucharest"), ("Bulgaria", "Sofia"), ("Serbia", "Belgrade"), ("Croatia", "Zagreb"), ("Turkey", "Ankara"),
    ("Egypt", "Cairo"), ("Kenya", "Nairobi"), ("Nigeria", "Abuja"), ("Ethiopia", "Addis Ababa"), ("Ghana", "Accra"),
    ("Morocco", "Rabat"), ("South Africa", "Pretoria"), ("Algeria", "Algiers"), ("Tunisia", "Tunis"), ("Senegal", "Dakar"),
    ("Uganda", "Kampala"), ("Zimbabwe", "Harare"), ("Sudan", "Khartoum"), ("Libya", "Tripoli"), ("Angola", "Luanda"),
    ("China", "Beijing|Peking"), ("Japan", "Tokyo"), ("South Korea", "Seoul"), ("North Korea", "Pyongyang"),
    ("India", "New Delhi"), ("Pakistan", "Islamabad"), ("Bangladesh", "Dhaka"), ("Nepal", "Kathmandu"),
    ("Thailand", "Bangkok"), ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"), ("Philippines", "Manila"),
    ("Malaysia", "Kuala Lumpur"), ("Iran", "Tehran"), ("Iraq", "Baghdad"), ("Saudi Arabia", "Riyadh"), ("Syria", "Damascus"),
    ("Jordan", "Amman"), ("Lebanon", "Beirut"), ("Afghanistan", "Kabul"), ("Mongolia", "Ulaanbaatar|Ulan Bator"),
    ("Australia", "Canberra"), ("New Zealand", "Wellington"), ("Fiji", "Suva"), ("Canada", "Ottawa"),
    ("United States|USA", "Washington"), ("Cuba", "Havana"), ("Jamaica", "Kingston"), ("Brazil", "Brasilia|Brasília"),
    ("Argentina", "Buenos Aires"), ("Chile", "Santiago"), ("Peru", "Lima"), ("Colombia", "Bogota|Bogotá"),
    ("Venezuela", "Caracas"), ("Ecuador", "Quito"), ("Uruguay", "Montevideo"), ("Paraguay", "Asuncion|Asunción"),
    ("Costa Rica", "San Jose|San José"), ("Honduras", "Tegucigalpa"), ("Nicaragua", "Managua"), ("Haiti", "Port-au-Prince"),
    ("Estonia", "Tallinn"), ("Latvia", "Riga"), ("Lithuania", "Vilnius"), ("Slovakia", "Bratislava"), ("Slovenia", "Ljubljana"),
    ("Belarus", "Minsk"), ("Georgia", "Tbilisi"), ("Armenia", "Yerevan"), ("Azerbaijan", "Baku"), ("Albania", "Tirana"),
    ("Cyprus", "Nicosia"), ("Malta", "Valletta"), ("Qatar", "Doha"), ("Oman", "Muscat"), ("Cambodia", "Phnom Penh"),
    ("Laos", "Vientiane"), ("Madagascar", "Antananarivo"),
];

/// Countries said with "the" (more are found by `needs_the`).
const THE_COUNTRIES: [&str; 12] = [
    "United Kingdom", "Netherlands", "Czech Republic", "Philippines", "United States", "Maldives", "Seychelles", "Comoros",
    "Bahamas", "Gambia", "Ivory Coast", "Vatican",
];

/// (element, symbol).
pub const ELEMENTS: [(&str, &str); 50] = [
    ("hydrogen", "H"), ("helium", "He"), ("lithium", "Li"), ("beryllium", "Be"), ("boron", "B"), ("carbon", "C"),
    ("nitrogen", "N"), ("oxygen", "O"), ("fluorine", "F"), ("neon", "Ne"), ("sodium", "Na"), ("magnesium", "Mg"),
    ("aluminium|aluminum", "Al"), ("silicon", "Si"), ("phosphorus", "P"), ("sulfur|sulphur", "S"), ("chlorine", "Cl"),
    ("argon", "Ar"), ("potassium", "K"), ("calcium", "Ca"), ("titanium", "Ti"), ("chromium", "Cr"), ("manganese", "Mn"),
    ("iron", "Fe"), ("cobalt", "Co"), ("nickel", "Ni"), ("copper", "Cu"), ("zinc", "Zn"), ("gallium", "Ga"), ("arsenic", "As"),
    ("selenium", "Se"), ("bromine", "Br"), ("krypton", "Kr"), ("strontium", "Sr"), ("silver", "Ag"), ("tin", "Sn"),
    ("antimony", "Sb"), ("iodine", "I"), ("xenon", "Xe"), ("cesium|caesium", "Cs"), ("barium", "Ba"), ("tungsten", "W"),
    ("platinum", "Pt"), ("gold", "Au"), ("mercury", "Hg"), ("lead", "Pb"), ("bismuth", "Bi"), ("radon", "Rn"),
    ("radium", "Ra"), ("uranium", "U"),
];

/// (animal singular|plural, class singular|plural).
pub const ANIMALS: [(&str, &str); 42] = [
    ("whale|whales", "mammal|mammals"), ("dolphin|dolphins", "mammal|mammals"), ("bat|bats", "mammal|mammals"),
    ("elephant|elephants", "mammal|mammals"), ("kangaroo|kangaroos", "mammal|mammals"), ("platypus|platypuses", "mammal|mammals"),
    ("walrus|walruses", "mammal|mammals"), ("giraffe|giraffes", "mammal|mammals"), ("horse|horses", "mammal|mammals"),
    ("tiger|tigers", "mammal|mammals"), ("eagle|eagles", "bird|birds"), ("penguin|penguins", "bird|birds"),
    ("ostrich|ostriches", "bird|birds"), ("owl|owls", "bird|birds"), ("sparrow|sparrows", "bird|birds"),
    ("parrot|parrots", "bird|birds"), ("hawk|hawks", "bird|birds"), ("flamingo|flamingos", "bird|birds"),
    ("snake|snakes", "reptile|reptiles"), ("lizard|lizards", "reptile|reptiles"), ("crocodile|crocodiles", "reptile|reptiles"),
    ("turtle|turtles", "reptile|reptiles"), ("alligator|alligators", "reptile|reptiles"), ("iguana|iguanas", "reptile|reptiles"),
    ("chameleon|chameleons", "reptile|reptiles"), ("tortoise|tortoises", "reptile|reptiles"), ("frog|frogs", "amphibian|amphibians"),
    ("toad|toads", "amphibian|amphibians"), ("salamander|salamanders", "amphibian|amphibians"), ("newt|newts", "amphibian|amphibians"),
    ("axolotl|axolotls", "amphibian|amphibians"), ("shark|sharks", "fish|fish"), ("salmon|salmon", "fish|fish"),
    ("trout|trout", "fish|fish"), ("tuna|tuna", "fish|fish"), ("goldfish|goldfish", "fish|fish"), ("seahorse|seahorses", "fish|fish"),
    ("ant|ants", "insect|insects"), ("bee|bees", "insect|insects"), ("butterfly|butterflies", "insect|insects"),
    ("beetle|beetles", "insect|insects"), ("mosquito|mosquitoes", "insect|insects"),
];

/// (planet, ordinal word, place).
pub const PLANETS: [(&str, &str, &str); 8] = [
    ("Mercury", "first", "one"), ("Venus", "second", "two"), ("Earth", "third", "three"), ("Mars", "fourth", "four"),
    ("Jupiter", "fifth", "five"), ("Saturn", "sixth", "six"), ("Uranus", "seventh", "seven"), ("Neptune", "eighth", "eight"),
];

pub const DEFAULT_SEED: u64 = 42;
pub const DEFAULT_DEV_PERCENT: u64 = 20;
const SPLIT_LABEL: u64 = 0x6661_6374_5f73_706c;
const ROUND_LABEL: u64 = 0x6661_6374_5f72_6e64;

/// Where the v42 test keys and templates live, and the only three files of that folder read here.
pub const KEYS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/nn/evalsets_v42");
pub const RESTATE_FILE: &str = "restate_fact_keys.txt";
pub const CONTROL_FILE: &str = "control_fact_keys.txt";
pub const TEMPLATES_FILE: &str = "test_templates.txt";
pub const WIKIDATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../data/downloads/wikidata");

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fact {
    pub rel: Rel,
    /// Subject forms (first = display).
    pub a: Vec<String>,
    /// Object forms (first = display).
    pub b: Vec<String>,
    /// Planets: the place as a word ("four").
    pub c: String,
}

impl Fact {
    pub fn id(&self) -> String {
        format!("{}:{}", self.rel.name(), self.a[0])
    }

    pub fn new(rel: Rel, a: &[&str], b: &[&str]) -> Fact {
        Fact { rel, a: a.iter().map(|s| s.to_string()).collect(), b: b.iter().map(|s| s.to_string()).collect(), c: String::new() }
    }
}

/// All 200 hand-written facts, in the order of the lists above.
pub fn all_facts() -> Vec<Fact> {
    let split = |s: &str| s.split('|').map(|x| x.to_string()).collect::<Vec<_>>();
    let mut v = Vec::new();
    for (a, b) in CAPITALS {
        v.push(Fact { rel: Rel::Capital, a: split(a), b: split(b), c: String::new() });
    }
    for (a, b) in ELEMENTS {
        v.push(Fact { rel: Rel::Symbol, a: split(a), b: split(b), c: String::new() });
    }
    for (a, b) in ANIMALS {
        v.push(Fact { rel: Rel::Class, a: split(a), b: split(b), c: String::new() });
    }
    for (a, b, c) in PLANETS {
        v.push(Fact { rel: Rel::Planet, a: vec![a.to_string()], b: vec![b.to_string()], c: c.to_string() });
    }
    v
}

fn fnv(s: &str) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in s.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// True when the fact is held out by the own split (a dev fact: never restated in training text).
pub fn is_dev(f: &Fact, seed: u64, dev_percent: u64) -> bool {
    mix64(derive_seed(seed, SPLIT_LABEL, fnv(&f.id()))) % 100 < dev_percent
}

fn cap(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

fn art(word: &str) -> &'static str {
    if word.starts_with(['a', 'e', 'i', 'o', 'u']) {
        "an"
    } else {
        "a"
    }
}

/// Whether a country's name is said with "the" ("the United Arab Emirates", "the Marshall Islands").
pub fn needs_the(name: &str) -> bool {
    THE_COUNTRIES.contains(&name)
        || ["United ", "Republic of ", "Kingdom of ", "People's Republic", "Federated States", "Democratic Republic"].iter().any(|p| name.starts_with(p))
        || name.ends_with(" Republic")
        || name.ends_with(" Islands")
}

/// A country's name as said inside a sentence.
pub fn country(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("The ") {
        format!("the {rest}")
    } else if needs_the(name) {
        format!("the {name}")
    } else {
        name.to_string()
    }
}

/// Every phrasing of one fact (6 or 7 sentences; both directions and a question form). None of them is
/// a v42 test wording ("What is the capital of X?", "What is the chemical symbol for X?", ...).
pub fn restatements(f: &Fact) -> Vec<String> {
    match f.rel {
        Rel::Capital => {
            let (c, x) = (country(&f.a[0]), f.b[0].as_str());
            vec![
                format!("The capital of {c} is {x}."),
                format!("{x} is the capital of {c}."),
                format!("Which country has {x} as its capital? {}.", cap(&c)),
                format!("Do you know the capital of {c}? It is {x}."),
                format!("{}'s capital city is {x}.", cap(&c)),
                format!("{x} is the capital city of {c}."),
                format!("The capital city of {c} is called {x}."),
            ]
        }
        Rel::Symbol => {
            let (e, sy) = (f.a[0].as_str(), f.b[0].as_str());
            vec![
                format!("The chemical symbol for {e} is {sy}."),
                format!("{sy} is the chemical symbol for {e}."),
                format!("Which element has the symbol {sy}? {}.", cap(e)),
                format!("Do you know the chemical symbol for {e}? It is {sy}."),
                format!("{} has the symbol {sy}.", cap(e)),
                format!("The symbol {sy} stands for {e}."),
            ]
        }
        Rel::Class => {
            let (a, ap, k, kp) = (f.a[0].as_str(), f.a[1].as_str(), f.b[0].as_str(), f.b[1].as_str());
            vec![
                format!("{} {a} is {} {k}.", cap(art(a)), art(k)),
                format!("{} are {kp}.", cap(ap)),
                format!("What kind of animal is {} {a}? {} {k}.", art(a), cap(art(k))),
                format!("The {a} is {} {k}.", art(k)),
                format!("One example of {} {k} is the {a}.", art(k)),
                format!("Which animal group does the {a} belong to? The {kp}."),
            ]
        }
        Rel::Planet => {
            let (p, o, n) = (f.a[0].as_str(), f.b[0].as_str(), f.c.as_str());
            vec![
                format!("{p} is the {o} planet from the Sun."),
                format!("The {o} planet from the Sun is {p}."),
                format!("Which planet is {o} from the Sun? {p}."),
                format!("What is the {o} planet from the Sun? {p}."),
                format!("Counting out from the Sun, {p} is planet number {n}."),
                format!("{p} comes {o} in order from the Sun."),
            ]
        }
    }
}

/// Restatement documents of exactly the given facts: `rounds` shuffled rounds of every phrasing,
/// cut into documents of `per_doc` sentences.
pub fn restatement_docs(facts: &[Fact], seed: u64, rounds: usize, per_doc: usize) -> Vec<String> {
    let mut all: Vec<String> = Vec::new();
    for r in 0..rounds {
        let mut round: Vec<String> = facts.iter().flat_map(restatements).collect();
        SplitMix64::new(derive_seed(seed, ROUND_LABEL, r as u64)).shuffle(&mut round);
        all.extend(round);
    }
    all.chunks(per_doc.max(1)).map(|c| c.join(" ")).collect()
}

// ---------------------------------------------------------------- reconciliation with the v42 test keys

/// What happens to a hand-written fact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// A test fact training may restate: restated.
    Restate,
    /// A test fact training must never restate.
    Control,
    /// A capital or element fact in neither list: not restated.
    Unlisted,
    /// Own split, train: restated.
    Train,
    /// Own split, dev: not restated.
    Dev,
}

impl Status {
    pub fn name(self) -> &'static str {
        match self {
            Status::Restate => "restate",
            Status::Control => "control",
            Status::Unlisted => "unlisted",
            Status::Train => "train",
            Status::Dev => "dev",
        }
    }
}

/// The two key lists of the v42 tests.
#[derive(Clone, Debug, Default)]
pub struct KeyLists {
    pub restate: BTreeSet<String>,
    pub control: BTreeSet<String>,
}

impl KeyLists {
    /// From the two files' texts (one key per line). A key in both lists is an error.
    pub fn parse(restate: &str, control: &str) -> Result<KeyLists, String> {
        let set = |t: &str| t.lines().map(|l| l.trim().to_string()).filter(|l| !l.is_empty()).collect::<BTreeSet<_>>();
        let k = KeyLists { restate: set(restate), control: set(control) };
        if let Some(both) = k.restate.intersection(&k.control).next() {
            return Err(format!("key {both:?} is in both lists"));
        }
        Ok(k)
    }

    /// Read `DIR/restate_fact_keys.txt` and `DIR/control_fact_keys.txt` (and nothing else).
    pub fn load(dir: impl AsRef<Path>) -> io::Result<KeyLists> {
        let d = dir.as_ref();
        let r = std::fs::read_to_string(d.join(RESTATE_FILE))?;
        let c = std::fs::read_to_string(d.join(CONTROL_FILE))?;
        KeyLists::parse(&r, &c).map_err(io::Error::other)
    }
}

fn same_name(x: &str, y: &str) -> bool {
    x.trim().eq_ignore_ascii_case(y.trim())
}

/// The test keys a hand-written capital or element fact goes by: `country:`/`element:` + each of its
/// names, the Wikidata country with one of its capitals, the Wikidata element with its symbol.
pub fn test_keys(f: &Fact, wd: &Wikidata) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    match f.rel {
        Rel::Capital => {
            for a in &f.a {
                v.push(format!("country:{a}"));
            }
            for c in wd.countries.values() {
                if c.capitals.iter().any(|cap| f.b.iter().any(|b| same_name(b, cap))) || f.a.iter().any(|a| same_name(a, &c.name)) {
                    v.push(format!("country:{}", c.name));
                }
            }
        }
        Rel::Symbol => {
            for a in &f.a {
                v.push(format!("element:{a}"));
            }
            for e in &wd.elements {
                if e.symbol == f.b[0] || f.a.iter().any(|a| same_name(a, &e.name)) {
                    v.push(format!("element:{}", e.name));
                }
            }
        }
        Rel::Class | Rel::Planet => {}
    }
    v.sort();
    v.dedup();
    v
}

/// A hand-written fact's status and the test key it was matched by.
pub fn own_status(f: &Fact, keys: &KeyLists, wd: &Wikidata, seed: u64, dev_percent: u64) -> (Status, Option<String>) {
    match f.rel {
        Rel::Capital | Rel::Symbol => {
            let ks = test_keys(f, wd);
            if let Some(k) = ks.iter().find(|k| keys.control.contains(*k)) {
                (Status::Control, Some(k.clone()))
            } else if let Some(k) = ks.iter().find(|k| keys.restate.contains(*k)) {
                (Status::Restate, Some(k.clone()))
            } else {
                (Status::Unlisted, None)
            }
        }
        Rel::Class | Rel::Planet => (if is_dev(f, seed, dev_percent) { Status::Dev } else { Status::Train }, None),
    }
}

/// Where a restated key's names came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeyRow {
    pub key: String,
    pub source: &'static str,
    pub fact: Fact,
}

/// The reconciled restatement plan.
#[derive(Clone, Debug)]
pub struct Plan {
    /// Every hand-written fact with its status and matched test key.
    pub own: Vec<(Fact, Status, Option<String>)>,
    /// The facts to restate: own train facts (animals, planets), then one fact per restated test key.
    pub restate: Vec<Fact>,
    /// One row per restated test key.
    pub keys: Vec<KeyRow>,
    /// Restate-list keys left out, with the reason.
    pub skipped: Vec<(String, &'static str)>,
    /// Every control key as a fact (all its names), for the no-mention check.
    pub control: Vec<Fact>,
}

fn add_forms(base: &mut Vec<String>, more: &[String]) {
    for m in more {
        if !base.iter().any(|x| same_name(x, m)) {
            base.push(m.clone());
        }
    }
}

/// Reconcile the hand-written facts with the test keys and Wikidata (see RECONCILIATION). With
/// `wd_elements` false, element keys are restated only when a hand-written fact has them.
pub fn plan(own: Vec<Fact>, keys: &KeyLists, wd: &Wikidata, seed: u64, dev_percent: u64, wd_elements: bool) -> Plan {
    let own: Vec<(Fact, Status, Option<String>)> = own
        .into_iter()
        .map(|f| {
            let (s, k) = own_status(&f, keys, wd, seed, dev_percent);
            (f, s, k)
        })
        .collect();
    let mut restate: Vec<Fact> = own.iter().filter(|(_, s, _)| *s == Status::Train).map(|(f, _, _)| f.clone()).collect();
    // hand-written facts by the key they were matched by
    let by_key = |k: &str, st: Status| own.iter().find(|(_, s, mk)| *s == st && mk.as_deref() == Some(k)).map(|(f, _, _)| f);
    let mut rows = Vec::new();
    let mut skipped = Vec::new();
    for k in &keys.restate {
        let mine = by_key(k, Status::Restate);
        let fact = if let Some(name) = k.strip_prefix("country:") {
            match wd.countries.get(name) {
                Some(c) if c.capitals.len() == 1 => {
                    let mut a = vec![c.name.clone()];
                    let mut b = vec![c.capitals[0].clone()];
                    if let Some(m) = mine {
                        add_forms(&mut a, &m.a);
                        add_forms(&mut b, &m.b);
                    }
                    Some((Fact { rel: Rel::Capital, a, b, c: String::new() }, if mine.is_some() { "wikidata+own" } else { "wikidata" }))
                }
                Some(_) => {
                    skipped.push((k.clone(), "several capitals"));
                    None
                }
                None => match mine {
                    Some(m) => Some((m.clone(), "own")),
                    None => {
                        skipped.push((k.clone(), "no data"));
                        None
                    }
                },
            }
        } else if let Some(name) = k.strip_prefix("element:") {
            match (wd.element(name), mine) {
                (Some(e), m) if wd_elements || m.is_some() => {
                    let mut a = vec![e.name.clone()];
                    if let Some(m) = m {
                        add_forms(&mut a, &m.a);
                    }
                    Some((Fact { rel: Rel::Symbol, a, b: vec![e.symbol.clone()], c: String::new() }, if m.is_some() { "wikidata+own" } else { "wikidata" }))
                }
                (None, Some(m)) => Some((m.clone(), "own")),
                (Some(_), _) => {
                    skipped.push((k.clone(), "element not in the hand list (wikidata elements off)"));
                    None
                }
                (None, None) => {
                    skipped.push((k.clone(), "no data"));
                    None
                }
            }
        } else {
            skipped.push((k.clone(), "unknown key kind"));
            None
        };
        if let Some((f, src)) = fact {
            restate.push(f.clone());
            rows.push(KeyRow { key: k.clone(), source: src, fact: f });
        }
    }
    // the control facts with every name they go by
    let mut control = Vec::new();
    for k in &keys.control {
        let mine = by_key(k, Status::Control);
        if let Some(name) = k.strip_prefix("country:") {
            let mut a = vec![name.to_string()];
            let mut b: Vec<String> = wd.countries.get(name).map(|c| c.capitals.clone()).unwrap_or_default();
            if let Some(m) = mine {
                add_forms(&mut a, &m.a);
                add_forms(&mut b, &m.b);
            }
            if !b.is_empty() {
                control.push(Fact { rel: Rel::Capital, a, b, c: String::new() });
            }
        } else if let Some(name) = k.strip_prefix("element:") {
            let mut a = vec![name.to_string()];
            let mut b: Vec<String> = wd.element(name).map(|e| vec![e.symbol.clone()]).unwrap_or_default();
            if let Some(m) = mine {
                add_forms(&mut a, &m.a);
                add_forms(&mut b, &m.b);
            }
            if !b.is_empty() {
                control.push(Fact { rel: Rel::Symbol, a, b, c: String::new() });
            }
        }
    }
    for (f, s, _) in &own {
        if *s == Status::Control && !control.iter().any(|c| c.rel == f.rel && c.a.iter().any(|x| same_name(x, &f.a[0]))) {
            control.push(f.clone());
        }
    }
    Plan { own, restate, keys: rows, skipped, control }
}

/// Every name a test template could be filled with: all forms of the hand-written facts, the plan's
/// facts, and every Wikidata name (countries, capitals, continents, elements, symbols).
pub fn guard_names(own: &[Fact], wd: &Wikidata) -> Vec<String> {
    let mut v: Vec<String> = Vec::new();
    for f in own {
        v.extend(f.a.iter().cloned());
        v.extend(f.b.iter().cloned());
    }
    v.extend(wd.names());
    v.sort();
    v.dedup();
    v
}

/// The two checks every restatement text must pass: no filled test template, and no sentence that
/// mentions a control fact. Returns a list of problems (empty = clean).
pub fn check_restatements(sentences: &[String], guard: &crate::wording_guard::WordingGuard, control: &[Fact]) -> Vec<String> {
    let mut bad = Vec::new();
    let mut ledger = Ledger::new(control.to_vec());
    for s in sentences {
        for h in guard.hits(s) {
            bad.push(format!("{}: {s:?}", guard.describe(&h)));
        }
        ledger.add_sentence(s);
    }
    for (i, f) in ledger.facts().iter().enumerate() {
        if ledger.mentions[i] > 0 {
            bad.push(format!("control fact {} mentioned {} times", f.id(), ledger.mentions[i]));
        }
    }
    bad
}

// ---------------------------------------------------------------- the ledger

/// Words of a sentence: (as written, lowercase). Letters of any script, or digit runs.
pub fn words(s: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut digits = false;
    let push = |cur: &mut String, out: &mut Vec<(String, String)>| {
        if !cur.is_empty() {
            out.push((cur.clone(), cur.to_lowercase()));
            cur.clear();
        }
    };
    for ch in s.chars() {
        if ch.is_alphabetic() {
            if digits {
                push(&mut cur, &mut out);
            }
            digits = false;
            cur.push(ch);
        } else if ch.is_ascii_digit() {
            if !digits {
                push(&mut cur, &mut out);
            }
            digits = true;
            cur.push(ch);
        } else {
            push(&mut cur, &mut out);
        }
    }
    push(&mut cur, &mut out);
    out
}

/// Split a text into sentences: after . ! ? followed by whitespace, and at every newline.
pub fn sentences(text: &str, mut f: impl FnMut(&str)) {
    for line in text.split('\n') {
        let b = line.as_bytes();
        let mut start = 0;
        for i in 0..b.len() {
            if matches!(b[i], b'.' | b'!' | b'?') && b.get(i + 1).is_some_and(|c| c.is_ascii_whitespace()) {
                f(&line[start..=i]);
                start = i + 1;
            }
        }
        if start < line.len() {
            f(&line[start..]);
        }
    }
}

struct Phrase {
    fact: usize,
    side: u8,
    words: Vec<String>,
    /// Must match this exact (capitalised) form.
    exact: Option<String>,
}

/// Counts mentions and statements of the facts.
pub struct Ledger {
    facts: Vec<Fact>,
    index: FastMap<String, Vec<usize>>,
    phrases: Vec<Phrase>,
    pub mentions: Vec<u64>,
    pub stated: Vec<u64>,
    pub sentences: u64,
}

impl Ledger {
    pub fn new(facts: Vec<Fact>) -> Ledger {
        let mut phrases = Vec::new();
        for (fi, f) in facts.iter().enumerate() {
            for (side, forms) in [(0u8, &f.a), (1u8, &f.b)] {
                for form in forms {
                    let ws: Vec<String> = words(form).into_iter().map(|(_, l)| l).collect();
                    let exact = match (f.rel, side) {
                        (Rel::Symbol, 1) | (Rel::Planet, 0) => Some(form.to_string()),
                        _ => None,
                    };
                    phrases.push(Phrase { fact: fi, side, words: ws, exact });
                }
            }
        }
        let mut index: FastMap<String, Vec<usize>> = new_map();
        for (pi, p) in phrases.iter().enumerate() {
            index.entry(p.words[0].clone()).or_default().push(pi);
        }
        let n = facts.len();
        Ledger { facts, index, phrases, mentions: vec![0; n], stated: vec![0; n], sentences: 0 }
    }

    pub fn facts(&self) -> &[Fact] {
        &self.facts
    }

    /// Count one sentence.
    pub fn add_sentence(&mut self, s: &str) {
        self.sentences += 1;
        let ws = words(s);
        // (fact, side, position)
        let mut found: Vec<(usize, u8, usize)> = Vec::new();
        for i in 0..ws.len() {
            let Some(cands) = self.index.get(&ws[i].1) else { continue };
            for &pi in cands {
                let p = &self.phrases[pi];
                if i + p.words.len() > ws.len() {
                    continue;
                }
                if !p.words.iter().enumerate().all(|(k, w)| ws[i + k].1 == *w) {
                    continue;
                }
                if let Some(e) = &p.exact {
                    if ws[i].0 != *e {
                        continue;
                    }
                }
                found.push((p.fact, p.side, i));
            }
        }
        if found.is_empty() {
            return;
        }
        found.sort_unstable();
        found.dedup();
        let has = |w: &str| ws.iter().any(|(_, l)| l == w);
        let mut i = 0;
        while i < found.len() {
            let f = found[i].0;
            let mut j = i;
            while j < found.len() && found[j].0 == f {
                j += 1;
            }
            let a: Vec<usize> = found[i..j].iter().filter(|x| x.1 == 0).map(|x| x.2).collect();
            let b: Vec<usize> = found[i..j].iter().filter(|x| x.1 == 1).map(|x| x.2).collect();
            if !a.is_empty() && !b.is_empty() {
                self.mentions[f] += 1;
                let stated = match self.facts[f].rel {
                    Rel::Capital => has("capital"),
                    Rel::Symbol => has("symbol"),
                    Rel::Planet => has("planet") && has("sun"),
                    Rel::Class => a.iter().any(|&pa| b.iter().any(|&pb| pb > pa && pb - pa <= 6)),
                };
                self.stated[f] += stated as u64;
            }
            i = j;
        }
    }

    /// Count every sentence of a text.
    pub fn add_text(&mut self, text: &str) {
        let mut buf: Vec<String> = Vec::new();
        sentences(text, |s| buf.push(s.to_string()));
        for s in buf {
            self.add_sentence(&s);
        }
    }

    /// Add another ledger's counts (same fact list).
    pub fn merge(&mut self, o: &Ledger) {
        for i in 0..self.mentions.len() {
            self.mentions[i] += o.mentions[i];
            self.stated[i] += o.stated[i];
        }
        self.sentences += o.sentences;
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::wording_guard::{parse_templates, WordingGuard};

    fn find(id: &str) -> Fact {
        all_facts().into_iter().find(|f| f.id() == id).unwrap()
    }

    #[test]
    fn two_hundred_unique_facts() {
        let f = all_facts();
        assert_eq!(f.len(), 200);
        let mut ids: Vec<String> = f.iter().map(|x| x.id()).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 200);
        for x in &f {
            if x.rel == Rel::Class {
                assert_eq!((x.a.len(), x.b.len()), (2, 2), "{}", x.id());
            }
        }
    }

    #[test]
    fn split_is_seeded_and_about_a_fifth() {
        let f = all_facts();
        let dev: Vec<bool> = f.iter().map(|x| is_dev(x, DEFAULT_SEED, 20)).collect();
        let n = dev.iter().filter(|&&d| d).count();
        assert!((25..=55).contains(&n), "{n} dev facts");
        let again: Vec<bool> = f.iter().map(|x| is_dev(x, DEFAULT_SEED, 20)).collect();
        assert_eq!(dev, again);
        let other: Vec<bool> = f.iter().map(|x| is_dev(x, 7, 20)).collect();
        assert_ne!(dev, other);
    }

    #[test]
    fn restatements_cover_both_directions_and_questions() {
        for f in all_facts() {
            let r = restatements(&f);
            assert!(r.len() >= 5, "{}", f.id());
            let mut u = r.clone();
            u.sort();
            u.dedup();
            assert_eq!(u.len(), r.len());
            assert!(r.iter().any(|s| s.contains('?')));
            for s in &r {
                assert!(s.chars().next().unwrap().is_uppercase(), "{s}");
                assert!(!s.contains("What is the capital of") && !s.contains("What is the chemical symbol for"), "{s}");
            }
        }
        let r = restatements(&find("capital:France"));
        assert!(r.contains(&"The capital of France is Paris.".to_string()));
        assert!(r.contains(&"Paris is the capital of France.".to_string()));
        assert!(r.contains(&"Which country has Paris as its capital? France.".to_string()));
        assert!(restatements(&find("capital:United Kingdom")).contains(&"The United Kingdom's capital city is London.".to_string()));
        assert!(restatements(&find("class:owl")).contains(&"An owl is a bird.".to_string()));
        assert!(restatements(&find("class:ant")).contains(&"What kind of animal is an ant? An insect.".to_string()));
        assert_eq!(country("Marshall Islands"), "the Marshall Islands");
        assert_eq!(country("The Bahamas"), "the Bahamas");
        assert_eq!(country("People's Republic of China"), "the People's Republic of China");
        assert_eq!(country("Central African Republic"), "the Central African Republic");
        assert_eq!(country("Chile"), "Chile");
    }

    #[test]
    fn docs_restate_exactly_the_given_facts() {
        let facts: Vec<Fact> = all_facts().into_iter().filter(|f| !is_dev(f, DEFAULT_SEED, 20)).collect();
        let docs = restatement_docs(&facts, DEFAULT_SEED, 2, 8);
        let text = docs.join(" ");
        for f in all_facts() {
            let n = text.matches(restatements(&f)[0].as_str()).count();
            assert_eq!(n, if is_dev(&f, DEFAULT_SEED, 20) { 0 } else { 2 }, "{}", f.id());
        }
        assert!(docs.iter().all(|d| d.matches(". ").count() + d.matches("? ").count() <= 16));
    }

    #[test]
    fn ledger_counts_mentions_and_statements() {
        let mut l = Ledger::new(all_facts());
        l.add_text("Paris is the capital of France. I visited France and Paris last year.\nKiev is the capital of Ukraine.");
        l.add_text("The symbol for gold is Au. He went to the gold mine. Gold, AU and helium.");
        l.add_text("Whales are mammals. Mammals eat; whales swim. Mars is the fourth planet from the Sun. The fourth earth layer.");
        l.add_text("Earth is the third planet from the Sun. Brasília is the capital of Brazil.");
        let idx = |id: &str| l.facts().iter().position(|f| f.id() == id).unwrap();
        assert_eq!((l.mentions[idx("capital:France")], l.stated[idx("capital:France")]), (2, 1));
        assert_eq!(l.stated[idx("capital:Ukraine")], 1);
        assert_eq!(l.stated[idx("capital:Brazil")], 1);
        assert_eq!((l.mentions[idx("symbol:gold")], l.stated[idx("symbol:gold")]), (1, 1));
        assert_eq!(l.mentions[idx("symbol:helium")], 0, "'He' as a pronoun is not a mention");
        assert_eq!((l.mentions[idx("class:whale")], l.stated[idx("class:whale")]), (2, 1));
        assert_eq!(l.stated[idx("planet:Mars")], 1);
        assert_eq!(l.stated[idx("planet:Earth")], 1);
        assert_eq!(l.sentences, 12);
    }

    #[test]
    fn sentence_split_and_words() {
        let mut v = Vec::new();
        sentences("One. Two? Three!\nFour 3.5 five", |s| v.push(s.to_string()));
        assert_eq!(v, vec!["One.", " Two?", " Three!", "Four 3.5 five"]);
        let w = words("Port-au-Prince, 4th São");
        let low: Vec<&str> = w.iter().map(|x| x.1.as_str()).collect();
        assert_eq!(low, vec!["port", "au", "prince", "4", "th", "são"]);
    }

    fn small_world() -> (KeyLists, Wikidata) {
        let q1 = "countryLabel,capitalLabel,continentLabel\n\
            France,Paris,Europe\nSpain,Madrid,Europe\nPeople's Republic of China,Beijing,Asia\nAndorra,Andorra la Vella,Europe\n\
            Bolivia,Sucre,South America\nBolivia,La Paz,South America\nKingdom of Denmark,Copenhagen,Europe\n";
        let q2 = "elementLabel,symbol,number\nhelium,He,2.0\ngold,Au,79.0\nunbipentium,Ubp,125.0\niron,Fe,26.0\n";
        let wd = Wikidata::parse(q1, q2).unwrap();
        let keys = KeyLists::parse(
            "country:Spain\ncountry:People's Republic of China\ncountry:Andorra\ncountry:Bolivia\nelement:helium\nelement:unbipentium\n",
            "country:France\ncountry:Kingdom of Denmark\nelement:gold\n",
        )
        .unwrap();
        (keys, wd)
    }

    #[test]
    fn key_lists_reject_a_key_in_both() {
        assert!(KeyLists::parse("country:Peru\n", "country:Peru\n").is_err());
    }

    #[test]
    fn plan_follows_the_key_lists() {
        let (keys, wd) = small_world();
        let p = plan(all_facts(), &keys, &wd, DEFAULT_SEED, 20, true);
        let st = |id: &str| p.own.iter().find(|(f, _, _)| f.id() == id).map(|(_, s, k)| (*s, k.clone())).unwrap();
        assert_eq!(st("capital:France").0, Status::Control);
        assert_eq!(st("capital:Denmark"), (Status::Control, Some("country:Kingdom of Denmark".into())), "matched by its capital");
        assert_eq!(st("capital:China"), (Status::Restate, Some("country:People's Republic of China".into())));
        assert_eq!(st("capital:Spain").0, Status::Restate);
        assert_eq!(st("capital:Peru").0, Status::Unlisted);
        assert_eq!(st("symbol:gold").0, Status::Control);
        assert_eq!(st("symbol:helium").0, Status::Restate);
        assert_eq!(st("symbol:iron").0, Status::Unlisted);
        // animals and planets keep the own split
        for (f, s, _) in &p.own {
            if matches!(f.rel, Rel::Class | Rel::Planet) {
                assert_eq!(*s == Status::Dev, is_dev(f, DEFAULT_SEED, 20));
            }
        }
        // restated: Spain, China (Wikidata name first, own forms kept), Andorra, helium, unbipentium; Bolivia skipped
        let ids: Vec<String> = p.keys.iter().map(|r| r.key.clone()).collect();
        assert_eq!(ids.len(), 5, "{ids:?}");
        assert_eq!(p.skipped, vec![("country:Bolivia".to_string(), "several capitals")]);
        let china = p.restate.iter().find(|f| f.a[0] == "People's Republic of China").unwrap();
        assert_eq!(china.a, vec!["People's Republic of China", "China"]);
        assert_eq!(china.b, vec!["Beijing", "Peking"]);
        assert!(p.restate.iter().any(|f| f.a[0] == "unbipentium" && f.b[0] == "Ubp"));
        // never France, Denmark, gold, Peru or iron
        let text = restatement_docs(&p.restate, 1, 1, 8).join(" ");
        for name in ["Paris", "Copenhagen", "Denmark", "Au", "Lima", "Fe", "Sucre"] {
            assert!(!text.split(|c: char| !c.is_alphanumeric()).any(|w| w == name), "{name} restated");
        }
        // without Wikidata elements only the hand-written element is restated
        let p2 = plan(all_facts(), &keys, &wd, DEFAULT_SEED, 20, false);
        assert!(p2.restate.iter().any(|f| f.a[0] == "helium") && !p2.restate.iter().any(|f| f.a[0] == "unbipentium"));
        // the control facts carry every name
        assert!(p.control.iter().any(|f| f.a.contains(&"Kingdom of Denmark".to_string()) && f.a.contains(&"Denmark".to_string())));
        let sentences: Vec<String> = p.restate.iter().flat_map(restatements).collect();
        let g = WordingGuard::new(parse_templates("Q: What is the capital of {}?\\nA:").unwrap(), guard_names(&all_facts(), &wd));
        assert!(check_restatements(&sentences, &g, &p.control).is_empty());
        // and the check does catch both kinds of problem
        let bad = vec!["Q: What is the capital of Spain?".to_string(), "Paris is the capital of France.".to_string()];
        assert_eq!(check_restatements(&bad, &g, &p.control).len(), 2);
    }

    /// The real v42 key lists, templates and Wikidata (skipped when they are not on this machine): every
    /// template filled with every name is absent from every restatement, checked by plain search.
    #[test]
    fn real_restatements_hold_no_filled_template() {
        let dir = Path::new(KEYS_DIR);
        if !dir.join(TEMPLATES_FILE).exists() || !Path::new(WIKIDATA_DIR).join("q1.csv").exists() {
            eprintln!("v42 key files not found; skipped");
            return;
        }
        let keys = KeyLists::load(dir).unwrap();
        let wd = Wikidata::load(WIKIDATA_DIR).unwrap();
        let p = plan(all_facts(), &keys, &wd, DEFAULT_SEED, DEFAULT_DEV_PERCENT, true);
        let names = guard_names(&all_facts(), &wd);
        let g = WordingGuard::load(dir.join(TEMPLATES_FILE), names.clone()).unwrap();
        let mut sentences: Vec<String> = p.restate.iter().flat_map(restatements).collect();
        sentences.sort();
        sentences.dedup();
        let low: Vec<String> = sentences.iter().map(|s| s.to_ascii_lowercase()).collect();
        let mut filled = 0usize;
        for t in 0..g.templates.len() {
            for n in &names {
                let core = g.filled_core(t, n).to_ascii_lowercase();
                filled += 1;
                assert!(!low.iter().any(|s| s.contains(&core)), "filled template in a restatement: {core:?}");
            }
        }
        assert!(filled >= 6 * 500, "{filled}");
        assert!(check_restatements(&sentences, &g, &p.control).is_empty());
        // no control key is restated, and every restated key is in the restate list
        for r in &p.keys {
            assert!(keys.restate.contains(&r.key) && !keys.control.contains(&r.key));
        }
        assert!(p.keys.len() >= 150, "{} keys restated", p.keys.len());
    }
}
