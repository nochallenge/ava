// TESTING / DESIGN ONLY
//! bios: synthetic biographies to measure how many bits of knowledge she can store (M4), after
//! Allen-Zhu and Li, "Physics of Language Models, Part 3.3" (the bioS data and its bit count).
//!
//! A PERSON has a unique made-up full name (first name, two middle initials, last name: 64 x 26 x 26
//! x 96 = 4,153,344 possible names) and six attributes, each drawn uniformly and independently:
//!   birth_day 1..=28 (28), birth_month (12), birth_year 1900..=1999 (100), birth_city (64 made-up
//!   towns), job (64), employer (32 made-up companies).
//! So every person carries exactly log2(28*12*100*64*64*32) = 32.04 bits that no other text can give.
//!
//! AN EXPOSURE is one short biography: four sentences (birth date, birth town, job, employer), each
//! from one of three templates picked at random, in a random order, each naming the person in full
//! (the paper's "multi-template, permuted, full name" setting, which lets knowledge be stored so it can
//! later be pulled out). Round r writes one exposure of every person, in a fresh seeded order; `E`
//! rounds give every person E exposures, spread through the whole file.
//!
//! PROBES. One prompt per person and attribute, ending right before the value (`probes`); the answer
//! is one of the attribute's known choices (`choices`). Score a model by the likelihood of every choice
//! after the prompt: accuracy = the right choice wins; loss = -log2 of the right choice's probability
//! renormalised over the choices.
//!
//! BITS STORED (the paper's measure, per attribute with S equally likely values, over N people):
//!   from the loss L (bits):     N * max(0, log2 S - L)
//!   from the accuracy a (a lower bound by Fano's inequality):
//!                               N * max(0, log2 S - h(a) - (1 - a) * log2(S - 1)),  h = binary entropy
//! summed over the six attributes. Divide by the parameter count for bits per parameter (the paper
//! finds about 2 bits per parameter for well-trained transformers at enough exposures).

use crate::hash::FastSet;
use crate::rng::{derive_seed, SplitMix64};

pub const FIRST: [&str; 64] = [
    "Alder", "Brenna", "Calder", "Delphine", "Emrys", "Fenella", "Garrick", "Hester", "Ivo", "Jessamy", "Kendrick", "Linnea",
    "Marek", "Nerys", "Osric", "Perpetua", "Quillon", "Rosalind", "Soren", "Tamsin", "Ulric", "Verity", "Wystan", "Xanthe",
    "Yorick", "Zinnia", "Ambrose", "Briony", "Cosmo", "Dulcie", "Evander", "Fenwick", "Gwendolyn", "Hollis", "Isolde", "Jasper",
    "Kestrel", "Leopold", "Marigold", "Niall", "Ottilie", "Peregrine", "Rowena", "Silas", "Thea", "Upton", "Vesper", "Winslow",
    "Yara", "Zeb", "Anselm", "Bettany", "Crispin", "Dorcas", "Elowen", "Florian", "Greer", "Hamish", "Ines", "Jory", "Kit",
    "Lorcan", "Mirren", "Nyle",
];

pub const LAST: [&str; 96] = [
    "Ashcombe", "Blackwood", "Carrow", "Dunmore", "Elsworth", "Fairbairn", "Glenister", "Harwell", "Ingleby", "Jessop", "Kettering",
    "Lockhart", "Maybury", "Northcott", "Oakshott", "Pellow", "Quarrie", "Rendall", "Saltmarsh", "Tredwell", "Upcott", "Vardy",
    "Whitlock", "Yelland", "Ackroyd", "Brackley", "Cheverton", "Dimmock", "Eversley", "Fothergill", "Gascoigne", "Hindley",
    "Ivers", "Jolliffe", "Kingdon", "Lavington", "Merriman", "Nettleship", "Ormerod", "Pargeter", "Quilter", "Rowntree",
    "Sherwin", "Thistlewood", "Underhill", "Venning", "Wetherby", "Yardley", "Alcott", "Bramhall", "Coldwell", "Darnell",
    "Eggleston", "Fenimore", "Gorringe", "Hazeldine", "Ilsley", "Jarrold", "Kimber", "Lumsden", "Mottram", "Nuttall", "Oddie",
    "Penhaligon", "Ruddock", "Satterly", "Tolhurst", "Uttley", "Varley", "Wakeling", "Yeoman", "Abernethy", "Bellingham",
    "Cottingham", "Dewhurst", "Entwistle", "Farrimond", "Grimshaw", "Hepplewhite", "Kershaw", "Lindqvist", "Mallory",
    "Nightingale", "Ollerenshaw", "Pickersgill", "Ravenscroft", "Sowerby", "Tunnicliffe", "Wainwright", "Ashdown", "Birtwistle",
    "Cruikshank", "Duckworth", "Fairclough", "Greatorex", "Huxtable",
];

pub const MONTHS: [&str; 12] =
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];

const TOWN_HEADS: [&str; 8] = ["Ash", "Bram", "Cold", "Dun", "Elm", "Fen", "Gray", "Hol"];
const TOWN_TAILS: [&str; 8] = ["bury", "ford", "stead", "field", "port", "combe", "by", "worth"];

pub const JOBS: [&str; 64] = [
    "accountant", "actuary", "architect", "archivist", "astronomer", "auditor", "baker", "barber", "biologist", "bookbinder",
    "botanist", "bricklayer", "butcher", "carpenter", "cartographer", "chemist", "chef", "clockmaker", "dentist", "designer",
    "economist", "editor", "electrician", "engineer", "farmer", "firefighter", "florist", "geologist", "glassblower", "goldsmith",
    "historian", "illustrator", "interpreter", "jeweller", "journalist", "judge", "lawyer", "librarian", "locksmith", "mechanic",
    "midwife", "miller", "musician", "nurse", "optician", "painter", "pharmacist", "photographer", "physicist", "pilot",
    "plumber", "potter", "printer", "sailor", "sculptor", "surgeon", "surveyor", "tailor", "teacher", "translator", "upholsterer",
    "veterinarian", "weaver", "zoologist",
];

const COMPANY_HEADS: [&str; 8] = ["Northwind", "Bluestone", "Redfern", "Silverline", "Oakridge", "Greenhill", "Starling", "Ironbridge"];
const COMPANY_TAILS: [&str; 4] = ["Mills", "Works", "Labs", "Trading"];

/// (attribute, number of equally likely values).
pub const ATTRS: [(&str, usize); 6] = [("birth_day", 28), ("birth_month", 12), ("birth_year", 100), ("birth_city", 64), ("job", 64), ("employer", 32)];
pub const YEAR0: u16 = 1900;
pub const DEFAULT_SEED: u64 = 33;
const PEOPLE_LABEL: u64 = 0x6269_6f73_5f70_706c;
const ROUND_LABEL: u64 = 0x6269_6f73_5f72_6e64;

pub fn town(i: usize) -> String {
    format!("{}{}", TOWN_HEADS[i / 8], TOWN_TAILS[i % 8])
}

pub fn company(i: usize) -> String {
    format!("{} {}", COMPANY_HEADS[i / 4], COMPANY_TAILS[i % 4])
}

fn art(w: &str) -> &'static str {
    if w.starts_with(['a', 'e', 'i', 'o', 'u']) {
        "an"
    } else {
        "a"
    }
}

/// The choices of an attribute, as they follow a probe prompt (with the leading space).
pub fn choices(attr: usize) -> Vec<String> {
    match attr {
        0 => (1..=28).map(|d| format!(" {d}")).collect(),
        1 => MONTHS.iter().map(|m| format!(" {m}")).collect(),
        2 => (0..100).map(|y| format!(" {}", YEAR0 + y)).collect(),
        3 => (0..64).map(|i| format!(" {}", town(i))).collect(),
        4 => JOBS.iter().map(|j| format!(" {} {j}", art(j))).collect(),
        5 => (0..32).map(|i| format!(" {}", company(i))).collect(),
        _ => panic!("no attribute {attr}"),
    }
}

/// Bits of one person's attributes: log2 of the product of the choice counts.
pub fn entropy_bits() -> f64 {
    ATTRS.iter().map(|&(_, s)| (s as f64).log2()).sum()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Person {
    pub name: String,
    /// Index of each attribute's value (day 0 = the 1st, year 0 = 1900).
    pub vals: [usize; 6],
}

impl Person {
    pub fn value(&self, attr: usize) -> String {
        choices(attr)[self.vals[attr]].trim_start().to_string()
    }
}

/// `n` people with unique names (n <= 4,153,344), all attributes uniform.
pub fn people(n: usize, seed: u64) -> Vec<Person> {
    let space = FIRST.len() * 26 * 26 * LAST.len();
    assert!(n <= space, "at most {space} unique names");
    let mut r = SplitMix64::new(derive_seed(seed, PEOPLE_LABEL, 0));
    let mut seen: FastSet<(usize, usize, usize, usize)> = FastSet::default();
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let key = (r.below(FIRST.len() as u64) as usize, r.below(26) as usize, r.below(26) as usize, r.below(LAST.len() as u64) as usize);
        if !seen.insert(key) {
            continue;
        }
        let name = format!("{} {}. {}. {}", FIRST[key.0], (b'A' + key.1 as u8) as char, (b'A' + key.2 as u8) as char, LAST[key.3]);
        let mut vals = [0usize; 6];
        for (k, v) in vals.iter_mut().enumerate() {
            *v = r.below(ATTRS[k].1 as u64) as usize;
        }
        out.push(Person { name, vals });
    }
    out
}

/// One exposure: four sentences, random templates, random order, the full name in each.
pub fn render(p: &Person, r: &mut SplitMix64) -> String {
    let n = &p.name;
    let (d, m, y) = (p.vals[0] + 1, MONTHS[p.vals[1]], YEAR0 as usize + p.vals[2]);
    let (city, job, firm) = (town(p.vals[3]), JOBS[p.vals[4]], company(p.vals[5]));
    let a = art(job);
    let date = [
        format!("{n} was born on {m} {d}, {y}."),
        format!("{n} came into this world on {m} {d}, {y}."),
        format!("{n} celebrates a birthday on {m} {d}, having been born in {y}."),
    ];
    let place = [format!("{n} was born in {city}."), format!("{n} spent the early years of life in {city}."), format!("{n}'s birthplace is {city}.")];
    let work = [format!("{n} works as {a} {job}."), format!("{n} earns a living as {a} {job}."), format!("{n} has a career as {a} {job}.")];
    let boss = [format!("{n} is employed by {firm}."), format!("{n} works for {firm}."), format!("{n} draws a salary from {firm}.")];
    let mut s = vec![
        date[r.below(3) as usize].clone(),
        place[r.below(3) as usize].clone(),
        work[r.below(3) as usize].clone(),
        boss[r.below(3) as usize].clone(),
    ];
    r.shuffle(&mut s);
    s.join(" ")
}

/// Round `round` of the training text: one exposure of every person, in a seeded order.
pub fn round_docs(ps: &[Person], seed: u64, round: u64) -> Vec<String> {
    let mut r = SplitMix64::new(derive_seed(seed, ROUND_LABEL, round));
    let mut order: Vec<usize> = (0..ps.len()).collect();
    r.shuffle(&mut order);
    order.into_iter().map(|i| render(&ps[i], &mut r)).collect()
}

/// (attribute index, prompt, answer with its leading space) for one person.
pub fn probes(p: &Person) -> Vec<(usize, String, String)> {
    let n = &p.name;
    let m = MONTHS[p.vals[1]];
    let d = p.vals[0] + 1;
    let c = |k: usize| choices(k)[p.vals[k]].clone();
    vec![
        (0, format!("{n} was born on {m}"), c(0)),
        (1, format!("{n} was born on"), c(1)),
        (2, format!("{n} was born on {m} {d},"), c(2)),
        (3, format!("{n} was born in"), c(3)),
        (4, format!("{n} works as"), c(4)),
        (5, format!("{n} is employed by"), c(5)),
    ]
}

fn h2(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        0.0
    } else {
        -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
    }
}

/// Bits stored, from per-attribute accuracy: `n * sum max(0, log2 S - h(a) - (1-a) log2(S-1))`.
pub fn bits_from_accuracy(n_people: u64, per_attr: &[(usize, f64)]) -> f64 {
    per_attr
        .iter()
        .map(|&(s, a)| {
            let s = s as f64;
            let a = a.clamp(0.0, 1.0);
            let b = s.log2() - h2(a) - (1.0 - a) * (s - 1.0).max(1.0).log2();
            n_people as f64 * b.max(0.0)
        })
        .sum()
}

/// Bits stored, from per-attribute mean loss in bits (over the S choices): `n * sum max(0, log2 S - L)`.
pub fn bits_from_loss(n_people: u64, per_attr: &[(usize, f64)]) -> f64 {
    per_attr.iter().map(|&(s, l)| n_people as f64 * ((s as f64).log2() - l).max(0.0)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn people_are_unique_and_seeded() {
        let ps = people(5000, 1);
        let mut names: Vec<&str> = ps.iter().map(|p| p.name.as_str()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), 5000);
        assert_eq!(ps, people(5000, 1));
        assert_ne!(ps[0], people(5000, 2)[0]);
        for p in &ps {
            for k in 0..6 {
                assert!(p.vals[k] < ATTRS[k].1);
            }
        }
        // uniform enough: every month is used
        let mut months = [0; 12];
        for p in &ps {
            months[p.vals[1]] += 1;
        }
        assert!(months.iter().all(|&c| c > 300));
    }

    #[test]
    fn entropy_is_the_sum_of_choice_bits() {
        assert!((entropy_bits() - (28.0f64 * 12.0 * 100.0 * 64.0 * 64.0 * 32.0).log2()).abs() < 1e-9);
        for (k, &(_, s)) in ATTRS.iter().enumerate() {
            let c = choices(k);
            assert_eq!(c.len(), s);
            let mut u = c.clone();
            u.sort();
            u.dedup();
            assert_eq!(u.len(), s, "choices of {k} are distinct");
        }
    }

    #[test]
    fn render_holds_every_value_and_the_name_four_times() {
        let ps = people(50, 3);
        let mut r = SplitMix64::new(9);
        for p in &ps {
            let t = render(p, &mut r);
            assert_eq!(t.matches(&p.name).count(), 4, "{t}");
            for (k, prompt, ans) in probes(p) {
                assert!(t.contains(ans.trim_start()), "{k}: {ans:?} in {t}");
                assert!(prompt.starts_with(&p.name));
            }
            assert!(t.contains(&format!("{}", YEAR0 as usize + p.vals[2])));
        }
        let rd = round_docs(&ps, 3, 0);
        assert_eq!(rd.len(), 50);
        assert_ne!(rd, round_docs(&ps, 3, 1));
    }

    #[test]
    fn probes_match_the_first_templates() {
        let p = &people(1, 4)[0];
        let mut r = SplitMix64::new(0);
        // some exposure uses the first date template, so prompt + answer appears verbatim
        let mut seen = false;
        for _ in 0..200 {
            let t = render(p, &mut r);
            let (_, prompt, ans) = &probes(p)[2];
            if t.contains(&format!("{prompt}{ans}")) {
                seen = true;
            }
        }
        assert!(seen);
    }

    #[test]
    fn bit_formulas() {
        // perfect recall: all bits; chance: none
        assert!((bits_from_accuracy(10, &[(64, 1.0)]) - 60.0).abs() < 1e-9);
        assert!(bits_from_accuracy(10, &[(64, 1.0 / 64.0)]).abs() < 1e-9);
        assert!((bits_from_loss(10, &[(64, 0.0)]) - 60.0).abs() < 1e-9);
        assert_eq!(bits_from_loss(10, &[(64, 7.0)]), 0.0);
        // more accuracy, more bits
        let b = |a| bits_from_accuracy(1, &[(28, a)]);
        assert!(b(0.5) < b(0.8) && b(0.8) < b(0.99));
        // the accuracy bound is never above the exact loss bits of the same model when its wrong
        // guesses are uniform: L = -a log2 a - (1-a) log2((1-a)/(S-1)) gives equality
        let (s, a) = (28.0f64, 0.7f64);
        let l = -a * a.log2() - (1.0 - a) * ((1.0 - a) / (s - 1.0)).log2();
        assert!((bits_from_loss(1, &[(28, l)]) - bits_from_accuracy(1, &[(28, a)])).abs() < 1e-9);
    }
}
