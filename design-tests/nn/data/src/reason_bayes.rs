// TESTING / DESIGN ONLY
//! reason_bayes (D8): beliefs updated turn by turn as evidence arrives, in words and in exact fractions.
//!
//! FRAMES (none of them the test's bags of red and blue balls): coins in a box (fair, bent, two-headed),
//! dice (plain, loaded, two sixes), a wet or dry spell of weather, a sensor that is sometimes wrong, lost
//! keys searched for room by room (a search that can miss), and spinners. One hypothesis is the truth
//! (drawn from the prior); the evidence is drawn from it, so most runs drift toward the truth, some do not,
//! and sometimes the truth is shown at the end, even when the evidence pointed the other way.
//!
//! EACH TURN shows one update in one of two forms: prior times likelihood for every hypothesis, summed and
//! divided (probabilities), or odds multiplied by the likelihood ratio (two hypotheses). Then a sentence in
//! words (which is ahead, by how much, and whether it moved). Some runs end with the chance of the next
//! observation. All arithmetic is exact (u128 fractions, reduced).

use crate::reason::{chance, gcd, pick, range, sample_k};
use crate::rng::SplitMix64;

/// An exact fraction, kept reduced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Frac {
    pub n: u128,
    pub d: u128,
}

impl Frac {
    pub fn new(n: u128, d: u128) -> Frac {
        assert!(d > 0);
        let g = gcd(n, d).max(1);
        Frac { n: n / g, d: d / g }
    }
    pub fn mul(self, o: Frac) -> Frac {
        Frac::new(self.n * o.n, self.d * o.d)
    }
    pub fn add(self, o: Frac) -> Frac {
        Frac::new(self.n * o.d + o.n * self.d, self.d * o.d)
    }
    pub fn div(self, o: Frac) -> Frac {
        assert!(o.n > 0);
        Frac::new(self.n * o.d, self.d * o.n)
    }
    pub fn one_minus(self) -> Frac {
        Frac::new(self.d - self.n, self.d)
    }
    pub fn f(self) -> f64 {
        self.n as f64 / self.d as f64
    }
    pub fn text(self) -> String {
        if self.n == 0 {
            "0".into()
        } else if self.n == self.d {
            "1".into()
        } else {
            format!("{}/{}", self.n, self.d)
        }
    }
}

#[derive(Clone, Debug)]
pub struct Hyp {
    pub name: String,
    pub prior: Frac,
    /// Likelihood of each kind of observation.
    pub lik: Vec<Frac>,
}

#[derive(Clone, Debug)]
pub struct Frame {
    pub intro: Vec<String>,
    pub hyps: Vec<Hyp>,
    /// How each kind of observation is told ("the coin lands heads").
    pub obs: Vec<String>,
    /// The thing predicted at the end ("heads"), as an observation index, if any.
    pub predict: Option<usize>,
    /// Keys search: each turn looks in one place (the observation kinds are "found" / "not found" there).
    pub search: Option<Search>,
}

#[derive(Clone, Debug)]
pub struct Search {
    pub detect: Frac,
}

fn coin_frame(r: &mut SplitMix64) -> Frame {
    let kinds: [(&str, Frac); 5] = [
        ("the fair coin", Frac::new(1, 2)),
        ("the two-headed coin", Frac::new(1, 1)),
        ("the coin bent toward heads", Frac::new(3, 4)),
        ("the coin bent toward tails", Frac::new(1, 4)),
        ("the slightly bent coin", Frac::new(2, 3)),
    ];
    let k = if chance(r, 1, 4) { 3 } else { 2 };
    let pickk = sample_k(r, kinds.len(), k);
    let counts: Vec<u128> = (0..k).map(|_| range(r, 1, 3) as u128).collect();
    let total: u128 = counts.iter().sum();
    let hyps: Vec<Hyp> = pickk
        .iter()
        .zip(&counts)
        .map(|(&i, &c)| Hyp { name: kinds[i].0.to_string(), prior: Frac::new(c, total), lik: vec![kinds[i].1, kinds[i].1.one_minus()] })
        .collect();
    let parts: Vec<String> = pickk.iter().zip(&counts).map(|(&i, &c)| format!("{} {}", c, if c == 1 { kinds[i].0.replacen("the ", "", 1) } else { kinds[i].0.replacen("the ", "", 1).replacen("coin", "coins", 1) })).collect();
    let intro = vec![
        format!("A box holds {} coins: {}.", total, parts.join(", ")),
        "One coin is taken out without looking, and we flip it again and again.".to_string(),
        format!("Before any flip, the chances follow the counts: {}.", hyps.iter().map(|h| format!("{} {}", h.name, h.prior.text())).collect::<Vec<_>>().join(", ")),
    ];
    Frame { intro, hyps, obs: vec!["The coin lands heads.".into(), "The coin lands tails.".into()], predict: Some(0), search: None }
}

fn dice_frame(r: &mut SplitMix64) -> Frame {
    let kinds: [(&str, Frac); 3] = [("the plain die", Frac::new(1, 6)), ("the loaded die", Frac::new(1, 2)), ("the die with two sixes", Frac::new(1, 3))];
    let pk = sample_k(r, 3, 2);
    let (a, b) = (range(r, 1, 3) as u128, range(r, 1, 3) as u128);
    let hyps = vec![
        Hyp { name: kinds[pk[0]].0.into(), prior: Frac::new(a, a + b), lik: vec![kinds[pk[0]].1, kinds[pk[0]].1.one_minus()] },
        Hyp { name: kinds[pk[1]].0.into(), prior: Frac::new(b, a + b), lik: vec![kinds[pk[1]].1, kinds[pk[1]].1.one_minus()] },
    ];
    let intro = vec![
        format!("A cup holds {} dice of two kinds: {a} like {} and {b} like {}.", a + b, kinds[pk[0]].0, kinds[pk[1]].0),
        format!("A six comes up {} of the time on {}, and {} of the time on {}.", kinds[pk[0]].1.text(), kinds[pk[0]].0, kinds[pk[1]].1.text(), kinds[pk[1]].0),
        "One die is shaken out and rolled, over and over; we only hear whether it shows a six.".to_string(),
    ];
    Frame { intro, hyps, obs: vec!["It shows a six.".into(), "No six this time.".into()], predict: Some(0), search: None }
}

fn weather_frame(r: &mut SplitMix64) -> Frame {
    let wet = *pick(r, &[Frac::new(3, 4), Frac::new(2, 3), Frac::new(4, 5)]);
    let dry = *pick(r, &[Frac::new(1, 5), Frac::new(1, 4), Frac::new(1, 3)]);
    let pw = *pick(r, &[Frac::new(1, 2), Frac::new(1, 3), Frac::new(1, 4), Frac::new(2, 3)]);
    let hyps = vec![
        Hyp { name: "a wet spell".into(), prior: pw, lik: vec![wet, wet.one_minus()] },
        Hyp { name: "a dry spell".into(), prior: pw.one_minus(), lik: vec![dry, dry.one_minus()] },
    ];
    let intro = vec![
        "The village keeps a weather book. Each month is either a wet spell or a dry spell.".to_string(),
        format!("In a wet spell it rains on {} of the days; in a dry spell on {} of them.", wet.text(), dry.text()),
        format!("A month starts, and from the book a wet spell has a chance of {}. We note each day as it comes.", pw.text()),
    ];
    Frame { intro, hyps, obs: vec!["It rains today.".into(), "A day with no rain.".into()], predict: Some(0), search: None }
}

fn sensor_frame(r: &mut SplitMix64) -> Frame {
    let (thing, yes, no, reading_yes, reading_no) = *pick(
        r,
        &[
            ("the cellar is damp", "damp", "dry", "The sensor reads damp.", "The sensor reads dry."),
            ("there is smoke in the attic", "smoke", "no smoke", "The detector beeps.", "The detector stays quiet."),
            ("the bridge is icy", "ice", "no ice", "The road sign blinks ICE.", "The road sign stays dark."),
            ("the plant needs water", "thirsty", "fine", "The soil probe says thirsty.", "The soil probe says fine."),
        ],
    );
    let hit = *pick(r, &[Frac::new(9, 10), Frac::new(4, 5), Frac::new(3, 4)]);
    let false_alarm = *pick(r, &[Frac::new(1, 10), Frac::new(1, 5), Frac::new(1, 4)]);
    let prior = *pick(r, &[Frac::new(1, 10), Frac::new(1, 5), Frac::new(1, 4), Frac::new(1, 2)]);
    let hyps = vec![
        Hyp { name: yes.to_string(), prior, lik: vec![hit, hit.one_minus()] },
        Hyp { name: no.to_string(), prior: prior.one_minus(), lik: vec![false_alarm, false_alarm.one_minus()] },
    ];
    let intro = vec![
        format!("We want to know whether {thing}. Before checking, the chance of {yes} is {}.", prior.text()),
        format!("The sensor is sometimes wrong: with {yes} it says so {} of the time; with {no} it still says {yes} {} of the time.", hit.text(), false_alarm.text()),
        "Each reading is taken fresh, so the readings do not depend on each other, only on the truth.".to_string(),
    ];
    Frame { intro, hyps, obs: vec![reading_yes.into(), reading_no.into()], predict: None, search: None }
}

fn keys_frame(r: &mut SplitMix64) -> Frame {
    let places = ["the kitchen", "the hall", "the car", "the garden shed", "the office"];
    let k = 3;
    let pk = sample_k(r, places.len(), k);
    let w: Vec<u128> = (0..k).map(|_| range(r, 1, 4) as u128).collect();
    let tot: u128 = w.iter().sum();
    let detect = *pick(r, &[Frac::new(2, 3), Frac::new(3, 4), Frac::new(1, 2)]);
    let hyps: Vec<Hyp> = pk.iter().zip(&w).map(|(&i, &c)| Hyp { name: places[i].to_string(), prior: Frac::new(c, tot), lik: vec![] }).collect();
    let intro = vec![
        "The keys are lost, and they are in one of three places.".to_string(),
        format!("From past days the chances are: {}.", hyps.iter().map(|h| format!("{} {}", h.name, h.prior.text())).collect::<Vec<_>>().join(", ")),
        format!("A quick look in the right place finds them only {} of the time, so a miss does not rule a place out.", detect.text()),
    ];
    Frame { intro, hyps, obs: vec![], predict: None, search: Some(Search { detect }) }
}

fn spinner_frame(r: &mut SplitMix64) -> Frame {
    let a = range(r, 1, 5) as u128;
    let mut b = range(r, 1, 5) as u128;
    if b == a {
        b = if a == 5 { 1 } else { a + 1 };
    }
    let hyps = vec![
        Hyp { name: "the striped spinner".into(), prior: Frac::new(1, 2), lik: vec![Frac::new(a, 6), Frac::new(6 - a, 6)] },
        Hyp { name: "the dotted spinner".into(), prior: Frac::new(1, 2), lik: vec![Frac::new(b, 6), Frac::new(6 - b, 6)] },
    ];
    let intro = vec![
        "Two spinners are cut into six equal parts each.".to_string(),
        format!("The striped spinner has {a} green parts out of 6; the dotted spinner has {b} out of 6."),
        "A friend spins one of them behind a screen, each just as likely, and calls out what it lands on.".to_string(),
    ];
    Frame { intro, hyps, obs: vec!["It lands on green.".into(), "It lands on a part that is not green.".into()], predict: Some(0), search: None }
}

fn belief_words(hyps: &[Hyp], post: &[Frac], before: &[Frac]) -> String {
    let lead = (0..post.len()).max_by(|&a, &b| post[a].f().partial_cmp(&post[b].f()).unwrap()).unwrap();
    let ties = post.iter().filter(|p| **p == post[lead]).count();
    if ties > 1 {
        return "The evidence and the starting chances now balance exactly: no hypothesis is ahead.".to_string();
    }
    let p = post[lead].f();
    let how = if p >= 0.99 {
        "almost certainly"
    } else if p >= 0.9 {
        "very likely"
    } else if p >= 0.7 {
        "fairly likely"
    } else if p > 0.5 {
        "a little more likely than not"
    } else {
        "the best guess, but less likely than all the others together"
    };
    let moved = if post[lead].f() > before[lead].f() {
        "up"
    } else if post[lead].f() < before[lead].f() {
        "down"
    } else {
        "unchanged"
    };
    format!("In words: it is {how} {} ({}, {moved} from {}).", hyps[lead].name, post[lead].text(), before[lead].text())
}

/// One full document.
fn run(r: &mut SplitMix64, f: &Frame) -> String {
    let mut out: Vec<String> = Vec::new();
    out.push(pick(r, &["Updating a belief, step by step.", "Evidence over several turns.", "Which is it? A belief that changes as news arrives."]).to_string());
    out.extend(f.intro.iter().cloned());
    // the truth, drawn from the prior
    let mut x = r.below(1_000_000) as f64 / 1e6;
    let mut truth = f.hyps.len() - 1;
    for (i, h) in f.hyps.iter().enumerate() {
        if x < h.prior.f() {
            truth = i;
            break;
        }
        x -= h.prior.f();
    }
    let mut post: Vec<Frac> = f.hyps.iter().map(|h| h.prior).collect();
    let turns = range(r, 2, 8) as usize;
    let odds_form = f.hyps.len() == 2 && f.search.is_none() && chance(r, 1, 3);
    let dialogue = chance(r, 1, 4);
    let (sa, sb) = *pick(r, &[("Kim", "Jo"), ("Ana", "Bo"), ("Lee", "Max"), ("Rosa", "Finn")]);
    for t in 1..=turns {
        // the evidence of this turn
        let (said, lik): (String, Vec<Frac>) = if let Some(s) = &f.search {
            let lead = (0..post.len()).max_by(|&a, &b| post[a].f().partial_cmp(&post[b].f()).unwrap().then(b.cmp(&a))).unwrap();
            let found = lead == truth && r.below(1_000_000) as f64 / 1e6 < s.detect.f();
            let lik: Vec<Frac> = (0..post.len())
                .map(|i| match (i == lead, found) {
                    (true, true) => s.detect,
                    (true, false) => s.detect.one_minus(),
                    (false, true) => Frac::new(0, 1),
                    (false, false) => Frac::new(1, 1),
                })
                .collect();
            let said = if found { format!("We look in {}, the most likely place: the keys are there!", f.hyps[lead].name) } else { format!("We look in {}, the most likely place, and do not find them.", f.hyps[lead].name) };
            (said, lik)
        } else {
            let p = f.hyps[truth].lik[0].f();
            let o = if (r.below(1_000_000) as f64 / 1e6) < p { 0 } else { 1 };
            (f.obs[o].clone(), f.hyps.iter().map(|h| h.lik[o]).collect())
        };
        let found_now = f.search.is_some() && said.ends_with("there!");
        out.push(if dialogue { format!("{sa}: {said}") } else { format!("Turn {t}. {said}") });
        let prods: Vec<Frac> = post.iter().zip(&lik).map(|(p, l)| p.mul(*l)).collect();
        let total = prods.iter().fold(Frac::new(0, 1), |a, b| a.add(*b));
        let new: Vec<Frac> = prods.iter().map(|p| p.div(total)).collect();
        let who = if dialogue { format!("{sb}: ") } else { String::new() };
        if odds_form {
            // odds a:b times the likelihood ratio, both scaled to whole numbers
            let (a, b) = (post[0], post[1]);
            let (oa, ob) = (a.n * b.d, b.n * a.d);
            let g = gcd(oa, ob).max(1);
            let (oa, ob) = (oa / g, ob / g);
            let den = lik[0].d * lik[1].d / gcd(lik[0].d, lik[1].d);
            let (la, lb) = (lik[0].n * (den / lik[0].d), lik[1].n * (den / lik[1].d));
            let (na, nb) = (oa * la, ob * lb);
            let g2 = gcd(na, nb).max(1);
            out.push(format!(
                "{who}Odds for {} against {} were {oa} to {ob}. This has a chance of {} with {} and {} with {}, a ratio of {la} to {lb}, so the odds become {na} to {nb}{}. That makes {} {}.",
                f.hyps[0].name,
                f.hyps[1].name,
                lik[0].text(),
                f.hyps[0].name,
                lik[1].text(),
                f.hyps[1].name,
                if g2 > 1 { format!(", which is {} to {}", na / g2, nb / g2) } else { String::new() },
                f.hyps[0].name,
                new[0].text()
            ));
        } else {
            let lines: Vec<String> = (0..post.len()).map(|i| format!("{} {} times {} = {}", f.hyps[i].name, post[i].text(), lik[i].text(), prods[i].text())).collect();
            out.push(format!("{who}Update: {}. These add up to {}; dividing each by that: {}.", lines.join("; "), total.text(), (0..post.len()).map(|i| format!("{} {}", f.hyps[i].name, new[i].text())).collect::<Vec<_>>().join(", ")));
        }
        out.push(format!("{who}{}", belief_words(&f.hyps, &new, &post)));
        post = new;
        if found_now {
            break;
        }
    }
    let lead = (0..post.len()).max_by(|&a, &b| post[a].f().partial_cmp(&post[b].f()).unwrap()).unwrap();
    if post.iter().filter(|p| **p == post[lead]).count() == 1 {
        out.push(format!("Best guess now: {}, with a chance of {}.", f.hyps[lead].name, post[lead].text()));
    } else {
        out.push("Best guess now: none; the top chances are equal.".to_string());
    }
    if let (Some(k), true) = (f.predict, chance(r, 1, 2)) {
        let parts: Vec<String> = (0..post.len()).map(|i| format!("{} times {}", post[i].text(), f.hyps[i].lik[k].text())).collect();
        let p = (0..post.len()).fold(Frac::new(0, 1), |a, i| a.add(post[i].mul(f.hyps[i].lik[k])));
        out.push(format!("Chance that the next one is like this: \"{}\": {} = {}.", f.obs[k].trim_end_matches('.'), parts.join(" + "), p.text()));
    }
    if chance(r, 1, 3) {
        let agree = truth == lead;
        out.push(format!(
            "Afterwards the truth came out: it was {}.{}",
            f.hyps[truth].name,
            if agree { " The evidence had pointed the right way." } else { " The evidence had pointed elsewhere; that happens, and the chances were honest about it." }
        ));
    }
    out.join("\n")
}

pub fn doc(r: &mut SplitMix64) -> String {
    let f = match r.below(6) {
        0 => coin_frame(r),
        1 => dice_frame(r),
        2 => weather_frame(r),
        3 => sensor_frame(r),
        4 => keys_frame(r),
        _ => spinner_frame(r),
    };
    run(r, &f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractions_are_exact() {
        let a = Frac::new(2, 4);
        assert_eq!(a, Frac::new(1, 2));
        assert_eq!(a.mul(Frac::new(3, 4)), Frac::new(3, 8));
        assert_eq!(Frac::new(1, 4).add(Frac::new(3, 8)), Frac::new(5, 8));
        assert_eq!(Frac::new(1, 4).div(Frac::new(5, 8)), Frac::new(2, 5));
        assert_eq!(Frac::new(3, 8).text(), "3/8");
        assert_eq!(Frac::new(1, 3).one_minus(), Frac::new(2, 3));
    }

    #[test]
    fn a_tails_rules_out_the_two_headed_coin() {
        let h = [
            Hyp { name: "fair".into(), prior: Frac::new(1, 2), lik: vec![Frac::new(1, 2), Frac::new(1, 2)] },
            Hyp { name: "two".into(), prior: Frac::new(1, 2), lik: vec![Frac::new(1, 1), Frac::new(0, 1)] },
        ];
        let prods: Vec<Frac> = h.iter().map(|x| x.prior.mul(x.lik[1])).collect();
        let tot = prods[0].add(prods[1]);
        assert_eq!(prods[0].div(tot), Frac::new(1, 1));
        assert_eq!(prods[1].div(tot), Frac::new(0, 1));
    }

    #[test]
    fn docs_update_with_fractions_that_sum_to_one() {
        let mut r = SplitMix64::new(5);
        for _ in 0..300 {
            let d = doc(&mut r);
            assert!(d.contains("Turn 1.") || d.contains(": "), "{d}");
            assert!(d.contains("Best guess now"));
            // every "dividing each by that" list sums to exactly 1
            for l in d.lines().filter(|l| l.contains("dividing each by that: ")) {
                let tail = l.split("dividing each by that: ").nth(1).unwrap().trim_end_matches('.');
                let mut s = Frac::new(0, 1);
                for part in tail.split(", ") {
                    let v = part.rsplit(' ').next().unwrap();
                    let f = match v.split_once('/') {
                        Some((a, b)) => Frac::new(a.parse().unwrap(), b.parse().unwrap()),
                        None => Frac::new(v.parse().unwrap(), 1),
                    };
                    s = s.add(f);
                }
                assert_eq!(s, Frac::new(1, 1), "{l}");
            }
        }
    }
}
