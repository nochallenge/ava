// TESTING / DESIGN ONLY
//! reason_causal (C10, D3): cause and effect in small made-up worlds, every answer with a short why (I5).
//!
//! WORLD. Everyday or invented events (a farm, the weather, a kitchen, a class, a night garden, a town, a
//! marble run, a vegetable garden), linked by rules: "Whenever the gate opens, the geese run out", or an
//! event that needs two others ("the bread rises only when both ..."). Events are tied in a chain (2 to
//! 10 long), a fork (one cause, two effects), a merge (two causes, one effect) or a random web. A closing
//! line says that nothing happens without a rule, except what we are told.
//!
//! QUESTIONS (3 to 6 a world, each its own day): forward along long chains; no and can't-be-sure cases;
//! working back from an effect to a cause (one possible cause, two, one ruled out, one already explained);
//! seeing an effect of a common cause versus making it happen ourselves; stepping in ("we make ... happen
//! ourselves", "we make sure ... does not"); and would-have-been questions with and without a backup cause.
//! Every answer is CHECKED by an exact engine (all unknown starting events enumerated; for would-have-been,
//! the worlds that fit the facts, replayed with the change), and a question whose wording disagrees with
//! the engine is dropped.
//!
//! AWAY FROM THE TEST. The causal test uses devices that are on or off ("the red lamp", "the buzzer"), the
//! frames "When X is on, Y turns on", "Only X can turn on Y", "Q: Is X on?" and "If X had been off, would Y
//! have been on?", and the answer "cannot tell". None of these appear here (`reason_guard` checks it).

use crate::reason::{and_list, cap, chance, pick, sample_k};
use crate::rng::SplitMix64;

/// One event: subject, present (3rd person), base, past, past participle; plural subject.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ev {
    pub subj: &'static str,
    pub pres: &'static str,
    pub base: &'static str,
    pub past: &'static str,
    pub part: &'static str,
    pub plural: bool,
}

fn ev(s: &'static str) -> Ev {
    let f: Vec<&'static str> = s.split('|').collect();
    Ev { subj: f[0], pres: f[1], base: f[2], past: f[3], part: f[4], plural: f.len() > 5 }
}

/// (domain intro, events).
const DOMAINS: [(&str, [&str; 12]); 8] = [
    (
        "On a small farm",
        [
            "the rooster|crows|crow|crowed|crowed",
            "the dog|barks|bark|barked|barked",
            "the cows|wake up|wake up|woke up|woken up|p",
            "the farmer|gets up|get up|got up|got up",
            "the gate|opens|open|opened|opened",
            "the geese|run out|run out|ran out|run out|p",
            "the hens|lay eggs|lay eggs|laid eggs|laid eggs|p",
            "the cat|hides|hide|hid|hidden",
            "the milk cart|leaves|leave|left|left",
            "the sheep|go up the hill|go up the hill|went up the hill|gone up the hill|p",
            "the barn door|swings shut|swing shut|swung shut|swung shut",
            "the goat|bleats|bleat|bleated|bleated",
        ],
    ),
    (
        "In a valley with strange weather",
        [
            "the wind|rises|rise|rose|risen",
            "the clouds|gather|gather|gathered|gathered|p",
            "it|rains|rain|rained|rained",
            "the river|floods|flood|flooded|flooded",
            "the frogs|sing|sing|sang|sung|p",
            "the snow|melts|melt|melted|melted",
            "the path|turns muddy|turn muddy|turned muddy|turned muddy",
            "the birds|fly south|fly south|flew south|flown south|p",
            "the pond|freezes|freeze|froze|frozen",
            "the leaves|fall|fall|fell|fallen|p",
            "the mushrooms|come up|come up|came up|come up|p",
            "the fog|rolls in|roll in|rolled in|rolled in",
        ],
    ),
    (
        "In a busy kitchen",
        [
            "the kettle|boils|boil|boiled|boiled",
            "the toast|pops up|pop up|popped up|popped up",
            "the oven|gets hot|get hot|got hot|got hot",
            "the bread|rises|rise|rose|risen",
            "the soup|bubbles|bubble|bubbled|bubbled",
            "the window|steams up|steam up|steamed up|steamed up",
            "the smoke|drifts out|drift out|drifted out|drifted out",
            "the butter|melts|melt|melted|melted",
            "the puppy|begs for food|beg for food|begged for food|begged for food",
            "the cook|opens the window|open the window|opened the window|opened the window",
            "the tea|gets made|get made|got made|got made",
            "the timer|rings|ring|rang|rung",
        ],
    ),
    (
        "In a noisy classroom",
        [
            "Mira|laughs|laugh|laughed|laughed",
            "Tom|smiles|smile|smiled|smiled",
            "the teacher|claps|clap|clapped|clapped",
            "the class|goes quiet|go quiet|went quiet|gone quiet",
            "Ravi|sings|sing|sang|sung",
            "Lena|starts dancing|start dancing|started dancing|started dancing",
            "the baby brother|wakes up|wake up|woke up|woken up",
            "Grandpa|tells a story|tell a story|told a story|told a story",
            "Omar|yawns|yawn|yawned|yawned",
            "the twins|giggle|giggle|giggled|giggled|p",
            "Kai|waves|wave|waved|waved",
            "Zoe|claps along|clap along|clapped along|clapped along",
        ],
    ),
    (
        "In a night garden that follows its own laws",
        [
            "the glowworms|shine|shine|shone|shone|p",
            "the moonflowers|open|open|opened|opened|p",
            "the tree frogs|call|call|called|called|p",
            "the bats|come out|come out|came out|come out|p",
            "the moths|gather|gather|gathered|gathered|p",
            "the owl|hunts|hunt|hunted|hunted",
            "the beetles|hide|hide|hid|hidden|p",
            "the vines|curl up|curl up|curled up|curled up|p",
            "the spring|bubbles over|bubble over|bubbled over|bubbled over",
            "the fireflies|blink|blink|blinked|blinked|p",
            "the lilies|close|close|closed|closed|p",
            "the crickets|chirp|chirp|chirped|chirped|p",
        ],
    ),
    (
        "In a river town",
        [
            "the mill wheel|turns|turn|turned|turned",
            "the flour|gets ground|get ground|got ground|got ground",
            "the baker|bakes bread|bake bread|baked bread|baked bread",
            "the market|opens|open|opened|opened",
            "the ferry|sails|sail|sailed|sailed",
            "the bridge|lifts|lift|lifted|lifted",
            "the boats|pass|pass|passed|passed|p",
            "the clock tower|strikes noon|strike noon|struck noon|struck noon",
            "the shops|close|close|closed|closed|p",
            "the square|fills with people|fill with people|filled with people|filled with people",
            "the fountain|flows|flow|flowed|flowed",
            "the train|leaves|leave|left|left",
        ],
    ),
    (
        "In a marble run on the table",
        [
            "the marble|rolls|roll|rolled|rolled",
            "the cup|tips over|tip over|tipped over|tipped over",
            "the first domino|falls|fall|fell|fallen",
            "the last domino|falls|fall|fell|fallen",
            "the flag|goes up|go up|went up|gone up",
            "the lever|drops|drop|dropped|dropped",
            "the seesaw|tips|tip|tipped|tipped",
            "the paddle|spins|spin|spun|spun",
            "the string|pulls tight|pull tight|pulled tight|pulled tight",
            "the trapdoor|opens|open|opened|opened",
            "the weight|swings|swing|swung|swung",
            "the toy car|moves|move|moved|moved",
        ],
    ),
    (
        "In a vegetable garden",
        [
            "the sprinkler|runs|run|ran|run",
            "the soil|gets wet|get wet|got wet|got wet",
            "the seeds|sprout|sprout|sprouted|sprouted|p",
            "the tomatoes|ripen|ripen|ripened|ripened|p",
            "the snails|come out|come out|came out|come out|p",
            "the roses|bloom|bloom|bloomed|bloomed|p",
            "the bees|visit|visit|visited|visited|p",
            "the gardener|picks fruit|pick fruit|picked fruit|picked fruit",
            "the weeds|grow|grow|grew|grown|p",
            "the hedge|gets trimmed|get trimmed|got trimmed|got trimmed",
            "the robins|dig for worms|dig for worms|dug for worms|dug for worms|p",
            "the compost|warms up|warm up|warmed up|warmed up",
        ],
    ),
];

impl Ev {
    pub fn pres(&self) -> String {
        format!("{} {}", self.subj, self.pres)
    }
    pub fn past(&self) -> String {
        format!("{} {}", self.subj, self.past)
    }
    pub fn not_past(&self) -> String {
        format!("{} did not {}", self.subj, self.base)
    }
    pub fn did_q(&self) -> String {
        format!("did {} {}", self.subj, self.base)
    }
    pub fn make(&self) -> String {
        format!("make {} {}", self.subj, self.base)
    }
    pub fn make_not(&self) -> String {
        format!("make sure {} {} not {}", self.subj, if self.plural { "do" } else { "does" }, self.base)
    }
    pub fn had_not(&self) -> String {
        format!("had {} not {}", self.subj, self.part)
    }
    pub fn would(&self) -> String {
        format!("would {} have {}", self.subj, self.part)
    }
    pub fn would_still(&self) -> String {
        format!("would {} still have {}", self.subj, self.part)
    }
    pub fn for_to(&self) -> String {
        format!("for {} to {}", self.subj, self.base)
    }
}

/// A rule world: nodes in causal order (parents come first). A node with no parents is a starting event.
#[derive(Clone, Debug)]
pub struct World {
    pub ev: Vec<Ev>,
    pub parents: Vec<Vec<usize>>,
    /// Needs all its parents (else any one parent is enough).
    pub and: Vec<bool>,
}

/// Yes, no, or not enough to say.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ans {
    Yes,
    No,
    Unsure,
}

impl World {
    pub fn n(&self) -> usize {
        self.ev.len()
    }

    pub fn roots(&self) -> Vec<usize> {
        (0..self.n()).filter(|&i| self.parents[i].is_empty()).collect()
    }

    /// Every event's value, given the starting events and interventions (node, value).
    pub fn eval(&self, root: &[bool], dos: &[(usize, bool)]) -> Vec<bool> {
        let mut v = vec![false; self.n()];
        for i in 0..self.n() {
            v[i] = if let Some(&(_, x)) = dos.iter().find(|(d, _)| *d == i) {
                x
            } else if self.parents[i].is_empty() {
                root[i]
            } else if self.and[i] {
                self.parents[i].iter().all(|&p| v[p])
            } else {
                self.parents[i].iter().any(|&p| v[p])
            };
        }
        v
    }

    /// Every assignment of the starting events that fits the facts (node, value) seen in the world
    /// shaped by `dos`.
    fn worlds(&self, facts: &[(usize, bool)], dos: &[(usize, bool)]) -> Vec<Vec<bool>> {
        let roots = self.roots();
        let mut out = Vec::new();
        for mask in 0..(1u32 << roots.len()) {
            let mut root = vec![false; self.n()];
            for (k, &rt) in roots.iter().enumerate() {
                root[rt] = mask >> k & 1 == 1;
            }
            let v = self.eval(&root, dos);
            if facts.iter().all(|&(i, x)| v[i] == x) {
                out.push(root);
            }
        }
        out
    }

    fn collect(vals: impl Iterator<Item = bool>) -> Option<Ans> {
        let (mut t, mut f) = (false, false);
        for x in vals {
            if x {
                t = true
            } else {
                f = true
            }
        }
        match (t, f) {
            (true, false) => Some(Ans::Yes),
            (false, true) => Some(Ans::No),
            (true, true) => Some(Ans::Unsure),
            _ => None,
        }
    }

    /// Does `q` happen, given facts seen (in the world with `dos` applied)? None: the facts cannot all hold.
    pub fn ask(&self, facts: &[(usize, bool)], dos: &[(usize, bool)], q: usize) -> Option<Ans> {
        let ws = self.worlds(facts, dos);
        World::collect(ws.iter().map(|root| self.eval(root, dos)[q]))
    }

    /// Would `q` have happened if `change` had been forced, in every world that fits what actually happened?
    pub fn would(&self, facts: &[(usize, bool)], change: &[(usize, bool)], q: usize) -> Option<Ans> {
        let ws = self.worlds(facts, &[]);
        World::collect(ws.iter().map(|root| self.eval(root, change)[q]))
    }

    pub fn children(&self, i: usize) -> Vec<usize> {
        (0..self.n()).filter(|&j| self.parents[j].contains(&i)).collect()
    }

    /// The starting events with a path to `x` (x itself when it is one).
    pub fn root_ancestors(&self, x: usize) -> Vec<usize> {
        let mut seen = vec![false; self.n()];
        let mut stack = vec![x];
        let mut out = Vec::new();
        while let Some(i) = stack.pop() {
            if seen[i] {
                continue;
            }
            seen[i] = true;
            if self.parents[i].is_empty() {
                out.push(i);
            }
            stack.extend(self.parents[i].iter().copied());
        }
        out.sort_unstable();
        out
    }

    /// A path from `a` to `b` through "any one parent is enough" links only (b may need all; then None).
    pub fn or_path(&self, a: usize, b: usize) -> Option<Vec<usize>> {
        let mut prev = vec![usize::MAX; self.n()];
        let mut q = std::collections::VecDeque::from([a]);
        prev[a] = a;
        while let Some(i) = q.pop_front() {
            if i == b {
                let mut p = vec![b];
                let mut c = b;
                while c != a {
                    c = prev[c];
                    p.push(c);
                }
                p.reverse();
                return Some(p);
            }
            for c in self.children(i) {
                if prev[c] == usize::MAX && !self.and[c] {
                    prev[c] = i;
                    q.push_back(c);
                }
            }
        }
        None
    }
}

fn make_world(r: &mut SplitMix64, evs: &[Ev]) -> World {
    let shape = r.below(10);
    let mut parents: Vec<Vec<usize>> = Vec::new();
    let mut and: Vec<bool> = Vec::new();
    match shape {
        // a chain of 3 to 10 events
        0..=2 => {
            let n = 3 + r.below(8) as usize;
            for i in 0..n {
                parents.push(if i == 0 { vec![] } else { vec![i - 1] });
                and.push(false);
            }
        }
        // a fork: one start, two branches
        3..=4 => {
            let (a, b) = (1 + r.below(3) as usize, 1 + r.below(3) as usize);
            parents.push(vec![]);
            for k in 0..a {
                parents.push(vec![if k == 0 { 0 } else { parents.len() - 1 }]);
            }
            for k in 0..b {
                parents.push(vec![if k == 0 { 0 } else { parents.len() - 1 }]);
            }
            and = vec![false; parents.len()];
        }
        // a merge: two chains into one event (any one enough, or both needed)
        5..=6 => {
            let (a, b) = (1 + r.below(3) as usize, 1 + r.below(3) as usize);
            for k in 0..a {
                parents.push(if k == 0 { vec![] } else { vec![k - 1] });
            }
            for k in 0..b {
                parents.push(if k == 0 { vec![] } else { vec![a + k - 1] });
            }
            parents.push(vec![a - 1, a + b - 1]);
            and = vec![false; parents.len()];
            *and.last_mut().unwrap() = chance(r, 1, 4);
            if chance(r, 1, 2) {
                let last = parents.len() - 1;
                parents.push(vec![last]);
                and.push(false);
            }
        }
        // a random web
        _ => {
            let n = 5 + r.below(5) as usize;
            let nroots = 1 + r.below(3) as usize;
            for i in 0..n {
                if i < nroots {
                    parents.push(vec![]);
                    and.push(false);
                    continue;
                }
                let two = i >= 2 && chance(r, 1, 3);
                let mut ps = vec![r.below(i as u64) as usize];
                if two {
                    let q = r.below(i as u64) as usize;
                    if q != ps[0] {
                        ps.push(q);
                    }
                }
                ps.sort_unstable();
                and.push(ps.len() == 2 && chance(r, 1, 5));
                parents.push(ps);
            }
        }
    }
    let pick_ev = sample_k(r, evs.len(), parents.len());
    World { ev: pick_ev.iter().map(|&k| evs[k]).collect(), parents, and }
}

fn rule_text(r: &mut SplitMix64, w: &World, i: usize) -> String {
    let x = &w.ev[i];
    let ps: Vec<&Ev> = w.parents[i].iter().map(|&p| &w.ev[p]).collect();
    if w.and[i] {
        let both = format!("{} and {}", ps[0].pres(), ps[1].pres());
        return match r.below(3) {
            0 => format!("{} only when both {both}.", cap(&x.pres())),
            1 => format!("If {both}, then {}; one of them alone is not enough.", x.pres()),
            _ => format!("It takes both to make {} {}: {both}.", x.subj, x.base),
        };
    }
    let mut lines = Vec::new();
    for p in ps {
        lines.push(match r.below(4) {
            0 => format!("Whenever {}, {}.", p.pres(), x.pres()),
            1 => format!("If {}, then {}.", p.pres(), x.pres()),
            2 => format!("{} whenever {}.", cap(&x.pres()), p.pres()),
            _ => format!("As soon as {}, {}.", p.pres(), x.pres()),
        });
    }
    lines.join(" ")
}

fn past_chain(w: &World, path: &[usize]) -> String {
    let parts: Vec<String> = path.iter().map(|&i| w.ev[i].past()).collect();
    cap(&parts.join(", so "))
}

fn unsure(r: &mut SplitMix64) -> &'static str {
    *pick(r, &["Not enough to say.", "We can't be sure.", "There is no way to know from this."])
}

/// "the dog barked and the cat hid" (or the "did not" forms).
fn pasts(w: &World, v: &[usize], happened: bool) -> String {
    and_list(&v.iter().map(|&i| if happened { w.ev[i].past() } else { w.ev[i].not_past() }).collect::<Vec<_>>())
}

/// One question: (what we are told, the question, the answer, why).
struct Q {
    told: String,
    ask: String,
    ans: Ans,
    why: String,
}

/// Facts given as sentences: "the gate opened. the cat did not hide."
fn facts_text(w: &World, facts: &[(usize, bool)]) -> String {
    facts.iter().map(|&(i, x)| cap(&if x { w.ev[i].past() } else { w.ev[i].not_past() }) + ".").collect::<Vec<_>>().join(" ")
}

fn q_forward_yes(r: &mut SplitMix64, w: &World) -> Option<Q> {
    let x = r.below(w.n() as u64) as usize;
    let ra = w.root_ancestors(x);
    let rt = *pick(r, &ra);
    if rt == x {
        return None;
    }
    let path = w.or_path(rt, x)?;
    let facts = vec![(rt, true)];
    let ans = w.ask(&facts, &[], x)?;
    Some(Q { told: facts_text(w, &facts), ask: format!("{}?", cap(&w.ev[x].did_q())), ans, why: format!("{}.", past_chain(w, &path)) })
}

fn q_forward_no(r: &mut SplitMix64, w: &World) -> Option<Q> {
    let x = r.below(w.n() as u64) as usize;
    let ra = w.root_ancestors(x);
    if ra.contains(&x) {
        return None;
    }
    let facts: Vec<(usize, bool)> = ra.iter().map(|&i| (i, false)).collect();
    let ans = w.ask(&facts, &[], x)?;
    let why = if ra.len() == 1 {
        format!("{}, and every way {} starts there.", cap(&w.ev[ra[0]].not_past()), w.ev[x].for_to())
    } else {
        format!("{}, and every way {} starts with one of those.", cap(&pasts(w, &ra, false)), w.ev[x].for_to())
    };
    Some(Q { told: facts_text(w, &facts), ask: format!("{}?", cap(&w.ev[x].did_q())), ans, why })
}

fn q_forward_unsure(r: &mut SplitMix64, w: &World) -> Option<Q> {
    let x = r.below(w.n() as u64) as usize;
    let ra = w.root_ancestors(x);
    if ra.contains(&x) {
        return None;
    }
    let open = *pick(r, &ra);
    let facts: Vec<(usize, bool)> = ra.iter().filter(|&&i| i != open).map(|&i| (i, false)).collect();
    let ans = w.ask(&facts, &[], x)?;
    let told = if facts.is_empty() { format!("We are not told whether {} {}.", w.ev[open].subj, w.ev[open].past) } else { facts_text(w, &facts) };
    let why = format!("It depends on whether {} {}, and nothing tells us that.", w.ev[open].subj, w.ev[open].past);
    Some(Q { told, ask: format!("{}?", cap(&w.ev[x].did_q())), ans, why })
}

fn q_back_one(r: &mut SplitMix64, w: &World) -> Option<Q> {
    let e = r.below(w.n() as u64) as usize;
    let ra = w.root_ancestors(e);
    if ra.len() != 1 || ra[0] == e {
        return None;
    }
    let c = ra[0];
    let facts = vec![(e, true)];
    let ans = w.ask(&facts, &[], c)?;
    let why = format!("The only way {} is that {} first, so {} must have {}.", w.ev[e].for_to(), w.ev[c].pres(), w.ev[c].subj, w.ev[c].part);
    Some(Q { told: facts_text(w, &facts), ask: format!("{}?", cap(&w.ev[c].did_q())), ans, why })
}

fn q_back_two(r: &mut SplitMix64, w: &World) -> Option<Q> {
    let e = r.below(w.n() as u64) as usize;
    let ra = w.root_ancestors(e);
    if ra.len() < 2 || w.and[e] {
        return None;
    }
    let c = *pick(r, &ra);
    let other: Vec<usize> = ra.iter().copied().filter(|&i| i != c).collect();
    let mode = r.below(3);
    let mut facts = vec![(e, true)];
    let why;
    if mode == 0 {
        let alts: Vec<String> = other.iter().map(|&o| w.ev[o].past()).collect();
        why = format!("That {} could be because {} or because {}; seeing it does not say which.", w.ev[e].past(), w.ev[c].past(), alts.join(" or because "));
    } else if mode == 1 {
        // the other causes are ruled out
        for &o in &other {
            facts.push((o, false));
        }
        why = format!("Something had to lead to it, and {}, so {} must have {}.", pasts(w, &other, false), w.ev[c].subj, w.ev[c].part);
    } else {
        // another cause is already known: it explains the effect
        facts.push((other[0], true));
        why = format!("That {} is already enough to explain it, so it says nothing about whether {}.", w.ev[other[0]].past(), w.ev[c].past());
    }
    let ans = w.ask(&facts, &[], c)?;
    Some(Q { told: facts_text(w, &facts), ask: format!("{}?", cap(&w.ev[c].did_q())), ans, why })
}

/// A common cause: seeing one of its effects, versus making that effect happen (or not) ourselves.
fn q_see_do(r: &mut SplitMix64, w: &World) -> Option<Q> {
    let a = r.below(w.n() as u64) as usize;
    let kids = w.children(a);
    if kids.len() < 2 {
        // look further down: two different descendants through different children
        return None;
    }
    let (b, c) = (kids[0], kids[1]);
    if w.parents[b] != [a] || w.parents[c] != [a] {
        return None;
    }
    let ra = w.root_ancestors(a);
    let (ea, eb, ec) = (&w.ev[a], &w.ev[b], &w.ev[c]);
    match r.below(4) {
        0 => {
            let facts = vec![(b, true)];
            let ans = w.ask(&facts, &[], c)?;
            Some(Q {
                told: facts_text(w, &facts),
                ask: format!("{}?", cap(&ec.did_q())),
                ans,
                why: format!("{} only because {}, and that also makes {} {}.", cap(&eb.past()), ea.past(), ec.subj, ec.base),
            })
        }
        1 => {
            let facts = vec![(b, false)];
            let ans = w.ask(&facts, &[], c)?;
            Some(Q {
                told: facts_text(w, &facts),
                ask: format!("{}?", cap(&ec.did_q())),
                ans,
                why: format!(
                    "Had {} {}, {} would have {} too. That did not happen, so {}, and without that there is no way {}.",
                    ea.subj,
                    ea.part,
                    eb.subj,
                    eb.part,
                    ea.not_past(),
                    ec.for_to()
                ),
            })
        }
        2 => {
            if ra.len() != 1 {
                return None;
            }
            let dos = vec![(b, true)];
            let ans = w.ask(&[], &dos, c)?;
            Some(Q {
                told: format!("This time nobody knows what started things, but we {} ourselves.", eb.make()),
                ask: format!("{}?", cap(&ec.did_q())),
                ans,
                why: format!(
                    "We made {} {} by hand, so it tells us nothing about {}. Whether {} still depends on whether {}.",
                    eb.subj,
                    eb.base,
                    ea.subj,
                    ec.pres(),
                    ea.pres()
                ),
            })
        }
        _ => {
            let facts = vec![(a, true)];
            let dos = vec![(b, false)];
            let ans = w.ask(&facts, &dos, c)?;
            Some(Q {
                told: format!("{} But we step in and {}.", facts_text(w, &facts), eb.make_not()),
                ask: format!("{}?", cap(&ec.did_q())),
                ans,
                why: format!("We only blocked one effect. {} anyway, and that still makes {} {}.", cap(&ea.past()), ec.subj, ec.base),
            })
        }
    }
}

/// Stepping in: making an event happen ourselves. Its effects follow; its causes are not touched.
fn q_do(r: &mut SplitMix64, w: &World) -> Option<Q> {
    let x = r.below(w.n() as u64) as usize;
    if w.parents[x].is_empty() {
        return None;
    }
    let dos = vec![(x, true)];
    if chance(r, 1, 2) {
        let kids: Vec<usize> = w.children(x).into_iter().filter(|&c| !w.and[c]).collect();
        let y = *kids.first()?;
        let ans = w.ask(&[], &dos, y)?;
        Some(Q {
            told: format!("Nobody knows how the day began, but we {} ourselves.", w.ev[x].make()),
            ask: format!("{}?", cap(&w.ev[y].did_q())),
            ans,
            why: format!("{} {}, and that is enough to make {} {}.", cap(w.ev[x].subj), w.ev[x].past, w.ev[y].subj, w.ev[y].base),
        })
    } else {
        let p = *pick(r, &w.parents[x]);
        let ans = w.ask(&[], &dos, p)?;
        Some(Q {
            told: format!("Nobody knows how the day began, but we {} ourselves.", w.ev[x].make()),
            ask: format!("{}?", cap(&w.ev[p].did_q())),
            ans,
            why: format!("Making {} {} by hand does not reach back: whether {} is still unknown.", w.ev[x].subj, w.ev[x].base, w.ev[p].past()),
        })
    }
}

/// Would-have-been: no backup cause, a backup cause, or an earlier event left alone.
fn q_would(r: &mut SplitMix64, w: &World) -> Option<Q> {
    let y = r.below(w.n() as u64) as usize;
    let ra = w.root_ancestors(y);
    if ra.contains(&y) {
        return None;
    }
    match r.below(3) {
        0 => {
            // every start true; take one away
            let a = *pick(r, &ra);
            let facts: Vec<(usize, bool)> = ra.iter().map(|&i| (i, true)).collect();
            let ans = w.would(&facts, &[(a, false)], y)?;
            let others: Vec<usize> = ra.iter().copied().filter(|&i| i != a).collect();
            let why = match ans {
                Ans::Yes => format!("{} as well, and that is still enough {}.", cap(&pasts(w, &others, true)), w.ev[y].for_to()),
                Ans::No if others.is_empty() => format!("It all started because {}; without that, nothing else would have made it happen.", w.ev[a].past()),
                Ans::No => format!("It also needed that {}; the rest alone is not enough.", w.ev[a].past()),
                Ans::Unsure => return None,
            };
            Some(Q { told: facts_text(w, &facts), ask: format!("{}, {}?", cap(&w.ev[a].had_not()), if ans == Ans::Yes { w.ev[y].would_still() } else { w.ev[y].would() }), ans, why })
        }
        1 => {
            // a middle event taken away
            if w.parents[y].len() != 1 || w.parents[w.parents[y][0]].is_empty() {
                return None;
            }
            let m = w.parents[y][0];
            let facts: Vec<(usize, bool)> = ra.iter().map(|&i| (i, true)).collect();
            let ans = w.would(&facts, &[(m, false)], y)?;
            if ans != Ans::No {
                return None;
            }
            Some(Q {
                told: facts_text(w, &facts),
                ask: format!("{}, {}?", cap(&w.ev[m].had_not()), w.ev[y].would()),
                ans,
                why: format!("Its only direct cause is that {}, so without that it would not have happened.", w.ev[m].past()),
            })
        }
        _ => {
            // an earlier event stays as it was when a later one is changed
            let a = *pick(r, &ra);
            let facts: Vec<(usize, bool)> = ra.iter().map(|&i| (i, true)).collect();
            let ans = w.would(&facts, &[(y, false)], a)?;
            Some(Q {
                told: facts_text(w, &facts),
                ask: format!("{}, {}?", cap(&w.ev[y].had_not()), w.ev[a].would_still()),
                ans,
                why: format!("{} first, and changing a later event cannot undo that.", cap(&w.ev[a].past())),
            })
        }
    }
}

/// An event that needs two others.
fn q_and(r: &mut SplitMix64, w: &World) -> Option<Q> {
    let x = (0..w.n()).find(|&i| w.and[i])?;
    let (p, q) = (w.parents[x][0], w.parents[x][1]);
    let both = chance(r, 1, 2);
    let mut facts: Vec<(usize, bool)> = Vec::new();
    for &rt in &w.root_ancestors(p) {
        facts.push((rt, true));
    }
    for &rt in &w.root_ancestors(q) {
        if !facts.iter().any(|f| f.0 == rt) {
            facts.push((rt, both));
        }
    }
    let ans = w.ask(&facts, &[], x)?;
    let why = format!(
        "It needs two things: {} and {}. {}",
        w.ev[p].pres(),
        w.ev[q].pres(),
        if ans == Ans::Yes { "Both happened." } else { "They did not both happen." }
    );
    Some(Q { told: facts_text(w, &facts), ask: format!("{}?", cap(&w.ev[x].did_q())), ans, why })
}

/// The answer in words as the doc's question wording expects; the engine's answer must match the template's
/// intent, which each template checks by building its why from the engine's answer or by construction.
fn ans_text(r: &mut SplitMix64, a: Ans) -> String {
    match a {
        Ans::Yes => "Yes.".to_string(),
        Ans::No => "No.".to_string(),
        Ans::Unsure => unsure(r).to_string(),
    }
}

/// Every template's why is written for one answer; this says which (None: written from the answer).
fn expected(kind: usize, q: &Q) -> Option<Ans> {
    match kind {
        0 => Some(Ans::Yes),
        1 => Some(Ans::No),
        2 => Some(Ans::Unsure),
        3 => Some(Ans::Yes),
        4 => {
            if q.why.contains("does not say which") || q.why.contains("says nothing about") {
                Some(Ans::Unsure)
            } else {
                Some(Ans::Yes)
            }
        }
        5 => {
            if q.why.contains("tells us nothing about") {
                Some(Ans::Unsure)
            } else if q.why.contains("did not") && q.why.starts_with("Had") {
                Some(Ans::No)
            } else {
                Some(Ans::Yes)
            }
        }
        6 => {
            if q.why.contains("still unknown") {
                Some(Ans::Unsure)
            } else {
                Some(Ans::Yes)
            }
        }
        _ => None,
    }
}

pub fn doc(r: &mut SplitMix64) -> String {
    loop {
        let (intro, list) = pick(r, &DOMAINS);
        let evs: Vec<Ev> = list.iter().map(|s| ev(s)).collect();
        let w = make_world(r, &evs);
        let mut out = Vec::new();
        out.push(format!(
            "{} {}",
            pick(r, &["A made-up world.", "Cause and effect.", "A small world with its own rules.", "Rules of a pretend place."]),
            format!("{intro}, these are the rules:")
        ));
        let mut order: Vec<usize> = (0..w.n()).filter(|&i| !w.parents[i].is_empty()).collect();
        if chance(r, 1, 2) {
            r.shuffle(&mut order);
        }
        let numbered = chance(r, 1, 2);
        for (k, &i) in order.iter().enumerate() {
            let t = rule_text(r, &w, i);
            out.push(if numbered { format!("{}. {t}", k + 1) } else { t });
        }
        out.push(
            pick(
                r,
                &[
                    "Those are all the rules. Nothing happens here unless a rule makes it happen, apart from what we are told.",
                    "There are no other causes: things happen only through these rules, or because we are told they did.",
                    "Nothing else makes anything happen in this world.",
                ],
            )
            .to_string(),
        );
        let want = 3 + r.below(4) as usize;
        let mut qs: Vec<Q> = Vec::new();
        let mut tries = 0;
        while qs.len() < want && tries < 60 {
            tries += 1;
            let kind = r.below(9) as usize;
            let q = match kind {
                0 => q_forward_yes(r, &w),
                1 => q_forward_no(r, &w),
                2 => q_forward_unsure(r, &w),
                3 => q_back_one(r, &w),
                4 => q_back_two(r, &w),
                5 => q_see_do(r, &w),
                6 => q_do(r, &w),
                7 => q_would(r, &w),
                _ => q_and(r, &w),
            };
            let Some(q) = q else { continue };
            // the template's wording must agree with the engine
            if let Some(e) = expected(kind, &q) {
                if e != q.ans {
                    continue;
                }
            }
            if qs.iter().any(|o| o.ask == q.ask && o.told == q.told) {
                continue;
            }
            qs.push(q);
        }
        if qs.len() < 2 {
            continue;
        }
        for q in qs {
            let a = ans_text(r, q.ans);
            let lead = pick(r, &["One day:", "Another day:", "Suppose this:", "New day:"]);
            out.push(format!("{lead} {}\nQuestion: {}\nAnswer: {a} {}", q.told, q.ask, q.why));
        }
        return out.join("\n");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(n: usize) -> World {
        let evs: Vec<Ev> = DOMAINS[0].1.iter().map(|s| ev(s)).collect();
        World { ev: evs[..n].to_vec(), parents: (0..n).map(|i| if i == 0 { vec![] } else { vec![i - 1] }).collect(), and: vec![false; n] }
    }

    #[test]
    fn engine_on_a_chain() {
        let w = chain(4);
        assert_eq!(w.ask(&[(0, true)], &[], 3), Some(Ans::Yes));
        assert_eq!(w.ask(&[(0, false)], &[], 3), Some(Ans::No));
        assert_eq!(w.ask(&[], &[], 3), Some(Ans::Unsure));
        // seeing the end tells us the start; making the middle happen does not
        assert_eq!(w.ask(&[(3, true)], &[], 0), Some(Ans::Yes));
        assert_eq!(w.ask(&[], &[(2, true)], 0), Some(Ans::Unsure));
        assert_eq!(w.ask(&[], &[(2, true)], 3), Some(Ans::Yes));
        // would-have-been
        assert_eq!(w.would(&[(0, true)], &[(1, false)], 3), Some(Ans::No));
        assert_eq!(w.would(&[(0, true)], &[(2, false)], 0), Some(Ans::Yes));
        assert_eq!(w.ask(&[(0, false), (3, true)], &[], 1), None, "facts that cannot all hold");
    }

    #[test]
    fn engine_two_causes_and_and() {
        let evs: Vec<Ev> = DOMAINS[1].1.iter().map(|s| ev(s)).collect();
        let mut w = World { ev: evs[..3].to_vec(), parents: vec![vec![], vec![], vec![0, 1]], and: vec![false, false, false] };
        assert_eq!(w.ask(&[(2, true)], &[], 0), Some(Ans::Unsure));
        assert_eq!(w.ask(&[(2, true), (1, false)], &[], 0), Some(Ans::Yes));
        assert_eq!(w.ask(&[(2, true), (1, true)], &[], 0), Some(Ans::Unsure), "explained away");
        assert_eq!(w.would(&[(0, true), (1, true)], &[(0, false)], 2), Some(Ans::Yes), "backup cause");
        w.and[2] = true;
        assert_eq!(w.ask(&[(2, true)], &[], 0), Some(Ans::Yes));
        assert_eq!(w.would(&[(0, true), (1, true)], &[(0, false)], 2), Some(Ans::No));
    }

    #[test]
    fn grammar() {
        let e = ev("the cows|wake up|wake up|woke up|woken up|p");
        assert_eq!(e.make_not(), "make sure the cows do not wake up");
        assert_eq!(e.had_not(), "had the cows not woken up");
        let r = ev("it|rains|rain|rained|rained");
        assert_eq!(r.did_q(), "did it rain");
        assert_eq!(r.make(), "make it rain");
    }

    #[test]
    fn docs_have_questions_with_reasons() {
        let mut r = SplitMix64::new(2);
        let mut kinds = [0usize; 3];
        for _ in 0..300 {
            let d = doc(&mut r);
            assert!(d.matches("Question:").count() >= 2, "{d}");
            for l in d.lines().filter(|l| l.starts_with("Answer:")) {
                if l.starts_with("Answer: Yes.") {
                    kinds[0] += 1;
                } else if l.starts_with("Answer: No.") {
                    kinds[1] += 1;
                } else {
                    kinds[2] += 1;
                }
                assert!(l.len() > 20, "every answer has a why: {l}");
            }
        }
        assert!(kinds.iter().all(|&k| k > 100), "{kinds:?}");
    }
}
