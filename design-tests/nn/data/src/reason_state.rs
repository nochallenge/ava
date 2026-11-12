// TESTING / DESIGN ONLY
//! reason_state (C5): keeping track of a changing state over 2 to 64 steps, with no running totals.
//!
//! TASKS: a walker on a ring of named stops; a robot on a lettered grid moved by compass steps; a pile of
//! plates (put on, take off); people trading the things they hold; a panel of switches (flip, set, flip
//! all); a queue (join, served, let someone ahead); people getting on and off a bus. Half the documents
//! write the state after every step; the other half give only the steps and then the end state. Every
//! document ends with a question about the end state and its answer.
//!
//! AWAY FROM THE TEST. The horizon test (C5) is a running total ("Start with 5. Add 2. Take away 1. What
//! number do you have now?") and the A4 state test hides a coin under cups ("Swap the red cup and the blue
//! cup"). Neither the frames, nor cups, nor a running total appear here; things change hands by "trading",
//! never by "swap"; switches are written "on" and "off", never as 0 and 1.

use crate::reason::{and_list, cap, chance, num_word, pick, range, sample_k};
use crate::rng::SplitMix64;

const LENGTHS: [usize; 12] = [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 64];
const PEOPLE: [&str; 12] = ["Ana", "Ben", "Cleo", "Dev", "Esme", "Femi", "Gus", "Hana", "Iris", "Jude", "Kofi", "Lina"];

fn length(r: &mut SplitMix64) -> usize {
    // more short and middle lengths than very long ones
    let i = r.below(LENGTHS.len() as u64 + 3) as usize;
    LENGTHS[i.min(LENGTHS.len() - 1)]
}

struct Out {
    lines: Vec<String>,
    each: bool,
}

impl Out {
    fn step(&mut self, k: usize, what: String, state: String) {
        if self.each {
            self.lines.push(format!("Step {k}: {what} {state}"));
        } else {
            self.lines.push(format!("Step {k}: {what}"));
        }
    }
}

fn ring(r: &mut SplitMix64, o: &mut Out, steps: usize) -> (String, String) {
    const STOPS: [&str; 12] = ["the oak", "the well", "the mill", "the pond", "the gate", "the bridge", "the barn", "the hill", "the cave", "the tower", "the market", "the dock"];
    let n = range(r, 5, 12) as usize;
    let stops: Vec<&str> = sample_k(r, STOPS.len(), n).into_iter().map(|i| STOPS[i]).collect();
    let mut at = r.below(n as u64) as usize;
    o.lines.push(format!("{} stops stand in a ring. Going clockwise they are: {}; after {} comes {} again.", cap(&num_word(n as u64)), stops.join(", "), stops[n - 1], stops[0]));
    o.lines.push(format!("A walker starts at {}.", stops[at]));
    for k in 1..=steps {
        let d = range(r, 1, (n - 1).min(4) as i64) as usize;
        let cw = chance(r, 1, 2);
        at = if cw { (at + d) % n } else { (at + n - d) % n };
        let what = format!("{} {} {}.", cap(&num_word(d as u64)), if d == 1 { "stop" } else { "stops" }, if cw { "clockwise" } else { "back, anticlockwise" });
        o.step(k, what, format!("Now at {}.", stops[at]));
    }
    ("Where is the walker now?".into(), format!("At {}.", stops[at]))
}

fn grid(r: &mut SplitMix64, o: &mut Out, steps: usize) -> (String, String) {
    let (w, h) = (range(r, 4, 8) as i64, range(r, 4, 8) as i64);
    let name = |x: i64, y: i64| format!("{}{}", (b'a' + x as u8) as char, y + 1);
    let (mut x, mut y) = (r.below(w as u64) as i64, r.below(h as u64) as i64);
    o.lines.push(format!(
        "A robot stands on a grid with columns a to {} and rows 1 to {h}. North is toward row 1, east toward column {}.",
        (b'a' + (w - 1) as u8) as char,
        (b'a' + (w - 1) as u8) as char
    ));
    o.lines.push(format!("It starts on {}.", name(x, y)));
    for k in 1..=steps {
        loop {
            let (dir, dx, dy) = *pick(r, &[("north", 0i64, -1i64), ("south", 0, 1), ("east", 1, 0), ("west", -1, 0)]);
            let d = range(r, 1, 3);
            let (nx, ny) = (x + dx * d, y + dy * d);
            if nx < 0 || ny < 0 || nx >= w || ny >= h {
                continue;
            }
            x = nx;
            y = ny;
            o.step(k, format!("{} {dir}.", cap(&num_word(d as u64))), format!("On {}.", name(x, y)));
            break;
        }
    }
    ("Which square is the robot on?".into(), format!("{}.", name(x, y)))
}

fn plates(r: &mut SplitMix64, o: &mut Out, steps: usize) -> (String, String) {
    const COLS: [&str; 8] = ["red", "white", "yellow", "black", "orange", "purple", "grey", "pink"];
    let mut pile: Vec<&str> = Vec::new();
    o.lines.push("Plates are piled on a shelf, one on top of another. The shelf starts empty.".into());
    let show = |p: &[&str]| if p.is_empty() { "The pile is empty.".to_string() } else { format!("Pile, bottom to top: {}.", p.join(", ")) };
    for k in 1..=steps {
        if !pile.is_empty() && (pile.len() >= 7 || chance(r, 2, 5)) {
            let c = pile.pop().unwrap();
            o.step(k, format!("Take the top plate off (the {c} one)."), show(&pile));
        } else {
            let c = *pick(r, &COLS);
            pile.push(c);
            o.step(k, format!("Put a {c} plate on the pile."), show(&pile));
        }
    }
    if pile.is_empty() || chance(r, 1, 2) {
        ("What does the pile look like now?".into(), show(&pile))
    } else {
        ("Which plate is on top?".into(), format!("The {} one.", pile.last().unwrap()))
    }
}

fn trades(r: &mut SplitMix64, o: &mut Out, steps: usize) -> (String, String) {
    const THINGS: [&str; 12] = ["a kite", "a drum", "a map", "a lantern", "a shell", "a flute", "a compass", "an umbrella", "a scarf", "a ribbon", "a key", "a whistle"];
    let n = range(r, 3, 5) as usize;
    let who: Vec<&str> = sample_k(r, PEOPLE.len(), n).into_iter().map(|i| PEOPLE[i]).collect();
    let mut has: Vec<&str> = sample_k(r, THINGS.len(), n).into_iter().map(|i| THINGS[i]).collect();
    let start: Vec<String> = (0..n).map(|i| format!("{} holds {}", who[i], has[i])).collect();
    o.lines.push(format!("{}.", and_list(&start)));
    let show = |has: &[&str]| format!("Now {}.", (0..n).map(|i| format!("{} has {}", who[i], has[i])).collect::<Vec<_>>().join(", "));
    for k in 1..=steps {
        let p = sample_k(r, n, 2);
        let (a, b) = (p[0], p[1]);
        let what = match r.below(3) {
            0 => format!("{} and {} trade what they hold.", who[a], who[b]),
            1 => format!("{} gives {} to {}, who hands back {}.", who[a], has[a], who[b], has[b]),
            _ => format!("{} and {} trade.", who[a], who[b]),
        };
        has.swap(a, b);
        o.step(k, what, show(&has));
    }
    let t = r.below(n as u64) as usize;
    if chance(r, 1, 2) {
        (format!("Who has {} now?", has[t]), format!("{}.", who[t]))
    } else {
        (format!("What does {} hold now?", who[t]), format!("{}.", cap(has[t])))
    }
}

fn switches(r: &mut SplitMix64, o: &mut Out, steps: usize) -> (String, String) {
    let n = range(r, 4, 8) as usize;
    let lab: Vec<char> = (0..n).map(|i| (b'A' + i as u8) as char).collect();
    let mut on: Vec<bool> = (0..n).map(|_| chance(r, 1, 2)).collect();
    let show = |on: &[bool]| {
        let v: Vec<String> = (0..n).filter(|&i| on[i]).map(|i| lab[i].to_string()).collect();
        if v.is_empty() {
            "All are off.".to_string()
        } else if v.len() == n {
            "All are on.".to_string()
        } else {
            format!("On: {}.", v.join(", "))
        }
    };
    let first: Vec<String> = (0..n).filter(|&i| on[i]).map(|i| lab[i].to_string()).collect();
    let start = match first.len() {
        0 => "all of them are off.".to_string(),
        k if k == n => "all of them are on.".to_string(),
        1 => format!("{} is on and the rest are off.", first[0]),
        _ => format!("{} are on and the rest are off.", and_list(&first)),
    };
    o.lines.push(format!("A panel has {} switches, {} to {}. At the start {start}", num_word(n as u64), lab[0], lab[n - 1]));
    for k in 1..=steps {
        let what = match r.below(8) {
            0..=3 => {
                let i = r.below(n as u64) as usize;
                on[i] = !on[i];
                format!("Flip {}.", lab[i])
            }
            4 => {
                let p = sample_k(r, n, 2);
                on[p[0]] = !on[p[0]];
                on[p[1]] = !on[p[1]];
                format!("Flip {} and {}.", lab[p[0]], lab[p[1]])
            }
            5 => {
                let i = r.below(n as u64) as usize;
                on[i] = true;
                format!("Set {} on (it stays on if it already was).", lab[i])
            }
            6 => {
                let i = r.below(n as u64) as usize;
                on[i] = false;
                format!("Set {} off.", lab[i])
            }
            _ => {
                on.iter_mut().for_each(|b| *b = !*b);
                "Flip every switch.".to_string()
            }
        };
        o.step(k, what, show(&on));
    }
    ("Which switches are on at the end?".into(), show(&on))
}

fn queue(r: &mut SplitMix64, o: &mut Out, steps: usize) -> (String, String) {
    let mut free: Vec<&str> = PEOPLE.to_vec();
    r.shuffle(&mut free);
    let n0 = range(r, 2, 4) as usize;
    let mut q: Vec<&str> = free.drain(..n0).collect();
    let mut gone: Vec<&str> = Vec::new();
    let place = *pick(r, &["the bakery", "the ticket desk", "the post office", "the ice cream van"]);
    o.lines.push(format!("A queue at {place}, front to back: {}.", q.join(", ")));
    let show = |q: &[&str]| if q.is_empty() { "The queue is empty.".to_string() } else { format!("Queue: {}.", q.join(", ")) };
    for k in 1..=steps {
        let what = match r.below(3) {
            0 if !free.is_empty() || !gone.is_empty() => {
                let p = if !free.is_empty() { free.pop().unwrap() } else { gone.remove(0) };
                q.push(p);
                format!("{p} joins the back.")
            }
            1 if q.len() >= 2 => {
                let i = 1 + r.below(q.len() as u64 - 1) as usize;
                let (a, b) = (q[i - 1], q[i]);
                q.swap(i - 1, i);
                format!("{a} lets {b} go ahead.")
            }
            _ if !q.is_empty() => {
                let p = q.remove(0);
                gone.push(p);
                format!("{p}, at the front, is served and leaves.")
            }
            _ => {
                let p = if !free.is_empty() { free.pop().unwrap() } else { gone.remove(0) };
                q.push(p);
                format!("{p} joins the back.")
            }
        };
        o.step(k, what, show(&q));
    }
    if q.is_empty() {
        ("Who is still in the queue?".into(), "Nobody.".into())
    } else if chance(r, 1, 2) {
        ("Who is at the front of the queue?".into(), format!("{}.", q[0]))
    } else {
        ("What is the queue now, front to back?".into(), format!("{}.", q.join(", ")))
    }
}

fn bus(r: &mut SplitMix64, o: &mut Out, steps: usize) -> (String, String) {
    let mut on: Vec<&str> = Vec::new();
    o.lines.push("A bus leaves the depot empty and stops along its route.".into());
    let show = |on: &[&str]| if on.is_empty() { "Nobody is on the bus.".to_string() } else { format!("On the bus: {}.", on.join(", ")) };
    for k in 1..=steps {
        let off: Vec<&str> = on.iter().copied().filter(|_| chance(r, 1, 3)).collect();
        on.retain(|p| !off.contains(p));
        let pool: Vec<&str> = PEOPLE.iter().copied().filter(|p| !on.contains(p) && !off.contains(p)).collect();
        let k_on = if pool.is_empty() { 0 } else { range(r, 0, 2.min(pool.len() as i64)) as usize };
        let get_on: Vec<&str> = sample_k(r, pool.len(), k_on).into_iter().map(|i| pool[i]).collect();
        on.extend(get_on.iter().copied());
        let mut parts = Vec::new();
        if !off.is_empty() {
            parts.push(format!("{} {} off", and_list(&off.iter().map(|s| s.to_string()).collect::<Vec<_>>()), if off.len() == 1 { "gets" } else { "get" }));
        }
        if !get_on.is_empty() {
            parts.push(format!("{} {} on", and_list(&get_on.iter().map(|s| s.to_string()).collect::<Vec<_>>()), if get_on.len() == 1 { "gets" } else { "get" }));
        }
        let what = if parts.is_empty() { "Nobody gets on or off.".to_string() } else { format!("{}.", cap(&parts.join(", and "))) };
        o.step(k, format!("A stop. {what}"), show(&on));
    }
    ("Who is on the bus at the end?".into(), show(&on))
}

pub fn doc(r: &mut SplitMix64) -> String {
    let steps = length(r);
    let each = chance(r, 1, 2);
    let mut o = Out { lines: Vec::new(), each };
    o.lines.push(pick(r, &["Keep track.", "Follow the changes.", "Tracking, step by step.", "Watch the state change."]).to_string());
    let (q, a) = match r.below(7) {
        0 => ring(r, &mut o, steps),
        1 => grid(r, &mut o, steps),
        2 => plates(r, &mut o, steps),
        3 => trades(r, &mut o, steps),
        4 => switches(r, &mut o, steps),
        5 => queue(r, &mut o, steps),
        _ => bus(r, &mut o, steps),
    };
    o.lines.push(format!("{} {}.", if each { "That was" } else { "Those were" }, if steps == 1 { "one step".to_string() } else { format!("{} steps", num_word(steps as u64)) }));
    o.lines.push(format!("Question: {q}\nAnswer: {a}"));
    o.lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths_cover_2_to_64() {
        let mut r = SplitMix64::new(1);
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..2000 {
            seen.insert(length(&mut r));
        }
        assert!(seen.contains(&2) && seen.contains(&64), "{seen:?}");
    }

    #[test]
    fn step_by_step_states_agree_with_the_answer() {
        // with the state written each step, the last written state must say what the answer says
        let mut r = SplitMix64::new(3);
        let mut checked = 0;
        for _ in 0..400 {
            let d = doc(&mut r);
            assert!(d.contains("Question:") && d.contains("Answer:"));
            assert!(!d.to_lowercase().contains("swap"));
            let last = d.lines().filter(|l| l.starts_with("Step ")).last().unwrap();
            let ans = d.lines().last().unwrap().trim_start_matches("Answer: ");
            if last.ends_with(ans) {
                checked += 1;
            }
        }
        assert!(checked > 50, "{checked}");
    }

    #[test]
    fn trades_keep_one_thing_each() {
        let mut r = SplitMix64::new(2);
        for _ in 0..50 {
            let mut o = Out { lines: Vec::new(), each: true };
            trades(&mut r, &mut o, 20);
            let last = o.lines.last().unwrap();
            let n = last.matches(" has ").count();
            assert!((3..=5).contains(&n), "{last}");
        }
    }
}
