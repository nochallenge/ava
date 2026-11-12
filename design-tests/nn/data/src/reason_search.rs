// TESTING / DESIGN ONLY
//! reason_search (C6): search written out as it happens (Stream of Search, Gandhi et al. 2024).
//!
//! COUNTDOWN. Reach a target from 3 to 5 numbers with + - * /, every result a whole number above zero
//! (no x1 or /1 steps). Two rule sets: use every number exactly once, or each at most once. A step joins two
//! numbers of the list into one. The search is written as it runs:
//! - depth first, steps in plain order or the step landing closest to the target first, indented by depth,
//!   with dead ends and backing up;
//! - breadth first, level by level (3 numbers, or 4 when the answer is two steps deep);
//! - best first: always expand the list whose nearest number is closest to the target;
//! - no solution (3 numbers): an exhaustive search that ends "there is no way", checked by a full solver.
//! A trace stops at `CAP` steps and says so honestly ("a solution may still exist").
//!
//! MAZES. Grids of 4 to 8 squares a side (columns a.., rows 1..), walls `#`, start S, goal G; depth first
//! (plain or toward the goal), breadth first (distance rings, then the shortest path), and A* (moves so
//! far + straight-line guess). Some mazes have no way through; the trace ends after every reachable square.

use crate::reason::{chance, pick, range};
use crate::rng::SplitMix64;
use std::collections::{BTreeSet, HashSet, VecDeque};

const MAX_VALUE: u32 = 100_000;
/// Most step lines a countdown trace writes before it stops.
pub const CAP: usize = 48;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

const OPS: [Op; 4] = [Op::Add, Op::Sub, Op::Mul, Op::Div];

impl Op {
    pub fn sym(self) -> char {
        match self {
            Op::Add => '+',
            Op::Sub => '-',
            Op::Mul => '*',
            Op::Div => '/',
        }
    }
}

/// a op b, with a >= b; None when not a whole number above zero, or a useless x1 / 1 step.
pub fn apply(a: u32, b: u32, op: Op) -> Option<u32> {
    let c = match op {
        Op::Add => a + b,
        Op::Sub if a > b => a - b,
        Op::Mul if a > 1 && b > 1 => a.checked_mul(b)?,
        Op::Div if b > 1 && a % b == 0 => a / b,
        _ => return None,
    };
    (c > 0 && c <= MAX_VALUE).then_some(c)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub a: u32,
    pub op: Op,
    pub b: u32,
    pub c: u32,
}

impl Step {
    pub fn text(&self) -> String {
        format!("{} {} {} = {}", self.a, self.op.sym(), self.b, self.c)
    }
}

/// Every step from a list (kept largest first), with the list it leaves; repeats removed.
pub fn moves(state: &[u32]) -> Vec<(Step, Vec<u32>)> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for i in 0..state.len() {
        for j in i + 1..state.len() {
            let (a, b) = (state[i], state[j]);
            for op in OPS {
                let Some(c) = apply(a, b, op) else { continue };
                if !seen.insert((a, b, op)) {
                    continue;
                }
                let mut ns: Vec<u32> = state.iter().enumerate().filter(|&(k, _)| k != i && k != j).map(|(_, &v)| v).collect();
                ns.push(c);
                ns.sort_unstable_by(|x, y| y.cmp(x));
                out.push((Step { a, op, b, c }, ns));
            }
        }
    }
    out
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Countdown {
    /// Largest first.
    pub nums: Vec<u32>,
    pub target: u32,
    /// Every number must be used (else: each at most once).
    pub use_all: bool,
}

impl Countdown {
    pub fn goal(&self, s: &[u32]) -> bool {
        if self.use_all {
            s.len() == 1 && s[0] == self.target
        } else {
            s.contains(&self.target)
        }
    }

    /// How far the list's nearest number is from the target.
    pub fn dist(&self, s: &[u32]) -> u32 {
        s.iter().map(|&n| n.abs_diff(self.target)).min().unwrap_or(u32::MAX)
    }
}

/// Every value a puzzle can end on (use_all: the last number; else any number of any reachable list).
pub fn reachable(nums: &[u32], use_all: bool) -> BTreeSet<u32> {
    let mut start = nums.to_vec();
    start.sort_unstable_by(|x, y| y.cmp(x));
    let mut seen: HashSet<Vec<u32>> = HashSet::new();
    let mut stack = vec![start];
    let mut vals = BTreeSet::new();
    while let Some(s) = stack.pop() {
        if !seen.insert(s.clone()) {
            continue;
        }
        if use_all {
            if s.len() == 1 {
                vals.insert(s[0]);
            }
        } else {
            vals.extend(s.iter().copied());
        }
        for (_, ns) in moves(&s) {
            if !seen.contains(&ns) {
                stack.push(ns);
            }
        }
    }
    vals
}

/// The fewest steps to the goal (breadth first over lists), or None.
pub fn shortest(p: &Countdown) -> Option<usize> {
    let mut q = VecDeque::from([(p.nums.clone(), 0usize)]);
    let mut seen = HashSet::new();
    while let Some((s, d)) = q.pop_front() {
        if p.goal(&s) {
            return Some(d);
        }
        if !seen.insert(s.clone()) {
            continue;
        }
        for (_, ns) in moves(&s) {
            q.push_back((ns, d + 1));
        }
    }
    None
}

fn list(s: &[u32]) -> String {
    s.iter().map(|n| n.to_string()).collect::<Vec<_>>().join(", ")
}

/// How a trace ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Found(Vec<Step>),
    NoWay,
    Stopped,
}

struct Style {
    left: usize,
    dead: usize,
}

fn step_line(st: &Style, step: &Step, left: &[u32]) -> String {
    match st.left {
        0 => format!("{}. Left: {}.", step.text(), list(left)),
        1 => format!("{}, leaving {}.", step.text(), list(left)),
        _ => format!("Try {}: now {}.", step.text(), list(left)),
    }
}

fn dead_line(st: &Style, p: &Countdown, left: &[u32]) -> String {
    match st.dead {
        0 => format!("Not {}. Dead end.", p.target),
        1 => format!("{} is not {}.", left[0], p.target),
        _ => "No good.".to_string(),
    }
}

/// Depth first. `closest`: children sorted by distance to the target.
pub fn dfs_trace(p: &Countdown, closest: bool, cap: usize, st_left: usize, st_dead: usize, out: &mut Vec<String>) -> Outcome {
    struct D<'a> {
        p: &'a Countdown,
        closest: bool,
        cap: usize,
        n: usize,
        st: Style,
        path: Vec<Step>,
    }
    enum R {
        Found,
        Exhausted,
        Capped,
    }
    fn go(d: &mut D, s: &[u32], depth: usize, out: &mut Vec<String>) -> R {
        let mut kids = moves(s);
        if d.closest {
            kids.sort_by_key(|(_, ns)| d.p.dist(ns));
        }
        let ind = "  ".repeat(depth);
        for (step, ns) in kids {
            if d.n >= d.cap {
                return R::Capped;
            }
            d.n += 1;
            d.path.push(step);
            let line = format!("{ind}{}", step_line(&d.st, &step, &ns));
            if d.p.goal(&ns) {
                out.push(format!("{line} That makes {}!", d.p.target));
                return R::Found;
            }
            if ns.len() == 1 {
                out.push(format!("{line} {}", dead_line(&d.st, d.p, &ns)));
                d.path.pop();
                continue;
            }
            out.push(line);
            match go(d, &ns, depth + 1, out) {
                R::Found => return R::Found,
                R::Capped => return R::Capped,
                R::Exhausted => out.push(format!("{ind}  Nothing from {} reaches {}; back up.", list(&ns), d.p.target)),
            }
            d.path.pop();
        }
        R::Exhausted
    }
    let mut d = D { p, closest, cap, n: 0, st: Style { left: st_left, dead: st_dead }, path: Vec::new() };
    match go(&mut d, &p.nums, 0, out) {
        R::Found => Outcome::Found(d.path),
        R::Exhausted => Outcome::NoWay,
        R::Capped => Outcome::Stopped,
    }
}

/// Breadth first, two levels: level 1 lists every step from the start, level 2 every result from each
/// level-1 list. Used for 3 numbers, or when the goal is at most two steps deep.
pub fn bfs_trace(p: &Countdown, out: &mut Vec<String>) -> Outcome {
    out.push("Level 1, every first step:".to_string());
    let first = moves(&p.nums);
    for (step, ns) in &first {
        out.push(format!("  {}, giving {}", step.text(), list(ns)));
        if p.goal(ns) {
            out.push(format!("{} is there after one step.", p.target));
            return Outcome::Found(vec![*step]);
        }
    }
    out.push("No target yet. Level 2, every step from each of those lists:".to_string());
    for (s1, ns) in &first {
        let kids = moves(ns);
        if kids.is_empty() {
            continue;
        }
        let mut got = Vec::new();
        for (s2, nns) in &kids {
            got.push(s2.c.to_string());
            if p.goal(nns) {
                out.push(format!("  from {} we can make {}. Found it.", list(ns), got.join(", ")));
                return Outcome::Found(vec![*s1, *s2]);
            }
        }
        out.push(format!("  from {} we can make {}", list(ns), got.join(", ")));
    }
    Outcome::NoWay
}

/// Best first: expand the list whose nearest number is closest to the target (fewer steps first on ties).
pub fn best_first_trace(p: &Countdown, cap: usize, out: &mut Vec<String>) -> Outcome {
    let mut frontier: Vec<(u32, usize, usize, Vec<u32>, Vec<Step>)> = vec![(p.dist(&p.nums), 0, 0, p.nums.clone(), Vec::new())];
    let mut seen: HashSet<Vec<u32>> = HashSet::new();
    seen.insert(p.nums.clone());
    let mut seq = 1;
    for _ in 0..cap {
        if frontier.is_empty() {
            out.push("Nothing is left to expand.".to_string());
            return Outcome::NoWay;
        }
        let k = (0..frontier.len()).min_by_key(|&i| (frontier[i].0, frontier[i].1, frontier[i].2)).unwrap();
        let (h, _, _, s, path) = frontier.swap_remove(k);
        out.push(format!("Expand {} (nearest is {} away).", list(&s), h));
        let mut kids: Vec<(Step, Vec<u32>)> = moves(&s).into_iter().filter(|(_, ns)| !seen.contains(ns)).collect();
        if let Some((step, _)) = kids.iter().find(|(_, ns)| p.goal(ns)) {
            out.push(format!("  {} gives {}. Found it.", step.text(), p.target));
            let mut pa = path.clone();
            pa.push(*step);
            return Outcome::Found(pa);
        }
        if kids.is_empty() {
            out.push("  No new lists from here.".to_string());
            continue;
        }
        kids.sort_by_key(|(_, ns)| p.dist(ns));
        let best: Vec<String> = kids.iter().take(3).map(|(st, ns)| format!("{} ({} away)", st.text(), p.dist(ns))).collect();
        out.push(format!("  {} new lists; nearest: {}.", kids.len(), best.join(", ")));
        for (st, ns) in kids {
            seen.insert(ns.clone());
            let mut pa = path.clone();
            pa.push(st);
            frontier.push((p.dist(&ns), pa.len(), seq, ns, pa));
            seq += 1;
        }
    }
    Outcome::Stopped
}

/// Replay a solution: true when its steps are legal from the start and end on the goal.
pub fn check_solution(p: &Countdown, steps: &[Step]) -> bool {
    let mut s = p.nums.clone();
    for st in steps {
        let Some((_, ns)) = moves(&s).into_iter().find(|(m, _)| m == st) else { return false };
        s = ns;
    }
    p.goal(&s)
}

fn pick_numbers(r: &mut SplitMix64, n: usize) -> Vec<u32> {
    let big = chance(r, 1, 4);
    let mut v: Vec<u32> = (0..n).map(|_| range(r, 1, 10) as u32).collect();
    if big {
        v[0] = *pick(r, &[25u32, 50, 75, 100]);
    }
    v.sort_unstable_by(|x, y| y.cmp(x));
    v
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Method {
    DfsPlain,
    DfsClosest,
    Bfs,
    BestFirst,
}

/// A solvable or unsolvable puzzle for a method (None: try again).
fn make_puzzle(r: &mut SplitMix64, method: Method, solvable: bool) -> Option<Countdown> {
    let n = match (method, solvable) {
        (_, false) => 3,
        (Method::Bfs, _) => *pick(r, &[3usize, 3, 4]),
        (Method::BestFirst, _) => *pick(r, &[4usize, 4, 5]),
        _ => *pick(r, &[3usize, 4, 4, 4, 5]),
    };
    let use_all = chance(r, 2, 5);
    let nums = pick_numbers(r, n);
    let reach = reachable(&nums, use_all);
    let target = if solvable {
        let opts: Vec<u32> = reach.iter().copied().filter(|&v| (10..=999).contains(&v) && !nums.contains(&v)).collect();
        if opts.is_empty() {
            return None;
        }
        *pick(r, &opts)
    } else {
        let t = range(r, 10, 120) as u32;
        if reach.contains(&t) || nums.contains(&t) {
            return None;
        }
        t
    };
    let p = Countdown { nums, target, use_all };
    if method == Method::Bfs && solvable && n == 4 && shortest(&p).is_none_or(|d| d > 2) {
        return None;
    }
    Some(p)
}

/// Run one method's trace (its "Search: ..." line first). Picks are drawn from `r`.
fn method_trace(r: &mut SplitMix64, p: &Countdown, method: Method, out: &mut Vec<String>) -> Outcome {
    match method {
        Method::DfsPlain | Method::DfsClosest => {
            out.push(
                if method == Method::DfsPlain {
                    "Search: depth first, trying the steps in order and backing up from dead ends."
                } else {
                    "Search: depth first, trying first the step that lands closest to the target."
                }
                .to_string(),
            );
            let cap = if p.nums.len() == 3 { 80 } else { CAP };
            let (sl, sd) = (r.below(3) as usize, r.below(3) as usize);
            dfs_trace(p, method == Method::DfsClosest, cap, sl, sd, out)
        }
        Method::Bfs => {
            out.push("Search: breadth first, one level of steps at a time.".to_string());
            bfs_trace(p, out)
        }
        Method::BestFirst => {
            out.push("Search: best first. Always expand the list whose nearest number is closest to the target.".to_string());
            best_first_trace(p, 14, out)
        }
    }
}

/// A puzzle and its trace. A trace that hits its cap is kept only 1 time in 20 (an honest "I stop here"
/// stays in the data, but most docs end in a solution or a proof that none exists).
fn countdown_case(r: &mut SplitMix64) -> (Countdown, Vec<String>, Outcome) {
    let roll = r.below(100);
    let (method, solvable) = match roll {
        0..=19 => (Method::DfsPlain, true),
        20..=44 => (Method::DfsClosest, true),
        45..=62 => (Method::Bfs, true),
        63..=84 => (Method::BestFirst, true),
        85..=92 => (*pick(r, &[Method::DfsPlain, Method::DfsClosest]), false),
        _ => (Method::Bfs, false),
    };
    loop {
        let Some(p) = make_puzzle(r, method, solvable) else { continue };
        let mut tr = Vec::new();
        let o = method_trace(r, &p, method, &mut tr);
        if o == Outcome::Stopped && !chance(r, 1, 20) {
            continue;
        }
        return (p, tr, o);
    }
}

fn countdown_doc(r: &mut SplitMix64) -> String {
    let (p, trace, outcome) = countdown_case(r);
    let mut out: Vec<String> = Vec::new();
    let head = ["Countdown.", "Number puzzle.", "Countdown search.", "Make the target."];
    out.push(format!("{} Numbers: {}. Target: {}.", pick(r, &head), list(&p.nums), p.target));
    out.push(
        if p.use_all {
            *pick(r, &["Every number must be used exactly once.", "Use all of the numbers, each one once."])
        } else {
            *pick(r, &["Each number may be used once at most; some can be left over.", "Use each number at most once. Not all are needed."])
        }
        .to_string(),
    );
    out.push(pick(r, &["Steps: + - * /, and every result must be a whole number above zero.", "Allowed: add, subtract, multiply, divide; whole numbers above zero only."]).to_string());
    out.extend(trace);
    match outcome {
        Outcome::Found(steps) => {
            debug_assert!(check_solution(&p, &steps));
            let s: Vec<String> = steps.iter().map(|s| s.text()).collect();
            out.push(format!("{} {}.", pick(r, &["Solution:", "Answer:", "So:"]), s.join("; ")));
        }
        Outcome::NoWay => {
            out.push(format!(
                "{} There is no way to make {} from {}.",
                pick(r, &["Every possible step has been tried.", "That was every way.", "All the options are used up."]),
                p.target,
                list(&p.nums)
            ));
        }
        Outcome::Stopped => {
            out.push(format!("That is a lot of tries without reaching {}, so I stop here. A solution may still exist.", p.target));
        }
    }
    out.join("\n")
}

// ---------------------------------------------------------------------------------------------------
// Mazes
// ---------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Maze {
    pub w: usize,
    pub h: usize,
    pub wall: Vec<bool>,
    pub start: usize,
    pub goal: usize,
}

const DIRS: [(&str, i64, i64); 4] = [("up", 0, -1), ("right", 1, 0), ("down", 0, 1), ("left", -1, 0)];

impl Maze {
    pub fn name(&self, c: usize) -> String {
        format!("{}{}", (b'a' + (c % self.w) as u8) as char, c / self.w + 1)
    }

    /// Open neighbours in the order up, right, down, left, with the direction word.
    pub fn neighbours(&self, c: usize) -> Vec<(usize, &'static str)> {
        let (x, y) = ((c % self.w) as i64, (c / self.w) as i64);
        DIRS.iter()
            .filter_map(|&(d, dx, dy)| {
                let (nx, ny) = (x + dx, y + dy);
                if nx < 0 || ny < 0 || nx >= self.w as i64 || ny >= self.h as i64 {
                    return None;
                }
                let n = ny as usize * self.w + nx as usize;
                (!self.wall[n]).then_some((n, d))
            })
            .collect()
    }

    pub fn manhattan(&self, a: usize, b: usize) -> usize {
        (a % self.w).abs_diff(b % self.w) + (a / self.w).abs_diff(b / self.w)
    }

    pub fn render(&self) -> String {
        let mut s = String::from(" ");
        for x in 0..self.w {
            s.push(' ');
            s.push((b'a' + x as u8) as char);
        }
        for y in 0..self.h {
            s.push('\n');
            s.push_str(&(y + 1).to_string());
            for x in 0..self.w {
                let c = y * self.w + x;
                s.push(' ');
                s.push(if c == self.start {
                    'S'
                } else if c == self.goal {
                    'G'
                } else if self.wall[c] {
                    '#'
                } else {
                    '.'
                });
            }
        }
        s
    }

    /// Fewest moves from the start to every square.
    pub fn distances(&self) -> Vec<Option<usize>> {
        let mut d = vec![None; self.w * self.h];
        d[self.start] = Some(0);
        let mut q = VecDeque::from([self.start]);
        while let Some(c) = q.pop_front() {
            for (n, _) in self.neighbours(c) {
                if d[n].is_none() {
                    d[n] = Some(d[c].unwrap() + 1);
                    q.push_back(n);
                }
            }
        }
        d
    }
}

fn make_maze(r: &mut SplitMix64, reachable_wanted: bool) -> Maze {
    loop {
        let (w, h) = (range(r, 4, 8) as usize, range(r, 4, 8) as usize);
        let density = range(r, 20, 36) as u64;
        let wall: Vec<bool> = (0..w * h).map(|_| r.below(100) < density).collect();
        let open: Vec<usize> = (0..w * h).filter(|&c| !wall[c]).collect();
        if open.len() < 6 {
            continue;
        }
        let start = *pick(r, &open);
        let goal = *pick(r, &open);
        let mut m = Maze { w, h, wall, start, goal };
        if goal == start || m.manhattan(start, goal) < (w + h) / 2 {
            continue;
        }
        m.wall[start] = false;
        if m.distances()[goal].is_some() == reachable_wanted {
            return m;
        }
    }
}

fn path_to(m: &Maze, parent: &[Option<usize>], end: usize) -> Vec<usize> {
    let mut p = vec![end];
    let mut c = end;
    while let Some(q) = parent[c] {
        p.push(q);
        c = q;
    }
    p.reverse();
    debug_assert_eq!(p[0], m.start);
    p
}

/// Depth first; `toward`: neighbours nearest the goal first. Returns the path when found.
pub fn maze_dfs(m: &Maze, toward: bool, out: &mut Vec<String>) -> Option<Vec<usize>> {
    let n = m.w * m.h;
    let mut seen = vec![false; n];
    let mut parent: Vec<Option<usize>> = vec![None; n];
    seen[m.start] = true;
    let mut cur = m.start;
    out.push(format!("At {} (S).", m.name(cur)));
    loop {
        if cur == m.goal {
            out.push(format!("{} is G. Found the goal.", m.name(cur)));
            return Some(path_to(m, &parent, cur));
        }
        let mut nb: Vec<(usize, &str)> = m.neighbours(cur).into_iter().filter(|(c, _)| !seen[*c]).collect();
        if toward {
            nb.sort_by_key(|(c, _)| m.manhattan(*c, m.goal));
        }
        if let Some(&(c, d)) = nb.first() {
            seen[c] = true;
            parent[c] = Some(cur);
            out.push(format!("Go {d} to {}.", m.name(c)));
            cur = c;
            continue;
        }
        // back up until a square with an unvisited open neighbour
        let dead = cur;
        let mut through = Vec::new();
        loop {
            match parent[cur] {
                None => {
                    out.push(format!(
                        "{} is a dead end, and back at the start there is nowhere new to go. Every square reachable from S has been visited; G is not among them. No way through.",
                        m.name(dead)
                    ));
                    return None;
                }
                Some(p) => {
                    cur = p;
                    through.push(m.name(p));
                    if m.neighbours(cur).iter().any(|(c, _)| !seen[*c]) {
                        break;
                    }
                }
            }
        }
        let last = through.pop().unwrap();
        if through.is_empty() {
            out.push(format!("{} is a dead end. Back to {last}.", m.name(dead)));
        } else {
            out.push(format!("{} is a dead end. Back through {} to {last}.", m.name(dead), through.join(", ")));
        }
    }
}

/// Breadth first: squares by distance, then the shortest path.
pub fn maze_bfs(m: &Maze, out: &mut Vec<String>) -> Option<Vec<usize>> {
    let n = m.w * m.h;
    let mut seen = vec![false; n];
    let mut parent: Vec<Option<usize>> = vec![None; n];
    seen[m.start] = true;
    let mut ring = vec![m.start];
    let mut d = 0;
    out.push(format!("Distance 0: {}.", m.name(m.start)));
    loop {
        let mut next = Vec::new();
        for &c in &ring {
            for (nb, _) in m.neighbours(c) {
                if !seen[nb] {
                    seen[nb] = true;
                    parent[nb] = Some(c);
                    next.push(nb);
                }
            }
        }
        d += 1;
        if next.is_empty() {
            out.push(format!("Distance {d}: nothing new. Every square that can be reached has been reached, and G is not one of them. There is no way from S to G."));
            return None;
        }
        let names: Vec<String> = next.iter().map(|&c| if c == m.goal { format!("{} (G)", m.name(c)) } else { m.name(c) }).collect();
        out.push(format!("Distance {d}: {}.", names.join(", ")));
        if next.contains(&m.goal) {
            out.push(format!("G is reached in {d} moves, and no shorter way exists because every nearer square was checked first."));
            return Some(path_to(m, &parent, m.goal));
        }
        ring = next;
    }
}

/// A*: moves so far + straight-line (grid) guess to the goal.
pub fn maze_astar(m: &Maze, out: &mut Vec<String>) -> Option<Vec<usize>> {
    let n = m.w * m.h;
    let mut g: Vec<Option<usize>> = vec![None; n];
    let mut parent: Vec<Option<usize>> = vec![None; n];
    let mut closed = vec![false; n];
    g[m.start] = Some(0);
    let mut open: Vec<usize> = vec![m.start];
    let f = |g: usize, c: usize| g + m.manhattan(c, m.goal);
    loop {
        if open.is_empty() {
            out.push("The open list is empty, and G was never reached. There is no way through.".to_string());
            return None;
        }
        let k = (0..open.len()).min_by_key(|&i| (f(g[open[i]].unwrap(), open[i]), m.manhattan(open[i], m.goal))).unwrap();
        let c = open.remove(k);
        closed[c] = true;
        let gc = g[c].unwrap();
        if c == m.goal {
            out.push(format!("Take {}: it is G, after {gc} moves.", m.name(c)));
            return Some(path_to(m, &parent, c));
        }
        let mut added = Vec::new();
        for (nb, _) in m.neighbours(c) {
            if closed[nb] {
                continue;
            }
            if g[nb].is_none_or(|old| gc + 1 < old) {
                g[nb] = Some(gc + 1);
                parent[nb] = Some(c);
                if !open.contains(&nb) {
                    open.push(nb);
                }
                added.push(format!("{} ({} + {} = {})", m.name(nb), gc + 1, m.manhattan(nb, m.goal), f(gc + 1, nb)));
            }
        }
        let tail = if added.is_empty() { "nothing new".to_string() } else { added.join(", ") };
        out.push(format!("Take {} ({gc} + {} = {}). Add: {tail}.", m.name(c), m.manhattan(c, m.goal), f(gc, c)));
    }
}

fn maze_doc(r: &mut SplitMix64) -> String {
    let reach = !chance(r, 3, 20);
    let m = make_maze(r, reach);
    let mut out = vec![
        pick(r, &["Maze search.", "A maze.", "Find a way through the maze."]).to_string(),
        "S is the start and G the goal; # is a wall and . is open floor. Moves go up, down, left or right.".to_string(),
        m.render(),
        format!("Start: {}. Goal: {}.", m.name(m.start), m.name(m.goal)),
    ];
    let method = r.below(4);
    out.push(
        match method {
            0 => "Search: depth first, trying up, right, down, left in that order.",
            1 => "Search: depth first, trying first the way that heads toward G.",
            2 => "Search: breadth first, all squares one move away, then two, and so on.",
            _ => "Search: A*. Each square scores moves so far plus a straight guess of the moves left; take the lowest.",
        }
        .to_string(),
    );
    let path = match method {
        0 => maze_dfs(&m, false, &mut out),
        1 => maze_dfs(&m, true, &mut out),
        2 => maze_bfs(&m, &mut out),
        _ => maze_astar(&m, &mut out),
    };
    if let Some(p) = path {
        let names: Vec<String> = p.iter().map(|&c| m.name(c)).collect();
        let note = if method < 2 && p.len() - 1 > m.distances()[m.goal].unwrap() { " Depth first found a way, but not the shortest one." } else { "" };
        out.push(format!("Path: {} ({} moves).{note}", names.join(" "), p.len() - 1));
    }
    out.join("\n")
}

/// One search document: a Countdown puzzle (3 in 4) or a maze.
pub fn doc(r: &mut SplitMix64) -> String {
    if r.below(4) < 3 {
        countdown_doc(r)
    } else {
        maze_doc(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_are_exact_whole_numbers() {
        assert_eq!(apply(7, 3, Op::Sub), Some(4));
        assert_eq!(apply(3, 3, Op::Sub), None);
        assert_eq!(apply(7, 2, Op::Div), None);
        assert_eq!(apply(8, 2, Op::Div), Some(4));
        assert_eq!(apply(8, 1, Op::Mul), None);
        assert_eq!(apply(8, 1, Op::Div), None);
        let m = moves(&[8, 3, 2]);
        assert!(m.iter().all(|(s, ns)| ns.len() == 2 && ns.contains(&s.c)));
    }

    #[test]
    fn traces_end_correctly_and_solutions_replay() {
        let mut r = SplitMix64::new(9);
        let (mut found, mut noway, mut stopped) = (0, 0, 0);
        for _ in 0..300 {
            let solvable = r.below(5) > 0;
            let method = *pick(&mut r, &[Method::DfsPlain, Method::DfsClosest, Method::Bfs, Method::BestFirst]);
            let Some(p) = make_puzzle(&mut r, method, solvable) else { continue };
            let reach = reachable(&p.nums, p.use_all).contains(&p.target);
            assert_eq!(reach, solvable);
            let mut out = Vec::new();
            let o = match method {
                Method::DfsPlain => dfs_trace(&p, false, 80, 0, 0, &mut out),
                Method::DfsClosest => dfs_trace(&p, true, 80, 1, 1, &mut out),
                Method::Bfs => bfs_trace(&p, &mut out),
                Method::BestFirst => best_first_trace(&p, 14, &mut out),
            };
            match o {
                Outcome::Found(s) => {
                    assert!(check_solution(&p, &s), "{p:?} {s:?}");
                    found += 1;
                }
                // "no way" is only ever said when the full solver agrees
                Outcome::NoWay => {
                    assert!(!reach, "{p:?}\n{}", out.join("\n"));
                    noway += 1;
                }
                Outcome::Stopped => stopped += 1,
            }
        }
        assert!(found > 100 && noway > 20, "{found} {noway} {stopped}");
        // the docs keep a capped trace only 1 time in 20, so most end in a solution or a proof of none
        let (mut f, mut n, mut s) = (0, 0, 0);
        for _ in 0..400 {
            let (p, _, o) = countdown_case(&mut r);
            match o {
                Outcome::Found(st) => {
                    assert!(check_solution(&p, &st));
                    f += 1
                }
                Outcome::NoWay => n += 1,
                Outcome::Stopped => s += 1,
            }
        }
        assert!(f > 280 && n > 30 && s < 40, "docs: {f} {n} {s}");
    }

    #[test]
    fn mazes_search_and_paths_are_legal() {
        let mut r = SplitMix64::new(4);
        for k in 0..200 {
            let m = make_maze(&mut r, k % 5 != 0);
            let best = m.distances()[m.goal];
            for method in 0..4 {
                let mut out = Vec::new();
                let p = match method {
                    0 => maze_dfs(&m, false, &mut out),
                    1 => maze_dfs(&m, true, &mut out),
                    2 => maze_bfs(&m, &mut out),
                    _ => maze_astar(&m, &mut out),
                };
                assert_eq!(p.is_some(), best.is_some());
                if let Some(p) = p {
                    assert_eq!((p[0], *p.last().unwrap()), (m.start, m.goal));
                    for w in p.windows(2) {
                        assert!(m.neighbours(w[0]).iter().any(|(c, _)| *c == w[1]));
                    }
                    if method >= 2 {
                        assert_eq!(p.len() - 1, best.unwrap(), "breadth first and A* find a shortest path");
                    }
                }
            }
        }
    }

    #[test]
    fn docs_look_right() {
        let mut r = SplitMix64::new(3);
        let d = countdown_doc(&mut r);
        assert!(d.contains("Target:"));
        let m = maze_doc(&mut r);
        assert!(m.contains("Start:") && m.contains('S') && m.contains('G'));
    }
}
