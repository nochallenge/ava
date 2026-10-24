// TESTING / DESIGN ONLY. (v29 WP12) ARC-AGI-1 grid puzzles by rule search.
//
//   cargo run --release --example arc -- <training|evaluation> [per-task.tsv]
//   AVA_THREADS (default 4)
//   AVA_ARC_HOLDOUT=k  the few-shot stress test (v31, k = 1 or 2): hide each puzzle's last k example
//                      pairs and score them as tests; see hold_out()
//
// A person solving an ARC puzzle looks at the example pairs, guesses a rule, checks it on every
// example, and switches rule when one fails (the prefrontal rule search of the Wisconsin card
// sort). Here:
//   perception  ~50 operations on a grid or its objects: turn, flip, crop to an object, scale,
//               tile, gravity, complete a symmetry, keep/remove objects, count, compress
//   learning    some steps are learned from the pairs instead of chosen: a colour mapping, a
//               local rule (a cell's neighbourhood -> its new colour), an object recolouring
//               (an object's size/shape/colour -> its new colour), how the halves of a grid
//               combine, how a grid tiles into a bigger one, which colour is "noise" to repair
//   search      chains of up to 2 operations, then optionally one learned step; a rule must
//               reproduce EVERY training pair exactly; simplest first (fewest steps, then
//               cheapest learned step) -- Occam
//   answer      the two simplest distinct predictions (ARC Prize: 2 attempts); if no rule fits
//               she abstains
// The data files are read as data; nothing from the downloaded repository is run.
// Pre-registered in "ava v29 - in depth" (WP12): develop on training, evaluation run once.
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::time::Instant;

type Grid = Vec<Vec<u8>>;
type Op = (String, Box<dyn Fn(&Grid) -> Option<Grid> + Send + Sync>);
/// (name, cost, raw input only, fit on (train in, train out) and predict the test inputs)
type Learner = (String, u32, bool, Box<dyn Fn(&[Grid], &[Grid], &[Grid]) -> Option<Vec<Grid>> + Send + Sync>);

const DATA: &str = "../data/arc-agi-1/ARC-AGI-master/data";

// ---------------------------------------------------------------- a tiny JSON reader (ARC files)
enum J { Arr(Vec<J>), Obj(Vec<(String, J)>), Num(i64), Str(String) }

struct Rd<'a> { b: &'a [u8], i: usize }

impl Rd<'_> {
    fn ws(&mut self) { while self.i < self.b.len() && self.b[self.i].is_ascii_whitespace() { self.i += 1; } }
    fn val(&mut self) -> J {
        self.ws();
        match self.b[self.i] {
            b'[' => {
                self.i += 1;
                let mut v = Vec::new();
                loop {
                    self.ws();
                    if self.b[self.i] == b']' { self.i += 1; break; }
                    v.push(self.val());
                    self.ws();
                    if self.b[self.i] == b',' { self.i += 1; }
                }
                J::Arr(v)
            }
            b'{' => {
                self.i += 1;
                let mut v = Vec::new();
                loop {
                    self.ws();
                    if self.b[self.i] == b'}' { self.i += 1; break; }
                    let J::Str(k) = self.val() else { panic!("object key") };
                    self.ws();
                    self.i += 1; // ':'
                    let x = self.val();
                    v.push((k, x));
                    self.ws();
                    if self.b[self.i] == b',' { self.i += 1; }
                }
                J::Obj(v)
            }
            b'"' => {
                self.i += 1;
                let s = self.i;
                while self.b[self.i] != b'"' { self.i += 1; }
                self.i += 1;
                J::Str(String::from_utf8_lossy(&self.b[s..self.i - 1]).into_owned())
            }
            _ => {
                let s = self.i;
                while self.i < self.b.len() && (self.b[self.i] == b'-' || self.b[self.i].is_ascii_digit()) { self.i += 1; }
                J::Num(std::str::from_utf8(&self.b[s..self.i]).unwrap().parse().expect("number"))
            }
        }
    }
}

fn get<'a>(o: &'a J, k: &str) -> Option<&'a J> {
    if let J::Obj(v) = o { v.iter().find(|(a, _)| a == k).map(|(_, b)| b) } else { None }
}

fn to_grid(j: &J) -> Grid {
    let J::Arr(rows) = j else { return Vec::new() };
    rows.iter().map(|r| match r { J::Arr(c) => c.iter().map(|x| if let J::Num(n) = x { *n as u8 } else { 0 }).collect(), _ => Vec::new() }).collect()
}

struct Task { name: String, train: Vec<(Grid, Grid)>, test: Vec<(Grid, Grid)> }

fn load(p: &std::path::Path) -> Task {
    let text = std::fs::read(p).expect("read a puzzle");
    let j = Rd { b: &text, i: 0 }.val();
    let pairs = |k: &str| -> Vec<(Grid, Grid)> {
        let Some(J::Arr(v)) = get(&j, k) else { return Vec::new() };
        v.iter().map(|x| (to_grid(get(x, "input").expect("input")), get(x, "output").map(to_grid).unwrap_or_default())).collect()
    };
    Task { name: p.file_stem().unwrap().to_string_lossy().into_owned(), train: pairs("train"), test: pairs("test") }
}

// ---------------------------------------------------------------- grids
fn hw(g: &Grid) -> (usize, usize) { (g.len(), g.first().map_or(0, |r| r.len())) }

fn ok(g: &Grid) -> bool {
    let (h, w) = hw(g);
    h > 0 && w > 0 && h <= 30 && w <= 30 && g.iter().all(|r| r.len() == w)
}

fn blank(h: usize, w: usize, c: u8) -> Grid { vec![vec![c; w]; h] }

/// the background: black if the grid has any, else its most common colour
fn bg(g: &Grid) -> u8 {
    let mut n = [0usize; 10];
    for r in g { for &c in r { n[c as usize] += 1; } }
    if n[0] > 0 { return 0; }
    (0..10).max_by_key(|&c| (n[c], std::cmp::Reverse(c))).unwrap() as u8
}

fn has(g: &Grid, c: u8) -> bool { g.iter().any(|r| r.contains(&c)) }

fn sub(g: &Grid, r0: usize, c0: usize, r1: usize, c1: usize) -> Grid { (r0..=r1).map(|r| g[r][c0..=c1].to_vec()).collect() }

fn rot90(g: &Grid) -> Grid { let (h, w) = hw(g); (0..w).map(|i| (0..h).map(|j| g[h - 1 - j][i]).collect()).collect() }
fn flip_lr(g: &Grid) -> Grid { g.iter().map(|r| r.iter().rev().copied().collect()).collect() }
fn flip_ud(g: &Grid) -> Grid { g.iter().rev().cloned().collect() }
fn transpose(g: &Grid) -> Grid { let (h, w) = hw(g); (0..w).map(|i| (0..h).map(|j| g[j][i]).collect()).collect() }

/// the 8 turns and flips of a square: 0 as is, 1-3 quarter turns clockwise, 4 left-right,
/// 5 up-down, 6 transpose, 7 anti-transpose
fn d4(g: &Grid, k: usize) -> Grid {
    match k {
        0 => g.clone(),
        1 => rot90(g),
        2 => flip_ud(&flip_lr(g)),
        3 => rot90(&flip_ud(&flip_lr(g))),
        4 => flip_lr(g),
        5 => flip_ud(g),
        6 => transpose(g),
        _ => transpose(&flip_ud(&flip_lr(g))),
    }
}

fn up(g: &Grid, a: usize, b: usize) -> Grid {
    let (h, w) = hw(g);
    (0..h * a).map(|r| (0..w * b).map(|c| g[r / a][c / b]).collect()).collect()
}

/// shrink by k when every k x k block is one colour
fn down(g: &Grid, k: usize) -> Option<Grid> {
    let (h, w) = hw(g);
    if h % k != 0 || w % k != 0 { return None; }
    for r in 0..h { for c in 0..w { if g[r][c] != g[r / k * k][c / k * k] { return None; } } }
    Some((0..h / k).map(|r| (0..w / k).map(|c| g[r * k][c * k]).collect()).collect())
}

// ---------------------------------------------------------------- objects
struct Obj { cells: Vec<(usize, usize)>, r0: usize, c0: usize, r1: usize, c1: usize, color: u8 }

/// connected non-background cells: one colour or many, 4 or 8 neighbours
fn objects(g: &Grid, multi: bool, diag: bool) -> Vec<Obj> {
    let (h, w) = hw(g);
    let b = bg(g);
    let mut seen = vec![vec![false; w]; h];
    let mut out = Vec::new();
    for r in 0..h {
        for c in 0..w {
            if seen[r][c] || g[r][c] == b { continue; }
            let col = g[r][c];
            let (mut st, mut cells) = (vec![(r, c)], Vec::new());
            seen[r][c] = true;
            while let Some((y, x)) = st.pop() {
                cells.push((y, x));
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if (dy == 0 && dx == 0) || (!diag && dy != 0 && dx != 0) { continue; }
                        let (ny, nx) = (y as i32 + dy, x as i32 + dx);
                        if ny < 0 || nx < 0 || ny >= h as i32 || nx >= w as i32 { continue; }
                        let (ny, nx) = (ny as usize, nx as usize);
                        if seen[ny][nx] || g[ny][nx] == b || (!multi && g[ny][nx] != col) { continue; }
                        seen[ny][nx] = true;
                        st.push((ny, nx));
                    }
                }
            }
            cells.sort_unstable();
            let (r0, r1) = (cells.iter().map(|p| p.0).min().unwrap(), cells.iter().map(|p| p.0).max().unwrap());
            let (c0, c1) = (cells.iter().map(|p| p.1).min().unwrap(), cells.iter().map(|p| p.1).max().unwrap());
            let mut n = [0usize; 10];
            for &(y, x) in &cells { n[g[y][x] as usize] += 1; }
            let color = (0..10).max_by_key(|&k| (n[k], std::cmp::Reverse(k))).unwrap() as u8;
            out.push(Obj { cells, r0, c0, r1, c1, color });
        }
    }
    out
}

fn shape(o: &Obj) -> Vec<Vec<bool>> {
    let mut m = vec![vec![false; o.c1 - o.c0 + 1]; o.r1 - o.r0 + 1];
    for &(r, c) in &o.cells { m[r - o.r0][c - o.c0] = true; }
    m
}

fn ncolors(o: &Obj, g: &Grid) -> usize { o.cells.iter().map(|&(r, c)| g[r][c]).collect::<HashSet<_>>().len() }

#[derive(Clone, Copy)]
enum Pick { Largest, Smallest, BiggestBox, UniqueColor, UniqueShape, MostColors }

/// which object: only ever a unique winner (a tie means she can't tell, so no answer)
fn pick(objs: &[Obj], g: &Grid, p: Pick) -> Option<usize> {
    let unique_best = |score: &dyn Fn(&Obj) -> i64| -> Option<usize> {
        let best = objs.iter().map(score).max()?;
        let ids: Vec<usize> = (0..objs.len()).filter(|&i| score(&objs[i]) == best).collect();
        (ids.len() == 1).then(|| ids[0])
    };
    let only = |ids: Vec<usize>| (ids.len() == 1).then(|| ids[0]);
    match p {
        Pick::Largest => unique_best(&|o| o.cells.len() as i64),
        Pick::Smallest => unique_best(&|o| -(o.cells.len() as i64)),
        Pick::BiggestBox => unique_best(&|o| ((o.r1 - o.r0 + 1) * (o.c1 - o.c0 + 1)) as i64),
        Pick::UniqueColor => only((0..objs.len()).filter(|&i| objs.iter().filter(|o| o.color == objs[i].color).count() == 1).collect()),
        Pick::UniqueShape => {
            let sh: Vec<_> = objs.iter().map(shape).collect();
            only((0..objs.len()).filter(|&i| sh.iter().filter(|s| **s == sh[i]).count() == 1).collect())
        }
        Pick::MostColors => unique_best(&|o| ncolors(o, g) as i64),
    }
}

fn crop_pick(g: &Grid, multi: bool, diag: bool, p: Pick, mask: bool) -> Option<Grid> {
    let objs = objects(g, multi, diag);
    let o = &objs[pick(&objs, g, p)?];
    if !mask { return Some(sub(g, o.r0, o.c0, o.r1, o.c1)); }
    let mut out = blank(o.r1 - o.r0 + 1, o.c1 - o.c0 + 1, bg(g));
    for &(r, c) in &o.cells { out[r - o.r0][c - o.c0] = g[r][c]; }
    Some(out)
}

fn crop_all(g: &Grid) -> Option<Grid> {
    let b = bg(g);
    let cells: Vec<(usize, usize)> = (0..g.len()).flat_map(|r| (0..g[r].len()).map(move |c| (r, c))).filter(|&(r, c)| g[r][c] != b).collect();
    if cells.is_empty() { return None; }
    let (r0, r1) = (cells.iter().map(|p| p.0).min()?, cells.iter().map(|p| p.0).max()?);
    let (c0, c1) = (cells.iter().map(|p| p.1).min()?, cells.iter().map(|p| p.1).max()?);
    Some(sub(g, r0, c0, r1, c1))
}

// ---------------------------------------------------------------- whole-grid operations
fn fall_down(g: &Grid) -> Grid {
    let (h, w) = hw(g);
    let b = bg(g);
    let mut out = blank(h, w, b);
    for c in 0..w {
        let col: Vec<u8> = (0..h).map(|r| g[r][c]).filter(|&x| x != b).collect();
        for (i, &x) in col.iter().enumerate() { out[h - col.len() + i][c] = x; }
    }
    out
}

/// complete a symmetric picture: a background cell takes the colour its mirror image has
fn sym_fill(g: &Grid, ks: &[usize]) -> Option<Grid> {
    let b = bg(g);
    let imgs: Vec<Grid> = ks.iter().map(|&k| d4(g, k)).filter(|t| hw(t) == hw(g)).collect();
    if imgs.is_empty() { return None; }
    let mut out = g.clone();
    for r in 0..g.len() {
        for c in 0..g[r].len() {
            if out[r][c] != b { continue; }
            if let Some(t) = imgs.iter().find(|t| t[r][c] != b) { out[r][c] = t[r][c]; }
        }
    }
    Some(out)
}

/// each coloured cell becomes a copy of the whole grid (or each background cell does)
fn kron(g: &Grid, inv: bool) -> Option<Grid> {
    let (h, w) = hw(g);
    if h * h > 30 || w * w > 30 { return None; }
    let b = bg(g);
    let mut out = blank(h * h, w * w, b);
    for r in 0..h {
        for c in 0..w {
            if (g[r][c] != b) == inv { continue; }
            for i in 0..h { for j in 0..w { out[r * h + i][c * w + j] = g[i][j]; } }
        }
    }
    Some(out)
}

fn colour_counts(g: &Grid) -> [usize; 10] {
    let b = bg(g);
    let mut n = [0usize; 10];
    for r in g { for &c in r { if c != b { n[c as usize] += 1; } } }
    n
}

fn dedup_rows(g: &Grid) -> Grid {
    let mut out: Grid = Vec::new();
    for r in g { if out.last() != Some(r) { out.push(r.clone()); } }
    out
}

fn drop_bg_lines(g: &Grid) -> Option<Grid> {
    let b = bg(g);
    let rows: Grid = g.iter().filter(|r| r.iter().any(|&c| c != b)).cloned().collect();
    if rows.is_empty() { return None; }
    let t: Grid = transpose(&rows).into_iter().filter(|r| r.iter().any(|&c| c != b)).collect();
    Some(transpose(&t))
}

/// shrink by k, each k x k block taking its most common colour
fn down_major(g: &Grid, k: usize) -> Option<Grid> {
    let (h, w) = hw(g);
    if h % k != 0 || w % k != 0 || h == k && w == k { return None; }
    Some((0..h / k).map(|r| (0..w / k).map(|c| {
        let mut n = [0usize; 10];
        for i in 0..k { for j in 0..k { n[g[r * k + i][c * k + j] as usize] += 1; } }
        (0..10).max_by_key(|&x| (n[x], std::cmp::Reverse(x))).unwrap() as u8
    }).collect()).collect())
}

/// a colour carries on downwards over background until the next colour (the other directions
/// by turning the grid)
fn carry_down(g: &Grid) -> Grid {
    let b = bg(g);
    let mut out = g.clone();
    for c in 0..g[0].len() {
        let mut cur = b;
        for r in 0..g.len() { if g[r][c] != b { cur = g[r][c]; } else { out[r][c] = cur; } }
    }
    out
}

/// two cells of the same colour in a row or column are joined by a line of that colour (over
/// background only)
fn connect(g: &Grid) -> Grid {
    let b = bg(g);
    let mut out = g.clone();
    for (r, row) in g.iter().enumerate() {
        let pts: Vec<usize> = (0..row.len()).filter(|&c| row[c] != b).collect();
        for p in pts.windows(2) { if row[p[0]] == row[p[1]] { for c in p[0] + 1..p[1] { out[r][c] = row[p[0]]; } } }
    }
    for c in 0..g[0].len() {
        let pts: Vec<usize> = (0..g.len()).filter(|&r| g[r][c] != b).collect();
        for p in pts.windows(2) { if g[p[0]][c] == g[p[1]][c] { for r in p[0] + 1..p[1] { if out[r][c] == b { out[r][c] = g[p[0]][c]; } } } }
    }
    out
}

fn op(n: impl Into<String>, f: impl Fn(&Grid) -> Option<Grid> + Send + Sync + 'static) -> Op { (n.into(), Box::new(f)) }


/// `AVA_ARC_FRAME=1`: look through the window. The largest rectangle outlined in one colour (its
/// border whole, whatever is inside); the answer is what it frames. Only a single such largest frame
/// counts. Off by default.
fn frame_on() -> bool { std::env::var("AVA_ARC_FRAME").map_or(false, |v| v == "1") }

fn crop_in_frame(g: &Grid) -> Option<Grid> {
    let (h, w) = hw(g);
    if h < 3 || w < 3 { return None; }
    let b = bg(g);
    let mut best: Option<(usize, (usize, usize, usize, usize))> = None;
    let mut ties = 0;
    for k in (0..10u8).filter(|&k| k != b) {
        if g.iter().flatten().filter(|&&x| x == k).count() < 8 { continue; }
        // run lengths of colour k to the right and downwards
        let mut right = vec![vec![0usize; w + 1]; h];
        let mut down = vec![vec![0usize; w]; h + 1];
        for r in 0..h { for c in (0..w).rev() { right[r][c] = if g[r][c] == k { right[r][c + 1] + 1 } else { 0 }; } }
        for r in (0..h).rev() { for c in 0..w { down[r][c] = if g[r][c] == k { down[r + 1][c] + 1 } else { 0 }; } }
        for r0 in 0..h { for c0 in 0..w {
            if right[r0][c0] < 3 || down[r0][c0] < 3 { continue; }
            for r1 in r0 + 2..r0 + down[r0][c0] { for c1 in c0 + 2..c0 + right[r0][c0] {
                if right[r1][c0] < c1 - c0 + 1 || down[r0][c1] < r1 - r0 + 1 { continue; }
                let area = (r1 - r0 + 1) * (c1 - c0 + 1);
                match best { Some((a, _)) if a > area => {}, Some((a, _)) if a == area => ties += 1,
                             _ => { best = Some((area, (r0, c0, r1, c1))); ties = 0; } }
            } }
        } }
    }
    let (_, (r0, c0, r1, c1)) = best?;
    (ties == 0).then(|| sub(g, r0 + 1, c0 + 1, r1 - 1, c1 - 1))
}
fn ops() -> Vec<Op> {
    let mut v: Vec<Op> = Vec::new();
    for (k, n) in [(1, "rot90"), (2, "rot180"), (3, "rot270"), (4, "flip_lr"), (5, "flip_ud"), (6, "transpose"), (7, "antitranspose")] {
        v.push(op(n, move |g| Some(d4(g, k))));
    }
    v.push(op("crop_all", crop_all));
    if frame_on() { v.push(op("crop_in_frame", crop_in_frame)); }
    let picks = [(Pick::Largest, "largest"), (Pick::Smallest, "smallest"), (Pick::BiggestBox, "biggest_box"),
                 (Pick::UniqueColor, "unique_colour"), (Pick::UniqueShape, "unique_shape"), (Pick::MostColors, "most_colours")];
    for (multi, diag, tag) in [(false, false, "1c"), (true, true, "mc")] {
        for (p, pn) in picks { v.push(op(format!("crop_{pn}_{tag}"), move |g| crop_pick(g, multi, diag, p, false))); }
    }
    for (p, pn) in picks { v.push(op(format!("cut_{pn}"), move |g| crop_pick(g, true, true, p, true))); }
    v.push(op("fall_down", |g| Some(fall_down(g))));
    v.push(op("fall_up", |g| Some(flip_ud(&fall_down(&flip_ud(g))))));
    v.push(op("fall_right", |g| Some(transpose(&fall_down(&transpose(g))))));
    v.push(op("fall_left", |g| Some(flip_lr(&transpose(&fall_down(&transpose(&flip_lr(g))))))));
    v.push(op("mirror_lr", |g| sym_fill(g, &[4])));
    v.push(op("mirror_ud", |g| sym_fill(g, &[5])));
    v.push(op("mirror_both", |g| sym_fill(g, &[4, 5, 2])));
    v.push(op("mirror_diag", |g| sym_fill(g, &[6])));
    v.push(op("mirror_all", |g| sym_fill(g, &[4, 5, 2, 6, 7, 1, 3])));
    for k in [2, 3, 4] { v.push(op(format!("up{k}"), move |g| Some(up(g, k, k)))); }
    for k in [2, 3] { v.push(op(format!("down{k}"), move |g| down(g, k))); }
    for k in [2, 3] { v.push(op(format!("down{k}_majority"), move |g| down_major(g, k))); }
    v.push(op("carry_down", |g| Some(carry_down(g))));
    v.push(op("carry_up", |g| Some(flip_ud(&carry_down(&flip_ud(g))))));
    v.push(op("carry_right", |g| Some(transpose(&carry_down(&transpose(g))))));
    v.push(op("carry_left", |g| Some(flip_lr(&transpose(&carry_down(&transpose(&flip_lr(g))))))));
    v.push(op("connect", |g| Some(connect(g))));
    v.push(op("up_by_colours", |g| { let k = colour_counts(g).iter().filter(|&&n| n > 0).count(); (k >= 1).then(|| up(g, k, k)) }));
    v.push(op("kron", |g| kron(g, false)));
    v.push(op("kron_inv", |g| kron(g, true)));
    v.push(op("count_row", |g| {
        let n = colour_counts(g);
        let cs: Vec<usize> = (0..10).filter(|&c| n[c] > 0).collect();
        (cs.len() == 1).then(|| vec![vec![cs[0] as u8; n[cs[0]]]])
    }));
    v.push(op("count_col", |g| {
        let n = colour_counts(g);
        let cs: Vec<usize> = (0..10).filter(|&c| n[c] > 0).collect();
        (cs.len() == 1).then(|| vec![vec![cs[0] as u8]; n[cs[0]]])
    }));
    v.push(op("most_common_1x1", |g| {
        let n = colour_counts(g);
        let best = *n.iter().max()?;
        let cs: Vec<usize> = (0..10).filter(|&c| n[c] == best && best > 0).collect();
        (cs.len() == 1).then(|| vec![vec![cs[0] as u8]])
    }));
    v.push(op("least_common_1x1", |g| {
        let n = colour_counts(g);
        let least = *n.iter().filter(|&&x| x > 0).min()?;
        let cs: Vec<usize> = (0..10).filter(|&c| n[c] == least).collect();
        (cs.len() == 1).then(|| vec![vec![cs[0] as u8]])
    }));
    v.push(op("fill_most_common", |g| {
        let mut n = [0usize; 10];
        for r in g { for &c in r { n[c as usize] += 1; } }
        let best = *n.iter().max()?;
        let cs: Vec<usize> = (0..10).filter(|&c| n[c] == best).collect();
        (cs.len() == 1).then(|| blank(g.len(), g[0].len(), cs[0] as u8))
    }));
    v.push(op("dedup_rows", |g| Some(dedup_rows(g))));
    v.push(op("dedup_cols", |g| Some(transpose(&dedup_rows(&transpose(g))))));
    v.push(op("dedup_both", |g| Some(transpose(&dedup_rows(&transpose(&dedup_rows(g)))))));
    v.push(op("drop_bg_lines", drop_bg_lines));
    v.push(op("keep_largest", |g| {
        let objs = objects(g, true, true);
        let o = &objs[pick(&objs, g, Pick::Largest)?];
        let mut out = blank(g.len(), g[0].len(), bg(g));
        for &(r, c) in &o.cells { out[r][c] = g[r][c]; }
        Some(out)
    }));
    v.push(op("remove_specks", |g| {
        let mut out = g.clone();
        let b = bg(g);
        for o in objects(g, true, true) { if o.cells.len() == 1 { out[o.r0][o.c0] = b; } }
        Some(out)
    }));
    v
}

// ---------------------------------------------------------------- learned steps
/// what a cell "sees", for a learned local rule
fn cell_feats(g: &Grid, kind: usize) -> Vec<Vec<u64>> {
    let (h, w) = hw(g);
    let b = bg(g);
    let at = |r: i32, c: i32| -> u64 { if r < 0 || c < 0 || r >= h as i32 || c >= w as i32 { 10 } else { g[r as usize][c as usize] as u64 } };
    let encl = if kind == 3 { enclosed(g) } else { Vec::new() };
    let rows: Vec<u64> = g.iter().map(|r| r.iter().fold(0u64, |m, &c| m | 1 << c)).collect();
    let cols: Vec<u64> = (0..w).map(|c| (0..h).fold(0u64, |m, r| m | 1 << g[r][c])).collect();
    (0..h).map(|r| (0..w).map(|c| {
        let (ri, ci) = (r as i32, c as i32);
        let own = g[r][c] as u64;
        match kind {
            0 => own,
            1 => [(-1, 0), (0, -1), (0, 1), (1, 0)].iter().fold(own, |k, &(dr, dc)| k * 11 + at(ri + dr, ci + dc)),
            2 => [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)].iter().fold(own, |k, &(dr, dc)| k * 11 + at(ri + dr, ci + dc)),
            3 => own * 2 + encl[r][c] as u64,
            4 => own | rows[r] << 4 | cols[c] << 14,
            // the first colour seen looking up, left, right and down (10 = the edge)
            6 => [(-1, 0), (0, -1), (0, 1), (1, 0)].iter().fold(own, |k, &(dr, dc)| {
                let (mut y, mut x) = (ri + dr, ci + dc);
                while at(y, x) != 10 && at(y, x) == b as u64 { y += dr; x += dc; }
                k * 11 + at(y, x)
            }),
            7 => own * 4 + (r % 2 * 2 + c % 2) as u64,
            _ => own * 16 + [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)].iter()
                .filter(|&&(dr, dc)| { let x = at(ri + dr, ci + dc); x != 10 && x != b as u64 }).count() as u64,
        }
    }).collect()).collect()
}

/// background cells that can't reach the border through background
fn enclosed(g: &Grid) -> Vec<Vec<bool>> {
    let (h, w) = hw(g);
    let b = bg(g);
    let mut out = vec![vec![false; w]; h];
    let mut seen = vec![vec![false; w]; h];
    let mut st: Vec<(usize, usize)> = Vec::new();
    for r in 0..h { for c in 0..w { if (r == 0 || c == 0 || r == h - 1 || c == w - 1) && g[r][c] == b { seen[r][c] = true; st.push((r, c)); } } }
    while let Some((r, c)) = st.pop() {
        for (dr, dc) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
            let (nr, nc) = (r as i32 + dr, c as i32 + dc);
            if nr < 0 || nc < 0 || nr >= h as i32 || nc >= w as i32 { continue; }
            let (nr, nc) = (nr as usize, nc as usize);
            if !seen[nr][nc] && g[nr][nc] == b { seen[nr][nc] = true; st.push((nr, nc)); }
        }
    }
    for r in 0..h { for c in 0..w { out[r][c] = g[r][c] == b && !seen[r][c]; } }
    out
}

/// a cell's new colour as a function of what it sees, learned from every training cell; a test
/// cell that sees something never seen in training leaves no answer (except a plain colour,
/// which keeps itself)
fn local_rule(a: &[Grid], o: &[Grid], t: &[Grid], kind: usize) -> Option<Vec<Grid>> {
    let mut table: HashMap<u64, u8> = HashMap::new();
    for (x, y) in a.iter().zip(o) {
        if hw(x) != hw(y) { return None; }
        let f = cell_feats(x, kind);
        for r in 0..x.len() {
            for c in 0..x[r].len() {
                if *table.entry(f[r][c]).or_insert(y[r][c]) != y[r][c] { return None; }
            }
        }
    }
    t.iter().map(|x| {
        let f = cell_feats(x, kind);
        let mut g = x.clone();
        for r in 0..x.len() {
            for c in 0..x[r].len() {
                g[r][c] = match table.get(&f[r][c]) { Some(&v) => v, None if kind == 0 => x[r][c], None => return None };
            }
        }
        Some(g)
    }).collect()
}

// ---------------------------------------------------------------- WP12b: which aspects matter
/// `AVA_ARC_SELECT=1`: a learned local rule that first finds WHICH things a cell's new colour
/// depends on -- the fewest features that explain every training cell and cover every test cell --
/// as in the Wisconsin card sort, where what is learned is which dimension counts. Each feature
/// is read either as raw colours or colour-blind (background / same as me / another colour /
/// the edge), and a cell's new colour can be kept, copied from what it attends to, or a fixed
/// colour. Off by default (the WP12 search is unchanged).
fn select_on() -> bool { std::env::var("AVA_ARC_SELECT").map_or(false, |v| v != "0" && !v.is_empty()) }
/// held-out check (dev): 0 none, 1 a left-out example must be predicted right where it is covered, 2 it must be covered and right
fn select_check() -> u8 { std::env::var("AVA_ARC_SELECT").ok().and_then(|v| v.parse::<u8>().ok()).map_or(0, |v| v.saturating_sub(1)) }

const NF: usize = 34;
/// `AVA_ARC_POOL=1` adds 5 pooled and relational features (34-38): which colour touches me straight,
/// diagonally, or at all, whatever the side (pooling over position, as complex cells in visual cortex
/// do), and the nearest colour unlike mine, in any direction or along the 4 straight lines. Off by default.
const NF_ALL: usize = 41;
fn pool_on() -> bool { std::env::var("AVA_ARC_POOL").map_or(false, |v| v == "1") }
fn nf() -> usize { if pool_on() { NF_ALL } else { NF } }
const DIRS8: [(i32, i32); 8] = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)];

/// colour-valued features: own (0), the 8 neighbours (1-8), the first colour seen in 8
/// directions (9-16), the colour a cell sits between on 4 lines (25-28)
fn is_colour_feat(k: usize) -> bool { k <= 16 || (25..=28).contains(&k) || (34..39).contains(&k) }

/// what a cell can attend to: NF planes of small values (colours 0-9, 10 = the edge, 11 = none)
fn feat_planes(g: &Grid) -> Vec<Vec<Vec<u8>>> {
    let (h, w) = hw(g);
    let b = bg(g);
    let at = |r: i32, c: i32| -> u8 { if r < 0 || c < 0 || r >= h as i32 || c >= w as i32 { 10 } else { g[r as usize][c as usize] } };
    let first = |r: usize, c: usize, (dr, dc): (i32, i32)| -> u8 {
        let (mut y, mut x) = (r as i32 + dr, c as i32 + dc);
        while at(y, x) != 10 && at(y, x) == b { y += dr; x += dc; }
        at(y, x)
    };
    let encl = enclosed(g);
    let mut comp = vec![vec![0u8; w]; h];
    for o in objects(g, false, false) {
        let bucket = match o.cells.len() { 1 => 1, 2 => 2, 3 => 3, 4 => 4, 5..=8 => 5, _ => 6 };
        for &(r, c) in &o.cells { comp[r][c] = bucket; }
    }
    let mut inbox = vec![vec![0u8; w]; h];
    for o in objects(g, true, true) { for r in o.r0..=o.r1 { for c in o.c0..=o.c1 { inbox[r][c] = 1; } } }
    let row_any: Vec<u8> = g.iter().map(|r| r.iter().any(|&x| x != b) as u8).collect();
    let col_any: Vec<u8> = (0..w).map(|c| (0..h).any(|r| g[r][c] != b) as u8).collect();
    let mut p = vec![vec![vec![0u8; w]; h]; NF_ALL];
    for r in 0..h {
        for c in 0..w {
            let (ri, ci) = (r as i32, c as i32);
            let mut f = [0u8; NF_ALL];
            f[0] = g[r][c];
            for (i, &(dr, dc)) in DIRS8.iter().enumerate() {
                f[1 + i] = at(ri + dr, ci + dc);
                f[9 + i] = first(r, c, (dr, dc));
            }
            f[17] = encl[r][c] as u8;
            f[18] = (1..=8).filter(|&i| f[i] != 10 && f[i] != b).count() as u8;
            f[19] = (1..=4).filter(|&i| f[i] != 10 && f[i] != b).count() as u8;
            f[20] = (r % 2) as u8;
            f[21] = (c % 2) as u8;
            f[22] = ((r + c) % 2) as u8;
            f[23] = (r % 3) as u8;
            f[24] = (c % 3) as u8;
            let between = |x: u8, y: u8| if x == y && x < 10 && x != b { x } else { 11 };
            f[25] = between(f[11], f[12]); // left and right
            f[26] = between(f[9], f[10]); // up and down
            f[27] = between(f[13], f[16]); // the two diagonals
            f[28] = between(f[14], f[15]);
            f[29] = comp[r][c];
            f[30] = (r == 0 || c == 0 || r == h - 1 || c == w - 1) as u8;
            f[31] = row_any[r];
            f[32] = col_any[c];
            f[33] = inbox[r][c];
            // pooled: the one colour touching me (11 none, 12 several), straight, diagonal, any side
            let nb = f;
            let pool = |ids: &[usize]| { let mut s: Vec<u8> = ids.iter().map(|&i| nb[i]).filter(|&x| x != 10 && x != b).collect(); s.sort(); s.dedup(); match s.len() { 0 => 11, 1 => s[0], _ => 12 } };
            f[34] = pool(&[1, 2, 3, 4]);
            f[35] = pool(&[5, 6, 7, 8]);
            f[36] = pool(&[1, 2, 3, 4, 5, 6, 7, 8]);
            // the nearest colour unlike mine (Chebyshev rings), and the nearest along the 4 straight lines
            f[37] = 11;
            for d in 1..h.max(w) as i32 {
                let mut s: Vec<u8> = Vec::new();
                for y in ri - d..=ri + d { for x in ci - d..=ci + d {
                    if (y - ri).abs() != d && (x - ci).abs() != d { continue; }
                    let v = at(y, x);
                    if v != 10 && v != b && v != g[r][c] { s.push(v); }
                } }
                if !s.is_empty() { s.sort(); s.dedup(); f[37] = if s.len() == 1 { s[0] } else { 12 }; break; }
            }
            f[38] = 11;
            let mut best = i32::MAX;
            for &(dr, dc) in &DIRS8[..4] {
                let (mut y, mut x, mut n) = (ri + dr, ci + dc, 1);
                while at(y, x) != 10 && at(y, x) == b { y += dr; x += dc; n += 1; }
                let v = at(y, x);
                if v == 10 { continue; }
                if n < best { best = n; f[38] = v; } else if n == best && f[38] != v { f[38] = 12; }
            }
            // is my whole row / column one colour?
            f[39] = g[r].iter().all(|&x| x == g[r][0]) as u8;
            f[40] = (0..h).all(|y| g[y][c] == g[0][c]) as u8;
            for k in 0..NF_ALL { p[k][r][c] = f[k]; }
        }
    }
    p
}

/// a colour-blind reading of a colour feature: background / same as me / the edge / another colour / none
fn rel(v: u8, own: u8, b: u8) -> u8 {
    if v == 10 { 2 } else if v == 11 { 4 } else if v == 12 { 5 } else if v == b { 0 } else if v == own { 1 } else { 3 }
}

/// `AVA_ARC_KEEP=1`: a case never seen in the examples leaves the cell as it is (things stay as
/// they are unless something I learned says they change -- inertia). The held-out check still asks
/// whether that default would have been right on each example left out. Off by default.
fn keep_on() -> bool { std::env::var("AVA_ARC_KEEP").map_or(false, |v| v == "1") }

/// one candidate rule: the features attended to, read raw or colour-blind; returns (table entries,
/// predictions) when it explains every training cell and every test cell was seen in training
fn try_rule(sub: &[usize], blind: bool, fa: &[Vec<Vec<Vec<u8>>>], ba: &[u8], o: &[Grid], ft: &[Vec<Vec<Vec<u8>>>], bt: &[u8], table: &mut Vec<u16>) -> Option<(usize, Vec<Grid>)> {
    // actions: bit 0 keep my colour, bits 1-3 copy the colour of attended feature i, bits 4-13 colour 0-9
    table.iter_mut().for_each(|x| *x = 0);
    let key = |f: &Vec<Vec<Vec<u8>>>, b: u8, r: usize, c: usize| -> usize {
        sub.iter().fold(0usize, |k, &j| { let v = f[j][r][c]; k * 16 + if blind && is_colour_feat(j) { rel(v, f[0][r][c], b) } else { v } as usize })
    };
    let mut entries = 0usize;
    for ((f, &b), y) in fa.iter().zip(ba).zip(o) {
        for r in 0..y.len() {
            for c in 0..y[r].len() {
                let want = y[r][c];
                let mut m: u16 = 1 << (4 + want);
                if f[0][r][c] == want { m |= 1; }
                for (i, &j) in sub.iter().enumerate() { if is_colour_feat(j) && f[j][r][c] == want { m |= 1 << (1 + i); } }
                let k = key(f, b, r, c);
                let e = &mut table[k];
                if *e == 0 { *e = m | 0x8000; entries += 1; } else { *e &= m | 0x8000; if *e == 0x8000 { return None; } }
            }
        }
    }
    let preds: Option<Vec<Grid>> = ft.iter().zip(bt).map(|(f, &b)| {
        let (h, w) = (f[0].len(), f[0][0].len());
        let mut g = vec![vec![0u8; w]; h];
        for r in 0..h {
            for c in 0..w {
                let e = table[key(f, b, r, c)] & 0x7fff;
                if e == 0 && keep_on() { g[r][c] = f[0][r][c]; continue; }
                if e == 0 { return None; }
                let bit = e.trailing_zeros() as usize;
                g[r][c] = if bit == 0 { f[0][r][c] } else if bit <= 3 { f[sub[bit - 1]][r][c] } else { (bit - 4) as u8 };
                if g[r][c] > 9 { return None; }
            }
        }
        Some(g)
    }).collect();
    preds.map(|p| (entries, p))
}

/// would the rule, learned without one example pair, have predicted that pair? (each pair in turn)
/// groups: which example pair each (possibly imagined) view came from; the first view of a group is
/// the pair as given. Leaving a pair out leaves out all its views, and the rule must predict it as given.
fn held_out_ok(sub: &[usize], blind: bool, fa: &[Vec<Vec<Vec<u8>>>], ba: &[u8], o: &[Grid], groups: &[usize], table: &mut Vec<u16>) -> bool {
    let check = select_check();
    let ng = groups.iter().max().map_or(0, |g| g + 1);
    if check == 0 || ng < 2 { return true; }
    (0..ng).all(|g| {
        let i = groups.iter().position(|&x| x == g).unwrap();
        let keep: Vec<usize> = (0..o.len()).filter(|&j| groups[j] != g).collect();
        let fk: Vec<_> = keep.iter().map(|&j| fa[j].clone()).collect();
        let bk: Vec<u8> = keep.iter().map(|&j| ba[j]).collect();
        let ok_: Vec<Grid> = keep.iter().map(|&j| o[j].clone()).collect();
        match try_rule(sub, blind, &fk, &bk, &ok_, std::slice::from_ref(&fa[i]), std::slice::from_ref(&ba[i]), table) {
            Some((_, p)) => p[0] == o[i],
            None => check == 1,
        }
    })
}

/// the fewest features first; among those, the rule with the fewest table entries (Occam)
/// `AVA_ARC_TURN=1`: when no rule fits the examples as given, she pictures each example pair turned
/// and flipped (the 8 views of a square, input and output alike -- mental rotation) and looks for one
/// rule that fits every view. A rule that only holds one way round finds no consistent table and
/// is not taken. Off by default.
fn turn_on() -> bool { std::env::var("AVA_ARC_TURN").map_or(false, |v| v == "1") }

fn rule_select(a: &[Grid], o: &[Grid], t: &[Grid]) -> Option<Vec<Grid>> {
    let groups: Vec<usize> = (0..a.len()).collect();
    if let Some(p) = rule_select_views(a, o, t, &groups) { return Some(p); }
    if !turn_on() { return None; }
    let (mut a2, mut o2, mut g2) = (Vec::new(), Vec::new(), Vec::new());
    for (i, (x, y)) in a.iter().zip(o).enumerate() {
        let mut seen: Vec<(Grid, Grid)> = Vec::new();
        for k in 0..8 {
            let v = (d4(x, k), d4(y, k));
            if seen.contains(&v) { continue; }
            seen.push(v.clone());
            a2.push(v.0); o2.push(v.1); g2.push(i);
        }
    }
    rule_select_views(&a2, &o2, t, &g2)
}

fn rule_select_views(a: &[Grid], o: &[Grid], t: &[Grid], groups: &[usize]) -> Option<Vec<Grid>> {
    if a.iter().zip(o).any(|(x, y)| hw(x) != hw(y)) || a == o { return None; }
    let fa: Vec<_> = a.iter().map(feat_planes).collect();
    let ft: Vec<_> = t.iter().map(feat_planes).collect();
    let ba: Vec<u8> = a.iter().map(bg).collect();
    let bt: Vec<u8> = t.iter().map(bg).collect();
    let mut table = vec![0u16; 1 << 12];
    let mut best: Option<(usize, Vec<Grid>, Vec<usize>, bool)> = None;
    let mut subs: Vec<Vec<usize>> = Vec::new();
    for size in 1..=3usize {
        subs.clear();
        let nf = nf();
        for i in 0..nf {
            if size == 1 { subs.push(vec![i]); continue; }
            for j in i + 1..nf {
                if size == 2 { subs.push(vec![i, j]); continue; }
                for k in j + 1..nf { subs.push(vec![i, j, k]); }
            }
        }
        for s in &subs {
            for blind in [false, true] {
                if let Some((n, p)) = try_rule(s, blind, &fa, &ba, o, &ft, &bt, &mut table) {
                    if !held_out_ok(s, blind, &fa, &ba, o, groups, &mut table) { continue; }
                    if best.as_ref().map_or(true, |b| n < b.0) { best = Some((n, p, s.clone(), blind)); }
                }
            }
        }
        if best.is_some() { break; }
    }
    if let (Some(b), Ok(_)) = (&best, std::env::var("AVA_ARC_DEBUG")) { eprintln!("rule_select: features {:?} colour-blind {} entries {}", b.2, b.3, b.0); }
    best.map(|b| b.1)
}

fn obj_key(o: &Obj, objs: &[Obj], kind: usize) -> u64 {
    let size = o.cells.len() as u64;
    let mut sh = DefaultHasher::new();
    shape(o).hash(&mut sh);
    match kind {
        0 => size,
        1 => objs.iter().filter(|p| p.cells.len() as u64 > size).count() as u64,
        2 => objs.iter().filter(|p| (p.cells.len() as u64) < size).count() as u64,
        3 => o.color as u64,
        4 => sh.finish(),
        5 => (o.r1 - o.r0 + 1) as u64,
        6 => (o.c1 - o.c0 + 1) as u64,
        7 => (o.r0 == 0 || o.c0 == 0) as u64, // finished below with the grid size
        8 => { let s = shape(o); objs.iter().filter(|p| shape(p) == s).count().min(2) as u64 }
        9 => objs.iter().filter(|p| p.color == o.color).count().min(2) as u64,
        _ => o.color as u64 * 1000 + size,
    }
}

/// each object takes one new colour, chosen by one property of the object (its size, size
/// rank, colour, shape, height, width, touching the border, being the only one of its shape
/// or colour); everything else stays as it is
fn obj_recolour(a: &[Grid], o: &[Grid], t: &[Grid]) -> Option<Vec<Grid>> {
    if a.iter().zip(o).any(|(x, y)| hw(x) != hw(y)) { return None; }
    for (multi, diag) in [(false, false), (false, true), (true, true)] {
        let ao: Vec<Vec<Obj>> = a.iter().map(|x| objects(x, multi, diag)).collect();
        let to: Vec<Vec<Obj>> = t.iter().map(|x| objects(x, multi, diag)).collect();
        let (ae, te): (Vec<_>, Vec<_>) = (a.iter().map(enclosed).collect(), t.iter().map(enclosed).collect());
        // kind 7 touches the border; kind 11 holds a hole (enclosed background inside its box)
        let key = |ob: &Obj, objs: &[Obj], g: &Grid, enc: &[Vec<bool>], kind: usize| -> u64 {
            match kind {
                7 => (ob.r0 == 0 || ob.c0 == 0 || ob.r1 == g.len() - 1 || ob.c1 == g[0].len() - 1) as u64,
                11 => (ob.r0..=ob.r1).any(|r| (ob.c0..=ob.c1).any(|c| enc[r][c])) as u64,
                _ => obj_key(ob, objs, kind),
            }
        };
        'kind: for kind in 0..12 {
            let mut table: HashMap<u64, u8> = HashMap::new();
            for (((x, y), objs), enc) in a.iter().zip(o).zip(&ao).zip(&ae) {
                let mut covered = vec![vec![false; x[0].len()]; x.len()];
                for ob in objs {
                    let col = y[ob.cells[0].0][ob.cells[0].1];
                    if ob.cells.iter().any(|&(r, c)| y[r][c] != col) { continue 'kind; }
                    if *table.entry(key(ob, objs, x, enc, kind)).or_insert(col) != col { continue 'kind; }
                    for &(r, c) in &ob.cells { covered[r][c] = true; }
                }
                for r in 0..x.len() { for c in 0..x[r].len() { if !covered[r][c] && x[r][c] != y[r][c] { continue 'kind; } } }
            }
            let preds: Option<Vec<Grid>> = t.iter().zip(&to).zip(&te).map(|((x, objs), enc)| {
                let mut g = x.clone();
                for ob in objs {
                    let col = *table.get(&key(ob, objs, x, enc, kind))?;
                    for &(r, c) in &ob.cells { g[r][c] = col; }
                }
                Some(g)
            }).collect();
            if preds.is_some() { return preds; }
        }
    }
    None
}

/// the parts of a grid: cut by full-length lines of one colour, or into equal halves / thirds
fn parts(g: &Grid, mode: usize) -> Option<Vec<Grid>> {
    let (h, w) = hw(g);
    let segs = |n: usize, k: usize| -> Option<Vec<(usize, usize)>> {
        if n % k != 0 { return None; }
        let s = n / k;
        Some((0..k).map(|i| (i * s, i * s + s - 1)).collect())
    };
    let runs = |sep: &[bool]| -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        let mut i = 0;
        while i < sep.len() {
            if sep[i] { i += 1; continue; }
            let s = i;
            while i < sep.len() && !sep[i] { i += 1; }
            v.push((s, i - 1));
        }
        v
    };
    let (rs, cs) = match mode {
        0 => {
            let b = bg(g);
            let sr: Vec<bool> = (0..h).map(|r| g[r][0] != b && g[r].iter().all(|&c| c == g[r][0])).collect();
            let sc: Vec<bool> = (0..w).map(|c| {
                let rows: Vec<usize> = (0..h).filter(|&r| !sr[r]).collect();
                !rows.is_empty() && g[rows[0]][c] != b && rows.iter().all(|&r| g[r][c] == g[rows[0]][c])
            }).collect();
            if !sr.contains(&true) && !sc.contains(&true) { return None; }
            (runs(&sr), runs(&sc))
        }
        5 => {
            // dividers: whole lines of a colour used for nothing else (a coincidental line of a shape's
            // own colour is not a divider)
            let b = bg(g);
            let mut found = None;
            for k in (0..10u8).filter(|&k| k != b) {
                let sr: Vec<bool> = (0..h).map(|r| g[r].iter().all(|&x| x == k)).collect();
                let rows: Vec<usize> = (0..h).filter(|&r| !sr[r]).collect();
                if rows.is_empty() { continue; }
                let sc: Vec<bool> = (0..w).map(|c| rows.iter().all(|&r| g[r][c] == k)).collect();
                if !sr.contains(&true) && !sc.contains(&true) { continue; }
                if rows.iter().any(|&r| (0..w).any(|c| g[r][c] == k && !sc[c])) { continue; }
                found = Some((runs(&sr), runs(&sc)));
                break;
            }
            found?
        }
        6 => {
            // groups kept apart by empty rows and columns
            let b = bg(g);
            let er: Vec<bool> = (0..h).map(|r| g[r].iter().all(|&x| x == b)).collect();
            let ec: Vec<bool> = (0..w).map(|c| (0..h).all(|r| g[r][c] == b)).collect();
            if !er.contains(&true) && !ec.contains(&true) { return None; }
            (runs(&er), runs(&ec))
        }
        1 => (segs(h, 2)?, vec![(0, w - 1)]),
        2 => (vec![(0, h - 1)], segs(w, 2)?),
        3 => (segs(h, 3)?, vec![(0, w - 1)]),
        _ => (vec![(0, h - 1)], segs(w, 3)?),
    };
    let ps: Vec<Grid> = rs.iter().flat_map(|&(r0, r1)| cs.iter().map(move |&(c0, c1)| sub(g, r0, c0, r1, c1))).collect();
    if mode == 6 {
        let b = bg(g);
        let ps: Vec<Grid> = ps.into_iter().filter(|p| p.iter().flatten().any(|&c| c != b)).collect();
        return (ps.len() >= 2).then_some(ps);
    }
    if ps.len() < 2 || ps.iter().any(|p| hw(p) != hw(&ps[0])) { return None; }
    Some(ps)
}

fn select_part(ps: &[Grid], b: u8, rule: usize) -> Option<usize> {
    let fill = |p: &Grid| p.iter().flatten().filter(|&&c| c != b).count() as i64;
    let ncol = |p: &Grid| p.iter().flatten().filter(|&&c| c != b).collect::<HashSet<_>>().len() as i64;
    let unique_best = |s: &dyn Fn(&Grid) -> i64| -> Option<usize> {
        let best = ps.iter().map(s).max()?;
        let ids: Vec<usize> = (0..ps.len()).filter(|&i| s(&ps[i]) == best).collect();
        (ids.len() == 1).then(|| ids[0])
    };
    match rule {
        0 => { let ids: Vec<usize> = (0..ps.len()).filter(|&i| ps.iter().filter(|q| **q == ps[i]).count() == 1).collect(); (ids.len() == 1).then(|| ids[0]) }
        1 => unique_best(&|p| fill(p)),
        2 => unique_best(&|p| -fill(p)),
        3 => unique_best(&|p| ncol(p)),
        4 => unique_best(&|p| -ncol(p)),
        _ => {
            let cs: Vec<Vec<u8>> = ps.iter().map(|p| { let mut v: Vec<u8> = p.iter().flatten().copied().filter(|&c| c != b).collect(); v.sort(); v.dedup(); v }).collect();
            let ids: Vec<usize> = (0..ps.len()).filter(|&i| !cs[i].is_empty() && cs.iter().filter(|q| **q == cs[i]).count() == 1).collect();
            (ids.len() == 1).then(|| ids[0])
        }
    }
}

fn perms(k: usize) -> Vec<Vec<usize>> {
    if k == 0 { return vec![Vec::new()]; }
    let mut out = Vec::new();
    for p in perms(k - 1) { for i in 0..k { let mut q = p.clone(); q.insert(i, k - 1); out.push(q); } }
    out.sort();
    out
}

/// `AVA_ARC_DIVIDE=1`: also try dividers seen as a colour used for nothing else (mode 5). Off by default.
fn divide_on() -> bool { std::env::var("AVA_ARC_DIVIDE").map_or(false, |v| v == "1") }

/// `AVA_ARC_GAPS=1`: also see parts as groups kept apart by empty space (mode 6; parts may differ
/// in size, so only picking one applies), and pick the part whose colours no other part has. Off by
/// default.
fn gaps_on() -> bool { std::env::var("AVA_ARC_GAPS").map_or(false, |v| v == "1") }

/// how the parts of a grid make the answer: pick one part, lay them over each other in some
/// order, or a cell's colour from which parts are filled there (AND / OR / XOR ... learned as a
/// table)
fn halves(a: &[Grid], o: &[Grid], t: &[Grid]) -> Option<Vec<Grid>> {
    for mode in 0..7 {
        if (mode == 5 && !divide_on()) || (mode == 6 && !gaps_on()) { continue; }
        let pa: Option<Vec<Vec<Grid>>> = a.iter().map(|g| parts(g, mode)).collect();
        let pt: Option<Vec<Vec<Grid>>> = t.iter().map(|g| parts(g, mode)).collect();
        let (Some(pa), Some(pt)) = (pa, pt) else { continue };
        let (ba, bt): (Vec<u8>, Vec<u8>) = (a.iter().map(bg).collect(), t.iter().map(bg).collect());
        for rule in 0..if gaps_on() { 6 } else { 5 } {
            if pa.iter().zip(o).zip(&ba).all(|((p, y), &b)| select_part(p, b, rule).map_or(false, |i| p[i] == *y)) {
                let r: Option<Vec<Grid>> = pt.iter().zip(&bt).map(|(p, &b)| select_part(p, b, rule).map(|i| p[i].clone())).collect();
                if r.is_some() { return r; }
            }
        }
        let k = pa[0].len();
        if pa.iter().chain(pt.iter()).any(|p| p.iter().any(|q| hw(q) != hw(&p[0]))) { continue; }
        if k > 4 || pa.iter().chain(pt.iter()).any(|p| p.len() != k) || pa.iter().zip(o).any(|(p, y)| hw(&p[0]) != hw(y)) { continue; }
        let lay = |p: &[Grid], b: u8, order: &[usize]| -> Grid {
            let (h, w) = hw(&p[0]);
            (0..h).map(|r| (0..w).map(|c| order.iter().map(|&i| p[i][r][c]).find(|&x| x != b).unwrap_or(b)).collect()).collect()
        };
        for order in perms(k) {
            if pa.iter().zip(o).zip(&ba).all(|((p, y), &b)| lay(p, b, &order) == *y) {
                return Some(pt.iter().zip(&bt).map(|(p, &b)| lay(p, b, &order)).collect());
            }
        }
        let mut table: HashMap<usize, u8> = HashMap::new();
        let mut fits = true;
        'pairs: for ((p, y), &b) in pa.iter().zip(o).zip(&ba) {
            for r in 0..y.len() {
                for c in 0..y[r].len() {
                    let key = (0..k).fold(0, |m, i| m | ((p[i][r][c] != b) as usize) << i);
                    if *table.entry(key).or_insert(y[r][c]) != y[r][c] { fits = false; break 'pairs; }
                }
            }
        }
        if !fits { continue; }
        let r: Option<Vec<Grid>> = pt.iter().zip(&bt).map(|(p, &b)| {
            let (h, w) = hw(&p[0]);
            (0..h).map(|r| (0..w).map(|c| table.get(&(0..k).fold(0, |m, i| m | ((p[i][r][c] != b) as usize) << i)).copied()).collect()).collect()
        }).collect();
        if r.is_some() { return r; }
    }
    None
}

/// repair a symmetric picture: cells of the "noise" colour take the colour of their mirror
/// images; the mirror lines are found in each grid (they need not be the grid's centre)
fn repair(g: &Grid, n: u8, crop: bool) -> Option<Grid> {
    let (h, w) = (g.len() as i64, g[0].len() as i64);
    let known = |r: i64, c: i64| r >= 0 && c >= 0 && r < h && c < w && g[r as usize][c as usize] != n;
    let nknown = g.iter().flatten().filter(|&&c| c != n).count();
    let fits = |f: &dyn Fn(i64, i64) -> (i64, i64)| -> bool {
        let (mut agree, mut dis) = (0usize, 0usize);
        for r in 0..h {
            for c in 0..w {
                if !known(r, c) { continue; }
                let (r2, c2) = f(r, c);
                if known(r2, c2) { if g[r as usize][c as usize] == g[r2 as usize][c2 as usize] { agree += 1 } else { dis += 1 } }
            }
        }
        dis == 0 && agree * 2 >= nknown
    };
    let mut maps: Vec<Box<dyn Fn(i64, i64) -> (i64, i64)>> = Vec::new();
    for s in 1..2 * w - 2 { let f = move |r: i64, c: i64| (r, s - c); if fits(&f) { maps.push(Box::new(f)); } }
    for s in 1..2 * h - 2 { let f = move |r: i64, c: i64| (s - r, c); if fits(&f) { maps.push(Box::new(f)); } }
    for d in -w..h { let f = move |r: i64, c: i64| (c + d, r - d); if fits(&f) { maps.push(Box::new(f)); } }
    for s in 0..h + w { let f = move |r: i64, c: i64| (s - c, s - r); if fits(&f) { maps.push(Box::new(f)); } }
    // a repeating pattern: a shift that maps the picture onto itself
    for dy in 0..h {
        for dx in 1 - w..w {
            if dy == 0 && dx <= 0 { continue; }
            let f = move |r: i64, c: i64| (r + dy, c + dx);
            if fits(&f) { maps.push(Box::new(f)); maps.push(Box::new(move |r: i64, c: i64| (r - dy, c - dx))); }
        }
    }
    if maps.is_empty() { return None; }
    let mut out = g.clone();
    loop {
        let mut changed = false;
        for r in 0..h {
            for c in 0..w {
                if out[r as usize][c as usize] != n { continue; }
                for f in &maps {
                    let (r2, c2) = f(r, c);
                    if r2 >= 0 && c2 >= 0 && r2 < h && c2 < w && out[r2 as usize][c2 as usize] != n {
                        out[r as usize][c as usize] = out[r2 as usize][c2 as usize];
                        changed = true;
                        break;
                    }
                }
            }
        }
        if !changed { break; }
    }
    if out.iter().flatten().any(|&c| c == n) { return None; }
    if !crop { return Some(out); }
    let cells: Vec<(usize, usize)> = (0..g.len()).flat_map(|r| (0..g[0].len()).map(move |c| (r, c))).filter(|&(r, c)| g[r][c] == n).collect();
    let (r0, r1) = (cells.iter().map(|p| p.0).min()?, cells.iter().map(|p| p.0).max()?);
    let (c0, c1) = (cells.iter().map(|p| p.1).min()?, cells.iter().map(|p| p.1).max()?);
    Some(sub(&out, r0, c0, r1, c1))
}

fn noise_repair(a: &[Grid], o: &[Grid], t: &[Grid]) -> Option<Vec<Grid>> {
    for n in 0..10u8 {
        if !a.iter().all(|g| has(g, n)) || o.iter().any(|g| has(g, n)) { continue; }
        for crop in [false, true] {
            if a.iter().zip(o).all(|(x, y)| repair(x, n, crop).as_ref() == Some(y)) {
                let r: Option<Vec<Grid>> = t.iter().map(|x| repair(x, n, crop)).collect();
                if r.is_some() { return r; }
            }
        }
    }
    None
}

/// the answer is the grid tiled a x b times, each tile turned or flipped its own way (or left
/// empty), learned per tile position
fn tile_sym(a: &[Grid], o: &[Grid], t: &[Grid]) -> Option<Vec<Grid>> {
    let ((h0, w0), (hh, ww)) = (hw(&a[0]), hw(&o[0]));
    if hh % h0 != 0 || ww % w0 != 0 { return None; }
    let (ra, rb) = (hh / h0, ww / w0);
    if ra * rb < 2 || a.iter().zip(o).any(|(x, y)| { let (h, w) = hw(x); hw(y) != (h * ra, w * rb) }) { return None; }
    let mut choice = Vec::new();
    for bi in 0..ra {
        for bj in 0..rb {
            let k = (0..9).find(|&k| a.iter().zip(o).all(|(x, y)| {
                let (h, w) = hw(x);
                let blk = sub(y, bi * h, bj * w, bi * h + h - 1, bj * w + w - 1);
                if k == 8 { let b = bg(x); return blk.iter().flatten().all(|&c| c == b); }
                let tt = d4(x, k);
                hw(&tt) == (h, w) && tt == blk
            }))?;
            choice.push(k);
        }
    }
    t.iter().map(|x| {
        let (h, w) = hw(x);
        let mut g = blank(h * ra, w * rb, bg(x));
        for (idx, &k) in choice.iter().enumerate() {
            if k == 8 { continue; }
            let (bi, bj) = (idx / rb, idx % rb);
            let tt = d4(x, k);
            if hw(&tt) != (h, w) { return None; }
            for r in 0..h { for c in 0..w { g[bi * h + r][bj * w + c] = tt[r][c]; } }
        }
        Some(g)
    }).collect()
}

/// the answer is what lies inside the box of one colour (with or without its frame)
fn crop_colour(a: &[Grid], o: &[Grid], t: &[Grid]) -> Option<Vec<Grid>> {
    for col in 0..10u8 {
        for inner in [false, true] {
            let f = |g: &Grid| -> Option<Grid> {
                let cells: Vec<(usize, usize)> = (0..g.len()).flat_map(|r| (0..g[0].len()).map(move |c| (r, c))).filter(|&(r, c)| g[r][c] == col).collect();
                let (mut r0, mut r1) = (cells.iter().map(|p| p.0).min()?, cells.iter().map(|p| p.0).max()?);
                let (mut c0, mut c1) = (cells.iter().map(|p| p.1).min()?, cells.iter().map(|p| p.1).max()?);
                if inner {
                    if r1 < r0 + 2 || c1 < c0 + 2 { return None; }
                    (r0, r1, c0, c1) = (r0 + 1, r1 - 1, c0 + 1, c1 - 1);
                }
                Some(sub(g, r0, c0, r1, c1))
            };
            if a.iter().zip(o).all(|(x, y)| f(x).as_ref() == Some(y)) {
                let r: Option<Vec<Grid>> = t.iter().map(f).collect();
                if r.is_some() { return r; }
            }
        }
    }
    None
}

fn up_ratio(a: &[Grid], o: &[Grid], t: &[Grid]) -> Option<Vec<Grid>> {
    let ((h0, w0), (hh, ww)) = (hw(&a[0]), hw(&o[0]));
    if hh % h0 != 0 || ww % w0 != 0 || hh * ww == h0 * w0 { return None; }
    let (ra, rb) = (hh / h0, ww / w0);
    if !a.iter().zip(o).all(|(x, y)| up(x, ra, rb) == *y) { return None; }
    Some(t.iter().map(|x| up(x, ra, rb)).collect())
}

fn learners() -> Vec<Learner> {
    let mut v: Vec<Learner> = Vec::new();
    let mut add = |n: &str, cost: u32, raw: bool, f: Box<dyn Fn(&[Grid], &[Grid], &[Grid]) -> Option<Vec<Grid>> + Send + Sync>| v.push((n.to_string(), cost, raw, f));
    add("colour_map", 1, false, Box::new(|a, o, t| local_rule(a, o, t, 0)));
    add("up_ratio", 1, true, Box::new(up_ratio));
    add("tile", 2, true, Box::new(tile_sym));
    add("crop_colour", 2, true, Box::new(crop_colour));
    add("halves", 2, true, Box::new(halves));
    add("repair_symmetry", 2, true, Box::new(noise_repair));
    add("recolour_objects", 2, false, Box::new(obj_recolour));
    add("rule_enclosed", 2, false, Box::new(|a, o, t| local_rule(a, o, t, 3)));
    add("rule_cross", 3, false, Box::new(|a, o, t| local_rule(a, o, t, 1)));
    add("rule_row_col", 3, false, Box::new(|a, o, t| local_rule(a, o, t, 4)));
    add("rule_count", 3, false, Box::new(|a, o, t| local_rule(a, o, t, 5)));
    add("rule_parity", 3, false, Box::new(|a, o, t| local_rule(a, o, t, 7)));
    add("rule_rays", 4, false, Box::new(|a, o, t| local_rule(a, o, t, 6)));
    add("rule_square", 4, false, Box::new(|a, o, t| local_rule(a, o, t, 2)));
    if select_on() { add("rule_select", 6, true, Box::new(rule_select)); }
    if stamp_on() { add("stamp", 4, true, Box::new(stamp)); }
    if rhythm_on() { add("rhythm", 3, true, Box::new(rhythm)); }
    if slide_on() { add("slide", 3, true, Box::new(slide)); }
    v
}


// ---------------------------------------------------------------- v30: marks around things
/// `AVA_ARC_STAMP=1`: each kind of thing leaves its own mark around it. For every object (one colour,
/// touching cells) she learns, by its colour (or colour and shape), what appears in the empty cells
/// around it, out to `pad` cells beyond its box. She learns only from cells near one object; the
/// marks must then rebuild every example exactly, and each example left out must be predicted from
/// the others (a kind never seen leaves no mark). Off by default.
fn stamp_on() -> bool { std::env::var("AVA_ARC_STAMP").map_or(false, |v| v == "1") }

type StampKey = (u8, u64);
type StampTable = HashMap<StampKey, HashMap<(i32, i32), u8>>;

/// an object's colours in the order they first appear (its parts' roles: kind 3 reads marks by role,
/// so a cross of new colours is still a cross)
fn stamp_roles(o: &Obj, x: &Grid) -> Vec<u8> {
    let mut v: Vec<u8> = Vec::new();
    for &(r, c) in &o.cells { if !v.contains(&x[r][c]) { v.push(x[r][c]); } }
    v
}

fn stamp_key(o: &Obj, kind: usize, x: &Grid) -> StampKey {
    if kind == 0 { return (o.color, 0); }
    let mut h = DefaultHasher::new();
    shape(o).hash(&mut h);
    if kind == 2 { for &(r, c) in &o.cells { x[r][c].hash(&mut h); } }
    if kind == 3 {
        let roles = stamp_roles(o, x);
        for &(r, c) in &o.cells { roles.iter().position(|&k| k == x[r][c]).hash(&mut h); }
        return (0, h.finish());
    }
    (o.color, h.finish())
}

/// for each empty cell, the objects whose surround (box + pad) covers it, with the offset
fn stamp_cover(x: &Grid, objs: &[Obj], pad: i32) -> HashMap<(usize, usize), Vec<(usize, (i32, i32))>> {
    let (h, w) = hw(x);
    let b = bg(x);
    let mut m: HashMap<(usize, usize), Vec<(usize, (i32, i32))>> = HashMap::new();
    for (i, o) in objs.iter().enumerate() {
        for r in o.r0 as i32 - pad..=o.r1 as i32 + pad {
            for c in o.c0 as i32 - pad..=o.c1 as i32 + pad {
                if r < 0 || c < 0 || r >= h as i32 || c >= w as i32 { continue; }
                let (ru, cu) = (r as usize, c as usize);
                if x[ru][cu] != b { continue; }
                m.entry((ru, cu)).or_default().push((i, (r - o.r0 as i32, c - o.c0 as i32)));
            }
        }
    }
    m
}

fn stamp_fit(a: &[Grid], o: &[Grid], kind: usize, pad: i32) -> Option<StampTable> {
    let mut t: StampTable = HashMap::new();
    for (x, y) in a.iter().zip(o) {
        let objs = objects(x, kind >= 2, true);
        for ((r, c), who) in stamp_cover(x, &objs, pad) {
            if who.len() != 1 { continue; }
            let (i, d) = who[0];
            let want = if kind == 3 { stamp_roles(&objs[i], x).iter().position(|&k| k == y[r][c]).map_or(y[r][c], |p| 10 + p as u8) } else { y[r][c] };
            let e = t.entry(stamp_key(&objs[i], kind, x)).or_default();
            match e.get(&d) { Some(&v) if v != want => return None, _ => { e.insert(d, want); } }
        }
    }
    Some(t)
}

fn stamp_apply(x: &Grid, t: &StampTable, kind: usize, pad: i32) -> Option<Grid> {
    let b = bg(x);
    let objs = objects(x, kind >= 2, true);
    let mut g = x.clone();
    for ((r, c), who) in stamp_cover(x, &objs, pad) {
        let mut mark: Option<u8> = None;
        for (i, d) in who {
            let v = t.get(&stamp_key(&objs[i], kind, x)).and_then(|e| e.get(&d)).copied().unwrap_or(b);
            let v = if v >= 10 { *stamp_roles(&objs[i], x).get(v as usize - 10)? } else { v };
            if v == b { continue; }
            if mark.map_or(false, |m| m != v) { return None; }
            mark = Some(v);
        }
        if let Some(v) = mark { g[r][c] = v; }
    }
    Some(g)
}

fn stamp(a: &[Grid], o: &[Grid], t: &[Grid]) -> Option<Vec<Grid>> {
    if a.iter().zip(o).any(|(x, y)| hw(x) != hw(y)) || a == o { return None; }
    // only empty cells may change
    for (x, y) in a.iter().zip(o) {
        let b = bg(x);
        if x.iter().zip(y).any(|(rx, ry)| rx.iter().zip(ry).any(|(&p, &q)| p != b && p != q)) { return None; }
    }
    for kind in 0..4 {
        for pad in 1..=2 {
            let Some(tab) = stamp_fit(a, o, kind, pad) else { continue };
            if !a.iter().zip(o).all(|(x, y)| stamp_apply(x, &tab, kind, pad).as_ref() == Some(y)) { continue; }
            let held = a.len() < 2 || (0..a.len()).all(|i| {
                let keep: Vec<usize> = (0..a.len()).filter(|&j| j != i).collect();
                let ak: Vec<Grid> = keep.iter().map(|&j| a[j].clone()).collect();
                let ok_: Vec<Grid> = keep.iter().map(|&j| o[j].clone()).collect();
                stamp_fit(&ak, &ok_, kind, pad).and_then(|tk| stamp_apply(&a[i], &tk, kind, pad)).as_ref() == Some(&o[i])
            });
            if !held { continue; }
            let p: Option<Vec<Grid>> = t.iter().map(|x| stamp_apply(x, &tab, kind, pad).filter(|g| g != x)).collect();
            if p.is_some() { return p; }
        }
    }
    None
}

// ---------------------------------------------------------------- v30: rhythm
/// `AVA_ARC_RHYTHM=1`: hearing the beat. She tries ways the picture could repeat (every k cells
/// along rows, columns or diagonals, or a k1 x k2 tile), takes the colour each place-in-the-beat
/// already shows (all its coloured cells must agree, and every place must have one), and fills the
/// whole grid with it. The simplest beat that rebuilds every example is used. Off by default.
fn rhythm_on() -> bool { std::env::var("AVA_ARC_RHYTHM").map_or(false, |v| v == "1") }

fn rhythm_class(kind: usize, k1: usize, k2: usize, r: usize, c: usize) -> usize {
    match kind {
        0 => (r + c) % k1,
        1 => (r + k1 * 32 - c) % k1,
        2 => r % k1,
        3 => c % k1,
        _ => (r % k1) * 16 + c % k2,
    }
}

fn rhythm_fill(x: &Grid, kind: usize, k1: usize, k2: usize) -> Option<Grid> {
    let b = bg(x);
    let mut col: HashMap<usize, u8> = HashMap::new();
    for (r, row) in x.iter().enumerate() {
        for (c, &v) in row.iter().enumerate() {
            if v == b { continue; }
            let e = col.entry(rhythm_class(kind, k1, k2, r, c)).or_insert(v);
            if *e != v { return None; }
        }
    }
    x.iter().enumerate().map(|(r, row)| (0..row.len()).map(|c| col.get(&rhythm_class(kind, k1, k2, r, c)).copied()).collect()).collect()
}

fn rhythm(a: &[Grid], o: &[Grid], t: &[Grid]) -> Option<Vec<Grid>> {
    if a.iter().zip(o).any(|(x, y)| hw(x) != hw(y)) || a == o { return None; }
    let mut beats: Vec<(usize, usize, usize, usize)> = Vec::new(); // (size, kind, k1, k2)
    for k in 2..=8 { for kind in 0..4 { beats.push((k, kind, k, 0)); } }
    for k1 in 1..=5 { for k2 in 1..=5 { if k1 * k2 >= 2 { beats.push((k1 * k2, 4, k1, k2)); } } }
    beats.sort();
    for (_, kind, k1, k2) in beats {
        if !a.iter().zip(o).all(|(x, y)| rhythm_fill(x, kind, k1, k2).as_ref() == Some(y)) { continue; }
        let p: Option<Vec<Grid>> = t.iter().map(|x| rhythm_fill(x, kind, k1, k2)).collect();
        if p.is_some() { return p; }
    }
    None
}

// ---------------------------------------------------------------- v30: sliding until contact
/// `AVA_ARC_SLIDE=1`: things slide until they touch. The cells of one colour move together, as one
/// rigid body, straight toward the cells of another colour (along the axis where the two face each
/// other) until they touch it; they may not pass through anything. Which colour moves and which
/// stays is learned from the examples. Off by default.
fn slide_on() -> bool { std::env::var("AVA_ARC_SLIDE").map_or(false, |v| v == "1") }

fn slide_one(x: &Grid, m: u8, s: u8) -> Option<Grid> {
    let (h, w) = hw(x);
    let b = bg(x);
    let cells = |k: u8| -> Vec<(i32, i32)> { (0..h).flat_map(|r| (0..w).map(move |c| (r, c))).filter(|&(r, c)| x[r][c] == k).map(|(r, c)| (r as i32, c as i32)).collect() };
    let (mv, st) = (cells(m), cells(s));
    if mv.is_empty() || st.is_empty() { return None; }
    let span = |v: &[(i32, i32)], row: bool| { let it = v.iter().map(|p| if row { p.0 } else { p.1 }); (it.clone().min().unwrap(), it.max().unwrap()) };
    let ((mr0, mr1), (mc0, mc1)) = (span(&mv, true), span(&mv, false));
    let ((sr0, sr1), (sc0, sc1)) = (span(&st, true), span(&st, false));
    let (dr, dc) = if mc0 <= sc1 && sc0 <= mc1 && (mr1 < sr0 || sr1 < mr0) { (if mr1 < sr0 { 1 } else { -1 }, 0) }
                   else if mr0 <= sr1 && sr0 <= mr1 && (mc1 < sc0 || sc1 < mc0) { (0, if mc1 < sc0 { 1 } else { -1 }) }
                   else { return None; };
    let touching = |k: i32| mv.iter().any(|&(r, c)| st.iter().any(|&(y, z)| (r + k * dr - y).abs() + (c + k * dc - z).abs() == 1));
    let free = |k: i32| mv.iter().all(|&(r, c)| {
        let (y, z) = (r + k * dr, c + k * dc);
        y >= 0 && z >= 0 && y < h as i32 && z < w as i32 && (x[y as usize][z as usize] == b || x[y as usize][z as usize] == m)
    });
    let mut k = 0;
    while !touching(k) { if !free(k + 1) { return None; } k += 1; }
    if k == 0 { return None; }
    let mut g = x.clone();
    for &(r, c) in &mv { g[r as usize][c as usize] = b; }
    for &(r, c) in &mv { g[(r + k * dr) as usize][(c + k * dc) as usize] = m; }
    Some(g)
}

fn slide(a: &[Grid], o: &[Grid], t: &[Grid]) -> Option<Vec<Grid>> {
    if a.iter().zip(o).any(|(x, y)| hw(x) != hw(y)) || a == o { return None; }
    for m in 0..10u8 {
        for s in (0..10u8).filter(|&s| s != m) {
            if !a.iter().zip(o).all(|(x, y)| slide_one(x, m, s).as_ref() == Some(y)) { continue; }
            let p: Option<Vec<Grid>> = t.iter().map(|x| slide_one(x, m, s)).collect();
            if p.is_some() { return p; }
        }
    }
    None
}
// ---------------------------------------------------------------- search
fn hash_of(a: &[Grid], b: &[Grid]) -> u64 {
    let mut h = DefaultHasher::new();
    a.hash(&mut h);
    b.hash(&mut h);
    h.finish()
}

/// every rule that reproduces all the training pairs, simplest first: (cost, name, predictions)
/// `AVA_ARC_DEPTH3=1`: chains of three operations as well as two. Off by default.
fn depth3_on() -> bool { std::env::var("AVA_ARC_DEPTH3").map_or(false, |v| v == "1") }

fn solve(task: &Task, ops: &[Op], learners: &[Learner]) -> Vec<(u32, String, Vec<Grid>)> {
    let ins: Vec<Grid> = task.train.iter().map(|p| p.0.clone()).collect();
    let outs: Vec<Grid> = task.train.iter().map(|p| p.1.clone()).collect();
    let tins: Vec<Grid> = task.test.iter().map(|p| p.0.clone()).collect();
    // chains of up to 2 operations; a chain that gives the same grids as a simpler one is dropped
    let mut seen: HashSet<u64> = HashSet::new();
    seen.insert(hash_of(&ins, &tins));
    let mut all: Vec<(String, u32, Vec<Grid>, Vec<Grid>)> = vec![("input".into(), 0, ins.clone(), tins.clone())];
    let mut level = all.clone();
    for _ in 0..2 {
        let mut next = Vec::new();
        for (name, cost, a, b) in &level {
            for (on, f) in ops {
                let Some(a2) = a.iter().map(|g| f(g).filter(ok)).collect::<Option<Vec<Grid>>>() else { continue };
                let Some(b2) = b.iter().map(|g| f(g).filter(ok)).collect::<Option<Vec<Grid>>>() else { continue };
                if !seen.insert(hash_of(&a2, &b2)) { continue; }
                next.push((format!("{name} > {on}"), cost + 1, a2, b2));
            }
        }
        all.extend(next.iter().cloned());
        level = next;
    }
    let mut progs: Vec<(u32, String, Vec<Grid>)> = Vec::new();
    for (name, cost, a, b) in &all {
        if *a == outs { progs.push((*cost, name.clone(), b.clone())); }
    }
    if depth3_on() {
        // a third step, kept only when it rebuilds every example exactly (checked one example at a time)
        for (name, cost, a, b) in &level {
            for (on, f) in ops {
                if !a.iter().zip(&outs).all(|(g, y)| f(g).map_or(false, |x| x == *y)) { continue; }
                let Some(b2) = b.iter().map(|g| f(g).filter(ok)).collect::<Option<Vec<Grid>>>() else { continue };
                progs.push((cost + 1, format!("{name} > {on}"), b2));
            }
        }
    }
    for (name, cost, a, b) in &all {
        for (ln, lc, raw, fit) in learners {
            if (*raw && *cost > 0) || (*cost > 1 && ln != "colour_map") { continue; }
            if let Some(p) = fit(a, &outs, b) {
                if p.iter().all(ok) { progs.push((cost + lc, format!("{name} > {ln}"), p)); }
            }
        }
    }
    progs.sort_by_key(|p| p.0); // stable: within a cost, the order they were tried
    progs
}

struct Res { name: String, n_test: usize, right: Vec<bool>, answered: bool, first_right: bool, used: Vec<String> }

fn run_task(task: &Task, ops: &[Op], learners: &[Learner]) -> Res {
    let progs = solve(task, ops, learners);
    let mut attempts: Vec<(String, Vec<Grid>)> = Vec::new();
    for (_, name, p) in progs {
        if attempts.iter().any(|(_, q)| *q == p) { continue; }
        attempts.push((name, p));
        if attempts.len() == 2 { break; }
    }
    let right: Vec<bool> = (0..task.test.len()).map(|i| attempts.iter().any(|(_, p)| p[i] == task.test[i].1)).collect();
    let first_right = attempts.first().map_or(false, |(_, p)| (0..task.test.len()).all(|i| p[i] == task.test[i].1));
    Res { name: task.name.clone(), n_test: task.test.len(), right, answered: !attempts.is_empty(), first_right, used: attempts.into_iter().map(|a| a.0).collect() }
}

// ---------------------------------------------------------------- the few-shot stress test (v31 A0)
/// `AVA_ARC_HOLDOUT=k` (k = 1 or 2): fewer examples, on practice data. In every puzzle with at least
/// k + 2 example pairs, the LAST k pairs are taken away from the examples and scored as extra tests
/// beside the real ones. She sees only their inputs, as with any test; their outputs are used for
/// scoring alone. k + 2 leaves at least 2 examples to learn from, the fewest any ARC puzzle gives.
/// A puzzle with fewer pairs is skipped and left out of every count. It measures how honest her
/// "it fits every example" check stays when examples are few (C1, ConceptARC, found it was not).
/// This measures; it adds no thinking. Off by default: only 1 and 2 turn it on (the plan's two
/// settings); any other value, 0 or not a number means off, and the header then shows a normal run.
fn holdout_k() -> usize { std::env::var("AVA_ARC_HOLDOUT").ok().and_then(|v| v.parse::<usize>().ok()).filter(|k| (1..=2).contains(k)).unwrap_or(0) }

/// the puzzle with its last k example pairs moved behind its real tests; None when it has too few
fn hold_out(mut t: Task, k: usize) -> Option<Task> {
    if t.train.len() < k + 2 { return None; }
    let hidden = t.train.split_off(t.train.len() - k);
    t.test.extend(hidden);
    Some(t)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let set = args.get(1).map(String::as_str).unwrap_or("training");
    let dir = format!("{DATA}/{set}");
    let mut files: Vec<_> = std::fs::read_dir(&dir).expect("read the ARC folder").filter_map(|e| e.ok()).map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |x| x == "json")).collect();
    files.sort();
    let mut tasks: Vec<Task> = files.iter().map(|p| load(p)).collect();
    // the stress test: hide the last k example pairs; n_real[i] = how many real tests puzzle i has
    let k = holdout_k();
    let (mut skipped, mut n_real) = (0usize, Vec::new());
    if k > 0 {
        let before = tasks.len();
        tasks = tasks.into_iter().filter_map(|t| { let nr = t.test.len(); hold_out(t, k).map(|t| { n_real.push(nr); t }) }).collect();
        skipped = before - tasks.len();
    }
    let threads: usize = std::env::var("AVA_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(4).max(1);
    let (ops, learners) = (ops(), learners());
    if k == 0 {
        println!("ARC-AGI-1 {set}: {} puzzles; {} operations, {} learned steps; {threads} threads", tasks.len(), ops.len(), learners.len());
    } else {
        println!("ARC-AGI-1 {set} HOLDOUT {k} (stress test: the last {k} example pair(s) of each puzzle hidden and scored as tests): {} puzzles, {skipped} skipped (fewer than {} example pairs); {} operations, {} learned steps; {threads} threads",
                 tasks.len(), k + 2, ops.len(), learners.len());
    }
    let t0 = Instant::now();
    let chunk = tasks.len().div_ceil(threads);
    let res: Vec<Res> = std::thread::scope(|s| {
        let hs: Vec<_> = tasks.chunks(chunk.max(1)).map(|ch| { let (ops, learners) = (&ops, &learners); s.spawn(move || ch.iter().map(|t| run_task(t, ops, learners)).collect::<Vec<_>>()) }).collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let n = res.len();
    let full = res.iter().filter(|r| r.right.iter().all(|&x| x)).count();
    let first = res.iter().filter(|r| r.first_right).count();
    let answered = res.iter().filter(|r| r.answered).count();
    let per_output: f64 = res.iter().map(|r| r.right.iter().filter(|&&x| x).count() as f64 / r.n_test as f64).sum();
    println!("  {:.1} s", t0.elapsed().as_secs_f64());
    println!("  fully right (2 attempts): {full} of {n} ({:.2}%)   first attempt alone: {first}   per-test-output score: {per_output:.2}", 100.0 * full as f64 / n as f64);
    println!("  answered {answered}: right {full}, wrong {}; abstained {}; share right among answered {:.1}%",
             answered - full, n - answered, 100.0 * full as f64 / answered.max(1) as f64);
    if k > 0 {
        // the wrong answers, split by how many of their test outputs were right and by which tests
        // they missed: the hidden example pairs, the real tests, or both
        let mut split: std::collections::BTreeMap<(usize, usize), usize> = std::collections::BTreeMap::new();
        let (mut held_only, mut real_only, mut both) = (0usize, 0usize, 0usize);
        let (mut rr, mut rn, mut hr, mut hn) = (0usize, 0usize, 0usize, 0usize);
        for (r, &nr) in res.iter().zip(&n_real) {
            if !r.answered { continue; }
            let real_ok = r.right[..nr].iter().filter(|&&x| x).count();
            let held_ok = r.right[nr..].iter().filter(|&&x| x).count();
            (rr, rn, hr, hn) = (rr + real_ok, rn + nr, hr + held_ok, hn + r.n_test - nr);
            if r.right.iter().all(|&x| x) { continue; }
            *split.entry((real_ok + held_ok, r.n_test)).or_default() += 1;
            match (real_ok == nr, held_ok == r.n_test - nr) { (true, _) => held_only += 1, (_, true) => real_only += 1, _ => both += 1 }
        }
        let none_right: usize = split.iter().filter(|((g, _), _)| *g == 0).map(|(_, v)| v).sum();
        println!("  HOLDOUT {k}: wrong answers by test outputs right: none right {none_right}, some right {} [{}]",
                 answered - full - none_right, split.iter().map(|((g, t), v)| format!("{g}/{t} {v}")).collect::<Vec<_>>().join(", "));
        println!("  HOLDOUT {k}: wrong answers that missed: hidden pairs only {held_only}, real tests only {real_only}, both {both}");
        println!("  HOLDOUT {k}: among answered puzzles, real test outputs right {rr} of {rn}, hidden-pair outputs right {hr} of {hn}");
    }
    let mut fam: HashMap<String, usize> = HashMap::new();
    for r in res.iter().filter(|r| r.right.iter().all(|&x| x)) {
        let winner = r.used.iter().position(|_| true).map(|_| r.used[0].clone()).unwrap_or_default();
        let last = winner.rsplit(" > ").next().unwrap_or("").to_string();
        *fam.entry(last).or_default() += 1;
    }
    let mut fv: Vec<_> = fam.into_iter().collect();
    fv.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    println!("  solved, by last step of the first attempt: {}", fv.iter().map(|(k, v)| format!("{k} {v}")).collect::<Vec<_>>().join(", "));
    if let Some(path) = args.get(2) {
        let mut out = String::from("task\tstatus\ttests_right\tattempts\n");
        for r in &res {
            let st = if r.right.iter().all(|&x| x) { "right" } else if r.answered { "wrong" } else { "abstain" };
            out.push_str(&format!("{}\t{st}\t{}/{}\t{}\n", r.name, r.right.iter().filter(|&&x| x).count(), r.n_test, r.used.join(" | ")));
        }
        std::fs::write(path, out).expect("write the per-puzzle file");
        println!("  per-puzzle results: {path}");
    }
}
