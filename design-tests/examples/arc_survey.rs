// TESTING / DESIGN ONLY. (v29 WP12b) Survey of the ARC training puzzles she abstains on: what kind of change each asks for.
//   cargo run --release --example arc_survey -- <per-task.tsv from arc.rs>
use std::collections::HashMap;
type Grid = Vec<Vec<u8>>;
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

fn hw(g: &Grid) -> (usize, usize) { (g.len(), g.first().map_or(0, |r| r.len())) }
fn hist(g: &Grid) -> [usize; 10] { let mut n = [0; 10]; for r in g { for &c in r { n[c as usize] += 1; } } n }

/// is `small` a sub-rectangle of `big`?
fn is_crop(big: &Grid, small: &Grid) -> bool {
    let ((bh, bw), (sh, sw)) = (hw(big), hw(small));
    if sh > bh || sw > bw { return false; }
    (0..=bh - sh).any(|r| (0..=bw - sw).any(|c| (0..sh).all(|i| big[r + i][c..c + sw] == small[i][..])))
}

fn kind(t: &Task) -> String {
    let same = t.train.iter().all(|(a, b)| hw(a) == hw(b));
    let smaller = t.train.iter().all(|(a, b)| { let ((ah, aw), (bh, bw)) = (hw(a), hw(b)); bh * bw < ah * aw });
    let bigger = t.train.iter().all(|(a, b)| { let ((ah, aw), (bh, bw)) = (hw(a), hw(b)); bh * bw > ah * aw });
    if same {
        // what happens to the changed cells (background = 0)
        let (mut add, mut del, mut rec, mut n, mut tot) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let mut moves = true;
        for (a, b) in &t.train {
            let (h, w) = hw(a);
            tot += h * w;
            for r in 0..h { for c in 0..w {
                let (x, y) = (a[r][c], b[r][c]);
                if x == y { continue; }
                n += 1;
                if x == 0 { add += 1 } else if y == 0 { del += 1 } else { rec += 1 }
            } }
            if hist(a) != hist(b) { moves = false; }
        }
        let frac = n as f64 / tot.max(1) as f64;
        let what = if moves && n > 0 { "move (same colours)" } else if add == n { "draw (adds colour on background)" }
            else if del == n { "erase" } else if rec == n { "recolour" } else if add > 0 && del > 0 { "move/redraw (adds and erases)" }
            else if add > 0 { "draw + recolour" } else { "erase + recolour" };
        let band = if frac < 0.05 { "<5% cells" } else if frac < 0.2 { "5-20% cells" } else { ">20% cells" };
        format!("same size | {what} | {band}")
    } else if smaller {
        if t.train.iter().all(|(a, b)| is_crop(a, b)) { "smaller | a crop of the input".into() } else { "smaller | not a crop".into() }
    } else if bigger { "bigger".into() } else { "size varies".into() }
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.get(1).map(String::as_str) == Some("compare") { compare(&a[2], &a[3]); return; }
    if a.get(1).map(String::as_str) == Some("needs") { needs(&a[2]); return; }
    if a.get(1).map(String::as_str) == Some("show") { for n in &a[2..] { show(n); } return; }
    let tsv = std::fs::read_to_string(a.get(1).expect("per-task tsv")).expect("read tsv");
    let status: HashMap<String, String> = tsv.lines().skip(1).filter_map(|l| { let p: Vec<&str> = l.split('\t').collect(); (p.len() > 1).then(|| (p[0].to_string(), p[1].to_string())) }).collect();
    let mut k: HashMap<(String, String), usize> = HashMap::new();
    let mut files: Vec<_> = std::fs::read_dir(format!("{DATA}/training")).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).collect();
    files.sort();
    for p in files {
        let t = load(&p);
        let st = status.get(&t.name).cloned().unwrap_or_default();
        *k.entry((kind(&t), st)).or_default() += 1;
    }
    let mut v: Vec<_> = k.into_iter().collect();
    v.sort_by(|x, y| x.0.cmp(&y.0));
    let mut by: HashMap<String, [usize; 3]> = HashMap::new();
    for ((kd, st), n) in &v { let e = by.entry(kd.clone()).or_default(); match st.as_str() { "right" => e[0] += n, "abstain" => e[1] += n, _ => e[2] += n } }
    let mut bv: Vec<_> = by.into_iter().collect();
    bv.sort_by(|x, y| (y.1[1]).cmp(&x.1[1]));
    println!("{:<52} {:>6} {:>8} {:>6}", "kind of puzzle (training)", "right", "abstain", "wrong");
    for (kd, [r, ab, w]) in bv { println!("{kd:<52} {r:>6} {ab:>8} {w:>6}"); }
}

// ---------------------------------------------------------------- compare two per-puzzle files
// arc_survey compare <base.tsv> <new.tsv>: per puzzle +1 right, -1 wrong, 0 abstain; the paired
// difference with a bootstrap 95% CI (splitmix64 seed 1, 20,000 resamples)
pub fn compare(base: &str, new: &str) {
    let read = |p: &str| -> HashMap<String, i64> {
        std::fs::read_to_string(p).expect("read tsv").lines().skip(1).filter_map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            (f.len() > 1).then(|| (f[0].to_string(), match f[1] { "right" => 1, "wrong" => -1, _ => 0 }))
        }).collect()
    };
    let (a, b) = (read(base), read(new));
    let mut names: Vec<&String> = a.keys().collect();
    names.sort();
    let d: Vec<i64> = names.iter().map(|k| b.get(*k).copied().unwrap_or(0) - a[*k]).collect();
    let mut s = 1u64;
    let mut next = || { s = s.wrapping_add(0x9E3779B97F4A7C15); let mut z = s; z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB); z ^ (z >> 31) };
    let n = d.len();
    let mut v: Vec<i64> = (0..20_000).map(|_| (0..n).map(|_| d[(next() % n as u64) as usize]).sum()).collect();
    v.sort();
    let gained = d.iter().filter(|&&x| x > 0).count();
    let lost = d.iter().filter(|&&x| x < 0).count();
    println!("puzzles {n}: right-wrong base {:+}, new {:+}; difference {:+}  95% CI {:+} .. {:+}  (puzzles better {gained}, worse {lost})",
             a.values().sum::<i64>(), names.iter().map(|k| b.get(*k).copied().unwrap_or(0)).sum::<i64>(), d.iter().sum::<i64>(), v[499], v[19_499]);
    // v31 A0: each file's own counts, and which puzzles are right in one file but not the other.
    // The first line above stays as it was, so older logs stay comparable.
    let rows = |p: &str| -> Vec<(String, String, usize)> {
        // (puzzle, status, test outputs right); tests_right is "right/total"
        std::fs::read_to_string(p).expect("read tsv").lines().skip(1).filter_map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            let got = f.get(2).and_then(|x| x.split('/').next()).and_then(|x| x.parse().ok()).unwrap_or(0);
            (f.len() > 1).then(|| (f[0].to_string(), f[1].to_string(), got))
        }).collect()
    };
    let (ra, rb) = (rows(base), rows(new));
    for (label, r) in [("base", &ra), ("new", &rb)] {
        let count = |s: &str| r.iter().filter(|x| x.1 == s).count();
        let (right, wrong) = (count("right"), count("wrong"));
        let part = r.iter().filter(|x| x.1 == "wrong" && x.2 > 0).count();
        let answered = right + wrong;
        let share = if answered == 0 { "n/a".to_string() } else { format!("{:.1}%", 100.0 * right as f64 / answered as f64) };
        println!("  {label} ({} puzzles): right {right}, wrong {wrong} (some test outputs right {part}, none right {}), abstained {}; share right among answered {share}",
                 r.len(), wrong - part, r.len() - answered);
    }
    let right_set = |r: &[(String, String, usize)]| -> Vec<String> { let mut v: Vec<String> = r.iter().filter(|x| x.1 == "right").map(|x| x.0.clone()).collect(); v.sort(); v };
    let (sa, sb) = (right_set(&ra), right_set(&rb));
    let only = |x: &[String], y: &[String]| -> Vec<String> { x.iter().filter(|n| !y.contains(n)).cloned().collect() };
    for (label, v) in [("base only", only(&sa, &sb)), ("new only", only(&sb, &sa))] {
        println!("  right in {label} ({}): {}", v.len(), if v.is_empty() { "-".to_string() } else { v.join(" ") });
    }
    // a warning when the two files do not cover the same puzzles (a stress run against a normal run)
    let (na, nb): (Vec<&String>, Vec<&String>) = (ra.iter().map(|x| &x.0).collect(), rb.iter().map(|x| &x.0).collect());
    let (oa, ob) = (na.iter().filter(|x| !nb.contains(x)).count(), nb.iter().filter(|x| !na.contains(x)).count());
    if oa + ob > 0 { println!("  note: the files cover different puzzles ({oa} only in base, {ob} only in new); the first line counts base's puzzles"); }
}

// ---------------------------------------------------------------- v30 recon: what each unsolved puzzle asks for
fn comps(g: &Grid, mask: &dyn Fn(usize, usize) -> bool) -> Vec<Vec<(usize, usize)>> {
    let (h, w) = hw(g);
    let mut seen = vec![vec![false; w]; h];
    let mut out = Vec::new();
    for r in 0..h { for c in 0..w {
        if seen[r][c] || !mask(r, c) { continue; }
        let mut st = vec![(r, c)]; seen[r][c] = true; let mut cells = Vec::new();
        while let Some((y, x)) = st.pop() {
            cells.push((y, x));
            for (dy, dx) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] {
                let (ny, nx) = (y as i32 + dy, x as i32 + dx);
                if ny < 0 || nx < 0 || ny >= h as i32 || nx >= w as i32 { continue; }
                let (ny, nx) = (ny as usize, nx as usize);
                if !seen[ny][nx] && mask(ny, nx) && g[ny][nx] == g[y][x] { seen[ny][nx] = true; st.push((ny, nx)); }
            }
        }
        out.push(cells);
    } }
    out
}
fn shape_of(cells: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let r0 = cells.iter().map(|p| p.0).min().unwrap(); let c0 = cells.iter().map(|p| p.1).min().unwrap();
    let mut s: Vec<(usize, usize)> = cells.iter().map(|&(r, c)| (r - r0, c - c0)).collect(); s.sort(); s
}
/// one tag per puzzle: the most specific description that fits every training pair
fn need(t: &Task) -> &'static str {
    let same = t.train.iter().all(|(a, b)| hw(a) == hw(b));
    if same {
        let (mut add, mut n, mut on_line, mut copied, mut whole_recol, mut recol) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let mut moved = true;
        for (a, b) in &t.train {
            let (h, w) = hw(a);
            if hist(a) != hist(b) { moved = false; }
            for r in 0..h { for c in 0..w {
                if a[r][c] == b[r][c] { continue; }
                n += 1;
                if a[r][c] == 0 {
                    add += 1;
                    // on a straight line (row, column or diagonal) through an input cell of the new colour
                    let col = b[r][c];
                    let line = (0..w).any(|x| a[r][x] == col) || (0..h).any(|y| a[y][c] == col)
                        || (-30i32..30).any(|k| { let (y, x) = (r as i32 + k, c as i32 + k); y >= 0 && x >= 0 && y < h as i32 && x < w as i32 && k != 0 && a[y as usize][x as usize] == col })
                        || (-30i32..30).any(|k| { let (y, x) = (r as i32 + k, c as i32 - k); y >= 0 && x >= 0 && y < h as i32 && x < w as i32 && k != 0 && a[y as usize][x as usize] == col });
                    if line { on_line += 1; }
                } else { recol += 1; }
            } }
            // added shapes that copy an input shape
            let in_shapes: Vec<_> = comps(a, &|r, c| a[r][c] != 0).iter().map(|x| shape_of(x)).collect();
            for cp in comps(b, &|r, c| a[r][c] == 0 && b[r][c] != 0) { if cp.len() >= 3 && in_shapes.contains(&shape_of(&cp)) { copied += cp.len(); } }
            // recolours that change whole objects
            for cp in comps(a, &|r, c| a[r][c] != 0) {
                let cols: std::collections::HashSet<u8> = cp.iter().map(|&(r, c)| b[r][c]).collect();
                if cols.len() == 1 && b[cp[0].0][cp[0].1] != a[cp[0].0][cp[0].1] { whole_recol += cp.len(); }
            }
        }
        if n == 0 { return "same size: identity?"; }
        if moved { return "same size: objects move (same colours)"; }
        if copied * 2 >= add.max(1) && add * 2 >= n { return "same size: copy / stamp a shape"; }
        if add * 10 >= n * 9 && on_line * 10 >= add * 9 { return "same size: draw lines from/between cells"; }
        if add * 10 >= n * 9 { return "same size: draw other shapes / fill"; }
        if recol * 10 >= n * 9 && whole_recol * 10 >= recol * 9 { return "same size: recolour whole objects (by a property)"; }
        if recol * 10 >= n * 9 { return "same size: recolour parts"; }
        return "same size: mixed edit (add + erase/recolour)";
    }
    let smaller = t.train.iter().all(|(a, b)| { let ((ah, aw), (bh, bw)) = (hw(a), hw(b)); bh * bw < ah * aw });
    if smaller {
        if t.train.iter().all(|(a, b)| is_crop(a, b)) { return "smaller: a crop (which part?)"; }
        if t.train.iter().all(|(_, b)| { let (h, w) = hw(b); h * w <= 9 || h == 1 || w == 1 }) { return "smaller: a tiny answer (count / pick a colour)"; }
        return "smaller: summarise / combine parts";
    }
    let bigger = t.train.iter().all(|(a, b)| { let ((ah, aw), (bh, bw)) = (hw(a), hw(b)); bh * bw > ah * aw });
    if bigger {
        if t.train.iter().all(|(a, b)| { let ((ah, aw), (bh, bw)) = (hw(a), hw(b)); bh % ah == 0 && bw % aw == 0 }) { return "bigger: whole multiple (tile / scale)"; }
        return "bigger: grow / extend";
    }
    "size varies by example"
}

pub fn needs(tsv: &str) {
    let status: HashMap<String, String> = std::fs::read_to_string(tsv).expect("tsv").lines().skip(1).filter_map(|l| { let p: Vec<&str> = l.split('\t').collect(); (p.len() > 1).then(|| (p[0].to_string(), p[1].to_string())) }).collect();
    let mut files: Vec<_> = std::fs::read_dir(format!("{DATA}/training")).unwrap().filter_map(|e| e.ok()).map(|e| e.path()).collect();
    files.sort();
    let mut k: HashMap<&str, (usize, usize, Vec<String>)> = HashMap::new();
    for p in files {
        let t = load(&p);
        let e = k.entry(need(&t)).or_default();
        if status.get(&t.name).map(String::as_str) == Some("right") { e.0 += 1; } else { e.1 += 1; if e.2.len() < 4 { e.2.push(t.name.clone()); } }
    }
    let mut v: Vec<_> = k.into_iter().collect();
    v.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
    println!("{:<52} {:>6} {:>9}  examples not solved", "what the puzzle asks (training)", "solved", "unsolved");
    for (kd, (s, u, ex)) in v { println!("{kd:<52} {s:>6} {u:>9}  {}", ex.join(" ")); }
}

pub fn show(name: &str) {
    let t = load(std::path::Path::new(&format!("{DATA}/training/{name}.json")));
    for (i, (a, b)) in t.train.iter().enumerate().take(2) {
        println!("{name} pair {i}: {}x{} -> {}x{}", a.len(), a[0].len(), b.len(), b[0].len());
        let rows = a.len().max(b.len());
        for r in 0..rows {
            let f = |g: &Grid| if r < g.len() { g[r].iter().map(|&c| if c == 0 { '.' } else { (b'0' + c) as char }).collect::<String>() } else { " ".repeat(g[0].len()) };
            println!("  {}   {}", f(a), f(b));
        }
    }
}
