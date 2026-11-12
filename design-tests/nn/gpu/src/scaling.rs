// TESTING / DESIGN ONLY — scaling (v42): is one family's BPB curve steeper or flatter than another's, under more
// than one curve shape, with honest intervals?
//
// v34's gap rule fits BPB = a + b·ln N per family and reads the ground change G = (b_A − b_B)·ln(N_hi/N_lo) against
// 2σ of the d128 seed spread. A straight line in ln N has no floor, so it can turn a flattening curve into a "losing
// ground" verdict (or hide one). This module checks the same question under a second form,
//   BPB = E + A·N^−α   (Chinchilla-style, one floor E shared by both families: the same data has one entropy),
// and puts bootstrap intervals on the gap under each form (Besiroglu et al. 2024: fits from a few points need them):
//   * over seeds: every (family, size) cell is re-averaged from its seeds drawn with replacement; a cell with one seed
//     gets a draw from the pooled seed deviations of the multi-seed cells;
//   * over residuals: the fitted curve plus residuals drawn with replacement (pooled over both families, scaled up by
//     sqrt(n / (n − p)) for the fitted parameters).
// Fits use cell means (one point per family and size). The power form is fitted by profiling: for a given floor E and
// exponent α the best A is closed-form least squares; α is found per family by a log grid then golden section, and E
// by a grid then golden section on [0, min BPB).

use ava_data::rng::SplitMix64;

/// One measured run.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub family: String,
    /// size: weights, FLOPs per token, or anything positive (fits use ln x)
    pub x: f64,
    pub bpb: f64,
    pub seed: u64,
}

/// The runs of one family at one size.
#[derive(Clone, Debug, PartialEq)]
pub struct Cell {
    pub x: f64,
    pub ys: Vec<f64>,
}

impl Cell {
    pub fn mean(&self) -> f64 {
        self.ys.iter().sum::<f64>() / self.ys.len() as f64
    }
}

/// A family's cells, smallest size first.
pub fn cells(rows: &[Row], family: &str) -> Vec<Cell> {
    let mut out: Vec<Cell> = Vec::new();
    for r in rows.iter().filter(|r| r.family == family) {
        match out.iter_mut().find(|c| c.x == r.x) {
            Some(c) => c.ys.push(r.bpb),
            None => out.push(Cell { x: r.x, ys: vec![r.bpb] }),
        }
    }
    out.sort_by(|a, b| a.x.total_cmp(&b.x));
    out
}

/// The pooled seed spread: every seed's deviation from its cell mean, times sqrt(k / (k − 1)) for a cell of k seeds
/// (so the deviations have the spread of single runs), over every multi-seed cell of every family in `rows`.
pub fn seed_deviations(rows: &[Row]) -> Vec<f64> {
    let mut fams: Vec<&str> = rows.iter().map(|r| r.family.as_str()).collect();
    fams.sort();
    fams.dedup();
    let mut out = Vec::new();
    for f in fams {
        for c in cells(rows, f).iter().filter(|c| c.ys.len() > 1) {
            let (m, k) = (c.mean(), c.ys.len() as f64);
            out.extend(c.ys.iter().map(|y| (y - m) * (k / (k - 1.0)).sqrt()));
        }
    }
    out
}

/// The pooled seed standard deviation (the v34 rule's σ): sqrt of the pooled within-cell variance.
pub fn seed_sd(rows: &[Row]) -> Option<f64> {
    let d = seed_deviations(rows);
    if d.is_empty() {
        return None;
    }
    // the deviations already carry the k/(k−1) correction, so their mean square is the pooled variance
    Some((d.iter().map(|x| x * x).sum::<f64>() / d.len() as f64).sqrt())
}

/// The two curve shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// BPB = a + b·ln x, per family
    LogLinear,
    /// BPB = E + A·x^−α, E shared by both families
    PowerFloor,
}

impl Form {
    pub fn name(self) -> &'static str {
        match self {
            Form::LogLinear => "log-linear  a + b ln N",
            Form::PowerFloor => "power law   E + A N^-alpha (shared E)",
        }
    }

    /// Fitted parameters for both families together.
    pub fn params(self) -> usize {
        match self {
            Form::LogLinear => 4,
            Form::PowerFloor => 5,
        }
    }
}

/// One family's fitted curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Curve {
    Lin { a: f64, b: f64 },
    Pow { e: f64, amp: f64, alpha: f64 },
}

impl Curve {
    pub fn at(&self, x: f64) -> f64 {
        match *self {
            Curve::Lin { a, b } => a + b * x.ln(),
            Curve::Pow { e, amp, alpha } => e + amp * x.powf(-alpha),
        }
    }

    /// d BPB / d ln x at x (the local slope per e-fold of size).
    pub fn slope(&self, x: f64) -> f64 {
        match *self {
            Curve::Lin { b, .. } => b,
            Curve::Pow { amp, alpha, .. } => -alpha * amp * x.powf(-alpha),
        }
    }
}

/// Both families' curves under one form.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pair {
    pub a: Curve,
    pub b: Curve,
}

impl Pair {
    /// gap(x) = a(x) − b(x): positive when family a is worse (higher BPB).
    pub fn gap(&self, x: f64) -> f64 {
        self.a.at(x) - self.b.at(x)
    }

    /// Ground change from x_lo to x_hi: gap(x_hi) − gap(x_lo). Positive = a loses ground.
    pub fn change(&self, x_lo: f64, x_hi: f64) -> f64 {
        self.gap(x_hi) - self.gap(x_lo)
    }

    /// Where the gap is zero, searched in [lo, hi] (the first sign change on a log grid, then bisection).
    pub fn crossing(&self, lo: f64, hi: f64) -> Option<f64> {
        let (l0, l1) = (lo.ln(), hi.ln());
        let steps = 400;
        let g = |t: f64| self.gap(t.exp());
        let mut prev = (l0, g(l0));
        for i in 1..=steps {
            let t = l0 + (l1 - l0) * i as f64 / steps as f64;
            let v = g(t);
            if prev.1 == 0.0 {
                return Some(prev.0.exp());
            }
            if prev.1.signum() != v.signum() {
                let (mut a, mut b) = (prev.0, t);
                for _ in 0..100 {
                    let m = 0.5 * (a + b);
                    if g(m).signum() == g(a).signum() {
                        a = m;
                    } else {
                        b = m;
                    }
                }
                return Some((0.5 * (a + b)).exp());
            }
            prev = (t, v);
        }
        None
    }
}

/// Least squares a + b·ln x.
pub fn fit_loglinear(xs: &[f64], ys: &[f64]) -> Curve {
    let n = xs.len() as f64;
    let lx: Vec<f64> = xs.iter().map(|x| x.ln()).collect();
    let (mx, my) = (lx.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let sxy: f64 = lx.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = lx.iter().map(|x| (x - mx) * (x - mx)).sum();
    let b = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    Curve::Lin { a: my - b * mx, b }
}

/// For a floor e and exponent alpha: the best amplitude (closed form) and the squared error.
fn pow_amp_sse(xs: &[f64], ys: &[f64], e: f64, alpha: f64) -> (f64, f64) {
    let z: Vec<f64> = xs.iter().map(|x| x.powf(-alpha)).collect();
    let szz: f64 = z.iter().map(|v| v * v).sum();
    let szy: f64 = z.iter().zip(ys).map(|(v, y)| v * (y - e)).sum();
    let amp = (szy / szz).max(0.0);
    let sse = z.iter().zip(ys).map(|(v, y)| (y - e - amp * v).powi(2)).sum();
    (amp, sse)
}

/// Minimise f on [lo, hi]: the best of `grid` log-spaced points (or evenly spaced if `log` is false), then golden
/// section between its neighbours. Returns (argmin, min).
fn minimise(f: &dyn Fn(f64) -> f64, lo: f64, hi: f64, grid: usize, log: bool) -> (f64, f64) {
    let at = |i: usize| {
        let t = i as f64 / (grid - 1) as f64;
        if log {
            (lo.ln() + (hi.ln() - lo.ln()) * t).exp()
        } else {
            lo + (hi - lo) * t
        }
    };
    let (mut bi, mut bv) = (0usize, f64::INFINITY);
    for i in 0..grid {
        let v = f(at(i));
        if v < bv {
            bi = i;
            bv = v;
        }
    }
    let (mut a, mut b) = (at(bi.saturating_sub(1)), at((bi + 1).min(grid - 1)));
    let r = (5f64.sqrt() - 1.0) / 2.0;
    let (mut c, mut d) = (b - r * (b - a), a + r * (b - a));
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..60 {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - r * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + r * (b - a);
            fd = f(d);
        }
    }
    let (x, v) = if fc < fd { (c, fc) } else { (d, fd) };
    if v <= bv {
        (x, v)
    } else {
        (at(bi), bv)
    }
}

/// The best exponent for one family at floor e: (alpha, amp, sse).
fn fit_pow_at(xs: &[f64], ys: &[f64], e: f64) -> (f64, f64, f64) {
    let f = |alpha: f64| pow_amp_sse(xs, ys, e, alpha).1;
    let (alpha, sse) = minimise(&f, 1e-3, 3.0, 48, true);
    (alpha, pow_amp_sse(xs, ys, e, alpha).0, sse)
}

/// E + A·x^−α for two families with one shared floor E in [0, min y). Returns the pair and the total squared error.
pub fn fit_power_shared(xa: &[f64], ya: &[f64], xb: &[f64], yb: &[f64]) -> (Pair, f64) {
    let ymin = ya.iter().chain(yb).copied().fold(f64::INFINITY, f64::min);
    let top = (ymin - 1e-6).max(0.0);
    let total = |e: f64| fit_pow_at(xa, ya, e).2 + fit_pow_at(xb, yb, e).2;
    let (e, sse) = minimise(&total, 0.0, top, 41, false);
    let (aa, ma, _) = fit_pow_at(xa, ya, e);
    let (ab, mb, _) = fit_pow_at(xb, yb, e);
    (Pair { a: Curve::Pow { e, amp: ma, alpha: aa }, b: Curve::Pow { e, amp: mb, alpha: ab } }, sse)
}

/// Fit both families' cell means under `form`. Returns the pair and the residuals (family a's cells, then b's).
pub fn fit(form: Form, xa: &[f64], ya: &[f64], xb: &[f64], yb: &[f64]) -> (Pair, Vec<f64>) {
    let pair = match form {
        Form::LogLinear => Pair { a: fit_loglinear(xa, ya), b: fit_loglinear(xb, yb) },
        Form::PowerFloor => fit_power_shared(xa, ya, xb, yb).0,
    };
    let res = xa.iter().zip(ya).map(|(&x, &y)| y - pair.a.at(x)).chain(xb.iter().zip(yb).map(|(&x, &y)| y - pair.b.at(x))).collect();
    (pair, res)
}

/// What the analysis reports for one form.
#[derive(Clone, Debug)]
pub struct Report {
    pub form: Form,
    pub pair: Pair,
    /// root mean square residual of the cell means
    pub rms: f64,
    /// ground change from x_lo to x_hi (a's smallest and largest sizes), and the gap at each target
    pub change: f64,
    pub gaps: Vec<f64>,
    pub crossing: Option<f64>,
    /// 95% percentile intervals: (change, gaps per target), by the seed bootstrap and by the residual bootstrap
    pub seed_ci: Option<((f64, f64), Vec<(f64, f64)>)>,
    pub resid_ci: ((f64, f64), Vec<(f64, f64)>),
}

/// The p-th quantile (0..1) of a sample, by linear interpolation.
pub fn quantile(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    let h = p * (v.len() - 1) as f64;
    let (i, f) = (h.floor() as usize, h - h.floor());
    if i + 1 < v.len() {
        v[i] * (1.0 - f) + v[i + 1] * f
    } else {
        v[i]
    }
}

fn ci(samples: &[Vec<f64>], k: usize) -> (f64, f64) {
    let mut v: Vec<f64> = samples.iter().map(|s| s[k]).collect();
    (quantile(&mut v, 0.025), quantile(&mut v, 0.975))
}

fn uniform(rng: &mut SplitMix64, n: usize) -> usize {
    rng.below(n as u64) as usize
}

/// The full check of family `a` against yardstick `b` under one form: fit, ground change between a's smallest and
/// largest size, gaps at `targets`, the crossing, and both bootstraps (`boot` replicates each, SplitMix64 `seed`).
pub fn analyse(rows: &[Row], a: &str, b: &str, form: Form, targets: &[f64], boot: usize, seed: u64) -> Result<Report, String> {
    let (ca, cb) = (cells(rows, a), cells(rows, b));
    if ca.len() < 2 || cb.len() < 2 {
        return Err(format!("need 2+ sizes per family: {a} has {}, {b} has {}", ca.len(), cb.len()));
    }
    if form == Form::PowerFloor && ca.len() + cb.len() < 6 {
        return Err("the power form needs 6+ sizes in all (5 parameters)".into());
    }
    let xa: Vec<f64> = ca.iter().map(|c| c.x).collect();
    let xb: Vec<f64> = cb.iter().map(|c| c.x).collect();
    let ya: Vec<f64> = ca.iter().map(Cell::mean).collect();
    let yb: Vec<f64> = cb.iter().map(Cell::mean).collect();
    let (x_lo, x_hi) = (xa[0], xa[xa.len() - 1]);
    let stats = |p: &Pair| -> Vec<f64> {
        let mut v = vec![p.change(x_lo, x_hi)];
        v.extend(targets.iter().map(|&t| p.gap(t)));
        v
    };
    let (pair, res) = fit(form, &xa, &ya, &xb, &yb);
    let n = res.len() as f64;
    let rms = (res.iter().map(|r| r * r).sum::<f64>() / n).sqrt();
    let point = stats(&pair);
    let span_lo = xa[0].min(xb[0]) / 100.0;
    let span_hi = xa[xa.len() - 1].max(xb[xb.len() - 1]) * 1000.0;
    let mut rng = SplitMix64::new(seed);

    // residual bootstrap: fitted + pooled residuals (scaled for the fitted parameters)
    let scale = if n > form.params() as f64 { (n / (n - form.params() as f64)).sqrt() } else { 1.0 };
    let pool: Vec<f64> = res.iter().map(|r| r * scale).collect();
    let mut rs = Vec::with_capacity(boot);
    for _ in 0..boot {
        let ya2: Vec<f64> = xa.iter().map(|&x| pair.a.at(x) + pool[uniform(&mut rng, pool.len())]).collect();
        let yb2: Vec<f64> = xb.iter().map(|&x| pair.b.at(x) + pool[uniform(&mut rng, pool.len())]).collect();
        rs.push(stats(&fit(form, &xa, &ya2, &xb, &yb2).0));
    }
    let resid_ci = (ci(&rs, 0), (0..targets.len()).map(|k| ci(&rs, k + 1)).collect());

    // seed bootstrap: re-average each cell from its seeds; single-seed cells get a pooled seed deviation
    let dev = seed_deviations(rows);
    let seed_ci = if dev.is_empty() {
        None
    } else {
        let redo = |cs: &[Cell], rng: &mut SplitMix64| -> Vec<f64> {
            cs.iter()
                .map(|c| {
                    if c.ys.len() > 1 {
                        (0..c.ys.len()).map(|_| c.ys[uniform(rng, c.ys.len())]).sum::<f64>() / c.ys.len() as f64
                    } else {
                        c.ys[0] + dev[uniform(rng, dev.len())]
                    }
                })
                .collect()
        };
        let mut ss = Vec::with_capacity(boot);
        for _ in 0..boot {
            let ya2 = redo(&ca, &mut rng);
            let yb2 = redo(&cb, &mut rng);
            ss.push(stats(&fit(form, &xa, &ya2, &xb, &yb2).0));
        }
        Some((ci(&ss, 0), (0..targets.len()).map(|k| ci(&ss, k + 1)).collect()))
    };
    Ok(Report { form, pair, rms, change: point[0], gaps: point[1..].to_vec(), crossing: pair.crossing(span_lo, span_hi), seed_ci, resid_ci })
}

/// The v34 rule's words for a ground change against a bar (2σ) and its intervals: "loses ground" only when the
/// change is above the bar and every interval lies above 0; "gains ground" the mirror; else "keeps pace (unclear)".
pub fn verdict(change: f64, bar: f64, cis: &[(f64, f64)]) -> &'static str {
    if change > bar && cis.iter().all(|c| c.0 > 0.0) {
        "loses ground"
    } else if change < -bar && cis.iter().all(|c| c.1 < 0.0) {
        "gains ground"
    } else {
        "keeps pace (unclear)"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows_from(fam: &str, curve: Curve, xs: &[f64], seeds: u64, noise: f64, rng: &mut SplitMix64) -> Vec<Row> {
        let mut out = Vec::new();
        for &x in xs {
            for s in 1..=seeds {
                let u = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                out.push(Row { family: fam.into(), x, bpb: curve.at(x) + noise * u * 3.46, seed: s });
            }
        }
        out
    }

    #[test]
    fn log_linear_fits_recover_a_known_line() {
        let xs = [1e5, 3e5, 1e6, 3e6, 1e7];
        let ys: Vec<f64> = xs.iter().map(|x: &f64| 4.2 - 0.165 * x.ln()).collect();
        let Curve::Lin { a, b } = fit_loglinear(&xs, &ys) else { panic!() };
        assert!((a - 4.2).abs() < 1e-9 && (b + 0.165).abs() < 1e-10);
    }

    #[test]
    fn the_power_form_recovers_a_shared_floor_and_the_gap() {
        // a: flatter (alpha 0.12), b: steeper (0.20), one floor 1.1; noise-free
        let (ta, tb) = (Curve::Pow { e: 1.1, amp: 6.0, alpha: 0.12 }, Curve::Pow { e: 1.1, amp: 14.0, alpha: 0.20 });
        let xa = [1e6, 3e6, 1.8e7];
        let xb = [1e5, 3e5, 8e5, 2.7e6, 6.3e6];
        let ya: Vec<f64> = xa.iter().map(|&x| ta.at(x)).collect();
        let yb: Vec<f64> = xb.iter().map(|&x| tb.at(x)).collect();
        let (p, sse) = fit_power_shared(&xa, &ya, &xb, &yb);
        assert!(sse < 1e-10, "sse {sse}");
        let truth = Pair { a: ta, b: tb };
        for x in [1e6, 1.8e7, 1e8, 1e9] {
            assert!((p.gap(x) - truth.gap(x)).abs() < 2e-3, "gap at {x}: {} vs {}", p.gap(x), truth.gap(x));
        }
        let Curve::Pow { e, .. } = p.a else { panic!() };
        assert!((e - 1.1).abs() < 0.02, "floor {e}");
        // the crossing of the true curves is found where the gap changes sign
        let c = truth.crossing(1e3, 1e12).unwrap();
        assert!(truth.gap(c).abs() < 1e-9);
    }

    #[test]
    fn bootstrap_intervals_cover_the_truth_on_noisy_synthetic_runs() {
        let mut rng = SplitMix64::new(5);
        let (ta, tb) = (Curve::Lin { a: 3.75, b: -0.130 }, Curve::Lin { a: 4.20, b: -0.165 });
        let mut rows = rows_from("S", ta, &[8e4, 2.8e5, 6.6e5, 2.2e6, 5.2e6], 1, 0.002, &mut rng);
        rows.extend(rows_from("T", tb, &[1e5, 3.3e5, 7.9e5, 2.7e6, 6.3e6], 1, 0.002, &mut rng));
        rows.extend(rows_from("S", ta, &[6.6e5], 2, 0.002, &mut rng).into_iter().map(|mut r| {
            r.seed += 1;
            r
        }));
        let truth = Pair { a: ta, b: tb };
        let r = analyse(&rows, "S", "T", Form::LogLinear, &[1.8e7], 2000, 1).unwrap();
        let want = truth.change(8e4, 5.2e6);
        assert!((r.change - want).abs() < 0.03, "change {} vs {want}", r.change);
        let (lo, hi) = r.resid_ci.0;
        assert!(lo < want && want < hi, "residual interval [{lo}, {hi}] misses {want}");
        let (slo, shi) = r.seed_ci.as_ref().unwrap().0;
        assert!(slo < r.change && r.change < shi);
        assert_eq!(verdict(r.change, 0.004, &[r.resid_ci.0, (slo, shi)]), "loses ground");
        // the same data under the power form still says S loses ground over the measured span
        let p = analyse(&rows, "S", "T", Form::PowerFloor, &[1.8e7], 300, 2).unwrap();
        assert!(p.change > 0.05, "power-form change {}", p.change);
    }

    #[test]
    fn seed_spread_is_pooled_across_cells() {
        let r = |f: &str, x: f64, y: f64| Row { family: f.into(), x, bpb: y, seed: 1 };
        let rows = vec![r("S", 1.0, 2.013102), r("S", 1.0, 2.009738), r("S", 1.0, 2.009726), r("T", 1.0, 1.959365), r("T", 1.0, 1.958558), r("T", 1.0, 1.961777), r("T", 2.0, 1.5)];
        // v34: S sd 0.00195, T sd 0.00168, pooled 0.00182
        assert!((seed_sd(&rows).unwrap() - 0.00182).abs() < 0.00001, "{}", seed_sd(&rows).unwrap());
        assert_eq!(cells(&rows, "T").len(), 2);
        assert_eq!(verdict(0.001, 0.0036, &[(-0.01, 0.02)]), "keeps pace (unclear)");
        assert_eq!(verdict(-0.05, 0.0036, &[(-0.08, -0.02)]), "gains ground");
    }
}
