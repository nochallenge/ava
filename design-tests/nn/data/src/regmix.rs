// TESTING / DESIGN ONLY
//! regmix: choosing the data mixture by regression over small proxy runs (V3), after RegMix (Liu et al.
//! 2024, "RegMix: Data Mixture as Regression for Language Model Pre-training") and the data mixing laws
//! of Ye et al. 2024 (loss as the exponential of a linear function of the mixture).
//!
//! 1. PROPOSE. N mixtures are drawn from a Dirichlet over the sources (alpha_s = concentration * center_s
//!    when a center mixture is given, else alpha_s = 1), then put inside per-source bounds [lo_s, hi_s]
//!    (`fit_bounds`): hi_s is usually what a source holds divided by the proxy's train tokens, so every
//!    proxy mixture can really be built. Each mixture is one proxy run (GPU work, queued, not run here).
//! 2. RESULTS. Each proxy run reports its mixture and its val loss on every source's val slice.
//! 3. FIT. For each target source t, a ridge regression of its loss on the mixture weights:
//!      linear     L_t(w) = sum_s c_ts * w_s
//!      loglinear  L_t(w) = exp(sum_s c_ts * w_s)      (Ye et al.; no loss floor, so it can miss the shape)
//!      logfeat    L_t(w) = sum_s [c_ts w_s + d_ts ln(w_s + 0.01) + e_ts ln(w_s + 0.1)]   (the default: a loss
//!                 that falls fast for the first bit of a source and then flattens, as real losses do; on the
//!                 two test worlds one log floor alone missed the best by about as much as the best proxy run)
//!    There is no separate intercept: the weights sum to 1, so sum_s c_ts * w_s already holds one. Ridge
//!    (lambda, default 1e-4) keeps the fit stable with few runs. Fit quality is the leave-one-out RMSE of
//!    the loss and the in-sample R^2.
//! 4. SEARCH. The target is sum_t a_t * L_t(w) (a_t from the target weights, normalised; default equal over
//!    all sources). Random Dirichlet points (alpha 1 and alpha 0.3, put inside the bounds) and the uniform
//!    point are scored. `best` (the default) then walks from the lowest one by pairwise transfers (move d
//!    from source j to source i while that lowers the target; d halves from 0.05 down to 1e-5). `topK` is
//!    RegMix's own rule: the mean of the K lowest points, which stays nearer to where proxies were run.
//!
//! All randomness is SplitMix64 from a seed, so every run of the tool is reproducible.

use crate::rng::SplitMix64;

/// Uniform in [0, 1).
pub fn uniform(r: &mut SplitMix64) -> f64 {
    (r.next_u64() >> 11) as f64 / (1u64 << 53) as f64
}

/// Standard normal (Box-Muller).
pub fn normal(r: &mut SplitMix64) -> f64 {
    let u1 = uniform(r).max(1e-300);
    let u2 = uniform(r);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Gamma(shape, 1) (Marsaglia and Tsang; shape < 1 by the boost u^(1/shape)).
pub fn gamma(r: &mut SplitMix64, shape: f64) -> f64 {
    if shape < 1.0 {
        let u = uniform(r).max(1e-300);
        return gamma(r, shape + 1.0) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = normal(r);
        let v = (1.0 + c * x).powi(3);
        if v <= 0.0 {
            continue;
        }
        let u = uniform(r);
        if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v;
        }
    }
}

/// A Dirichlet(alpha) draw.
pub fn dirichlet(r: &mut SplitMix64, alpha: &[f64]) -> Vec<f64> {
    let g: Vec<f64> = alpha.iter().map(|&a| gamma(r, a.max(1e-3)).max(1e-300)).collect();
    let s: f64 = g.iter().sum();
    g.iter().map(|x| x / s).collect()
}

/// Put a mixture inside per-source bounds, keeping the sum 1: clamp, then move the missing (or extra)
/// mass onto the sources in proportion to their room. None when the bounds cannot hold a mixture.
pub fn fit_bounds(w: &[f64], lo: &[f64], hi: &[f64]) -> Option<Vec<f64>> {
    let (slo, shi): (f64, f64) = (lo.iter().sum(), hi.iter().sum());
    if slo > 1.0 + 1e-9 || shi < 1.0 - 1e-9 || lo.iter().zip(hi).any(|(l, h)| l > h) {
        return None;
    }
    let mut v: Vec<f64> = w.iter().zip(lo.iter().zip(hi)).map(|(&x, (&l, &h))| x.clamp(l, h)).collect();
    let diff = 1.0 - v.iter().sum::<f64>();
    if diff > 0.0 {
        let room: f64 = v.iter().zip(hi).map(|(x, h)| h - x).sum();
        for (x, h) in v.iter_mut().zip(hi) {
            *x += diff * (h - *x) / room;
        }
    } else if diff < 0.0 {
        let room: f64 = v.iter().zip(lo).map(|(x, l)| x - l).sum();
        for (x, l) in v.iter_mut().zip(lo) {
            *x += diff * (*x - l) / room;
        }
    }
    Some(v)
}

/// N proposal mixtures (see PROPOSE).
pub fn propose(n: usize, alpha: &[f64], lo: &[f64], hi: &[f64], seed: u64) -> Option<Vec<Vec<f64>>> {
    let mut r = SplitMix64::new(seed);
    (0..n).map(|_| fit_bounds(&dirichlet(&mut r, alpha), lo, hi)).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// L = c . w
    Linear,
    /// L = exp(c . w)
    LogLinear,
    /// L = c . w + d . ln(w + 0.01) + e . ln(w + 0.1)
    LogFeat,
}

/// The floors inside the log features: a source at weight 0 still has finite features, and two floors
/// let the fit bend fast near 0 (0.01) and flatten later (0.1), as a loss with a floor does.
pub const LOG_EPS: [f64; 2] = [0.01, 0.1];

impl Kind {
    pub fn parse(s: &str) -> Result<Kind, String> {
        match s {
            "linear" => Ok(Kind::Linear),
            "loglinear" => Ok(Kind::LogLinear),
            "logfeat" => Ok(Kind::LogFeat),
            _ => Err(format!("unknown model {s:?} (linear, loglinear, logfeat)")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Kind::Linear => "linear",
            Kind::LogLinear => "loglinear",
            Kind::LogFeat => "logfeat",
        }
    }

    /// The regression features of a mixture.
    pub fn features(self, w: &[f64]) -> Vec<f64> {
        match self {
            Kind::Linear | Kind::LogLinear => w.to_vec(),
            Kind::LogFeat => w.iter().copied().chain(LOG_EPS.iter().flat_map(|e| w.iter().map(move |x| (x + e).ln()))).collect(),
        }
    }
}

/// One proxy run: its mixture and its val loss per source.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub mix: Vec<f64>,
    pub loss: Vec<f64>,
}

/// Solve (X'X + lambda I) c = X'y by Gaussian elimination with partial pivoting.
pub fn ridge(x: &[Vec<f64>], y: &[f64], lambda: f64) -> Vec<f64> {
    let p = x.first().map_or(0, |r| r.len());
    let mut a = vec![vec![0.0; p + 1]; p];
    for (row, &yy) in x.iter().zip(y) {
        for i in 0..p {
            for j in 0..p {
                a[i][j] += row[i] * row[j];
            }
            a[i][p] += row[i] * yy;
        }
    }
    for (i, r) in a.iter_mut().enumerate() {
        r[i] += lambda;
    }
    for col in 0..p {
        let piv = (col..p).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
        a.swap(col, piv);
        let d = a[col][col];
        if d.abs() < 1e-300 {
            continue;
        }
        for i in 0..p {
            if i != col {
                let f = a[i][col] / d;
                if f != 0.0 {
                    for j in col..=p {
                        a[i][j] -= f * a[col][j];
                    }
                }
            }
        }
    }
    (0..p).map(|i| if a[i][i].abs() < 1e-300 { 0.0 } else { a[i][p] / a[i][i] }).collect()
}

/// A fitted model: per target source, one coefficient per feature.
#[derive(Clone, Debug)]
pub struct Model {
    pub kind: Kind,
    pub coef: Vec<Vec<f64>>,
}

impl Model {
    pub fn fit(rows: &[Row], kind: Kind, lambda: f64) -> Model {
        let x: Vec<Vec<f64>> = rows.iter().map(|r| kind.features(&r.mix)).collect();
        let targets = rows.first().map_or(0, |r| r.loss.len());
        let coef = (0..targets)
            .map(|t| {
                let y: Vec<f64> = rows
                    .iter()
                    .map(|r| match kind {
                        Kind::LogLinear => r.loss[t].max(1e-12).ln(),
                        Kind::Linear | Kind::LogFeat => r.loss[t],
                    })
                    .collect();
                ridge(&x, &y, lambda)
            })
            .collect();
        Model { kind, coef }
    }

    /// Predicted loss of every target source.
    pub fn predict(&self, w: &[f64]) -> Vec<f64> {
        let f = self.kind.features(w);
        self.coef
            .iter()
            .map(|c| {
                let z: f64 = c.iter().zip(&f).map(|(a, b)| a * b).sum();
                match self.kind {
                    Kind::LogLinear => z.exp(),
                    Kind::Linear | Kind::LogFeat => z,
                }
            })
            .collect()
    }

    /// The target: the losses weighted by `a` (normalised to sum 1).
    pub fn target(&self, w: &[f64], a: &[f64]) -> f64 {
        let s: f64 = a.iter().sum();
        self.predict(w).iter().zip(a).map(|(l, x)| l * x / s).sum()
    }
}

/// Fit quality per target: (leave-one-out RMSE of the loss, in-sample R^2).
pub fn quality(rows: &[Row], kind: Kind, lambda: f64) -> Vec<(f64, f64)> {
    let targets = rows.first().map_or(0, |r| r.loss.len());
    let full = Model::fit(rows, kind, lambda);
    let mut sq = vec![0.0; targets];
    for k in 0..rows.len() {
        let rest: Vec<Row> = rows.iter().enumerate().filter(|(i, _)| *i != k).map(|(_, r)| r.clone()).collect();
        let p = Model::fit(&rest, kind, lambda).predict(&rows[k].mix);
        for t in 0..targets {
            sq[t] += (p[t] - rows[k].loss[t]).powi(2);
        }
    }
    (0..targets)
        .map(|t| {
            let mean = rows.iter().map(|r| r.loss[t]).sum::<f64>() / rows.len() as f64;
            let tot: f64 = rows.iter().map(|r| (r.loss[t] - mean).powi(2)).sum();
            let res: f64 = rows.iter().map(|r| (full.predict(&r.mix)[t] - r.loss[t]).powi(2)).sum();
            ((sq[t] / rows.len() as f64).sqrt(), if tot > 0.0 { 1.0 - res / tot } else { 1.0 })
        })
        .collect()
}

/// How the best mixture is searched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Search {
    /// The lowest predicted target: random points, then pairwise transfers.
    Best,
    /// RegMix's rule: the mean of the `k` best of the random points (stays near where proxies were run).
    TopK(usize),
}

impl Search {
    pub fn parse(s: &str) -> Result<Search, String> {
        match s {
            "best" => Ok(Search::Best),
            _ => match s.strip_prefix("top") {
                Some(k) => k.parse::<usize>().map(|k| Search::TopK(k.max(1))).map_err(|_| format!("bad search {s:?}")),
                None => Err(format!("unknown search {s:?} (best, topK such as top100)")),
            },
        }
    }
}

/// The best mixture inside the bounds by `how` (see SEARCH). Returns it and its predicted target.
pub fn search(m: &Model, a: &[f64], lo: &[f64], hi: &[f64], seed: u64, samples: usize, how: Search) -> Option<(Vec<f64>, f64)> {
    let n = lo.len();
    let f = |w: &[f64]| m.target(w, a);
    let mut r = SplitMix64::new(seed);
    let mut pts: Vec<(f64, Vec<f64>)> = Vec::with_capacity(samples + 1);
    let uni = fit_bounds(&vec![1.0 / n as f64; n], lo, hi)?;
    pts.push((f(&uni), uni));
    for k in 0..samples {
        let alpha = vec![if k % 2 == 0 { 1.0 } else { 0.3 }; n];
        let w = fit_bounds(&dirichlet(&mut r, &alpha), lo, hi)?;
        pts.push((f(&w), w));
    }
    pts.sort_by(|x, y| x.0.total_cmp(&y.0));
    if let Search::TopK(k) = how {
        let k = k.min(pts.len());
        let mut mean = vec![0.0; n];
        for (_, w) in &pts[..k] {
            for i in 0..n {
                mean[i] += w[i] / k as f64;
            }
        }
        let v = f(&mean);
        return Some((mean, v));
    }
    let (mut fb, mut best) = pts.swap_remove(0);
    let mut d: f64 = 0.05;
    while d >= 1e-5 {
        let mut improved = true;
        while improved {
            improved = false;
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let step = d.min(best[j] - lo[j]).min(hi[i] - best[i]);
                    if step <= 0.0 {
                        continue;
                    }
                    let mut w = best.clone();
                    w[i] += step;
                    w[j] -= step;
                    let v = f(&w);
                    if v < fb - 1e-15 {
                        (best, fb) = (w, v);
                        improved = true;
                    }
                }
            }
        }
        d /= 2.0;
    }
    Some((best, fb))
}

/// `a=0.123456,b=...` (the form `v42_shards --mix` reads).
pub fn mix_string(names: &[String], w: &[f64]) -> String {
    names.iter().zip(w).map(|(n, x)| format!("{n}={x:.6}")).collect::<Vec<_>>().join(",")
}

/// Read proxy results: a TSV whose header has `w_NAME` and `loss_NAME` for every source name; other
/// columns are ignored, and a row with an empty or unreadable loss is skipped (a run not done yet).
pub fn read_results(text: &str, names: &[String]) -> Result<(Vec<Row>, usize), String> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
    let head: Vec<&str> = lines.next().ok_or("empty results file")?.split('\t').map(str::trim).collect();
    let col = |p: &str, n: &str| head.iter().position(|h| *h == format!("{p}{n}")).ok_or_else(|| format!("results header lacks {p}{n}"));
    let wc: Vec<usize> = names.iter().map(|n| col("w_", n)).collect::<Result<_, _>>()?;
    let lc: Vec<usize> = names.iter().map(|n| col("loss_", n)).collect::<Result<_, _>>()?;
    let (mut rows, mut skipped) = (Vec::new(), 0usize);
    for l in lines {
        let f: Vec<&str> = l.split('\t').map(str::trim).collect();
        let get = |c: &[usize]| c.iter().map(|&i| f.get(i).and_then(|x| x.parse::<f64>().ok())).collect::<Option<Vec<f64>>>();
        match (get(&wc), get(&lc)) {
            (Some(mix), Some(loss)) if loss.iter().all(|x| x.is_finite() && *x > 0.0) => {
                let s: f64 = mix.iter().sum();
                rows.push(Row { mix: mix.iter().map(|x| x / s).collect(), loss });
            }
            _ => skipped += 1,
        }
    }
    Ok((rows, skipped))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two known worlds (4 sources): the loss on source t falls as its own weight grows, towards a floor.
    /// World 0: b + k exp(-g w_t) + a small cross term. World 1: b + k (w_t + 0.02)^-p.
    fn world(kind: usize, w: &[f64]) -> Vec<f64> {
        let b = [2.0, 2.5, 3.0, 1.5];
        let k = [1.0, 1.5, 0.8, 2.0];
        let g = [6.0, 3.0, 9.0, 4.0];
        let p = [0.3, 0.5, 0.2, 0.4];
        (0..4)
            .map(|t| match kind {
                0 => b[t] + k[t] * (-g[t] * w[t]).exp() + 0.1 * w[(t + 1) % 4],
                _ => b[t] + 0.3 * k[t] * (w[t] + 0.02).powf(-p[t]),
            })
            .collect()
    }

    fn tval(kind: usize, w: &[f64], a: &[f64]) -> f64 {
        world(kind, w).iter().zip(a).map(|(l, x)| l * x).sum::<f64>() / a.iter().sum::<f64>()
    }

    /// The true best on a 0.01 grid of the simplex.
    fn brute_best(kind: usize, a: &[f64], lo: &[f64], hi: &[f64]) -> (Vec<f64>, f64) {
        let mut best = (vec![], f64::INFINITY);
        for i in 0..=100 {
            for j in 0..=(100 - i) {
                for k in 0..=(100 - i - j) {
                    let w = [i as f64 / 100.0, j as f64 / 100.0, k as f64 / 100.0, (100 - i - j - k) as f64 / 100.0];
                    if w.iter().zip(lo.iter().zip(hi)).all(|(x, (l, h))| *x >= l - 1e-9 && *x <= h + 1e-9) {
                        let v = tval(kind, &w, a);
                        if v < best.1 {
                            best = (w.to_vec(), v);
                        }
                    }
                }
            }
        }
        best
    }

    #[test]
    fn dirichlet_means_and_bounds() {
        let mut r = SplitMix64::new(1);
        let alpha = [2.0, 1.0, 1.0];
        let mut mean = [0.0; 3];
        for _ in 0..20000 {
            let w = dirichlet(&mut r, &alpha);
            assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9 && w.iter().all(|x| *x > 0.0));
            for i in 0..3 {
                mean[i] += w[i] / 20000.0;
            }
        }
        assert!((mean[0] - 0.5).abs() < 0.01 && (mean[1] - 0.25).abs() < 0.01, "{mean:?}");
        let v = fit_bounds(&[0.7, 0.2, 0.1], &[0.0, 0.0, 0.05], &[0.3, 1.0, 1.0]).unwrap();
        assert!((v.iter().sum::<f64>() - 1.0).abs() < 1e-12 && v[0] <= 0.3 + 1e-12 && v[2] >= 0.05);
        assert!(fit_bounds(&[0.5, 0.5], &[0.0, 0.0], &[0.3, 0.3]).is_none());
        let p = propose(50, &[1.0; 4], &[0.0; 4], &[0.5; 4], 9).unwrap();
        assert!(p.iter().all(|w| w.iter().all(|x| *x <= 0.5 + 1e-12)));
    }

    #[test]
    fn ridge_recovers_a_plane() {
        let x: Vec<Vec<f64>> = (0..30).map(|i| vec![1.0, (i % 7) as f64, (i % 5) as f64 * 0.5]).collect();
        let y: Vec<f64> = x.iter().map(|r| 2.0 + 3.0 * r[1] - 1.5 * r[2]).collect();
        let c = ridge(&x, &y, 1e-9);
        assert!((c[0] - 2.0).abs() < 1e-6 && (c[1] - 3.0).abs() < 1e-6 && (c[2] + 1.5).abs() < 1e-6, "{c:?}");
    }

    /// The default (logfeat, best) finds a known best mixture from 64 noisy proxy runs in two worlds of
    /// different shape: its true target is within 0.005 of the true best and no worse than the best proxy run.
    #[test]
    fn finds_a_known_best_mixture_from_noisy_proxies() {
        let (lo, hi) = (vec![0.0; 4], vec![1.0, 1.0, 1.0, 0.4]);
        let a = [1.0, 1.0, 2.0, 1.0];
        for kind in [0usize, 1] {
            let props = propose(64, &[1.0; 4], &lo, &hi, 7 + kind as u64).unwrap();
            let mut r = SplitMix64::new(3);
            let rows: Vec<Row> = props.iter().map(|w| Row { mix: w.clone(), loss: world(kind, w).iter().map(|l| l + 0.005 * normal(&mut r)).collect() }).collect();
            let (truth, tv) = brute_best(kind, &a, &lo, &hi);
            let best_proxy = rows.iter().map(|r| tval(kind, &r.mix, &a)).fold(f64::INFINITY, f64::min);
            for m in [Kind::LogFeat, Kind::LogLinear, Kind::Linear] {
                for how in [Search::Best, Search::TopK(100)] {
                    let model = Model::fit(&rows, m, 1e-4);
                    let (w, _) = search(&model, &a, &lo, &hi, 11, 20000, how).unwrap();
                    assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9 && w[3] <= 0.4 + 1e-9);
                    let got = tval(kind, &w, &a);
                    let l1: f64 = w.iter().zip(&truth).map(|(x, y)| (x - y).abs()).sum();
                    eprintln!("world {kind} {:>9} {how:?}: gap {:.4} (best proxy gap {:.4}) L1 {l1:.3} {w:.3?} vs {truth:.2?}", m.name(), got - tv, best_proxy - tv);
                    if (m, how) == (Kind::LogFeat, Search::Best) {
                        assert!(got - tv < 0.005, "world {kind}: gap {}", got - tv);
                        assert!(got <= best_proxy + 1e-9, "world {kind}");
                    }
                }
            }
            let q = quality(&rows, Kind::LogFeat, 1e-4);
            assert!(q.iter().all(|(rmse, r2)| *rmse < 0.05 && *r2 > 0.9), "world {kind}: {q:?}");
        }
    }

    #[test]
    fn parse_names() {
        assert_eq!(Kind::parse("logfeat").unwrap(), Kind::LogFeat);
        assert!(Kind::parse("tree").is_err());
        assert_eq!(Search::parse("top100").unwrap(), Search::TopK(100));
        assert_eq!(Search::parse("best").unwrap(), Search::Best);
        assert!(Search::parse("worst").is_err());
    }

    #[test]
    fn results_round_trip() {
        let names = vec!["a".to_string(), "b".to_string()];
        let text = "run\tw_a\tw_b\tloss_a\tloss_b\textra\n0\t0.25\t0.75\t3.1\t2.9\tx\n1\t0.5\t0.5\t\t\t\n# note\n2\t1\t3\t3.0\t2.8\t\n";
        let (rows, skipped) = read_results(text, &names).unwrap();
        assert_eq!(skipped, 1);
        assert_eq!(rows[1].mix, vec![0.25, 0.75], "weights are normalised");
        assert!(read_results("run\tw_a\n", &names).is_err());
        assert_eq!(mix_string(&names, &[0.25, 0.75]), "a=0.250000,b=0.750000");
    }
}
