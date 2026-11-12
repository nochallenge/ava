// TESTING / DESIGN ONLY
//! K5 vision, a small version: a retina and the first visual cortex (V1).
//! - Retina: centre-surround cells (a difference of Gaussians, Kuffler 1953), divided by the local mean
//!   light (Weber's law, light adaptation), so the code is the same in dim and bright light:
//!     r = (centre − surround) / (local mean + ε)        ON = max(r − θ, 0), OFF = max(−r − θ, 0)
//!   Flat light gives no signal at all: only edges and changes cost spikes (sparse, like `crate::event`).
//! - V1 simple cells sum the retina's output along a short oriented line (Hubel & Wiesel); complex cells
//!   pool |simple| over position, so they say "a line at this angle is here somewhere".
//! - For Ava: `patches` cuts the ON/OFF maps into patch tokens that a learned projection maps to her width
//!   (the way a vision transformer reads an image), and `v1_features` are the pooled angles.
//!
//! The oriented summation is fixed here, a developmental prior; a learned sparse code (`crate::sae`)
//! trained on real images can replace it once images are downloaded.

pub struct Image {
    pub w: usize,
    pub h: usize,
    pub px: Vec<f32>,
}

impl Image {
    pub fn new(w: usize, h: usize, v: f32) -> Self {
        Self { w, h, px: vec![v; w * h] }
    }
    fn at(&self, x: isize, y: isize) -> f32 {
        let x = x.clamp(0, self.w as isize - 1) as usize;
        let y = y.clamp(0, self.h as isize - 1) as usize;
        self.px[y * self.w + x]
    }
    pub fn scaled(&self, k: f32) -> Image {
        Image { w: self.w, h: self.h, px: self.px.iter().map(|v| v * k).collect() }
    }
}

fn gauss(img: &Image, sigma: f32) -> Image {
    let r = (3.0 * sigma).ceil() as isize;
    let k: Vec<f32> = (-r..=r).map(|i| (-(i * i) as f32 / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = k.iter().sum();
    let k: Vec<f32> = k.iter().map(|v| v / s).collect();
    let mut tmp = Image::new(img.w, img.h, 0.0);
    for y in 0..img.h {
        for x in 0..img.w {
            tmp.px[y * img.w + x] = (-r..=r).map(|i| k[(i + r) as usize] * img.at(x as isize + i, y as isize)).sum();
        }
    }
    let mut out = Image::new(img.w, img.h, 0.0);
    for y in 0..img.h {
        for x in 0..img.w {
            out.px[y * img.w + x] = (-r..=r).map(|i| k[(i + r) as usize] * tmp.at(x as isize, y as isize + i)).sum();
        }
    }
    out
}

pub struct Retina {
    pub centre: f32,
    pub surround: f32,
    pub adapt: f32,
    pub theta: f32,
    pub eps: f32,
}

impl Default for Retina {
    fn default() -> Self {
        Self { centre: 0.8, surround: 2.0, adapt: 4.0, theta: 0.05, eps: 1e-5 }
    }
}

impl Retina {
    /// The light-adapted centre-surround signal r at every pixel.
    pub fn signal(&self, img: &Image) -> Image {
        let (c, s, m) = (gauss(img, self.centre), gauss(img, self.surround), gauss(img, self.adapt));
        let px = (0..img.px.len()).map(|i| (c.px[i] - s.px[i]) / (m.px[i] + self.eps)).collect();
        Image { w: img.w, h: img.h, px }
    }

    /// ON and OFF cell rates.
    pub fn on_off(&self, img: &Image) -> (Vec<f32>, Vec<f32>) {
        let r = self.signal(img);
        (r.px.iter().map(|v| (v - self.theta).max(0.0)).collect(), r.px.iter().map(|v| (-v - self.theta).max(0.0)).collect())
    }

    /// Patch tokens: [n_patches][2·p·p] (ON then OFF), row-major patches of size p.
    pub fn patches(&self, img: &Image, p: usize) -> Vec<Vec<f32>> {
        let (on, off) = self.on_off(img);
        let mut out = Vec::new();
        for py in 0..img.h / p {
            for px in 0..img.w / p {
                let mut t = Vec::with_capacity(2 * p * p);
                for map in [&on, &off] {
                    for y in 0..p {
                        for x in 0..p {
                            t.push(map[(py * p + y) * img.w + px * p + x]);
                        }
                    }
                }
                out.push(t);
            }
        }
        out
    }

    /// V1: for 4 angles, simple cells sum r along a line of 7 pixels; complex cells pool |simple| by its
    /// mean and its max over position. Returns [mean at 4 angles, max at 4 angles], each group divided by
    /// its own total, so the code says which angles, not how much contrast.
    pub fn v1_features(&self, img: &Image) -> Vec<f32> {
        let r = self.signal(img);
        let mut mean = [0f32; 4];
        let mut max = [0f32; 4];
        for (a, m) in mean.iter_mut().enumerate() {
            let th = a as f32 * std::f32::consts::FRAC_PI_4;
            let (dx, dy) = (th.cos(), th.sin());
            for y in 0..img.h {
                for x in 0..img.w {
                    let s: f32 = (-3..=3)
                        .map(|t| {
                            let (fx, fy) = (x as f32 + t as f32 * dx, y as f32 + t as f32 * dy);
                            bilinear(&r, fx, fy)
                        })
                        .sum::<f32>()
                        .abs();
                    *m += s;
                    max[a] = max[a].max(s);
                }
            }
            *m /= (img.w * img.h) as f32;
        }
        let (sm, sx) = (mean.iter().sum::<f32>().max(1e-9), max.iter().sum::<f32>().max(1e-9));
        mean.iter().map(|v| v / sm).chain(max.iter().map(|v| v / sx)).collect()
    }
}

fn bilinear(img: &Image, x: f32, y: f32) -> f32 {
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let (x0, y0) = (x0 as isize, y0 as isize);
    img.at(x0, y0) * (1.0 - fx) * (1.0 - fy) + img.at(x0 + 1, y0) * fx * (1.0 - fy) + img.at(x0, y0 + 1) * (1.0 - fx) * fy + img.at(x0 + 1, y0 + 1) * fx * fy
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::readout::Readout;

    struct Rng(u64);
    impl Rng {
        fn u(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    const SHAPES: [&str; 7] = ["horizontal", "vertical", "rising", "falling", "plus", "cross", "circle"];
    const N: usize = 32;

    /// A shape of random size, place, thickness and contrast sign on a random background, at light level k.
    fn draw(class: usize, r: &mut Rng, k: f32) -> Image {
        let bg = 0.3 + 0.5 * r.u();
        let fig = if r.u() < 0.5 { bg * (1.4 + 0.6 * r.u()) } else { bg * (0.2 + 0.4 * r.u()) };
        let mut img = Image::new(N, N, bg);
        let half = 5.0 + 4.0 * r.u();
        let (cx, cy) = (half + 2.0 + (N as f32 - 2.0 * half - 4.0) * r.u(), half + 2.0 + (N as f32 - 2.0 * half - 4.0) * r.u());
        let thick = 0.6 + 0.6 * r.u();
        let seg = |px: f32, py: f32, a: f32| -> bool {
            // Distance from the point to a segment through the centre at angle a, of half-length `half`.
            let (dx, dy) = (px - cx, py - cy);
            let (c, s) = (a.cos(), a.sin());
            let along = dx * c + dy * s;
            let across = (-dx * s + dy * c).abs();
            along.abs() <= half && across <= thick
        };
        let q = std::f32::consts::FRAC_PI_4;
        for y in 0..N {
            for x in 0..N {
                let (px, py) = (x as f32, y as f32);
                let on = match SHAPES[class] {
                    "horizontal" => seg(px, py, 0.0),
                    "vertical" => seg(px, py, 2.0 * q),
                    "rising" => seg(px, py, q),
                    "falling" => seg(px, py, 3.0 * q),
                    "plus" => seg(px, py, 0.0) || seg(px, py, 2.0 * q),
                    "cross" => seg(px, py, q) || seg(px, py, 3.0 * q),
                    _ => (((px - cx).powi(2) + (py - cy).powi(2)).sqrt() - half).abs() <= thick,
                };
                if on {
                    img.px[y * N + x] = fig;
                }
            }
        }
        img.scaled(k)
    }

    fn set(n: usize, seed: u64, k: f32, ret: &Retina) -> (Vec<Vec<f32>>, Vec<usize>) {
        let mut r = Rng(seed);
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..n {
            let c = i % SHAPES.len();
            let light = k * (0.5 + r.u());
            x.push(ret.v1_features(&draw(c, &mut r, light)));
            y.push(c);
        }
        (x, y)
    }

    #[test]
    fn she_tells_shapes_apart_anywhere_in_view() {
        let ret = Retina::default();
        let t0 = std::time::Instant::now();
        let (x, y) = set(700, 1, 1.0, &ret);
        let ro = Readout::train(&x, &y, SHAPES.len(), 400, 0.05).unwrap();
        let (tx, ty) = set(350, 2, 1.0, &ret);
        let acc = ro.accuracy(&tx, &ty).unwrap();
        eprintln!("shapes at new places and sizes: {:.1}% right (chance {:.1}%), {:.2} s for 1050 images", 100.0 * acc, 100.0 / 7.0, t0.elapsed().as_secs_f32());
        assert!(acc > 0.9, "{acc}");
        // In dim light (a tenth) and in bright light (ten times), with no retraining.
        for k in [0.1f32, 10.0] {
            let (tx, ty) = set(350, 3, k, &ret);
            let a = ro.accuracy(&tx, &ty).unwrap();
            eprintln!("  at light x{k}: {:.1}% right", 100.0 * a);
            assert!(a > 0.85, "{k}: {a}");
        }
    }

    #[test]
    fn the_retina_code_is_the_same_in_any_light_and_silent_on_flat_light() {
        let ret = Retina::default();
        let mut r = Rng(9);
        let img = draw(6, &mut r, 1.0);
        let base = ret.signal(&img);
        for k in [0.1f32, 10.0] {
            let s = ret.signal(&img.scaled(k));
            let d = base.px.iter().zip(&s.px).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            eprintln!("  code change at light x{k}: {d:.2e}");
            assert!(d < 2e-3, "{k}: {d}");
        }
        let (on, off) = ret.on_off(&img);
        let active = on.iter().chain(&off).filter(|&&v| v > 0.0).count() as f32 / (2 * N * N) as f32;
        eprintln!("cells active on a shape: {:.1}%", 100.0 * active);
        assert!(active < 0.2, "{active}");
        let (on, off) = ret.on_off(&Image::new(N, N, 0.7));
        assert!(on.iter().chain(&off).all(|&v| v == 0.0));
        let p = ret.patches(&img, 8);
        assert_eq!((p.len(), p[0].len()), (16, 128));
    }
}
