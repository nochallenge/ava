// TESTING / DESIGN ONLY — tests of src/model/fwin_f2.rs (v51 fast2: R, W, TH and CB on the fused WIN step).
//
// CPU: the candle-op twins against candle's autograd of the reference path's own functions (v51::letter_core,
// v51::col_mm, v51b::th_read / th_gates / th_res, v51b::cb_code), f64 and f32; the parts that are the reference's own
// ops must agree bit for bit. Twins whose backward differs from autograd's only in summation order: f64 within 1e-12
// relative, f32 within 2e-6 (5e-6 for sums). R's backward kernels' row lists are checked by a host walk of them (f64).
// The step's hooks themselves (F2) on a CPU model, in the order src/model/fwin.rs calls them: R's letters, a three-block
// relay chain (toy blocks around TH's hooks), CB's head, W's fold and unfold, against autograd (f32, 2e-5: chains).
// CUDA (AVA_GPU_TEST_CUDA=1, a second): the kernels against the twins (f32: the forwards within 1e-6, the sums within
// 1e-5, CB's winners the same units).

use super::*;
use crate::arch::Arch;
use crate::model::{Config, Kind};
use crate::testkit::unit_noise;
use candle_core::{Device, Var};
use candle_nn::{VarBuilder, VarMap};

const CPU: Device = Device::Cpu;

fn noise(dev: &Device, shape: &[usize], seed: u64, scale: f64, offset: f64, dt: DType) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_vec(unit_noise(n, seed), shape, dev).unwrap().to_dtype(dt).unwrap().affine(scale, offset).unwrap()
}

/// max |a - b| / max(max |a|, floor), in f64.
fn rel(a: &Tensor, b: &Tensor, floor: f64) -> f64 {
    let a = a.to_dtype(DType::F64).unwrap().flatten_all().unwrap();
    let b = b.to_dtype(DType::F64).unwrap().flatten_all().unwrap();
    assert_eq!(a.dims(), b.dims(), "shapes");
    let d = a.sub(&b).unwrap().abs().unwrap().max(0).unwrap().to_scalar::<f64>().unwrap();
    let m = a.abs().unwrap().max(0).unwrap().to_scalar::<f64>().unwrap();
    d / m.max(floor)
}

fn same(a: &Tensor, b: &Tensor) -> bool {
    rel(a, b, 1e-30) == 0.0
}

fn tol(dt: DType, sum: bool) -> f64 {
    match (dt, sum) {
        (DType::F64, _) => 1e-12,
        (_, false) => 2e-6,
        (_, true) => 5e-6,
    }
}

const CHAIN: f64 = 2e-5;

fn check(tag: &str, want: &Tensor, got: &Tensor, bar: f64) {
    let e = rel(want, got, 1e-30);
    eprintln!("{tag}: rel {e:.2e} (bar {bar:.0e})");
    assert!(e <= bar, "{tag}: rel {e} > {bar}");
}

/// A letter table of `v` tokens: byte strings of 0..=20 bytes (past 16 cut, as v51's builder), from a seed.
fn table(v: usize, seed: u64) -> Vec<Vec<u8>> {
    let r = unit_noise(v * 22, seed);
    (0..v)
        .map(|t| {
            let len = ((r[t * 22] + 1.0) * 10.5) as usize % 21;
            (0..len).map(|j| (((r[t * 22 + 1 + j] + 1.0) * 128.0) as usize).min(255) as u8).collect()
        })
        .collect()
}

/// v51's tables of `table` (its builder's code): ids, eidx (u32 [V, 16]), mask [V, 16], inv [V, 1].
fn rows(table: &[Vec<u8>], dev: &Device) -> (Tensor, Tensor, Tensor, Tensor) {
    let (v, lm) = (table.len(), crate::v51::LETTER_MAX);
    let (mut ids, mut eidx, mut mask, mut inv) = (vec![0u32; v * lm], vec![0u32; v * lm], vec![0f32; v * lm], vec![0f32; v]);
    for (t, bytes) in table.iter().enumerate() {
        let len = bytes.len();
        let used = len.min(lm);
        for j in 0..used {
            ids[t * lm + j] = bytes[j] as u32;
            eidx[t * lm + j] = (len - 1 - j).min(lm - 1) as u32;
            mask[t * lm + j] = 1.0;
        }
        if used > 0 {
            inv[t] = 1.0 / (used as f32).sqrt();
        }
    }
    (Tensor::from_vec(ids, (v, lm), dev).unwrap(), Tensor::from_vec(eidx, (v, lm), dev).unwrap(), Tensor::from_vec(mask, (v, lm), dev).unwrap(), Tensor::from_vec(inv, (v, 1), dev).unwrap())
}

/// R's backward as the two kernels compute it, walked on the host from the row lists (f64).
fn let_bwd_host(t: &LetTabs, dl: &Tensor) -> Vec<Vec<f64>> {
    let csr = let_csr(t).unwrap();
    let lm = crate::v51::LETTER_MAX;
    let f = |x: &Tensor| -> Vec<Vec<f64>> { x.to_dtype(DType::F64).unwrap().to_vec2().unwrap() };
    let (e, p, q, g) = (f(t.byte), f(t.pos), f(t.end), f(dl));
    let inv: Vec<f64> = t.inv.to_dtype(DType::F64).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let ids: Vec<u32> = t.ids.flatten_all().unwrap().to_vec1().unwrap();
    let eidx: Vec<u32> = t.eidx.flatten_all().unwrap().to_vec1().unwrap();
    let pos: Vec<u32> = csr.pos.to_vec1().unwrap();
    let cbeg: Vec<u32> = csr.cbeg.to_vec1().unwrap();
    let rbeg: Vec<u32> = csr.rbeg.to_vec1().unwrap();
    let dd = crate::v51::LETTER_DIM;
    let part: Vec<Vec<f64>> = (0..csr.nch)
        .map(|ch| {
            (0..dd)
                .map(|c| {
                    let mut acc = 0.0;
                    for &pp in &pos[cbeg[ch] as usize..cbeg[ch + 1] as usize] {
                        let pp = pp as usize;
                        let (v, j) = (pp / lm, pp % lm);
                        let s = ((e[ids[pp] as usize][c] + q[eidx[pp] as usize][c]) + p[j][c]).tanh();
                        acc += (g[v][c] * inv[v]) * (1.0 - s * s);
                    }
                    acc
                })
                .collect()
        })
        .collect();
    (0..LET_ROWS).map(|r| (0..dd).map(|c| (rbeg[r]..rbeg[r + 1]).map(|ch| part[ch as usize][c]).sum()).collect()).collect()
}

#[test]
fn letters_twins_match_autograd() {
    for dt in [DType::F64, DType::F32] {
        let tb = table(41, 7);
        let (ids, eidx, mask, inv) = rows(&tb, &CPU);
        let dd = crate::v51::LETTER_DIM;
        let byte = Var::from_tensor(&noise(&CPU, &[256, dd], 1, 0.6, 0.0, dt)).unwrap();
        let pos = Var::from_tensor(&noise(&CPU, &[16, dd], 2, 0.6, 0.0, dt)).unwrap();
        let end = Var::from_tensor(&noise(&CPU, &[16, dd], 3, 0.6, 0.0, dt)).unwrap();
        let dl = noise(&CPU, &[41, dd], 4, 1.0, 0.0, dt);
        let lv = crate::v51::letter_core(byte.as_tensor(), pos.as_tensor(), end.as_tensor(), &ids, &eidx, &mask, &inv).unwrap();
        let gr = lv.mul(&dl).unwrap().sum_all().unwrap().backward().unwrap();
        let t = LetTabs { byte: byte.as_tensor(), pos: pos.as_tensor(), end: end.as_tensor(), ids: &ids, eidx: &eidx, mask: &mask, inv: &inv };
        assert!(same(&lv, &let_fwd(&t).unwrap()), "{dt:?}: let_fwd on the CPU is letter_core, bit for bit");
        let (de, dp, dq) = let_bwd(&t, &dl).unwrap();
        for (name, v, g) in [("byte", &byte, &de), ("pos", &pos, &dp), ("end", &end, &dq)] {
            check(&format!("letters {dt:?} d{name} (twin vs autograd)"), gr.get(v.as_tensor()).unwrap(), g, tol(dt, true));
        }
        if dt == DType::F64 {
            // the kernels' row lists (chunks of one row each, every masked-in entry once per table row it uses)
            let host = let_bwd_host(&t, &dl);
            let host = Tensor::from_vec(host.concat(), (LET_ROWS, dd), &CPU).unwrap();
            let twin = Tensor::cat(&[&de, &dp, &dq], 0).unwrap();
            check("letters f64: the kernels' row-list walk vs the twin", &twin, &host, 1e-12);
            let csr = let_csr(&t).unwrap();
            let used: f32 = mask.sum_all().unwrap().to_scalar().unwrap();
            assert_eq!(csr.pos.dim(0).unwrap(), 3 * used as usize, "each entry in three rows' lists");
        }
    }
}

#[test]
fn columns_fold_matches_col_mm() {
    for dt in [DType::F64, DType::F32] {
        let (n, di, dout, k, r) = (11usize, 24usize, 40usize, 4usize, 8usize);
        let h = Var::from_tensor(&noise(&CPU, &[n, di], 21, 1.0, 0.0, dt)).unwrap();
        let w = Var::from_tensor(&noise(&CPU, &[di, dout], 22, 0.5, 0.0, dt)).unwrap();
        let a = Var::from_tensor(&noise(&CPU, &[di, r], 23, 0.4, 0.0, dt)).unwrap();
        let c = Var::from_tensor(&noise(&CPU, &[r, dout], 24, 0.4, 0.0, dt)).unwrap();
        let dy = noise(&CPU, &[n, dout], 25, 1.0, 0.0, dt);
        let y = crate::v51::col_mm(h.as_tensor(), w.as_tensor(), a.as_tensor(), c.as_tensor(), k).unwrap();
        let gr = y.mul(&dy).unwrap().sum_all().unwrap().backward().unwrap();
        let we = col_fold(w.as_tensor(), a.as_tensor(), c.as_tensor(), k).unwrap();
        check(&format!("columns {dt:?}: h fold(W) vs col_mm"), &y, &h.as_tensor().matmul(&we).unwrap(), tol(dt, true));
        let dwe = h.as_tensor().t().unwrap().matmul(&dy).unwrap();
        let (dw, da, dc) = col_unfold(&dwe, a.as_tensor(), c.as_tensor(), k).unwrap();
        let dh = dy.matmul(&we.t().unwrap()).unwrap();
        for (name, v, g) in [("W", &w, &dw), ("A", &a, &da), ("B", &c, &dc), ("h", &h, &dh)] {
            check(&format!("columns {dt:?} d{name}"), gr.get(v.as_tensor()).unwrap(), g, tol(dt, true));
        }
        // the off-diagonal blocks get exactly 0 (autograd's narrow), as the unfold's mask
        let m = col_mask(di, dout, k, dt, &CPU).unwrap();
        assert!(same(&dw.mul(&m.affine(-1.0, 1.0).unwrap()).unwrap(), &dw.zeros_like().unwrap()), "{dt:?}: dW off the blocks is 0");
    }
}

#[test]
fn relay_twins_match_autograd() {
    for dt in [DType::F64, DType::F32] {
        let (n, d, kk) = (9usize, 20usize, crate::v51b::TH_K);
        let acc = Var::from_tensor(&noise(&CPU, &[n, kk], 31, 2.0, 0.1, dt)).unwrap();
        let g = Var::from_tensor(&noise(&CPU, &[kk, 2], 32, 0.7, 0.0, dt)).unwrap();
        let c = Var::from_tensor(&noise(&CPU, &[2], 33, 0.5, 0.0, dt)).unwrap();
        let dg = noise(&CPU, &[n, 2], 34, 1.0, 0.0, dt);
        let reads = 3;
        let want = crate::v51b::th_gates(acc.as_tensor(), reads, g.as_tensor(), c.as_tensor()).unwrap();
        let gr = want.mul(&dg).unwrap().sum_all().unwrap().backward().unwrap();
        let (f, gates) = th_gates_keep(acc.as_tensor(), reads, g.as_tensor(), c.as_tensor()).unwrap();
        assert!(same(&want, &gates), "{dt:?}: the gates are v51b::th_gates's, bit for bit");
        let (dacc, dgw, dc) = th_gates_bwd(&f, &gates, g.as_tensor(), reads, &dg).unwrap();
        for (name, v, got) in [("acc", &acc, &dacc), ("G", &g, &dgw), ("c", &c, &dc)] {
            check(&format!("relay gates {dt:?} d{name}"), gr.get(v.as_tensor()).unwrap(), got, tol(dt, true));
        }
        let x = Var::from_tensor(&noise(&CPU, &[n, d], 35, 1.5, 0.2, dt)).unwrap();
        let r = Var::from_tensor(&noise(&CPU, &[d, kk], 36, 0.3, 0.0, dt)).unwrap();
        let ones = Tensor::ones(d, dt, &CPU).unwrap();
        let da = noise(&CPU, &[n, kk], 37, 1.0, 0.0, dt);
        let rd = crate::v51b::th_read(x.as_tensor(), &ones, r.as_tensor()).unwrap();
        let gr = rd.mul(&da).unwrap().sum_all().unwrap().backward().unwrap();
        let xn = norm(x.as_tensor(), &ones).unwrap();
        assert!(same(&rd, &xn.matmul(r.as_tensor()).unwrap()), "{dt:?}: the read is v51b::th_read's");
        let (dx, dr) = th_read_bwd(x.as_tensor(), &xn, r.as_tensor(), &da, &ones).unwrap();
        check(&format!("relay read {dt:?} dx"), gr.get(x.as_tensor()).unwrap(), &dx, tol(dt, true));
        check(&format!("relay read {dt:?} dR"), gr.get(r.as_tensor()).unwrap(), &dr, tol(dt, true));
    }
}

#[test]
fn granule_code_is_v51b_cb_code() {
    for (d, g, n) in [(64usize, 512usize, 13usize), (128, 4096, 5)] {
        let xf = noise(&CPU, &[n, d], 41, 1.0, 0.0, DType::F32);
        let emb = noise(&CPU, &[n, d], 42, 0.7, 0.1, DType::F32);
        let ones = Tensor::ones(d, DType::F32, &CPU).unwrap();
        let (idx, sgn) = cb_wiring(2 * d, g, &CPU).unwrap();
        let (i2, s2) = crate::v51b::cb_wiring(2 * d, g, crate::v51b::CB_FAN_IN, crate::v51b::CB_SEED);
        assert_eq!(idx.to_vec1::<u32>().unwrap(), i2, "the wiring is v51b's");
        assert_eq!(sgn.flatten_all().unwrap().to_vec1::<f32>().unwrap(), s2, "its signs too");
        let k = g / crate::v51b::CB_SPARSITY;
        let want = crate::v51b::cb_code(&xf, &emb, &ones, &idx, &sgn, k).unwrap();
        let got = cb_code(&cb_input(&xf, &emb, &ones).unwrap(), &idx, &sgn, k).unwrap();
        assert!(same(&want, &got), "d{d} G{g}: the code is v51b::cb_code's, bit for bit");
        let nz: f32 = got.ne(0.0).unwrap().to_dtype(DType::F32).unwrap().sum_all().unwrap().to_scalar().unwrap();
        eprintln!("granule code d{d} G{g}: {nz} of {} units on (k {k} per row)", n * g);
        assert!(nz > 0.0 && nz <= (n * k) as f32 + n as f32 * 2.0, "about k winners per row");
    }
}

// ---------------------------------------------------------------- the hooks on a CPU model

fn cpu_model(vocab: usize, d: usize, layers: usize, flags: &str) -> (VarMap, Model) {
    let cfg = Config::with_kind(Kind::D, vocab, d, layers, 16).with_arch(Arch::from_flags(flags).unwrap()).unwrap();
    let vm = VarMap::new();
    let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
    (vm, m)
}

fn set(vm: &VarMap, name: &str, t: &Tensor) {
    vm.data().lock().unwrap()[name].set(t).unwrap();
}

fn ids_of(vocab: usize, n: usize, seed: u64) -> Tensor {
    let r = unit_noise(n, seed);
    let v: Vec<u32> = r.iter().map(|x| (((x + 1.0) * 0.5 * vocab as f32) as usize).min(vocab - 1) as u32).collect();
    Tensor::from_vec(v, (1, n), &CPU).unwrap()
}

fn empty_grads() -> GradStore {
    Tensor::zeros((), DType::F32, &CPU).unwrap().backward().unwrap()
}

const MM: Mm = Mm { bf16: false };

#[test]
fn letters_hooks_match_autograd() {
    // vocab 293: a table of this test's own (the other tests' tables are kept per vocabulary size)
    let vocab = 293;
    crate::v51::set_letters(table(vocab, 51));
    let (vm, m) = cpu_model(vocab, 64, 2, "--letters");
    set(&vm, "v51.letters.wo", &noise(&CPU, &[64, 64], 52, 0.3, 0.0, DType::F32));
    let ids = ids_of(vocab, 320, 53); // more tokens than the vocabulary: v51::embed's training branch
    let idf = ids.flatten_all().unwrap();
    let x = Var::from_tensor(&noise(&CPU, &[320, 64], 54, 1.0, 0.0, DType::F32)).unwrap();
    let r = noise(&CPU, &[320, 64], 55, 1.0, 0.0, DType::F32);
    let want = crate::v51::embed(&m, &ids, x.as_tensor().clone()).unwrap();
    let gr = want.mul(&r).unwrap().sum_all().unwrap().backward().unwrap();
    let mut f2 = F2::new(&m, &idf, &MM).unwrap();
    let got = f2.embed(x.as_tensor().detach()).unwrap();
    assert!(same(&want, &got), "F2::embed is v51::embed, bit for bit");
    let mut grads = empty_grads();
    f2.finish(&r, &mut grads).unwrap();
    for name in ["byte", "pos", "end", "wo"] {
        let v = &vm.data().lock().unwrap()[&format!("v51.letters.{name}")].clone();
        check(&format!("letters hooks d{name}"), gr.get(v.as_tensor()).unwrap(), grads.get(v.as_tensor()).expect("a gradient"), CHAIN);
    }
}

#[test]
fn relay_hooks_match_autograd() {
    // three toy blocks around TH's hooks, called in src/model/fwin.rs's order: y = x A_i, x_m2 = x + y g_mem,
    // mlp = tanh(x_m2 B_i), x_out = th_res(x, x_m2 + mlp, g_res), the relay reading every block's input
    let (l, d, n) = (3usize, 64usize, 24usize);
    let (vm, m) = cpu_model(258, d, l, "--thalamus");
    set(&vm, "th.gate", &noise(&CPU, &[l, crate::v51b::TH_K, 2], 61, 0.8, 0.0, DType::F32));
    set(&vm, "th.bias", &noise(&CPU, &[l, 2], 62, 0.5, 0.0, DType::F32));
    let a: Vec<Tensor> = (0..l).map(|i| noise(&CPU, &[d, d], 63 + i as u64, 0.15, 0.0, DType::F32)).collect();
    let b: Vec<Tensor> = (0..l).map(|i| noise(&CPU, &[d, d], 73 + i as u64, 0.2, 0.0, DType::F32)).collect();
    let x0 = Var::from_tensor(&noise(&CPU, &[n, d], 64, 1.0, 0.0, DType::F32)).unwrap();
    let r = noise(&CPU, &[n, d], 65, 1.0, 0.0, DType::F32);
    let th = m.v51b().unwrap().th.as_ref().unwrap();
    let ones = Tensor::ones(d, DType::F32, &CPU).unwrap();
    // reference: v51b's own functions, autograd
    let mut acc = crate::v51b::th_read(x0.as_tensor(), &ones, &th.read.get(0).unwrap()).unwrap();
    let mut reads = 1;
    let mut xr = x0.as_tensor().clone();
    for i in 0..l {
        let g = crate::v51b::th_gates(&acc, reads, &th.gate.get(i).unwrap(), &th.bias.get(i).unwrap()).unwrap();
        let x_m2 = xr.add(&xr.matmul(&a[i]).unwrap().broadcast_mul(&g.narrow(1, 1, 1).unwrap()).unwrap()).unwrap();
        let x_raw = x_m2.add(&x_m2.matmul(&b[i]).unwrap().tanh().unwrap()).unwrap();
        xr = crate::v51b::th_res(&xr, &x_raw, &g.narrow(1, 0, 1).unwrap()).unwrap();
        if i + 1 < l {
            acc = acc.add(&crate::v51b::th_read(&xr, &ones, &th.read.get(i + 1).unwrap()).unwrap()).unwrap();
            reads += 1;
        }
    }
    let gr = xr.mul(&r).unwrap().sum_all().unwrap().backward().unwrap();
    // the hooks
    let idf = ids_of(258, n, 66).flatten_all().unwrap();
    let mut f2 = F2::new(&m, &idf, &MM).unwrap();
    let mut x = x0.as_tensor().detach();
    f2.relay_begin(&x).unwrap();
    let mut keep = Vec::new();
    for i in 0..l {
        f2.block_begin(i).unwrap();
        let x_m2 = x.add(&f2.mem_out(i, x.matmul(&a[i]).unwrap()).unwrap()).unwrap();
        let mlp = x_m2.matmul(&b[i]).unwrap().tanh().unwrap();
        let (xa, xb) = f2.res_in(i, &x, x_m2.clone(), mlp.clone()).unwrap();
        let out = xa.add(&xb).unwrap();
        f2.relay_after(i, &out).unwrap();
        keep.push((x.clone(), mlp));
        x = out;
    }
    assert!(same(&xr, &x), "the hooks' forward is v51b's, bit for bit");
    let mut dx = r.clone();
    for i in (0..l).rev() {
        let (x_in, mlp) = &keep[i];
        if let Some(rr) = f2.res_back(i, &dx).unwrap() {
            dx = dx.add(&rr).unwrap();
        }
        dx = dx.add(&dx.mul(&mlp.sqr().unwrap().affine(-1.0, 1.0).unwrap()).unwrap().matmul(&b[i].t().unwrap()).unwrap()).unwrap();
        let dy = f2.mem_back(i, &dx, &dx).unwrap();
        dx = dx.add(&dy.matmul(&a[i].t().unwrap()).unwrap()).unwrap();
        if let Some(rr) = f2.block_back(i).unwrap() {
            dx = dx.add(&rr).unwrap();
        }
        if let Some(rr) = f2.read_grad(i, x_in).unwrap() {
            dx = dx.add(&rr).unwrap();
        }
    }
    let mut grads = empty_grads();
    f2.finish(&dx, &mut grads).unwrap();
    check("relay hooks dx_0", gr.get(x0.as_tensor()).unwrap(), &dx, CHAIN);
    for (name, t) in [("read", &th.read), ("gate", &th.gate), ("bias", &th.bias)] {
        check(&format!("relay hooks d{name}"), gr.get(t).unwrap(), grads.get(t).expect("a gradient"), CHAIN);
    }
}

#[test]
fn head_hooks_match_autograd() {
    let (d, n) = (64usize, 16usize);
    let (vm, m) = cpu_model(258, d, 2, "--cerebellum-head 512");
    set(&vm, "cbh.wo", &noise(&CPU, &[512, d], 81, 0.2, 0.0, DType::F32));
    let ids = ids_of(258, n, 82);
    let idf = ids.flatten_all().unwrap();
    let xf = noise(&CPU, &[n, d], 83, 1.0, 0.0, DType::F32);
    let r = noise(&CPU, &[n, d], 84, 1.0, 0.0, DType::F32);
    let want = m.v51b_cb(xf.clone(), &ids).unwrap();
    let gr = want.mul(&r).unwrap().sum_all().unwrap().backward().unwrap();
    let mut f2 = F2::new(&m, &idf, &MM).unwrap();
    let gf = Tensor::ones(d, DType::F32, &CPU).unwrap();
    let got = f2.head_in(&xf, &gf, xf.clone()).unwrap();
    check("head hooks: xf + correction (dense read-out vs v51b's CPU gather)", &want, &got, 2e-6);
    let mut grads = empty_grads();
    f2.head_bwd(&r, &mut grads).unwrap();
    let w = m.cb_weight().unwrap();
    check("head hooks dW", gr.get(w).unwrap(), grads.get(w).expect("a gradient"), tol(DType::F32, true));
}

#[test]
fn column_hooks_match_autograd() {
    let (d, n) = (64usize, 10usize);
    let (vm, m) = cpu_model(258, d, 2, "--columns 4");
    let hid = m.cfg.hidden();
    let names: Vec<String> = vm.data().lock().unwrap().keys().filter(|k| k.contains(".v51.col_")).cloned().collect();
    for (k, name) in names.iter().enumerate() {
        let t = vm.data().lock().unwrap()[name].as_tensor().clone();
        set(&vm, name, &noise(&CPU, t.dims(), 90 + k as u64, 0.2, 0.0, DType::F32));
    }
    let idf = ids_of(258, n, 91).flatten_all().unwrap();
    let f2 = F2::new(&m, &idf, &MM).unwrap();
    for i in 0..2 {
        let l = &m.blocks[i].mlp;
        let cols = f2.cols_of(i).expect("column weights");
        let h = noise(&CPU, &[n, d], 92, 1.0, 0.0, DType::F32);
        let a = noise(&CPU, &[n, hid], 93, 1.0, 0.0, DType::F32);
        let (r1, r2) = (noise(&CPU, &[n, hid], 94, 1.0, 0.0, DType::F32), noise(&CPU, &[n, d], 95, 1.0, 0.0, DType::F32));
        let u = crate::v51::col_mm(&h, &l.w1, &cols[0], &cols[1], 4).unwrap();
        let y = crate::v51::col_mm(&a, &l.w2, &cols[2], &cols[3], 4).unwrap();
        let gr = u.mul(&r1).unwrap().sum_all().unwrap().add(&y.mul(&r2).unwrap().sum_all().unwrap()).unwrap().backward().unwrap();
        let (w1, w2) = (f2.w1(i, &l.w1).unwrap(), f2.w2(i, &l.w2).unwrap());
        check(&format!("column hooks block {i}: u"), &u, &h.matmul(&w1).unwrap(), tol(DType::F32, true));
        check(&format!("column hooks block {i}: y"), &y, &a.matmul(&w2).unwrap(), tol(DType::F32, true));
        let mut grads = empty_grads();
        f2.put_w(i, false, &l.w1, h.t().unwrap().matmul(&r1).unwrap(), &mut grads).unwrap();
        f2.put_w(i, true, &l.w2, a.t().unwrap().matmul(&r2).unwrap(), &mut grads).unwrap();
        for (name, t) in [("W1", &l.w1), ("W2", &l.w2), ("A1", &cols[0]), ("B1", &cols[1]), ("A2", &cols[2]), ("B2", &cols[3])] {
            check(&format!("column hooks block {i} d{name}"), gr.get(t).unwrap(), grads.get(t).expect("a gradient"), tol(DType::F32, true));
        }
        // kWTA per column: the rows reshaped [N K, hidden / K] with k / K, and back
        let (uc, kc) = f2.cols_in(u.clone(), m.cfg.kwta_k()).unwrap();
        assert_eq!((uc.dims(), kc), (&[n * 4, hid / 4][..], m.cfg.kwta_k() / 4));
        let (_, back) = f2.cols_out(None, uc, n).unwrap();
        assert!(same(&back, &u), "the column view goes back to the rows");
    }
}

#[test]
fn routing_takes_the_four_parts() {
    const RECIPE: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense --resid-fix \
                          --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1";
    let c = |extra: &str| Config::with_kind(Kind::D, 258, 64, 2, 16).with_arch(Arch::from_flags(&format!("{RECIPE} {extra}")).unwrap());
    for extra in ["--letters", "--columns 4", "--thalamus", "--cerebellum-head 512", "--letters --columns 4 --thalamus --cerebellum-head 512"] {
        for nm in ["", " --neuromod 3"] {
            let cfg = c(&format!("{extra}{nm}")).unwrap();
            assert!(super::super::step_ok(&cfg), "{extra}{nm}: the fused WIN step takes it");
            assert!(super::super::would_take(&cfg, true) || !super::super::on(), "{extra}{nm}");
        }
    }
    // merge_f (runs/v51/merge_f.md): with v51 fast1 merged, the four with fast1's six take the fused step too; --loop
    // still keeps autograd
    for extra in ["--letters --dendrite", "--columns 4 --dopamine", "--thalamus --rhythm", "--cerebellum-head 512 --mood"] {
        let cfg = c(extra).unwrap();
        assert!(super::super::step_ok(&cfg), "{extra}: the fused WIN step takes it (merged)");
    }
    for extra in ["--thalamus --loop 2"] {
        if let Ok(cfg) = c(extra) {
            assert!(!super::super::step_ok(&cfg), "{extra}: keeps autograd");
        }
    }
    // v51 cbfix (runs/v51/cbfix.md): the device's own CB bound is CB_MAX_G on the CPU (and the CPU takes no fused step)
    assert_eq!(cb_max_g(&Device::Cpu), CB_MAX_G);
    for g in [512, CB_MAX_G] {
        let cfg = c(&format!("--cerebellum-head {g}")).unwrap();
        assert!(device_ok(&cfg, &Device::Cpu) && !super::super::would_take_on(&cfg, &Device::Cpu), "CB {g} on the CPU");
    }
    // a model whose talk-time state the step does not carry keeps autograd
    let (_, m) = cpu_model(258, 64, 2, "--cerebellum-head 512");
    assert!(model_ok(&m));
    m.cb_online_begin(1, 0.1).unwrap();
    assert!(!model_ok(&m), "an online table");
    let (_, m) = cpu_model(258, 64, 2, "--thalamus");
    m.set_part_signals(true);
    assert!(!model_ok(&m), "the chat app's part signals");
    m.set_part_signals(false);
    assert!(model_ok(&m));
}

// ---------------------------------------------------------------- CUDA: the kernels against the twins

fn cuda() -> Option<Device> {
    if std::env::var("AVA_GPU_TEST_CUDA").map(|v| v == "1").unwrap_or(false) {
        Some(Device::new_cuda(0).expect("CUDA device 0"))
    } else {
        eprintln!("fwin_f2 kernels: skipped (set AVA_GPU_TEST_CUDA=1)");
        None
    }
}

#[test]
fn kernels_match_the_twins() {
    let Some(dev) = cuda() else { return };
    let dt = DType::F32;
    // R at a real vocabulary's size (and a small one)
    for (v, seed) in [(16384usize, 101u64), (300, 102)] {
        let (ids, eidx, mask, inv) = rows(&table(v, seed), &dev);
        let dd = crate::v51::LETTER_DIM;
        let (byte, pos, end) = (noise(&dev, &[256, dd], seed + 1, 0.6, 0.0, dt), noise(&dev, &[16, dd], seed + 2, 0.6, 0.0, dt), noise(&dev, &[16, dd], seed + 3, 0.6, 0.0, dt));
        let t = LetTabs { byte: &byte, pos: &pos, end: &end, ids: &ids, eidx: &eidx, mask: &mask, inv: &inv };
        let (k, o) = (let_fwd_k(&t).unwrap(), let_fwd_ops(&t).unwrap());
        eprintln!("f2_let_fwd V{v}: bit for bit {}", same(&o, &k));
        check(&format!("f2_let_fwd V{v}"), &o, &k, 1e-6);
        let dl = noise(&dev, &[v, dd], seed + 4, 1.0, 0.0, dt);
        let (ke, kp, kq) = let_bwd_k(&t, &dl).unwrap();
        let (oe, op, oq) = let_bwd_ops(&t, &dl).unwrap();
        for (name, a, b) in [("byte", &oe, &ke), ("pos", &op, &kp), ("end", &oq, &kq)] {
            check(&format!("f2_let_bwd V{v} d{name}"), a, b, 1e-5);
        }
    }
    // CB at d64, d128 and d384 (G 12288: the shared-memory opt-in); v51 cbfix: and d640 (G 20480 = CB_MAX_G)
    for (d, g, n) in [(64usize, 512usize, 37usize), (128, 4096, 300), (384, 12288, 9), (640, 20480, 9)] {
        let u = Tensor::cat(&[noise(&dev, &[n, d], 111, 1.0, 0.0, dt), noise(&dev, &[n, d], 112, 1.2, 0.0, dt)], 1).unwrap();
        let (idx, sgn) = cb_wiring(2 * d, g, &dev).unwrap();
        let k = g / crate::v51b::CB_SPARSITY;
        let (kc, oc) = (cb_code_k(&u, &idx, &sgn, k).unwrap(), cb_code_ops(&u, &idx, &sgn, k).unwrap());
        let (mk, mo) = (kc.ne(0.0).unwrap(), oc.ne(0.0).unwrap());
        assert!(same(&mk.to_dtype(dt).unwrap(), &mo.to_dtype(dt).unwrap()), "f2_cb_code d{d} G{g}: the same winners");
        eprintln!("f2_cb_code d{d} G{g}: bit for bit {}", same(&oc, &kc));
        check(&format!("f2_cb_code d{d} G{g}"), &oc, &kc, 1e-6);
    }
    // v51 cbfix (runs/v51/cbfix.md): the device's limits and the largest G the step takes on it; kwta_mask_rows (the
    // twin's and the reference path's kWTA, which failed at G 12288) on wide rows against the CPU's sort, up to the
    // widest row this device holds; one past it a plain error, not the driver's "invalid argument"
    let Device::Cuda(cd) = &dev else { unreachable!("cuda() gives a CUDA device") };
    let (base, optin) = crate::lif::smem_limits(cd).unwrap();
    let st = |name: &str| cd.get_or_load_custom_func(name, MODULE, PTX).unwrap().shared_size_bytes().unwrap() as usize;
    let (st_cb, st_kw) = (st("f2_cb_code"), st("kwta_mask_rows"));
    let gmax = cb_max_g(&dev);
    eprintln!("shared memory per block: {base} B default, {optin} B opt-in; static: f2_cb_code {st_cb} B, kwta_mask_rows {st_kw} B");
    eprintln!("cb_max_g: {gmax} (CB_MAX_G {CB_MAX_G}; the kernel's fit {}); kwta_mask_rows widest row {}", (optin - st_cb) / 4, (optin - st_kw) / 4);
    assert_eq!(gmax, ((optin - st_cb) / 4).min(CB_MAX_G), "cb_max_g");
    let hmax = (optin - st_kw) / 4;
    for (h, n) in [(12030usize, 5usize), (12288, 5), (20480, 5), (hmax, 3)] {
        let x = noise(&dev, &[n, h], 120 + h as u64, 1.0, 0.0, dt);
        let k = h / crate::v51b::CB_SPARSITY;
        let gpu = crate::lif::kwta_mask_fast(&x, k).unwrap().to_device(&Device::Cpu).unwrap();
        let cpu = crate::lif::kwta_mask_fast(&x.to_device(&Device::Cpu).unwrap(), k).unwrap();
        let ok = same(&gpu, &cpu);
        eprintln!("kwta_mask_rows H{h}: the CPU sort's mask bit for bit {ok}");
        assert!(ok, "kwta_mask_rows H{h}: the mask differs from the CPU sort's");
    }
    let x = noise(&dev, &[2, hmax + 1], 130, 1.0, 0.0, dt);
    let e = crate::lif::kwta_mask_fast(&x, 8).unwrap_err().to_string();
    eprintln!("kwta_mask_rows H{}: {e}", hmax + 1);
    assert!(e.contains("opt-in") && !e.contains("invalid argument"), "kwta_mask_rows past the limit: {e}");
    // the step's routing on this device: G within cb_max_g taken, the next multiple of 32 past it (if a model) refused
    const RECIPE: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense --resid-fix \
                          --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1";
    for g in [12288usize, 20480, gmax / 32 * 32, gmax / 32 * 32 + 32] {
        let cfg = Config::with_kind(Kind::D, 258, 64, 2, 16).with_arch(Arch::from_flags(&format!("{RECIPE} --cerebellum-head {g}")).unwrap()).unwrap();
        let ok = device_ok(&cfg, &dev);
        eprintln!("routing on this device, CB G{g}: device_ok {ok}, would_take_on {}", super::super::would_take_on(&cfg, &dev));
        assert_eq!(ok, g <= gmax, "device_ok G{g}");
        let take = super::super::would_take_on(&cfg, &dev);
        assert!(if ok && super::super::step_ok(&cfg) { take || !super::super::on() } else { !take }, "would_take_on G{g}");
    }
}
