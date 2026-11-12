// TESTING / DESIGN ONLY — merge_f (runs/v51/merge_f.md): round 2's ten parts with --neuromod 3 together on the fused
// WIN step, the CPU half. The step itself needs CUDA: tests/fast2_parts.rs merged_ten_parts_step_matches_autograd, run
// by runs/v51/fast_merged_check.sh.
// * Routing: on Ava 0.4b's recipe every subset of the ten parts (P S R DA M W Pr Y TH CB), with and without
//   --neuromod 3, that Arch::validate allows (all but those with both W and Pr: validate refuses --prune with
//   --columns) takes the fused step; --loop 2, --dale and CB past 20480 units keep autograd; a model whose talk-time
//   state the step does not carry (the chat app's part signals) keeps it too.
// * Pieces: all ten cannot be on at once, so two CPU models with nine parts and --neuromod 3, set A = all but W and set
//   B = all but Pr (d128, 3 blocks, vocabulary 301 with a letter table of this test's own, the zero-start weights moved
//   off 0). On each, the fused step's pieces where the two builders' parts meet, against the reference path's own
//   functions (forward: bit for bit) and candle's autograd of them (backward: f32 within 2e-5, the bar both builders
//   used for chains; 1e-4 for what passes through the rhythm scan's sums, as F-2):
//   - R then Y: F2::embed (the letters), fwin_r2::begin on its output (the rhythms read the lettered embedding, as
//     v51::begin after v51::embed), every block's read gate, back through fwin_r2::finish (Y's embedding share) and
//     F2::finish (the letter tables);
//   - P and TH at the memory read: the dendrites read the read before TH's memory gate; the merged backward
//     (fwin::dread_op) against autograd, with and without the dendrites' share;
//   - W's fold and unfold (set B), DA's and M's signals after block 0 (fwin_r2::after0 vs v51::after0) and CB's head
//     (F2::head_in / head_bwd vs v51b_cb) on the same models.

use super::f2::{model_ok, F2};
use super::{dread_op, on, step_ok, would_take};
use crate::arch::Arch;
use crate::fwin_r2::{self as r2, Blk, Bwd};
use crate::model::{head_norm, Config, Kind, Mm, Model};
use crate::testkit::unit_noise;
use crate::v51::gain;
use candle_core::backprop::GradStore;
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::{VarBuilder, VarMap};

const CPU: Device = Device::Cpu;
const MM: Mm = Mm { bf16: false };
const CHAIN: f64 = 2e-5;
const SCAN: f64 = 1e-4;
const RECIPE: &str = "--head-size 64 --timescales fixed --value-residual --kwta 0.25 --mlp-act relu2 --residual dense --resid-fix \
                      --lif-init logspaced --lif-scale --gates channel --mem-momentum 0.5 --spikes int2 --alif --homeostasis 0.1";
const PARTS: [&str; 10] = ["--dendrite", "--stp", "--letters", "--dopamine", "--mood", "--columns 4", "--prune", "--rhythm", "--thalamus", "--cerebellum-head 512"];
const W: usize = 5;
const PR: usize = 6;
const SET_A: &str = "--dendrite --stp --letters --dopamine --mood --prune --rhythm --thalamus --cerebellum-head 512";
const SET_B: &str = "--dendrite --stp --letters --dopamine --mood --columns 4 --rhythm --thalamus --cerebellum-head 512";
const V: usize = 301;

fn cfg_of(vocab: usize, d: usize, layers: usize, extra: &str) -> std::result::Result<Config, String> {
    Config::with_kind(Kind::D, vocab, d, layers, 16).with_arch(Arch::from_flags(&format!("{RECIPE} {extra}"))?)
}

fn noise(shape: &[usize], seed: u64, scale: f64) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor::from_vec(unit_noise(n, seed), shape, &CPU).unwrap().affine(scale, 0.0).unwrap()
}

/// max |a - b| / max |a| (f64).
fn rel(a: &Tensor, b: &Tensor) -> f64 {
    let a = a.to_dtype(DType::F64).unwrap().flatten_all().unwrap();
    let b = b.to_dtype(DType::F64).unwrap().flatten_all().unwrap();
    assert_eq!(a.dims(), b.dims(), "shapes");
    let d = a.sub(&b).unwrap().abs().unwrap().max(0).unwrap().to_scalar::<f64>().unwrap();
    let m = a.abs().unwrap().max(0).unwrap().to_scalar::<f64>().unwrap();
    d / m.max(1e-30)
}

fn same(tag: &str, a: &Tensor, b: &Tensor) {
    let e = rel(a, b);
    eprintln!("{tag}: rel {e:.2e} (bit for bit)");
    assert!(e == 0.0, "{tag}: not bit for bit (rel {e})");
}

fn check(tag: &str, want: &Tensor, got: &Tensor, bar: f64) {
    let e = rel(want, got);
    eprintln!("{tag}: rel {e:.2e} (bar {bar:.0e})");
    assert!(e <= bar, "{tag}: rel {e} > {bar}");
}

fn empty_grads() -> GradStore {
    Tensor::zeros((), DType::F32, &CPU).unwrap().backward().unwrap()
}

/// The letter table of this test's own vocabulary (0..=20 bytes per token, some empty).
fn letters() {
    let r = unit_noise(V * 22, 301);
    let t: Vec<Vec<u8>> = (0..V)
        .map(|v| {
            let len = ((r[v * 22] + 1.0) * 10.5) as usize % 21;
            (0..len).map(|j| (((r[v * 22 + 1 + j] + 1.0) * 128.0) as usize).min(255) as u8).collect()
        })
        .collect();
    crate::v51::set_letters(t);
}

/// A d128 L3 model with `set` + --neuromod 3: every weight that starts at 0 moved off it (0.3), the relay's gates more
/// (0.8, as fast2's tests), the v51 weights moved as fwin_r2's tests move them (0.4; the reset bias 1.0), Pr's cut
/// units left as built.
fn model_of(set: &str, seed: u64) -> (VarMap, Model) {
    letters();
    let cfg = cfg_of(V, 128, 3, &format!("{set} --neuromod 3")).unwrap();
    let vm = VarMap::new();
    let m = Model::new(cfg, VarBuilder::from_varmap(&vm, DType::F32, &CPU)).unwrap();
    let mut names: Vec<(String, Var)> = vm.data().lock().unwrap().iter().map(|(k, v)| (k.clone(), v.clone())).collect();
    names.sort_by(|a, b| a.0.cmp(&b.0));
    for (k, (name, var)) in names.into_iter().enumerate() {
        let t = var.as_tensor();
        let s = seed * 1000 + k as u64;
        let zero = t.flatten_all().unwrap().to_vec1::<f32>().unwrap().iter().all(|&x| x == 0.0);
        let new = if name == "th.gate" {
            noise(t.dims(), s, 0.8)
        } else if name.ends_with(".alive") || name.ends_with(".use") || name.contains(".letters.") && !name.ends_with(".wo") {
            continue;
        } else if name.contains("v51") && !name.contains(".col_") {
            t.add(&noise(t.dims(), s, if name.ends_with(".br") { 1.0 } else { 0.4 })).unwrap()
        } else if zero {
            noise(t.dims(), s, 0.3)
        } else {
            continue;
        };
        var.set(&new).unwrap();
    }
    m.v51_schedule(0, 1000);
    (vm, m)
}

fn ids_of(b: usize, t: usize, seed: u64) -> Tensor {
    let v: Vec<u32> = unit_noise(b * t, seed).iter().map(|x| (((x + 1.0) * 0.5 * V as f32) as usize).min(V - 1) as u32).collect();
    Tensor::from_vec(v, (b, t), &CPU).unwrap()
}

fn v51(m: &Model) -> &crate::v51::V51Ext {
    &m.ext.as_ref().unwrap().v51
}

/// R then Y (and their backward): the letters join the embedding before the rhythms read it.
fn letters_then_rhythms(tag: &str, m: &Model) {
    let (b, t, d, l) = (2usize, 24usize, m.cfg.d, m.blocks.len());
    let (n, nh) = (b * t, d / 64);
    let ids = ids_of(b, t, 11);
    let idf = ids.flatten_all().unwrap();
    let v = v51(m);
    let (lt, rp) = (v.letters.as_ref().expect("letters"), v.rhythm.as_ref().expect("rhythm"));
    // the reference: v51::embed, then v51::begin's ops on its output, every block's read gate (v51's rh_gate on the
    // read-out per head), autograd
    let x0 = Var::from_tensor(&noise(&[n, d], 12, 1.0)).unwrap();
    let xl = crate::v51::embed(m, &ids, x0.as_tensor().clone()).unwrap();
    let pre = head_norm(&xl).unwrap().matmul(&rp.wr).unwrap().broadcast_add(&rp.br).unwrap().reshape((b, t, 2)).unwrap();
    let a = crate::v51::rhythm_scan(&pre, &Tensor::zeros((b, 2), DType::F32, &CPU).unwrap()).unwrap();
    let fr = crate::v51::rhythm_feats(&a, &rp.lw).unwrap();
    crate::v51::begin(m, &xl.detach(), b, t, None).unwrap();
    let rh_ref = v.sig.borrow().rh.clone().expect("v51::begin's phases");
    crate::v51::end(m);
    let r = noise(&[n, d], 13, 1.0);
    let mut loss = xl.mul(&r).unwrap().sum_all().unwrap();
    let mut os = Vec::new();
    for j in 0..l {
        let (_, gr) = v.blocks[j].rh.as_ref().expect("Y's gates");
        let o = Var::from_tensor(&noise(&[n, d], 20 + j as u64, 1.0)).unwrap();
        let r2 = noise(&[n, d], 30 + j as u64, 1.0);
        let gate = gain(&fr.matmul(gr).unwrap()).unwrap();
        let og = o.as_tensor().reshape((n, nh, d / nh)).unwrap().broadcast_mul(&gate.unsqueeze(2).unwrap()).unwrap().reshape((n, d)).unwrap();
        loss = loss.add(&og.mul(&r2).unwrap().sum_all().unwrap()).unwrap();
        os.push((o, r2, og, gr.clone()));
    }
    let gr = loss.backward().unwrap();
    // the fused step's pieces, in src/model/fwin.rs's order
    let mut f2 = F2::new(m, &idf, &MM).unwrap();
    let xl2 = f2.embed(x0.as_tensor().detach()).unwrap();
    same(&format!("{tag} R: F2::embed is v51::embed"), &xl, &xl2);
    let rf = r2::begin(m, &xl2, b, t).unwrap();
    same(&format!("{tag} R then Y: the phases of the lettered embedding are v51::begin's"), &rh_ref, &rf.rh.as_ref().expect("Y").f);
    let mut bw = Bwd::default();
    let mut gs = empty_grads();
    for (j, (o, r2v, og, _)) in os.iter().enumerate() {
        let (g, th) = r2::rgate(m, j, &rf).unwrap().expect("Y's read gate");
        same(&format!("{tag} Y block {j}: the gated read"), og, &r2::rgate_fwd(o.as_tensor(), &g).unwrap());
        let sv = Blk { rg: Some((g, th, o.as_tensor().detach())), ..Default::default() };
        let d_o = r2::rgate_bwd(m, j, &rf, &sv, r2v, &mut bw, &mut gs).unwrap();
        check(&format!("{tag} Y block {j}: d(read-out)"), gr.get(o.as_tensor()).unwrap(), &d_o, CHAIN);
    }
    let dxr = r2::finish(m, &rf, bw, &xl2, &mut gs).unwrap().expect("Y's embedding share");
    let dx = r.add(&dxr).unwrap();
    check(&format!("{tag} R then Y: dL/d(the embedding)"), gr.get(x0.as_tensor()).unwrap(), &dx, SCAN);
    f2.finish(&dx, &mut gs).unwrap();
    for (name, t) in [("byte", &lt.byte), ("pos", &lt.pos), ("end", &lt.end), ("wo", &lt.wo)] {
        check(&format!("{tag} R: d letters.{name} (through Y's share)"), gr.get(t).unwrap(), gs.get(t).expect("a gradient"), SCAN);
    }
    for (name, t) in [("lw", &rp.lw), ("wr", &rp.wr), ("br", &rp.br)] {
        check(&format!("{tag} Y: d rhythm.{name}"), gr.get(t).unwrap(), gs.get(t).expect("a gradient"), SCAN);
    }
    for (j, (_, _, _, g)) in os.iter().enumerate() {
        check(&format!("{tag} Y block {j}: d read gate"), gr.get(g).unwrap(), gs.get(g).expect("a gradient"), CHAIN);
    }
}

/// P and TH at block i's memory read y: the dendrites read y, TH gates it into the stream.
fn dendrites_and_relay(tag: &str, m: &Model) {
    let (b, t, d, hid) = (2usize, 24usize, m.cfg.d, m.cfg.hidden());
    let n = b * t;
    let i = 1;
    let idf = ids_of(b, t, 41).flatten_all().unwrap();
    let th = m.v51b().and_then(|e| e.th.as_ref()).expect("the relay");
    let (wa, wb) = v51(m).blocks[i].dend.as_ref().expect("the dendrites");
    let ones = Tensor::ones(d, DType::F32, &CPU).unwrap();
    let x = noise(&[n, d], 42, 1.0);
    // the reference: v51b's gates (block 1's, from the relay's read of the blocks' input), v51's dendrite, autograd
    let acc = crate::v51b::th_read(&x, &ones, &th.read.get(0).unwrap()).unwrap();
    let gates = crate::v51b::th_gates(&acc, 1, &th.gate.get(i).unwrap(), &th.bias.get(i).unwrap()).unwrap();
    let g_mem = gates.narrow(1, 1, 1).unwrap();
    let y = Var::from_tensor(&noise(&[n, d], 43, 1.0)).unwrap();
    let a = Var::from_tensor(&noise(&[n, hid], 44, 1.0).relu().unwrap()).unwrap();
    let (r1, r2) = (noise(&[n, d], 45, 1.0), noise(&[n, hid], 46, 1.0));
    let want_mem = y.as_tensor().broadcast_mul(&g_mem).unwrap();
    let want_p = a.as_tensor().mul(&gain(&head_norm(y.as_tensor()).unwrap().matmul(wa).unwrap().matmul(wb).unwrap()).unwrap()).unwrap();
    let gr = want_mem.mul(&r1).unwrap().sum_all().unwrap().add(&want_p.mul(&r2).unwrap().sum_all().unwrap()).unwrap().backward().unwrap();
    // the pieces: the read kept for the dendrites before the gate (r2.ctx), then F2::mem_out
    let mut f2 = F2::new(m, &idf, &MM).unwrap();
    f2.relay_begin(&x).unwrap();
    f2.block_begin(0).unwrap();
    f2.block_begin(1).unwrap();
    let mut sv = Blk { ctx: Some(y.as_tensor().detach()), ..Default::default() };
    let got_mem = f2.mem_out(i, y.as_tensor().detach()).unwrap();
    same(&format!("{tag} TH: the gated memory read"), &want_mem, &got_mem);
    let got_p = r2::dend_fwd(m, i, a.as_tensor().detach(), &mut sv).unwrap();
    same(&format!("{tag} P: the dendrites on the ungated read"), &want_p, &got_p);
    let mut gs = empty_grads();
    let (da, dctx) = r2::dend_bwd(m, i, &sv, &r2, &mut gs).unwrap().expect("P's backward");
    check(&format!("{tag} P: d(units)"), gr.get(a.as_tensor()).unwrap(), &da, CHAIN);
    let dread = dread_op(&mut f2, &MM, i, &r1, &r1, Some(&dctx)).unwrap();
    check(&format!("{tag} P + TH: dread_op = dL/d(the read)"), gr.get(y.as_tensor()).unwrap(), &dread, CHAIN);
    let only_th = dread_op(&mut f2, &MM, i, &r1, &r1, None).unwrap();
    same(&format!("{tag} TH alone: dread_op = dx times the memory gate"), &r1.broadcast_mul(&g_mem).unwrap(), &only_th);
    for (name, t) in [("A", wa), ("B", wb)] {
        check(&format!("{tag} P: d{name}"), gr.get(t).unwrap(), gs.get(t).expect("a gradient"), CHAIN);
    }
}

/// W's fold and unfold on block 0 (set B).
fn columns(tag: &str, m: &Model) {
    let (n, d, hid) = (10usize, m.cfg.d, m.cfg.hidden());
    let idf = ids_of(1, n, 51).flatten_all().unwrap();
    let f2 = F2::new(m, &idf, &MM).unwrap();
    let l = &m.blocks[0].mlp;
    let cols = v51(m).blocks[0].cols.as_ref().expect("column weights");
    let h = noise(&[n, d], 52, 1.0);
    let a = noise(&[n, hid], 53, 1.0);
    let (r1, r2) = (noise(&[n, hid], 54, 1.0), noise(&[n, d], 55, 1.0));
    let u = crate::v51::col_mm(&h, &l.w1, &cols[0], &cols[1], 4).unwrap();
    let y = crate::v51::col_mm(&a, &l.w2, &cols[2], &cols[3], 4).unwrap();
    let gr = u.mul(&r1).unwrap().sum_all().unwrap().add(&y.mul(&r2).unwrap().sum_all().unwrap()).unwrap().backward().unwrap();
    check(&format!("{tag} W: u"), &u, &h.matmul(&f2.w1(0, &l.w1).unwrap()).unwrap(), 5e-6);
    check(&format!("{tag} W: y"), &y, &a.matmul(&f2.w2(0, &l.w2).unwrap()).unwrap(), 5e-6);
    let mut gs = empty_grads();
    f2.put_w(0, false, &l.w1, h.t().unwrap().matmul(&r1).unwrap(), &mut gs).unwrap();
    f2.put_w(0, true, &l.w2, a.t().unwrap().matmul(&r2).unwrap(), &mut gs).unwrap();
    for (name, t) in [("W1", &l.w1), ("W2", &l.w2), ("A1", &cols[0]), ("B1", &cols[1]), ("A2", &cols[2]), ("B2", &cols[3])] {
        check(&format!("{tag} W: d{name}"), gr.get(t).unwrap(), gs.get(t).expect("a gradient"), 5e-6);
    }
}

/// DA's and M's signals after block 0, and CB's head, on the same model.
fn signals_and_head(tag: &str, m: &Model) {
    let (b, t, d) = (2usize, 24usize, m.cfg.d);
    let n = b * t;
    let ids = ids_of(b, t, 61);
    let idf = ids.flatten_all().unwrap();
    let x0 = noise(&[n, d], 62, 1.0);
    crate::v51::after0(m, &x0, &ids, b, t, None).unwrap();
    let (z_ref, mood_ref) = {
        let s = v51(m).sig.borrow();
        (s.z.clone().expect("the surprise"), s.mood.clone().expect("the mood"))
    };
    crate::v51::end(m);
    let mut f = r2::begin(m, &noise(&[n, d], 63, 1.0), b, t).unwrap();
    r2::after0(m, &mut f, &x0, &ids, &m.tok.detach()).unwrap();
    same(&format!("{tag} DA, M: the surprise is v51::after0's"), &z_ref, f.z.as_ref().unwrap());
    same(&format!("{tag} M: the mood is v51::after0's"), &mood_ref, &f.mood.as_ref().unwrap().m2);
    // CB: the dense read-out against v51b's CPU gather (fast2's bar), dW against autograd
    let xf = noise(&[n, d], 64, 1.0);
    let r = noise(&[n, d], 65, 1.0);
    let want = m.v51b_cb(xf.clone(), &ids).unwrap();
    let gr = want.mul(&r).unwrap().sum_all().unwrap().backward().unwrap();
    let mut f2 = F2::new(m, &idf, &MM).unwrap();
    let got = f2.head_in(&xf, &Tensor::ones(d, DType::F32, &CPU).unwrap(), xf.clone()).unwrap();
    check(&format!("{tag} CB: xf + the correction"), &want, &got, 2e-6);
    let mut gs = empty_grads();
    f2.head_bwd(&r, &mut gs).unwrap();
    let w = m.cb_weight().unwrap();
    check(&format!("{tag} CB: dW"), gr.get(w).unwrap(), gs.get(w).expect("a gradient"), 5e-6);
}

#[test]
fn ten_parts_with_neuromod_route_and_their_pieces_match_autograd() {
    // ---- routing: every subset validate allows takes the fused step
    let (mut taken, mut refused) = (0usize, 0usize);
    for nm in ["", " --neuromod 3"] {
        for mask in 0u32..1024 {
            let extra: Vec<&str> = (0..10).filter(|j| mask >> j & 1 == 1).map(|j| PARTS[j]).collect();
            let extra = format!("{}{nm}", extra.join(" "));
            let both = mask >> W & 1 == 1 && mask >> PR & 1 == 1;
            match cfg_of(258, 64, 3, &extra) {
                Err(e) => {
                    assert!(both, "{extra}: Arch::validate refused it ({e})");
                    refused += 1;
                }
                Ok(cfg) => {
                    assert!(!both, "{extra}: validate let W and Pr through");
                    assert!(step_ok(&cfg), "{extra}: the fused WIN step must take it");
                    assert!(would_take(&cfg, true) || !on(), "{extra}: would_take");
                    taken += 1;
                }
            }
        }
    }
    eprintln!("routing: {taken} combinations take the fused step, {refused} refused by Arch::validate (W with Pr)");
    assert_eq!((taken, refused), (2 * 768, 2 * 256));
    // what neither builder fused keeps autograd
    let big = |s: &str| s.replace("--cerebellum-head 512", "--cerebellum-head 20512");
    for extra in [format!("{SET_A} --neuromod 3 --loop 2"), format!("{SET_B} --loop 2"), format!("{SET_B} --neuromod 3 --dale 0.8"), format!("{} --neuromod 3", big(SET_A)), big(SET_B)] {
        let cfg = cfg_of(258, 64, 3, &extra).unwrap_or_else(|e| panic!("{extra}: {e}"));
        assert!(!step_ok(&cfg), "{extra}: must keep autograd");
    }
    let cfg = cfg_of(258, 64, 3, &format!("{} --neuromod 3", SET_A.replace("--cerebellum-head 512", "--cerebellum-head 20480"))).unwrap();
    assert!(step_ok(&cfg), "CB at 20480 units (the kernel's limit) takes the fused step");
    // ---- the pieces on set A and set B (+ --neuromod 3)
    for (tag, set) in [("A (all but W)", SET_A), ("B (all but Pr)", SET_B)] {
        let (_, m) = model_of(set, if set == SET_A { 1 } else { 2 });
        assert!(step_ok(&m.cfg), "{tag}: the fused WIN step takes the model");
        assert!(model_ok(&m), "{tag}: nothing talk-time on");
        m.set_part_signals(true);
        assert!(!model_ok(&m), "{tag}: the chat app's part signals keep autograd");
        m.set_part_signals(false);
        letters_then_rhythms(tag, &m);
        dendrites_and_relay(tag, &m);
        if set == SET_B {
            columns(tag, &m);
        }
        signals_and_head(tag, &m);
    }
}
