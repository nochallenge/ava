// TESTING / DESIGN ONLY — mem_probe: per-step CUDA memory of the training step, to find what grows.
//
// mem_probe <fwd|bwd|full> [steps 60] [d 384] [layers 8] [ctx 256] [batch 32]
// Prints, every 5 steps after a device sync: the default pool's live bytes (USED) and reserved bytes
// (RESERVED), and the peak live bytes so far. USED growing = tensors kept alive (a leak in our code or candle); RESERVED growing with
// USED flat = the pool holding freed memory.

use ava_gpu::model::{Config, Model, VOCAB};
use ava_gpu::optim::{AdamW, Param};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use cudarc::driver::sys::{cuDeviceGetDefaultMemPool, cuMemPoolGetAttribute, CUmemPool_attribute, CUresult};

fn pool_attr(attr: CUmemPool_attribute) -> u64 {
    unsafe {
        let mut pool = std::mem::MaybeUninit::uninit();
        assert_eq!(cuDeviceGetDefaultMemPool(pool.as_mut_ptr(), 0), CUresult::CUDA_SUCCESS);
        let mut v: u64 = 0;
        assert_eq!(cuMemPoolGetAttribute(pool.assume_init(), attr, &mut v as *mut u64 as *mut _), CUresult::CUDA_SUCCESS);
        v
    }
}

fn main() -> candle_core::Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let mode = a.first().map(String::as_str).unwrap_or("full").to_string();
    let num = |i: usize, def: usize| a.get(i).and_then(|s| s.parse().ok()).unwrap_or(def);
    let (steps, d, layers, ctx, batch) = (num(1, 60), num(2, 384), num(3, 8), num(4, 256), num(5, 32));
    let dev = Device::new_cuda(0)?;
    dev.set_seed(1)?;
    let varmap = VarMap::new();
    let model = Model::new(Config::new(VOCAB, d, layers, ctx), VarBuilder::from_varmap(&varmap, DType::F32, &dev))?;
    let params: Vec<Param> = varmap.data().lock().unwrap().iter().map(|(k, v)| Param { name: k.clone(), var: v.clone(), decay: Model::decays(k) }).collect();
    let mut opt = AdamW::new(params, 0.1)?;
    let ids = Tensor::rand(0f32, VOCAB as f32, (batch, ctx), &dev)?.to_dtype(DType::U32)?;
    let tg = Tensor::rand(0f32, VOCAB as f32, (batch, ctx), &dev)?.to_dtype(DType::U32)?;
    println!("mode {mode} d {d} layers {layers} ctx {ctx} batch {batch}");
    for step in 0..steps {
        let (loss, act) = model.loss(&ids, &tg)?;
        if mode != "fwd" {
            let grads = loss.backward()?;
            if mode == "full" {
                let g = opt.grad_norm(&grads)?;
                opt.step(&grads, 1e-4, if g > 1.0 { 1.0 / g } else { 1.0 })?;
            }
        }
        let l = loss.to_scalar::<f32>()?;
        drop((loss, act));
        if step % 5 == 0 || step + 1 == steps {
            dev.synchronize()?;
            let used = pool_attr(CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT) as f64 / 1048576.0;
            let res = pool_attr(CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT) as f64 / 1048576.0;
            let peak = pool_attr(CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH) as f64 / 1048576.0;
            println!("step {step:4} loss {l:.4} used {used:8.1} MiB reserved {res:8.1} MiB peak live {peak:8.1} MiB");
        }
    }
    Ok(())
}
