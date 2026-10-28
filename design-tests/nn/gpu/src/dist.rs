// TESTING / DESIGN ONLY — v40: data-parallel training over several processes (the roadmap's "many GPUs").
//
// Every process (rank 0 .. world-1) holds the whole model and reads its own share of every batch. After the
// backward, the gradients are summed over all ranks (all-reduce), so every rank takes the same AdamW step and the
// weights stay identical. The loss is written so that the sum over ranks is the whole batch's loss (each rank
// weights its rows by 1 / the batch's total), so W ranks compute what one process computes for the whole batch.
//
// Transport: plain TCP from the standard library. Rank 0 listens, receives every rank's buffer, adds them in rank
// order (the same order every time, so every rank gets bit-identical sums) and sends the sum back. It runs
// anywhere, including Windows and one GPU shared by two processes, which is how it is tested here. It is not
// built for speed at scale: a cluster should replace `all_reduce_sum` with NCCL's ring all-reduce (NVLink /
// InfiniBand); nothing else changes.

use candle_core::backprop::GradStore;
use candle_core::{DType, Tensor, Var};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{Duration, Instant};

/// One rank's links to the others.
pub struct Dist {
    pub rank: usize,
    pub world: usize,
    /// rank 0: one stream per other rank, in rank order; any other rank: one stream to rank 0
    peers: Vec<TcpStream>,
}

fn io<E: std::fmt::Display>(what: &str) -> impl Fn(E) -> String + '_ {
    move |e| format!("dist {what}: {e}")
}

fn write_f32s(s: &mut TcpStream, v: &[f32]) -> Result<(), String> {
    let mut b = Vec::with_capacity(8 + 4 * v.len());
    b.extend_from_slice(&(v.len() as u64).to_le_bytes());
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    s.write_all(&b).map_err(io("send"))
}

fn read_f32s(s: &mut TcpStream, out: &mut [f32]) -> Result<(), String> {
    let mut n = [0u8; 8];
    s.read_exact(&mut n).map_err(io("receive"))?;
    let n = u64::from_le_bytes(n) as usize;
    if n != out.len() {
        return Err(format!("dist: a peer sent {n} numbers, expected {}", out.len()));
    }
    let mut b = vec![0u8; 4 * n];
    s.read_exact(&mut b).map_err(io("receive"))?;
    for (o, c) in out.iter_mut().zip(b.chunks_exact(4)) {
        *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    Ok(())
}

impl Dist {
    /// A world of one (every call is a no-op).
    pub fn single() -> Dist {
        Dist { rank: 0, world: 1, peers: Vec::new() }
    }

    /// Join the world: rank 0 listens on `addr` and waits for the others; the others connect (retrying for a
    /// minute, so the ranks can start in any order) and say who they are.
    pub fn connect(rank: usize, world: usize, addr: &str) -> Result<Dist, String> {
        if world <= 1 {
            return Ok(Dist::single());
        }
        if rank >= world {
            return Err(format!("dist: rank {rank} >= world {world}"));
        }
        let peers = if rank == 0 {
            let l = TcpListener::bind(addr).map_err(io("listen"))?;
            let mut slots: Vec<Option<TcpStream>> = (1..world).map(|_| None).collect();
            for _ in 1..world {
                let (mut s, _) = l.accept().map_err(io("accept"))?;
                s.set_nodelay(true).map_err(io("nodelay"))?;
                let mut b = [0u8; 8];
                s.read_exact(&mut b).map_err(io("hello"))?;
                let r = u64::from_le_bytes(b) as usize;
                if r == 0 || r >= world || slots[r - 1].is_some() {
                    return Err(format!("dist: a peer says it is rank {r}"));
                }
                slots[r - 1] = Some(s);
            }
            slots.into_iter().map(|s| s.expect("every rank joined")).collect()
        } else {
            let t0 = Instant::now();
            let mut s = loop {
                match TcpStream::connect(addr) {
                    Ok(s) => break s,
                    Err(e) if t0.elapsed() < Duration::from_secs(60) => {
                        let _ = e;
                        std::thread::sleep(Duration::from_millis(200));
                    }
                    Err(e) => return Err(format!("dist: connect to {addr}: {e}")),
                }
            };
            s.set_nodelay(true).map_err(io("nodelay"))?;
            s.write_all(&(rank as u64).to_le_bytes()).map_err(io("hello"))?;
            vec![s]
        };
        Ok(Dist { rank, world, peers })
    }

    /// Sum `buf` over all ranks, in place. Rank 0 adds in rank order, so every rank ends with the same bits.
    pub fn all_reduce_sum(&mut self, buf: &mut [f32]) -> Result<(), String> {
        if self.world == 1 {
            return Ok(());
        }
        if self.rank == 0 {
            let mut tmp = vec![0f32; buf.len()];
            for p in self.peers.iter_mut() {
                read_f32s(p, &mut tmp)?;
                for (a, b) in buf.iter_mut().zip(&tmp) {
                    *a += *b;
                }
            }
            for p in self.peers.iter_mut() {
                write_f32s(p, buf)?;
            }
        } else {
            write_f32s(&mut self.peers[0], buf)?;
            read_f32s(&mut self.peers[0], buf)?;
        }
        Ok(())
    }

    /// Sum every parameter's gradient over the ranks. A parameter without a gradient on a rank counts as 0 there;
    /// after the sum it has one on every rank if any rank had one.
    pub fn all_reduce_grads(&mut self, vars: &[Var], grads: &mut GradStore) -> Result<(), String> {
        if self.world == 1 {
            return Ok(());
        }
        let err = |e: candle_core::Error| format!("dist grads: {e}");
        let (mut flat, mut has) = (Vec::new(), Vec::with_capacity(vars.len()));
        for v in vars {
            match grads.get(v.as_tensor()) {
                Some(g) => {
                    flat.extend(g.flatten_all().and_then(|g| g.to_dtype(DType::F32)).and_then(|g| g.to_vec1::<f32>()).map_err(err)?);
                    has.push(1.0f32);
                }
                None => {
                    flat.extend(std::iter::repeat(0f32).take(v.elem_count()));
                    has.push(0.0);
                }
            }
        }
        self.all_reduce_sum(&mut has)?;
        self.all_reduce_sum(&mut flat)?;
        let mut off = 0;
        for (v, &h) in vars.iter().zip(&has) {
            let n = v.elem_count();
            if h > 0.0 {
                let t = Tensor::from_slice(&flat[off..off + n], v.dims(), v.device()).map_err(err)?;
                grads.insert(v.as_tensor(), t);
            }
            off += n;
        }
        Ok(())
    }

    /// Rank 0's weights to every rank, so all start identical.
    pub fn broadcast_vars(&mut self, vars: &[Var]) -> Result<(), String> {
        if self.world == 1 {
            return Ok(());
        }
        let err = |e: candle_core::Error| format!("dist broadcast: {e}");
        let total: usize = vars.iter().map(|v| v.elem_count()).sum();
        let mut flat = Vec::with_capacity(total);
        if self.rank == 0 {
            for v in vars {
                flat.extend(v.as_tensor().flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(err)?);
            }
            for p in self.peers.iter_mut() {
                write_f32s(p, &flat)?;
            }
        } else {
            flat.resize(total, 0.0);
            read_f32s(&mut self.peers[0], &mut flat)?;
            let mut off = 0;
            for v in vars {
                let n = v.elem_count();
                v.set(&Tensor::from_slice(&flat[off..off + n], v.dims(), v.device()).map_err(err)?).map_err(err)?;
                off += n;
            }
        }
        Ok(())
    }
}
