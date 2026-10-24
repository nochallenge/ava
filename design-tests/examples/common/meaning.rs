// TESTING / DESIGN ONLY. (v28 idea 1) WORD MEANING AS SPARSE CODES, shared by meaning_build.rs
// (learns them from reading) and compose.rs (AVA_MEANING=<file>: recall matches words by
// meaning). A word's meaning is which of D units are most active after reading: K of D (about
// 3%), like a sparse cortical code. Two words mean alike to the degree their codes overlap:
// shared units / K, from 0 (unrelated; random words share about K/D of their units) to 1.
#![allow(dead_code)]
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::path::Path;

pub const D: usize = 1024;
pub const K: usize = 32;

pub struct Meaning {
    pub id: HashMap<String, u32>,
    pub code: Vec<[u16; K]>, // each sorted ascending
}

impl Meaning {
    /// overlap of two words' codes, 0..1; a word unknown to her reading has no meaning yet (0)
    pub fn overlap(&self, a: &str, b: &str) -> f32 {
        if a == b { return 1.0; }
        let (Some(&i), Some(&j)) = (self.id.get(a), self.id.get(b)) else { return 0.0 };
        let (x, y) = (&self.code[i as usize], &self.code[j as usize]);
        let (mut p, mut q, mut n) = (0, 0, 0);
        while p < K && q < K {
            if x[p] == y[q] { n += 1; p += 1; q += 1; } else if x[p] < y[q] { p += 1; } else { q += 1; }
        }
        n as f32 / K as f32
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut b: Vec<u8> = b"AVAMEAN1".to_vec();
        b.extend((D as u32).to_le_bytes());
        b.extend((K as u32).to_le_bytes());
        let mut words: Vec<(&String, &u32)> = self.id.iter().collect();
        words.sort_by_key(|x| *x.1);
        b.extend((words.len() as u32).to_le_bytes());
        for (w, &i) in words {
            b.extend((w.len() as u16).to_le_bytes());
            b.extend(w.as_bytes());
            for &u in &self.code[i as usize] { b.extend(u.to_le_bytes()); }
        }
        std::fs::File::create(path)?.write_all(&b)
    }

    pub fn load(path: &Path) -> io::Result<Meaning> {
        let mut b = Vec::new();
        std::fs::File::open(path)?.read_to_end(&mut b)?;
        let bad = || io::Error::new(io::ErrorKind::InvalidData, "not an AVAMEAN1 file");
        if b.len() < 20 || &b[..8] != b"AVAMEAN1" { return Err(bad()); }
        let u32at = |p: usize| u32::from_le_bytes(b[p..p + 4].try_into().unwrap()) as usize;
        if u32at(8) != D || u32at(12) != K { return Err(bad()); }
        let n = u32at(16);
        let (mut id, mut code, mut p) = (HashMap::with_capacity(n), Vec::with_capacity(n), 20);
        for i in 0..n {
            let l = u16::from_le_bytes([b[p], b[p + 1]]) as usize;
            p += 2;
            let w = String::from_utf8(b[p..p + l].to_vec()).map_err(|_| bad())?;
            p += l;
            let mut c = [0u16; K];
            for (k, x) in c.iter_mut().enumerate() { *x = u16::from_le_bytes([b[p + 2 * k], b[p + 2 * k + 1]]); }
            p += 2 * K;
            id.insert(w, i as u32);
            code.push(c);
        }
        Ok(Meaning { id, code })
    }
}
