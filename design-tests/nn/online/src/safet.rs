// TESTING / DESIGN ONLY
//! F4: reading a safetensors header, and checking that a new brain can still read the old one's
//! memory codes before it is swapped in.
//!
//! The rule: every tensor of the live brain must still exist in the candidate with the same shape
//! and type. New tensors are fine (growth adds parts). A changed shape means old states, diary keys
//! and engram codes would be read wrongly, so it needs an explicit migration and is refused otherwise.

use crate::json::{parse, J};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
}

pub type Header = BTreeMap<String, TensorInfo>;

/// Reads the tensor names, types and shapes of a safetensors file (the data is not read).
pub fn read_header(path: &Path) -> Result<Header, String> {
    let err = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut f = std::fs::File::open(path).map_err(err)?;
    let mut n8 = [0u8; 8];
    f.read_exact(&mut n8).map_err(err)?;
    let n = u64::from_le_bytes(n8) as usize;
    if n > 100 << 20 {
        return Err(format!("{}: header of {n} bytes is not believable", path.display()));
    }
    let mut h = vec![0u8; n];
    f.read_exact(&mut h).map_err(err)?;
    let s = std::str::from_utf8(&h).map_err(|_| format!("{}: header is not UTF-8", path.display()))?;
    header_from_json(&parse(s.trim_end()).map_err(|e| format!("{}: {e}", path.display()))?)
}

pub fn header_from_json(j: &J) -> Result<Header, String> {
    let J::Obj(items) = j else { return Err("header is not an object".into()) };
    let mut out = Header::new();
    for (k, v) in items {
        if k == "__metadata__" {
            continue;
        }
        let dtype = v.get("dtype").and_then(J::as_str).ok_or(format!("{k}: no dtype"))?.to_string();
        let shape = v
            .get("shape")
            .and_then(J::as_arr)
            .ok_or(format!("{k}: no shape"))?
            .iter()
            .map(|x| x.as_f64().map(|f| f as usize).ok_or(format!("{k}: bad shape")))
            .collect::<Result<Vec<_>, _>>()?;
        out.insert(k.clone(), TensorInfo { dtype, shape });
    }
    Ok(out)
}

/// What stops `new` from reading `old`'s codes. Empty means compatible.
pub fn problems(old: &Header, new: &Header) -> Vec<String> {
    let mut p = Vec::new();
    for (k, o) in old {
        match new.get(k) {
            None => p.push(format!("{k} is missing")),
            Some(n) if n != o => p.push(format!("{k}: {} {:?} -> {} {:?}", o.dtype, o.shape, n.dtype, n.shape)),
            _ => {}
        }
    }
    p
}

/// Writes a small all-zero F32 safetensors file (for tests and dry runs).
pub fn write_dummy(path: &Path, tensors: &[(&str, &[usize])]) -> Result<(), String> {
    let mut off = 0usize;
    let mut parts = Vec::new();
    for (name, shape) in tensors {
        let n: usize = shape.iter().product::<usize>() * 4;
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        parts.push(format!("\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{off},{}]}}", dims.join(","), off + n));
        off += n;
    }
    let mut head = format!("{{{}}}", parts.join(",")).into_bytes();
    while head.len() % 8 != 0 {
        head.push(b' ');
    }
    let mut bytes = (head.len() as u64).to_le_bytes().to_vec();
    bytes.extend(head);
    bytes.extend(std::iter::repeat(0u8).take(off));
    std::fs::write(path, bytes).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growth_is_allowed_but_reshaping_is_not() {
        let d = crate::log::test_dir("safet");
        let (a, b, c) = (d.join("a.safetensors"), d.join("b.safetensors"), d.join("c.safetensors"));
        write_dummy(&a, &[("emb", &[10, 4]), ("w", &[4, 4])]).unwrap();
        write_dummy(&b, &[("emb", &[10, 4]), ("w", &[4, 4]), ("engram", &[8, 4])]).unwrap();
        write_dummy(&c, &[("emb", &[12, 4])]).unwrap();
        let (ha, hb, hc) = (read_header(&a).unwrap(), read_header(&b).unwrap(), read_header(&c).unwrap());
        assert_eq!(ha["w"].shape, vec![4, 4]);
        assert!(problems(&ha, &hb).is_empty());
        let p = problems(&ha, &hc);
        assert_eq!(p.len(), 2, "{p:?}");
        assert!(p.iter().any(|x| x.contains("w is missing")));
        assert!(p.iter().any(|x| x.starts_with("emb:")));
    }
}
