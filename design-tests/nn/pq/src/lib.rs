// TESTING / DESIGN ONLY
//! A small Parquet reader, std only, made for one job: pulling text columns out of downloaded
//! datasets (Simple English Wikipedia, FineWeb-Edu) so they can become Ava's reading.
//!
//! What it supports:
//! - the Thrift compact protocol used by the file footer and page headers;
//! - data pages v1 and v2, dictionary pages;
//! - PLAIN and dictionary (RLE_DICTIONARY / PLAIN_DICTIONARY) encodings of BYTE_ARRAY values;
//! - optional (nullable) top-level columns, through definition levels;
//! - the codecs UNCOMPRESSED and SNAPPY.
//!
//! Anything else (nested columns, other codecs, delta encodings) is reported as an error, never
//! silently skipped. Row groups are read one at a time, so a 2 GB file never sits in memory whole.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub type Res<T> = Result<T, String>;

// ---------------------------------------------------------------------------------------------
// Thrift compact protocol: parsed into a small value tree, then read by field id.
// ---------------------------------------------------------------------------------------------

/// One Thrift value.
#[derive(Clone, Debug, PartialEq)]
pub enum TV {
    Bool(bool),
    Int(i64),
    Double(f64),
    Bin(Vec<u8>),
    List(Vec<TV>),
    Map(Vec<(TV, TV)>),
    Struct(Vec<(i16, TV)>),
}

impl TV {
    /// The field with this id, if this is a struct that has it.
    pub fn get(&self, id: i16) -> Option<&TV> {
        match self {
            TV::Struct(fs) => fs.iter().find(|(f, _)| *f == id).map(|(_, v)| v),
            _ => None,
        }
    }
    pub fn int(&self, id: i16) -> Option<i64> {
        match self.get(id) {
            Some(TV::Int(v)) => Some(*v),
            _ => None,
        }
    }
    pub fn bool(&self, id: i16) -> Option<bool> {
        match self.get(id) {
            Some(TV::Bool(v)) => Some(*v),
            _ => None,
        }
    }
    pub fn str(&self, id: i16) -> Option<String> {
        match self.get(id) {
            Some(TV::Bin(b)) => Some(String::from_utf8_lossy(b).into_owned()),
            _ => None,
        }
    }
    pub fn list(&self, id: i16) -> &[TV] {
        match self.get(id) {
            Some(TV::List(v)) => v,
            _ => &[],
        }
    }
}

/// A cursor over Thrift compact bytes.
pub struct Thrift<'a> {
    pub buf: &'a [u8],
    pub pos: usize,
}

impl<'a> Thrift<'a> {
    pub fn new(buf: &'a [u8]) -> Self {
        Thrift { buf, pos: 0 }
    }
    fn byte(&mut self) -> Res<u8> {
        let b = *self.buf.get(self.pos).ok_or("thrift: unexpected end")?;
        self.pos += 1;
        Ok(b)
    }
    fn varint(&mut self) -> Res<u64> {
        let (mut v, mut shift) = (0u64, 0u32);
        loop {
            let b = self.byte()?;
            v |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
            shift += 7;
            if shift > 63 {
                return Err("thrift: varint too long".into());
            }
        }
    }
    fn zigzag(&mut self) -> Res<i64> {
        let v = self.varint()?;
        Ok(((v >> 1) as i64) ^ -((v & 1) as i64))
    }
    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.buf.len()).ok_or("thrift: length past end")?;
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    /// Reads one value of compact type `t`.
    fn value(&mut self, t: u8, depth: u32) -> Res<TV> {
        if depth > 64 {
            return Err("thrift: nested too deep".into());
        }
        Ok(match t {
            1 => TV::Bool(true),
            2 => TV::Bool(false),
            3 => TV::Int(self.byte()? as i8 as i64),
            4 | 5 | 6 => TV::Int(self.zigzag()?),
            7 => {
                let b = self.take(8)?;
                TV::Double(f64::from_le_bytes(b.try_into().unwrap()))
            }
            8 => {
                let n = self.varint()? as usize;
                TV::Bin(self.take(n)?.to_vec())
            }
            9 | 10 => {
                let h = self.byte()?;
                let et = h & 0x0f;
                let n = if h >> 4 == 15 { self.varint()? as usize } else { (h >> 4) as usize };
                let mut v = Vec::with_capacity(n.min(1 << 16));
                for _ in 0..n {
                    // Booleans inside a list are one byte each (1 = true).
                    v.push(if et == 1 || et == 2 { TV::Bool(self.byte()? == 1) } else { self.value(et, depth + 1)? });
                }
                TV::List(v)
            }
            11 => {
                let n = self.varint()? as usize;
                let mut v = Vec::new();
                if n > 0 {
                    let kv = self.byte()?;
                    for _ in 0..n {
                        let k = self.value(kv >> 4, depth + 1)?;
                        let val = self.value(kv & 0x0f, depth + 1)?;
                        v.push((k, val));
                    }
                }
                TV::Map(v)
            }
            12 => self.read_struct(depth + 1)?,
            _ => return Err(format!("thrift: unknown type {t}")),
        })
    }

    /// Reads a struct (fields until STOP).
    pub fn read_struct(&mut self, depth: u32) -> Res<TV> {
        let mut fields = Vec::new();
        let mut last: i16 = 0;
        loop {
            let h = self.byte()?;
            if h == 0 {
                return Ok(TV::Struct(fields));
            }
            let t = h & 0x0f;
            let delta = (h >> 4) as i16;
            let id = if delta == 0 { self.zigzag()? as i16 } else { last + delta };
            last = id;
            let v = self.value(t, depth)?;
            fields.push((id, v));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Snappy (raw block format, as Parquet stores it).
// ---------------------------------------------------------------------------------------------

/// Decompresses one raw Snappy block.
pub fn snappy_decompress(src: &[u8]) -> Res<Vec<u8>> {
    let mut t = Thrift::new(src);
    let n = t.varint()? as usize;
    let mut out: Vec<u8> = Vec::with_capacity(n);
    let mut p = t.pos;
    let get = |p: usize| -> Res<u8> { src.get(p).copied().ok_or_else(|| "snappy: unexpected end".to_string()) };
    while p < src.len() {
        let tag = src[p];
        p += 1;
        match tag & 3 {
            0 => {
                let mut len = (tag >> 2) as usize;
                if len >= 60 {
                    let k = len - 59;
                    let mut l = 0usize;
                    for i in 0..k {
                        l |= (get(p + i)? as usize) << (8 * i);
                    }
                    p += k;
                    len = l;
                }
                len += 1;
                let end = p.checked_add(len).filter(|&e| e <= src.len()).ok_or("snappy: literal past end")?;
                out.extend_from_slice(&src[p..end]);
                p = end;
            }
            k => {
                let (len, off) = match k {
                    1 => {
                        let o = (((tag >> 5) as usize) << 8) | get(p)? as usize;
                        p += 1;
                        (((tag >> 2) & 7) as usize + 4, o)
                    }
                    2 => {
                        let o = get(p)? as usize | (get(p + 1)? as usize) << 8;
                        p += 2;
                        ((tag >> 2) as usize + 1, o)
                    }
                    _ => {
                        let mut o = 0usize;
                        for i in 0..4 {
                            o |= (get(p + i)? as usize) << (8 * i);
                        }
                        p += 4;
                        ((tag >> 2) as usize + 1, o)
                    }
                };
                if off == 0 || off > out.len() {
                    return Err("snappy: bad copy offset".into());
                }
                let start = out.len() - off;
                for i in 0..len {
                    // Byte by byte, because a copy may overlap what it is writing.
                    let b = out[start + i];
                    out.push(b);
                }
            }
        }
    }
    if out.len() != n {
        return Err(format!("snappy: got {} bytes, header said {n}", out.len()));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------
// The RLE / bit-packed hybrid (definition levels and dictionary indices).
// ---------------------------------------------------------------------------------------------

/// Decodes `count` values of `width` bits from the hybrid encoding in `buf`.
pub fn rle_hybrid(buf: &[u8], width: u32, count: usize) -> Res<Vec<u32>> {
    let mut out = Vec::with_capacity(count);
    if width == 0 {
        out.resize(count, 0);
        return Ok(out);
    }
    if width > 32 {
        return Err(format!("rle: bit width {width} > 32"));
    }
    let mut t = Thrift::new(buf);
    let bytes_w = ((width + 7) / 8) as usize;
    while out.len() < count {
        let h = t.varint()?;
        if h & 1 == 0 {
            let run = (h >> 1) as usize;
            let b = t.take(bytes_w)?;
            let mut v = 0u32;
            for (i, &x) in b.iter().enumerate() {
                v |= (x as u32) << (8 * i);
            }
            let take = run.min(count - out.len());
            out.extend(std::iter::repeat(v).take(take));
        } else {
            let groups = (h >> 1) as usize;
            let nbytes = groups * width as usize;
            // The last group may be cut short at the end of a page.
            let avail = nbytes.min(buf.len() - t.pos);
            let b = t.take(avail)?;
            let total = groups * 8;
            let mask: u64 = if width == 32 { u32::MAX as u64 } else { (1u64 << width) - 1 };
            for i in 0..total {
                if out.len() >= count {
                    break;
                }
                let bit = i * width as usize;
                let (byte, sh) = (bit / 8, bit % 8);
                let mut w = 0u64;
                for j in 0..5 {
                    if let Some(&x) = b.get(byte + j) {
                        w |= (x as u64) << (8 * j);
                    }
                }
                out.push(((w >> sh) & mask) as u32);
            }
        }
    }
    Ok(out)
}

/// Bits needed to hold values up to `max`.
pub fn bit_width(max: u32) -> u32 {
    32 - max.leading_zeros()
}

// ---------------------------------------------------------------------------------------------
// The file.
// ---------------------------------------------------------------------------------------------

/// One top-level column as the schema describes it.
#[derive(Clone, Debug)]
pub struct Column {
    pub name: String,
    pub physical: i64,
    pub optional: bool,
}

/// An open Parquet file with its footer parsed.
pub struct ParquetFile {
    pub file: File,
    pub meta: TV,
    pub columns: Vec<Column>,
    pub num_rows: i64,
    pub created_by: String,
}

pub const CODEC_NAMES: [&str; 8] = ["UNCOMPRESSED", "SNAPPY", "GZIP", "LZO", "BROTLI", "LZ4", "ZSTD", "LZ4_RAW"];

impl ParquetFile {
    pub fn open(path: &Path) -> Res<ParquetFile> {
        let mut file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let len = file.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
        if len < 12 {
            return Err("parquet: file too short".into());
        }
        let mut tail = [0u8; 8];
        file.seek(SeekFrom::End(-8)).map_err(|e| e.to_string())?;
        file.read_exact(&mut tail).map_err(|e| e.to_string())?;
        if &tail[4..] != b"PAR1" {
            return Err("parquet: no PAR1 at the end (not a parquet file, or cut short)".into());
        }
        let flen = u32::from_le_bytes(tail[..4].try_into().unwrap()) as u64;
        if flen + 8 > len {
            return Err("parquet: footer length past the start".into());
        }
        let mut foot = vec![0u8; flen as usize];
        file.seek(SeekFrom::Start(len - 8 - flen)).map_err(|e| e.to_string())?;
        file.read_exact(&mut foot).map_err(|e| e.to_string())?;
        let meta = Thrift::new(&foot).read_struct(0)?;
        // Schema: element 0 is the root; its children are the top-level columns.
        let schema = meta.list(2);
        let mut columns = Vec::new();
        for el in schema.iter().skip(1) {
            if el.int(5).unwrap_or(0) > 0 {
                return Err("parquet: nested columns are not supported".into());
            }
            columns.push(Column {
                name: el.str(4).unwrap_or_default(),
                physical: el.int(1).unwrap_or(-1),
                optional: el.int(3) == Some(1),
            });
        }
        let num_rows = meta.int(3).unwrap_or(0);
        let created_by = meta.str(6).unwrap_or_default();
        Ok(ParquetFile { file, meta, columns, num_rows, created_by })
    }

    pub fn row_groups(&self) -> usize {
        self.meta.list(4).len()
    }

    /// The codecs used by the file's column chunks (by name), for reporting.
    pub fn codecs(&self) -> Vec<String> {
        let mut seen: Vec<String> = Vec::new();
        for rg in self.meta.list(4) {
            for cc in rg.list(1) {
                if let Some(md) = cc.get(3) {
                    let c = md.int(4).unwrap_or(-1);
                    let name = CODEC_NAMES.get(c as usize).map(|s| s.to_string()).unwrap_or(format!("codec {c}"));
                    if !seen.contains(&name) {
                        seen.push(name);
                    }
                }
            }
        }
        seen
    }

    /// Reads one BYTE_ARRAY column of row group `rg`: one entry per row, `None` for a null.
    pub fn read_bytes_column(&mut self, rg: usize, name: &str) -> Res<Vec<Option<Vec<u8>>>> {
        let ci = self.columns.iter().position(|c| c.name == name).ok_or_else(|| format!("parquet: no column {name:?}"))?;
        let col = self.columns[ci].clone();
        if col.physical != 6 {
            return Err(format!("parquet: column {name:?} is not BYTE_ARRAY (type {})", col.physical));
        }
        let group = self.meta.list(4).get(rg).ok_or("parquet: no such row group")?.clone();
        let rows = group.int(3).unwrap_or(0) as usize;
        let cc = group.list(1).get(ci).ok_or("parquet: row group lacks the column")?;
        let md = cc.get(3).ok_or("parquet: column chunk without metadata")?;
        let codec = md.int(4).unwrap_or(0);
        if codec != 0 && codec != 1 {
            let n = CODEC_NAMES.get(codec as usize).copied().unwrap_or("?");
            return Err(format!("parquet: codec {n} is not supported (only UNCOMPRESSED and SNAPPY)"));
        }
        let data_off = md.int(9).ok_or("parquet: no data page offset")?;
        let start = match md.int(11) {
            Some(d) if d > 0 && d < data_off => d,
            _ => data_off,
        } as u64;
        let size = md.int(7).ok_or("parquet: no compressed size")? as usize;
        let mut chunk = vec![0u8; size];
        self.file.seek(SeekFrom::Start(start)).map_err(|e| e.to_string())?;
        self.file.read_exact(&mut chunk).map_err(|e| e.to_string())?;

        let max_def: u32 = if col.optional { 1 } else { 0 };
        let mut dict: Vec<Vec<u8>> = Vec::new();
        let mut out: Vec<Option<Vec<u8>>> = Vec::with_capacity(rows);
        let mut pos = 0usize;
        while pos < chunk.len() && out.len() < rows {
            let mut t = Thrift::new(&chunk[pos..]);
            let ph = t.read_struct(0)?;
            pos += t.pos;
            let ptype = ph.int(1).unwrap_or(-1);
            let usize_ = ph.int(2).unwrap_or(0) as usize;
            let csize = ph.int(3).unwrap_or(0) as usize;
            let end = pos.checked_add(csize).filter(|&e| e <= chunk.len()).ok_or("parquet: page past chunk end")?;
            let body = &chunk[pos..end];
            pos = end;
            let unpack = |b: &[u8]| -> Res<Vec<u8>> { if codec == 1 { snappy_decompress(b) } else { Ok(b.to_vec()) } };
            match ptype {
                2 => {
                    // Dictionary page: PLAIN values.
                    let raw = unpack(body)?;
                    let n = ph.get(7).and_then(|d| d.int(1)).unwrap_or(0) as usize;
                    dict = plain_byte_arrays(&raw, n)?;
                }
                0 => {
                    let h = ph.get(5).ok_or("parquet: data page without header")?;
                    let n = h.int(1).unwrap_or(0) as usize;
                    let enc = h.int(2).unwrap_or(0);
                    let raw = unpack(body)?;
                    if raw.len() != usize_ {
                        return Err(format!("parquet: page size {} != {usize_}", raw.len()));
                    }
                    let mut p = 0usize;
                    let defs = if max_def > 0 {
                        let l = u32::from_le_bytes(raw.get(0..4).ok_or("parquet: short page")?.try_into().unwrap()) as usize;
                        let d = rle_hybrid(raw.get(4..4 + l).ok_or("parquet: def levels past page")?, bit_width(max_def), n)?;
                        p = 4 + l;
                        d
                    } else {
                        vec![0; n]
                    };
                    decode_values(&raw[p..], enc, &defs, max_def, &dict, &mut out)?;
                }
                3 => {
                    let h = ph.get(8).ok_or("parquet: v2 page without header")?;
                    let n = h.int(1).unwrap_or(0) as usize;
                    let enc = h.int(4).unwrap_or(0);
                    let dl = h.int(5).unwrap_or(0) as usize;
                    let rl = h.int(6).unwrap_or(0) as usize;
                    let compressed = h.bool(7).unwrap_or(true);
                    if rl != 0 {
                        return Err("parquet: repetition levels in a flat column".into());
                    }
                    let defs = if max_def > 0 {
                        rle_hybrid(body.get(..dl).ok_or("parquet: def levels past page")?, bit_width(max_def), n)?
                    } else {
                        vec![0; n]
                    };
                    let vals = &body[dl..];
                    let raw = if compressed { unpack(vals)? } else { vals.to_vec() };
                    decode_values(&raw, enc, &defs, max_def, &dict, &mut out)?;
                }
                1 => {} // index page: ignore
                _ => return Err(format!("parquet: unknown page type {ptype}")),
            }
        }
        if out.len() != rows {
            return Err(format!("parquet: column {name:?} gave {} values for {rows} rows", out.len()));
        }
        Ok(out)
    }
}

/// `n` PLAIN BYTE_ARRAY values (4-byte LE length, then the bytes).
pub fn plain_byte_arrays(raw: &[u8], n: usize) -> Res<Vec<Vec<u8>>> {
    let mut v = Vec::with_capacity(n);
    let mut p = 0usize;
    for _ in 0..n {
        let l = u32::from_le_bytes(raw.get(p..p + 4).ok_or("parquet: plain value past page")?.try_into().unwrap()) as usize;
        p += 4;
        v.push(raw.get(p..p + l).ok_or("parquet: plain value past page")?.to_vec());
        p += l;
    }
    Ok(v)
}

/// Decodes one page's values and appends one entry per level (None where the level is a null).
fn decode_values(raw: &[u8], enc: i64, defs: &[u32], max_def: u32, dict: &[Vec<u8>], out: &mut Vec<Option<Vec<u8>>>) -> Res<()> {
    let present = defs.iter().filter(|&&d| d == max_def).count();
    let vals: Vec<Vec<u8>> = match enc {
        0 => plain_byte_arrays(raw, present)?,
        2 | 8 => {
            let w = *raw.first().ok_or("parquet: empty dictionary-index page")? as u32;
            let idx = rle_hybrid(&raw[1..], w, present)?;
            let mut v = Vec::with_capacity(present);
            for i in idx {
                v.push(dict.get(i as usize).ok_or("parquet: dictionary index out of range")?.clone());
            }
            v
        }
        e => return Err(format!("parquet: value encoding {e} is not supported")),
    };
    let mut it = vals.into_iter();
    for &d in defs {
        out.push(if d == max_def { it.next() } else { None });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zz(v: i64) -> Vec<u8> {
        let mut u = ((v << 1) ^ (v >> 63)) as u64;
        let mut o = Vec::new();
        loop {
            let b = (u & 0x7f) as u8;
            u >>= 7;
            if u == 0 {
                o.push(b);
                return o;
            }
            o.push(b | 0x80);
        }
    }

    #[test]
    fn thrift_struct_with_ints_strings_lists() {
        // struct { 1: i32 = 7, 2: binary = "ab", 4: list<i32> = [1, -2], 20: i64 = 300 }
        let mut b = vec![0x15];
        b.extend(zz(7));
        b.push(0x18);
        b.extend([2, b'a', b'b']);
        b.push(0x29); // delta 2 -> id 4, list
        b.push(0x25); // 2 elements of i32
        b.extend(zz(1));
        b.extend(zz(-2));
        b.push(0x06); // long form: delta 0, type i64, then zigzag id
        b.extend(zz(20));
        b.extend(zz(300));
        b.push(0);
        let v = Thrift::new(&b).read_struct(0).unwrap();
        assert_eq!(v.int(1), Some(7));
        assert_eq!(v.str(2).as_deref(), Some("ab"));
        assert_eq!(v.list(4), &[TV::Int(1), TV::Int(-2)]);
        assert_eq!(v.int(20), Some(300));
    }

    #[test]
    fn snappy_literal_and_overlapping_copy() {
        // "abcabcabcab": literal "abc" then a 1-byte-offset copy of length 8 at offset 3.
        let mut s = vec![11u8];
        s.push((2 << 2) | 0); // literal of 3
        s.extend(b"abc");
        s.push((((8 - 4) as u8) << 2) | 1); // copy-1, len 8, offset high bits 0
        s.push(3);
        assert_eq!(snappy_decompress(&s).unwrap(), b"abcabcabcab");
    }

    #[test]
    fn snappy_rejects_wrong_length() {
        let s = vec![5u8, (2 << 2), b'a', b'b', b'c'];
        assert!(snappy_decompress(&s).is_err());
    }

    #[test]
    fn rle_run_and_bitpacked() {
        // run of 5 x value 1 (width 1), then 1 bit-packed group of 8 values 0..=7 at width 3.
        let mut b = vec![5 << 1, 1];
        b.push((1 << 1) | 1);
        // values 0..7 at 3 bits each, LSB first: 0b10001000, 0b11000110, 0b11111010
        b.extend([0x88, 0xC6, 0xFA]);
        let v = rle_hybrid(&b, 1, 5).unwrap();
        assert_eq!(v, vec![1, 1, 1, 1, 1]);
        let v = rle_hybrid(&b[2..], 3, 8).unwrap();
        assert_eq!(v, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn plain_values_and_nulls() {
        let mut raw = Vec::new();
        for s in ["hi", "yo"] {
            raw.extend((s.len() as u32).to_le_bytes());
            raw.extend(s.as_bytes());
        }
        let mut out = Vec::new();
        decode_values(&raw, 0, &[1, 0, 1], 1, &[], &mut out).unwrap();
        assert_eq!(out, vec![Some(b"hi".to_vec()), None, Some(b"yo".to_vec())]);
    }

    #[test]
    fn bit_widths() {
        assert_eq!(bit_width(0), 0);
        assert_eq!(bit_width(1), 1);
        assert_eq!(bit_width(7), 3);
        assert_eq!(bit_width(8), 4);
    }
}
