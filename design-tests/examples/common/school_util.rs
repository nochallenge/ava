// TESTING / DESIGN ONLY. Shared helpers for the school tools (ported from Python in v27 so the
// whole project is Rust): text read and written the way the Python tools did, Python's line
// splitting and white space, a JSON reader and writer, MD5 and SHA-256, and URL joining.
// No crates. Included by the examples with `#[path = "common/school_util.rs"] mod util;`.
#![allow(dead_code)]

use std::fs;
use std::io;
use std::path::Path;

// ---------------------------------------------------------------- text
/// Python's str.isspace(): Unicode white space plus the separators 1c-1f
pub fn py_space(c: char) -> bool { c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c) }
pub fn py_strip(s: &str) -> &str { s.trim_matches(py_space) }
pub fn py_rstrip(s: &str) -> &str { s.trim_end_matches(py_space) }

/// \r\n and lone \r become \n (Python's universal newlines on read)
pub fn universal(s: String) -> String {
    if s.contains('\r') { s.replace("\r\n", "\n").replace('\r', "\n") } else { s }
}
/// a file read the way Python's text mode reads it: strict UTF-8, universal newlines
pub fn read_text(p: &Path) -> io::Result<String> {
    let s = String::from_utf8(fs::read(p)?).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(universal(s))
}
/// a file written the way Python's text mode writes it: \n becomes the platform's line end
pub fn write_text(p: &Path, s: &str) -> io::Result<()> {
    if cfg!(windows) { fs::write(p, s.replace('\n', "\r\n")) } else { fs::write(p, s) }
}

/// Python's str.splitlines(): every line boundary Python knows, no trailing empty line
pub fn splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        if matches!(c, '\n' | '\r' | '\u{b}' | '\u{c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}') {
            out.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' {
                if let Some(&(_, '\n')) = it.peek() { it.next(); end += 1; }
            }
            start = end;
        }
    }
    if start < s.len() { out.push(&s[start..]); }
    out
}

/// a string as Python's repr() shows it
pub fn py_repr(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut o = String::new();
    o.push(q);
    for c in s.chars() {
        let u = c as u32;
        let unprintable = u < 0x20 || (0x7f..0xa1).contains(&u) || u == 0xad || u == 0x1680
            || (0x2000..0x2010).contains(&u) || (0x2028..0x2030).contains(&u) || (0x205f..0x2065).contains(&u)
            || (0x2066..0x2070).contains(&u) || u == 0x3000 || u == 0xfeff || (0xfff9..0xfffc).contains(&u)
            || (0xe000..0xf900).contains(&u) || u >= 0xf0000;
        match c {
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if c == q => { o.push('\\'); o.push(c); }
            _ if unprintable => o.push_str(&if u < 0x100 { format!("\\x{u:02x}") } else if u < 0x10000 { format!("\\u{u:04x}") } else { format!("\\U{u:08x}") }),
            c => o.push(c),
        }
    }
    o.push(q);
    o
}

// ---------------------------------------------------------------- JSON
#[derive(Clone, Debug, PartialEq)]
pub enum J { Null, Bool(bool), Num(f64), Str(String), Arr(Vec<J>), Obj(Vec<(String, J)>) }

impl J {
    pub fn get(&self, k: &str) -> Option<&J> {
        if let J::Obj(v) = self { v.iter().find(|x| x.0 == k).map(|x| &x.1) } else { None }
    }
    pub fn str(&self) -> Option<&str> { if let J::Str(s) = self { Some(s) } else { None } }
    pub fn num(&self) -> Option<f64> { if let J::Num(x) = self { Some(*x) } else { None } }
    pub fn arr(&self) -> Option<&[J]> { if let J::Arr(v) = self { Some(v) } else { None } }
    pub fn truthy(&self) -> bool {
        match self { J::Null => false, J::Bool(b) => *b, J::Num(x) => *x != 0.0, J::Str(s) => !s.is_empty(),
                     J::Arr(v) => !v.is_empty(), J::Obj(v) => !v.is_empty() }
    }
}

struct P<'a> { s: &'a [u8], i: usize }
impl<'a> P<'a> {
    fn ws(&mut self) { while self.i < self.s.len() && b" \t\r\n".contains(&self.s[self.i]) { self.i += 1; } }
    fn val(&mut self) -> Result<J, String> {
        self.ws();
        match self.s.get(self.i) {
            Some(b'{') => {
                self.i += 1;
                let mut o = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b'}') { self.i += 1; return Ok(J::Obj(o)); }
                loop {
                    self.ws();
                    let k = self.string()?;
                    self.ws();
                    if self.s.get(self.i) != Some(&b':') { return Err(format!("expected ':' at {}", self.i)); }
                    self.i += 1;
                    let v = self.val()?;
                    // a repeated key keeps the last value, as Python's dict does
                    if let Some(x) = o.iter_mut().find(|x: &&mut (String, J)| x.0 == k) { x.1 = v; } else { o.push((k, v)); }
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b'}') => { self.i += 1; return Ok(J::Obj(o)); }
                        _ => return Err(format!("expected ',' or '}}' at {}", self.i)),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut a = Vec::new();
                self.ws();
                if self.s.get(self.i) == Some(&b']') { self.i += 1; return Ok(J::Arr(a)); }
                loop {
                    a.push(self.val()?);
                    self.ws();
                    match self.s.get(self.i) {
                        Some(b',') => self.i += 1,
                        Some(b']') => { self.i += 1; return Ok(J::Arr(a)); }
                        _ => return Err(format!("expected ',' or ']' at {}", self.i)),
                    }
                }
            }
            Some(b'"') => Ok(J::Str(self.string()?)),
            Some(b't') if self.s[self.i..].starts_with(b"true") => { self.i += 4; Ok(J::Bool(true)) }
            Some(b'f') if self.s[self.i..].starts_with(b"false") => { self.i += 5; Ok(J::Bool(false)) }
            Some(b'n') if self.s[self.i..].starts_with(b"null") => { self.i += 4; Ok(J::Null) }
            Some(_) => {
                let st = self.i;
                while self.i < self.s.len() && b"+-0123456789.eE".contains(&self.s[self.i]) { self.i += 1; }
                let t = std::str::from_utf8(&self.s[st..self.i]).map_err(|e| e.to_string())?;
                t.parse::<f64>().map(J::Num).map_err(|_| format!("bad value at {st}"))
            }
            None => Err("unexpected end".into()),
        }
    }
    fn hex4(&mut self) -> Result<u32, String> {
        let h = self.s.get(self.i..self.i + 4).ok_or("short \\u escape")?;
        self.i += 4;
        u32::from_str_radix(std::str::from_utf8(h).map_err(|e| e.to_string())?, 16).map_err(|e| e.to_string())
    }
    fn string(&mut self) -> Result<String, String> {
        if self.s.get(self.i) != Some(&b'"') { return Err(format!("expected a string at {}", self.i)); }
        self.i += 1;
        let mut b: Vec<u8> = Vec::new();
        loop {
            let c = *self.s.get(self.i).ok_or("unterminated string")?;
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = *self.s.get(self.i).ok_or("bad escape")?;
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"', b'\\' => '\\', b'/' => '/', b'b' => '\u{8}', b'f' => '\u{c}',
                        b'n' => '\n', b'r' => '\r', b't' => '\t',
                        b'u' => {
                            let mut u = self.hex4()?;
                            if (0xd800..0xdc00).contains(&u) && self.s[self.i..].starts_with(b"\\u") {
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xdc00..0xe000).contains(&lo) { u = 0x10000 + ((u - 0xd800) << 10) + (lo - 0xdc00); } else { self.i = save; }
                            }
                            char::from_u32(u).unwrap_or('\u{fffd}')
                        }
                        _ => return Err(format!("bad escape at {}", self.i)),
                    };
                    let mut t = [0u8; 4];
                    b.extend_from_slice(ch.encode_utf8(&mut t).as_bytes());
                }
                _ => b.push(c),
            }
        }
        String::from_utf8(b).map_err(|e| e.to_string())
    }
}
pub fn json_parse(s: &str) -> Result<J, String> {
    let mut p = P { s: s.as_bytes(), i: 0 };
    let v = p.val()?;
    p.ws();
    if p.i != p.s.len() { return Err(format!("extra data at {}", p.i)); }
    Ok(v)
}

/// Python's json.dumps(x) (indent None: ", " and ": ") or json.dumps(x, indent=n), ensure_ascii
pub fn json_dumps(j: &J, indent: Option<usize>) -> String {
    let mut o = String::new();
    dump(j, indent, 0, &mut o);
    o
}
fn dump_str(s: &str, o: &mut String) {
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            '\u{8}' => o.push_str("\\b"),
            '\u{c}' => o.push_str("\\f"),
            ' '..='~' => o.push(c),
            _ => {
                let mut b = [0u16; 2];
                for u in c.encode_utf16(&mut b) { o.push_str(&format!("\\u{:04x}", u)); }
            }
        }
    }
    o.push('"');
}
fn dump(j: &J, ind: Option<usize>, lvl: usize, o: &mut String) {
    let nl = |o: &mut String, l: usize| if let Some(n) = ind { o.push('\n'); o.push_str(&" ".repeat(n * l)); };
    match j {
        J::Null => o.push_str("null"),
        J::Bool(b) => o.push_str(if *b { "true" } else { "false" }),
        J::Num(x) => if x.fract() == 0.0 && x.abs() < 1e15 { o.push_str(&format!("{}", *x as i64)) } else { o.push_str(&format!("{x:?}")) },
        J::Str(s) => dump_str(s, o),
        J::Arr(v) => {
            if v.is_empty() { o.push_str("[]"); return; }
            o.push('[');
            for (i, x) in v.iter().enumerate() {
                if i > 0 { o.push(','); if ind.is_none() { o.push(' '); } }
                nl(o, lvl + 1);
                dump(x, ind, lvl + 1, o);
            }
            nl(o, lvl);
            o.push(']');
        }
        J::Obj(v) => {
            if v.is_empty() { o.push_str("{}"); return; }
            o.push('{');
            for (i, (k, x)) in v.iter().enumerate() {
                if i > 0 { o.push(','); if ind.is_none() { o.push(' '); } }
                nl(o, lvl + 1);
                dump_str(k, o);
                o.push_str(": ");
                dump(x, ind, lvl + 1, o);
            }
            nl(o, lvl);
            o.push('}');
        }
    }
}
pub fn obj(kv: Vec<(&str, J)>) -> J { J::Obj(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect()) }
pub fn num(x: usize) -> J { J::Num(x as f64) }
pub fn s(x: &str) -> J { J::Str(x.to_string()) }

// ---------------------------------------------------------------- hashes
pub fn md5(data: &[u8]) -> [u8; 16] {
    const S: [u32; 64] = [7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21];
    let k: Vec<u32> = (0..64).map(|i| ((i as f64 + 1.0).sin().abs() * 4294967296.0) as u32).collect();
    let (mut a0, mut b0, mut c0, mut d0) = (0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32);
    let mut m = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    m.push(0x80);
    while m.len() % 64 != 56 { m.push(0); }
    m.extend_from_slice(&bits.to_le_bytes());
    for chunk in m.chunks(64) {
        let w: Vec<u32> = (0..16).map(|i| u32::from_le_bytes(chunk[4 * i..4 * i + 4].try_into().unwrap())).collect();
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let f2 = f.wrapping_add(a).wrapping_add(k[i]).wrapping_add(w[g]);
            a = d; d = c; c = b;
            b = b.wrapping_add(f2.rotate_left(S[i]));
        }
        a0 = a0.wrapping_add(a); b0 = b0.wrapping_add(b); c0 = c0.wrapping_add(c); d0 = d0.wrapping_add(d);
    }
    let mut out = [0u8; 16];
    for (i, x) in [a0, b0, c0, d0].iter().enumerate() { out[4 * i..4 * i + 4].copy_from_slice(&x.to_le_bytes()); }
    out
}
/// Python's int(hashlib.md5(key.encode()).hexdigest(), 16) % m
pub fn bucket(key: &str, m: u128) -> u128 { u128::from_be_bytes(md5(key.as_bytes())) % m }

pub fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2];
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut m = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    m.push(0x80);
    while m.len() % 64 != 56 { m.push(0); }
    m.extend_from_slice(&bits.to_be_bytes());
    for chunk in m.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 { w[i] = u32::from_be_bytes(chunk[4 * i..4 * i + 4].try_into().unwrap()); }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for i in 0..8 { h[i] = h[i].wrapping_add(v[i]); }
    }
    let mut out = [0u8; 32];
    for (i, x) in h.iter().enumerate() { out[4 * i..4 * i + 4].copy_from_slice(&x.to_be_bytes()); }
    out
}
pub fn hex(b: &[u8]) -> String { b.iter().map(|x| format!("{x:02x}")).collect() }

// ---------------------------------------------------------------- URLs
/// Python's urllib.parse.urljoin for an http(s) base without query or fragment (the crawl's case)
pub fn urljoin(base: &str, url: &str) -> String {
    if url.is_empty() { return base.to_string(); }
    let (bscheme, rest) = base.split_once("://").unwrap_or(("", base));
    let (bnetloc, bpath) = match rest.find('/') { Some(i) => (&rest[..i], &rest[i..]), None => (rest, "") };
    let mut scheme = bscheme.to_string();
    let mut after = url;
    if let Some(i) = url.find(':') {
        let head = &url[..i];
        if i > 0 && head.chars().next().map_or(false, |c| c.is_ascii_alphabetic())
            && head.chars().all(|c| c.is_ascii_alphanumeric() || "+-.".contains(c)) {
            scheme = head.to_ascii_lowercase();
            after = &url[i + 1..];
        }
    }
    if scheme != bscheme || !["http", "https"].contains(&scheme.as_str()) { return url.to_string(); }
    if let Some(r) = after.strip_prefix("//") {
        let (netloc, path) = match r.find('/') { Some(i) => (&r[..i], &r[i..]), None => (r, "") };
        return format!("{scheme}://{netloc}{path}");
    }
    if after.is_empty() { return format!("{scheme}://{bnetloc}{bpath}"); }
    let mut segs: Vec<&str> = if after.starts_with('/') { after.split('/').collect() } else {
        let mut b: Vec<&str> = bpath.split('/').collect();
        if b.last() != Some(&"") { b.pop(); }
        b.extend(after.split('/'));
        b
    };
    if segs.len() > 2 {
        let last = segs.pop().unwrap();
        let first = segs.remove(0);
        let mid: Vec<&str> = segs.into_iter().filter(|x| !x.is_empty()).collect();
        segs = std::iter::once(first).chain(mid).chain(std::iter::once(last)).collect();
    }
    let mut res: Vec<&str> = Vec::new();
    for sg in &segs {
        match *sg { ".." => { res.pop(); } "." => {} x => res.push(x) }
    }
    if matches!(segs.last(), Some(&".") | Some(&"..")) { res.push(""); }
    let mut p = res.join("/");
    if p.is_empty() { p = "/".into(); }
    if !p.starts_with('/') { p.insert(0, '/'); }
    format!("{scheme}://{bnetloc}{p}")
}

// ---------------------------------------------------------------- paths
/// every file under `dir` whose name ends in `ext` (case-insensitive, as Windows globbing is),
/// not following symbolic links
pub fn rglob(dir: &Path, ext: &str, out: &mut Vec<std::path::PathBuf>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let Ok(t) = e.file_type() else { continue };
        let p = e.path();
        if t.is_dir() { rglob(&p, ext, out); }
        else if p.file_name().map_or(false, |n| n.to_string_lossy().to_lowercase().ends_with(ext)) { out.push(p); }
    }
}
/// the order Python's sorted() gives Windows paths: by lower-cased parts
pub fn path_key(p: &Path) -> Vec<String> {
    let s = p.to_string_lossy().to_string();
    let s = if cfg!(windows) { s.replace('/', "\\").to_lowercase() } else { s };
    s.split(if cfg!(windows) { '\\' } else { '/' }).map(String::from).collect()
}
