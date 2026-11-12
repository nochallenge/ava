// TESTING / DESIGN ONLY
//! G2: a crash-safe, checksummed, append-only record log, and atomic file writes.
//!
//! Each record is `length (u32 LE) | crc32 (u32 LE) | payload`. A crash can leave a torn last
//! record. `Log::open` keeps every whole record whose checksum is good, stops at the first bad one and
//! cuts the file back to the end of the last good record, so the next append starts clean. Each
//! append is synced to disk before it returns, so an acknowledged record survives a power cut.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const TABLE: [u32; 256] = make_table();

const fn make_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}

/// CRC-32 (the IEEE one used by zip and PNG).
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = !0u32;
    for &b in data {
        c = TABLE[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
    }
    !c
}

/// No record is bigger than this; a bigger length can only be damage.
pub const MAX_RECORD: usize = 64 << 20;

/// Reads whole, good records from the start of `buf`. Returns them and the byte count they cover.
pub fn parse(buf: &[u8]) -> (Vec<Vec<u8>>, usize) {
    let (mut out, mut pos) = (Vec::new(), 0usize);
    while pos + 8 <= buf.len() {
        let len = u32::from_le_bytes(buf[pos..pos + 4].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(buf[pos + 4..pos + 8].try_into().unwrap());
        if len > MAX_RECORD || pos + 8 + len > buf.len() {
            break;
        }
        let payload = &buf[pos + 8..pos + 8 + len];
        if crc32(payload) != crc {
            break;
        }
        out.push(payload.to_vec());
        pos += 8 + len;
    }
    (out, pos)
}

/// One record as bytes on disk.
pub fn frame(payload: &[u8]) -> Vec<u8> {
    let mut r = Vec::with_capacity(8 + payload.len());
    r.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    r.extend_from_slice(&crc32(payload).to_le_bytes());
    r.extend_from_slice(payload);
    r
}

pub struct Log {
    pub path: PathBuf,
    file: File,
}

/// What `Log::open` found.
pub struct Opened {
    pub log: Log,
    pub records: Vec<Vec<u8>>,
    /// Bytes cut from a torn or damaged end (0 after a clean shutdown).
    pub cut_bytes: u64,
}

impl Log {
    /// Opens (or creates) a log, repairing a torn end.
    pub fn open(path: &Path) -> Result<Opened, String> {
        let err = |e: std::io::Error| format!("{}: {e}", path.display());
        let mut file = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path).map_err(err)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).map_err(err)?;
        let (records, good) = parse(&buf);
        let cut_bytes = (buf.len() - good) as u64;
        if cut_bytes > 0 {
            file.set_len(good as u64).map_err(err)?;
            file.sync_all().map_err(err)?;
        }
        file.seek(SeekFrom::End(0)).map_err(err)?;
        Ok(Opened { log: Log { path: path.to_path_buf(), file }, records, cut_bytes })
    }

    /// Appends one record and syncs it to disk.
    pub fn append(&mut self, payload: &[u8]) -> Result<(), String> {
        if payload.len() > MAX_RECORD {
            return Err(format!("record of {} bytes is too big", payload.len()));
        }
        let err = |e: std::io::Error| format!("{}: {e}", self.path.display());
        self.file.write_all(&frame(payload)).map_err(err)?;
        self.file.sync_data().map_err(err)
    }

    /// Replaces the whole log with these records, atomically (used to drop erased or forgotten text
    /// for good). Returns the reopened log.
    pub fn rewrite(self, records: &[Vec<u8>]) -> Result<Log, String> {
        let path = self.path.clone();
        drop(self.file);
        let bytes: Vec<u8> = records.iter().flat_map(|r| frame(r)).collect();
        atomic_write(&path, &bytes)?;
        Ok(Log::open(&path)?.log)
    }
}

/// `path` with ".tmp" added to its file name.
pub fn tmp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".tmp");
    path.with_file_name(name)
}

/// Writes a file so a reader sees either the old one or the whole new one, never a half-written one:
/// write a temporary file, sync it, then rename it over the old one.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp = tmp_path(path);
    let err = |e: std::io::Error| format!("{}: {e}", path.display());
    {
        let mut f = File::create(&tmp).map_err(err)?;
        f.write_all(bytes).map_err(err)?;
        f.sync_all().map_err(err)?;
    }
    std::fs::rename(&tmp, path).map_err(err)
}

/// Copies a file atomically (to a temporary name, synced, then renamed), and returns its FNV-64 hash.
pub fn atomic_copy(src: &Path, dst: &Path) -> Result<u64, String> {
    let tmp = tmp_path(dst);
    let err = |e: std::io::Error| format!("{} -> {}: {e}", src.display(), dst.display());
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    {
        let mut r = File::open(src).map_err(err)?;
        let mut w = File::create(&tmp).map_err(err)?;
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = r.read(&mut buf).map_err(err)?;
            if n == 0 {
                break;
            }
            h = crate::fnv_more(h, &buf[..n]);
            w.write_all(&buf[..n]).map_err(err)?;
        }
        w.sync_all().map_err(err)?;
    }
    std::fs::rename(&tmp, dst).map_err(err)?;
    Ok(h)
}

/// FNV-64 hash of a whole file, read in pieces.
pub fn hash_file(path: &Path) -> Result<u64, String> {
    let err = |e: std::io::Error| format!("{}: {e}", path.display());
    let mut r = File::open(path).map_err(err)?;
    let (mut h, mut buf) = (0xcbf2_9ce4_8422_2325u64, vec![0u8; 1 << 20]);
    loop {
        let n = r.read(&mut buf).map_err(err)?;
        if n == 0 {
            return Ok(h);
        }
        h = crate::fnv_more(h, &buf[..n]);
    }
}

/// A fresh, empty folder for a test.
#[cfg(test)]
pub fn test_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("ava_online_test_{name}_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn a_torn_end_is_cut_and_appends_continue() {
        let d = test_dir("torn");
        let p = d.join("a.log");
        {
            let mut l = Log::open(&p).unwrap().log;
            for r in [b"one".as_slice(), b"two", b"three"] {
                l.append(r).unwrap();
            }
        }
        // A crash in the middle of the last write.
        let len = std::fs::metadata(&p).unwrap().len();
        OpenOptions::new().write(true).open(&p).unwrap().set_len(len - 2).unwrap();
        let o = Log::open(&p).unwrap();
        assert_eq!(o.records, vec![b"one".to_vec(), b"two".to_vec()]);
        assert_eq!(o.cut_bytes, 8 + 5 - 2);
        let mut l = o.log;
        l.append(b"four").unwrap();
        drop(l);
        let o = Log::open(&p).unwrap();
        assert_eq!(o.records.len(), 3);
        assert_eq!(o.records[2], b"four");
        assert_eq!(o.cut_bytes, 0);
    }

    #[test]
    fn a_damaged_record_stops_the_read() {
        let mut bytes = frame(b"good");
        let mut bad = frame(b"bad!");
        bad[9] ^= 1;
        bytes.extend(bad);
        bytes.extend(frame(b"after"));
        let (r, n) = parse(&bytes);
        assert_eq!(r, vec![b"good".to_vec()]);
        assert_eq!(n, 12);
    }

    #[test]
    fn atomic_write_replaces_and_rewrite_keeps_only_new_records() {
        let d = test_dir("atomic");
        let p = d.join("x.txt");
        atomic_write(&p, b"old").unwrap();
        atomic_write(&p, b"new").unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"new");
        assert!(!tmp_path(&p).exists());
        let lp = d.join("b.log");
        let mut l = Log::open(&lp).unwrap().log;
        l.append(b"secret").unwrap();
        let l = l.rewrite(&[b"kept".to_vec()]).unwrap();
        drop(l);
        let raw = std::fs::read(&lp).unwrap();
        assert!(!raw.windows(6).any(|w| w == b"secret"));
        assert_eq!(Log::open(&lp).unwrap().records, vec![b"kept".to_vec()]);
        let c = d.join("copy.bin");
        assert_eq!(atomic_copy(&p, &c).unwrap(), crate::fnv(b"new"));
        assert_eq!(hash_file(&c).unwrap(), crate::fnv(b"new"));
    }
}
