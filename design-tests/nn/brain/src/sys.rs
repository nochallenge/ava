// TESTING / DESIGN ONLY — ava_brain::sys: the process's peak working set (resident memory), for
// reporting how much RAM a configuration needs. Windows: K32GetProcessMemoryInfo from kernel32
// (declared by hand, no crates). Other systems: VmHWM from /proc/self/status, else None.

#[cfg(windows)]
mod imp {
    #[repr(C)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn K32GetProcessMemoryInfo(process: isize, counters: *mut ProcessMemoryCounters, cb: u32) -> i32;
    }

    pub fn peak_rss_bytes() -> Option<u64> {
        let mut c = ProcessMemoryCounters {
            cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
            page_fault_count: 0,
            peak_working_set_size: 0,
            working_set_size: 0,
            quota_peak_paged_pool_usage: 0,
            quota_paged_pool_usage: 0,
            quota_peak_non_paged_pool_usage: 0,
            quota_non_paged_pool_usage: 0,
            pagefile_usage: 0,
            peak_pagefile_usage: 0,
        };
        // SAFETY: GetCurrentProcess returns a pseudo-handle that needs no closing; the struct is
        // laid out as PROCESS_MEMORY_COUNTERS and cb holds its size.
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
        if ok != 0 {
            Some(c.peak_working_set_size as u64)
        } else {
            None
        }
    }
}

#[cfg(not(windows))]
mod imp {
    pub fn peak_rss_bytes() -> Option<u64> {
        let s = std::fs::read_to_string("/proc/self/status").ok()?;
        let line = s.lines().find(|l| l.starts_with("VmHWM:"))?;
        let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
        Some(kb * 1024)
    }
}

/// Peak resident memory of this process so far, in MiB.
pub fn peak_rss_mib() -> Option<f64> {
    imp::peak_rss_bytes().map(|b| b as f64 / (1024.0 * 1024.0))
}
