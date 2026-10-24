// How many neurons will this machine actually hold? Measured, not estimated:
// allocate populations of increasing size and read the real process working set.
use design_tests::v4::{Brain, Params};
use std::time::Instant;

#[cfg(windows)]
fn rss_bytes() -> u64 {
    // GetProcessMemoryInfo via a tiny inline binding would need a crate; instead ask
    // the OS through the same counter Task Manager shows.
    use std::process::Command;
    let pid = std::process::id();
    let out = Command::new("powershell")
        .args(["-NoProfile", "-Command",
               &format!("(Get-Process -Id {pid}).WorkingSet64")])
        .output();
    out.ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}
#[cfg(not(windows))]
fn rss_bytes() -> u64 { 0 }

fn main() {
    let base = rss_bytes();
    println!("baseline process memory: {:.0} MB\n", base as f64 / 1e6);

    println!("=== empty neurons: what does one cost before it learns anything? ===");
    println!("{:>14} {:>12} {:>16} {:>10}", "neurons", "RSS MB", "bytes/neuron", "alloc time");
    let mut per_neuron = 0f64;
    for &n in &[1_000_000u32, 5_000_000, 20_000_000] {
        let t = Instant::now();
        let b = Brain::new(n, Params::default());
        let dt = t.elapsed();
        let rss = rss_bytes();
        let used = rss.saturating_sub(base);
        per_neuron = used as f64 / n as f64;
        println!("{:>14} {:>12.0} {:>16.1} {:>10.2?}", n, used as f64 / 1e6, per_neuron, dt);
        std::hint::black_box(&b);
        drop(b);
    }

    println!("\n=== the real cost is synapses, not neurons ===");
    println!("a synapse is (u32 target, u16 strength), 8 bytes once the Vec pads it.");
    println!("synapses per fact = k(k-1):\n");
    println!("{:>5} {:>16} {:>14} {:>14} {:>14}", "k", "syn/fact", "652k facts", "2.7M facts", "10M facts");
    for k in [10u64, 12, 16, 24] {
        let s = k * (k - 1);
        println!("{:>5} {:>16} {:>13.1}GB {:>13.1}GB {:>13.1}GB",
            k, s,
            s as f64 * 652_477.0 * 8.0 / 1e9,
            s as f64 * 2_689_118.0 * 8.0 / 1e9,
            s as f64 * 10_000_000.0 * 8.0 / 1e9);
    }

    println!("\n=== what fits in 31.8 GB (leaving 8 GB for windows and everything else) ===");
    let budget = 24.0e9;
    println!("{:>14} {:>14} {:>16} {:>14}", "neurons", "neuron cost", "left for synapses", "facts at k=12");
    for &n in &[1_000_000u64, 10_000_000, 50_000_000, 100_000_000, 200_000_000] {
        let ncost = n as f64 * per_neuron.max(48.0);
        let left = budget - ncost;
        let facts = if left > 0.0 { left / (12.0 * 11.0 * 8.0) } else { 0.0 };
        println!("{:>14} {:>13.1}GB {:>15.1}GB {:>14.0}",
            n, ncost / 1e9, left.max(0.0) / 1e9, facts.max(0.0));
    }
}
