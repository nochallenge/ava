// TESTING / DESIGN ONLY — compile src/lif.cu to PTX with nvcc (already required by candle's cuda
// feature). CUDA_COMPUTE_CAP picks the virtual architecture (default 75); NVCC overrides the compiler.
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/lif.cu");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo:rerun-if-env-changed=NVCC");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("lif.ptx");
    let cap = std::env::var("CUDA_COMPUTE_CAP").unwrap_or_else(|_| "75".into());
    let nvcc = std::env::var("NVCC").unwrap_or_else(|_| "nvcc".into());
    let status = Command::new(&nvcc)
        .args(["--ptx", "-O3", &format!("-arch=compute_{cap}"), "src/lif.cu", "-o"])
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("could not run {nvcc} (the CUDA toolkit's compiler): {e}"));
    assert!(status.success(), "nvcc failed to compile src/lif.cu");
}
