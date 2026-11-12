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
    // v49 (the chat app for download): with AVA_DELAYLOAD_CUDA=1 (MSVC), ava_live links the CUDA libraries it imports
    // delay-loaded (/DELAYLOAD): it starts on a PC without them and loads them only if a CUDA device is ever used (--gpu).
    // The same code and numbers; unset (every other build), nothing here changes. Check the result with
    // `dumpbin /dependents ava_live.exe` (the libraries then appear under "delay load dependencies" only).
    println!("cargo:rerun-if-env-changed=AVA_DELAYLOAD_CUDA");
    if std::env::var("AVA_DELAYLOAD_CUDA").as_deref() == Ok("1") && std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        for dll in ["cublas64_13.dll", "curand64_10.dll"] {
            println!("cargo:rustc-link-arg-bin=ava_live=/DELAYLOAD:{dll}");
        }
        println!("cargo:rustc-link-arg-bin=ava_live=delayimp.lib");
    }
}
