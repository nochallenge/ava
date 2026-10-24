#!/bin/sh
# TESTING / DESIGN ONLY - build helper for ava_gpu (candle + CUDA on Windows/MSVC). Usage: ./build.sh build --release
# * MSVC cl.exe/link.exe on PATH: nvcc needs cl.exe; the final link uses link.exe because the machine-wide
#   cargo config selects rust-lld, which cannot resolve strcat_s for onig_sys (pulled in by candle-core 0.11).
# * CUDA_COMPUTE_CAP=89 (RTX 4070); NVCC_APPEND_FLAGS: CCCL in CUDA 13.x needs the conforming MSVC preprocessor.
# * CUDA_PATH is taken from the environment (v13.2 here; nvcc 13.2 is first on PATH).
MSVC="/c/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
export PATH="$MSVC:$PATH"
export CUDA_COMPUTE_CAP=89
export CARGO_BUILD_JOBS=6
export NVCC_APPEND_FLAGS="-Xcompiler /Zc:preprocessor"
export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=link.exe
export RUSTFLAGS="-C linker-flavor=msvc"
cd "$(dirname "$0")" || exit 1
exec cargo "$@"
