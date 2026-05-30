use std::process::Command;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/matmul.cu");

    let out_dir  = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("matmul.ptx");
    let cu_path  = "src/matmul.cu";

    // Compile to PTX for sm_75 (Turing baseline).
    // CUDA JIT-compiles to native code on first launch and caches it —
    // startup overhead only, not per-kernel-call.
    // sm_75 covers GTX 16xx and RTX 20xx natively;
    // RTX 30xx/40xx/50xx get JIT-compiled to their native arch.
    let status = Command::new("nvcc")
        .args([
            "-ptx",
            "-arch=sm_75",
            "--use_fast_math",
            "-O3",
            "-o", ptx_path.to_str().unwrap(),
            cu_path,
        ])
        .status()
        .expect("nvcc not found — install CUDA toolkit and ensure nvcc is on PATH");

    assert!(status.success(), "nvcc compilation failed");

    println!("cargo:rustc-env=MATMUL_PTX_PATH={}", ptx_path.display());
}
