use std::process::Command;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/matmul.cu");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("matmul.ptx");
    let cu_path = "src/matmul.cu";

    // Compile .cu → .ptx targeting sm_75 (Turing+, covers 16xx/20xx/30xx/40xx/50xx via JIT)
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
