use std::process::Command;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/matmul.cu");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("matmul.ptx");
    let cu_path = "src/matmul.cu";

    // Compile .cu → .ptx
    // sm_75: base target (Turing — GTX 16xx, RTX 20xx); JIT compiles to sm_86/89/100 on newer GPUs
    // WMMA INT8 is guarded by __CUDA_ARCH__ >= 720 in the .cu file
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
