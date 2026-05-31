use std::path::PathBuf;
use std::process::Command;

fn try_nvcc(prefix: &str, cu: &str, out: &PathBuf, arch: &str) -> Option<PathBuf> {
    let ptx = out.join(format!("{}_{}.ptx", prefix, arch));
    let ok = Command::new("nvcc")
        .args([
            "-ptx",
            &format!("-arch={}", arch),
            "--use_fast_math",
            "-O3",
            "-o", ptx.to_str().unwrap(),
            cu,
        ])
        .status()
        .ok()
        .map(|s| s.success())
        .unwrap_or(false);
    if ok { Some(ptx) } else { None }
}

fn main() {
    println!("cargo:rerun-if-changed=src/matmul.cu");
    println!("cargo:rerun-if-changed=src/blake3_solver.cu");
    let out  = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let cu   = "src/matmul.cu";
    let cu_b = "src/blake3_solver.cu";

    // sm_75 — mandatory baseline (Turing: GTX 16xx, RTX 20xx)
    let ptx75 = try_nvcc("matmul", cu, &out, "sm_75")
        .expect("nvcc failed for sm_75 — ensure CUDA toolkit is on PATH");
    println!("cargo:rustc-env=MATMUL_PTX_SM75={}", ptx75.display());

    let b3_75 = try_nvcc("blake3", cu_b, &out, "sm_75")
        .expect("nvcc failed blake3_solver for sm_75");
    println!("cargo:rustc-env=BLAKE3_PTX_SM75={}", b3_75.display());

    // sm_86 — Ampere (RTX 30xx), optional
    if let Some(p) = try_nvcc("matmul", cu, &out, "sm_86") {
        println!("cargo:rustc-env=MATMUL_PTX_SM86={}", p.display());
        println!("cargo:rustc-cfg=has_sm86_ptx");
    }
    if let Some(p) = try_nvcc("blake3", cu_b, &out, "sm_86") {
        println!("cargo:rustc-env=BLAKE3_PTX_SM86={}", p.display());
        println!("cargo:rustc-cfg=has_blake3_sm86_ptx");
    }

    // sm_89 — Ada Lovelace (RTX 40xx), optional
    if let Some(p) = try_nvcc("matmul", cu, &out, "sm_89") {
        println!("cargo:rustc-env=MATMUL_PTX_SM89={}", p.display());
        println!("cargo:rustc-cfg=has_sm89_ptx");
    }
    if let Some(p) = try_nvcc("blake3", cu_b, &out, "sm_89") {
        println!("cargo:rustc-env=BLAKE3_PTX_SM89={}", p.display());
        println!("cargo:rustc-cfg=has_blake3_sm89_ptx");
    }
}