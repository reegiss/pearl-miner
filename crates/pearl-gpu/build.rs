use std::process::Command;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=kernels/matmul.cu");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("matmul.ptx");

    let status = Command::new("nvcc")
        .args(["--ptx", "-arch=sm_75",
               "kernels/matmul.cu",
               "-o"])
        .arg(&ptx_path)
        .status()
        .expect("nvcc not found — install CUDA toolkit");

    if !status.success() {
        panic!("nvcc failed to compile kernels/matmul.cu");
    }
}
