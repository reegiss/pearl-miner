# Multi-Arch WMMA Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compile native PTX for sm_75/sm_86/sm_89 and add an Ampere WMMA kernel using `m16n8k32` fragments, eliminating the JIT-from-sm_75 penalty on RTX 30xx/40xx.

**Architecture:** `build.rs` compiles up to three PTX files (sm_75 mandatory, sm_86/sm_89 optional). `lib.rs` detects the device SM version at construction and loads the best available PTX. `matmul.cu` selects the right WMMA kernel at compile time via `__CUDA_ARCH__` guards: Turing (sm_72–79) keeps the existing `m16n16k16` path; Ampere/Ada (sm_80+) uses `m16n8k32` with one sub-step per depth step for r=32.

**Tech Stack:** Rust, CUDA (nvcc), cudarc 0.19, nvcuda::wmma, cargo build scripts.

---

## Files

| File | Change |
|------|--------|
| `crates/pearl-gpu/build.rs` | Add optional sm_86 / sm_89 compilation; emit `rustc-cfg` flags |
| `crates/pearl-gpu/src/matmul.cu` | Split WMMA kernel into Turing path (sm_72–79) and Ampere path (sm_80+) |
| `crates/pearl-gpu/src/lib.rs` | Embed all PTX statics; detect device SM at runtime; replace `MATMUL_PTX` single static |

---

## Task 1: Multi-arch build in `build.rs`

**Files:**
- Modify: `crates/pearl-gpu/build.rs`

- [ ] **Replace `build.rs` entirely with multi-arch version**

```rust
use std::path::PathBuf;
use std::process::Command;

fn try_nvcc(cu: &str, out: &PathBuf, arch: &str) -> Option<PathBuf> {
    let ptx = out.join(format!("matmul_{}.ptx", arch));
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
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let cu  = "src/matmul.cu";

    // sm_75 — mandatory baseline (Turing: GTX 16xx, RTX 20xx)
    let ptx75 = try_nvcc(cu, &out, "sm_75")
        .expect("nvcc failed for sm_75 — ensure CUDA toolkit is on PATH");
    println!("cargo:rustc-env=MATMUL_PTX_SM75={}", ptx75.display());

    // sm_86 — Ampere (RTX 30xx), optional
    if let Some(p) = try_nvcc(cu, &out, "sm_86") {
        println!("cargo:rustc-env=MATMUL_PTX_SM86={}", p.display());
        println!("cargo:rustc-cfg=has_sm86_ptx");
    }

    // sm_89 — Ada Lovelace (RTX 40xx), optional
    if let Some(p) = try_nvcc(cu, &out, "sm_89") {
        println!("cargo:rustc-env=MATMUL_PTX_SM89={}", p.display());
        println!("cargo:rustc-cfg=has_sm89_ptx");
    }
}
```

- [ ] **Verify build still compiles on this machine (sm_75 only)**

```bash
cargo build --release 2>&1 | grep -E "^error|Finished"
```
Expected: `Finished` (sm_86/sm_89 silently skipped — local GPU is GTX 1660 SUPER)

- [ ] **Commit**

```bash
git add crates/pearl-gpu/build.rs
git commit -m "build: multi-arch PTX — sm_75 mandatory + sm_86/sm_89 optional"
```

---

## Task 2: Ampere WMMA kernel (`m16n8k32`) in `matmul.cu`

**Files:**
- Modify: `crates/pearl-gpu/src/matmul.cu`

Background: Ampere (sm_80+) prefers `m16n8k32` INT8 WMMA over Turing's `m16n16k16`. For r=32 the `m16n8k32` variant needs only 1 sub-step per depth step (k=32 matches r), using two accumulator halves per 16×16 tile (left cols 0–7, right cols 8–15). The M-state XOR remains 8 elements per thread (4 from each half) — identical total information to Turing's single 8-element accumulator.

- [ ] **Replace the `#if __CUDA_ARCH__ >= 720` WMMA block**

Find this in `matmul.cu`:
```c
/* ------------------------------------------------------------------ */
/*  Kernel B: WMMA INT8 tensor cores (sm_72+, RTX cards)               */
/* ------------------------------------------------------------------ */

#if __CUDA_ARCH__ >= 720
```

Replace the entire WMMA section (everything from that comment through the closing `#else` stub) with:

```c
/* ------------------------------------------------------------------ */
/*  Kernel B-Ampere: WMMA INT8, m16n8k32 (sm_80+, RTX 30xx/40xx)      */
/*  Each warp: two 16×8 acc halves covering the 16×16 output tile.     */
/*  sub_steps = r/32 (= 1 for the default r=32).                       */
/* ------------------------------------------------------------------ */

#if __CUDA_ARCH__ >= 800

#include <mma.h>
using namespace nvcuda::wmma;

extern "C" __global__ void tiled_matmul_wmma(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t*      __restrict__ C,    // unused — ABI compat
    uint32_t*     __restrict__ M_out,
    int m, int n, int k, int r
) {
    /*
     * blockDim = (32, 8)  — 8 warps, each owns one 16×16 output tile
     * gridDim  = (n/16, (m+127)/128)
     * smem     = 9 × r × 16 bytes  (Bs shared + 8 × wAs per warp)
     *
     * Two m16n8k32 accumulators per warp:
     *   acc_l → C[tile_i*16..(+16)][tile_j*16..(+8)]   (left  8 cols)
     *   acc_r → C[tile_i*16..(+16)][tile_j*16+8..(+8)] (right 8 cols)
     * Together they cover the full 16×16 output tile.
     */
    extern __shared__ int32_t smem_wmma[];

    const int warp_id    = threadIdx.y;   // 0..7
    const int lane       = threadIdx.x;   // 0..31
    const int tid        = warp_id * 32 + lane;

    const int tile_i     = blockIdx.y * 8 + warp_id;
    const int tile_j     = blockIdx.x;
    const bool active    = (tile_i * 16 < m && tile_j * 16 < n);
    const int  a_row_base = tile_i * 16;
    const int  b_col_base = tile_j * 16;

    // Shared memory: Bs[r×16] then 8 × wAs[16×r]
    int8_t* Bs  = (int8_t*)smem_wmma;
    int8_t* wAs = Bs + r * 16 + warp_id * 16 * r;

    // Two accumulator halves for the 16×16 output tile
    fragment<accumulator, 16, 8, 32, int32_t> acc_l, acc_r;
    fill_fragment(acc_l, 0);
    fill_fragment(acc_r, 0);

    uint32_t M[16];
    #pragma unroll
    for (int q = 0; q < 16; q++) M[q] = 0;

    const int num_steps = k / r;
    const int sub_steps = r / 32;   // = 1 for r=32 (default)

    for (int ell = 0; ell < num_steps; ell++) {
        const int s = ell * r;

        // All 256 threads cooperatively load Bs[r × 16]
        for (int idx = tid; idx < r * 16; idx += 256) {
            int row = idx / 16, col = idx % 16;
            int gcol = b_col_base + col;
            Bs[row * 16 + col] = ((s + row) < k && gcol < n)
                ? B[(s + row) * n + gcol] : (int8_t)0;
        }
        __syncthreads();

        if (active) {
            // Each warp loads its own As[16 × r]
            for (int idx = lane; idx < 16 * r; idx += 32) {
                int row = idx / r, col = idx % r;
                int grow = a_row_base + row;
                wAs[idx] = (grow < m && (s + col) < k)
                    ? A[grow * k + s + col] : (int8_t)0;
            }
            __syncwarp();

            // m16n8k32 sub-steps: 1 iteration for r=32
            for (int sub = 0; sub < sub_steps; sub++) {
                fragment<matrix_a, 16, 8, 32, int8_t, row_major> a_frag;
                fragment<matrix_b, 16, 8, 32, int8_t, row_major> b_frag_l, b_frag_r;

                // A strip: wAs[0..15][sub*32..sub*32+31], stride = r
                load_matrix_sync(a_frag, wAs + sub * 32, r);

                // B left half:  Bs[sub*32..sub*32+31][0..7],  stride = 16
                load_matrix_sync(b_frag_l, Bs + sub * 32 * 16,     16);
                // B right half: Bs[sub*32..sub*32+31][8..15], stride = 16
                load_matrix_sync(b_frag_r, Bs + sub * 32 * 16 + 8, 16);

                mma_sync(acc_l, a_frag, b_frag_l, acc_l);
                mma_sync(acc_r, a_frag, b_frag_r, acc_r);
            }

            // M-state XOR: 4 elements from each half = 8 total per thread
            // (same information as Turing's single 16×16 accumulator with 8 elements)
            uint32_t local_xor = 0;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                local_xor ^= (uint32_t)acc_l.x[i];
                local_xor ^= (uint32_t)acc_r.x[i];
            }
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor, 16);
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  8);
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  4);
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  2);
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  1);
            if (lane == 0)
                M[ell & 15] = rol32(M[ell & 15], 13) ^ local_xor;
        }

        __syncthreads();
    }

    if (!active) return;

    if (lane == 0) {
        int num_tiles_n = (n + 15) / 16;
        int base = (tile_i * num_tiles_n + tile_j) * 16;
        #pragma unroll
        for (int q = 0; q < 16; q++) M_out[base + q] = M[q];
    }
}

/* ------------------------------------------------------------------ */
/*  Kernel B-Turing: WMMA INT8, m16n16k16 (sm_72–79, RTX 20xx)        */
/* ------------------------------------------------------------------ */

#elif __CUDA_ARCH__ >= 720

#include <mma.h>
using namespace nvcuda::wmma;

extern "C" __global__ void tiled_matmul_wmma(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t*      __restrict__ C,
    uint32_t*     __restrict__ M_out,
    int m, int n, int k, int r
) {
    /*
     * blockDim = (32, 8) — 8 warps, each owns one 16×16 output tile.
     * gridDim  = (n/16, (m+127)/128)
     * smem     = 9 × r × 16 bytes
     */
    extern __shared__ int32_t smem_wmma[];

    const int warp_id    = threadIdx.y;
    const int lane       = threadIdx.x;
    const int tid        = warp_id * 32 + lane;

    const int tile_i     = blockIdx.y * 8 + warp_id;
    const int tile_j     = blockIdx.x;
    const bool active    = (tile_i * 16 < m && tile_j * 16 < n);
    const int  a_row_base = tile_i * 16;
    const int  b_col_base = tile_j * 16;

    int8_t* Bs  = (int8_t*)smem_wmma;
    int8_t* wAs = Bs + r * 16 + warp_id * 16 * r;

    fragment<accumulator, 16, 16, 16, int32_t> acc;
    fill_fragment(acc, 0);

    uint32_t M[16];
    #pragma unroll
    for (int q = 0; q < 16; q++) M[q] = 0;

    const int num_steps = k / r;

    for (int ell = 0; ell < num_steps; ell++) {
        const int s = ell * r;

        for (int idx = tid; idx < r * 16; idx += 256) {
            int row = idx / 16, col = idx % 16;
            int gcol = b_col_base + col;
            Bs[row * 16 + col] = ((s + row) < k && gcol < n)
                ? B[(s + row) * n + gcol] : (int8_t)0;
        }
        __syncthreads();

        if (active) {
            for (int idx = lane; idx < 16 * r; idx += 32) {
                int row = idx / r, col = idx % r;
                int grow = a_row_base + row;
                wAs[idx] = (grow < m && (s + col) < k)
                    ? A[grow * k + s + col] : (int8_t)0;
            }
            __syncwarp();

            const int sub_steps = r / 16;
            for (int sub = 0; sub < sub_steps; sub++) {
                fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
                fragment<matrix_b, 16, 16, 16, int8_t, row_major> b_frag;
                load_matrix_sync(a_frag, wAs + sub * 16,       r);
                load_matrix_sync(b_frag, Bs  + sub * 16 * 16, 16);
                mma_sync(acc, a_frag, b_frag, acc);
            }

            uint32_t local_xor = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) local_xor ^= (uint32_t)acc.x[i];
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor, 16);
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  8);
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  4);
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  2);
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  1);
            if (lane == 0)
                M[ell & 15] = rol32(M[ell & 15], 13) ^ local_xor;
        }

        __syncthreads();
    }

    if (!active) return;

    if (lane == 0) {
        int num_tiles_n = (n + 15) / 16;
        int base = (tile_i * num_tiles_n + tile_j) * 16;
        #pragma unroll
        for (int q = 0; q < 16; q++) M_out[base + q] = M[q];
    }
}

#else

extern "C" __global__ void tiled_matmul_wmma(
    const int8_t*, const int8_t*, int32_t*, uint32_t*,
    int, int, int, int) {}

#endif
```

- [ ] **Verify sm_75 build still compiles (Turing path taken on this machine)**

```bash
touch crates/pearl-gpu/src/matmul.cu && cargo build --release 2>&1 | grep -E "^error|Finished"
```
Expected: `Finished`

- [ ] **Commit**

```bash
git add crates/pearl-gpu/src/matmul.cu
git commit -m "perf: Ampere WMMA kernel — m16n8k32 under __CUDA_ARCH__ >= 800"
```

---

## Task 3: Runtime PTX selection in `lib.rs`

**Files:**
- Modify: `crates/pearl-gpu/src/lib.rs`

- [ ] **Replace the single `MATMUL_PTX` static and `new()` PTX loading**

Replace the top of `lib.rs` (lines 1–20 up to and including the `static MATMUL_PTX` line) with:

```rust
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::driver::sys::CUdevice_attribute::{
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as CC_MAJOR,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as CC_MINOR,
};
use cudarc::nvrtc::Ptx;
use pearl_types::{FoundBlock, MiningParams};
use std::sync::{Arc, Mutex};
use thiserror::Error;

static PTX_SM75: &str = include_str!(env!("MATMUL_PTX_SM75"));

#[cfg(has_sm86_ptx)]
static PTX_SM86: &str = include_str!(env!("MATMUL_PTX_SM86"));

#[cfg(has_sm89_ptx)]
static PTX_SM89: &str = include_str!(env!("MATMUL_PTX_SM89"));

fn best_ptx(sm: u32) -> &'static str {
    #[cfg(has_sm89_ptx)]
    if sm >= 89 { return PTX_SM89; }
    #[cfg(has_sm86_ptx)]
    if sm >= 86 { return PTX_SM86; }
    PTX_SM75
}
```

- [ ] **Update `GpuMiner::new()` to use `best_ptx()`**

In the `GpuMiner::new()` function, replace these lines:

```rust
        let module = ctx.load_module(Ptx::from_src(MATMUL_PTX))?;
```

with:

```rust
        let major = ctx.attribute(CC_MAJOR).unwrap_or(7) as u32;
        let minor = ctx.attribute(CC_MINOR).unwrap_or(5) as u32;
        let sm    = major * 10 + minor;
        let module = ctx.load_module(Ptx::from_src(best_ptx(sm)))?;
```

Also update the `GpuInfo` and startup log to show SM version. In `main.rs`, `gpu.info.name` already prints. Optionally add `sm` to `GpuInfo` for the log:

Add field to `GpuInfo`:
```rust
pub struct GpuInfo {
    pub name:     String,
    pub mem_mb:   usize,
    pub use_wmma: bool,
    pub sm:       u32,      // e.g. 86 for RTX 3080
}
```

In `GpuMiner::new()` update the `GpuInfo` construction:
```rust
            info: GpuInfo { name, mem_mb, use_wmma, sm },
```

In `src/main.rs` (the GPU init print), add `sm` display:
```rust
println!("[gpu:{}] {} · {} MB · {} · sm_{}", id, gpu.info.name, gpu.info.mem_mb, kernel, gpu.info.sm);
```

- [ ] **Build and verify**

```bash
cargo build --release 2>&1 | grep -E "^error|Finished"
```
Expected: `Finished`

- [ ] **Commit**

```bash
git add crates/pearl-gpu/src/lib.rs src/main.rs
git commit -m "perf: runtime PTX selection — load sm_86/sm_89 native code when available"
```

---

## Verification

After deploying to the RTX machine, the startup log should show:

```
[gpu:0] NVIDIA GeForce RTX 3080 · 10240 MB · WMMA tensor cores · sm_86
[gpu:1] NVIDIA GeForce RTX 3080 · 10240 MB · WMMA tensor cores · sm_86
```

And after 5 seconds of mining:

```
[miner] X.XX MH/s [wmma] kernel=YYms dtoh=0.1ms
```

If `kernel` drops significantly (target: < 20ms for RTX 30xx), the multi-arch compilation is working. If `kernel` stays at ~51ms, the bottleneck is not the JIT penalty and needs deeper profiling (shared memory layout, pipeline stalls, etc.).
