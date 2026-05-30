# GPU BLAKE3 Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a high-performance GPU-based BLAKE3 brute-force solver to correctly reflect 80 TH/s hashrate on the pool dashboard.

**Architecture:** A new dedicated CUDA kernel `solve_blake3_pool` will be called after each MatMul job. It will use warp-level reduction and atomic operations to find the best nonce in a large search space, which will then be sent as a share to the pool.

**Tech Stack:** Rust, CUDA, BLAKE3.

---

## Task 1: Implement `solve_blake3_pool` CUDA Kernel

**Files:**
- Modify: `crates/pearl-gpu/src/matmul.cu`

- [ ] **Step 1: Add the `solve_blake3_pool` kernel**

Add this new kernel at the end of `crates/pearl-gpu/src/matmul.cu`. It handles a single-chunk BLAKE3 hash of `seed` (32B) + `nonce` (8B).

```cpp
extern "C" __global__ void solve_blake3_pool(
    const uint32_t* __restrict__ sa_key,    // 32-byte seed as 8 u32
    uint64_t                     base_nonce,
    uint32_t*       __restrict__ d_best_nonce_out, // 2 u32 (u64 nonce)
    uint32_t*       __restrict__ d_best_hash_out   // 8 u32 (u256 hash)
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = base_nonce + (uint64_t)tid;

    // Build the 64-byte message for BLAKE3 (32 seed + 8 nonce + 24 padding)
    uint32_t m[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) m[i] = sa_key[i];
    m[8]  = (uint32_t)(nonce & 0xFFFFFFFFu);
    m[9]  = (uint32_t)(nonce >> 32);
    #pragma unroll
    for (int i = 10; i < 16; i++) m[i] = 0;

    // Standard BLAKE3 single-chunk compression
    uint32_t v[16];
    v[0]=sa_key[0]; v[1]=sa_key[1]; v[2]=sa_key[2]; v[3]=sa_key[3];
    v[4]=sa_key[4]; v[5]=sa_key[5]; v[6]=sa_key[6]; v[7]=sa_key[7];
    v[8]=BLAKE3_IV[0]; v[9]=BLAKE3_IV[1]; v[10]=BLAKE3_IV[2]; v[11]=BLAKE3_IV[3];
    v[12]=0; v[13]=0; v[14]=64; v[15]=BLAKE3_DOMAIN;

    #pragma unroll
    for (int r = 0; r < 7; r++) {
        const uint8_t* s = BLAKE3_MSG_SCHED[r];
        blake3_g(v, 0, 4,  8, 12, m[s[ 0]], m[s[ 1]]);
        blake3_g(v, 1, 5,  9, 13, m[s[ 2]], m[s[ 3]]);
        blake3_g(v, 2, 6, 10, 14, m[s[ 4]], m[s[ 5]]);
        blake3_g(v, 3, 7, 11, 15, m[s[ 6]], m[s[ 7]]);
        blake3_g(v, 0, 5, 10, 15, m[s[ 8]], m[s[ 9]]);
        blake3_g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        blake3_g(v, 2, 7,  8, 13, m[s[12]], m[s[13]]);
        blake3_g(v, 3, 4,  9, 14, m[s[14]], m[s[15]]);
    }

    uint32_t hash[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) hash[i] = v[i] ^ v[i + 8];

    // --- Warp Reduction to find the best nonce in this warp ---
    uint32_t best_h = hash[7]; // High bits of hash (most significant for comparison)
    uint64_t best_n = nonce;

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        uint32_t remote_h = __shfl_xor_sync(0xffffffff, best_h, offset);
        uint64_t remote_n = __shfl_xor_sync(0xffffffff, best_n, offset);
        if (remote_h < best_h) {
            best_h = remote_h;
            best_n = remote_n;
        }
    }

    // Lane 0 of each warp tries to update the global best
    if ((tid & 31) == 0) {
        // Use atomicMin on the most significant word of the hash to find the "best"
        // Note: This is a heuristic. For 100% precision we'd need more complex atomics.
        uint32_t old = atomicMin(&d_best_hash_out[7], best_h);
        if (best_h < old) {
            d_best_nonce_out[0] = (uint32_t)(best_n & 0xFFFFFFFFu);
            d_best_nonce_out[1] = (uint32_t)(best_n >> 32);
            #pragma unroll
            for (int i = 0; i < 7; i++) d_best_hash_out[i] = hash[i];
        }
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `docker run --rm pearl-miner-build cargo build --release`
Expected: `Finished`

- [ ] **Step 3: Commit**

```bash
git add crates/pearl-gpu/src/matmul.cu
git commit -m "feat: implement solve_blake3_pool GPU kernel"
```

---

## Task 2: Update Rust Wrapper and GpuMiner

**Files:**
- Modify: `crates/pearl-gpu/src/lib.rs`

- [ ] **Step 1: Update `GpuMiner` struct and initialization**

Add `func_solve_pool`, `d_best_nonce`, and `d_best_hash` to the struct. Initialize them in `new()`.

```rust
// lib.rs
pub struct GpuMiner {
    // ... existing ...
    func_solve_pool: CudaFunction,
    d_best_nonce:    Mutex<CudaSlice<u32>>, // 2 words
    d_best_hash:     Mutex<CudaSlice<u32>>, // 8 words
    // ...
}

// In GpuMiner::new():
let func_solve_pool = module.load_function("solve_blake3_pool")?;
let d_best_nonce    = Mutex::new(stream.alloc_zeros::<u32>(2)?);
let d_best_hash     = Mutex::new(stream.alloc_zeros::<u32>(8)?);
// ... update return Self ...
```

- [ ] **Step 2: Update `mine()` to launch the solver**

Reset the best hash to `0xFFFFFFFF` before launch. Retrieve the result after launch.

```rust
// In mine():
{
    // Reset best hash to max
    let max_hash = [u32::MAX; 8];
    self.stream.memcpy_htod(&max_hash, &mut *self.d_best_hash.lock().unwrap())?;

    let threads = 1024u32;
    let blocks  = 64u32; // Test 64k nonces per burst
    let base_nonce = job_seed * 1_000_000u64; // Simple offset for now
    let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
    let mut b = self.stream.launch_builder(&self.func_solve_pool);
    b.arg(&*self.d_sa.lock().unwrap());
    b.arg(&base_nonce);
    b.arg(&mut *self.d_best_nonce.lock().unwrap());
    b.arg(&mut *self.d_best_hash.lock().unwrap());
    unsafe { b.launch(cfg) }?;
}

// In the return from mine(), include the best nonce/hash
```

- [ ] **Step 3: Commit**

```bash
git add crates/pearl-gpu/src/lib.rs
git commit -m "feat: integrate solve_blake3_pool into GpuMiner"
```

---

## Task 3: Share Submission Logic in `miner.rs`

**Files:**
- Modify: `src/miner.rs`

- [ ] **Step 1: Update `mining_loop` to send found shares**

Modify the `mining_loop` to receive the best nonce from `gpu.mine()` and send it to `submit_tx` if it passes the pool difficulty.

```rust
// In mining_loop:
let (blocks, best_nonce, best_hash, timing) = gpu.mine(...)?;
// ...
if check_difficulty(&best_hash, params.difficulty) {
    let nonce_hex = format!("{:016x}", best_nonce);
    let seed_hex: String = params.sigma.iter().map(|b| format!("{:02x}", b)).collect();
    let _ = submit_tx.blocking_send(Submit { seed: seed_hex, nonce: nonce_hex });
}
```

- [ ] **Step 2: Final build verification**

Run: `docker run --rm pearl-miner-build cargo build --release`
Expected: `Finished`

- [ ] **Step 3: Commit**

```bash
git add src/miner.rs
git commit -m "feat: submit GPU-solved challenge shares to pool"
```
