# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pearl Miner — a Rust implementation of a miner for the Pearl blockchain, a Layer-1 Proof-of-Useful-Work (PoUW) protocol where mining work is INT8 matrix multiplication. GPU compute (CUDA) is the primary workhorse; CPU handles commitment hashing and pool communication.

Full protocol specification (in Portuguese): `prompt.md`.  
Architecture design: `docs/superpowers/specs/2026-05-27-architecture-design.md`.  
Pool protocol fix spec: `docs/superpowers/specs/2026-05-30-pool-protocol-fix.md`.

**Current status:** Fully implemented and running against `us1.alphapool.tech:5566`. All crates have real code.

## Build & Test Commands

```bash
# Build release (requires CUDA toolkit with nvcc on PATH)
cargo build --release

# Check all crates compile (no CUDA needed for type-checking only)
cargo check

# Run all tests
cargo test

# Run tests for a single crate
cargo test -p pearl-commitment

# Run a specific test by name
cargo test -p pearl-commitment test_compute_is_deterministic

# Lint
cargo clippy --all-targets

# Run the miner
./target/release/pearl-miner \
  --wallet prl1p0jfkajyjgeu9eszntdfztcxyfjk5lh9mmvna69hd5ny83l8laqtq45rjx4 \
  --pool us1.alphapool.tech:5566
```

`pearl-gpu` requires CUDA. `build.rs` compiles `src/matmul.cu` via `nvcc -ptx -arch=sm_75` and embeds the PTX. CUDA JIT-compiles PTX to native code on first launch (then cached). To force a CUDA rebuild: `touch crates/pearl-gpu/src/matmul.cu && cargo build --release`.

To build a portable binary (avoids AVX512 crashes on machines without it): `RUSTFLAGS="-C target-cpu=x86-64" cargo build --release`.

## Workspace Architecture

```
pearl-miner/
├── src/
│   ├── main.rs           # CLI (clap), GPU discovery, channel wiring
│   ├── pool.rs           # TCP pool client — JSON-RPC, challenge/submit loop
│   └── miner.rs          # Mining orchestrator + BLAKE3 challenge solver
└── crates/
    ├── pearl-types/      # Shared structs only: MiningParams, FoundBlock, Commitments
    ├── pearl-commitment/ # BLAKE3 commitment chain (κ, HA, HB, sB, sA)
    ├── pearl-noise/      # Noise matrix generation (EL·ER, FL·FR) via BLAKE3 PRNG
    └── pearl-gpu/        # CUDA kernels (DP4A + WMMA), GpuMiner struct
```

`pearl-types` is the only shared dependency. No logic crate depends on another logic crate — only the binary imports all of them.

## Two-Layer Mining

The pool runs two independent challenges simultaneously:

### Layer 1 — BLAKE3 hash challenge (share submission)
When the pool sends `pearl.challenge { seed, difficulty }`, find nonce `N` (u64) such that:
```
BLAKE3(seed_bytes ++ N.to_le_bytes()) ≤ 2^(256 − difficulty)
```
Found nonce sent as `pearl.challenge_response { seed, nonce }` where nonce = `format!("{:016x}", N)`. Implemented in `solve_blake3_challenge()` in `src/miner.rs` using a dedicated rayon ThreadPool with N/2 cores to avoid starving noise generation.

### Layer 2 — PoUW GPU mining (block finding)
GPU computes noised matmul A'·B' and checks `BLAKE3(M-state, key=sA) ≤ threshold` per tile. Found tiles are logged as `[PoUW]` but **not yet submitted** to the pool — proof format not yet confirmed. Runs in parallel with the BLAKE3 solver.

## Channel Architecture (src/main.rs wiring)

```
pool::run()  ──watch::Sender<Option<(seed_hex, difficulty)>>──▶  Miner::run()
              ◀──mpsc::Sender<Submit { seed, nonce }>──────────  (BLAKE3 solver)
```

`watch` channel: pool pushes challenges; miner `await`s `changed()`. On each new challenge, all running threads (GPU miners + BLAKE3 solver) are cancelled via a shared `Arc<AtomicBool>`.

## Mining Pipeline Per Challenge (src/miner.rs)

1. Extract `sigma` (32 bytes) and `difficulty` from challenge
2. Pre-compute B' = B + FL·FR once per challenge (sB doesn't depend on A — paper §4.2)
3. Upload B' to each GPU via `GpuMiner::set_b()`
4. Spawn: one `spawn_blocking` per GPU + one for the BLAKE3 solver

**Per GPU job (`mining_loop`):**
- `random_matrix_i8()` → A (LCG PRNG)
- `compute_sa(&a, cc)` → sA (2 BLAKE3 calls)
- `generate_e(m, k, r, &sA)` → ENoise (rayon-parallel over rows)
- `apply_e(&a, &noise)` → A' (rayon-parallel over rows)
- `gpu.mine(&a_prime, params, &sA)` → Vec\<FoundBlock\>

Hashrate is logged every 100 jobs as `[miner] <rate> H/s · [profile/<kernel>] ...`.

## GpuMiner (crates/pearl-gpu/src/lib.rs)

Pre-allocates all device buffers at construction (`d_a`, `d_b`, `d_c`, `d_m`) — each a `Mutex<CudaSlice<T>>` to satisfy cudarc 0.19's builder-pattern borrow checker.

**Kernel selection:** `has_tensor_cores(name)` checks for RTX/A100/H100/V100/T4 in GPU name → WMMA kernel. Otherwise → DP4A kernel.

**WMMA kernel** (`tiled_matmul_wmma` in `matmul.cu`):
- `blockDim=(32,4)` — 4 warps per block, each warp handles one 16×16 tile
- All 4 warps cooperatively load the shared B-strip (128 threads)
- Grid: `(n/16, (m+63)/64)`. Shared memory: `5 × r × 16` bytes
- Requires `sm_72+`; stub exported for sm_75 PTX but only runs on RTX cards

**DP4A kernel** (`tiled_matmul_dp4a`):
- `blockDim=(16,16)` — 8 warps per block, one tile per block

**Difficulty check (CPU):** `BLAKE3(M[16 × u32], key=sA) ≤ 2^(256−b) × r × tm × tn` (little-endian uint256 comparison).

## Noise Matrices (crates/pearl-noise/src/lib.rs)

`ENoise`: EL is m×r dense INT8 in [-32,31]; ER encoded as sparse column pairs `(pos_row, neg_row)` — each column has exactly one +1 and one -1. `apply_e` is O(m×k×2), not O(m×k×r).

`FNoise`: FL encoded as sparse row pairs; FR is r×k dense INT8.

PRNG: `prng_u64(seed, domain, row, col)` = `blake3::keyed_hash(seed, [domain ‖ row_le64 ‖ col_le64])`. `gen_dense`, `gen_sparse_cols`, `gen_sparse_rows`, `apply_e`, and `apply_f` are all parallelized with rayon.

## Pool Protocol (src/pool.rs)

JSON-RPC over TCP. Pool sends:
- `pearl.challenge { seed: "<64-hex>", difficulty: <u32> }` — starts a new challenge
- `pearl.set_mining_params` — currently ignored (hardcoded params used)

Miner sends:
- `mining.authorize [wallet, password]` — on every connect
- `pearl.challenge_response { seed, nonce }` — BLAKE3 challenge solution

Reconnects immediately on clean close; waits 5s after errors.

## Key Constants (src/miner.rs)

```rust
DEFAULT_R=32, DEFAULT_K=512, DEFAULT_TM=16, DEFAULT_TN=16, DEFAULT_M=1024, DEFAULT_N=1024
```

## Algorithm Details

### Commitment Hash Chain
```
κ   = BLAKE3(σ ‖ μ)           // μ encodes r,k,tm,tn,m,n as 32 bytes
HB  = BLAKE3(Flatten(B^T), key=κ)   // B column-major
sB  = BLAKE3(κ ‖ HB)
HA  = BLAKE3(Flatten(A), key=κ)     // A row-major
sA  = BLAKE3(sB ‖ HA)
```

### Tiled MatMul Block Condition
Per output tile (i,j) over k/r depth steps ℓ:
1. Accumulate `Cblk` in INT32
2. `X = XOR(Cblk)`, `M[ℓ mod 16] = rotate_left(M[ℓ mod 16], 13) XOR X`
3. Block found when `BLAKE3(M, key=sA) ≤ 2^(256−b) · r · tm · tn`

### Noise Peel (not yet implemented — PoUW proof submission pending)
```
A·B = A'·B' − (A·FL)·FR − EL·(ER·B')
```

### Parameter Constraints
- r ∈ {32,64,128,256,512,1024}; `16r ≤ k ≤ 4r²`; `64 | k`; `k ≤ 2^16`
- `tm·tn ≥ 32`; `k·(tm+tn) ≤ 2^22`; m, n ≤ 2^24
- A, B in INT8 [-64,64]; noise in [-63,63] — no overflow in A'=A+E, B'=B+F
