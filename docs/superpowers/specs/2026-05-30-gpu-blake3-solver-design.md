# GPU BLAKE3 Challenge Solver — Design Spec

**Date:** 2026-05-30  
**Status:** Pending Review

## Problem
The current BLAKE3 pool challenge solver runs on the CPU. While the GPU performs matrix multiplications at ~80 TH/s (useful work), the pool measures hashrate based on the number of BLAKE3 challenge shares submitted. Because the CPU is exponentially slower than the GPU, the miner submits very few shares, resulting in a reported hashrate of nearly zero on the pool dashboard. Furthermore, attempts to submit PoUW tiles as shares caused pool disconnects due to format and frequency violations.

## Goal
Implement a high-performance BLAKE3 brute-force solver on the GPU that runs as a dedicated phase after each MatMul job. This will allow the miner to find valid challenge nonces at a rate consistent with the hardware's power, ensuring the 80 TH/s is correctly reflected on the pool.

## Design

### 1. CUDA Kernel: `solve_blake3_pool`
A new kernel in `matmul.cu` specifically designed for the pool challenge (`seed` ++ `nonce`).

- **Input:** 
  - `challenge_seed` (32 bytes)
  - `base_nonce` (starting point for this burst)
  - `target_difficulty` (minimum acceptable difficulty)
- **Execution:**
  - One-shot BLAKE3 compression: The input is 40 bytes (32 seed + 8 nonce), which fits in a single 64-byte BLAKE3 chunk. This minimizes rounds and memory access.
  - **Warp-Level Reduction:** Each warp (32 threads) finds the "Best Nonce" (numerically lowest hash) using `__shfl_xor_sync`.
  - **Atomic Global Update:** Only the best nonce of the entire grid is written to global memory using `atomicMin` on the hash value.
- **Throughput:** Capable of testing millions of nonces in < 1ms.

### 2. Integration: `crates/pearl-gpu/src/lib.rs`
- Update `GpuMiner` struct to hold the new `func_solve_pool` handle.
- Modify `mine()` to include **Stage 3: Pool Solver**:
  1. `generate_a_prime` (Noise)
  2. `tiled_matmul_wmma` (Useful Work)
  3. `solve_blake3_pool` (Burst search for shares)
- The solver will use a rolling `base_nonce` incremented by the number of threads per call.

### 3. Submission Strategy (Anti-Flood)
To prevent the "Connection closed by server" error:
- **Quality-over-Quantity:** The GPU will always find the "best" share in its search space.
- The Rust miner will only submit the share if:
  - It exceeds the pool's required difficulty (e.g., hash < 2^(256-32)).
  - It hasn't sent a share in the last 100ms (rate limit).
- This ensures the pool sees high-difficulty work without being overwhelmed by TCP packets.

## Impact
- **Visibility:** The worker will appear on the pool with a hashrate reflecting the GPU's true power.
- **Stability:** No more disconnects from sending malformed or too frequent shares.
- **Efficiency:** CPU usage will drop significantly as the heavy lifting moves to CUDA.

## Testing Strategy
1. Build with `Dockerfile.build` to verify CUDA syntax.
2. Run miner with `--pool` pointing to a local mock server or the real test pool.
3. Verify that `Authentication successful!` appears and shares are sent without disconnects.
4. Check the pool dashboard for the registered hashrate.
