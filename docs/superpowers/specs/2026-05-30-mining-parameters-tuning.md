# Hashrate Optimization via Protocol Parameter Tuning — Design Spec

**Date:** 2026-05-30  
**Status:** Pending Review

## Problem
The miner's current hashrate is sub-optimal compared to the hardware's theoretical maximum. While the CUDA kernels have been highly optimized (WMMA Tensor Cores, DP4A, memory prefetching), the cryptographic boundary conditions defined by the Pearl PoUW protocol dictate how often a valid block is found. The current default configuration uses a low noise rank (`r = 32`), which mathematically suppresses the block acceptance probability per tile.

## Goal
Quadruple the probability of finding a valid block (effectively a 4x multiplier on the "useful hashrate") by exploiting the protocol's acceptance inequality, without modifying the underlying CUDA kernel architectures or violating any blockchain verifier rules.

## Protocol Mechanics & Analysis
The Pearl whitepaper (§4.5) defines the block opening condition as:
`BLAKE3(M, key=sA) ≤ 2^(256-b) * r * tm * tn`

The acceptance target grows linearly with `r` (noise rank), `tm` (tile height), and `tn` (tile width). 

### Current State (`src/miner.rs`)
- `r = 32`
- `tm = 16`, `tn = 16`
- Target Multiplier: `32 * 16 * 16 = 8,192`

### Constraints (from `prompt.md`)
1. `r` must be in `{32, 64, 128, 256, 512, 1024}`.
2. `k` (common dimension) must satisfy: `16*r <= k <= 4*r^2` and `k % 64 == 0`.
3. `tm * tn >= 32` and `k * (tm + tn) <= 2^22`.
4. CUDA kernels (`matmul.cu`) are hardcoded for `16x16` tiles for WMMA Tensor Cores. Changing `tm` or `tn` would require rewriting the CUDA fragments.

## Proposed Design (The "Sweet Spot")

By increasing `r`, we increase the multiplier. However, increasing `r` forces us to increase `k` (due to the `16*r <= k` rule), which linearly increases the depth of the matrix multiplication, slowing down the tile computation. We need a balance.

We will increase `r` from 32 to **128**.
- **New Target Multiplier:** `128 * 16 * 16 = 32,768` (**4x increase** in acceptance probability per tile).

To satisfy the verifier rules for `r = 128`:
- Minimum `k = 16 * 128 = 2048`.
- We will set `k = 2048`.

To offset the deeper `k` and ensure the GPU's Streaming Multiprocessors (SMs) are fully saturated with work (avoiding CPU synchronization stalls), we will double the global matrix dimensions `M` and `N`.

### Configuration Changes (`src/miner.rs`)
```rust
const DEFAULT_R:  usize = 128;   // Increased from 32 (4x multiplier)
const DEFAULT_K:  usize = 2048;  // Increased from 512 (Minimum legal k for r=128)
const DEFAULT_TM: usize = 16;    // Unchanged (WMMA compat)
const DEFAULT_TN: usize = 16;    // Unchanged (WMMA compat)
const DEFAULT_M:  usize = 8192;  // Increased from 4096 (Better GPU saturation)
const DEFAULT_N:  usize = 8192;  // Increased from 4096 (Better GPU saturation)
```

## Impact
1. **Mathematical Advantage:** The threshold for a valid block is 4x larger.
2. **Hardware Efficiency:** Matrices of `8192x8192` provide `262,144` output tiles per job (vs `65,536` previously), keeping the GPU running hot longer between D2H memory transfers and CPU PRNG stalls.
3. **Safety:** Fully compliant with all whitepaper and prompt constraints. No CUDA code needs to be modified. No ZK-SNARK verifier rules are broken.

## Testing Strategy
- Compile the code with the new parameters.
- Verify that the program runs and the pool accepts the `[PoUW]` blocks without rejecting them for parameter violations.
- Observe the hashrate log to confirm the proportional increase in MH/s.