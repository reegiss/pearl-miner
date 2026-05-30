# GPU Hashrate Optimization — Design Spec

**Date:** 2026-05-30  
**Status:** Approved

## Problem

Current throughput: ~226 KH/s (2× RTX 30xx) with 36ms/job.  
After previous optimizations (rayon noise + WMMA 4-warp) expected ~1 MH/s.  
Goal: maximize tiles/second by eliminating wasted GPU and CPU time.

## Two-Phase Approach

### Phase 1 — Larger matrices (M=N=4096)

**Change:** `DEFAULT_M` and `DEFAULT_N` from 1024 → 4096 in `src/miner.rs`.  
Everything downstream adjusts automatically (buffer sizes, grid dims, tile counts).

**Why it helps:**
- Tiles/job: 4,096 → 65,536 (16×)
- At 1024 tiles with 4-warp WMMA: 256 blocks → poor SM wave coverage on 46 SMs
- At 65,536 tiles: 16,384 blocks → many waves, SMs stay busy
- Per-tile efficiency improves; kernel launch and memory transfer amortize over 16× more work

**GPU memory impact:**
- d_c grows from 4 MB → 64 MB (m×n×4 bytes INT32)
- Total per-GPU: ~72 MB (well within 7851 MB for RTX 3070/3060 Ti)

**Verify:** WMMA grid is `(n/16, (m+63)/64)` — with m=4096, gridDim.y=64. ✓  
DP4A grid is `(n/16, m/16)` — with m=4096, gridDim.y=256. ✓

### Phase 2 — Double-buffer CUDA stream pipelining

**Problem with current loop:**
```
job 0: [CPU noise] → [upload A'] → [GPU kernel] → [download M] → [BLAKE3 checks]
job 1:                                                              [CPU noise] → ...
```
GPU is idle during CPU noise generation; CPU is idle during GPU execution.

**Solution — two slots, two CUDA streams:**

```
Stream A: upload(0) → kernel(0) → download(0)       upload(2) → kernel(2) → ...
Stream B:              upload(1) → kernel(1) → download(1)         upload(3) → ...
CPU:      noise(0)  noise(1)  check(0)  noise(2)  check(1)  noise(3)  check(2) ...
```

CPU prepares job N+1 while GPU runs job N. Effective per-job time ≈ `max(cpu_time, gpu_time)`.

**API changes to `GpuMiner`:**

Add two methods alongside existing `mine()`:
- `mine_async(slot: usize, a_prime: &[i8], params, s_a)` — uploads A' to slot's `d_a` and launches kernel on slot's stream; returns immediately (non-blocking)
- `sync_and_check(slot: usize, params, s_a) -> Vec<FoundBlock>` — synchronizes slot's stream, downloads `d_m`, runs BLAKE3 difficulty checks

**New fields in `GpuMiner`:**
- `stream_b: Arc<CudaStream>` — second CUDA stream for slot 1
- `d_a_1: Mutex<CudaSlice<i8>>` — second A' buffer
- `d_m_1: Mutex<CudaSlice<u32>>` — second M-state buffer
- `d_c` and `d_b` remain single (B' is read-only, C output can be reused)

**Updated `mining_loop` pattern:**
```rust
// Startup: prepare slot 0
let a0 = prepare(job=0); gpu.mine_async(0, &a0, ...);

loop {
    // Prepare next job on CPU while GPU runs current
    let a_next = prepare(job + 1);
    // Collect results from current GPU job
    let found = gpu.sync_and_check(slot % 2, ...);
    // Launch next GPU job
    gpu.mine_async((slot + 1) % 2, &a_next, ...);
    // Process found blocks, log hashrate, check cancel
    ...
    slot += 1; job += 1;
}

// Drain last slot before exit
gpu.sync_and_check(slot % 2, ...);
```

**CUDA stream semantics:** Operations on different streams may overlap if there is no data dependency and GPU has enough resources. Upload on stream B can overlap with kernel on stream A (they use different device buffers). The `sync_and_check` call on stream A blocks only that stream, not stream B.

## Files Changed

- `src/miner.rs` — change `DEFAULT_M`, `DEFAULT_N`; update `mining_loop` to double-buffer pattern
- `crates/pearl-gpu/src/lib.rs` — add `stream_b`, `d_a_1`, `d_m_1`; add `mine_async()` and `sync_and_check()` methods
- No changes to CUDA kernels or other crates

## Expected Outcome

| Phase | tiles/job | Expected hashrate (2 GPU) |
|-------|-----------|--------------------------|
| Before | 4,096 | ~226 KH/s |
| After noise+WMMA fix | 4,096 | ~1 MH/s |
| + Phase 1 (4096×4096) | 65,536 | ~5–10 MH/s |
| + Phase 2 (pipeline) | 65,536 | ~8–15 MH/s |
