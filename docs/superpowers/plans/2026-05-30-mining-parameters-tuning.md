# Hashrate Optimization Parameter Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Modify the default mining parameters in `src/miner.rs` to quadruple the probability of finding a valid block while fully respecting all protocol and hardware rules.

**Architecture:** We are updating constant parameters to `r=128`, `k=2048`, `m=8192`, `n=8192`. No structural changes are needed because the math automatically adjusts within the existing logic.

**Tech Stack:** Rust, Pearl Miner Codebase.

---

## Task 1: Update Constants in `miner.rs`

**Files:**
- Modify: `src/miner.rs`

- [ ] **Step 1: Write the updated constants**

Modify the constant block at the top of `src/miner.rs` (around line 12).

Replace:
```rust
const DEFAULT_R:  usize = 32;
const DEFAULT_K:  usize = 512;
const DEFAULT_TM: usize = 16;
const DEFAULT_TN: usize = 16;
const DEFAULT_M:  usize = 4096;
const DEFAULT_N:  usize = 4096;
```

With:
```rust
const DEFAULT_R:  usize = 128;
const DEFAULT_K:  usize = 2048;
const DEFAULT_TM: usize = 16;
const DEFAULT_TN: usize = 16;
const DEFAULT_M:  usize = 8192;
const DEFAULT_N:  usize = 8192;
```

- [ ] **Step 2: Run test to verify it compiles**

Run: `cargo check`
Expected: `Finished`

- [ ] **Step 3: Commit**

```bash
git add src/miner.rs
git commit -m "perf: optimize mining parameters (r=128, k=2048) for 4x target multiplier"
```

- [ ] **Step 4: Execute the miner to observe the hashrate**

Run a brief execution to ensure it works correctly with the pool.
Run: `cargo run --release -- --wallet test --pool 127.0.0.1:0`
*(Note: A local run might fail on pool connection, but we just want to ensure it boots without parameter validation errors. In the real world, the user runs the binary.)*