# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pearl Miner — a Rust implementation of a miner for the Pearl blockchain, a Layer-1 Proof-of-Useful-Work (PoUW) protocol where mining work is INT8 matrix multiplication. GPU compute used for AI inference is simultaneously used for mining.

Full protocol specification (in Portuguese): `prompt.md`.  
Architecture design and public APIs: `docs/superpowers/specs/2026-05-27-architecture-design.md`.  
Step-by-step implementation plan with TDD scaffolding: `docs/superpowers/plans/2026-05-27-pure-rust-foundation.md`.

**Current status:** Cargo workspace scaffold only — all crate `lib.rs` files are empty stubs.

## Build & Test Commands

```bash
# Check all crates compile
cargo check

# Run all tests across the workspace
cargo test

# Run tests for a single crate
cargo test -p pearl-commitment

# Run a specific test by name
cargo test -p pearl-commitment test_compute_is_deterministic

# Lint
cargo clippy --all-targets

# Build release
cargo build --release
```

`pearl-gpu` (Part 2) requires CUDA; its `build.rs` will compile the `.cu` kernel via `cudarc`.

## Workspace Architecture

```
pearl-miner/
├── Cargo.toml              # workspace root
├── crates/
│   ├── pearl-types/        # shared structs only — no logic, no internal deps
│   ├── pearl-commitment/   # CommitmentHash + MerkleTree (BLAKE3)
│   ├── pearl-noise/        # noise matrix generation EL·ER, FL·FR (BLAKE3 PRNG)
│   ├── pearl-gpu/          # CUDA TiledMatMul kernel — Part 2
│   ├── pearl-peel/         # clean product recovery A·B
│   └── pearl-block/        # block serialization and certificate validation
└── src/
    └── main.rs             # pearl-miner binary — Part 2
```

### Crate Dependency Rules

`pearl-types` is the only shared dependency. No logic crate depends on another logic crate — only the binary imports all of them. This keeps crates independently testable.

```
pearl-types  ←  all crates
    ↑
pearl-commitment   pearl-noise   pearl-block
         ↑               ↑
         └──── pearl-gpu ─────────────────┐
                    ↑                     │
              pearl-peel                  │
                    ↑                     ↓
              pearl-miner (binary) ←──────┘
```

## Mining Pipeline (Algorithm Flow)

```
Given A (m×k), B (k×n), miner config μ, chain state σ:

1. (sA, sB) ← CommitmentHash(A, B, μ, σ)
2. (EL, ER) ← NoiseGeneration(m, k; key=sA)    // E = EL·ER
3. (FR^T, FL^T) ← NoiseGeneration(n, k; key=sB) // F = FL·FR
4. A′ ← A + EL·ER,  B′ ← B + FL·FR
5. (C′, Blocks) ← TiledMatMul(A′, B′, sA, b)   // mine during matmul
6. C ← C′ − (A·FL)·FR − EL·(ER·B′)             // noise peel
7. Return C, Blocks
```

**Key optimization:** F depends only on sB (which depends only on B, μ, σ — not A), so B′ can be pre-noised once per chain-state update and reused across many A inputs.

## Key Algorithm Details

### Commitment Hash
```
κ   = BLAKE3(σ || μ)
HA  = BLAKE3(Flatten(A),   key=κ)   // A row-major
HB  = BLAKE3(Flatten(B^T), key=κ)   // B column-major
sB  = BLAKE3(κ || HB)
sA  = BLAKE3(sB || HA)
```
BLAKE3 is a Merkle tree internally — this enables Merkle path proofs for individual rows/columns without revealing the full matrix.

### Noise Matrices (seeded from sA / sB)
- `E = EL · ER` (seed=sA): EL is m×r in [-32,31]; ER is r×k — each column has exactly one +1 and one -1
- `F = FL · FR` (seed=sB): FL follows ER^T distribution; FR follows EL^T distribution
- A, B quantized to INT8 in [-64, 64]; noise in [-63, 63] — no INT8 overflow in A'=A+E, B'=B+F

### Tiled MatMul & Block Condition
Per output tile (i, j) of size tm×tn, across k/r depth steps:
1. Accumulate `Cblk` in INT32
2. Every depth step ℓ: `X = XOR(Cblk)`, `M[ℓ mod 16] = rotate_left(M[ℓ mod 16], 13) XOR X`
3. Block found when: `BLAKE3(M, key=sA) ≤ 2^(256−b) · r · tm · tn` (uint256 little-endian)

### Noise Peel (clean product recovery)
```
A·B = A'·B' − (A·FL)·FR − EL·(ER·B')
```
All three correction terms are O(n²·r) — negligible vs the O(n²·k) main product.

### Parameter Constraints
- r ∈ {32, 64, 128, 256, 512, 1024}; `16r ≤ k ≤ 4r²`; `k ≤ 2^16`; `64 | k`
- `tm·tn ≥ 32`; `k·(tm+tn) ≤ 2^22`; m, n ≤ 2^24

## Mining Pipeline (binary orchestration)

```
AI workload → A (m×k INT8), B (k×n INT8)
  → pearl_commitment::compute()        → Commitments { κ, HA, HB, sA, sB }
  → generate_e(sA), generate_f(sB)    → EL, ER, FL, FR
  → A' = A + EL·ER,  B' = B + FL·FR
  → GpuMiner::mine(A', B', …)          → Vec<FoundTile>
  → pearl_peel::recover()              → clean A·B → deliver to AI workload
  → if tile found: assemble PearlBlock + BlockCertificate → submit to P2P
```

- Commitment + noise generation: CPU-bound, dedicated thread
- `GpuMiner`: dedicated CUDA stream, separate from the inference stream (overhead target < 5%)
- Multi-GPU: one `GpuMiner` per device, all fed from a shared `(A, B)` queue via Tokio

## Blockchain Specifics

- Bitcoin UTXO fork; Taproot-only addresses; `OP_CAT` enabled
- Post-quantum XMSS signatures (`OP_CHECKXMSSSIG`, opcode #222)
- Block identity: `dSHA256(version || prev_hash || tx_root || time || nBits || pouw_meta)` where `pouw_meta = SHA256(zkSNARK public witness)` — stubbed as `SHA256(final_hash)` until `pearl-proof` is implemented
- Block time: 194s target; difficulty via WTEMA-N (N=2016, T=194s, τ=7 days)
- Supply: 2.1B PEARL; emission `E*(t) = S·H / ((t+H)·(t+H−1))`, H=650226 blocks

## zkSNARK (Plonky2) — Out of Scope for Part 1

- No trusted setup; post-quantum; 3-layer recursion → proof < 60 KB
- Public inputs: r, k, tm, tn, m, n, i, j, HA, HB, `BLAKE3(M, key=sA)`
- Private inputs: actual strips of A and B
