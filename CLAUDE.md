# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pearl Miner — implementation of a miner for the Pearl blockchain (`pearlresearch.ai`), a Layer-1 Proof-of-Useful-Work (PoUW) protocol where mining work is INT8 matrix multiplication (MatMul), making GPU compute used for AI inference simultaneously useful for mining.

The full protocol specification is in `prompt.md`.

## Planned Module Structure

| Module | Purpose |
|---|---|
| `commitment.rs` / `.py` | `CommitmentHash(A, B, μ, σ) → (sA, sB)`; Merkle tree over rows of A (row-major) and columns of B (col-major) using BLAKE3 |
| `noise_gen.rs` / `.py` | `NoiseGeneration(m, k, r, seed) → (EL, ER)`; BLAKE3-based PRNG with per-entry domain separation |
| `matmul_kernel.cu` | CUDA kernel: INT8 tiled MatMul with INT32 accumulation; BLAKE3 condition check per tile; GPU interleaving overhead < 5% |
| `noise_peel.rs` / `.py` | Recover clean product `A·B = A'·B' − (A·FL)·FR − EL·(ER·B')` |
| `proof.rs` / `.py` | Block Opening Proof (Merkle-based); Plonky2 zkSNARK integration for production |
| `block.rs` / `.py` | Pearl block serialization/deserialization; certificate validation |
| `miner.rs` / `.py` | Main loop: receives A, B from AI workload, runs full pipeline, submits valid blocks; multi-GPU support (DP/TP/PP) |

## Key Algorithm Details

### Commitment Hash
```
κ      = BLAKE3(σ || μ)
HA     = BLAKE3(Flatten(A),   key=κ)   // A row-major
HB     = BLAKE3(Flatten(B^T), key=κ)   // B column-major
sB     = BLAKE3(κ || HB)
sA     = BLAKE3(sB || HA)
```

### Noise Matrices
- `E = EL · ER` (rank-r, seed = sA): EL is m×r in [-32,31]; ER is r×k, each column has exactly one +1 and one -1
- `F = FL · FR` (rank-r, seed = sB): FL follows ER^T distribution; FR follows EL^T distribution
- A, B quantized to INT8 in [-64, 64]; noise E, F in [-63, 63] (no INT8 overflow)

### Tiled MatMul & Block Condition
Per output tile (i, j) of size tm×tn:
1. Accumulate `Cblk` in INT32
2. Maintain state `M[16]` (INT32); every r-depth step: `X = XOR(Cblk elements)`, `M[ℓ mod 16] = rotate_left(M[ℓ mod 16], 13) XOR X`
3. Block found when: `BLAKE3(M, key=sA) ≤ 2^(256−b) · r · tm · tn` (uint256 little-endian)

### Parameter Constraints
- `16r ≤ k ≤ 4r²`, `k ≤ 2^16`, `64 | k`
- r ∈ {32, 64, 128, 256, 512, 1024}
- `tm·tn ≥ 32`; `k·(tm+tn) ≤ 2^22`
- m, n ≤ 2^24

## Blockchain Specifics

- Bitcoin UTXO fork; Taproot-only addresses
- Post-quantum XMSS signatures (`OP_CHECKXMSSSIG`, opcode #222); `OP_CAT` enabled
- Block identity: `dSHA256(version || prev_hash || tx_root || time || nBits || pouw_meta)` where `pouw_meta = SHA256(zkSNARK public witness)`
- Block time: 194s target; difficulty via WTEMA-N (N=2016, T=194s, τ=7 days)
- Supply: 2.1B PEARL; emission `E*(t) = S·H / ((t+H)·(t+H−1))`, H=650226 blocks

## zkSNARK (Plonky2)
- No trusted setup; post-quantum
- Public: r, k, tm, tn, m, n, i, j, HA, HB, `BLAKE3(M, key=sA)`
- Private: actual strips of A and B
- Target: 3-layer recursion → proof < 60 KB

## Reference Hardware & Performance
- 4×H200 GPUs; reference throughput ~806–981 TMADs/s (useful)
- Reference model: LLaMA 3.3 70B (Pearl-certified)
- Official repo: github.com/pearl-research-labs/pearl
