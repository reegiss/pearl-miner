# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pearl Miner — implementation of a miner for the Pearl blockchain (`pearlresearch.ai`), a Layer-1 Proof-of-Useful-Work (PoUW) protocol where mining work is INT8 matrix multiplication (MatMul), making GPU compute used for AI inference simultaneously useful for mining. Chain launched April 27, 2025.

The full protocol specification is in `prompt.md`. The whitepaper is in `paper.md`.

## Planned Module Structure

| Module | Purpose |
|---|---|
| `commitment.rs` / `.py` | `CommitmentHash(A, B, μ, σ) → (sA, sB)`; BLAKE3 Merkle tree over rows of A (row-major) and columns of B (col-major) |
| `noise_gen.rs` / `.py` | `NoiseGeneration(m, k, r, seed) → (EL, ER)`; BLAKE3-based PRNG with per-entry domain separation |
| `matmul_kernel.cu` | CUDA kernel: INT8 tiled MatMul with INT32 accumulation; BLAKE3 condition check per tile; GPU interleaving overhead < 5% |
| `noise_peel.rs` / `.py` | Recover clean product `A·B = A′·B′ − (A·FL)·FR − EL·(ER·B′)` |
| `proof.rs` / `.py` | Block Opening Proof (Merkle-based); Plonky2 zkSNARK integration for production |
| `block.rs` / `.py` | Pearl block serialization/deserialization; certificate validation |
| `miner.rs` / `.py` | Main loop: receives A, B from AI workload, runs full pipeline, submits valid blocks; multi-GPU support (DP/TP/PP) |

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
κ      = BLAKE3(σ || μ)
HA     = BLAKE3(Flatten(A),   key=κ)   // A row-major
HB     = BLAKE3(Flatten(B^T), key=κ)   // B column-major (transpose)
sB     = BLAKE3(κ || HB)
sA     = BLAKE3(sB || HA)
```
BLAKE3 is a Merkle tree internally — this enables Merkle path proofs for individual rows/columns without revealing the full matrix.

### Noise Matrices
- `E = EL · ER` (rank-r, seed = sA): EL ∈ [−32, 31]^(m×r) uniform; ER ∈ {−1,0,+1}^(r×k), each column has exactly one +1 and one −1
- `F = FL · FR` (rank-r, seed = sB): FL shares ER^T distribution; FR shares EL^T distribution
- A, B quantized to INT8 in [−64, 64]; noise entries in [−63, 63] → noised matrices stay within INT8 (no overflow)

### Tiled MatMul & Block Condition
Per output tile (i, j) of size tm×tn, over ⌊k/r⌋ full-rank depth steps:
1. Accumulate `Cblk` in INT32 (add A′[i:i+tm, ℓr:(ℓ+1)r] · B′[ℓr:(ℓ+1)r, j:j+tn])
2. After each full-rank step ℓ: `X = XOR of all tm·tn entries in Cblk`, then update 512-bit state `M[16]` (INT32):
   ```
   M[ℓ mod 16] = rotate_left(M[ℓ mod 16], 13) XOR X
   ```
   (only full tiles where h=tm, w=tn, d=r update M)
3. Block opened when: `BLAKE3(M, key=sA) ≤ 2^(256−b) · r · tm · tn` (uint256 little-endian)

### Parameter Constraints
- `16r ≤ k ≤ 4r²`, `k ≤ 2^16`, `64 | k`
- r ∈ {32, 64, 128, 256, 512, 1024}
- `tm·tn ≥ 32`; `k·(tm+tn) ≤ 2^22`
- m, n ≤ 2^24
- Tile shapes are 3D arithmetic progressions; boundary tiles are ineligible for PoW

## Block Opening Proof

### Merkle-Based Proof (included in block certificate)
Contains: HA, HB, Merkle authentication paths for the row strips of A and column strips of B used in the tile, tile metadata (i, j, depth ℓ), matmul shape (m, n, k), rank r, tile shapes tm, tn.

Verifier steps:
1. Validate Merkle proofs for A-rows and B-columns against HA, HB
2. Derive sA, sB from HA, HB, σ, μ
3. Generate EL, ER, FL, FR from noise seeds
4. Reconstruct candidate tile Ctile from noised strips at (i, j)
5. Check difficulty: `BLAKE3(M, key=sA) ≤ 2^(256−b) · r · tm · tn`

### zkSNARK Proof (Plonky2 — production)
- Proves the statement: "∃ strips consistent with HA, HB s.t. their product yields Mi,j whose BLAKE3 equals h" without revealing strips
- **Public inputs:** r, k, tm, tn, m, n, i, j, HA, HB, `BLAKE3(M, key=sA)`
- **Private inputs:** actual strips of A and B
- **Optimizations:** direct AIR representation; preprocessed columns for noise derivation (2/3 of verifier time, agreed by both parties in plaintext); 3-layer recursion
- **Target:** proof size < 60 KB
- No trusted setup; post-quantum (hash-based only)

## Blockchain Specifics

### Block Structure
- Block identity: `dSHA256(version || prev_hash || tx_root || time || nBits || pouw_meta)` — 116 bytes
  - `pouw_meta = SHA256(zkSNARK public witness)` (32 bytes)
- Block certificate: `version || bytes`, variable-length, max **65 KB** (replaces Bitcoin nonce field)
- Multiple valid certificates can exist per block (zkSNARK is randomized); identity is determined by pouw_meta, not certificate bytes

### Addresses & Signatures
- Taproot-only (legacy P2PKH/P2SH/ECDSA/Schnorr removed)
- Post-quantum: `OP_CHECKXMSSSIG` (opcode #222) — SHAKE256, ~256-bit security, **32 signatures per key pair**, signature size **2340 bytes**
- XMSS key derivation path: `m/222'/<coin_type>'/account'/branch/index`
- P2MR (BIP-360): eliminates quantum-vulnerable key-path spend; all spends via script path + Merkle proof

### Script VM
- `OP_CAT` enabled: concatenates two stack items; output capped at **520 bytes**; dynamic cost ⌈len_out/64⌉ against tapscript budget

### Consensus & Timing
- Block time: **194 s** target (3:14 min)
- Difficulty: **WTEMA-N** (N=2016, T=194 s, τ=7 days):
  ```
  target_new = target_old + target_old · (t − T) / (N · T)
  ```
- Initial difficulty: `nBits = 0x1b00ffff`
- Timestamp rules: monotonic (≥ 1 s after parent); future-time limit **5 minutes** (vs Bitcoin's 120 min)
- Tie-breaking: prefer first-received tip when cumulative work differs by less than `min(work1, work2) / 4`

### Supply & Emission
- Total supply: **2.1B PEARL**; smallest unit: 10^−8 (grain)
- Smooth polynomial decay (no halving shocks): `E*(t) = S·H / ((t+H)·(t+H−1))`
  - H = 650,226 blocks (~4 years at 194 s/block)
  - R(H) = 1/2 (50% allocated at H blocks); R(t) = H/(t+H)

## Reference Hardware & Performance

Reference: LLaMA 3.3 70B (Pearl-certified) on 4×H200 GPUs

| Parallelism | MMLU Score | Throughput (tok/s) | Useful MADs (TMADs/s) |
|---|---|---|---|
| PP=4 | 0.8190 | 17,206 | **806** |
| TP=4 | 0.8180 | 13,264 | **620** |
| DP=4 | 0.8198 | 18,292 | **981** |

Useful-work miners pay ~10% overhead (τ ≈ 0.1) vs full compute cost (~$1.70/H100-hr) for non-useful miners.

## Planned Protocol Upgrade

Future upgrade (not yet scheduled) will extend PoUW to BF16/FP8/FP4 by using native quantization noise as the perturbation mechanism, enabling training workloads and state-of-the-art low-precision inference without adaptation.

## Official Resources
- Official repo: github.com/pearl-research-labs/pearl
- vLLM plugin: available at launch for Pearl-certified inference
