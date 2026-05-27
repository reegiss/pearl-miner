# Pearl Miner — Architecture Design

**Date:** 2026-05-27
**Status:** Approved
**Scope:** Workspace structure, crate boundaries, public interfaces, and data flow. zkSNARK (Plonky2) proof module is out of scope for this story.

---

## 1. Overview

Pearl Miner implements the Proof-of-Useful-Work (PoUW) protocol for the Pearl blockchain. Mining work is INT8 matrix multiplication — the same operation used in GPU-based AI inference — making GPU compute useful for both AI workloads and block production simultaneously.

**Language:** Rust  
**Build:** Single Cargo workspace; CUDA kernel compiled via `build.rs` in `pearl-gpu`

---

## 2. Workspace Structure

```
pearl-miner/
├── Cargo.toml              # workspace root
├── crates/
│   ├── pearl-types/        # shared types (no internal deps)
│   ├── pearl-commitment/   # CommitmentHash + Merkle tree
│   ├── pearl-noise/        # noise matrix generation EL·ER, FL·FR
│   ├── pearl-gpu/          # CUDA TiledMatMul kernel (build.rs compiles .cu)
│   ├── pearl-peel/         # clean product recovery A·B
│   └── pearl-block/        # Pearl block serialization and validation
└── src/
    └── main.rs             # pearl-miner binary, orchestrates all 6 crates
```

### Dependency Graph

```
pearl-types
    ↑
pearl-commitment   pearl-noise   pearl-block
         ↑               ↑
         └──── pearl-gpu ─────────────────┐
                    ↑                     │
              pearl-peel                  │
                    ↑                     ↓
              pearl-miner (binary) ←──────┘
```

`pearl-types` is the only crate all others depend on. No logic crate depends on another logic crate — only the `pearl-miner` binary imports all and orchestrates.

---

## 3. Shared Types (`pearl-types`)

No logic, only structs and type aliases. All crates import from here.

Matrix types (`Vec<i8>`, `Vec<i32>`) are **not** centralized here — each crate declares its own local representation to avoid imposing a global memory layout.

```rust
pub struct MatrixParams {
    pub m: u32, pub n: u32, pub k: u32,
    pub r: u32,             // noise rank ∈ {32,64,128,256,512,1024}
    pub tm: u32, pub tn: u32,
}

pub struct MiningConfig {
    pub params: MatrixParams,
    pub difficulty_bits: f64,   // fractional b
    pub sigma: Vec<u8>,         // blockchain state
    pub mu: Vec<u8>,            // miner config
}

pub struct Commitments {
    pub kappa: [u8; 32],
    pub ha: [u8; 32],
    pub hb: [u8; 32],
    pub s_a: [u8; 32],
    pub s_b: [u8; 32],
}

// Tile that satisfied the difficulty condition
pub struct FoundTile {
    pub tile_i: u32,
    pub tile_j: u32,
    pub m_state: [i32; 16],
    pub final_hash: [u8; 32],   // BLAKE3(M, key=sA)
}

pub struct BlockCertificate {
    pub commitments: Commitments,
    pub tile: FoundTile,
    pub merkle_proof_a: Vec<[u8; 32]>,
    pub merkle_proof_b: Vec<[u8; 32]>,
}
```

---

## 4. Public Interfaces

### `pearl-commitment`

```rust
// κ = BLAKE3(σ‖μ), HA = BLAKE3(Flatten(A), key=κ), HB = BLAKE3(Flatten(B^T), key=κ)
// sB = BLAKE3(κ‖HB), sA = BLAKE3(sB‖HA)
pub fn compute(
    a: &[i8],               // A row-major, shape m×k
    b: &[i8],               // B col-major (B^T), shape k×n
    config: &MiningConfig,
) -> Commitments;

pub struct MerkleTree { /* ... */ }
impl MerkleTree {
    pub fn from_rows(data: &[i8], row_len: usize) -> Self;
    pub fn root(&self) -> [u8; 32];
    pub fn proof(&self, idx: usize) -> Vec<[u8; 32]>;
}
```

### `pearl-noise`

```rust
// E = EL·ER: EL is m×r in [-32,31]; ER is r×k with exactly one +1 and one -1 per column
pub fn generate_e(m: u32, k: u32, r: u32, seed: &[u8; 32]) -> (Vec<i8>, Vec<i8>);

// F = FL·FR: FL follows ER^T distribution; FR follows EL^T distribution
pub fn generate_f(k: u32, n: u32, r: u32, seed: &[u8; 32]) -> (Vec<i8>, Vec<i8>);
```

PRNG: BLAKE3 with per-entry domain separation by index, as specified by the protocol.

### `pearl-gpu`

```rust
pub struct GpuMiner { /* device handle, CUDA streams, device memory buffers */ }

impl GpuMiner {
    pub fn new(device_id: u32) -> Result<Self>;

    // Runs TiledMatMul; returns tiles satisfying the difficulty condition
    // BLAKE3(M, key=sA) ≤ 2^(256−b) · r · tm · tn  (uint256 little-endian)
    pub fn mine(
        &self,
        a_noisy: &[i8],        // A' row-major, m×k
        b_noisy: &[i8],        // B' col-major, k×n
        commitments: &Commitments,
        config: &MiningConfig,
    ) -> Result<Vec<FoundTile>>;
}
```

The kernel accumulates in INT32, maintains `M[16]` state across depth steps, and checks the BLAKE3 condition per tile. Implemented in `.cu`, compiled via `build.rs` using `cudarc`.

### `pearl-peel`

```rust
// Recovers A·B from noisy product: A·B = A'·B' − (A·FL)·FR − EL·(ER·B')
// All corrections are O(n²·r) — asymptotically negligible vs O(n²·k) for the main product
pub fn recover(
    ab_noisy: &[i32],   // C' = A'·B', shape m×n
    a: &[i8],           // original A, shape m×k
    b_noisy: &[i8],     // B', shape k×n
    el: &[i8], er: &[i8],
    fl: &[i8], fr: &[i8],
    params: &MatrixParams,
) -> Vec<i32>;          // clean A·B, shape m×n
```

### `pearl-block`

```rust
pub struct PearlBlock {
    pub version: u32,
    pub prev_hash: [u8; 32],
    pub tx_root: [u8; 32],
    pub timestamp: u64,
    pub n_bits: u32,
    pub certificate: BlockCertificate,
    pub transactions: Vec<Vec<u8>>,
}

// dSHA256(version ‖ prev_hash ‖ tx_root ‖ time ‖ nBits ‖ pouw_meta)
// pouw_meta = SHA256(zkSNARK public witness) — stubbed until pearl-proof is added
pub fn block_identity(block: &PearlBlock) -> [u8; 32];

pub fn serialize(block: &PearlBlock) -> Vec<u8>;
pub fn deserialize(bytes: &[u8]) -> Result<PearlBlock>;

// Re-runs the BLAKE3 condition check against the certificate data
pub fn validate_certificate(cert: &BlockCertificate, config: &MiningConfig) -> bool;
```

---

## 5. Data Flow in `pearl-miner` Binary

```
AI workload (vLLM / PyTorch via FFI or socket)
    A: m×k (INT8)   B: k×n (INT8)
              │
              ▼
pearl_commitment::compute(A, B, config)
→ Commitments { κ, HA, HB, sA, sB }
→ MerkleTree(A rows), MerkleTree(B cols)
              │
    ┌─────────┴──────────┐
    ▼                    ▼
generate_e(sA)      generate_f(sB)
→ EL, ER             → FL, FR
    └─────────┬──────────┘
              │  A' = A + EL·ER
              │  B' = B + FL·FR
              ▼
GpuMiner::mine(A', B', commitments, config)
→ Vec<FoundTile>
              │
    ┌─────────┴──────────────────────────────┐
    │ tile found                             │ none → discard, await next (A, B)
    ▼                                        │
pearl_peel::recover(...)  ←──────────────────┘
→ clean A·B (delivered to AI workload)
    │
    ▼
pearl_block: assemble PearlBlock
           + BlockCertificate (Merkle proofs for winning tile)
           + block_identity()
    │
    ▼
P2P: submit block to Pearl network
```

### Pipeline Properties

- Commitment + noise generation are CPU-bound; run on a dedicated thread, not the GPU stream
- `GpuMiner` operates on a dedicated CUDA stream, separate from the inference stream (overhead target: < 5%)
- Even when no tile is found, `A'·B'` has been computed; the clean result is recovered and delivered to the AI workload
- **Multi-GPU:** one `GpuMiner` instance per device; the binary spawns one Tokio task per GPU, all fed from a shared `(A, B)` queue

---

## 6. Out of Scope

- **`pearl-proof` (Plonky2 zkSNARK):** deferred to a future story. `block_identity` stubs `pouw_meta` until then.
- P2P networking protocol
- Wallet / transaction construction
- Difficulty adjustment (WTEMA-N)
