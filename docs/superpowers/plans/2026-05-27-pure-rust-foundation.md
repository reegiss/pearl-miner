# Pearl Miner — Pure Rust Foundation Implementation Plan (Part 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up the Cargo workspace and implement the five pure-Rust crates that form the cryptographic and mathematical foundation of the Pearl PoUW miner protocol.

**Architecture:** Single Cargo workspace, one crate per protocol module. Each crate has a clean public API matching the design spec in `docs/superpowers/specs/2026-05-27-architecture-design.md`. `pearl-gpu` and the miner binary are Part 2.

**Tech Stack:** Rust 2021 edition · `blake3 1.x` · `sha2 0.10` · `num-bigint 0.4` · `num-traits 0.2`

---

## File Map

| File | Responsibility |
|---|---|
| `Cargo.toml` | Workspace root declaring all 5 crates |
| `crates/pearl-types/Cargo.toml` | No external deps |
| `crates/pearl-types/src/lib.rs` | `MatrixParams`, `MiningConfig`, `Commitments`, `FoundTile`, `BlockCertificate` |
| `crates/pearl-commitment/Cargo.toml` | Deps: `pearl-types`, `blake3` |
| `crates/pearl-commitment/src/lib.rs` | Re-exports from `commitment` and `merkle` modules |
| `crates/pearl-commitment/src/commitment.rs` | `compute()` — BLAKE3 commitment hash chain |
| `crates/pearl-commitment/src/merkle.rs` | `MerkleTree` — build, root, proof, verify |
| `crates/pearl-noise/Cargo.toml` | Deps: `pearl-types`, `blake3` |
| `crates/pearl-noise/src/lib.rs` | `generate_e()`, `generate_f()` — BLAKE3-seeded noise matrices |
| `crates/pearl-peel/Cargo.toml` | Deps: `pearl-types` · dev: `pearl-noise` |
| `crates/pearl-peel/src/lib.rs` | `recover()` — clean A·B from noisy MatMul |
| `crates/pearl-block/Cargo.toml` | Deps: `pearl-types`, `blake3`, `sha2`, `num-bigint`, `num-traits` |
| `crates/pearl-block/src/lib.rs` | Re-exports from `serial` and `validate` modules |
| `crates/pearl-block/src/serial.rs` | `PearlBlock`, `serialize()`, `deserialize()` |
| `crates/pearl-block/src/validate.rs` | `validate_certificate()`, `block_identity()` |

---

### Task 1: Workspace and Crate Scaffolding

**Files:**
- Create: `Cargo.toml`
- Create: `crates/pearl-types/Cargo.toml`
- Create: `crates/pearl-types/src/lib.rs`
- Create: `crates/pearl-commitment/Cargo.toml`
- Create: `crates/pearl-commitment/src/lib.rs`
- Create: `crates/pearl-noise/Cargo.toml`
- Create: `crates/pearl-noise/src/lib.rs`
- Create: `crates/pearl-peel/Cargo.toml`
- Create: `crates/pearl-peel/src/lib.rs`
- Create: `crates/pearl-block/Cargo.toml`
- Create: `crates/pearl-block/src/lib.rs`

- [ ] **Step 1: Create workspace Cargo.toml**

```toml
# Cargo.toml
[workspace]
members = [
    "crates/pearl-types",
    "crates/pearl-commitment",
    "crates/pearl-noise",
    "crates/pearl-peel",
    "crates/pearl-block",
]
resolver = "2"
```

- [ ] **Step 2: Create pearl-types crate**

```toml
# crates/pearl-types/Cargo.toml
[package]
name = "pearl-types"
version = "0.1.0"
edition = "2021"
```

```rust
// crates/pearl-types/src/lib.rs
// (empty for now — filled in Task 2)
```

- [ ] **Step 3: Create pearl-commitment crate**

```toml
# crates/pearl-commitment/Cargo.toml
[package]
name = "pearl-commitment"
version = "0.1.0"
edition = "2021"

[dependencies]
pearl-types = { path = "../pearl-types" }
blake3 = "1"
```

```rust
// crates/pearl-commitment/src/lib.rs
// (empty for now)
```

- [ ] **Step 4: Create pearl-noise crate**

```toml
# crates/pearl-noise/Cargo.toml
[package]
name = "pearl-noise"
version = "0.1.0"
edition = "2021"

[dependencies]
pearl-types = { path = "../pearl-types" }
blake3 = "1"
```

```rust
// crates/pearl-noise/src/lib.rs
// (empty for now)
```

- [ ] **Step 5: Create pearl-peel crate**

```toml
# crates/pearl-peel/Cargo.toml
[package]
name = "pearl-peel"
version = "0.1.0"
edition = "2021"

[dependencies]
pearl-types = { path = "../pearl-types" }

[dev-dependencies]
pearl-noise = { path = "../pearl-noise" }
```

```rust
// crates/pearl-peel/src/lib.rs
// (empty for now)
```

- [ ] **Step 6: Create pearl-block crate**

```toml
# crates/pearl-block/Cargo.toml
[package]
name = "pearl-block"
version = "0.1.0"
edition = "2021"

[dependencies]
pearl-types = { path = "../pearl-types" }
blake3 = "1"
sha2 = "0.10"
num-bigint = "0.4"
num-traits = "0.2"
```

```rust
// crates/pearl-block/src/lib.rs
// (empty for now)
```

- [ ] **Step 7: Verify workspace builds**

Run: `cargo check`

Expected: all 5 crates compile with 0 errors (0 warnings for empty libs is fine).

- [ ] **Step 8: Commit**

```bash
git add Cargo.toml Cargo.lock crates/
git commit -m "feat: initialize cargo workspace with 5 pure-rust crate stubs"
```

---

### Task 2: `pearl-types` — Shared Structs

**Files:**
- Modify: `crates/pearl-types/src/lib.rs`

- [ ] **Step 1: Write the types**

```rust
// crates/pearl-types/src/lib.rs

#[derive(Debug, Clone, PartialEq)]
pub struct MatrixParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub r: u32,   // noise rank ∈ {32, 64, 128, 256, 512, 1024}
    pub tm: u32,
    pub tn: u32,
}

#[derive(Debug, Clone)]
pub struct MiningConfig {
    pub params: MatrixParams,
    pub difficulty_bits: f64,  // fractional b
    pub sigma: Vec<u8>,        // blockchain state
    pub mu: Vec<u8>,           // miner config bytes
}

#[derive(Debug, Clone, PartialEq)]
pub struct Commitments {
    pub kappa: [u8; 32],
    pub ha: [u8; 32],
    pub hb: [u8; 32],
    pub s_a: [u8; 32],
    pub s_b: [u8; 32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct FoundTile {
    pub tile_i: u32,
    pub tile_j: u32,
    pub m_state: [i32; 16],
    pub final_hash: [u8; 32],  // BLAKE3(M, key=sA)
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockCertificate {
    pub commitments: Commitments,
    pub tile: FoundTile,
    pub merkle_proof_a: Vec<[u8; 32]>,
    pub merkle_proof_b: Vec<[u8; 32]>,
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p pearl-types`

Expected: Compiling pearl-types v0.1.0 — 0 errors.

- [ ] **Step 3: Commit**

```bash
git add crates/pearl-types/src/lib.rs
git commit -m "feat(pearl-types): add shared protocol structs"
```

---

### Task 3: `pearl-commitment` — CommitmentHash

**Files:**
- Create: `crates/pearl-commitment/src/commitment.rs`
- Modify: `crates/pearl-commitment/src/lib.rs`

The hash chain: `κ = BLAKE3(σ‖μ)` → `HA = BLAKE3(A, key=κ)` → `HB = BLAKE3(B^T, key=κ)` → `sB = BLAKE3(κ‖HB)` → `sA = BLAKE3(sB‖HA)`.

- [ ] **Step 1: Write the failing test**

```rust
// crates/pearl-commitment/src/commitment.rs  (bottom of file, after implementation placeholder)
#[cfg(test)]
mod tests {
    use super::*;
    use pearl_types::{MatrixParams, MiningConfig};

    fn cfg() -> MiningConfig {
        MiningConfig {
            params: MatrixParams { m: 4, n: 4, k: 64, r: 32, tm: 2, tn: 2 },
            difficulty_bits: 10.0,
            sigma: b"chain-state".to_vec(),
            mu: b"miner-cfg".to_vec(),
        }
    }

    #[test]
    fn test_compute_is_deterministic() {
        let a = vec![1i8; 4 * 64];
        let b = vec![2i8; 64 * 4];
        assert_eq!(compute(&a, &b, &cfg()), compute(&a, &b, &cfg()));
    }

    #[test]
    fn test_ha_depends_on_a_not_b() {
        let b = vec![3i8; 64 * 4];
        let c1 = compute(&vec![1i8; 4 * 64], &b, &cfg());
        let c2 = compute(&vec![2i8; 4 * 64], &b, &cfg());
        assert_ne!(c1.ha, c2.ha);
        assert_eq!(c1.hb, c2.hb);
    }

    #[test]
    fn test_hb_depends_on_b_not_a() {
        let a = vec![1i8; 4 * 64];
        let c1 = compute(&a, &vec![2i8; 64 * 4], &cfg());
        let c2 = compute(&a, &vec![3i8; 64 * 4], &cfg());
        assert_ne!(c1.hb, c2.hb);
        assert_eq!(c1.ha, c2.ha);
    }

    #[test]
    fn test_sa_changes_when_ha_changes() {
        let b = vec![0i8; 64 * 4];
        let c1 = compute(&vec![1i8; 4 * 64], &b, &cfg());
        let c2 = compute(&vec![2i8; 4 * 64], &b, &cfg());
        assert_ne!(c1.s_a, c2.s_a);
    }

    #[test]
    fn test_sa_changes_when_hb_changes() {
        let a = vec![0i8; 4 * 64];
        let c1 = compute(&a, &vec![1i8; 64 * 4], &cfg());
        let c2 = compute(&a, &vec![2i8; 64 * 4], &cfg());
        assert_ne!(c1.s_a, c2.s_a);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p pearl-commitment 2>&1 | head -20`

Expected: error[E0425]: cannot find function `compute` in this scope

- [ ] **Step 3: Implement `compute()`**

```rust
// crates/pearl-commitment/src/commitment.rs
use pearl_types::{Commitments, MiningConfig};

pub fn compute(a: &[i8], b: &[i8], config: &MiningConfig) -> Commitments {
    // κ = BLAKE3(σ || μ)
    let mut hasher = blake3::Hasher::new();
    hasher.update(&config.sigma);
    hasher.update(&config.mu);
    let kappa: [u8; 32] = *hasher.finalize().as_bytes();

    // HA = BLAKE3(Flatten(A), key=κ)  — A is already row-major
    let a_bytes: Vec<u8> = a.iter().map(|&x| x as u8).collect();
    let ha: [u8; 32] = *blake3::keyed_hash(&kappa, &a_bytes).as_bytes();

    // HB = BLAKE3(Flatten(B^T), key=κ) — B is passed col-major = B^T row-major
    let b_bytes: Vec<u8> = b.iter().map(|&x| x as u8).collect();
    let hb: [u8; 32] = *blake3::keyed_hash(&kappa, &b_bytes).as_bytes();

    // sB = BLAKE3(κ || HB)
    let mut sb_input = [0u8; 64];
    sb_input[..32].copy_from_slice(&kappa);
    sb_input[32..].copy_from_slice(&hb);
    let s_b: [u8; 32] = *blake3::hash(&sb_input).as_bytes();

    // sA = BLAKE3(sB || HA)
    let mut sa_input = [0u8; 64];
    sa_input[..32].copy_from_slice(&s_b);
    sa_input[32..].copy_from_slice(&ha);
    let s_a: [u8; 32] = *blake3::hash(&sa_input).as_bytes();

    Commitments { kappa, ha, hb, s_a, s_b }
}
```

- [ ] **Step 4: Wire up `lib.rs`**

```rust
// crates/pearl-commitment/src/lib.rs
mod commitment;
pub use commitment::compute;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p pearl-commitment`

Expected:
```
test commitment::tests::test_compute_is_deterministic ... ok
test commitment::tests::test_ha_depends_on_a_not_b ... ok
test commitment::tests::test_hb_depends_on_b_not_a ... ok
test commitment::tests::test_sa_changes_when_ha_changes ... ok
test commitment::tests::test_sa_changes_when_hb_changes ... ok
```

- [ ] **Step 6: Commit**

```bash
git add crates/pearl-commitment/src/
git commit -m "feat(pearl-commitment): implement commitment hash chain"
```

---

### Task 4: `pearl-commitment` — MerkleTree

**Files:**
- Create: `crates/pearl-commitment/src/merkle.rs`
- Modify: `crates/pearl-commitment/src/lib.rs`

1-indexed binary tree stored as a flat array. Leaves at `[padded_size..2*padded_size]`, where `padded_size` is the next power of 2 ≥ row count.

- [ ] **Step 1: Write the failing tests**

```rust
// crates/pearl-commitment/src/merkle.rs  (add at bottom, after struct placeholder)
#[cfg(test)]
mod tests {
    use super::*;

    fn row_bytes(row: &[i8]) -> Vec<u8> {
        row.iter().map(|&x| x as u8).collect()
    }

    #[test]
    fn test_root_is_deterministic() {
        let data = vec![1i8; 4 * 8]; // 4 rows × 8 cols
        let t1 = MerkleTree::from_rows(&data, 8);
        let t2 = MerkleTree::from_rows(&data, 8);
        assert_eq!(t1.root(), t2.root());
    }

    #[test]
    fn test_root_changes_with_data() {
        let t1 = MerkleTree::from_rows(&vec![1i8; 4 * 8], 8);
        let t2 = MerkleTree::from_rows(&vec![2i8; 4 * 8], 8);
        assert_ne!(t1.root(), t2.root());
    }

    #[test]
    fn test_proof_verifies_row0() {
        let data: Vec<i8> = (0..16).map(|i| i as i8).collect(); // 2 rows × 8
        let tree = MerkleTree::from_rows(&data, 8);
        let proof = tree.proof(0);
        let leaf = row_bytes(&data[0..8]);
        assert!(tree.verify_proof(0, &leaf, &proof));
    }

    #[test]
    fn test_proof_verifies_row1() {
        let data: Vec<i8> = (0..16).map(|i| i as i8).collect();
        let tree = MerkleTree::from_rows(&data, 8);
        let proof = tree.proof(1);
        let leaf = row_bytes(&data[8..16]);
        assert!(tree.verify_proof(1, &leaf, &proof));
    }

    #[test]
    fn test_wrong_leaf_fails_verification() {
        let data: Vec<i8> = (0..16).map(|i| i as i8).collect();
        let tree = MerkleTree::from_rows(&data, 8);
        let proof = tree.proof(0);
        let wrong_leaf = vec![0xffu8; 8];
        assert!(!tree.verify_proof(0, &wrong_leaf, &proof));
    }

    #[test]
    fn test_non_power_of_two_row_count() {
        // 3 rows — padded to 4
        let data: Vec<i8> = (0..24).map(|i| i as i8).collect();
        let tree = MerkleTree::from_rows(&data, 8);
        // Proofs for all 3 real rows must verify
        for idx in 0..3 {
            let proof = tree.proof(idx);
            let leaf = row_bytes(&data[idx * 8..(idx + 1) * 8]);
            assert!(tree.verify_proof(idx, &leaf, &proof), "row {} failed", idx);
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p pearl-commitment merkle 2>&1 | head -10`

Expected: error — `MerkleTree` not found.

- [ ] **Step 3: Implement `MerkleTree`**

```rust
// crates/pearl-commitment/src/merkle.rs
use blake3;

pub struct MerkleTree {
    nodes: Vec<[u8; 32]>,  // 1-indexed; nodes[1] = root
    padded_size: usize,
}

fn hash_leaf(data: &[u8]) -> [u8; 32] {
    *blake3::hash(data).as_bytes()
}

fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut input = [0u8; 64];
    input[..32].copy_from_slice(left);
    input[32..].copy_from_slice(right);
    *blake3::hash(&input).as_bytes()
}

impl MerkleTree {
    pub fn from_rows(data: &[i8], row_len: usize) -> Self {
        let row_count = if row_len == 0 { 0 } else { data.len() / row_len };
        let padded_size = row_count.next_power_of_two().max(1);
        let total = 2 * padded_size;
        let mut nodes = vec![[0u8; 32]; total];

        // Hash real leaves
        for i in 0..row_count {
            let bytes: Vec<u8> = data[i * row_len..(i + 1) * row_len]
                .iter().map(|&x| x as u8).collect();
            nodes[padded_size + i] = hash_leaf(&bytes);
        }
        // Padding leaves stay as [0u8; 32]

        // Build internal nodes bottom-up (skip index 0)
        for i in (1..padded_size).rev() {
            nodes[i] = hash_pair(&nodes[2 * i], &nodes[2 * i + 1]);
        }

        MerkleTree { nodes, padded_size }
    }

    pub fn root(&self) -> [u8; 32] {
        self.nodes[1]
    }

    /// Returns sibling hashes from leaf to root (leaf-first).
    pub fn proof(&self, idx: usize) -> Vec<[u8; 32]> {
        let mut path = Vec::new();
        let mut i = self.padded_size + idx;
        while i > 1 {
            path.push(self.nodes[i ^ 1]);
            i /= 2;
        }
        path
    }

    /// Reconstructs the root from a leaf and its proof and checks it matches.
    pub fn verify_proof(&self, idx: usize, leaf_bytes: &[u8], proof: &[[u8; 32]]) -> bool {
        let mut current = hash_leaf(leaf_bytes);
        let mut i = self.padded_size + idx;
        for sibling in proof {
            let (left, right) = if i % 2 == 0 {
                (&current, sibling)
            } else {
                (sibling, &current)
            };
            current = hash_pair(left, right);
            i /= 2;
        }
        current == self.root()
    }
}
```

- [ ] **Step 4: Export from `lib.rs`**

```rust
// crates/pearl-commitment/src/lib.rs
mod commitment;
mod merkle;

pub use commitment::compute;
pub use merkle::MerkleTree;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p pearl-commitment`

Expected: 9 tests, all `ok`.

- [ ] **Step 6: Commit**

```bash
git add crates/pearl-commitment/src/merkle.rs crates/pearl-commitment/src/lib.rs
git commit -m "feat(pearl-commitment): add blake3 merkle tree with proof generation"
```

---

### Task 5: `pearl-noise` — `generate_e` (EL and ER)

**Files:**
- Modify: `crates/pearl-noise/src/lib.rs`

Protocol spec: EL is m×r with entries in `[-32, 31]`; ER is r×k where each column has exactly one `+1` and one `-1` at distinct random positions. BLAKE3 is used as PRNG with per-index domain separation (tag byte distinguishes EL entries from ER column picks).

- [ ] **Step 1: Write the failing tests**

```rust
// Add at the bottom of crates/pearl-noise/src/lib.rs
#[cfg(test)]
mod tests {
    use super::*;

    const SEED: [u8; 32] = [42u8; 32];

    #[test]
    fn test_el_shape_and_range() {
        let (el, _) = generate_e(4, 64, 32, &SEED);
        assert_eq!(el.len(), 4 * 32);
        for &v in &el {
            assert!(v >= -32 && v <= 31, "EL entry {} out of [-32,31]", v);
        }
    }

    #[test]
    fn test_er_shape_and_column_structure() {
        let (r, k) = (32u32, 64u32);
        let (_, er) = generate_e(4, k, r, &SEED);
        assert_eq!(er.len(), (r * k) as usize);

        for col in 0..k as usize {
            let col_vals: Vec<i8> = (0..r as usize)
                .map(|row| er[row * k as usize + col])
                .collect();
            let ones   = col_vals.iter().filter(|&&v| v ==  1).count();
            let neg_ones = col_vals.iter().filter(|&&v| v == -1).count();
            let zeros  = col_vals.iter().filter(|&&v| v ==  0).count();
            assert_eq!(ones, 1,     "col {col}: expected 1 +1, got {ones}");
            assert_eq!(neg_ones, 1, "col {col}: expected 1 -1, got {neg_ones}");
            assert_eq!(zeros, (r - 2) as usize,
                "col {col}: expected {} zeros, got {zeros}", r - 2);
        }
    }

    #[test]
    fn test_generate_e_is_deterministic() {
        let (el1, er1) = generate_e(4, 64, 32, &SEED);
        let (el2, er2) = generate_e(4, 64, 32, &SEED);
        assert_eq!(el1, el2);
        assert_eq!(er1, er2);
    }

    #[test]
    fn test_different_seeds_give_different_el() {
        let (el1, _) = generate_e(4, 64, 32, &[1u8; 32]);
        let (el2, _) = generate_e(4, 64, 32, &[2u8; 32]);
        assert_ne!(el1, el2);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p pearl-noise test_el_shape 2>&1 | head -10`

Expected: error — `generate_e` not found.

- [ ] **Step 3: Implement `generate_e`**

```rust
// crates/pearl-noise/src/lib.rs

/// BLAKE3-based PRNG: hash(seed, tag‖idx) → 32 bytes.
/// `tag` provides domain separation between EL, ER, FL, FR.
fn prng(seed: &[u8; 32], tag: u8, idx: u64) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_keyed(seed);
    hasher.update(&[tag]);
    hasher.update(&idx.to_le_bytes());
    *hasher.finalize().as_bytes()
}

/// EL: m×r row-major, entries uniform in [-32, 31].
fn gen_el(m: u32, r: u32, seed: &[u8; 32]) -> Vec<i8> {
    let size = (m * r) as usize;
    let mut el = Vec::with_capacity(size);
    for idx in 0..size as u64 {
        let h = prng(seed, 0, idx);
        let val = u32::from_le_bytes([h[0], h[1], h[2], h[3]]);
        el.push((val % 64) as i8 - 32);
    }
    el
}

/// ER: r×k row-major. Each column j has exactly one +1 and one -1 at distinct rows.
fn gen_er(r: u32, k: u32, seed: &[u8; 32]) -> Vec<i8> {
    let mut er = vec![0i8; (r * k) as usize];
    for col in 0..k as u64 {
        let h = prng(seed, 1, col);
        let p1 = u32::from_le_bytes([h[0], h[1], h[2], h[3]]) as usize % r as usize;
        let mut p2 = u32::from_le_bytes([h[4], h[5], h[6], h[7]]) as usize % r as usize;
        if p2 == p1 {
            p2 = (p1 + 1) % r as usize;
        }
        er[p1 * k as usize + col as usize] =  1;
        er[p2 * k as usize + col as usize] = -1;
    }
    er
}

/// Returns (EL: m×r, ER: r×k).
/// E = EL·ER has entries in [-63, 63] and fits in i8 without overflow.
pub fn generate_e(m: u32, k: u32, r: u32, seed: &[u8; 32]) -> (Vec<i8>, Vec<i8>) {
    (gen_el(m, r, seed), gen_er(r, k, seed))
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p pearl-noise`

Expected: 4 tests all `ok`.

- [ ] **Step 5: Commit**

```bash
git add crates/pearl-noise/src/lib.rs
git commit -m "feat(pearl-noise): implement generate_e with blake3 prng"
```

---

### Task 6: `pearl-noise` — `generate_f` (FL and FR)

**Files:**
- Modify: `crates/pearl-noise/src/lib.rs`

FL follows the ER^T distribution (k×r, each **row** has one `+1` and one `-1`). FR follows the EL^T distribution (r×n, entries in `[-32, 31]`). F = FL·FR has shape k×n and is added to B.

- [ ] **Step 1: Write the failing tests**

```rust
// Add to the tests module at the bottom of crates/pearl-noise/src/lib.rs

    #[test]
    fn test_fl_shape_and_row_structure() {
        let (k, n, r) = (64u32, 4u32, 32u32);
        let (fl, _) = generate_f(k, n, r, &SEED);
        assert_eq!(fl.len(), (k * r) as usize);

        for row in 0..k as usize {
            let row_vals: Vec<i8> = (0..r as usize)
                .map(|col| fl[row * r as usize + col])
                .collect();
            let ones     = row_vals.iter().filter(|&&v| v ==  1).count();
            let neg_ones = row_vals.iter().filter(|&&v| v == -1).count();
            assert_eq!(ones, 1,     "row {row}: expected 1 +1");
            assert_eq!(neg_ones, 1, "row {row}: expected 1 -1");
        }
    }

    #[test]
    fn test_fr_shape_and_range() {
        let (k, n, r) = (64u32, 4u32, 32u32);
        let (_, fr) = generate_f(k, n, r, &SEED);
        assert_eq!(fr.len(), (r * n) as usize);
        for &v in &fr {
            assert!(v >= -32 && v <= 31, "FR entry {} out of [-32,31]", v);
        }
    }

    #[test]
    fn test_generate_f_is_deterministic() {
        let (fl1, fr1) = generate_f(64, 4, 32, &SEED);
        let (fl2, fr2) = generate_f(64, 4, 32, &SEED);
        assert_eq!(fl1, fl2);
        assert_eq!(fr1, fr2);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p pearl-noise test_fl 2>&1 | head -10`

Expected: error — `generate_f` not found.

- [ ] **Step 3: Implement `generate_f`**

```rust
// Add to crates/pearl-noise/src/lib.rs (after generate_e)

/// FL: k×r row-major. Each row i has exactly one +1 and one -1 (follows ER^T distribution).
fn gen_fl(k: u32, r: u32, seed: &[u8; 32]) -> Vec<i8> {
    let mut fl = vec![0i8; (k * r) as usize];
    for row in 0..k as u64 {
        let h = prng(seed, 2, row);
        let p1 = u32::from_le_bytes([h[0], h[1], h[2], h[3]]) as usize % r as usize;
        let mut p2 = u32::from_le_bytes([h[4], h[5], h[6], h[7]]) as usize % r as usize;
        if p2 == p1 {
            p2 = (p1 + 1) % r as usize;
        }
        fl[row as usize * r as usize + p1] =  1;
        fl[row as usize * r as usize + p2] = -1;
    }
    fl
}

/// FR: r×n row-major, entries uniform in [-32, 31] (follows EL^T distribution).
fn gen_fr(r: u32, n: u32, seed: &[u8; 32]) -> Vec<i8> {
    let size = (r * n) as usize;
    let mut fr = Vec::with_capacity(size);
    for idx in 0..size as u64 {
        let h = prng(seed, 3, idx);
        let val = u32::from_le_bytes([h[0], h[1], h[2], h[3]]);
        fr.push((val % 64) as i8 - 32);
    }
    fr
}

/// Returns (FL: k×r, FR: r×n).
/// F = FL·FR has entries in [-63, 63] and fits in i8 without overflow.
pub fn generate_f(k: u32, n: u32, r: u32, seed: &[u8; 32]) -> (Vec<i8>, Vec<i8>) {
    (gen_fl(k, r, seed), gen_fr(r, n, seed))
}
```

- [ ] **Step 4: Run all noise tests to verify they pass**

Run: `cargo test -p pearl-noise`

Expected: 7 tests all `ok`.

- [ ] **Step 5: Commit**

```bash
git add crates/pearl-noise/src/lib.rs
git commit -m "feat(pearl-noise): implement generate_f for F=FL·FR noise"
```

---

### Task 7: `pearl-peel` — Clean Product Recovery

**Files:**
- Modify: `crates/pearl-peel/src/lib.rs`

Formula: `A·B = A'·B' − (A·FL)·FR − EL·(ER·B')`. All intermediate products use i32 arithmetic to avoid overflow.

- [ ] **Step 1: Write the failing test**

```rust
// crates/pearl-peel/src/lib.rs (add at bottom)
#[cfg(test)]
mod tests {
    use super::*;
    use pearl_noise::{generate_e, generate_f};
    use pearl_types::MatrixParams;

    fn naive_matmul(a: &[i32], b: &[i32], m: usize, k: usize, n: usize) -> Vec<i32> {
        let mut c = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    c[i * n + j] += a[i * k + l] * b[l * n + j];
                }
            }
        }
        c
    }

    fn as_i32(v: &[i8]) -> Vec<i32> {
        v.iter().map(|&x| x as i32).collect()
    }

    #[test]
    fn test_recover_matches_direct_matmul() {
        let params = MatrixParams { m: 4, n: 4, k: 64, r: 32, tm: 2, tn: 2 };
        let (m, n, k, r) = (4usize, 4usize, 64usize, 32usize);

        let a: Vec<i8> = (0..m * k).map(|i| (i % 64) as i8 - 32).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (i % 32) as i8 - 16).collect();

        let (el, er) = generate_e(params.m, params.k, params.r, &[1u8; 32]);
        let (fl, fr) = generate_f(params.k, params.n, params.r, &[2u8; 32]);

        // E = EL·ER (m×k): each entry = EL[i,p1] - EL[i,p2] ∈ [-63,63], fits i8
        let e = naive_matmul(&as_i32(&el), &as_i32(&er), m, r, k);
        let f = naive_matmul(&as_i32(&fl), &as_i32(&fr), k, r, n);

        let a_noisy: Vec<i8> = a.iter().zip(e.iter()).map(|(&a, &e)| a + e as i8).collect();
        let b_noisy: Vec<i8> = b.iter().zip(f.iter()).map(|(&b, &f)| b + f as i8).collect();

        let ab_noisy = naive_matmul(&as_i32(&a_noisy), &as_i32(&b_noisy), m, k, n);

        let recovered = recover(&ab_noisy, &a, &b_noisy, &el, &er, &fl, &fr, &params);
        let direct    = naive_matmul(&as_i32(&a), &as_i32(&b), m, k, n);

        assert_eq!(recovered, direct, "recovered product does not match direct A·B");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p pearl-peel 2>&1 | head -10`

Expected: error — `recover` not found.

- [ ] **Step 3: Implement `recover()`**

```rust
// crates/pearl-peel/src/lib.rs
use pearl_types::MatrixParams;

fn matmul_i32(a: &[i32], b: &[i32], m: usize, k: usize, n: usize) -> Vec<i32> {
    let mut c = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            for l in 0..k {
                c[i * n + j] += a[i * k + l] * b[l * n + j];
            }
        }
    }
    c
}

fn cast_i32(v: &[i8]) -> Vec<i32> {
    v.iter().map(|&x| x as i32).collect()
}

/// Recovers the clean matrix product A·B from the noisy computation.
///
/// Inputs (all row-major flat arrays):
///   ab_noisy : C' = A'·B', shape m×n (i32)
///   a        : original A, shape m×k (i8)
///   b_noisy  : B', shape k×n (i8)
///   el, er   : factors of E = EL·ER; EL m×r, ER r×k (i8)
///   fl, fr   : factors of F = FL·FR; FL k×r, FR r×n (i8)
///
/// Formula: A·B = A'·B' − (A·FL)·FR − EL·(ER·B')
pub fn recover(
    ab_noisy: &[i32],
    a: &[i8],
    b_noisy: &[i8],
    el: &[i8], er: &[i8],
    fl: &[i8], fr: &[i8],
    params: &MatrixParams,
) -> Vec<i32> {
    let m = params.m as usize;
    let n = params.n as usize;
    let k = params.k as usize;
    let r = params.r as usize;

    let a_i32  = cast_i32(a);
    let fl_i32 = cast_i32(fl);
    let fr_i32 = cast_i32(fr);
    let el_i32 = cast_i32(el);
    let er_i32 = cast_i32(er);
    let bn_i32 = cast_i32(b_noisy);

    // term1 = (A · FL) · FR   [m×k · k×r = m×r; m×r · r×n = m×n]
    let a_fl  = matmul_i32(&a_i32,  &fl_i32, m, k, r);
    let term1 = matmul_i32(&a_fl,   &fr_i32, m, r, n);

    // term2 = EL · (ER · B')  [r×k · k×n = r×n; m×r · r×n = m×n]
    let er_b  = matmul_i32(&er_i32, &bn_i32, r, k, n);
    let term2 = matmul_i32(&el_i32, &er_b,   m, r, n);

    ab_noisy.iter()
        .zip(term1.iter())
        .zip(term2.iter())
        .map(|((&c, &t1), &t2)| c - t1 - t2)
        .collect()
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p pearl-peel`

Expected: `test tests::test_recover_matches_direct_matmul ... ok`

- [ ] **Step 5: Commit**

```bash
git add crates/pearl-peel/src/lib.rs
git commit -m "feat(pearl-peel): implement noise correction for clean A·B recovery"
```

---

### Task 8: `pearl-block` — Block Struct and Serialization

**Files:**
- Create: `crates/pearl-block/src/serial.rs`
- Modify: `crates/pearl-block/src/lib.rs`

All integers are little-endian. Variable-length fields (Merkle proofs, transactions) are prefixed with a `u32` length.

- [ ] **Step 1: Write the failing tests**

```rust
// crates/pearl-block/src/serial.rs (add at bottom)
#[cfg(test)]
mod tests {
    use super::*;
    use pearl_types::{BlockCertificate, Commitments, FoundTile};

    fn test_cert() -> BlockCertificate {
        BlockCertificate {
            commitments: Commitments {
                kappa: [0u8; 32], ha: [1u8; 32], hb: [2u8; 32],
                s_a: [3u8; 32],   s_b: [4u8; 32],
            },
            tile: FoundTile {
                tile_i: 7, tile_j: 13,
                m_state: [42i32; 16],
                final_hash: [5u8; 32],
            },
            merkle_proof_a: vec![[6u8; 32], [7u8; 32]],
            merkle_proof_b: vec![[8u8; 32]],
        }
    }

    #[test]
    fn test_roundtrip_empty_transactions() {
        let block = PearlBlock {
            version: 1,
            prev_hash: [0xABu8; 32],
            tx_root: [0xCDu8; 32],
            timestamp: 1_700_000_000u64,
            n_bits: 0x1b00ffff,
            certificate: test_cert(),
            transactions: vec![],
        };
        let bytes = serialize(&block);
        let back = deserialize(&bytes).expect("deserialize failed");
        assert_eq!(back.version, block.version);
        assert_eq!(back.prev_hash, block.prev_hash);
        assert_eq!(back.tx_root, block.tx_root);
        assert_eq!(back.timestamp, block.timestamp);
        assert_eq!(back.n_bits, block.n_bits);
        assert_eq!(back.certificate, block.certificate);
        assert_eq!(back.transactions, block.transactions);
    }

    #[test]
    fn test_roundtrip_with_transactions() {
        let block = PearlBlock {
            version: 2,
            prev_hash: [1u8; 32],
            tx_root: [2u8; 32],
            timestamp: 9999,
            n_bits: 0xdeadbeef,
            certificate: test_cert(),
            transactions: vec![
                vec![0x01, 0x02, 0x03],
                vec![0xff; 100],
            ],
        };
        let bytes = serialize(&block);
        let back = deserialize(&bytes).unwrap();
        assert_eq!(back.transactions, block.transactions);
    }

    #[test]
    fn test_deserialize_truncated_returns_error() {
        let block = PearlBlock {
            version: 1,
            prev_hash: [0u8; 32],
            tx_root: [0u8; 32],
            timestamp: 0,
            n_bits: 0,
            certificate: test_cert(),
            transactions: vec![],
        };
        let mut bytes = serialize(&block);
        bytes.truncate(bytes.len() / 2);
        assert!(deserialize(&bytes).is_err());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p pearl-block serial 2>&1 | head -10`

Expected: error — `PearlBlock` not found.

- [ ] **Step 3: Implement `PearlBlock`, `serialize`, `deserialize`**

```rust
// crates/pearl-block/src/serial.rs
use pearl_types::{BlockCertificate, Commitments, FoundTile};

#[derive(Debug, Clone, PartialEq)]
pub struct PearlBlock {
    pub version: u32,
    pub prev_hash: [u8; 32],
    pub tx_root: [u8; 32],
    pub timestamp: u64,
    pub n_bits: u32,
    pub certificate: BlockCertificate,
    pub transactions: Vec<Vec<u8>>,
}

pub fn serialize(block: &PearlBlock) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&block.version.to_le_bytes());
    buf.extend_from_slice(&block.prev_hash);
    buf.extend_from_slice(&block.tx_root);
    buf.extend_from_slice(&block.timestamp.to_le_bytes());
    buf.extend_from_slice(&block.n_bits.to_le_bytes());

    let c = &block.certificate;
    buf.extend_from_slice(&c.commitments.kappa);
    buf.extend_from_slice(&c.commitments.ha);
    buf.extend_from_slice(&c.commitments.hb);
    buf.extend_from_slice(&c.commitments.s_a);
    buf.extend_from_slice(&c.commitments.s_b);
    buf.extend_from_slice(&c.tile.tile_i.to_le_bytes());
    buf.extend_from_slice(&c.tile.tile_j.to_le_bytes());
    for &v in &c.tile.m_state {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf.extend_from_slice(&c.tile.final_hash);
    buf.extend_from_slice(&(c.merkle_proof_a.len() as u32).to_le_bytes());
    for h in &c.merkle_proof_a { buf.extend_from_slice(h); }
    buf.extend_from_slice(&(c.merkle_proof_b.len() as u32).to_le_bytes());
    for h in &c.merkle_proof_b { buf.extend_from_slice(h); }
    buf.extend_from_slice(&(block.transactions.len() as u32).to_le_bytes());
    for tx in &block.transactions {
        buf.extend_from_slice(&(tx.len() as u32).to_le_bytes());
        buf.extend_from_slice(tx);
    }
    buf
}

pub fn deserialize(bytes: &[u8]) -> Result<PearlBlock, String> {
    let mut pos = 0usize;

    // Inline macros avoid closure lifetime issues with &[u8] returns.
    macro_rules! take {
        ($n:expr) => {{
            if pos + $n > bytes.len() {
                return Err(format!("unexpected end of data at offset {pos}"));
            }
            let s = &bytes[pos..pos + $n];
            pos += $n;
            s
        }};
    }
    macro_rules! r32 { () => { u32::from_le_bytes(take!(4).try_into().unwrap()) }; }
    macro_rules! r64 { () => { u64::from_le_bytes(take!(8).try_into().unwrap()) }; }
    macro_rules! rh  { () => { <[u8;32]>::try_from(take!(32)).unwrap() }; }

    let version   = r32!();
    let prev_hash = rh!();
    let tx_root   = rh!();
    let timestamp = r64!();
    let n_bits    = r32!();

    let commitments = Commitments {
        kappa: rh!(), ha: rh!(), hb: rh!(), s_a: rh!(), s_b: rh!(),
    };

    let tile_i = r32!();
    let tile_j = r32!();
    let mut m_state = [0i32; 16];
    for v in &mut m_state {
        *v = i32::from_le_bytes(take!(4).try_into().unwrap());
    }
    let final_hash = rh!();
    let tile = FoundTile { tile_i, tile_j, m_state, final_hash };

    let pa_len = r32!() as usize;
    let mut merkle_proof_a = Vec::with_capacity(pa_len);
    for _ in 0..pa_len { merkle_proof_a.push(rh!()); }

    let pb_len = r32!() as usize;
    let mut merkle_proof_b = Vec::with_capacity(pb_len);
    for _ in 0..pb_len { merkle_proof_b.push(rh!()); }

    let certificate = BlockCertificate { commitments, tile, merkle_proof_a, merkle_proof_b };

    let tx_count = r32!() as usize;
    let mut transactions = Vec::with_capacity(tx_count);
    for _ in 0..tx_count {
        let tx_len = r32!() as usize;
        transactions.push(take!(tx_len).to_vec());
    }

    Ok(PearlBlock { version, prev_hash, tx_root, timestamp, n_bits, certificate, transactions })
}
```

- [ ] **Step 4: Wire up `lib.rs`**

```rust
// crates/pearl-block/src/lib.rs
mod serial;
mod validate;

pub use serial::{PearlBlock, deserialize, serialize};
pub use validate::{block_identity, validate_certificate};
```

Also create an empty `validate.rs` placeholder so the crate compiles:

```rust
// crates/pearl-block/src/validate.rs
use pearl_types::{BlockCertificate, MiningConfig};
use crate::serial::PearlBlock;

pub fn block_identity(_block: &PearlBlock) -> [u8; 32] { todo!() }
pub fn validate_certificate(_cert: &BlockCertificate, _config: &MiningConfig) -> bool { todo!() }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p pearl-block serial`

Expected: 3 serial tests all `ok`.

- [ ] **Step 6: Commit**

```bash
git add crates/pearl-block/src/
git commit -m "feat(pearl-block): add PearlBlock struct with serialize/deserialize"
```

---

### Task 9: `pearl-block` — Validation and Block Identity

**Files:**
- Modify: `crates/pearl-block/src/validate.rs`

`block_identity` = dSHA256 of the fixed header fields including a stubbed `pouw_meta`. `validate_certificate` re-runs the BLAKE3 difficulty check: `BLAKE3(M, key=sA) ≤ floor(2^(256 − floor(b))) · r · tm · tn`.

Note: `difficulty_bits` is treated as `floor(b)` in this prototype. The fractional part is deferred alongside the full zkSNARK integration.

- [ ] **Step 1: Write the failing tests**

```rust
// crates/pearl-block/src/validate.rs (add at bottom)
#[cfg(test)]
mod tests {
    use super::*;
    use pearl_types::{BlockCertificate, Commitments, FoundTile, MatrixParams, MiningConfig};
    use crate::serial::{PearlBlock, serialize};

    fn test_cert(s_a: [u8; 32]) -> BlockCertificate {
        BlockCertificate {
            commitments: Commitments {
                kappa: [0u8; 32], ha: [0u8; 32], hb: [0u8; 32],
                s_a, s_b: [0u8; 32],
            },
            tile: FoundTile {
                tile_i: 0, tile_j: 0,
                m_state: [99i32; 16],
                final_hash: [0u8; 32],
            },
            merkle_proof_a: vec![],
            merkle_proof_b: vec![],
        }
    }

    fn cfg(b: f64) -> MiningConfig {
        MiningConfig {
            params: MatrixParams { m: 4, n: 4, k: 64, r: 32, tm: 2, tn: 2 },
            difficulty_bits: b,
            sigma: vec![], mu: vec![],
        }
    }

    fn test_block(cert: BlockCertificate) -> PearlBlock {
        PearlBlock {
            version: 1,
            prev_hash: [1u8; 32],
            tx_root: [2u8; 32],
            timestamp: 12345,
            n_bits: 0x1b00ffff,
            certificate: cert,
            transactions: vec![],
        }
    }

    #[test]
    fn test_validate_certificate_passes_with_zero_difficulty() {
        // b=0 → threshold = 2^256 · r·tm·tn > any 256-bit hash → always passes
        let cert = test_cert([1u8; 32]);
        assert!(validate_certificate(&cert, &cfg(0.0)));
    }

    #[test]
    fn test_validate_certificate_fails_with_max_difficulty() {
        // b=256 → threshold = r·tm·tn = 128 → probability ≈ 0 of passing
        let cert = test_cert([1u8; 32]);
        assert!(!validate_certificate(&cert, &cfg(256.0)));
    }

    #[test]
    fn test_block_identity_is_deterministic() {
        let block = test_block(test_cert([0u8; 32]));
        assert_eq!(block_identity(&block), block_identity(&block));
    }

    #[test]
    fn test_block_identity_changes_with_prev_hash() {
        let mut b1 = test_block(test_cert([0u8; 32]));
        let mut b2 = b1.clone();
        b2.prev_hash = [0xffu8; 32];
        assert_ne!(block_identity(&b1), block_identity(&b2));
    }

    #[test]
    fn test_block_identity_changes_with_timestamp() {
        let mut b1 = test_block(test_cert([0u8; 32]));
        let mut b2 = b1.clone();
        b2.timestamp = b1.timestamp + 1;
        assert_ne!(block_identity(&b1), block_identity(&b2));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p pearl-block validate 2>&1 | head -10`

Expected: panics with `not yet implemented` (from the `todo!()` stubs).

- [ ] **Step 3: Implement `validate_certificate` and `block_identity`**

```rust
// crates/pearl-block/src/validate.rs
use num_bigint::BigUint;
use num_traits::One;
use sha2::{Digest, Sha256};

use pearl_types::{BlockCertificate, MiningConfig};
use crate::serial::PearlBlock;

pub fn block_identity(block: &PearlBlock) -> [u8; 32] {
    // pouw_meta = SHA256(final_hash) — stub until pearl-proof is added
    let pouw_meta = Sha256::digest(&block.certificate.tile.final_hash);

    let mut header = Vec::with_capacity(4 + 32 + 32 + 8 + 4 + 32);
    header.extend_from_slice(&block.version.to_le_bytes());
    header.extend_from_slice(&block.prev_hash);
    header.extend_from_slice(&block.tx_root);
    header.extend_from_slice(&block.timestamp.to_le_bytes());
    header.extend_from_slice(&block.n_bits.to_le_bytes());
    header.extend_from_slice(&pouw_meta);

    let first:  [u8; 32] = Sha256::digest(&header).into();
    let second: [u8; 32] = Sha256::digest(first).into();
    second
}

pub fn validate_certificate(cert: &BlockCertificate, config: &MiningConfig) -> bool {
    let mut m_bytes = [0u8; 64];
    for (i, &v) in cert.tile.m_state.iter().enumerate() {
        m_bytes[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
    }

    let hash = blake3::keyed_hash(&cert.commitments.s_a, &m_bytes);
    let hash_val = BigUint::from_bytes_le(hash.as_bytes());
    let threshold = difficulty_threshold(
        config.difficulty_bits,
        config.params.r,
        config.params.tm,
        config.params.tn,
    );
    hash_val <= threshold
}

/// threshold = floor(2^(256 - floor(b))) · r · tm · tn
/// Uses floor(b) as a prototype approximation for fractional difficulty bits.
fn difficulty_threshold(b: f64, r: u32, tm: u32, tn: u32) -> BigUint {
    let b_int = b.floor() as u32;
    let shift = 256u32.saturating_sub(b_int);
    (BigUint::one() << shift) * r * tm * tn
}
```

- [ ] **Step 4: Run all pearl-block tests to verify they pass**

Run: `cargo test -p pearl-block`

Expected: 8 tests all `ok`.

- [ ] **Step 5: Run the full workspace test suite**

Run: `cargo test`

Expected: all tests across all 5 crates pass with 0 failures.

- [ ] **Step 6: Commit**

```bash
git add crates/pearl-block/src/validate.rs
git commit -m "feat(pearl-block): implement block_identity and validate_certificate"
```

---

## What's Next (Part 2)

Part 2 covers:
1. `pearl-gpu` — CUDA TiledMatMul kernel (`cudarc`, `build.rs` compiling `.cu`)
2. `pearl-miner` binary — Tokio-based orchestration loop, multi-GPU task spawning, AI workload integration
