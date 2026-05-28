# Async Multi-GPU Mining Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the synchronous single-GPU main loop with a Tokio async pipeline that fans each (A, B) matrix pair out to N GPU workers simultaneously and returns the clean A·B product via a clone-able `PipelineHandle`.

**Architecture:** Three concurrent actors — one preprocessor task and N GPU worker tasks — linked by a custom `LastSlot` queue (raw jobs) and a Tokio `watch` channel (prepared jobs). The preprocessor handles CPU commitment + noise work; GPU workers independently compute A'·B' and recover clean A·B; the first worker to finish wins and delivers the result via a shared `oneshot` sender. Found blocks are emitted on a separate `mpsc` channel.

**Tech Stack:** Rust 2021 · `tokio 1` (sync, rt-multi-thread, macros) · existing `pearl-*` crates

---

## File Map

| File | Responsibility |
|---|---|
| `Cargo.toml` | Add `sync` to tokio features |
| `src/pipeline/mod.rs` | `PipelineError`, `PipelineHandle`, `BlockReceiver`, `MiningPipeline`, `start()` |
| `src/pipeline/slot.rs` | `LastSlot<T>` — single-slot last-write-wins async queue |
| `src/pipeline/jobs.rs` | `RawJob`, `PreparedJob` |
| `src/pipeline/worker.rs` | `apply_noise()`, `assemble_block()`, `preprocessor_loop()`, `gpu_worker_loop()` |
| `src/main.rs` | Simplified entry point using `MiningPipeline::start()` |

---

### Task 1: Enable `sync` feature and create module skeleton

**Files:**
- Modify: `Cargo.toml`
- Create: `src/pipeline/mod.rs`
- Create: `src/pipeline/slot.rs`
- Create: `src/pipeline/jobs.rs`
- Create: `src/pipeline/worker.rs`
- Modify: `src/main.rs` (add `mod pipeline;`)

- [ ] **Step 1: Add `sync` to tokio features**

In `Cargo.toml`, change:
```toml
tokio  = { version = "1", features = ["rt-multi-thread", "macros"] }
```
to:
```toml
tokio  = { version = "1", features = ["rt-multi-thread", "macros", "sync", "time"] }
```
(`time` is needed for `tokio::time::timeout` in tests.)

- [ ] **Step 2: Create the four empty module files**

```rust
// src/pipeline/mod.rs
mod slot;
mod jobs;
mod worker;
```

```rust
// src/pipeline/slot.rs
```

```rust
// src/pipeline/jobs.rs
```

```rust
// src/pipeline/worker.rs
```

- [ ] **Step 3: Declare the module in `src/main.rs`**

Add `mod pipeline;` as the first line of `src/main.rs`.

- [ ] **Step 4: Verify it compiles**

Run: `cargo check`

Expected: 0 errors (unused-module warnings are fine).

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/pipeline/mod.rs src/pipeline/slot.rs src/pipeline/jobs.rs src/pipeline/worker.rs src/main.rs
git commit -m "feat(pipeline): add pipeline module skeleton and enable tokio sync"
```

---

### Task 2: `LastSlot<T>` — single-slot last-write-wins queue

**Files:**
- Modify: `src/pipeline/slot.rs`

`put()` replaces any stored value and returns the displaced one. `take()` suspends until a value is available, then removes and returns it. This is used for the raw job queue because `RawJob` contains a non-Clone `oneshot::Sender`.

- [ ] **Step 1: Write the failing tests**

```rust
// src/pipeline/slot.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_take_after_put() {
        let slot = LastSlot::new();
        slot.put(42u32);
        assert_eq!(slot.take().await, 42);
    }

    #[tokio::test]
    async fn test_put_displaces_previous() {
        let slot = LastSlot::new();
        assert!(slot.put(1u32).is_none());   // slot was empty
        assert_eq!(slot.put(2u32), Some(1)); // previous value returned
        assert_eq!(slot.take().await, 2);    // latest value wins
    }

    #[tokio::test]
    async fn test_take_blocks_until_put() {
        use std::sync::Arc;
        use tokio::time::{sleep, Duration};

        let slot = Arc::new(LastSlot::new());
        let slot2 = Arc::clone(&slot);
        let handle = tokio::spawn(async move { slot2.take().await });
        sleep(Duration::from_millis(10)).await;
        slot.put(99u32);
        assert_eq!(handle.await.unwrap(), 99);
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p pearl-miner pipeline::slot 2>&1 | head -20`

Expected: error — `LastSlot` not found.

- [ ] **Step 3: Implement `LastSlot`**

```rust
// src/pipeline/slot.rs
use std::sync::Mutex;
use tokio::sync::Notify;

pub struct LastSlot<T> {
    inner:  Mutex<Option<T>>,
    notify: Notify,
}

impl<T: Send> LastSlot<T> {
    pub fn new() -> Self {
        Self { inner: Mutex::new(None), notify: Notify::new() }
    }

    /// Swap in `val`; return any displaced predecessor.
    pub fn put(&self, val: T) -> Option<T> {
        let old = self.inner.lock().unwrap().replace(val);
        self.notify.notify_one();
        old
    }

    /// Suspend until a value is available, then take it.
    pub async fn take(&self) -> T {
        loop {
            {
                let mut guard = self.inner.lock().unwrap();
                if let Some(val) = guard.take() {
                    return val;
                }
            }
            self.notify.notified().await;
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p pearl-miner pipeline::slot`

Expected: 3 tests, all `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/slot.rs
git commit -m "feat(pipeline): add LastSlot single-slot async queue"
```

---

### Task 3: `RawJob` and `PreparedJob` data types

**Files:**
- Modify: `src/pipeline/jobs.rs`

No tests needed — these are plain data structs. `PipelineError` is referenced here and will be defined in `mod.rs` in the next task; we forward-declare it with `use`.

- [ ] **Step 1: Write the types**

```rust
// src/pipeline/jobs.rs
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;
use pearl_types::{Commitments, MiningConfig};

use crate::pipeline::PipelineError;

/// Lives from submit() until the preprocessor calls LastSlot::take().
pub struct RawJob {
    pub a:         Arc<[i8]>,
    pub b:         Arc<[i8]>,
    pub config:    Arc<MiningConfig>,
    pub result_tx: oneshot::Sender<Result<Vec<i32>, PipelineError>>,
}

/// Lives from end-of-preprocessing until all GPU workers finish with it.
/// Shared across workers via Arc.
pub struct PreparedJob {
    pub a:       Arc<[i8]>,          // original A — needed for peel recovery
    pub b:       Arc<[i8]>,          // original B — needed for peel recovery
    pub a_noisy: Vec<i8>,            // A' = A + E
    pub b_noisy: Vec<i8>,            // B' = B + F
    pub el:      Vec<i8>,
    pub er:      Vec<i8>,
    pub fl:      Vec<i8>,
    pub fr:      Vec<i8>,
    pub commitments: Arc<Commitments>,
    pub config:      Arc<MiningConfig>,
    /// First GPU worker to take the sender wins; others see None.
    pub result_tx: Arc<Mutex<Option<oneshot::Sender<Result<Vec<i32>, PipelineError>>>>>,
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check`

Expected: 0 errors (unused-import warnings about `PipelineError` are fine; it will be defined next).

- [ ] **Step 3: Commit**

```bash
git add src/pipeline/jobs.rs
git commit -m "feat(pipeline): add RawJob and PreparedJob types"
```

---

### Task 4: Public API — `PipelineError`, `PipelineHandle`, `BlockReceiver`, `MiningPipeline`

**Files:**
- Modify: `src/pipeline/mod.rs`

`MiningPipeline::start()` is stubbed with `todo!()` — it will be filled in Task 8.

- [ ] **Step 1: Write the compilation test**

```rust
// src/pipeline/mod.rs — add at the bottom
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_handle_is_clone() {
        fn assert_clone<T: Clone>() {}
        assert_clone::<PipelineHandle>();
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p pearl-miner pipeline::tests::pipeline_handle_is_clone 2>&1 | head -10`

Expected: error — `PipelineHandle` not found.

- [ ] **Step 3: Implement the public types**

```rust
// src/pipeline/mod.rs
pub mod slot;
pub mod jobs;
pub mod worker;

use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use pearl_block::PearlBlock;
use pearl_types::MiningConfig;

use slot::LastSlot;
pub use jobs::{PreparedJob, RawJob};

#[derive(Debug)]
pub enum PipelineError {
    /// Job displaced by a newer submission before preprocessing began.
    Dropped,
    /// A worker task panicked or all GPU workers exited.
    WorkerPanic,
}

/// Clone-able submit handle. Multiple callers may call submit() concurrently.
#[derive(Clone)]
pub struct PipelineHandle {
    pub(crate) raw_slot: Arc<LastSlot<RawJob>>,
    pub(crate) config:   Arc<MiningConfig>,
}

impl PipelineHandle {
    /// Submit (A, B) for mining. Returns clean A·B when any GPU finishes.
    /// Returns Err(Dropped) if a newer submission displaced this one before
    /// the preprocessor started working on it.
    pub async fn submit(
        &self,
        a: Arc<[i8]>,
        b: Arc<[i8]>,
    ) -> Result<Vec<i32>, PipelineError> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let job = RawJob {
            a,
            b,
            config: Arc::clone(&self.config),
            result_tx,
        };
        // Swap in the new job; immediately signal any displaced predecessor.
        if let Some(displaced) = self.raw_slot.put(job) {
            let _ = displaced.result_tx.send(Err(PipelineError::Dropped));
        }
        result_rx.await.map_err(|_| PipelineError::WorkerPanic)?
    }
}

/// Receives found blocks. Not Clone — single consumer expected.
pub struct BlockReceiver {
    pub(crate) rx: mpsc::Receiver<PearlBlock>,
}

impl BlockReceiver {
    /// Returns None when the pipeline shuts down.
    pub async fn next(&mut self) -> Option<PearlBlock> {
        self.rx.recv().await
    }
}

/// Owns all worker JoinHandles. Drop to shut down.
pub struct MiningPipeline {
    _handles: Vec<JoinHandle<()>>,
}

impl MiningPipeline {
    /// Spawns preprocessor + one GPU worker per device_id.
    pub fn start(
        config: Arc<MiningConfig>,
        device_ids: &[u32],
    ) -> (MiningPipeline, PipelineHandle, BlockReceiver) {
        todo!("implemented in Task 8")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_handle_is_clone() {
        fn assert_clone<T: Clone>() {}
        assert_clone::<PipelineHandle>();
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p pearl-miner pipeline::tests`

Expected: `test pipeline::tests::pipeline_handle_is_clone ... ok`

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/mod.rs
git commit -m "feat(pipeline): add PipelineHandle, BlockReceiver, MiningPipeline public API"
```

---

### Task 5: `apply_noise()` and `assemble_block()` helpers

**Files:**
- Modify: `src/pipeline/worker.rs`

`apply_noise` adds the factored noise `EL·ER` to a matrix. `assemble_block` constructs a `PearlBlock` from a winning tile; `prev_hash` and `tx_root` are zero-filled until P2P is implemented.

- [ ] **Step 1: Write the failing test**

```rust
// src/pipeline/worker.rs
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use pearl_types::{Commitments, FoundTile, MatrixParams, MiningConfig};

    pub fn dummy_config() -> Arc<MiningConfig> {
        Arc::new(MiningConfig {
            params: MatrixParams { m: 4, n: 4, k: 64, r: 32, tm: 2, tn: 2 },
            difficulty_bits: 0.0,
            sigma: b"test".to_vec(),
            mu:    b"test".to_vec(),
        })
    }

    pub fn dummy_commitments() -> Arc<Commitments> {
        Arc::new(Commitments {
            kappa: [0u8; 32], ha: [1u8; 32], hb: [2u8; 32],
            s_a:   [3u8; 32], s_b: [4u8; 32],
        })
    }

    #[test]
    fn test_apply_noise_adds_el_er() {
        // EL: 2×2, all zeros; ER: 2×4, all zeros → noise is zero → result == base
        let base  = vec![1i8, 2, 3, 4, 5, 6, 7, 8]; // 2 rows × 4 cols
        let el    = vec![0i8; 2 * 2];
        let er    = vec![0i8; 2 * 4];
        let result = apply_noise(&base, &el, &er, 2, 4, 2);
        assert_eq!(result, base);
    }

    #[test]
    fn test_assemble_block_fields() {
        let a: Arc<[i8]> = vec![1i8; 4 * 64].into();
        let b: Arc<[i8]> = vec![2i8; 64 * 4].into();
        let config = dummy_config();
        let tile = FoundTile {
            tile_i: 1, tile_j: 2,
            m_state: [7i32; 16],
            final_hash: [0xAAu8; 32],
        };
        let commitments = dummy_commitments();

        let block = assemble_block(&tile, &a, &b, &commitments, &config);

        assert_eq!(block.version, 1);
        assert_eq!(block.certificate.tile, tile);
        assert_eq!(block.certificate.commitments, *commitments);
        assert!(!block.certificate.merkle_proof_a.is_empty());
        assert!(!block.certificate.merkle_proof_b.is_empty());
    }
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p pearl-miner pipeline::worker::tests 2>&1 | head -20`

Expected: error — `apply_noise` and `assemble_block` not found.

- [ ] **Step 3: Implement `apply_noise` and `assemble_block`**

```rust
// src/pipeline/worker.rs
use std::sync::Arc;
use pearl_block::PearlBlock;
use pearl_commitment::MerkleTree;
use pearl_types::{BlockCertificate, Commitments, FoundTile, MiningConfig};

/// Add noise matrix EL·ER to `base` element-wise (saturating i8 add).
/// base: rows×cols, EL: rows×rank, ER: rank×cols — all row-major flat arrays.
pub fn apply_noise(base: &[i8], el: &[i8], er: &[i8], rows: usize, cols: usize, rank: usize) -> Vec<i8> {
    let mut result = base.to_vec();
    for i in 0..rows {
        for j in 0..cols {
            let mut e: i32 = 0;
            for p in 0..rank {
                e += el[i * rank + p] as i32 * er[p * cols + j] as i32;
            }
            result[i * cols + j] = base[i * cols + j].saturating_add(e as i8);
        }
    }
    result
}

/// Build a PearlBlock from a winning tile.
/// prev_hash, tx_root are zero-filled placeholders until P2P is integrated.
pub fn assemble_block(
    tile: &FoundTile,
    a: &[i8],
    b: &[i8],
    commitments: &Commitments,
    config: &MiningConfig,
) -> PearlBlock {
    let k = config.params.k as usize;
    let m = config.params.m as usize;

    let proof_a = MerkleTree::from_rows(a, k).proof(tile.tile_i as usize);
    let proof_b = MerkleTree::from_rows(b, m).proof(tile.tile_j as usize);

    PearlBlock {
        version:     1,
        prev_hash:   [0u8; 32],
        tx_root:     [0u8; 32],
        timestamp:   std::time::SystemTime::now()
                         .duration_since(std::time::UNIX_EPOCH)
                         .unwrap_or_default()
                         .as_secs(),
        n_bits:      0x1b00ffff,
        certificate: BlockCertificate {
            commitments: commitments.clone(),
            tile:        tile.clone(),
            merkle_proof_a: proof_a,
            merkle_proof_b: proof_b,
        },
        transactions: vec![],
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p pearl-miner pipeline::worker::tests`

Expected: 2 tests, both `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/worker.rs
git commit -m "feat(pipeline): add apply_noise and assemble_block helpers"
```

---

### Task 6: `gpu_worker_loop`

**Files:**
- Modify: `src/pipeline/worker.rs`

The loop is parameterised over a generic `mine_fn` so tests can supply a mock without GPU hardware. Internally, `mine_fn` is wrapped in `Arc<Mutex<F>>` so it can be called via `spawn_blocking` across multiple loop iterations without being consumed.

- [ ] **Step 1: Write the failing tests**

```rust
// src/pipeline/worker.rs — add to the tests module (inside mod tests { ... })

    use crate::pipeline::{PipelineError, PreparedJob};
    use std::sync::Mutex;
    use tokio::sync::{mpsc, watch};

    fn make_prepared_job(
        config: Arc<MiningConfig>,
        result_tx: tokio::sync::oneshot::Sender<Result<Vec<i32>, PipelineError>>,
    ) -> Arc<PreparedJob> {
        let (m, k, n, r) = (4usize, 64usize, 4usize, 32usize);
        let a: Arc<[i8]> = vec![1i8; m * k].into();
        let b: Arc<[i8]> = vec![2i8; k * n].into();
        Arc::new(PreparedJob {
            a:       Arc::clone(&a),
            b:       Arc::clone(&b),
            a_noisy: a.to_vec(),
            b_noisy: b.to_vec(),
            el: vec![0i8; m * r],
            er: vec![0i8; r * k],
            fl: vec![0i8; k * r],
            fr: vec![0i8; r * n],
            commitments: dummy_commitments(),
            config,
            result_tx: Arc::new(Mutex::new(Some(result_tx))),
        })
    }

    #[tokio::test]
    async fn test_worker_delivers_clean_product() {
        let config = dummy_config();
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let job = make_prepared_job(Arc::clone(&config), result_tx);

        let (prepared_tx, prepared_rx) = watch::channel(None::<Arc<PreparedJob>>);
        let (block_tx, _block_rx) = mpsc::channel(8);

        let mock_mine = move |_a: &[i8], _b: &[i8], _c: &Commitments, cfg: &MiningConfig| {
            let sz = (cfg.params.m * cfg.params.n) as usize;
            Ok::<(Vec<FoundTile>, Vec<i32>), anyhow::Error>((vec![], vec![42i32; sz]))
        };

        tokio::spawn(gpu_worker_loop(prepared_rx, block_tx, mock_mine));
        prepared_tx.send_replace(Some(job));

        let result = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            result_rx,
        ).await.expect("timed out").unwrap();

        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 4 * 4);
    }

    #[tokio::test]
    async fn test_worker_emits_block_when_tile_found() {
        let config = dummy_config();
        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        let job = make_prepared_job(Arc::clone(&config), result_tx);

        let (prepared_tx, prepared_rx) = watch::channel(None::<Arc<PreparedJob>>);
        let (block_tx, mut block_rx) = mpsc::channel(8);

        let mock_mine = move |_a: &[i8], _b: &[i8], _c: &Commitments, cfg: &MiningConfig| {
            let tile = FoundTile {
                tile_i: 0, tile_j: 0,
                m_state: [1i32; 16],
                final_hash: [0u8; 32],
            };
            let sz = (cfg.params.m * cfg.params.n) as usize;
            Ok::<(Vec<FoundTile>, Vec<i32>), anyhow::Error>((vec![tile], vec![0i32; sz]))
        };

        tokio::spawn(gpu_worker_loop(prepared_rx, block_tx, mock_mine));
        prepared_tx.send_replace(Some(job));

        let block = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            block_rx.recv(),
        ).await.expect("timed out").expect("channel closed");
        assert_eq!(block.certificate.tile.tile_i, 0);
    }

    #[tokio::test]
    async fn test_two_workers_only_one_delivers_result() {
        let config = dummy_config();
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let job = make_prepared_job(Arc::clone(&config), result_tx);

        let (prepared_tx, prepared_rx1) = watch::channel(None::<Arc<PreparedJob>>);
        let prepared_rx2 = prepared_tx.subscribe();
        let (block_tx1, _) = mpsc::channel(8);
        let (block_tx2, _) = mpsc::channel(8);

        let mock_mine = move |_a: &[i8], _b: &[i8], _c: &Commitments, cfg: &MiningConfig| {
            let sz = (cfg.params.m * cfg.params.n) as usize;
            Ok::<(Vec<FoundTile>, Vec<i32>), anyhow::Error>((vec![], vec![0i32; sz]))
        };

        tokio::spawn(gpu_worker_loop(prepared_rx1, block_tx1, mock_mine));
        tokio::spawn(gpu_worker_loop(prepared_rx2, block_tx2, mock_mine));
        prepared_tx.send_replace(Some(job));

        let result = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            result_rx,
        ).await.expect("timed out").unwrap();
        assert!(result.is_ok(), "expected Ok, got {:?}", result);
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test -p pearl-miner pipeline::worker::tests::test_worker 2>&1 | head -20`

Expected: error — `gpu_worker_loop` not found.

- [ ] **Step 3: Implement `gpu_worker_loop`**

Add this above the `#[cfg(test)]` block in `src/pipeline/worker.rs`:

```rust
use anyhow::Result;
use tokio::sync::{mpsc, watch};
use pearl_types::FoundTile;

/// Run the GPU worker loop, calling `mine_fn` for each PreparedJob.
///
/// `mine_fn` is wrapped in Arc<Mutex<>> internally so it can be called
/// via spawn_blocking across multiple iterations without being consumed.
pub async fn gpu_worker_loop<F>(
    mut prepared_rx: watch::Receiver<Option<Arc<crate::pipeline::PreparedJob>>>,
    block_tx: mpsc::Sender<PearlBlock>,
    mine_fn: F,
)
where
    F: FnMut(&[i8], &[i8], &Commitments, &MiningConfig)
         -> Result<(Vec<FoundTile>, Vec<i32>)>
         + Send + 'static,
{
    use crate::pipeline::PipelineError;
    let mine_fn = Arc::new(std::sync::Mutex::new(mine_fn));

    loop {
        if prepared_rx.changed().await.is_err() {
            break; // sender dropped — pipeline shut down
        }

        let job = {
            let borrow = prepared_rx.borrow_and_update();
            match borrow.as_ref() {
                Some(j) => Arc::clone(j),
                None    => continue,
            }
        };

        // Clone everything needed inside spawn_blocking
        let a_noisy      = job.a_noisy.clone();
        let b_noisy      = job.b_noisy.clone();
        let el           = job.el.clone();
        let er           = job.er.clone();
        let fl           = job.fl.clone();
        let fr           = job.fr.clone();
        let a_in         = Arc::clone(&job.a);
        let b_in         = Arc::clone(&job.b);
        let commitments  = Arc::clone(&job.commitments);
        let config       = Arc::clone(&job.config);
        let params       = config.params.clone();
        let mine_fn2     = Arc::clone(&mine_fn);
        // Keep clones in async context for assemble_block after spawn_blocking
        let a_for_block  = Arc::clone(&job.a);
        let b_for_block  = Arc::clone(&job.b);
        let comm_block   = Arc::clone(&job.commitments);
        let cfg_block    = Arc::clone(&job.config);
        let result_tx    = Arc::clone(&job.result_tx);

        let mine_result = tokio::task::spawn_blocking(move || {
            let mut f = mine_fn2.lock().unwrap();
            let (found_tiles, ab_noisy) = f(&a_noisy, &b_noisy, &commitments, &config)?;
            let clean_ab = pearl_peel::recover(
                &ab_noisy, &a_in, &b_noisy,
                &el, &er, &fl, &fr, &params,
            );
            Ok::<(Vec<FoundTile>, Vec<i32>), anyhow::Error>((found_tiles, clean_ab))
        }).await;

        match mine_result {
            Ok(Ok((found_tiles, clean_ab))) => {
                // First worker to take result_tx wins; others see None and skip
                if let Some(tx) = result_tx.lock().unwrap().take() {
                    let _ = tx.send(Ok(clean_ab));
                }
                for tile in &found_tiles {
                    let block = assemble_block(tile, &a_for_block, &b_for_block,
                                               &comm_block, &cfg_block);
                    let _ = block_tx.send(block).await;
                }
            }
            Ok(Err(_gpu_err)) | Err(_panic) => {
                if let Some(tx) = result_tx.lock().unwrap().take() {
                    let _ = tx.send(Err(PipelineError::WorkerPanic));
                }
            }
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p pearl-miner pipeline::worker::tests`

Expected: 5 tests (2 from Task 5 + 3 new), all `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/worker.rs
git commit -m "feat(pipeline): add gpu_worker_loop with mockable mine function"
```

---

### Task 7: `preprocessor_loop`

**Files:**
- Modify: `src/pipeline/worker.rs`

The preprocessor takes `RawJob` from `LastSlot`, runs commitment + noise generation in `spawn_blocking`, and writes a `PreparedJob` into the `watch` channel. If all GPU workers exit (`receiver_count() == 0`), the loop exits.

- [ ] **Step 1: Write the failing test**

```rust
// src/pipeline/worker.rs — add to mod tests

    #[tokio::test]
    async fn test_preprocessor_dispatches_prepared_job() {
        use crate::pipeline::slot::LastSlot;
        use crate::pipeline::RawJob;

        let config = dummy_config();
        let raw_slot = Arc::new(LastSlot::<RawJob>::new());
        let (prepared_tx, mut prepared_rx) =
            watch::channel(None::<Arc<PreparedJob>>);

        let slot2 = Arc::clone(&raw_slot);
        tokio::spawn(preprocessor_loop(slot2, prepared_tx));

        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        let job = RawJob {
            a:         vec![1i8; 4 * 64].into(),
            b:         vec![2i8; 64 * 4].into(),
            config:    Arc::clone(&config),
            result_tx,
        };
        raw_slot.put(job);

        tokio::time::timeout(
            std::time::Duration::from_millis(500),
            prepared_rx.changed(),
        ).await.expect("preprocessor timed out").unwrap();

        let prepared = prepared_rx.borrow().clone()
            .expect("expected Some(PreparedJob)");
        assert_eq!(prepared.a_noisy.len(), 4 * 64);
        assert_eq!(prepared.b_noisy.len(), 64 * 4);
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p pearl-miner pipeline::worker::tests::test_preprocessor 2>&1 | head -20`

Expected: error — `preprocessor_loop` not found.

- [ ] **Step 3: Implement `preprocessor_loop`**

Add this above `gpu_worker_loop` in `src/pipeline/worker.rs`:

```rust
pub async fn preprocessor_loop(
    raw_slot:    Arc<crate::pipeline::slot::LastSlot<crate::pipeline::RawJob>>,
    prepared_tx: watch::Sender<Option<Arc<crate::pipeline::PreparedJob>>>,
) {
    use crate::pipeline::{PipelineError, PreparedJob};
    use std::sync::Mutex;

    loop {
        if prepared_tx.receiver_count() == 0 {
            break; // all GPU workers have exited — nothing to dispatch to
        }

        let job = raw_slot.take().await;

        let a   = Arc::clone(&job.a);
        let b   = Arc::clone(&job.b);
        let cfg = Arc::clone(&job.config);
        let result_tx_raw = job.result_tx;

        let preprocess_result = tokio::task::spawn_blocking(move || {
            let p = &cfg.params;
            let (m, n, k, r) = (p.m as usize, p.n as usize, p.k as usize, p.r as usize);
            let commitments  = pearl_commitment::compute(&a, &b, &cfg);
            let (el, er)     = pearl_noise::generate_e(p.m, p.k, p.r, &commitments.s_a);
            let (fl, fr)     = pearl_noise::generate_f(p.k, p.n, p.r, &commitments.s_b);
            let a_noisy      = apply_noise(&a, &el, &er, m, k, r);
            let b_noisy      = apply_noise(&b, &fl, &fr, k, n, r);
            PreparedJob {
                a:           Arc::clone(&a),
                b:           Arc::clone(&b),
                a_noisy,
                b_noisy,
                el, er, fl, fr,
                commitments: Arc::new(commitments),
                config:      cfg,
                result_tx:   Arc::new(Mutex::new(Some(result_tx_raw))),
            }
        }).await;

        match preprocess_result {
            Ok(prepared) => {
                // If all GPU workers exited while we were preprocessing, this send
                // still succeeds (watch doesn't close when receivers drop).
                // The PreparedJob will be discarded; result_tx is inside it and
                // will be dropped, closing the oneshot → caller gets WorkerPanic.
                let _ = prepared_tx.send_replace(Some(Arc::new(prepared)));
            }
            Err(_panic) => {
                // spawn_blocking panicked; result_tx_raw was moved in and is now
                // dropped, which closes the oneshot — caller gets WorkerPanic.
            }
        }
    }
}
```

- [ ] **Step 4: Add imports at the top of `src/pipeline/worker.rs`**

Ensure the top of the file has (add any missing lines):

```rust
use std::sync::Arc;
use anyhow::Result;
use tokio::sync::{mpsc, watch};
use pearl_block::PearlBlock;
use pearl_commitment::MerkleTree;
use pearl_types::{BlockCertificate, Commitments, FoundTile, MiningConfig};
```

- [ ] **Step 5: Run all worker tests to verify they pass**

Run: `cargo test -p pearl-miner pipeline::worker::tests`

Expected: 6 tests, all `ok`.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/worker.rs
git commit -m "feat(pipeline): add preprocessor_loop with commitment and noise generation"
```

---

### Task 8: Wire `MiningPipeline::start()` and integration tests

**Files:**
- Modify: `src/pipeline/mod.rs`

- [ ] **Step 1: Write the failing integration tests**

Add to the `tests` module in `src/pipeline/mod.rs`:

```rust
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::time::timeout;
    use pearl_types::MatrixParams;

    fn test_config() -> Arc<MiningConfig> {
        Arc::new(MiningConfig {
            params: MatrixParams { m: 4, n: 4, k: 64, r: 32, tm: 2, tn: 2 },
            difficulty_bits: 256.0, // no tiles win; A·B still returned
            sigma: b"test-sigma".to_vec(),
            mu:    b"test-mu".to_vec(),
        })
    }

    #[tokio::test]
    async fn test_drop_oldest_signals_dropped() {
        let raw_slot = Arc::new(slot::LastSlot::<RawJob>::new());
        let (result_tx1, result_rx1) = tokio::sync::oneshot::channel::<Result<Vec<i32>, PipelineError>>();
        let (result_tx2, _result_rx2) = tokio::sync::oneshot::channel::<Result<Vec<i32>, PipelineError>>();

        let config = test_config();
        let a: Arc<[i8]> = vec![0i8; 4 * 64].into();
        let b: Arc<[i8]> = vec![0i8; 64 * 4].into();

        // Put job1 — slot is empty, no displacement
        raw_slot.put(RawJob { a: Arc::clone(&a), b: Arc::clone(&b),
                               config: Arc::clone(&config), result_tx: result_tx1 });
        // Put job2 — displaces job1
        if let Some(displaced) = raw_slot.put(
            RawJob { a, b, config, result_tx: result_tx2 }
        ) {
            let _ = displaced.result_tx.send(Err(PipelineError::Dropped));
        }

        let result = result_rx1.await.unwrap();
        assert!(matches!(result, Err(PipelineError::Dropped)));
    }

    #[tokio::test]
    async fn test_pipeline_handle_is_clone_and_submit_works() {
        // Structural: verify Clone is usable and todo!() is gone after Task 8
        let (pipeline, handle, _blocks) =
            MiningPipeline::start(test_config(), &[]);
        let handle2 = handle.clone(); // must compile
        drop(handle2);
        drop(pipeline);
    }
```

- [ ] **Step 2: Run to verify `test_drop_oldest_signals_dropped` passes already**

Run: `cargo test -p pearl-miner pipeline::tests::test_drop_oldest_signals_dropped`

Expected: `ok` (this test exercises `LastSlot` directly, no `start()` needed).

- [ ] **Step 3: Implement `MiningPipeline::start()`**

Replace the `todo!("implemented in Task 8")` body in `src/pipeline/mod.rs`:

```rust
    pub fn start(
        config: Arc<MiningConfig>,
        device_ids: &[u32],
    ) -> (MiningPipeline, PipelineHandle, BlockReceiver) {
        let raw_slot = Arc::new(slot::LastSlot::<RawJob>::new());
        let (prepared_tx, _) = watch::channel(None::<Arc<PreparedJob>>);
        let (block_tx, block_rx) = mpsc::channel::<PearlBlock>(64);

        let mut handles: Vec<JoinHandle<()>> = Vec::new();

        // Spawn one GPU worker per device
        for &device_id in device_ids {
            let prepared_rx = prepared_tx.subscribe();
            let btx         = block_tx.clone();

            handles.push(tokio::spawn(async move {
                let gpu_result = tokio::task::spawn_blocking(
                    move || pearl_gpu::GpuMiner::new(device_id)
                ).await;

                match gpu_result {
                    Ok(Ok(gpu)) => {
                        let mine_fn = move |a: &[i8], b: &[i8],
                                            c: &pearl_types::Commitments,
                                            cfg: &pearl_types::MiningConfig| {
                            gpu.mine(a, b, c, cfg)
                        };
                        worker::gpu_worker_loop(prepared_rx, btx, mine_fn).await;
                    }
                    _ => {
                        // GPU init failed; receiver_count drops — preprocessor will
                        // eventually see 0 and exit cleanly.
                    }
                }
            }));
        }

        // Spawn the preprocessor after GPU workers so receiver_count > 0 at start
        let slot_clone = Arc::clone(&raw_slot);
        handles.push(tokio::spawn(worker::preprocessor_loop(slot_clone, prepared_tx)));

        let handle = PipelineHandle {
            raw_slot,
            config,
        };

        (MiningPipeline { _handles: handles }, handle, BlockReceiver { rx: block_rx })
    }
```

- [ ] **Step 4: Run all pipeline tests**

Run: `cargo test -p pearl-miner pipeline`

Expected: all pipeline tests pass.

- [ ] **Step 5: Run workspace tests (excluding GPU crate)**

Run: `cargo test --workspace --exclude pearl-gpu`

Expected: all tests pass across all non-GPU crates.

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/mod.rs
git commit -m "feat(pipeline): implement MiningPipeline::start() wiring all workers"
```

---

### Task 9: Update `src/main.rs`

**Files:**
- Modify: `src/main.rs`

Replace the existing synchronous pipeline with the new async one. The mining loop submits synthetic matrices in a tight loop — in production these come from the AI workload.

- [ ] **Step 1: Replace the body of `src/main.rs`**

```rust
// src/main.rs
mod pipeline;

use std::sync::Arc;
use anyhow::Result;
use pearl_types::{MatrixParams, MiningConfig};
use pipeline::{MiningPipeline, PipelineError};

#[tokio::main]
async fn main() -> Result<()> {
    let config = Arc::new(MiningConfig {
        params: MatrixParams {
            m: 32, n: 32, k: 512, r: 32,
            tm: 4,  tn: 4,
        },
        difficulty_bits: 1.0, // very low: almost every tile wins (for smoke-testing)
        sigma: b"pearl-genesis-block".to_vec(),
        mu:    b"miner-pubkey-placeholder".to_vec(),
    });

    // Try GPU devices 0–3; workers that fail to init exit silently
    let device_ids: Vec<u32> = (0u32..4).collect();
    eprintln!("[init] starting pipeline (up to {} GPU device(s))", device_ids.len());

    let (_pipeline, handle, mut blocks) =
        MiningPipeline::start(Arc::clone(&config), &device_ids);

    // Print found blocks in a background task
    tokio::spawn(async move {
        while let Some(block) = blocks.next().await {
            let id    = pearl_block::block_identity(&block);
            let bytes = pearl_block::serialize(&block);
            eprintln!("[block] identity={}  size={} bytes", hex(&id), bytes.len());
        }
        eprintln!("[block] block receiver closed");
    });

    // Mining loop — submit the same synthetic (A, B) pair repeatedly.
    // In production, A and B come from the AI inference workload.
    let p = &config.params;
    let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);
    let a: Arc<[i8]> = (0..m * k)
        .map(|i| ((i * 7 + 3) % 128) as i8 - 64)
        .collect::<Vec<_>>()
        .into();
    let b: Arc<[i8]> = (0..k * n)
        .map(|i| ((i * 11 + 5) % 128) as i8 - 64)
        .collect::<Vec<_>>()
        .into();
    eprintln!("[data] generated {}×{} A and {}×{} B (INT8)", m, k, k, n);

    loop {
        match handle.submit(Arc::clone(&a), Arc::clone(&b)).await {
            Ok(clean_ab) => {
                eprintln!("[peel] recovered {}×{} clean product ({} elements)",
                    m, n, clean_ab.len());
            }
            Err(PipelineError::Dropped) => {
                eprintln!("[submit] job dropped — superseded by newer submission");
            }
            Err(PipelineError::WorkerPanic) => {
                eprintln!("[submit] all GPU workers exited — shutting down");
                break;
            }
        }
    }

    Ok(())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check`

Expected: 0 errors.

- [ ] **Step 3: Run the full workspace test suite one final time**

Run: `cargo test --workspace --exclude pearl-gpu`

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat: replace sync main loop with async MiningPipeline"
```

---

### Optional: GPU smoke test

This test requires a CUDA-capable GPU. Skip it in CI; run it manually with `PEARL_GPU_TEST=1`.

Add to `src/pipeline/mod.rs` tests module:

```rust
    #[tokio::test]
    #[ignore = "requires CUDA GPU — run with PEARL_GPU_TEST=1 cargo test -- --ignored"]
    async fn test_gpu_smoke_zero_difficulty() {
        use std::time::Duration;
        use tokio::time::timeout;

        let config = Arc::new(MiningConfig {
            params: MatrixParams { m: 8, n: 8, k: 64, r: 32, tm: 4, tn: 4 },
            difficulty_bits: 0.0, // all tiles win
            sigma: b"smoke".to_vec(),
            mu:    b"smoke".to_vec(),
        });

        let (pipeline, handle, mut blocks) = MiningPipeline::start(Arc::clone(&config), &[0]);
        let a: Arc<[i8]> = vec![1i8; 8 * 64].into();
        let b: Arc<[i8]> = vec![2i8; 64 * 8].into();

        let result = timeout(Duration::from_secs(10), handle.submit(a, b))
            .await
            .expect("timed out")
            .expect("submit failed");

        assert_eq!(result.len(), 8 * 8, "clean product has wrong size");

        let block = timeout(Duration::from_secs(5), blocks.next())
            .await
            .expect("no block received")
            .expect("block channel closed");

        assert_eq!(block.version, 1);
        drop(pipeline);
    }
```

---

## Self-Review

**Spec coverage:**

| Spec section | Covered by |
|---|---|
| Architecture diagram (3 actors, 2 channels) | Tasks 2, 7, 8 |
| `LastSlot<T>` for raw job queue | Task 2 |
| `RawJob`, `PreparedJob` types | Task 3 |
| `PipelineHandle::submit()` with drop-oldest | Task 4, Task 8 integration test |
| `BlockReceiver::next()` | Task 4 |
| `MiningPipeline::start()` | Task 8 |
| `apply_noise()` + `assemble_block()` | Task 5 |
| `preprocessor_loop()` | Task 7 |
| `gpu_worker_loop()` with mock support | Task 6 |
| `PipelineError::Dropped` / `WorkerPanic` | Tasks 4, 6, 7, 8 |
| Error table (all 6 rows) | Tasks 6, 7, 8 |
| Unit tests (no GPU) | Tasks 2, 4, 6, 7, 8 |
| Integration tests (mock GPU) | Task 6 |
| GPU smoke test (`#[ignore]`) | Optional task at end |
| Updated `main.rs` | Task 9 |

**No placeholder issues found.** All steps contain complete code.

**Type consistency check:**
- `LastSlot::put()` returns `Option<T>` — used in Task 4 `submit()` ✓
- `LastSlot::take()` returns `T` — used in Task 7 `preprocessor_loop()` ✓
- `gpu_worker_loop` takes `mine_fn: F where F: FnMut(...)` — matches Task 8 closure ✓
- `PreparedJob.result_tx: Arc<Mutex<Option<oneshot::Sender<...>>>>` — taken in Task 6 ✓
- `assemble_block` signature `(tile, a, b, commitments, config)` — matches Task 5 definition and Task 6 call site ✓
- `MiningPipeline::start()` returns `(MiningPipeline, PipelineHandle, BlockReceiver)` — matches Task 9 destructure ✓
