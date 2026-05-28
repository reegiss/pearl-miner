# Pearl Miner — Async Multi-GPU Mining Pipeline

**Date:** 2026-05-28  
**Status:** Approved  
**Scope:** `src/pipeline.rs` — Tokio-based orchestration of commitment generation, noise preprocessing, GPU tiled matmul, peel recovery, and block assembly across N GPUs.

---

## 1. Overview

The current `src/main.rs` runs a single-GPU synchronous pipeline. This design replaces it with an async, multi-GPU pipeline that:

- Exposes a clean in-process Rust API (`PipelineHandle`) for the AI workload
- Distributes the same (A, B) matrix pair to all GPUs simultaneously (independent replicas)
- Applies drop-oldest backpressure at the preprocessing stage
- Returns the clean A·B product to the caller as soon as any one GPU finishes
- Emits assembled `PearlBlock` values on a channel whenever a tile satisfies difficulty

All orchestration lives in `src/pipeline.rs` (not a new crate) because it wires together all logic crates — a separate logic crate would violate the "no logic crate imports another" invariant.

---

## 2. Architecture

Three concurrent actors communicate through two Tokio `watch` channels:

```
submit(a, b)
    │
    ▼
[raw_watch]  ← watch::Sender<Option<Arc<RawJob>>>
    │             drop-oldest: submit() swaps in new job,
    │             signals Err(Dropped) to any displaced caller immediately
    ▼
Preprocessor task  (1×, spawn_blocking for CPU work)
    • pearl_commitment::compute()
    • pearl_noise::generate_e() / generate_f()
    • apply_noise() → A' = A + E,  B' = B + F
    │
    ▼
[prepared_watch]  ← watch::Sender<Option<Arc<PreparedJob>>>
    │
    ├── GPU Worker 0  (spawn_blocking → GpuMiner::mine() → pearl_peel::recover())
    ├── GPU Worker 1  (spawn_blocking → GpuMiner::mine() → pearl_peel::recover())
    └── GPU Worker N  …

First GPU to finish:
    → result_tx.send(Ok(clean_ab))   oneshot, taken from Arc<Mutex<Option<…>>>
    → block_tx.send(PearlBlock)      mpsc, one send per found tile
```

### Drop-Oldest Semantics

Drop-oldest is provided by `watch::Sender::send_replace()`:

- **Raw level:** if `submit()` is called while the preprocessor is idle, `send_replace` returns the old `RawJob`; `submit()` immediately sends `Err(Dropped)` on the displaced caller's `result_tx`. If the preprocessor already took the job, `send_replace` returns `None` — the old job is in-flight and will complete normally.
- **Prepared level:** if a second `PreparedJob` arrives before GPUs pick up the first, it silently overwrites it (the stale prepared job had not been started, so no caller is waiting on it).

---

## 3. Public API

```rust
// src/pipeline.rs

#[derive(Debug)]
pub enum PipelineError {
    /// Job displaced by a newer submission before preprocessing began.
    Dropped,
    /// A worker task panicked or all GPU workers exited.
    WorkerPanic,
}

/// Clone-able handle; multiple threads may call submit() concurrently.
#[derive(Clone)]
pub struct PipelineHandle {
    raw_tx:  watch::Sender<Option<Arc<RawJob>>>,
    block_rx: Arc<Mutex<mpsc::Receiver<PearlBlock>>>,  // shared across clones
}

impl PipelineHandle {
    /// Submit (A, B) for mining. Returns clean A·B when any GPU finishes.
    /// Err(Dropped) if a newer submission displaced this one before preprocessing.
    pub async fn submit(
        &self,
        a: Arc<[i8]>,
        b: Arc<[i8]>,
    ) -> Result<Vec<i32>, PipelineError>;

    /// Receive the next found block. Returns None when the pipeline shuts down.
    pub async fn next_block(&self) -> Option<PearlBlock>;
}

/// Owns all worker JoinHandles. Drop to shut down.
pub struct MiningPipeline {
    _handles: Vec<JoinHandle<()>>,
}

impl MiningPipeline {
    /// Spawns preprocessor + one GPU worker per device_id.
    /// Returns (pipeline_owner, shareable_handle).
    pub fn start(
        config: Arc<MiningConfig>,
        device_ids: &[u32],
    ) -> (MiningPipeline, PipelineHandle);
}
```

`a` and `b` are `Arc<[i8]>` so the pipeline shares ownership without copying matrices into each GPU task. `MiningConfig` is `Arc`-shared and read-only for the pipeline's lifetime.

---

## 4. Internal Types

```rust
/// Lives from submit() until preprocessing begins (or until Dropped).
struct RawJob {
    a:         Arc<[i8]>,
    b:         Arc<[i8]>,
    config:    Arc<MiningConfig>,
    result_tx: oneshot::Sender<Result<Vec<i32>, PipelineError>>,
}

/// Lives from end-of-preprocessing until all GPU workers finish with it.
struct PreparedJob {
    // Originals needed for peel recovery
    a:       Arc<[i8]>,
    b:       Arc<[i8]>,
    // Noisy inputs for the GPU
    a_noisy: Vec<i8>,
    b_noisy: Vec<i8>,
    // Noise factors for peel correction  
    el: Vec<i8>, er: Vec<i8>,
    fl: Vec<i8>, fr: Vec<i8>,
    commitments: Arc<Commitments>,
    config:      Arc<MiningConfig>,
    // Shared result slot — first GPU to take it wins
    result_tx: Arc<Mutex<Option<oneshot::Sender<Result<Vec<i32>, PipelineError>>>>>,
}
```

---

## 5. Task Logic

### submit()

```rust
pub async fn submit(&self, a: Arc<[i8]>, b: Arc<[i8]>) -> Result<Vec<i32>, PipelineError> {
    let (result_tx, result_rx) = oneshot::channel();
    let new_job = Arc::new(RawJob { a, b, config: self.config.clone(), result_tx });

    // Swap in new job; get back any displaced predecessor
    let old = self.raw_tx.send_replace(Some(Arc::clone(&new_job)));
    if let Some(displaced) = old.flatten() {
        // Arc::try_unwrap succeeds only if preprocessor hasn't taken it yet
        if let Ok(job) = Arc::try_unwrap(displaced) {
            let _ = job.result_tx.send(Err(PipelineError::Dropped));
        }
    }

    result_rx.await.map_err(|_| PipelineError::WorkerPanic)?
}
```

### Preprocessor Task

```
loop:
  raw_rx.changed().await
  let Some(job) = raw_rx.borrow_and_update().clone() else { continue }

  let result = spawn_blocking(move || {
      let commitments = pearl_commitment::compute(&job.a, &job.b, &job.config);
      let (el, er) = pearl_noise::generate_e(m, k, r, &commitments.s_a);
      let (fl, fr) = pearl_noise::generate_f(k, n, r, &commitments.s_b);
      let a_noisy = apply_noise(&job.a, &el, &er, m, k, r);
      let b_noisy = apply_noise(&job.b, &fl, &fr, k, n, r);
      PreparedJob { ..., result_tx: Arc::new(Mutex::new(Some(job.result_tx))) }
  }).await;

  match result {
      Ok(prepared) => { prepared_tx.send_replace(Some(Arc::new(prepared))); }
      Err(_panic)  => { /* take result_tx and send WorkerPanic */ }
  }
```

### GPU Worker Task (per device)

```
// One-time init
let gpu = spawn_blocking(move || GpuMiner::new(device_id)).await?;

loop:
  prepared_rx.changed().await
  let Some(job) = prepared_rx.borrow_and_update().clone() else { continue }

  let result = spawn_blocking(move || {
      let (found_tiles, ab_noisy) = gpu.mine(&job.a_noisy, &job.b_noisy,
                                              &job.commitments, &job.config)?;
      let clean_ab = pearl_peel::recover(&ab_noisy, &job.a, &job.b_noisy,
                                         &job.el, &job.er, &job.fl, &job.fr,
                                         &job.config.params);
      Ok::<_, anyhow::Error>((found_tiles, clean_ab))
  }).await;

  match result {
      Ok(Ok((found_tiles, clean_ab))) => {
          // First GPU to take the sender wins; others see None
          if let Some(tx) = job.result_tx.lock().unwrap().take() {
              let _ = tx.send(Ok(clean_ab));
          }
          for tile in found_tiles {
              let block = assemble_block(&tile, &job);
              let _ = block_tx.send(block);
          }
      }
      Ok(Err(gpu_err)) | Err(_panic) => {
          // Take result_tx and signal error so caller unblocks
          if let Some(tx) = job.result_tx.lock().unwrap().take() {
              let _ = tx.send(Err(PipelineError::WorkerPanic));
          }
      }
  }
```

---

## 6. Error Handling

| Situation | Behavior |
|---|---|
| New `submit()` displaces queued `RawJob` | `Err(Dropped)` sent immediately to displaced caller |
| Preprocessor `spawn_blocking` panics | Preprocessor catches `JoinError`, sends `Err(WorkerPanic)` on job's `result_tx` |
| `GpuMiner::new()` fails at startup | GPU task exits; if all GPU tasks exit, `block_tx` drops and `next_block()` returns `None` |
| `GpuMiner::mine()` returns `Err` | GPU task takes `result_tx` and sends `Err(WorkerPanic)`; loops to next job |
| All GPU tasks exit | Preprocessor detects `prepared_tx.receiver_count() == 0`; sends `Err(WorkerPanic)` on any future job |
| `MiningPipeline` dropped | `JoinHandle`s dropped; tasks exit at next channel poll — no forced abort needed |

---

## 7. File Layout

```
src/
├── main.rs         — entry point; calls MiningPipeline::start(), loops on next_block()
└── pipeline.rs     — MiningPipeline, PipelineHandle, RawJob, PreparedJob, worker tasks
```

`main.rs` is reduced to configuration, GPU device enumeration, and the top-level `run()` loop that submits synthetic (or real) matrices and prints found blocks.

---

## 8. Testing

### Unit tests (no GPU required)

- `submit()` returns `Err(Dropped)` when a second call displaces the first while the preprocessor is idle.
- `submit()` returns `Err(WorkerPanic)` when `MiningPipeline` is started with zero device IDs (no GPU workers, preprocessor detects `receiver_count() == 0`).
- Drop-oldest fires under backpressure: rapid-fire 5 submissions; verify first and last are processed, middle ones receive `Err(Dropped)`.

### Integration tests (mock GPU)

Extract the GPU worker loop body behind a `trait MineFn: FnMut(…) -> Result<(Vec<FoundTile>, Vec<i32>)>`. In tests, supply a mock that returns zero found tiles and a deterministic matrix. Verify:

1. `submit()` returns the correct clean product.
2. Concurrent submissions from two tasks both resolve.
3. A mock that always errors causes `submit()` to return `Err(WorkerPanic)`.

### GPU smoke test (`#[ignore]`, opt-in via `PEARL_GPU_TEST=1`)

End-to-end test through `MiningPipeline::start()` with `device_ids = &[0]`, `difficulty_bits = 0.0` (all tiles win). Asserts found blocks are emitted and `submit()` returns a non-empty clean product.

---

## 9. Out of Scope

- P2P block submission (caller of `next_block()` handles this)
- AI workload FFI / socket integration
- WTEMA-N difficulty adjustment
- Multi-machine coordination
