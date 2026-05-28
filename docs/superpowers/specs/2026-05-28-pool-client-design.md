# Pearl Miner — Pool Client

**Date:** 2026-05-28  
**Status:** Approved  
**Scope:** `src/pool.rs` + `src/main.rs` — TCP connection to mining pool, challenge reception, pipeline hot-reload on new challenge, block submission stub.

---

## 1. Overview

The miner currently runs with hardcoded `sigma` and `difficulty_bits`. This design connects the miner to the pool at `us1.alphapool.tech:5566`, receives live challenges (seed + difficulty), and dynamically restarts the mining pipeline whenever a new challenge arrives. Block submission is a stub pending protocol specification confirmation.

---

## 2. Pool Protocol (observed)

Transport: TCP, newline-delimited JSON (one JSON object per line).

**Server → Client** (sent immediately on connect, repeated on new blocks):
```json
{"id":null,"method":"pearl.challenge","params":{"seed":"a1b2c3...","difficulty":32}}
```
- `seed`: hex-encoded 32-byte string — becomes `sigma` in `MiningConfig`
- `difficulty`: integer — becomes `difficulty_bits` as `f64`

**Client → Server** (`pearl.submit`): format TBD — stub only in this implementation.

---

## 3. File Layout

```
src/
├── pool.rs    — new: PoolChallenge, pool_task(), submit_block() stub
└── main.rs    — modified: start pool_task, watch channel, select! loop
```

---

## 4. `src/pool.rs`

### Types

```rust
#[derive(Clone, Debug)]
pub struct PoolChallenge {
    pub seed:       [u8; 32],
    pub difficulty: f64,
}
```

### `pool_task()`

```rust
pub async fn pool_task(
    host_port:    String,
    wallet:       String,
    challenge_tx: watch::Sender<Option<PoolChallenge>>,
    mut block_rx: mpsc::Receiver<PearlBlock>,
)
```

Runs forever in a loop:

```
loop:
  TcpStream::connect(&host_port)
  → Ok(stream):
      let reader = BufReader::new(stream);
      loop:
        read_line() → EOF/err → break (reconnect)
        parse JSON → method == "pearl.challenge" →
            decode seed hex (32 bytes) →
                Ok  → challenge_tx.send_replace(Some(PoolChallenge { seed, difficulty }))
                Err → log and skip (don't disconnect)
            other method → ignore
  → Err(e): log "pool connect failed: {e}"
  sleep(backoff); backoff = min(backoff * 2, 60s)  // reset to 5s on success
```

**Reconnect backoff:** starts at 5s, doubles on each failed attempt, caps at 60s. Resets to 5s on a successful connection.

Block submission (stub):
```rust
async fn submit_block(block: &PearlBlock, wallet: &str, seed: &[u8; 32]) {
    // TODO: implement pearl.submit once protocol format is confirmed
    let id = pearl_block::block_identity(block);
    eprintln!("[pool] block found, submission pending protocol spec (identity={})", hex(id));
}
```

The `block_rx` is consumed inside `pool_task` using `tokio::select!` so submissions can be attempted concurrently with reading challenges.

---

## 5. Integration in `main.rs`

### Startup

```rust
let (challenge_tx, mut challenge_rx) = watch::channel(None::<PoolChallenge>);
let (block_tx, block_rx)             = mpsc::channel::<PearlBlock>(32);

tokio::spawn(pool::pool_task(
    args.pool.clone(),
    args.wallet.clone(),
    challenge_tx,
    block_rx,
));

// Wait for first challenge before starting the pipeline
eprintln!("[pool] waiting for first challenge...");
challenge_rx.changed().await
    .map_err(|_| anyhow::anyhow!("pool task exited before first challenge"))?;
let first_challenge = challenge_rx.borrow().clone().unwrap();
eprintln!("[pool] first challenge — seed={} difficulty={}",
    hex(&first_challenge.seed), first_challenge.difficulty);
```

### `make_config()` helper

```rust
fn make_config(challenge: &PoolChallenge, mu: &[u8]) -> Arc<MiningConfig> {
    Arc::new(MiningConfig {
        params: MatrixParams { m: 32, n: 32, k: 512, r: 32, tm: 4, tn: 4 },
        difficulty_bits: challenge.difficulty,
        sigma: challenge.seed.to_vec(),
        mu: mu.to_vec(),
    })
}
```

### Main `select!` loop

```rust
let config = make_config(&first_challenge, &mu);
let (mut pipeline, mut handle, mut blocks) =
    MiningPipeline::start(Arc::clone(&config), &device_ids);

let a: Arc<[i8]> = /* synthetic test matrices */;
let b: Arc<[i8]> = /* synthetic test matrices */;

loop {
    tokio::select! {
        // New challenge → restart pipeline with updated config
        Ok(_) = challenge_rx.changed() => {
            let ch = challenge_rx.borrow().clone().unwrap();
            eprintln!("[pool] new challenge — seed={} difficulty={}",
                hex(&ch.seed), ch.difficulty);
            let new_config = make_config(&ch, &mu);
            drop(pipeline);
            (pipeline, handle, blocks) =
                MiningPipeline::start(Arc::clone(&new_config), &device_ids);
        }

        // Block found → forward to pool task for submission
        Some(block) = blocks.next() => {
            let _ = block_tx.send(block).await;
        }

        // Pipeline result (clean A·B returned)
        result = handle.submit(Arc::clone(&a), Arc::clone(&b)) => {
            match result {
                Ok(clean_ab) => {
                    eprintln!("[peel] recovered {}×{} product ({} elements)",
                        32, 32, clean_ab.len());
                }
                Err(PipelineError::Dropped)     => {}
                Err(PipelineError::WorkerPanic) => {
                    eprintln!("[submit] GPU workers exited — shutting down");
                    break;
                }
            }
        }
    }
}
```

---

## 6. Error Handling

| Situation | Behavior |
|---|---|
| TCP connect fails | Log error, backoff (5s → 60s), retry |
| Connection drops mid-stream | EOF from `read_line` → break inner loop → reconnect with backoff |
| `seed` hex decode fails | Log warning, skip message, stay connected |
| `seed` wrong length (≠ 32 bytes) | Log warning, skip message, stay connected |
| Pool sends unknown method | Silently ignore |
| `challenge_rx.changed()` error (pool task exited) | Main loop continues mining on last known config |
| No challenge received in first 30s | `changed().await` completes once any challenge arrives (no timeout in this impl) |

---

## 7. Dependencies

No new crates needed — `tokio` already has `net`, `io`, `sync`, and `time` features enabled.

---

## 8. Testing

- `pool_task` with a mock TCP server that sends one `pearl.challenge` → verify `challenge_rx` fires with correct seed and difficulty.
- `pool_task` with a server that closes connection immediately → verify it reconnects.
- `pool_task` with a server that sends an invalid seed (bad hex) → verify it skips and stays connected.
- `make_config()` uses `challenge.seed` as `sigma` and `challenge.difficulty` as `difficulty_bits`.

---

## 9. Out of Scope

- `pearl.submit` implementation (protocol format TBD)
- Pool authentication / worker name
- Multiple pool failover
- Pool latency metrics
