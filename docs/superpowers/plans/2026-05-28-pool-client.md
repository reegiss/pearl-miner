# Pool Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect `pearl-miner` to a mining pool (TCP JSON-RPC), receive live `pearl.challenge` messages to drive `sigma`/`difficulty_bits`, and hot-reload the mining pipeline whenever a new challenge arrives.

**Architecture:** A new `src/pool.rs` exposes `pool_task()` (runs forever, reconnects on drop) and `parse_challenge()` (pure function, fully tested). `main.rs` spawns `pool_task`, waits for the first challenge, then drives a `tokio::select!` loop that restarts the mining pipeline on every new challenge and forwards found blocks to the pool for submission. Block submission is a logged stub — protocol format TBD.

**Tech Stack:** Rust 2021 · `tokio net + io-util` · `serde_json 1`

---

## File Map

| File | Change |
|---|---|
| `Cargo.toml` | Add `serde_json = "1"`; add `net`, `io-util` to tokio features |
| `src/pool.rs` | New — `PoolChallenge`, `parse_challenge()`, `pool_task()`, `submit_block()` stub |
| `src/main.rs` | Modified — `mod pool`, channels, `make_config()`, `select!` loop |

---

### Task 1: Add dependencies

**Files:**
- Modify: `Cargo.toml`

- [ ] **Step 1: Update `Cargo.toml`**

Change the tokio line and add serde_json:

```toml
tokio      = { version = "1", features = ["rt-multi-thread", "macros", "sync", "time", "net", "io-util"] }
serde_json = "1"
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check`

Expected: `Finished dev profile` with 0 errors.

- [ ] **Step 3: Commit**

```bash
git add Cargo.toml
git commit -m "chore: add serde_json and tokio net/io-util for pool client"
```

---

### Task 2: `parse_challenge()` — pure parsing, fully tested

**Files:**
- Create: `src/pool.rs`
- Modify: `src/main.rs` (add `mod pool;`)

`parse_challenge` takes a raw JSON line and returns `Some(PoolChallenge)` only when `method == "pearl.challenge"` and both fields decode successfully. All other inputs return `None` — never panics.

- [ ] **Step 1: Write failing tests**

Create `/home/regis/develop/pearl-miner/src/pool.rs` with just the test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    const VALID_LINE: &str = concat!(
        r#"{"id":null,"method":"pearl.challenge","params":"#,
        r#"{"seed":"a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90","difficulty":32}}"#
    );

    #[test]
    fn test_parse_valid_challenge() {
        let ch = parse_challenge(VALID_LINE).unwrap();
        assert_eq!(ch.difficulty, 32.0);
        assert_eq!(ch.seed[0],  0xa1);
        assert_eq!(ch.seed[31], 0x90);
    }

    #[test]
    fn test_parse_wrong_method() {
        let line = r#"{"id":null,"method":"other.method","params":{"seed":"aa","difficulty":1}}"#;
        assert!(parse_challenge(line).is_none());
    }

    #[test]
    fn test_parse_invalid_json() {
        assert!(parse_challenge("not json").is_none());
    }

    #[test]
    fn test_parse_bad_hex_seed() {
        let line = r#"{"id":null,"method":"pearl.challenge","params":{"seed":"zzzz","difficulty":1}}"#;
        assert!(parse_challenge(line).is_none());
    }

    #[test]
    fn test_parse_wrong_seed_length() {
        // 4 bytes instead of 32
        let line = r#"{"id":null,"method":"pearl.challenge","params":{"seed":"deadbeef","difficulty":1}}"#;
        assert!(parse_challenge(line).is_none());
    }

    #[test]
    fn test_parse_missing_difficulty() {
        let line = r#"{"id":null,"method":"pearl.challenge","params":{"seed":"a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90"}}"#;
        assert!(parse_challenge(line).is_none());
    }
}
```

- [ ] **Step 2: Add `mod pool;` to `src/main.rs`**

Add `mod pool;` as the third line of `src/main.rs` (after `mod pipeline;` and `mod args;`).

- [ ] **Step 3: Run to verify tests fail**

Run: `cargo test -p pearl-miner pool::tests::test_parse 2>&1 | head -20`

Expected: compile error — `parse_challenge` not found.

- [ ] **Step 4: Implement `PoolChallenge` and `parse_challenge()`**

Prepend this before the `#[cfg(test)]` block in `src/pool.rs`:

```rust
use tokio::sync::{mpsc, watch};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::net::TcpStream;
use std::time::Duration;
use pearl_block::PearlBlock;

#[derive(Clone, Debug, PartialEq)]
pub struct PoolChallenge {
    pub seed:       [u8; 32],
    pub difficulty: f64,
}

fn decode_hex32(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 { return None; }
    let mut out = [0u8; 32];
    for (i, chunk) in s.as_bytes().chunks(2).enumerate() {
        let hi = (chunk[0] as char).to_digit(16)? as u8;
        let lo = (chunk[1] as char).to_digit(16)? as u8;
        out[i] = (hi << 4) | lo;
    }
    Some(out)
}

/// Parse a pool JSON-RPC line into a PoolChallenge.
/// Returns None for any unknown method, bad JSON, or malformed fields.
pub fn parse_challenge(line: &str) -> Option<PoolChallenge> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    if v["method"].as_str()? != "pearl.challenge" {
        return None;
    }
    let params     = &v["params"];
    let seed_hex   = params["seed"].as_str()?;
    let difficulty = params["difficulty"].as_f64()?;
    let seed       = decode_hex32(seed_hex)?;
    Some(PoolChallenge { seed, difficulty })
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p pearl-miner pool::tests::test_parse`

Expected: 6 tests, all `ok`.

- [ ] **Step 6: Commit**

```bash
git add src/pool.rs src/main.rs
git commit -m "feat(pool): add PoolChallenge and parse_challenge()"
```

---

### Task 3: `pool_task()` — TCP connection and reconnect

**Files:**
- Modify: `src/pool.rs`

`pool_task_inner` is the testable core; `pool_task` is the public entry point with production defaults. Tests use `pool_task_inner` with a 50 ms initial backoff so reconnect tests complete in < 500 ms.

- [ ] **Step 1: Add integration tests to `src/pool.rs`**

Add these tests inside the existing `mod tests` block (after the parse tests):

```rust
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;
    use tokio::sync::mpsc;

    const CHALLENGE_LINE_A: &str = concat!(
        "{\"id\":null,\"method\":\"pearl.challenge\",\"params\":",
        "{\"seed\":\"a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90\",",
        "\"difficulty\":32}}\n"
    );

    const CHALLENGE_LINE_B: &str = concat!(
        "{\"id\":null,\"method\":\"pearl.challenge\",\"params\":",
        "{\"seed\":\"b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90a1\",",
        "\"difficulty\":64}}\n"
    );

    #[tokio::test]
    async fn test_pool_task_receives_challenge() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr     = listener.local_addr().unwrap();

        let (challenge_tx, mut challenge_rx) = watch::channel(None::<PoolChallenge>);
        let (_block_tx, block_rx) = mpsc::channel::<PearlBlock>(4);

        tokio::spawn(pool_task_inner(
            addr.to_string(), "wallet".into(),
            challenge_tx, block_rx,
            Duration::from_millis(50),
        ));

        let (mut stream, _) = listener.accept().await.unwrap();
        stream.write_all(CHALLENGE_LINE_A.as_bytes()).await.unwrap();

        tokio::time::timeout(
            Duration::from_millis(500),
            challenge_rx.changed(),
        ).await.expect("timed out").unwrap();

        let ch = challenge_rx.borrow().clone().unwrap();
        assert_eq!(ch.difficulty, 32.0);
        assert_eq!(ch.seed[0], 0xa1);
    }

    #[tokio::test]
    async fn test_pool_task_ignores_unknown_method() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr     = listener.local_addr().unwrap();

        let (challenge_tx, mut challenge_rx) = watch::channel(None::<PoolChallenge>);
        let (_block_tx, block_rx) = mpsc::channel::<PearlBlock>(4);

        tokio::spawn(pool_task_inner(
            addr.to_string(), "wallet".into(),
            challenge_tx, block_rx,
            Duration::from_millis(50),
        ));

        let (mut stream, _) = listener.accept().await.unwrap();
        // Send an unknown method, then a valid challenge
        stream.write_all(b"{\"method\":\"unknown\"}\n").await.unwrap();
        stream.write_all(CHALLENGE_LINE_A.as_bytes()).await.unwrap();

        tokio::time::timeout(
            Duration::from_millis(500),
            challenge_rx.changed(),
        ).await.expect("timed out").unwrap();

        assert!(challenge_rx.borrow().is_some());
    }

    #[tokio::test]
    async fn test_pool_task_reconnects_after_disconnect() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr     = listener.local_addr().unwrap();

        let (challenge_tx, mut challenge_rx) = watch::channel(None::<PoolChallenge>);
        let (_block_tx, block_rx) = mpsc::channel::<PearlBlock>(4);

        tokio::spawn(pool_task_inner(
            addr.to_string(), "wallet".into(),
            challenge_tx, block_rx,
            Duration::from_millis(50),
        ));

        // First connection: accept and immediately drop (simulate server restart)
        let (stream1, _) = listener.accept().await.unwrap();
        drop(stream1);

        // Second connection: send a challenge
        let (mut stream2, _) = listener.accept().await.unwrap();
        stream2.write_all(CHALLENGE_LINE_B.as_bytes()).await.unwrap();

        tokio::time::timeout(
            Duration::from_millis(500),
            challenge_rx.changed(),
        ).await.expect("timed out after reconnect").unwrap();

        let ch = challenge_rx.borrow().clone().unwrap();
        assert_eq!(ch.difficulty, 64.0);
    }
```

- [ ] **Step 2: Run to verify tests fail**

Run: `cargo test -p pearl-miner pool::tests::test_pool_task 2>&1 | head -20`

Expected: compile error — `pool_task_inner` not found.

- [ ] **Step 3: Implement `pool_task_inner()`, `pool_task()`, and `submit_block()`**

Add this before the `#[cfg(test)]` block in `src/pool.rs` (after the existing `parse_challenge` code):

```rust
fn fmt_hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

async fn submit_block(block: &PearlBlock) {
    // Protocol format TBD — log until confirmed
    let id = pearl_block::block_identity(block);
    eprintln!("[pool] block found — submission pending protocol spec (identity={})",
        fmt_hex(&id));
}

/// Testable core — takes configurable initial_backoff so tests complete quickly.
pub async fn pool_task_inner(
    host_port:       String,
    wallet:          String,
    challenge_tx:    watch::Sender<Option<PoolChallenge>>,
    mut block_rx:    mpsc::Receiver<PearlBlock>,
    initial_backoff: Duration,
) {
    let _ = wallet; // used when pearl.submit is implemented
    let mut backoff = initial_backoff;

    loop {
        match TcpStream::connect(&host_port).await {
            Ok(stream) => {
                eprintln!("[pool] connected to {host_port}");
                backoff = initial_backoff; // reset on success

                let mut lines = BufReader::new(stream).lines();

                loop {
                    tokio::select! {
                        result = lines.next_line() => {
                            match result {
                                Ok(Some(line)) => {
                                    if let Some(ch) = parse_challenge(&line) {
                                        eprintln!("[pool] challenge seed={} difficulty={}",
                                            fmt_hex(&ch.seed), ch.difficulty);
                                        challenge_tx.send_replace(Some(ch));
                                    }
                                    // unknown method: silently ignore
                                }
                                _ => break, // EOF or error → reconnect
                            }
                        }
                        Some(block) = block_rx.recv() => {
                            submit_block(&block).await;
                        }
                    }
                }

                eprintln!("[pool] disconnected from {host_port}");
            }
            Err(e) => {
                eprintln!("[pool] connect failed ({host_port}): {e}");
            }
        }

        eprintln!("[pool] reconnecting in {}ms…", backoff.as_millis());
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(Duration::from_secs(60));
    }
}

/// Public entry point with production backoff (5 s initial, max 60 s).
pub async fn pool_task(
    host_port:    String,
    wallet:       String,
    challenge_tx: watch::Sender<Option<PoolChallenge>>,
    block_rx:     mpsc::Receiver<PearlBlock>,
) {
    pool_task_inner(host_port, wallet, challenge_tx, block_rx,
                   Duration::from_secs(5)).await
}
```

- [ ] **Step 4: Run all pool tests to verify they pass**

Run: `cargo test -p pearl-miner pool::tests`

Expected: 9 tests, all `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/pool.rs
git commit -m "feat(pool): add pool_task with TCP connection and reconnect logic"
```

---

### Task 4: Integrate pool client into `main.rs`

**Files:**
- Modify: `src/main.rs`

Replace the hardcoded `MiningConfig` and simple loop with a `select!` loop driven by pool challenges.

- [ ] **Step 1: Replace `src/main.rs`**

```rust
mod pipeline;
mod args;
mod pool;

use std::sync::Arc;
use anyhow::Result;
use clap::Parser;
use pearl_types::{MatrixParams, MiningConfig};
use pipeline::{MiningPipeline, PipelineError};
use args::Args;
use pool::{pool_task, PoolChallenge};
use tokio::sync::mpsc;

fn make_config(challenge: &PoolChallenge, mu: &[u8]) -> Arc<MiningConfig> {
    Arc::new(MiningConfig {
        params: MatrixParams { m: 32, n: 32, k: 512, r: 32, tm: 4, tn: 4 },
        difficulty_bits: challenge.difficulty,
        sigma: challenge.seed.to_vec(),
        mu: mu.to_vec(),
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let mu = args.wallet_bytes()
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    args.validate_pool()
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    eprintln!("[init] wallet: {}", args.wallet);
    eprintln!("[init] pool:   {}", args.pool);

    let n_devices = pearl_gpu::device_count();
    if n_devices == 0 {
        eprintln!("[init] no CUDA devices found — exiting");
        return Ok(());
    }
    let device_ids: Vec<u32> = (0..n_devices as u32).collect();
    eprintln!("[init] detected {} CUDA device(s): {:?}", n_devices, device_ids);

    // Pool channels
    let (challenge_tx, mut challenge_rx) = tokio::sync::watch::channel(None::<PoolChallenge>);
    let (block_tx, block_rx)             = mpsc::channel::<pearl_block::PearlBlock>(32);

    tokio::spawn(pool_task(
        args.pool.clone(),
        args.wallet.clone(),
        challenge_tx,
        block_rx,
    ));

    // Wait for the first challenge before starting the pipeline
    eprintln!("[pool] waiting for first challenge...");
    challenge_rx.changed().await
        .map_err(|_| anyhow::anyhow!("pool task exited before first challenge"))?;
    let first = challenge_rx.borrow().clone().unwrap();
    eprintln!("[pool] first challenge — seed={} difficulty={}",
        hex(&first.seed), first.difficulty);

    // Synthetic matrices (in production these come from the AI workload)
    let p = MatrixParams { m: 32, n: 32, k: 512, r: 32, tm: 4, tn: 4 };
    let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);
    let a: Arc<[i8]> = (0..m * k)
        .map(|i| ((i * 7 + 3) % 128) as i8 - 64)
        .collect::<Vec<_>>().into();
    let b: Arc<[i8]> = (0..k * n)
        .map(|i| ((i * 11 + 5) % 128) as i8 - 64)
        .collect::<Vec<_>>().into();
    eprintln!("[data] generated {}×{} A and {}×{} B (INT8)", m, k, k, n);

    let config = make_config(&first, &mu);
    let (mut pipeline, mut handle, mut blocks) =
        MiningPipeline::start(Arc::clone(&config), &device_ids);

    loop {
        tokio::select! {
            // New challenge → restart pipeline with updated sigma/difficulty
            Ok(_) = challenge_rx.changed() => {
                let ch = challenge_rx.borrow().clone().unwrap();
                eprintln!("[pool] new challenge — seed={} difficulty={}",
                    hex(&ch.seed), ch.difficulty);
                let new_config = make_config(&ch, &mu);
                drop(pipeline);
                (pipeline, handle, blocks) =
                    MiningPipeline::start(Arc::clone(&new_config), &device_ids);
            }

            // Block found → forward to pool_task for submission
            Some(block) = blocks.next() => {
                let _ = block_tx.send(block).await;
            }

            // Mining pipeline result (clean A·B)
            result = handle.submit(Arc::clone(&a), Arc::clone(&b)) => {
                match result {
                    Ok(clean_ab) => {
                        eprintln!("[peel] recovered {}×{} product ({} elements)",
                            m, n, clean_ab.len());
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

    Ok(())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check`

Expected: 0 errors.

- [ ] **Step 3: Run all workspace tests**

Run: `cargo test --workspace --exclude pearl-gpu 2>&1 | tail -15`

Expected: all tests pass (9 pool + 7 args + 12 pipeline + …).

- [ ] **Step 4: Commit**

```bash
git add src/main.rs
git commit -m "feat: integrate pool client — live challenge drives pipeline hot-reload"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `serde_json`, tokio `net`/`io-util` | Task 1 |
| `PoolChallenge { seed, difficulty }` | Task 2 |
| `parse_challenge()` — valid / wrong method / bad JSON / bad hex / wrong length / missing field | Task 2 |
| `pool_task()` public entry point (5s initial backoff) | Task 3 |
| `pool_task_inner()` testable with configurable backoff | Task 3 |
| TCP connect + `BufReader::lines()` read loop | Task 3 |
| `challenge_tx.send_replace()` on valid challenge | Task 3 |
| Silently ignore unknown methods | Task 3 |
| EOF / error → reconnect | Task 3 |
| Reconnect backoff: double per failure, cap 60s, reset on success | Task 3 |
| `submit_block()` stub — logs identity, no network | Task 3 |
| `block_rx` consumed in `pool_task_inner` select! | Task 3 |
| `make_config()` uses `challenge.seed` as `sigma` and `challenge.difficulty` as `difficulty_bits` | Task 4 |
| Spawn `pool_task` before waiting for first challenge | Task 4 |
| Wait for first challenge before starting pipeline | Task 4 |
| `select!` loop: challenge branch restarts pipeline | Task 4 |
| `select!` loop: blocks branch forwards to `block_tx` | Task 4 |
| `select!` loop: submit branch handles pipeline result | Task 4 |
| Tests: mock TCP server for challenge reception | Task 3 |
| Tests: reconnect after server drops connection | Task 3 |
| Tests: unknown method ignored | Task 3 |

**No placeholders.** All code is complete. `submit_block` stub is intentional and explicitly described.

**Type consistency:**
- `PoolChallenge` defined in Task 2 → used in Task 3 (`challenge_tx: watch::Sender<Option<PoolChallenge>>`) → used in Task 4 (`make_config(&ch, &mu)`) ✓
- `pool_task_inner` signature in Task 3 → called in test with 5-arg form ✓
- `pool_task` 4-arg public wrapper in Task 3 → called in Task 4 main.rs ✓
- `make_config(challenge: &PoolChallenge, mu: &[u8]) -> Arc<MiningConfig>` defined and used consistently ✓
