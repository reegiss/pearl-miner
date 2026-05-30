# Pool Protocol Fix — Design Spec

**Date:** 2026-05-30  
**Status:** Approved

## Problem

Two issues prevent the miner from registering shares:

1. **Wrong nonce format**: `pearl.challenge_response` submits the PoUW tile's
   `BLAKE3(M, key=sA)` hash as the `nonce`. The pool expects a simple proof-of-work:
   find `N` (u64) such that `BLAKE3(seed_bytes ++ N.to_le_bytes()) ≤ 2^(256−difficulty)`.
   Confirmed by running a CPU brute-force search: nonce=7419332574 found after 7.4B iters
   with hash ending in 4 zero bytes (difficulty=32 satisfied).

2. **Missing hashrate display**: `LOG_INTERVAL_S=5.0` never fires because the pool sends
   new challenges faster than 5 seconds, resetting the mining loop before the timer expires.

## Solution

### BLAKE3 Challenge Solver

A new function `solve_blake3_challenge(seed, difficulty, cancel)` in `src/miner.rs`:

- Input: 32-byte seed, difficulty, cancel AtomicBool
- Algorithm: iterate `nonce = 0, 1, 2, …`; compute `BLAKE3(seed ++ nonce.to_le_bytes())`;
  check `hash[28..32] == [0,0,0,0]` for difficulty=32 (general: top `difficulty` bits of
  LE uint256 must be zero)
- Parallelism: rayon `par_iter` over chunks of 1M nonces → uses all CPU cores
- Cancel: checks `cancel.load(Relaxed)` inside each chunk; returns `None` if cancelled

Nonce sent to pool: `format!("{:016x}", nonce)` (big-endian 16-char hex string).

Spawned as a `tokio::task::spawn_blocking` thread alongside GPU mining threads, sharing
the same cancel `Arc<AtomicBool>`. Sends found nonce via `submit_tx` (same channel).

### PoUW Tile Submission

Remove `submit_tx.blocking_send()` from `mining_loop` — tile hashes are not valid challenge
responses and caused pool rejections. Found tiles are logged as `[PoUW]` for debugging only.

### Hashrate Display

Change log trigger from `elapsed >= LOG_INTERVAL_S` to `job % 100 == 0`. This fires
regardless of challenge rotation speed, producing output every ~2-5 seconds at typical
job rates.

## Dependencies

Add `rayon = "1"` to workspace root `Cargo.toml`.

## Files Changed

- `Cargo.toml` — add rayon dependency
- `src/miner.rs` — add solver function, fix hashrate trigger, remove tile submit
