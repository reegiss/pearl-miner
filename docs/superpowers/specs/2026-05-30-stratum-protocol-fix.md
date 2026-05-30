# Stratum Protocol Fix — Design Spec

**Date:** 2026-05-30  
**Status:** Approved

## Problem

The miner does not appear in the AlphaPool worker dashboard and no shares are
registered as accepted or rejected. Analysis of the reference miner binary
(`alpha` v1.7.6) reveals three root causes:

1. **`mining.notify` not handled.** The pool sends this with a `job_id` before
   each `pearl.challenge`. Without parsing `job_id`, the miner cannot submit
   valid `mining.submit` shares — the message the pool uses to credit workers.

2. **`mining.submit` never sent.** The miner sends only `pearl.challenge_response`.
   The pool dashboard tracks workers exclusively via `mining.submit` shares.
   `pearl.challenge_response` alone is not sufficient for worker visibility.

3. **`mining.set_difficulty` and `pearl.set_mining_params` not handled.**
   The pool sends these on every connection. Ignoring them means the miner
   uses hardcoded params regardless of pool configuration.

## Protocol Flow (reference binary, confirmed)

```
Client → mining.subscribe ["pearl-miner/0.1.0", null, addr, 0]
Client → mining.authorize  ["address.worker", "password"]
  Pool → mining.set_difficulty  params[0]: float difficulty
  Pool → pearl.set_mining_params  { m, n, k, r, tm, tn, ... }
  Pool → mining.notify  [job_id, seed, target, clean_jobs]
  Pool → pearl.challenge  { seed, difficulty }
Client → pearl.challenge_response  { seed, nonce }   ← BLAKE3 answer
Client → mining.submit  ["address.worker", job_id, nonce]  ← share credit ← NEW
  Pool → { id: N, result: true/false, error: ... }   ← accepted/rejected
```

## Solution

### 1. New Incoming Message Handlers (`src/pool.rs`)

Add three new arms to the `match method.as_str()` block inside `connect()`:

**`mining.set_difficulty`**
- Parse `params[0]` as `f64`, cast to `u32`, store as `current_difficulty` local
  to the connection.
- Log `[pool] difficulty set to {n}`.
- If value is invalid, log and ignore.

**`pearl.set_mining_params`**
- Parse params object fields: `m`, `n`, `k`, `r`, `tm`, `tn` (all `usize`).
- Send `Some(MiningParams { sigma: [0u8;32], difficulty: 0, m, n, k, r, tm, tn })`
  via `params_tx` watch channel.
- Log `[pool] mining params: m={m} n={n} k={k} r={r} tm={tm} tn={tn}`.
- If any required field is missing, log and ignore.

**`mining.notify`**
- Parse `params[0]` as `job_id: String`.
- Store as `current_job_id` local to the connection.
- If `params[3]` (clean_jobs) is `true`, cancel current mining via existing
  channel by sending the current challenge again with the new job_id (see §2).
- Log `[pool] job {job_id}`.

### 2. Updated Challenge Channel

Change channel type from `Option<(String, u32)>` to `Option<(String, u32, String)>`
carrying `(seed_hex, difficulty, job_id)`.

When `pearl.challenge` arrives, populate `job_id` from `current_job_id`.
If `current_job_id` is empty (notify not yet received), use empty string.

**Files:** `src/main.rs` (channel declaration), `src/pool.rs` (sender),
`src/miner.rs` (receiver).

### 3. Updated `Submit` Struct

```rust
pub struct Submit {
    pub seed:   String,   // challenge seed (hex)
    pub nonce:  String,   // winning nonce (hex, 16 chars)
    pub job_id: String,   // job_id from last mining.notify
}
```

`src/miner.rs` populates `job_id` from the `job_id` field of the received
challenge tuple. `src/pool.rs` reads it when building the submit messages.

### 4. Dual Share Submission

When `submit_rx` yields a `Submit`, send two messages in sequence:

```
pearl.challenge_response  { "seed": submit.seed, "nonce": submit.nonce }
mining.submit  ["address.worker", submit.job_id, submit.nonce]
```

If `submit.job_id` is empty, skip `mining.submit` and log
`[pool] no job_id yet, skipping mining.submit`.

Track both message IDs in `pending_submits: HashMap<u64, &'static str>`:
- `msg_id` for `pearl.challenge_response` → `"challenge_response"`
- `msg_id + 1` for `mining.submit` → `"submit"`

### 5. Response Logging

When a message arrives with a numeric `id` that is in `pending_submits`:
- `result == true` → `[pool] ✓ {kind} accepted`
- `result == false` or `error != null` → `[pool] ✗ {kind} REJECTED: {error}`
- Remove id from `pending_submits`.

### 6. New Params Channel (wiring in `src/main.rs`)

```rust
let (params_tx, params_rx) = watch::channel(None::<MiningParams>);
```

Passed to `pool::run()` (sender) and `Miner::run()` (receiver).

`Miner::run()`: when a new challenge arrives, check `params_rx.borrow()`.
If `Some(p)`, use `p.m`, `p.n`, `p.k`, `p.r`, `p.tm`, `p.tn` instead of
defaults. If `None`, fall back to `DEFAULT_*` constants.

## Files Changed

| File | Change |
|------|--------|
| `src/pool.rs` | Add handlers for `mining.set_difficulty`, `pearl.set_mining_params`, `mining.notify`; send dual submission; track responses; add `params_tx` param |
| `src/miner.rs` | Update `challenge_rx` type to carry `job_id`; update `Submit` struct; read `job_id` and `params_rx` |
| `src/main.rs` | Add `params_tx`/`params_rx` channel; update wiring |

No changes to CUDA kernels or library crates.

## Expected Outcome

- Worker appears in AlphaPool dashboard within seconds of first BLAKE3 solve.
- Each solved share logs accepted or rejected confirmation.
- Pool-assigned matrix dimensions (`pearl.set_mining_params`) override hardcoded defaults.
