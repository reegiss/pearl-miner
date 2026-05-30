use pearl_gpu::GpuMiner;
use pearl_types::MiningParams;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, watch};
use tokio::time::sleep;

#[derive(Deserialize, Debug)]
struct PoolMessage {
    #[serde(default)]
    id:     serde_json::Value,
    method: Option<String>,
    params: Option<serde_json::Value>,
    result: Option<serde_json::Value>,
    error:  Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ChallengeParams {
    seed:       String,
    difficulty: u32,
}

/// A solved PoUW block the miner wants to submit.
#[derive(Clone, Debug)]
pub struct Submit {
    pub worker:  String,
    pub job_id:  String,
    pub payload: String, // base64-encoded share
}

/// Connect to the pool, handle the full Stratum handshake, forward PoUW jobs
/// to the miner via `challenge_tx`, and forward share submissions from `submit_rx`.
/// `gpu` is used for GPU-accelerated BLAKE3 pool challenge solving.
/// Reconnects on drop or error.
pub async fn run(
    addr:          &str,
    wallet:        &str,
    password:      &str,
    n_gpus:        usize,
    gpu:           Arc<GpuMiner>,
    challenge_tx:  watch::Sender<Option<(String, u32, String)>>,
    params_tx:     watch::Sender<Option<MiningParams>>,
    mut submit_rx: mpsc::Receiver<Submit>,
) {
    loop {
        println!("[pool] Connecting to {}...", addr);
        match connect(addr, wallet, password, n_gpus, Arc::clone(&gpu), &challenge_tx, &params_tx, &mut submit_rx).await {
            Ok(())  => println!("[pool] Connection closed by server."),
            Err(e)  => eprintln!("[pool] Connection error: {e}"),
        }
        sleep(Duration::from_secs(5)).await;
    }
}

async fn connect(
    addr:          &str,
    wallet:        &str,
    password:      &str,
    n_gpus:        usize,
    gpu:           Arc<GpuMiner>,
    challenge_tx:  &watch::Sender<Option<(String, u32, String)>>,
    params_tx:     &watch::Sender<Option<MiningParams>>,
    submit_rx:     &mut mpsc::Receiver<Submit>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let stream = TcpStream::connect(addr).await?;
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();

    // Worker base: wallet must contain a dot for pool to register it
    let worker_base = if wallet.contains('.') {
        wallet.to_string()
    } else {
        format!("{}.worker1", wallet)
    };

    // ────────────────────────────────────────────────────────────────────────
    // STEP 1 — wait for initial pearl.challenge (pool sends this immediately)
    // ────────────────────────────────────────────────────────────────────────
    println!("[pool] Connected. Waiting for initial pearl.challenge...");
    let initial_challenge = loop {
        let Some(line) = lines.next_line().await? else {
            return Err("connection closed before challenge".into());
        };
        if line.trim().is_empty() { continue; }
        println!("[pool] <- {}", line);
        let Ok(msg) = serde_json::from_str::<PoolMessage>(&line) else { continue };
        if msg.method.as_deref() == Some("pearl.challenge") {
            if let Some(p) = msg.params {
                if let Ok(ch) = serde_json::from_value::<ChallengeParams>(p) {
                    break ch;
                }
            }
        }
    };

    // ────────────────────────────────────────────────────────────────────────
    // STEP 2 — solve the BLAKE3 challenge using GPU
    // ────────────────────────────────────────────────────────────────────────
    let seed_hex   = initial_challenge.seed.clone();
    let difficulty = initial_challenge.difficulty;
    println!("[pool] Solving initial challenge diff={} on GPU...", difficulty);

    let sigma = hex_to_32bytes(&seed_hex)
        .ok_or_else(|| format!("bad seed hex: {}", seed_hex))?;

    let gpu_c = Arc::clone(&gpu);
    let nonce = tokio::task::spawn_blocking(move || {
        gpu_c.solve_pool_challenge(&sigma, difficulty, &AtomicBool::new(false))
    })
    .await?
    .map_err(|e| format!("GPU solver error: {e}"))?
    .ok_or("failed to solve initial challenge")?;

    let nonce_hex = format!("{:016x}", nonce);
    println!("[pool] Initial challenge solved: nonce={}", nonce_hex);

    // ────────────────────────────────────────────────────────────────────────
    // STEP 3 — send pearl.challenge_response (id=1)
    // ────────────────────────────────────────────────────────────────────────
    let cr = serde_json::json!({
        "id": 1,
        "method": "pearl.challenge_response",
        "params": { "seed": &seed_hex, "nonce": &nonce_hex }
    });
    writer.write_all(format!("{}\n", cr).as_bytes()).await?;
    println!("[pool] -> pearl.challenge_response nonce={}", nonce_hex);

    // wait for acceptance (id=1 result)
    loop {
        let Some(line) = lines.next_line().await? else { return Err("closed awaiting cr ack".into()); };
        if line.trim().is_empty() { continue; }
        println!("[pool] <- {}", line);
        let Ok(msg) = serde_json::from_str::<PoolMessage>(&line) else { continue };
        if msg.id == serde_json::json!(1) || msg.id == serde_json::json!("1") {
            if msg.result.as_ref().and_then(|r| r.as_bool()) == Some(true) {
                println!("[pool] Challenge accepted!");
            } else {
                let err = msg.error.map(|e| e.to_string()).unwrap_or_default();
                return Err(format!("challenge rejected: {}", err).into());
            }
            break;
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // STEP 4 — mining.configure (id=2)
    // ────────────────────────────────────────────────────────────────────────
    let configure = serde_json::json!({
        "id": 2,
        "method": "mining.configure",
        "params": [["pearl/v1"], {}]
    });
    writer.write_all(format!("{}\n", configure).as_bytes()).await?;
    println!("[pool] -> mining.configure");

    loop {
        let Some(line) = lines.next_line().await? else { return Err("closed awaiting configure ack".into()); };
        if line.trim().is_empty() { continue; }
        println!("[pool] <- {}", line);
        let Ok(msg) = serde_json::from_str::<PoolMessage>(&line) else { continue };
        if msg.id == serde_json::json!(2) || msg.id == serde_json::json!("2") {
            println!("[pool] Configure confirmed.");
            break;
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // STEP 5 — mining.subscribe (id=3)
    // ────────────────────────────────────────────────────────────────────────
    let subscribe = serde_json::json!({
        "id": 3,
        "method": "mining.subscribe",
        "params": ["pearl-miner/0.1.0"]
    });
    writer.write_all(format!("{}\n", subscribe).as_bytes()).await?;
    println!("[pool] -> mining.subscribe");

    // wait for subscribe result, then pearl.set_mining_params
    let mut got_subscribe = false;
    let mut got_params    = false;
    while !got_subscribe || !got_params {
        let Some(line) = lines.next_line().await? else { return Err("closed during subscribe phase".into()); };
        if line.trim().is_empty() { continue; }
        println!("[pool] <- {}", line);
        let Ok(msg) = serde_json::from_str::<PoolMessage>(&line) else { continue };

        if (msg.id == serde_json::json!(3) || msg.id == serde_json::json!("3")) && !got_subscribe {
            println!("[pool] Subscription confirmed.");
            got_subscribe = true;
            continue;
        }

        if msg.method.as_deref() == Some("pearl.set_mining_params") && !got_params {
            if let Some(mp) = parse_mining_params(&msg.params) {
                println!("[pool] mining params: m={} n={} k={} r={} tm={} tn={}",
                    mp.m, mp.n, mp.k, mp.r, mp.tm, mp.tn);
                let _ = params_tx.send(Some(mp));
            } else {
                eprintln!("[pool] malformed pearl.set_mining_params");
            }
            got_params = true;
            continue;
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // STEP 6 — mining.authorize per GPU (ids 4, 5, ...)
    // ────────────────────────────────────────────────────────────────────────
    let n_gpus_auth = n_gpus.max(1);
    for gpu_i in 0..n_gpus_auth {
        let worker = format!("{}.gpu{}", worker_base, gpu_i);
        let auth = serde_json::json!({
            "id": 4 + gpu_i,
            "method": "mining.authorize",
            "params": [&worker, password]
        });
        writer.write_all(format!("{}\n", auth).as_bytes()).await?;
        println!("[pool] -> mining.authorize for {}", worker);
    }

    // wait for all authorize results + mining.set_difficulty + mining.notify
    let mut auths_left      = n_gpus_auth;
    let mut pouw_difficulty = 32u32;
    let mut got_notify      = false;

    while auths_left > 0 || !got_notify {
        let Some(line) = lines.next_line().await? else { return Err("closed during authorize phase".into()); };
        if line.trim().is_empty() { continue; }
        println!("[pool] <- {}", line);
        let Ok(msg) = serde_json::from_str::<PoolMessage>(&line) else { continue };

        // authorize responses
        if let Some(id) = msg.id.as_u64() {
            if (4..4 + n_gpus_auth as u64).contains(&id) {
                if msg.result.as_ref().and_then(|r| r.as_bool()) == Some(true) {
                    println!("[pool] GPU {} authorized.", id - 4);
                } else {
                    eprintln!("[pool] GPU {} authorize FAILED: {:?}", id - 4, msg.error);
                }
                auths_left = auths_left.saturating_sub(1);
                continue;
            }
        }

        let Some(method) = msg.method else { continue };
        match method.as_str() {
            "mining.set_difficulty" => {
                if let Some(d) = msg.params.as_ref().and_then(|p| p.get(0)).and_then(|v| v.as_u64()) {
                    pouw_difficulty = d as u32;
                    println!("[pool] PoUW difficulty set to {}", pouw_difficulty);
                }
            }
            "mining.notify" => {
                if let Some((job_id, sigma_hex)) = parse_notify(&msg.params) {
                    println!("[pool] Job {} sigma={:.16}... diff={}",
                        job_id, sigma_hex, pouw_difficulty);
                    let _ = challenge_tx.send(Some((sigma_hex, pouw_difficulty, job_id)));
                    got_notify = true;
                }
            }
            "pearl.set_mining_params" => {
                if let Some(mp) = parse_mining_params(&msg.params) {
                    let _ = params_tx.send(Some(mp));
                }
            }
            _ => {}
        }
    }

    println!("[pool] Handshake complete. Mining...");

    // ────────────────────────────────────────────────────────────────────────
    // MAIN LOOP
    // ────────────────────────────────────────────────────────────────────────
    let mut msg_id = 4 + n_gpus_auth as u64;
    let mut current_job_id = String::new();
    let mut pending: HashMap<u64, &'static str> = HashMap::new();

    // Channel for BLAKE3 solver results: (seed_hex, nonce_hex)
    let (solved_tx, mut solved_rx) = mpsc::channel::<(String, String)>(8);
    // Cancel token for the currently running BLAKE3 solver
    let mut blake3_cancel = Arc::new(AtomicBool::new(false));

    loop {
        tokio::select! {
            // ── Incoming from pool ──────────────────────────────────────────
            line = lines.next_line() => {
                let Some(line) = line? else { break; };
                if line.trim().is_empty() { continue; }
                println!("[pool] <- {}", line);

                let Ok(msg) = serde_json::from_str::<PoolMessage>(&line) else { continue };

                // responses to our submissions
                if let Some(id) = msg.id.as_u64() {
                    if let Some(kind) = pending.remove(&id) {
                        match msg.result.as_ref().and_then(|r| r.as_bool()) {
                            Some(true) => println!("[pool] ✓ {kind} accepted"),
                            _ => {
                                let err = msg.error.as_ref()
                                    .map(|e| e.to_string())
                                    .or_else(|| msg.result.as_ref().map(|r| r.to_string()))
                                    .unwrap_or_default();
                                println!("[pool] ✗ {kind} REJECTED: {err}");
                            }
                        }
                        continue;
                    }
                }

                let Some(method) = msg.method else { continue };
                match method.as_str() {
                    "pearl.challenge" => {
                        if let Some(p) = msg.params {
                            if let Ok(ch) = serde_json::from_value::<ChallengeParams>(p) {
                                println!("[pool] New BLAKE3 challenge diff={}", ch.difficulty);
                                // cancel previous solver
                                blake3_cancel.store(true, Ordering::Relaxed);
                                blake3_cancel = Arc::new(AtomicBool::new(false));

                                let cancel  = Arc::clone(&blake3_cancel);
                                let stx     = solved_tx.clone();
                                let seed_h  = ch.seed.clone();
                                let diff    = ch.difficulty;
                                let sigma   = match hex_to_32bytes(&seed_h) {
                                    Some(s) => s,
                                    None => { eprintln!("[pool] bad challenge seed"); continue; }
                                };
                                let gpu_c = Arc::clone(&gpu);
                                tokio::task::spawn_blocking(move || {
                                    match gpu_c.solve_pool_challenge(&sigma, diff, &cancel) {
                                        Ok(Some(n)) => {
                                            let _ = stx.blocking_send((seed_h, format!("{:016x}", n)));
                                        }
                                        Ok(None) => {} // cancelled
                                        Err(e) => {
                                            // GPU fallback: CPU solver
                                            eprintln!("[pool] GPU solver error {e}, falling back to CPU");
                                            if let Some(n) = solve_blake3_challenge(sigma, diff, &cancel) {
                                                let _ = stx.blocking_send((seed_h, format!("{:016x}", n)));
                                            }
                                        }
                                    }
                                });
                            }
                        }
                    }

                    "mining.notify" => {
                        if let Some((job_id, sigma_hex)) = parse_notify(&msg.params) {
                            current_job_id = job_id.clone();
                            println!("[pool] New job {} sigma={:.16}...", job_id, sigma_hex);
                            let _ = challenge_tx.send(Some((sigma_hex, pouw_difficulty, job_id)));
                        }
                    }

                    "mining.set_difficulty" => {
                        if let Some(d) = msg.params.as_ref()
                            .and_then(|p| p.get(0))
                            .and_then(|v| v.as_u64())
                        {
                            pouw_difficulty = d as u32;
                            println!("[pool] PoUW difficulty updated to {}", pouw_difficulty);
                        }
                    }

                    "pearl.set_mining_params" => {
                        if let Some(mp) = parse_mining_params(&msg.params) {
                            println!("[pool] mining params updated: m={} n={} k={} r={}",
                                mp.m, mp.n, mp.k, mp.r);
                            let _ = params_tx.send(Some(mp));
                        }
                    }

                    _ => {}
                }
            }

            // ── BLAKE3 solver result ────────────────────────────────────────
            Some((seed, nonce)) = solved_rx.recv() => {
                let cr = serde_json::json!({
                    "id": msg_id,
                    "method": "pearl.challenge_response",
                    "params": { "seed": &seed, "nonce": &nonce }
                });
                pending.insert(msg_id, "challenge_response");
                msg_id += 1;
                println!("[pool] -> pearl.challenge_response nonce={}", nonce);
                writer.write_all(format!("{}\n", cr).as_bytes()).await?;
            }

            // ── PoUW share submission ───────────────────────────────────────
            Some(submit) = submit_rx.recv() => {
                let ms = serde_json::json!({
                    "id": msg_id,
                    "method": "mining.submit",
                    "params": [&submit.worker, &submit.job_id, &submit.payload]
                });
                pending.insert(msg_id, "mining.submit");
                msg_id += 1;
                println!("[pool] -> mining.submit job={} worker={}", submit.job_id, submit.worker);
                writer.write_all(format!("{}\n", ms).as_bytes()).await?;
            }
        }
    }

    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

fn parse_notify(params: &Option<serde_json::Value>) -> Option<(String, String)> {
    let p = params.as_ref()?;
    let job_id   = p.get(0)?.as_str()?.to_string();
    let sigma    = p.get(1)?.as_str()?.to_string();
    if sigma.len() != 64 { return None; }
    Some((job_id, sigma))
}

fn parse_mining_params(params: &Option<serde_json::Value>) -> Option<MiningParams> {
    let p   = params.as_ref()?;
    let obj = if p.is_array() { p.get(0)? } else { p };
    Some(MiningParams {
        sigma:      [0u8; 32],
        difficulty: 0,
        m:  obj.get("m")?.as_u64()? as usize,
        n:  obj.get("n")?.as_u64()? as usize,
        k:  obj.get("k")?.as_u64()? as usize,
        r:  (obj.get("rank").or_else(|| obj.get("r")))?.as_u64()? as usize,
        tm: obj.get("tm").and_then(|v| v.as_u64()).unwrap_or(16) as usize,
        tn: obj.get("tn").and_then(|v| v.as_u64()).unwrap_or(16) as usize,
    })
}

fn check_difficulty(hash: &[u8; 32], difficulty: u32) -> bool {
    // Pool requires LEADING zero bits: hash[0..] must start with `difficulty` zero bits
    let full_bytes = (difficulty / 8) as usize;
    let remainder  = (difficulty % 8) as u8;
    for i in 0..full_bytes {
        if hash[i] != 0 { return false; }
    }
    if remainder > 0 {
        // Check top `remainder` bits of the next byte (MSBs in output byte order)
        let mask = 0xFF_u8 << (8 - remainder);
        if hash[full_bytes] & mask != 0 { return false; }
    }
    true
}

fn solve_blake3_challenge(seed: [u8; 32], difficulty: u32, cancel: &AtomicBool) -> Option<u64> {
    const CHUNK: u64 = 4_000_000;
    let mut offset = 0u64;
    loop {
        if cancel.load(Ordering::Relaxed) { return None; }
        let result = (offset..offset + CHUNK).into_par_iter().find_any(|&nonce| {
            if cancel.load(Ordering::Relaxed) { return false; }
            let mut input = [0u8; 40];
            input[..32].copy_from_slice(&seed);
            input[32..].copy_from_slice(&nonce.to_le_bytes());
            check_difficulty(blake3::hash(&input).as_bytes(), difficulty)
        });
        if let Some(nonce) = result { return Some(nonce); }
        offset += CHUNK;
        if offset == 0 { return None; }
    }
}

fn hex_to_32bytes(hex: &str) -> Option<[u8; 32]> {
    if hex.len() != 64 { return None; }
    let mut out = [0u8; 32];
    for (i, chunk) in hex.as_bytes().chunks(2).enumerate() {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        out[i] = (hi << 4) | lo;
    }
    Some(out)
}

fn hex_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}
