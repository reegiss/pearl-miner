use pearl_types::MiningParams;
use serde::Deserialize;
use std::collections::HashMap;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, watch};
use tokio::time::sleep;

#[derive(Deserialize, Debug)]
struct PoolMessage {
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

/// A solved challenge the miner wants to submit to the pool.
#[derive(Clone, Debug)]
pub struct Submit {
    pub seed:   String,   // challenge seed (hex)
    pub nonce:  String,   // winning nonce (16-char hex)
    pub job_id: String,   // job_id from last mining.notify
}

/// Connect to the pool, push challenges to `challenge_tx`, forward mining params
/// via `params_tx`, and forward submit messages from `submit_rx` to the pool.
/// Reconnects on drop or error.
pub async fn run(
    addr:          &str,
    wallet:        &str,
    password:      &str,
    challenge_tx:  watch::Sender<Option<(String, u32, String)>>,
    params_tx:     watch::Sender<Option<MiningParams>>,
    mut submit_rx: mpsc::Receiver<Submit>,
) {
    loop {
        println!("[pool] Connecting to {}...", addr);
        match connect(addr, wallet, password, &challenge_tx, &params_tx, &mut submit_rx).await {
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
    challenge_tx:  &watch::Sender<Option<(String, u32, String)>>,
    params_tx:     &watch::Sender<Option<MiningParams>>,
    submit_rx:     &mut mpsc::Receiver<Submit>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let stream = TcpStream::connect(addr).await?;
    let (reader, mut writer) = stream.into_split();

    // Worker name must contain a dot for pool to register it
    let worker = if wallet.contains('.') {
        wallet.to_string()
    } else {
        format!("{}.worker1", wallet)
    };

    // Stratum handshake: subscribe + authorize back-to-back
    let subscribe = serde_json::json!({
        "id": 1,
        "method": "mining.subscribe",
        "params": ["pearl-miner/0.1.0", null, addr, 0]
    });
    let auth = serde_json::json!({
        "id": 2,
        "method": "mining.authorize",
        "params": [&worker, password]
    });
    writer.write_all(format!("{}\n{}\n", subscribe, auth).as_bytes()).await?;
    println!("[pool] Connected. Handshaking as {}...", worker);

    let mut last_seed                          = String::new();
    let mut current_job_id                     = String::new();
    let mut msg_id: u64                        = 3;
    let mut pending: HashMap<u64, &'static str> = HashMap::new();
    let mut lines = BufReader::new(reader).lines();

    loop {
        tokio::select! {
            // ── Incoming from pool ──────────────────────────────────────────
            line = lines.next_line() => {
                let Some(line) = line? else { break; };
                if line.trim().is_empty() { continue; }

                println!("[pool] <- {}", line);

                let Ok(msg) = serde_json::from_str::<PoolMessage>(&line) else {
                    continue;
                };

                // ── Responses to our submissions (id ≥ 3) ──────────────────
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

                // ── Subscribe response (id=1) ───────────────────────────────
                if msg.id == serde_json::json!(1) || msg.id == serde_json::json!("1") {
                    println!("[pool] Subscription confirmed.");
                    continue;
                }

                // ── Authorize response (id=2) ───────────────────────────────
                if msg.id == serde_json::json!(2) || msg.id == serde_json::json!("2") {
                    if let Some(res) = &msg.result {
                        if res.as_bool() == Some(true) || !res.is_null() {
                            println!("[pool] Authentication successful! Worker is now active.");
                        } else {
                            println!("[pool] Authentication FAILED: {:?}", res);
                        }
                    } else if let Some(err) = &msg.error {
                        println!("[pool] Authentication ERROR: {:?}", err);
                    }
                    continue;
                }

                // ── Notifications ───────────────────────────────────────────
                let Some(method) = msg.method else { continue; };
                match method.as_str() {

                    "mining.set_difficulty" => {
                        if let Some(d) = msg.params
                            .as_ref()
                            .and_then(|p| p.get(0))
                            .and_then(|v| v.as_f64())
                        {
                            println!("[pool] difficulty set to {}", d as u32);
                        } else {
                            eprintln!("[pool] malformed mining.set_difficulty");
                        }
                    }

                    "pearl.set_mining_params" => {
                        let parsed = msg.params.as_ref().and_then(|p| {
                            // params may be wrapped in an array or sent as object directly
                            let obj = if p.is_array() { p.get(0)? } else { p };
                            Some(MiningParams {
                                sigma:      [0u8; 32],
                                difficulty: 0,
                                m:  obj.get("m")?.as_u64()? as usize,
                                n:  obj.get("n")?.as_u64()? as usize,
                                k:  obj.get("k")?.as_u64()? as usize,
                                r:  obj.get("r")?.as_u64()? as usize,
                                tm: obj.get("tm").and_then(|v| v.as_u64()).unwrap_or(16) as usize,
                                tn: obj.get("tn").and_then(|v| v.as_u64()).unwrap_or(16) as usize,
                            })
                        });
                        if let Some(mp) = parsed {
                            println!("[pool] mining params: m={} n={} k={} r={} tm={} tn={}",
                                mp.m, mp.n, mp.k, mp.r, mp.tm, mp.tn);
                            let _ = params_tx.send(Some(mp));
                        } else {
                            eprintln!("[pool] malformed pearl.set_mining_params");
                        }
                    }

                    "mining.notify" => {
                        if let Some(jid) = msg.params
                            .as_ref()
                            .and_then(|p| p.get(0))
                            .and_then(|v| v.as_str())
                        {
                            current_job_id = jid.to_string();
                            println!("[pool] job {}", current_job_id);
                        } else {
                            eprintln!("[pool] malformed mining.notify");
                        }
                    }

                    "pearl.challenge" => {
                        if let Some(p) = msg.params {
                            if let Ok(ch) = serde_json::from_value::<ChallengeParams>(p) {
                                if ch.seed == last_seed { continue; }
                                last_seed = ch.seed.clone();
                                let _ = challenge_tx.send(Some((
                                    ch.seed,
                                    ch.difficulty,
                                    current_job_id.clone(),
                                )));
                            }
                        }
                    }

                    _ => {}
                }
            }

            // ── Outgoing to pool ────────────────────────────────────────────
            Some(submit) = submit_rx.recv() => {
                // pearl.challenge_response — BLAKE3 answer
                let cr = serde_json::json!({
                    "id": msg_id,
                    "method": "pearl.challenge_response",
                    "params": { "seed": &submit.seed, "nonce": &submit.nonce }
                });
                pending.insert(msg_id, "challenge_response");
                msg_id += 1;

                if submit.job_id.is_empty() {
                    println!("[pool] no job_id yet, skipping mining.submit");
                    writer.write_all(format!("{cr}\n").as_bytes()).await?;
                } else {
                    // mining.submit — credits the share on the pool dashboard
                    let ms = serde_json::json!({
                        "id": msg_id,
                        "method": "mining.submit",
                        "params": [&worker, &submit.job_id, &submit.nonce]
                    });
                    pending.insert(msg_id, "submit");
                    msg_id += 1;
                    writer.write_all(format!("{cr}\n{ms}\n").as_bytes()).await?;
                }
            }
        }
    }

    Ok(())
}
