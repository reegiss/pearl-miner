use serde::Deserialize;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, watch};
use tokio::time::sleep;

#[derive(Deserialize)]
struct PoolMessage {
    method: String,
    params: serde_json::Value,
}

#[derive(Deserialize)]
struct ChallengeParams {
    seed: String,
    difficulty: u32,
}

/// A found block the miner wants to submit to the pool.
#[derive(Clone, Debug)]
pub struct Submit {
    pub seed:  String,   // challenge seed that was solved
    pub nonce: String,   // winning BLAKE3 hash (hex)
}

/// Connect to the pool, push challenges to `challenge_tx`, and forward
/// submit messages from `submit_rx` back to the pool. Reconnects on drop.
pub async fn run(
    addr:         &str,
    wallet:       &str,
    password:     &str,
    challenge_tx: watch::Sender<Option<(String, u32)>>,
    mut submit_rx: mpsc::Receiver<Submit>,
) {
    loop {
        println!("[pool] Connecting to {}...", addr);
        match connect(addr, wallet, password, &challenge_tx, &mut submit_rx).await {
            Ok(()) => {} // clean close — pool resets after submit, reconnect immediately
            Err(e) => {
                eprintln!("[pool] Connection error: {e}");
                sleep(Duration::from_secs(5)).await;
            }
        }
    }
}

async fn connect(
    addr:         &str,
    wallet:       &str,
    password:     &str,
    challenge_tx: &watch::Sender<Option<(String, u32)>>,
    submit_rx:    &mut mpsc::Receiver<Submit>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let stream = TcpStream::connect(addr).await?;
    let (reader, mut writer) = stream.into_split();

    let auth = serde_json::json!({
        "id": 1,
        "method": "mining.authorize",
        "params": [wallet, password]
    });
    writer.write_all(format!("{auth}\n").as_bytes()).await?;
    println!("[pool] Connected. Authenticating...");

    let mut last_seed  = String::new();
    let mut msg_id: u64 = 2;
    let mut lines = BufReader::new(reader).lines();

    loop {
        tokio::select! {
            // Incoming message from pool
            line = lines.next_line() => {
                let Some(line) = line? else { break; };
                if line.trim().is_empty() { continue; }

                let Ok(msg) = serde_json::from_str::<PoolMessage>(&line) else {
                    continue;
                };

                match msg.method.as_str() {
                    "pearl.challenge" => {
                        if let Ok(ch) = serde_json::from_value::<ChallengeParams>(msg.params) {
                            if ch.seed == last_seed { continue; }
                            last_seed = ch.seed.clone();
                            println!("[pool] New challenge: seed={} difficulty={}", ch.seed, ch.difficulty);
                            let _ = challenge_tx.send(Some((ch.seed, ch.difficulty)));
                        }
                    }
                    other => {
                        println!("[pool] <- {other}");
                    }
                }
            }

            // Submit a found block — only pearl.challenge_response
            Some(submit) = submit_rx.recv() => {
                let msg = serde_json::json!({
                    "id": msg_id,
                    "method": "pearl.challenge_response",
                    "params": { "seed": submit.seed, "nonce": submit.nonce }
                });
                msg_id += 1;
                println!("[pool] -> submit nonce={}", &submit.nonce[..16]);
                writer.write_all(format!("{msg}\n").as_bytes()).await?;
            }
        }
    }

    Ok(())
}
