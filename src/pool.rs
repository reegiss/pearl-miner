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
        let line = r#"{"id":null,"method":"pearl.challenge","params":{"seed":"deadbeef","difficulty":1}}"#;
        assert!(parse_challenge(line).is_none());
    }

    #[test]
    fn test_parse_missing_difficulty() {
        let line = r#"{"id":null,"method":"pearl.challenge","params":{"seed":"a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90"}}"#;
        assert!(parse_challenge(line).is_none());
    }

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

        // First connection: accept and immediately drop
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
}
