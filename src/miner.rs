use crate::pool::Submit;
use pearl_commitment::compute as commitment_compute;
use pearl_gpu::GpuMiner;
use pearl_noise::{apply_noise, generate_e, generate_f};
use pearl_types::MiningParams;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, watch};

const DEFAULT_R:  usize = 32;
const DEFAULT_K:  usize = 512;
const DEFAULT_TM: usize = 16;
const DEFAULT_TN: usize = 16;
const DEFAULT_M:  usize = 512;
const DEFAULT_N:  usize = 512;

pub struct Miner {
    gpu: Arc<GpuMiner>,
}

impl Miner {
    pub fn new(device_id: usize) -> Result<Self, pearl_gpu::GpuError> {
        let gpu = GpuMiner::new(device_id)?;
        println!(
            "[gpu:{device_id}] {} · {} MB",
            gpu.info.name, gpu.info.mem_mb,
        );
        Ok(Self { gpu: Arc::new(gpu) })
    }

    pub async fn run(
        self,
        wallet: String,
        mut challenge_rx: watch::Receiver<Option<(String, u32)>>,
        submit_tx: mpsc::Sender<Submit>,
    ) {
        println!("[miner] Waiting for first challenge...");

        // cancel flag shared with the blocking thread
        let mut cancel = Arc::new(AtomicBool::new(false));
        let mut current_job: Option<tokio::task::JoinHandle<()>> = None;

        loop {
            if challenge_rx.changed().await.is_err() {
                break;
            }
            let Some((seed_hex, difficulty)) = challenge_rx.borrow().clone() else {
                continue;
            };

            let sigma = match hex_to_32bytes(&seed_hex) {
                Some(s) => s,
                None => {
                    eprintln!("[miner] Invalid seed hex: {seed_hex}");
                    continue;
                }
            };

            // Signal old loop to stop and wait for clean exit
            cancel.store(true, Ordering::Relaxed);
            if let Some(h) = current_job.take() {
                let _ = h.await;
            }

            println!("[miner] New challenge — seed={seed_hex} difficulty={difficulty}");

            let params = MiningParams {
                sigma,
                difficulty,
                r:  DEFAULT_R,
                k:  DEFAULT_K,
                tm: DEFAULT_TM,
                tn: DEFAULT_TN,
                m:  DEFAULT_M,
                n:  DEFAULT_N,
            };

            // Fresh cancel flag for this challenge
            let new_cancel = Arc::new(AtomicBool::new(false));
            cancel = Arc::clone(&new_cancel);

            let gpu_c    = Arc::clone(&self.gpu);
            let wallet_c = wallet.clone();
            let seed_c   = seed_hex.clone();
            let submit_c = submit_tx.clone();

            current_job = Some(tokio::task::spawn_blocking(move || {
                mining_loop(&gpu_c, &params, &wallet_c, &seed_c, new_cancel, submit_c);
            }));
        }
    }
}

fn mining_loop(
    gpu:       &GpuMiner,
    params:    &MiningParams,
    wallet:    &str,
    seed_hex:  &str,
    cancel:    Arc<AtomicBool>,
    submit_tx: mpsc::Sender<Submit>,
) {
    let seed_short = &seed_hex[..8];
    let start      = Instant::now();
    let mut job    = 0u64;

    while !cancel.load(Ordering::Relaxed) {
        job += 1;
        // Generate fresh random matrices each job
        let a = random_matrix_i8(params.m, params.k, job);
        let b = random_matrix_i8(params.k, params.n, job ^ 0xFFFF_FFFF_FFFF_FFFF);

        let b_col = transpose_i8(&b, params.k, params.n);
        let commitments = commitment_compute(&a, &b_col, params);

        let (el, er) = generate_e(params.m, params.k, params.r, &commitments.s_a);
        let (fl, fr) = generate_f(params.n, params.k, params.r, &commitments.s_b);

        let a_prime = apply_noise(&a, &el, &er, params.m, params.k, params.r);
        let b_prime = apply_noise(&b, &fl, &fr, params.n, params.k, params.r);

        match gpu.mine(&a_prime, &b_prime, params, &commitments.s_a) {
            Ok(blocks) if !blocks.is_empty() => {
                for blk in &blocks {
                    let nonce = hex_bytes(&blk.hash);
                    println!(
                        "[miner] BLOCK FOUND! seed={seed_short} job={job} tile=({},{}) nonce={nonce}",
                        blk.tile_i, blk.tile_j,
                    );
                    println!("[miner] wallet={wallet}");
                    let _ = submit_tx.blocking_send(Submit {
                        seed:  seed_hex.to_string(),
                        nonce,
                    });
                }
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("[miner] GPU error: {e}");
                break;
            }
        }

        let total_s    = start.elapsed().as_secs_f64().max(0.001);
        let tiles_per_job = ((params.m / DEFAULT_TM) * (params.n / DEFAULT_TN)) as u64;
        let total_hashes  = job * tiles_per_job;
        let hashrate      = total_hashes as f64 / total_s;

        println!(
            "[miner] seed={seed_short} job={job}  {}",
            fmt_hashrate(hashrate),
        );
    }

    let total_s       = start.elapsed().as_secs_f64().max(0.001);
    let tiles_per_job = ((params.m / DEFAULT_TM) * (params.n / DEFAULT_TN)) as u64;
    let total_hashes  = job * tiles_per_job;
    println!(
        "[miner] seed={seed_short} stopped after {job} jobs  {}",
        fmt_hashrate(total_hashes as f64 / total_s),
    );
}

fn random_matrix_i8(rows: usize, cols: usize, seed: u64) -> Vec<i8> {
    let mut x = seed ^ 0xDEAD_BEEF_CAFE_1337;
    (0..rows * cols)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let v = ((x >> 33) as i32) & 0x7F;
            (if v > 64 { v - 128 } else { v }) as i8
        })
        .collect()
}

fn transpose_i8(src: &[i8], rows: usize, cols: usize) -> Vec<i8> {
    let mut dst = vec![0i8; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            dst[c * rows + r] = src[r * cols + c];
        }
    }
    dst
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

fn fmt_hashrate(h: f64) -> String {
    const T: f64 = 1_000_000_000_000.0;
    const G: f64 =     1_000_000_000.0;
    const M: f64 =         1_000_000.0;
    const K: f64 =             1_000.0;
    if h >= T      { format!("{:.2} TH/s", h / T) }
    else if h >= G { format!("{:.2} GH/s", h / G) }
    else if h >= M { format!("{:.2} MH/s", h / M) }
    else if h >= K { format!("{:.2} KH/s", h / K) }
    else           { format!("{:.0} H/s",  h)      }
}

fn hex_bytes(b: &[u8]) -> String {
    b.iter().map(|v| format!("{v:02x}")).collect()
}
