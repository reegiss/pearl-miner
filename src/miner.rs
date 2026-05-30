use crate::pool::Submit;
use pearl_commitment::compute as commitment_compute;
use pearl_gpu::GpuMiner;
use pearl_noise::{apply_noise, generate_e, generate_f};
use pearl_types::MiningParams;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
    gpus: Vec<Arc<GpuMiner>>,
}

impl Miner {
    pub fn new(device_ids: &[usize]) -> Result<Self, pearl_gpu::GpuError> {
        let mut gpus = Vec::with_capacity(device_ids.len());
        for &id in device_ids {
            let gpu = GpuMiner::new(id)?;
            println!("[gpu:{id}] {} · {} MB", gpu.info.name, gpu.info.mem_mb);
            gpus.push(Arc::new(gpu));
        }
        Ok(Self { gpus })
    }

    pub async fn run(
        self,
        wallet:       String,
        mut challenge_rx: watch::Receiver<Option<(String, u32)>>,
        submit_tx:    mpsc::Sender<Submit>,
    ) {
        let n_gpus = self.gpus.len();
        println!("[miner] {} GPU(s) ready. Waiting for first challenge...", n_gpus);

        // Shared hash counter across all GPU threads — for combined hashrate display
        let total_hashes = Arc::new(AtomicU64::new(0));

        // Cancel flag + join handles for all running GPU threads
        let mut cancel  = Arc::new(AtomicBool::new(false));
        let mut handles: Vec<tokio::task::JoinHandle<()>> = Vec::new();

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

            // Cancel all running GPU threads and wait for clean exit
            cancel.store(true, Ordering::Relaxed);
            for h in handles.drain(..) {
                let _ = h.await;
            }

            // Reset shared hash counter for this challenge
            total_hashes.store(0, Ordering::Relaxed);

            println!("[miner] New challenge — seed={seed_hex} difficulty={difficulty} gpus={n_gpus}");

            let params = Arc::new(MiningParams {
                sigma,
                difficulty,
                r:  DEFAULT_R,
                k:  DEFAULT_K,
                tm: DEFAULT_TM,
                tn: DEFAULT_TN,
                m:  DEFAULT_M,
                n:  DEFAULT_N,
            });

            let new_cancel = Arc::new(AtomicBool::new(false));
            cancel = Arc::clone(&new_cancel);

            // Spawn one blocking thread per GPU
            for (gpu_idx, gpu) in self.gpus.iter().enumerate() {
                let gpu_c       = Arc::clone(gpu);
                let params_c    = Arc::clone(&params);
                let wallet_c    = wallet.clone();
                let seed_c      = seed_hex.clone();
                let cancel_c    = Arc::clone(&new_cancel);
                let submit_c    = submit_tx.clone();
                let hashes_c    = Arc::clone(&total_hashes);

                handles.push(tokio::task::spawn_blocking(move || {
                    mining_loop(
                        &gpu_c, &params_c, &wallet_c, &seed_c,
                        gpu_idx, n_gpus,
                        cancel_c, submit_c, hashes_c,
                    );
                }));
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn mining_loop(
    gpu:          &GpuMiner,
    params:       &MiningParams,
    wallet:       &str,
    seed_hex:     &str,
    gpu_idx:      usize,   // which GPU this thread is
    n_gpus:       usize,   // total GPUs — used to stride job seeds
    cancel:       Arc<AtomicBool>,
    submit_tx:    mpsc::Sender<Submit>,
    total_hashes: Arc<AtomicU64>,
) {
    let seed_short    = &seed_hex[..8];
    let tiles_per_job = ((params.m / DEFAULT_TM) * (params.n / DEFAULT_TN)) as u64;
    let start         = Instant::now();
    let mut job       = 0u64;

    while !cancel.load(Ordering::Relaxed) {
        job += 1;
        // Stride seeds so each GPU works on distinct matrices
        let seed_a = (job - 1) * n_gpus as u64 + gpu_idx as u64;
        let seed_b = seed_a ^ 0xFFFF_FFFF_FFFF_FFFF;

        let a = random_matrix_i8(params.m, params.k, seed_a);
        let b = random_matrix_i8(params.k, params.n, seed_b);

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
                        "[gpu:{gpu_idx}] BLOCK FOUND! seed={seed_short} job={job} tile=({},{}) nonce={nonce}",
                        blk.tile_i, blk.tile_j,
                    );
                    println!("[gpu:{gpu_idx}] wallet={wallet}");
                    let _ = submit_tx.blocking_send(Submit {
                        seed:  seed_hex.to_string(),
                        nonce,
                    });
                }
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("[gpu:{gpu_idx}] error: {e}");
                break;
            }
        }

        // Update shared hash counter and print combined hashrate from GPU 0
        let new_total = total_hashes.fetch_add(tiles_per_job, Ordering::Relaxed) + tiles_per_job;
        if gpu_idx == 0 {
            let elapsed = start.elapsed().as_secs_f64().max(0.001);
            println!(
                "[miner] seed={seed_short} job={job} gpu:{gpu_idx}  {} total",
                fmt_hashrate(new_total as f64 / elapsed),
            );
        }
    }
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

fn hex_bytes(b: &[u8]) -> String {
    b.iter().map(|v| format!("{v:02x}")).collect()
}
