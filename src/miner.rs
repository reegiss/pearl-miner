use crate::pool::Submit;
use pearl_commitment::{compute_challenge, compute_sa};
use pearl_gpu::GpuMiner;
use pearl_noise::{apply_e, apply_f, generate_e, generate_f};
use pearl_types::MiningParams;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, watch};

const DEFAULT_R:  usize = 32;
const DEFAULT_K:  usize = 512;
const DEFAULT_TM: usize = 16;
const DEFAULT_TN: usize = 16;
const DEFAULT_M:  usize = 1024;
const DEFAULT_N:  usize = 1024;

pub struct Miner {
    gpus: Vec<Arc<GpuMiner>>,
}

impl Miner {
    pub fn new(device_ids: &[usize]) -> Result<Self, pearl_gpu::GpuError> {
        let params = MiningParams {
            sigma: [0u8; 32], difficulty: 32,
            r: DEFAULT_R, k: DEFAULT_K, tm: DEFAULT_TM, tn: DEFAULT_TN,
            m: DEFAULT_M, n: DEFAULT_N,
        };
        let mut gpus = Vec::with_capacity(device_ids.len());
        for &id in device_ids {
            let gpu = GpuMiner::new(id, &params)?;
            let kernel = if gpu.info.use_wmma { "WMMA tensor cores" } else { "DP4A CUDA cores" };
            println!("[gpu:{}] {} · {} MB · {}", id, gpu.info.name, gpu.info.mem_mb, kernel);
            gpus.push(Arc::new(gpu));
        }
        Ok(Self { gpus })
    }

    pub async fn run(
        self,
        wallet:          String,
        mut challenge_rx: watch::Receiver<Option<(String, u32)>>,
        submit_tx:       mpsc::Sender<Submit>,
    ) {
        let n_gpus = self.gpus.len();
        println!("[miner] {} GPU(s) ready.", n_gpus);

        let total_hashes = Arc::new(AtomicU64::new(0));
        let mut cancel   = Arc::new(AtomicBool::new(false));
        let mut handles: Vec<tokio::task::JoinHandle<()>> = Vec::new();

        loop {
            if challenge_rx.changed().await.is_err() { break; }
            let Some((seed_hex, difficulty)) = challenge_rx.borrow().clone() else { continue; };

            let sigma = match hex_to_32bytes(&seed_hex) {
                Some(s) => s,
                None => { eprintln!("[miner] bad seed: {seed_hex}"); continue; }
            };

            // Stop all running threads
            cancel.store(true, Ordering::Relaxed);
            for h in handles.drain(..) { let _ = h.await; }

            total_hashes.store(0, Ordering::Relaxed);
            println!("[pool] challenge seed={} difficulty={difficulty}", &seed_hex[..16]);

            let params = Arc::new(MiningParams {
                sigma, difficulty,
                r: DEFAULT_R, k: DEFAULT_K,
                tm: DEFAULT_TM, tn: DEFAULT_TN,
                m: DEFAULT_M, n: DEFAULT_N,
            });

            // --- Pre-compute B' once per challenge (paper §4.2 optimization) ---
            let b_seed = 0xB0B0_B0B0_B0B0_B0B0u64 ^ u64::from_le_bytes(sigma[..8].try_into().unwrap());
            let b = random_matrix_i8(params.k, params.n, b_seed);
            let b_col   = transpose_i8(&b, params.k, params.n);
            let cc      = compute_challenge(&b_col, &params);
            let f_noise = generate_f(params.n, params.k, params.r, &cc.s_b);
            let b_prime = apply_f(&b, &f_noise, params.n, params.k);
            let cc      = Arc::new(cc);

            // Upload B' to each GPU once — stays resident until next challenge
            for gpu in &self.gpus {
                if let Err(e) = gpu.set_b(&b_prime) {
                    eprintln!("[miner] set_b error: {e}");
                }
            }

            let new_cancel = Arc::new(AtomicBool::new(false));
            cancel = Arc::clone(&new_cancel);

            // Spawn BLAKE3 challenge solver (CPU, rayon parallel)
            {
                let cancel_c = Arc::clone(&new_cancel);
                let submit_c = submit_tx.clone();
                let seed_c   = seed_hex.clone();
                handles.push(tokio::task::spawn_blocking(move || {
                    if let Some(nonce) = solve_blake3_challenge(sigma, difficulty, &cancel_c) {
                        let nonce_hex = format!("{:016x}", nonce);
                        println!("[BLAKE3] Challenge solved! nonce={nonce_hex}");
                        let _ = submit_c.blocking_send(Submit { seed: seed_c, nonce: nonce_hex });
                    }
                }));
            }

            // Spawn PoUW GPU mining threads
            for (gpu_idx, gpu) in self.gpus.iter().enumerate() {
                let gpu_c    = Arc::clone(gpu);
                let params_c = Arc::clone(&params);
                let wallet_c = wallet.clone();
                let seed_c   = seed_hex.clone();
                let cancel_c = Arc::clone(&new_cancel);
                let hashes_c = Arc::clone(&total_hashes);
                let cc_c     = Arc::clone(&cc);

                handles.push(tokio::task::spawn_blocking(move || {
                    mining_loop(
                        &gpu_c, &params_c, &wallet_c, &seed_c,
                        gpu_idx, n_gpus,
                        &cc_c,
                        cancel_c, hashes_c,
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
    gpu_idx:      usize,
    n_gpus:       usize,
    cc:           &pearl_commitment::ChallengeCommitment,
    cancel:       Arc<AtomicBool>,
    total_hashes: Arc<AtomicU64>,
) {
    let tiles_per_job = ((params.m / DEFAULT_TM) * (params.n / DEFAULT_TN)) as u64;
    let mut job        = 0u64;
    let start          = Instant::now();
    let mut last_log   = start;

    // Profiling accumulators (µs)
    let mut t_rand   = 0u128;
    let mut t_commit = 0u128;
    let mut t_noise  = 0u128;
    let mut t_apply  = 0u128;
    let mut t_gpu    = 0u128;

    while !cancel.load(Ordering::Relaxed) {
        job += 1;
        let seed_a = (job - 1) * n_gpus as u64 + gpu_idx as u64;

        let t0 = Instant::now();
        let a = random_matrix_i8(params.m, params.k, seed_a);
        t_rand += t0.elapsed().as_micros();

        let t0 = Instant::now();
        let (s_a, _) = compute_sa(&a, cc);
        t_commit += t0.elapsed().as_micros();

        let t0 = Instant::now();
        let e_noise = generate_e(params.m, params.k, params.r, &s_a);
        t_noise += t0.elapsed().as_micros();

        let t0 = Instant::now();
        let a_prime = apply_e(&a, &e_noise, params.m, params.k, params.r);
        t_apply += t0.elapsed().as_micros();

        let t0 = Instant::now();
        match gpu.mine(&a_prime, params, &s_a) {
            Ok(blocks) if !blocks.is_empty() => {
                for blk in &blocks {
                    // Log found PoUW tiles — not submitted until proof format is confirmed
                    println!(
                        "[PoUW] gpu:{gpu_idx} tile=({},{}) hash={} wallet={wallet}",
                        blk.tile_i, blk.tile_j, hex_bytes(&blk.hash),
                    );
                }
            }
            Ok(_) => {}
            Err(e) => { eprintln!("[gpu:{gpu_idx}] error: {e}"); break; }
        }
        t_gpu += t0.elapsed().as_micros();

        let new_total = total_hashes.fetch_add(tiles_per_job, Ordering::Relaxed) + tiles_per_job;

        // GPU 0 logs combined hashrate every 100 jobs
        if gpu_idx == 0 && job % 100 == 0 {
            let now     = Instant::now();
            let elapsed = (now - start).as_secs_f64().max(0.001);
            let per     = |t: u128| t as f64 / job as f64 / 1000.0; // ms/job
            println!(
                "[miner] {} · seed={} · job={}",
                fmt_hashrate(new_total as f64 / elapsed),
                &seed_hex[..16],
                job * n_gpus as u64,
            );
            let kernel = if gpu.info.use_wmma { "wmma" } else { "dp4a" };
            println!(
                "[profile/{kernel}] rand={:.2}ms commit={:.2}ms noise={:.2}ms apply={:.2}ms gpu={:.2}ms  total={:.2}ms/job",
                per(t_rand), per(t_commit), per(t_noise), per(t_apply), per(t_gpu),
                per(t_rand + t_commit + t_noise + t_apply + t_gpu),
            );
            last_log = now;
        }
        let _ = last_log; // suppress unused warning
    }
}

// ────────────────────────────────────────────────────────────────────────────
// BLAKE3 challenge solver
// ────────────────────────────────────────────────────────────────────────────

fn check_difficulty(hash: &[u8; 32], difficulty: u32) -> bool {
    // Treat hash as LE uint256; need hash < 2^(256-difficulty)
    // = top `difficulty` bits (bytes[31..] downward) must be zero
    let full_bytes = (difficulty / 8) as usize;
    let remainder  = (difficulty % 8) as u8;
    for i in 0..full_bytes {
        if hash[31 - i] != 0 { return false; }
    }
    if remainder > 0 {
        let mask = 0xFF_u8 << (8 - remainder);
        if hash[31 - full_bytes] & mask != 0 { return false; }
    }
    true
}

fn solve_blake3_challenge(seed: [u8; 32], difficulty: u32, cancel: &AtomicBool) -> Option<u64> {
    // Use half the cores so we don't starve the noise-generation rayon pool
    let n_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let n_solver = (n_cores / 2).max(2);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(n_solver)
        .build()
        .expect("rayon pool");

    const CHUNK: u64 = 1_000_000;
    let mut offset = 0u64;
    loop {
        if cancel.load(Ordering::Relaxed) { return None; }
        let result = pool.install(|| {
            (offset..offset + CHUNK).into_par_iter().find_any(|&nonce| {
                if cancel.load(Ordering::Relaxed) { return false; }
                let mut input = [0u8; 40];
                input[..32].copy_from_slice(&seed);
                input[32..].copy_from_slice(&nonce.to_le_bytes());
                check_difficulty(blake3::hash(&input).as_bytes(), difficulty)
            })
        });
        if let Some(nonce) = result { return Some(nonce); }
        offset += CHUNK;
        if offset == 0 { return None; }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

fn random_matrix_i8(rows: usize, cols: usize, seed: u64) -> Vec<i8> {
    let mut x = seed ^ 0xDEAD_BEEF_CAFE_1337;
    (0..rows * cols).map(|_| {
        x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = ((x >> 33) as i32) & 0x7F;
        (if v > 64 { v - 128 } else { v }) as i8
    }).collect()
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
