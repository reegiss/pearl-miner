use crate::pool::Submit;
use pearl_commitment::compute_challenge;
use pearl_gpu::GpuMiner;
use pearl_noise::{apply_f, generate_f};
use pearl_types::MiningParams;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, watch};

const DEFAULT_R:  usize = 256;
const DEFAULT_K:  usize = 4096;
const DEFAULT_TM: usize = 16;
const DEFAULT_TN: usize = 16;
const DEFAULT_M:  usize = 8192;
const DEFAULT_N:  usize = 8192;

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
            println!("[gpu:{}] {} · {} MB · {} · sm_{}", id, gpu.info.name, gpu.info.mem_mb, kernel, gpu.info.sm);
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
            println!("[miner] challenge seed={}... diff={difficulty}", &seed_hex[..16]);

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
                let cancel_c = Arc::clone(&new_cancel);
                let hashes_c = Arc::clone(&total_hashes);
                let cc_c     = Arc::clone(&cc);
                let submit_p = submit_tx.clone();
                let seed_p   = seed_hex.clone();

                handles.push(tokio::task::spawn_blocking(move || {
                    mining_loop(
                        &gpu_c, &params_c, &wallet_c,
                        gpu_idx, n_gpus,
                        &cc_c,
                        cancel_c, hashes_c,
                        submit_p,
                    );
                }));
            }
        }
    }
}

fn mining_loop(
    gpu:          &GpuMiner,
    params:       &MiningParams,
    wallet:       &str,
    gpu_idx:      usize,
    n_gpus:       usize,
    cc:           &pearl_commitment::ChallengeCommitment,
    cancel:       Arc<AtomicBool>,
    total_hashes: Arc<AtomicU64>,
    submit_tx:    mpsc::Sender<Submit>,
) {
    let tiles_per_job = ((params.m / DEFAULT_TM) * (params.n / DEFAULT_TN)) as u64;
    let mut job        = 0u64;
    let start          = Instant::now();
    let mut last_log   = start;
    let mut t_kernel_acc = 0u128;
    let mut t_dtoh_acc   = 0u128;

    while !cancel.load(Ordering::Relaxed) {
        job += 1;
        let job_seed = (job - 1) * n_gpus as u64 + gpu_idx as u64;

        // One BLAKE3 call: virtual sA seeded from challenge kappa + job counter.
        // All heavy matrix work (A generation, noise, noise application) is on GPU.
        let virtual_sa = *blake3::keyed_hash(&cc.kappa, &job_seed.to_le_bytes()).as_bytes();

        let t0 = Instant::now();

        // GPU: tiled matmul A'·B', M-state accumulation, BLAKE3 difficulty check
        // A' is generated on-the-fly in the MatMul kernels (PRNG fusion)
        let ([t_kernel, t_dtoh, _], found_blocks) = match gpu.mine(params, job_seed, &virtual_sa) {
            Ok((blocks, timing)) => (timing, blocks),
            Err(e) => { eprintln!("[gpu:{gpu_idx}] mine error: {e}"); break; }
        };

        for fb in found_blocks {
            let nonce_hex: String = fb.hash.iter().map(|b| format!("{:02x}", b)).collect();
            println!("[PoUW] Found block! hash={}", &nonce_hex[..16]);
            // NOTE: Do not submit PoUW blocks to pearl.challenge_response.
            // That method is for BLAKE3 challenge nonces only.
        }

        let _ = wallet;
        let _ = t0;
        t_kernel_acc  += t_kernel;
        t_dtoh_acc    += t_dtoh;

        let new_total = total_hashes.fetch_add(tiles_per_job, Ordering::Relaxed) + tiles_per_job;

        if gpu_idx == 0 {
            let now = Instant::now();
            if (now - last_log).as_secs_f64() >= 5.0 {
                let elapsed = (now - start).as_secs_f64().max(0.001);
                let kernel  = if gpu.info.use_wmma { "wmma" } else { "dp4a" };
                let j = job as f64;
                
                // 1 Hash = 1 INT8 MAD (Multiply-Add). Each tile processes (tm * tn * k) MADs.
                let mads_per_tile = (params.tm * params.tn * params.k) as f64;
                let raw_mads_per_sec = (new_total as f64 * mads_per_tile) / elapsed;
                
                // Target multiplier normalized to baseline (r=32).
                let effective_mads_per_sec = raw_mads_per_sec * (params.r as f64 / 32.0);
                
                println!(
                    "[miner] {} (Effective) / {} (Raw) [{kernel}] kernel={:.1}ms dtoh={:.2}ms",
                    fmt_hashrate(effective_mads_per_sec),
                    fmt_hashrate(raw_mads_per_sec),
                    t_kernel_acc as f64 / j / 1000.0,
                    t_dtoh_acc   as f64 / j / 1000.0,
                );
                last_log = now;
            }
        }
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

