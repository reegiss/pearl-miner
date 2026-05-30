use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::driver::sys::CUdevice_attribute::{
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR as CC_MAJOR,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR as CC_MINOR,
};
use cudarc::nvrtc::Ptx;
use pearl_types::{FoundBlock, MiningParams};
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum GpuError {
    #[error("CUDA driver error: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("No CUDA device at index {0}")]
    NoDevice(usize),
}

const TM: usize = 16;
const TN: usize = 16;
const FOUND_STRIDE: usize = 26;
const MAX_FOUND: usize = 256;

static PTX_SM75: &str = include_str!(env!("MATMUL_PTX_SM75"));

#[cfg(has_sm86_ptx)]
static PTX_SM86: &str = include_str!(env!("MATMUL_PTX_SM86"));

#[cfg(has_sm89_ptx)]
static PTX_SM89: &str = include_str!(env!("MATMUL_PTX_SM89"));

fn best_ptx(sm: u32) -> &'static str {
    #[cfg(has_sm89_ptx)]
    if sm >= 89 { return PTX_SM89; }
    #[cfg(has_sm86_ptx)]
    if sm >= 86 { return PTX_SM86; }
    let _ = sm;
    PTX_SM75
}

pub struct GpuInfo {
    pub name:     String,
    pub mem_mb:   usize,
    pub use_wmma: bool,
    pub sm:       u32,
}

pub fn device_count() -> usize {
    CudaContext::device_count().unwrap_or(0) as usize
}

pub struct GpuMiner {
    #[allow(dead_code)]
    ctx:          Arc<CudaContext>,
    stream:       Arc<CudaStream>,
    #[allow(dead_code)]
    module:       Arc<CudaModule>,
    func_dp4a:      CudaFunction,
    func_wmma:      CudaFunction,
    func_blake3:    CudaFunction,
    func_gen_a:     CudaFunction,
    func_gen_el:    CudaFunction,
    func_solve_pool: CudaFunction,
    use_wmma:       bool,
    d_a:           Mutex<CudaSlice<i8>>,   // A' (m×k) — regenerated each job
    d_el:          Mutex<CudaSlice<i8>>,   // Noise factor matrix EL (m×r) — pre-computed once per job
    d_b:           Mutex<CudaSlice<i8>>,   // B' (k×n) — set once per challenge
    d_c:           Mutex<CudaSlice<i32>>,  // placeholder (kernel never writes C)
    d_m:           Mutex<CudaSlice<u32>>,  // M states (num_tiles×16)
    d_sa:          Mutex<CudaSlice<u32>>,  // sA key as 8 u32 words (per job)
    d_threshold:   Mutex<CudaSlice<u32>>,  // difficulty threshold as 8 u32 words
    d_found:       Mutex<CudaSlice<u32>>,  // found tiles: MAX_FOUND × FOUND_STRIDE
    d_found_count: Mutex<CudaSlice<u32>>,  // atomic counter (1 u32)
    d_best_nonce:  Mutex<CudaSlice<u32>>,  // u64 best nonce (2 words)
    d_best_hash:   Mutex<CudaSlice<u32>>,  // u256 best hash (8 words)
    m: usize, n: usize, k: usize, r: usize,
    num_tiles_m: usize, num_tiles_n: usize,
    pub info: GpuInfo,
}

impl GpuMiner {
    pub fn new(device_id: usize, params: &MiningParams) -> Result<Self, GpuError> {
        let ctx    = CudaContext::new(device_id).map_err(|_| GpuError::NoDevice(device_id))?;
        let name   = ctx.name().unwrap_or_else(|_| "Unknown GPU".into());
        let mem_mb = ctx.total_mem().unwrap_or(0) / (1024 * 1024);
        let major  = ctx.attribute(CC_MAJOR).unwrap_or(7) as u32;
        let minor  = ctx.attribute(CC_MINOR).unwrap_or(5) as u32;
        let sm     = major * 10 + minor;
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_src(best_ptx(sm)))?;

        let func_dp4a      = module.load_function("tiled_matmul_dp4a")?;
        let func_wmma      = module.load_function("tiled_matmul_wmma")?;
        let func_blake3    = module.load_function("blake3_check")?;
        let func_gen_a     = module.load_function("generate_a_prime")?;
        let func_gen_el    = module.load_function("generate_el_matrix")?;
        let func_solve_pool = module.load_function("solve_blake3_pool")?;
        let use_wmma       = sm >= 72;

        let (m, n, k, r) = (params.m, params.n, params.k, params.r);
        let num_tiles_m  = (m + TM - 1) / TM;
        let num_tiles_n  = (n + TN - 1) / TN;
        let num_tiles    = num_tiles_m * num_tiles_n;

        let d_a           = Mutex::new(stream.alloc_zeros::<i8>(m * k)?);
        let d_el          = Mutex::new(stream.alloc_zeros::<i8>(m * r)?);
        let d_b           = Mutex::new(stream.alloc_zeros::<i8>(k * n)?);
        let d_c           = Mutex::new(stream.alloc_zeros::<i32>(1)?);
        let d_m           = Mutex::new(stream.alloc_zeros::<u32>(num_tiles * 16)?);
        let d_sa          = Mutex::new(stream.alloc_zeros::<u32>(8)?);
        let d_threshold   = Mutex::new(stream.alloc_zeros::<u32>(8)?);
        let d_found       = Mutex::new(stream.alloc_zeros::<u32>(MAX_FOUND * FOUND_STRIDE)?);
        let d_found_count = Mutex::new(stream.alloc_zeros::<u32>(1)?);
        let d_best_nonce  = Mutex::new(stream.alloc_zeros::<u32>(2)?);
        let d_best_hash   = Mutex::new(stream.alloc_zeros::<u32>(8)?);

        Ok(Self {
            ctx, stream, module,
            func_dp4a, func_wmma, func_blake3, func_gen_a, func_gen_el, func_solve_pool, use_wmma,
            d_a, d_el, d_b, d_c, d_m, d_sa, d_threshold, d_found, d_found_count,
            d_best_nonce, d_best_hash,
            m, n, k, r, num_tiles_m, num_tiles_n,
            info: GpuInfo { name, mem_mb, use_wmma, sm },
        })
    }

    /// Upload B' to GPU — call once per challenge.
    pub fn set_b(&self, b_prime: &[i8]) -> Result<(), GpuError> {
        self.stream.memcpy_htod(b_prime, &mut *self.d_b.lock().unwrap())?;
        Ok(())
    }

    /// Run matmul + GPU-side BLAKE3 difficulty check + pool challenge solver.
    /// Returns found blocks, best challenge nonce, best challenge hash, and stage timings.
    pub fn mine(
        &self,
        params:   &MiningParams,
        job_seed: u64,
        s_a:      &[u8; 32],
    ) -> Result<(Vec<FoundBlock>, u64, [u8; 32], [u128; 3]), GpuError> {
        let (m, n, k, r)     = (self.m, self.n, self.k, self.r);
        let (ntm, ntn)       = (self.num_tiles_m, self.num_tiles_n);
        let (mi, ni, ki, ri) = (m as i32, n as i32, k as i32, r as i32);
        let num_tiles        = ntm * ntn;

        // Upload sA key (8 u32 LE words)
        let sa_words: Vec<u32> = s_a.chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        self.stream.memcpy_htod(&sa_words, &mut *self.d_sa.lock().unwrap())?;

        // sa_seed as u64 for compute_noisy_a in kernel
        let sa_u64 = u64::from_le_bytes(s_a[..8].try_into().unwrap());

        // Upload threshold (8 u32 LE words)
        let thresh_bytes = difficulty_threshold(params.difficulty, r, TM, TN);
        let thresh_words: Vec<u32> = thresh_bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        self.stream.memcpy_htod(&thresh_words, &mut *self.d_threshold.lock().unwrap())?;

        // Reset counters/buffers
        {
            let zero = [0u32; 1];
            self.stream.memcpy_htod(&zero, &mut *self.d_found_count.lock().unwrap())?;
            let max_hash = [u32::MAX; 8];
            self.stream.memcpy_htod(&max_hash, &mut *self.d_best_hash.lock().unwrap())?;
        }

        // Lock all buffers
        let mut d_a = self.d_a.lock().unwrap();
        let mut d_el = self.d_el.lock().unwrap();
        let d_b  = self.d_b.lock().unwrap();
        let mut d_c = self.d_c.lock().unwrap();
        let mut d_m = self.d_m.lock().unwrap();
        let d_sa  = self.d_sa.lock().unwrap();
        let d_thr = self.d_threshold.lock().unwrap();
        let mut d_found       = self.d_found.lock().unwrap();
        let mut d_found_count = self.d_found_count.lock().unwrap();
        let mut d_best_nonce  = self.d_best_nonce.lock().unwrap();
        let mut d_best_hash   = self.d_best_hash.lock().unwrap();

        // --- Matmul kernel ---
        let t_kernel_start = std::time::Instant::now();

        // Stage 0: Pre-compute EL matrix
        {
            let threads = 1024u32;
            let threads = 256u32; let blocks = (((mi * ri / 16) as u32) + threads - 1) / threads;
            let cfg_el  = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
            let mut b   = self.stream.launch_builder(&self.func_gen_el);
            b.arg(&mut *d_el); b.arg(&sa_u64); b.arg(&mi); b.arg(&ri);
            unsafe { b.launch(cfg_el) }?;
        }

        // Stage 1: generate A' = A + EL·ER on GPU (dedicated low-register kernel)
        {
            let bx: u32 = 32;
            let by: u32 = 16;
            let gx = (ki as u32 + bx - 1) / bx;
            let gy = (mi as u32 + by - 1) / by;
            let cfg_gen = LaunchConfig {
                grid_dim:         (gx, gy, 1),
                block_dim:        (bx, by, 1),
                shared_mem_bytes: 0,
            };
            let mut b = self.stream.launch_builder(&self.func_gen_a);
            b.arg(&mut *d_a); b.arg(&*d_el); b.arg(&job_seed); b.arg(&sa_u64);
            b.arg(&mi); b.arg(&ki); b.arg(&ri);
            unsafe { b.launch(cfg_gen) }?;
        }

        // Stage 2: tiled matmul A'·B' + M-state accumulation
        if self.use_wmma {
            // sm_80+: double-buffer Bs (10×r×16); sm_72-79: single-buffer (9×r×16).
            // Always allocate the larger size — safe to over-allocate shared memory.
            let smem = (10 * r * 16) as u32;
            let cfg = LaunchConfig {
                grid_dim:         (ntn as u32, ((ntm + 7) / 8) as u32, 1),
                block_dim:        (32, 8, 1),
                shared_mem_bytes: smem,
            };
            let mut b = self.stream.launch_builder(&self.func_wmma);
            b.arg(&*d_a); b.arg(&*d_b); b.arg(&mut *d_c); b.arg(&mut *d_m);
            b.arg(&mi); b.arg(&ni); b.arg(&ki); b.arg(&ri);
            unsafe { b.launch(cfg) }?;
        } else {
            let warps  = (TM * TN) / 32;
            let shared = ((TM + TN) * r + warps * 4 + 16 * 4) as u32;
            let cfg = LaunchConfig {
                grid_dim:         (ntn as u32, ntm as u32, 1),
                block_dim:        (TN as u32, TM as u32, 1),
                shared_mem_bytes: shared,
            };
            let mut b = self.stream.launch_builder(&self.func_dp4a);
            b.arg(&*d_a); b.arg(&*d_b); b.arg(&mut *d_c); b.arg(&mut *d_m);
            b.arg(&mi); b.arg(&ni); b.arg(&ki); b.arg(&ri);
            unsafe { b.launch(cfg) }?;
        }


        // --- BLAKE3 check kernel (on GPU, no CPU transfer needed) ---
        {
            let threads = 256u32;
            let blocks  = ((num_tiles as u32) + threads - 1) / threads;
            let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
            let num_tiles_i  = num_tiles as i32;
            let num_tiles_ni = ntn as i32;
            let mut b = self.stream.launch_builder(&self.func_blake3);
            b.arg(&*d_m); b.arg(&*d_sa); b.arg(&*d_thr);
            b.arg(&mut *d_found); b.arg(&mut *d_found_count);
            b.arg(&num_tiles_i); b.arg(&num_tiles_ni);
            unsafe { b.launch(cfg) }?;
        }

        // --- Stage 3: Pool challenge solver (burst) ---
        {
            let threads = 1024u32;
            let blocks  = 64u32; // 64k nonces per job
            let base_nonce = job_seed * 65536; 
            let cfg = LaunchConfig { grid_dim: (blocks, 1, 1), block_dim: (threads, 1, 1), shared_mem_bytes: 0 };
            let mut b = self.stream.launch_builder(&self.func_solve_pool);
            b.arg(&*d_sa);
            b.arg(&base_nonce);
            b.arg(&mut *d_best_nonce);
            b.arg(&mut *d_best_hash);
            unsafe { b.launch(cfg) }?;
        }

        self.stream.synchronize()?;
        let t_kernel = t_kernel_start.elapsed().as_micros();

        // --- D2H results ---
        let t_dtoh_start = std::time::Instant::now();
        let count_vec = self.stream.clone_dtoh(&*d_found_count)?;
        let count = (count_vec[0] as usize).min(MAX_FOUND);
        let found_flat = if count > 0 { self.stream.clone_dtoh(&*d_found)? } else { vec![] };
        
        let best_nonce_words = self.stream.clone_dtoh(&*d_best_nonce)?;
        let best_nonce = (best_nonce_words[0] as u64) | ((best_nonce_words[1] as u64) << 32);
        
        let best_hash_words = self.stream.clone_dtoh(&*d_best_hash)?;
        let mut best_hash = [0u8; 32];
        for (i, &w) in best_hash_words.iter().enumerate() {
            best_hash[i * 4..(i + 1) * 4].copy_from_slice(&w.to_le_bytes());
        }

        let t_dtoh = t_dtoh_start.elapsed().as_micros();

        // Decode found entries
        let found: Vec<FoundBlock> = (0..count).map(|i| {
            let base = i * FOUND_STRIDE;
            let tile_i  = found_flat[base]     as usize * TM;
            let tile_j  = found_flat[base + 1] as usize * TN;
            let mut m_state = [0u32; 16];
            m_state.copy_from_slice(&found_flat[base + 2..base + 18]);
            let mut hash = [0u8; 32];
            for (j, &w) in found_flat[base + 18..base + 26].iter().enumerate() {
                hash[j * 4..(j + 1) * 4].copy_from_slice(&w.to_le_bytes());
            }
            FoundBlock { tile_i, tile_j, m_state, hash }
        }).collect();

        Ok((found, best_nonce, best_hash, [t_kernel, t_dtoh, 0]))
    }

    /// Solve the Stratum pool BLAKE3 challenge on GPU.
    /// Computes BLAKE3(seed_32 ++ nonce_le_8) and looks for a nonce where
    /// the hash has at least `difficulty` leading zero bits.
    /// Returns Some(nonce) on success or None if cancelled.
    pub fn solve_pool_challenge(
        &self,
        seed:       &[u8; 32],
        difficulty: u32,
        cancel:     &std::sync::atomic::AtomicBool,
    ) -> Result<Option<u64>, GpuError> {
        use std::sync::atomic::Ordering;

        let func = self.module.load_function("solve_pool_challenge")?;

        // Upload seed
        let seed_words: Vec<u32> = seed.chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mut d_seed = self.stream.alloc_zeros::<u32>(8)?;
        self.stream.memcpy_htod(&seed_words, &mut d_seed)?;

        // Allocate output buffers
        let mut d_nonce = self.stream.alloc_zeros::<u32>(2)?;
        let mut d_found = self.stream.alloc_zeros::<u32>(1)?;

        // 64 blocks × 1024 threads = 65 536 nonces per kernel launch
        const THREADS: u32 = 1024;
        const BLOCKS:  u32 = 64;
        const PER_LAUNCH: u64 = THREADS as u64 * BLOCKS as u64;

        let mut base_nonce: u64 = 0;
        loop {
            if cancel.load(Ordering::Relaxed) { return Ok(None); }

            // Reset found flag
            self.stream.memcpy_htod(&[0u32], &mut d_found)?;

            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (BLOCKS, 1, 1),
                block_dim: (THREADS, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = self.stream.launch_builder(&func);
            b.arg(&d_seed);
            b.arg(&base_nonce);
            b.arg(&difficulty);
            b.arg(&mut d_nonce);
            b.arg(&mut d_found);
            unsafe { b.launch(cfg) }?;
            self.stream.synchronize()?;

            let found_flag = self.stream.clone_dtoh(&d_found)?;
            if found_flag[0] != 0 {
                let nonce_words = self.stream.clone_dtoh(&d_nonce)?;
                let nonce = (nonce_words[0] as u64) | ((nonce_words[1] as u64) << 32);
                return Ok(Some(nonce));
            }

            base_nonce = base_nonce.wrapping_add(PER_LAUNCH);
            if base_nonce == 0 { return Ok(None); } // exhausted 64-bit space
        }
    }
}


fn difficulty_threshold(b: u32, r: usize, tm: usize, tn: usize) -> [u8; 32] {
    let mut val = [0u8; 32];
    let shift    = 256u32.saturating_sub(b);
    let byte_idx = (shift / 8) as usize;
    let bit_idx  = (shift % 8) as usize;
    if byte_idx < 32 { val[byte_idx] = 1u8 << bit_idx; }
    let multiplier = (r * tm * tn) as u64;
    let mut carry: u64 = 0;
    for byte in val.iter_mut() {
        let prod = (*byte as u64) * multiplier + carry;
        *byte = prod as u8;
        carry = prod >> 8;
    }
    val
}

