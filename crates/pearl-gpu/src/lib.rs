use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
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
const FOUND_STRIDE: usize = 26; // u32s per found entry: 2 tile coords + 16 m_state + 8 hash
const MAX_FOUND: usize = 256;

static MATMUL_PTX: &str = include_str!(env!("MATMUL_PTX_PATH"));

pub struct GpuInfo {
    pub name:     String,
    pub mem_mb:   usize,
    pub use_wmma: bool,
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
    func_dp4a:    CudaFunction,
    func_wmma:    CudaFunction,
    func_gen_a:   CudaFunction,
    func_blake3:  CudaFunction,
    use_wmma:     bool,
    d_a:           Mutex<CudaSlice<i8>>,   // A' (m×k)
    d_b:           Mutex<CudaSlice<i8>>,   // B' (k×n) — set once per challenge
    d_c:           Mutex<CudaSlice<i32>>,  // placeholder (kernel never writes C)
    d_m:           Mutex<CudaSlice<u32>>,  // M states (num_tiles×16)
    d_sa:          Mutex<CudaSlice<u32>>,  // sA key as 8 u32 words (per job)
    d_threshold:   Mutex<CudaSlice<u32>>,  // difficulty threshold as 8 u32 words
    d_found:       Mutex<CudaSlice<u32>>,  // found tiles: MAX_FOUND × FOUND_STRIDE
    d_found_count: Mutex<CudaSlice<u32>>,  // atomic counter (1 u32)
    m: usize, n: usize, k: usize, r: usize,
    num_tiles_m: usize, num_tiles_n: usize,
    pub info: GpuInfo,
}

impl GpuMiner {
    pub fn new(device_id: usize, params: &MiningParams) -> Result<Self, GpuError> {
        let ctx    = CudaContext::new(device_id).map_err(|_| GpuError::NoDevice(device_id))?;
        let name   = ctx.name().unwrap_or_else(|_| "Unknown GPU".into());
        let mem_mb = ctx.total_mem().unwrap_or(0) / (1024 * 1024);
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_src(MATMUL_PTX))?;

        let func_dp4a   = module.load_function("tiled_matmul_dp4a")?;
        let func_wmma   = module.load_function("tiled_matmul_wmma")?;
        let func_gen_a  = module.load_function("generate_noisy_a")?;
        let func_blake3 = module.load_function("blake3_check")?;
        let use_wmma    = has_tensor_cores(&name);

        let (m, n, k, r) = (params.m, params.n, params.k, params.r);
        let num_tiles_m  = (m + TM - 1) / TM;
        let num_tiles_n  = (n + TN - 1) / TN;
        let num_tiles    = num_tiles_m * num_tiles_n;

        let d_a           = Mutex::new(stream.alloc_zeros::<i8>(m * k)?);
        let d_b           = Mutex::new(stream.alloc_zeros::<i8>(k * n)?);
        let d_c           = Mutex::new(stream.alloc_zeros::<i32>(1)?);
        let d_m           = Mutex::new(stream.alloc_zeros::<u32>(num_tiles * 16)?);
        let d_sa          = Mutex::new(stream.alloc_zeros::<u32>(8)?);
        let d_threshold   = Mutex::new(stream.alloc_zeros::<u32>(8)?);
        let d_found       = Mutex::new(stream.alloc_zeros::<u32>(MAX_FOUND * FOUND_STRIDE)?);
        let d_found_count = Mutex::new(stream.alloc_zeros::<u32>(1)?);

        Ok(Self {
            ctx, stream, module,
            func_dp4a, func_wmma, func_gen_a, func_blake3, use_wmma,
            d_a, d_b, d_c, d_m, d_sa, d_threshold, d_found, d_found_count,
            m, n, k, r, num_tiles_m, num_tiles_n,
            info: GpuInfo { name, mem_mb, use_wmma },
        })
    }

    /// Upload B' to GPU — call once per challenge.
    pub fn set_b(&self, b_prime: &[i8]) -> Result<(), GpuError> {
        self.stream.memcpy_htod(b_prime, &mut *self.d_b.lock().unwrap())?;
        Ok(())
    }

    /// Generate A' = A + EL·ER entirely on GPU using splitmix64 PRNG.
    pub fn generate_noisy_a(
        &self,
        job_seed: u64,
        sa_seed:  &[u8; 32],
        params:   &MiningParams,
    ) -> Result<(), GpuError> {
        let sa_u64: u64 = u64::from_le_bytes(sa_seed[..8].try_into().unwrap());
        let (mi, ki, ri) = (params.m as i32, params.k as i32, params.r as i32);
        let bx = 32u32; let by = 16u32;
        let gx = (params.k as u32 + bx - 1) / bx;
        let gy = (params.m as u32 + by - 1) / by;
        let cfg = LaunchConfig { grid_dim: (gx, gy, 1), block_dim: (bx, by, 1), shared_mem_bytes: 0 };
        let mut d_a = self.d_a.lock().unwrap();
        let mut b   = self.stream.launch_builder(&self.func_gen_a);
        b.arg(&mut *d_a); b.arg(&job_seed); b.arg(&sa_u64); b.arg(&mi); b.arg(&ki); b.arg(&ri);
        unsafe { b.launch(cfg) }?;
        Ok(())
    }

    /// Run matmul + GPU-side BLAKE3 difficulty check. Returns found blocks and stage timings.
    pub fn mine(
        &self,
        params: &MiningParams,
        s_a:    &[u8; 32],
    ) -> Result<(Vec<FoundBlock>, [u128; 3]), GpuError> {
        let (m, n, k, r)     = (self.m, self.n, self.k, self.r);
        let (ntm, ntn)       = (self.num_tiles_m, self.num_tiles_n);
        let (mi, ni, ki, ri) = (m as i32, n as i32, k as i32, r as i32);
        let num_tiles        = ntm * ntn;

        // Upload sA key (8 u32 LE words)
        let sa_words: Vec<u32> = s_a.chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        self.stream.memcpy_htod(&sa_words, &mut *self.d_sa.lock().unwrap())?;

        // Upload threshold (8 u32 LE words)
        let thresh_bytes = difficulty_threshold(params.difficulty, r, TM, TN);
        let thresh_words: Vec<u32> = thresh_bytes.chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        self.stream.memcpy_htod(&thresh_words, &mut *self.d_threshold.lock().unwrap())?;

        // Reset found counter to 0
        let zero = [0u32; 1];
        self.stream.memcpy_htod(&zero, &mut *self.d_found_count.lock().unwrap())?;

        // Lock all buffers
        let d_a  = self.d_a.lock().unwrap();
        let d_b  = self.d_b.lock().unwrap();
        let mut d_c = self.d_c.lock().unwrap();
        let mut d_m = self.d_m.lock().unwrap();
        let d_sa  = self.d_sa.lock().unwrap();
        let d_thr = self.d_threshold.lock().unwrap();
        let mut d_found       = self.d_found.lock().unwrap();
        let mut d_found_count = self.d_found_count.lock().unwrap();

        // --- Matmul kernel ---
        let t_kernel_start = std::time::Instant::now();
        if self.use_wmma {
            let smem = (9 * r * 16) as u32;
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

        self.stream.synchronize()?;
        let t_kernel = t_kernel_start.elapsed().as_micros();

        // --- D2H: only the count + winning tiles (bytes, not megabytes) ---
        let t_dtoh_start = std::time::Instant::now();
        let count_vec = self.stream.clone_dtoh(&*d_found_count)?;
        let count = (count_vec[0] as usize).min(MAX_FOUND);
        let found_flat = if count > 0 {
            self.stream.clone_dtoh(&*d_found)?
        } else {
            vec![]
        };
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

        Ok((found, [t_kernel, t_dtoh, 0]))
    }
}

fn has_tensor_cores(name: &str) -> bool {
    let n = name.to_ascii_uppercase();
    n.contains("RTX")
        || n.contains("A100") || n.contains("H100") || n.contains("H200")
        || n.contains("V100") || n.contains("T4")   || n.contains("A10")
        || n.contains("A30")  || n.contains("A40")
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

fn hash_le_threshold(hash: &[u8; 32], threshold: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        match hash[i].cmp(&threshold[i]) {
            std::cmp::Ordering::Less    => return true,
            std::cmp::Ordering::Greater => return false,
            std::cmp::Ordering::Equal   => continue,
        }
    }
    true
}
