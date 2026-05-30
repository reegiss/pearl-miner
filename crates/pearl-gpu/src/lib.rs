use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use pearl_types::{FoundBlock, MiningParams};
use rayon::prelude::*;
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
    ctx:        Arc<CudaContext>,
    stream:     Arc<CudaStream>,
    #[allow(dead_code)]
    module:     Arc<CudaModule>,
    func_dp4a:  CudaFunction,
    func_wmma:  CudaFunction,
    func_gen_a: CudaFunction,
    use_wmma:   bool,
    d_a: Mutex<CudaSlice<i8>>,   // A' (m×k) — written by generate_noisy_a kernel
    d_b: Mutex<CudaSlice<i8>>,   // B' (k×n) — uploaded once per challenge
    d_c: Mutex<CudaSlice<i32>>,  // C  (m×n) — written by matmul kernel
    d_m: Mutex<CudaSlice<u32>>,  // M states (tiles×16) — written by matmul kernel
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

        let func_dp4a  = module.load_function("tiled_matmul_dp4a")?;
        let func_wmma  = module.load_function("tiled_matmul_wmma")?;
        let func_gen_a = module.load_function("generate_noisy_a")?;
        let use_wmma   = has_tensor_cores(&name);

        let (m, n, k, r) = (params.m, params.n, params.k, params.r);
        let num_tiles_m  = (m + TM - 1) / TM;
        let num_tiles_n  = (n + TN - 1) / TN;
        let num_tiles    = num_tiles_m * num_tiles_n;

        let d_a = Mutex::new(stream.alloc_zeros::<i8>(m * k)?);
        let d_b = Mutex::new(stream.alloc_zeros::<i8>(k * n)?);
        let d_c = Mutex::new(stream.alloc_zeros::<i32>(1)?);  // placeholder — kernel never writes C
        let d_m = Mutex::new(stream.alloc_zeros::<u32>(num_tiles * 16)?);

        Ok(Self {
            ctx, stream, module,
            func_dp4a, func_wmma, func_gen_a, use_wmma,
            d_a, d_b, d_c, d_m,
            m, n, k, r, num_tiles_m, num_tiles_n,
            info: GpuInfo { name, mem_mb, use_wmma },
        })
    }

    /// Upload B' to GPU — call once per challenge (B' is constant per challenge).
    pub fn set_b(&self, b_prime: &[i8]) -> Result<(), GpuError> {
        self.stream.memcpy_htod(b_prime, &mut *self.d_b.lock().unwrap())?;
        Ok(())
    }

    /// Generate A' = A + EL·ER entirely on GPU using splitmix64 PRNG.
    /// Call before `mine()` each job. No CPU-side matrix work required.
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

    /// Run the matmul kernel on d_a (already filled) and check BLAKE3 difficulty.
    /// Returns (found_blocks, [t_kernel_us, t_dtoh_us, t_blake3_us]).
    pub fn mine(
        &self,
        params: &MiningParams,
        s_a:    &[u8; 32],
    ) -> Result<(Vec<FoundBlock>, [u128; 3]), GpuError> {
        let (m, n, k, r)    = (self.m, self.n, self.k, self.r);
        let (ntm, ntn)      = (self.num_tiles_m, self.num_tiles_n);
        let (mi, ni, ki, ri) = (m as i32, n as i32, k as i32, r as i32);

        // Acquire all four buffers for kernel launch (no contention — 1 thread/GPU)
        let d_a  = self.d_a.lock().unwrap();
        let d_b  = self.d_b.lock().unwrap();
        let mut d_c = self.d_c.lock().unwrap();
        let mut d_m = self.d_m.lock().unwrap();

        let t_kernel_start = std::time::Instant::now();
        if self.use_wmma {
            // 8 warps per block: Bs(r×16) + 8×As(16×r) = 9×r×16 bytes
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
        self.stream.synchronize()?;
        let t_kernel = t_kernel_start.elapsed().as_micros();

        let t_dtoh_start = std::time::Instant::now();
        let m_states = self.stream.clone_dtoh(&*d_m)?;
        let t_dtoh = t_dtoh_start.elapsed().as_micros();

        let threshold = difficulty_threshold(params.difficulty, r, TM, TN);
        let num_tiles = ntm * ntn;

        let t_blake3_start = std::time::Instant::now();
        let found: Vec<FoundBlock> = (0..num_tiles).into_par_iter().filter_map(|idx| {
            let ti = idx / ntn;
            let tj = idx % ntn;
            let base = idx * 16;
            let mut m_state = [0u32; 16];
            m_state.copy_from_slice(&m_states[base..base + 16]);
            let hash = blake3_m_hash(&m_state, s_a);
            if hash_le_threshold(&hash, &threshold) {
                Some(FoundBlock { tile_i: ti * TM, tile_j: tj * TN, m_state, hash })
            } else {
                None
            }
        }).collect();
        let t_blake3 = t_blake3_start.elapsed().as_micros();

        Ok((found, [t_kernel, t_dtoh, t_blake3]))
    }
}

fn has_tensor_cores(name: &str) -> bool {
    let n = name.to_ascii_uppercase();
    n.contains("RTX")
        || n.contains("A100") || n.contains("H100") || n.contains("H200")
        || n.contains("V100") || n.contains("T4")   || n.contains("A10")
        || n.contains("A30")  || n.contains("A40")
}

fn blake3_m_hash(m: &[u32; 16], s_a: &[u8; 32]) -> [u8; 32] {
    let mut msg = [0u8; 64];
    for (i, &v) in m.iter().enumerate() {
        msg[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
    }
    *blake3::keyed_hash(s_a, &msg).as_bytes()
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
