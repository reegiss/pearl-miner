use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use pearl_types::{FoundBlock, MiningParams};
use std::sync::Arc;
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
    ctx:       Arc<CudaContext>,
    stream:    Arc<CudaStream>,
    #[allow(dead_code)]
    module:    Arc<CudaModule>,
    func_dp4a: CudaFunction,
    func_wmma: CudaFunction,
    use_wmma:  bool,
    pub info:  GpuInfo,
}

impl GpuMiner {
    pub fn new(device_id: usize) -> Result<Self, GpuError> {
        let ctx    = CudaContext::new(device_id).map_err(|_| GpuError::NoDevice(device_id))?;
        let name   = ctx.name().unwrap_or_else(|_| "Unknown GPU".into());
        let mem_mb = ctx.total_mem().unwrap_or(0) / (1024 * 1024);
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_src(MATMUL_PTX))?;

        let func_dp4a = module.load_function("tiled_matmul_dp4a")?;
        let func_wmma = module.load_function("tiled_matmul_wmma")?;

        // RTX cards (Turing/Ampere/Ada/Blackwell) have INT8 tensor cores.
        // GTX 16xx is also sm75 but without tensor cores — detect by name.
        let use_wmma = has_tensor_cores(&name);

        Ok(Self {
            ctx, stream, module,
            func_dp4a, func_wmma, use_wmma,
            info: GpuInfo { name, mem_mb, use_wmma },
        })
    }

    pub fn mine(
        &self,
        a_prime: &[i8],
        b_prime: &[i8],
        params:  &MiningParams,
        s_a:     &[u8; 32],
    ) -> Result<Vec<FoundBlock>, GpuError> {
        let m = params.m;
        let n = params.n;
        let k = params.k;
        let r = params.r;

        let num_tiles_m = (m + TM - 1) / TM;
        let num_tiles_n = (n + TN - 1) / TN;
        let num_tiles   = num_tiles_m * num_tiles_n;

        let d_a = self.stream.clone_htod(a_prime)?;
        let d_b = self.stream.clone_htod(b_prime)?;
        let mut d_c = self.stream.alloc_zeros::<i32>(m * n)?;
        let mut d_m = self.stream.alloc_zeros::<u32>(num_tiles * 16)?;

        let (mi, ni, ki, ri) = (m as i32, n as i32, k as i32, r as i32);

        if self.use_wmma {
            // WMMA: 1 warp (32 threads) per tile, no shared memory needed
            let cfg = LaunchConfig {
                grid_dim:         (num_tiles_n as u32, num_tiles_m as u32, 1),
                block_dim:        (32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = self.stream.launch_builder(&self.func_wmma);
            b.arg(&d_a); b.arg(&d_b); b.arg(&mut d_c); b.arg(&mut d_m);
            b.arg(&mi); b.arg(&ni); b.arg(&ki); b.arg(&ri);
            unsafe { b.launch(cfg) }?;
        } else {
            // DP4A: TM×TN threads per tile, dynamic shared memory
            let warps        = (TM * TN) / 32;
            let shared_bytes = ((TM + TN) * r + warps * 4 + 16 * 4) as u32;
            let cfg = LaunchConfig {
                grid_dim:         (num_tiles_n as u32, num_tiles_m as u32, 1),
                block_dim:        (TN as u32, TM as u32, 1),
                shared_mem_bytes: shared_bytes,
            };
            let mut b = self.stream.launch_builder(&self.func_dp4a);
            b.arg(&d_a); b.arg(&d_b); b.arg(&mut d_c); b.arg(&mut d_m);
            b.arg(&mi); b.arg(&ni); b.arg(&ki); b.arg(&ri);
            unsafe { b.launch(cfg) }?;
        }

        let m_states  = self.stream.clone_dtoh(&d_m)?;
        let threshold = difficulty_threshold(params.difficulty, r, TM, TN);
        let mut found = Vec::new();

        for ti in 0..num_tiles_m {
            for tj in 0..num_tiles_n {
                let base = (ti * num_tiles_n + tj) * 16;
                let mut m_state = [0u32; 16];
                m_state.copy_from_slice(&m_states[base..base + 16]);
                let hash = blake3_m_hash(&m_state, s_a);
                if hash_le_threshold(&hash, &threshold) {
                    found.push(FoundBlock { tile_i: ti * TM, tile_j: tj * TN, m_state, hash });
                }
            }
        }

        Ok(found)
    }
}

/// Returns true if the GPU likely has dedicated INT8 tensor cores.
/// GTX 16xx is sm75 but manufactured without tensor units.
fn has_tensor_cores(name: &str) -> bool {
    let n = name.to_ascii_uppercase();
    // RTX series: tensor cores present
    if n.contains("RTX") { return true; }
    // Datacenter GPUs with tensor cores
    if n.contains("A100") || n.contains("H100") || n.contains("H200")
        || n.contains("V100") || n.contains("T4") || n.contains("A10")
        || n.contains("A30") || n.contains("A40") { return true; }
    false
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
