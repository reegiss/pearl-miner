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
    pub name:    String,
    pub mem_mb:  usize,
}

/// Number of CUDA-capable devices available on this machine.
pub fn device_count() -> usize {
    CudaContext::device_count().unwrap_or(0) as usize
}

pub struct GpuMiner {
    #[allow(dead_code)]
    ctx:    Arc<CudaContext>,
    stream: Arc<CudaStream>,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    func:   CudaFunction,
    pub info: GpuInfo,
}

impl GpuMiner {
    pub fn new(device_id: usize) -> Result<Self, GpuError> {
        let ctx = CudaContext::new(device_id)
            .map_err(|_| GpuError::NoDevice(device_id))?;
        let name   = ctx.name().unwrap_or_else(|_| "Unknown GPU".into());
        let mem_mb = ctx.total_mem().unwrap_or(0) / (1024 * 1024);
        let stream = ctx.default_stream();
        let module = ctx.load_module(Ptx::from_src(MATMUL_PTX))?;
        let func   = module.load_function("tiled_matmul")?;
        Ok(Self { ctx, stream, module, func, info: GpuInfo { name, mem_mb } })
    }

    /// Run tiled matmul on A' and B', return tiles that pass the BLAKE3 difficulty check.
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

        // Upload inputs
        let d_a = self.stream.clone_htod(a_prime)?;
        let d_b = self.stream.clone_htod(b_prime)?;
        let mut d_c = self.stream.alloc_zeros::<i32>(m * n)?;
        let mut d_m = self.stream.alloc_zeros::<u32>(num_tiles * 16)?;

        // Dynamic shared memory:
        //   As[TM×r] + Bs[r×TN] + warp_xors[WARPS×4] + M[16×4]
        let warps = (TM * TN) / 32;
        let shared_bytes = ((TM + TN) * r + warps * 4 + 16 * 4) as u32;

        let cfg = LaunchConfig {
            grid_dim:         (num_tiles_n as u32, num_tiles_m as u32, 1),
            block_dim:        (TN as u32, TM as u32, 1),
            shared_mem_bytes: shared_bytes,
        };

        let (mi, ni, ki, ri) = (m as i32, n as i32, k as i32, r as i32);
        let mut builder = self.stream.launch_builder(&self.func);
        builder.arg(&d_a);
        builder.arg(&d_b);
        builder.arg(&mut d_c);
        builder.arg(&mut d_m);
        builder.arg(&mi);
        builder.arg(&ni);
        builder.arg(&ki);
        builder.arg(&ri);
        unsafe { builder.launch(cfg) }?;

        // Download M states to CPU for BLAKE3 checking
        let m_states = self.stream.clone_dtoh(&d_m)?;

        // BLAKE3 difficulty check on CPU
        let threshold = difficulty_threshold(params.difficulty, r, TM, TN);
        let mut found = Vec::new();

        for ti in 0..num_tiles_m {
            for tj in 0..num_tiles_n {
                let base = (ti * num_tiles_n + tj) * 16;
                let mut m_state = [0u32; 16];
                m_state.copy_from_slice(&m_states[base..base + 16]);

                let hash = blake3_m_hash(&m_state, s_a);
                if hash_le_threshold(&hash, &threshold) {
                    found.push(FoundBlock {
                        tile_i: ti * TM,
                        tile_j: tj * TN,
                        m_state,
                        hash,
                    });
                }
            }
        }

        Ok(found)
    }
}

fn blake3_m_hash(m: &[u32; 16], s_a: &[u8; 32]) -> [u8; 32] {
    let mut msg = [0u8; 64];
    for (i, &v) in m.iter().enumerate() {
        msg[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
    }
    *blake3::keyed_hash(s_a, &msg).as_bytes()
}

/// Compute 2^(256−b) × r × tm × tn as a 32-byte little-endian uint256.
fn difficulty_threshold(b: u32, r: usize, tm: usize, tn: usize) -> [u8; 32] {
    let mut val = [0u8; 32];
    let shift = 256u32.saturating_sub(b);
    let byte_idx = (shift / 8) as usize;
    let bit_idx  = (shift % 8) as usize;
    if byte_idx < 32 {
        val[byte_idx] = 1u8 << bit_idx;
    }
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
