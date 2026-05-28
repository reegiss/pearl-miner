use std::ffi::c_void;
use std::sync::Arc;

use anyhow::{bail, Result};
use cudarc::driver::{CudaDevice, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use num_bigint::BigUint;
use num_traits::One;
use pearl_types::{Commitments, FoundTile, MiningConfig};

/// Returns the number of CUDA-capable devices available on this machine.
/// Returns 0 if CUDA is unavailable or no devices are found.
pub fn device_count() -> usize {
    CudaDevice::count().unwrap_or(0).max(0) as usize
}

// Pre-compiled PTX path — built by build.rs with -arch=sm_75
const PTX_FILE: &str = concat!(env!("OUT_DIR"), "/matmul.ptx");

// Must match FoundTileGpu in matmul.cu (104 bytes, 4-byte aligned)
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct GpuFoundTile {
    tile_i:     u32,
    tile_j:     u32,
    m_state:    [i32; 16],
    final_hash: [u8; 32],
}

unsafe impl cudarc::driver::DeviceRepr for GpuFoundTile {}
// All-zero bit pattern is valid: u32/i32/u8 fields have no invalid representations.
unsafe impl cudarc::driver::ValidAsZeroBits for GpuFoundTile {}

const MODULE:    &str = "pearl_matmul";
const KERNEL:    &str = "tiled_matmul_mine";
const MAX_FOUND: i32  = 4096;

macro_rules! gpu {
    ($e:expr) => {
        $e.map_err(|e| anyhow::anyhow!("CUDA: {e:?}"))?
    };
}

pub struct GpuMiner {
    device: Arc<CudaDevice>,
}

impl GpuMiner {
    pub fn new(device_id: u32) -> Result<Self> {
        let device = gpu!(CudaDevice::new(device_id as usize));

        // Load pre-compiled PTX (built with -arch=sm_75 by build.rs)
        let ptx = cudarc::nvrtc::Ptx::from_file(PTX_FILE);
        gpu!(device.load_ptx(ptx, MODULE, &[KERNEL]));

        Ok(GpuMiner { device })
    }

    /// Tiled INT8 matmul on GPU.
    /// Returns (found_tiles, C = A'·B' as m×n INT32 row-major).
    pub fn mine(
        &self,
        a_noisy: &[i8],
        b_noisy: &[i8],
        commitments: &Commitments,
        config: &MiningConfig,
    ) -> Result<(Vec<FoundTile>, Vec<i32>)> {
        let p = &config.params;
        let (m, n, k, r, tm, tn) = (p.m, p.n, p.k, p.r, p.tm, p.tn);

        if tm * tn > 1024 {
            bail!("tm*tn={} exceeds max 1024 CUDA threads/block", tm * tn);
        }
        if k % r != 0 {
            bail!("k={k} must be divisible by r={r}");
        }

        let dev = &self.device;
        let f = dev.get_func(MODULE, KERNEL)
            .ok_or_else(|| anyhow::anyhow!("kernel '{KERNEL}' not found in module '{MODULE}'"))?;

        // Host → device transfers
        let a_dev: CudaSlice<i8>  = gpu!(dev.htod_sync_copy(a_noisy));
        let b_dev: CudaSlice<i8>  = gpu!(dev.htod_sync_copy(b_noisy));
        let c_dev: CudaSlice<i32> = gpu!(dev.alloc_zeros::<i32>((m * n) as usize));

        let found_dev: CudaSlice<GpuFoundTile> =
            gpu!(dev.alloc_zeros::<GpuFoundTile>(MAX_FOUND as usize));
        let count_dev: CudaSlice<i32> = gpu!(dev.htod_sync_copy(&[0i32]));

        let sa_dev:     CudaSlice<u8> = gpu!(dev.htod_sync_copy(&commitments.s_a));
        let threshold                  = compute_threshold(config.difficulty_bits, r, tm, tn);
        let thresh_dev: CudaSlice<u8> = gpu!(dev.htod_sync_copy(&threshold));

        // Raw device pointers (CUdeviceptr = u64)
        let a_ptr:     u64 = *a_dev.device_ptr();
        let b_ptr:     u64 = *b_dev.device_ptr();
        let c_ptr:     u64 = *c_dev.device_ptr();
        let found_ptr: u64 = *found_dev.device_ptr();
        let count_ptr: u64 = *count_dev.device_ptr();
        let sa_ptr:    u64 = *sa_dev.device_ptr();
        let thr_ptr:   u64 = *thresh_dev.device_ptr();

        // Scalar args (must remain live until after launch)
        let max_f = MAX_FOUND;
        let (mv, nv, kv, rv, tv, tnv) = (m, n, k, r, tm, tn);

        // Build raw args list (kernel arg i → pointer to value of arg i)
        let mut args: Vec<*mut c_void> = vec![
            &a_ptr     as *const _ as *mut _,
            &b_ptr     as *const _ as *mut _,
            &c_ptr     as *const _ as *mut _,
            &found_ptr as *const _ as *mut _,
            &count_ptr as *const _ as *mut _,
            &max_f     as *const _ as *mut _,
            &sa_ptr    as *const _ as *mut _,
            &thr_ptr   as *const _ as *mut _,
            &mv  as *const _ as *mut _,
            &nv  as *const _ as *mut _,
            &kv  as *const _ as *mut _,
            &rv  as *const _ as *mut _,
            &tv  as *const _ as *mut _,
            &tnv as *const _ as *mut _,
        ];

        let grid_x = m.div_ceil(tm);
        let grid_y = n.div_ceil(tn);
        let smem   = (tm * tn + 16) * 4;

        let cfg = LaunchConfig {
            grid_dim:         (grid_x, grid_y, 1),
            block_dim:        (tm, tn, 1),
            shared_mem_bytes: smem,
        };

        unsafe { gpu!(f.launch(cfg, &mut args)) };
        gpu!(dev.synchronize());

        // Device → host
        let count     = gpu!(dev.dtoh_sync_copy(&count_dev))[0];
        let gpu_tiles = gpu!(dev.dtoh_sync_copy(&found_dev));
        let c_host    = gpu!(dev.dtoh_sync_copy(&c_dev));

        let n_found = (count.min(MAX_FOUND)) as usize;
        let found: Vec<FoundTile> = gpu_tiles[..n_found]
            .iter()
            .map(|t| FoundTile {
                tile_i:     t.tile_i,
                tile_j:     t.tile_j,
                m_state:    t.m_state,
                final_hash: t.final_hash,
            })
            .collect();

        Ok((found, c_host))
    }
}

/// 32-byte LE uint256 threshold: floor(2^(256 − floor(b))) · r · tm · tn
/// If the result exceeds 2^256 − 1, clamp to 0xFF…FF (all hashes pass).
pub fn compute_threshold(b: f64, r: u32, tm: u32, tn: u32) -> [u8; 32] {
    let shift = 256u32.saturating_sub(b.floor() as u32);
    let val = (BigUint::one() << shift) * r * tm * tn;
    let bytes = val.to_bytes_le();
    if bytes.len() > 32 {
        return [0xFF; 32];
    }
    let mut out = [0u8; 32];
    out[..bytes.len()].copy_from_slice(&bytes);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use pearl_types::{MatrixParams, MiningConfig};

    fn cfg() -> MiningConfig {
        MiningConfig {
            params: MatrixParams { m: 8, n: 8, k: 64, r: 32, tm: 4, tn: 4 },
            difficulty_bits: 0.0,  // threshold = max → every tile wins
            sigma: b"test".to_vec(),
            mu:    b"test".to_vec(),
        }
    }

    #[test]
    fn test_gpu_miner_finds_all_tiles_at_zero_difficulty() {
        let miner = GpuMiner::new(0).expect("GPU init failed");
        let config = cfg();
        let p = &config.params;
        let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);

        let a: Vec<i8> = (0..m * k).map(|i| (i % 64) as i8 - 32).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (i % 32) as i8 - 16).collect();

        let commitments = Commitments {
            kappa: [0u8; 32], ha: [0u8; 32], hb: [0u8; 32],
            s_a:   [1u8; 32], s_b: [0u8; 32],
        };

        let (found, c_out) = miner.mine(&a, &b, &commitments, &config)
            .expect("mine failed");

        // With b=0, threshold is effectively max; all (m/tm)*(n/tn) = 4 tiles should win
        let expected_tiles = (p.m / p.tm) * (p.n / p.tn);
        assert_eq!(found.len() as u32, expected_tiles,
            "expected {expected_tiles} winning tiles, got {}", found.len());

        // Output matrix has correct size
        assert_eq!(c_out.len(), m * n);
    }

    #[test]
    fn test_gpu_c_matches_cpu_naive() {
        let miner = GpuMiner::new(0).expect("GPU init failed");
        let config = cfg();
        let p = &config.params;
        let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);

        // Small deterministic matrices
        let a: Vec<i8> = (0..m * k).map(|i| (i % 7) as i8 - 3).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (i % 5) as i8 - 2).collect();

        let commitments = Commitments {
            kappa: [0u8; 32], ha: [0u8; 32], hb: [0u8; 32],
            s_a:   [1u8; 32], s_b: [0u8; 32],
        };

        let (_, c_gpu) = miner.mine(&a, &b, &commitments, &config)
            .expect("mine failed");

        // CPU reference: A row-major × B col-major → C[i*n+j] = sum_l A[i*k+l]*B[j*k+l]
        let mut c_cpu = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    c_cpu[i * n + j] += a[i * k + l] as i32 * b[j * k + l] as i32;
                }
            }
        }

        assert_eq!(c_gpu, c_cpu, "GPU output does not match CPU reference");
    }
}
