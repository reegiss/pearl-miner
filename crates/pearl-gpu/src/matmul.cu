/*
 * Pearl PoUW — Tiled INT8 MatMul kernels
 *
 * Two implementations selected at runtime:
 *   tiled_matmul_dp4a  — DP4A on CUDA cores (sm_61+, all NVIDIA GPUs)
 *   tiled_matmul_wmma  — Tensor cores via WMMA INT8 (sm_72+, RTX cards)
 *
 * WMMA layout (per tile):
 *   blockDim = (32, 1)  — 1 warp handles one 16×16 output tile
 *   gridDim  = (n/16, m/16)
 *   Depth step r=32 → 2 sub-steps of 16, each a 16×16×16 mma_sync call
 *   M-state XOR: each thread XORs its 8 accumulator elements, then
 *   warp butterfly reduction via __shfl_xor_sync
 */

#include <stdint.h>

/* ------------------------------------------------------------------ */
/*  Shared helpers                                                      */
/* ------------------------------------------------------------------ */

#define TM 16
#define TN 16
#define WARPS_PER_BLOCK_DP4A ((TM * TN) / 32)

__device__ __forceinline__ uint32_t rol32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

// splitmix64 — independent element seeding (no sequential state needed)
__device__ __forceinline__ uint64_t splitmix64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// Derive a unique seed for element (row, col) from base seed
__device__ __forceinline__ uint64_t elem_seed(uint64_t base, uint64_t row, uint64_t col) {
    return splitmix64(base ^ (row * 0x9e3779b97f4a7c15ULL) ^ (col * 0xd1b54a32d192ed03ULL));
}

/* ------------------------------------------------------------------ */
/*  Kernel C: generate A' = A + EL·ER entirely on GPU                  */
/* ------------------------------------------------------------------ */

/*
 * blockDim = (32, 16, 1) — 512 threads per block
 * gridDim  = ((k+31)/32, (m+15)/16, 1)
 *
 * Each thread computes one element of A' using splitmix64 PRNG:
 *   A[i][j]        ← elem_seed(job_seed, i, j)
 *   ER pos/neg_col ← elem_seed(sa_seed^0x100, 0, j) % r
 *   EL[i][pos_col] ← elem_seed(sa_seed^0x200, i, pos_col)
 *   A'[i][j]       = clamp(A[i][j] + EL[i][pos_col] − EL[i][neg_col])
 *
 * No inter-thread communication required — fully parallel.
 */
extern "C" __global__ void generate_noisy_a(
    int8_t*  __restrict__ A_prime,
    uint64_t job_seed,
    uint64_t sa_seed,
    int m, int k, int r
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= k) return;

    // A[row][col] in [-64, 63]
    int8_t a_val = (int8_t)((elem_seed(job_seed, (uint64_t)row, (uint64_t)col) & 0x7F) - 64);

    // ER: column `col` has one +1 at pos_col and one -1 at neg_col in [0, r)
    uint64_t er_s = elem_seed(sa_seed ^ 0x100ULL, 0ULL, (uint64_t)col);
    int pos_col   = (int)(er_s % (uint64_t)r);
    er_s = splitmix64(er_s);
    int neg_col   = (int)(er_s % (uint64_t)r);
    if (neg_col == pos_col) neg_col = (neg_col + 1) % r;

    // EL[row][pos_col] and EL[row][neg_col] in [-32, 31]
    int8_t el_pos = (int8_t)((elem_seed(sa_seed ^ 0x200ULL, (uint64_t)row, (uint64_t)pos_col) & 0x3F) - 32);
    int8_t el_neg = (int8_t)((elem_seed(sa_seed ^ 0x200ULL, (uint64_t)row, (uint64_t)neg_col) & 0x3F) - 32);

    int val = (int)a_val + (int)el_pos - (int)el_neg;
    if (val >  127) val =  127;
    if (val < -127) val = -127;
    A_prime[row * k + col] = (int8_t)val;
}

/* ------------------------------------------------------------------ */
/*  Kernel A: DP4A (CUDA cores, sm_61+)                                */
/* ------------------------------------------------------------------ */

extern "C" __global__ void tiled_matmul_dp4a(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t*      __restrict__ C,
    uint32_t*     __restrict__ M_out,
    int m, int n, int k, int r
) {
    extern __shared__ uint8_t smem[];

    int8_t*  As        = (int8_t*)smem;
    int8_t*  Bs        = (int8_t*)smem + TM * r;
    uint32_t* warp_xors = (uint32_t*)(smem + (TM + TN) * r);
    uint32_t* M         = warp_xors + WARPS_PER_BLOCK_DP4A;

    const int tile_i = blockIdx.y;
    const int tile_j = blockIdx.x;
    const int ty     = threadIdx.y;
    const int tx     = threadIdx.x;
    const int tid    = ty * TN + tx;
    const int lane   = tid & 31;
    const int warp   = tid >> 5;

    int32_t acc = 0;

    if (tid < 16) M[tid] = 0;
    __syncthreads();

    const int num_steps = k / r;

    for (int ell = 0; ell < num_steps; ell++) {
        const int s = ell * r;

        {
            int total = TM * r;
            for (int idx = tid; idx < total; idx += TM * TN) {
                int row = idx / r, col = idx % r;
                int grow = tile_i * TM + row;
                As[row * r + col] = (grow < m && (s + col) < k)
                    ? A[grow * k + s + col] : 0;
            }
        }
        {
            int total = r * TN;
            for (int idx = tid; idx < total; idx += TM * TN) {
                int row = idx / TN, col = idx % TN;
                int gcol = tile_j * TN + col;
                Bs[row * TN + col] = ((s + row) < k && gcol < n)
                    ? B[(s + row) * n + gcol] : 0;
            }
        }

        __syncthreads();

        const int r4 = r / 4;
        for (int q = 0; q < r4; q++) {
            int a_pack = ((int)As[ty * r + q*4+0] & 0xFF)
                       | ((int)As[ty * r + q*4+1] & 0xFF) << 8
                       | ((int)As[ty * r + q*4+2] & 0xFF) << 16
                       | ((int)As[ty * r + q*4+3] & 0xFF) << 24;
            int b_pack = ((int)Bs[(q*4+0) * TN + tx] & 0xFF)
                       | ((int)Bs[(q*4+1) * TN + tx] & 0xFF) << 8
                       | ((int)Bs[(q*4+2) * TN + tx] & 0xFF) << 16
                       | ((int)Bs[(q*4+3) * TN + tx] & 0xFF) << 24;
            acc = __dp4a(a_pack, b_pack, acc);
        }
        for (int q = r4 * 4; q < r; q++)
            acc += (int32_t)As[ty * r + q] * (int32_t)Bs[q * TN + tx];

        uint32_t wx = (uint32_t)acc;
        wx ^= __shfl_xor_sync(0xffffffff, wx, 16);
        wx ^= __shfl_xor_sync(0xffffffff, wx,  8);
        wx ^= __shfl_xor_sync(0xffffffff, wx,  4);
        wx ^= __shfl_xor_sync(0xffffffff, wx,  2);
        wx ^= __shfl_xor_sync(0xffffffff, wx,  1);
        if (lane == 0) warp_xors[warp] = wx;
        __syncthreads();

        if (tid == 0) {
            uint32_t X = 0;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK_DP4A; w++) X ^= warp_xors[w];
            M[ell & 15] = rol32(M[ell & 15], 13) ^ X;
        }
        __syncthreads();
    }

    if (tile_i * TM + ty < m && tile_j * TN + tx < n)
        C[(tile_i * TM + ty) * n + (tile_j * TN + tx)] = acc;

    if (tid == 0) {
        int num_tiles_n = (n + TN - 1) / TN;
        int base = (tile_i * num_tiles_n + tile_j) * 16;
        #pragma unroll
        for (int q = 0; q < 16; q++) M_out[base + q] = M[q];
    }
}

/* ------------------------------------------------------------------ */
/*  Kernel B: WMMA INT8 tensor cores (sm_72+, RTX cards)               */
/* ------------------------------------------------------------------ */

#if __CUDA_ARCH__ >= 720

#include <mma.h>
using namespace nvcuda::wmma;

extern "C" __global__ void tiled_matmul_wmma(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t*      __restrict__ C,
    uint32_t*     __restrict__ M_out,
    int m, int n, int k, int r
) {
    /*
     * 4 warps per block; each warp handles one 16×16 output tile.
     * blockDim = (32, 4, 1) → 128 threads = 4 warps
     * gridDim  = (n/16, (m+63)/64, 1)
     *
     * Shared memory per block (5 × r × 16 bytes):
     *   Bs[r × 16]          — B strip for this tile column (all 4 warps share)
     *   As_w[4][16 × r]     — A strip per warp (each warp writes its own section)
     *
     * This layout gives 3-4× more warps/SM than the old 1-warp design,
     * boosting SM occupancy from ~33% to ~100%.
     */
    extern __shared__ int32_t smem_wmma[];

    const int warp_id    = threadIdx.y;   // 0..3
    const int lane       = threadIdx.x;   // 0..31
    const int tid        = warp_id * 32 + lane;

    const int tile_i     = blockIdx.y * 4 + warp_id;
    const int tile_j     = blockIdx.x;
    const bool active    = (tile_i * 16 < m && tile_j * 16 < n);
    const int  a_row_base = tile_i * 16;
    const int  b_col_base = tile_j * 16;

    // Shared memory pointers
    int8_t* Bs   = (int8_t*)smem_wmma;
    int8_t* wAs  = Bs + r * 16 + warp_id * 16 * r;

    fragment<accumulator, 16, 16, 16, int32_t> acc;
    fill_fragment(acc, 0);

    uint32_t M[16];
    #pragma unroll
    for (int q = 0; q < 16; q++) M[q] = 0;

    const int num_steps = k / r;

    for (int ell = 0; ell < num_steps; ell++) {
        const int s = ell * r;

        // All 128 threads cooperatively load Bs[r × 16]
        for (int idx = tid; idx < r * 16; idx += 128) {
            int row = idx / 16, col = idx % 16;
            int gcol = b_col_base + col;
            Bs[row * 16 + col] = ((s + row) < k && gcol < n)
                ? B[(s + row) * n + gcol] : (int8_t)0;
        }
        __syncthreads();  // Bs ready for all warps

        // Each warp loads its own As[16 × r]
        if (active) {
            for (int idx = lane; idx < 16 * r; idx += 32) {
                int row = idx / r, col = idx % r;
                int grow = a_row_base + row;
                wAs[idx] = (grow < m && (s + col) < k)
                    ? A[grow * k + s + col] : (int8_t)0;
            }
        }
        __syncwarp();

        // WMMA sub-steps
        if (active) {
            const int sub_steps = r / 16;
            for (int sub = 0; sub < sub_steps; sub++) {
                fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
                fragment<matrix_b, 16, 16, 16, int8_t, row_major> b_frag;
                load_matrix_sync(a_frag, wAs + sub * 16,       r);
                load_matrix_sync(b_frag, Bs  + sub * 16 * 16, 16);
                mma_sync(acc, a_frag, b_frag, acc);
            }

            uint32_t local_xor = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++) local_xor ^= (uint32_t)acc.x[i];
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor, 16);
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  8);
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  4);
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  2);
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  1);
            if (lane == 0)
                M[ell & 15] = rol32(M[ell & 15], 13) ^ local_xor;
        }

        __syncthreads();  // all warps done before next Bs overwrite
    }

    if (!active) return;

    store_matrix_sync(C + a_row_base * n + b_col_base, acc, n, mem_row_major);

    if (lane == 0) {
        int num_tiles_n = (n + 15) / 16;
        int base = (tile_i * num_tiles_n + tile_j) * 16;
        #pragma unroll
        for (int q = 0; q < 16; q++) M_out[base + q] = M[q];
    }
}

#else

extern "C" __global__ void tiled_matmul_wmma(
    const int8_t*, const int8_t*, int32_t*, uint32_t*,
    int, int, int, int) {}

#endif
