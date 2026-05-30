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
     * Shared memory layout (per block = 1 warp):
     *   As[16 × r]   — A strip for this tile, row-major, stride=r
     *   Bs[r  × 16]  — B strip for this tile, row-major, stride=16
     *
     * All 32 lanes load As and Bs cooperatively (coalesced), then
     * call load_matrix_sync from shared memory for each WMMA sub-step.
     */
    // 4-byte aligned shared memory — required for load_matrix_sync with int8_t
    extern __shared__ int32_t smem_wmma_i32[];
    int8_t* As = (int8_t*)smem_wmma_i32;           // 16 × r bytes
    int8_t* Bs = (int8_t*)smem_wmma_i32 + 16 * r;  // r  × 16 bytes

    const int tile_i      = blockIdx.y;
    const int tile_j      = blockIdx.x;
    const int lane        = threadIdx.x & 31;
    const int a_row_start = tile_i * 16;
    const int b_col_start = tile_j * 16;

    if (a_row_start >= m || b_col_start >= n) return;

    fragment<accumulator, 16, 16, 16, int32_t> acc;
    fill_fragment(acc, 0);

    uint32_t M[16];
    #pragma unroll
    for (int q = 0; q < 16; q++) M[q] = 0;

    const int num_steps = k / r;

    for (int ell = 0; ell < num_steps; ell++) {
        const int s = ell * r;

        // Cooperative coalesced load: 32 threads fill 16×r and r×16 strips
        for (int idx = lane; idx < 16 * r; idx += 32) {
            int row = idx / r, col = idx % r;
            int grow = a_row_start + row;
            As[row * r + col] = (grow < m && (s + col) < k)
                ? A[grow * k + s + col] : (int8_t)0;
        }
        for (int idx = lane; idx < r * 16; idx += 32) {
            int row = idx / 16, col = idx % 16;
            int gcol = b_col_start + col;
            Bs[row * 16 + col] = ((s + row) < k && gcol < n)
                ? B[(s + row) * n + gcol] : (int8_t)0;
        }
        __syncwarp();

        // r/16 WMMA sub-steps, each 16×16×16 — reads from shared memory
        const int sub_steps = r / 16;
        for (int sub = 0; sub < sub_steps; sub++) {
            fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
            fragment<matrix_b, 16, 16, 16, int8_t, row_major> b_frag;
            // As[0:16, sub*16:sub*16+16], stride = r
            load_matrix_sync(a_frag, As + sub * 16, r);
            // Bs[sub*16:sub*16+16, 0:16], stride = 16
            load_matrix_sync(b_frag, Bs + sub * 16 * 16, 16);
            mma_sync(acc, a_frag, b_frag, acc);
        }

        // XOR reduction across the 16×16 accumulator
        uint32_t local_xor = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) local_xor ^= (uint32_t)acc.x[i];
        local_xor ^= __shfl_xor_sync(0xffffffff, local_xor, 16);
        local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  8);
        local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  4);
        local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  2);
        local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  1);

        // Only lane 0 writes M — no syncwarp needed (M is private to lane 0)
        if (lane == 0)
            M[ell & 15] = rol32(M[ell & 15], 13) ^ local_xor;
    }

    // Store C
    if (a_row_start < m && b_col_start < n)
        store_matrix_sync(C + a_row_start * n + b_col_start, acc, n, mem_row_major);

    // Store M (lane 0)
    if (lane == 0) {
        int num_tiles_n = (n + 15) / 16;
        int base = (tile_i * num_tiles_n + tile_j) * 16;
        #pragma unroll
        for (int q = 0; q < 16; q++) M_out[base + q] = M[q];
    }
}

#else

// Stub so the PTX always exports the symbol (called only when use_wmma=true
// which never happens below sm_72)
extern "C" __global__ void tiled_matmul_wmma(
    const int8_t*, const int8_t*, int32_t*, uint32_t*,
    int, int, int, int) {}

#endif
