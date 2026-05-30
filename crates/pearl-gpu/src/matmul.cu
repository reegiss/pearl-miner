/*
 * Pearl PoUW — Tiled INT8 MatMul with warp-level M-state accumulation
 *
 * One thread block per output tile (i,j), TM×TN threads.
 * Depth loop: for each of k/r depth steps, accumulate INT32, then
 * update M[16] using a two-phase XOR reduction:
 *   1. __reduce_xor_sync within each warp  (hardware, 5 cycles)
 *   2. thread 0 combines all warp results  (WARPS_PER_BLOCK iterations)
 *
 * Dynamic shared memory layout:
 *   [0            .. TM*r)       int8  As[TM][r]
 *   [TM*r         .. (TM+TN)*r)  int8  Bs[r][TN]
 *   [(TM+TN)*r    .. +4*WARPS)   u32   warp_xors[WARPS_PER_BLOCK]
 *   [above+4*WARPS.. +64)        u32   M[16]
 */

#include <stdint.h>

#define TM 16
#define TN 16
#define WARPS_PER_BLOCK ((TM * TN) / 32)

__device__ __forceinline__ uint32_t rol32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

extern "C" __global__ void tiled_matmul(
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
    uint32_t* M         = warp_xors + WARPS_PER_BLOCK;

    const int tile_i = blockIdx.y;
    const int tile_j = blockIdx.x;
    const int ty     = threadIdx.y;
    const int tx     = threadIdx.x;
    const int tid    = ty * TN + tx;
    const int lane   = tid & 31;
    const int warp   = tid >> 5;

    const int gi = tile_i * TM + ty;
    const int gj = tile_j * TN + tx;

    int32_t acc = 0;

    // Init M state
    if (tid < 16) M[tid] = 0;
    __syncthreads();

    const int num_steps = k / r;

    for (int ell = 0; ell < num_steps; ell++) {
        const int s = ell * r;

        // Load A strip: A[gi, s:s+r] into As[ty, 0:r]
        {
            int total = TM * r;
            for (int idx = tid; idx < total; idx += TM * TN) {
                int row = idx / r, col = idx % r;
                int grow = tile_i * TM + row;
                As[row * r + col] = (grow < m && (s + col) < k)
                    ? A[grow * k + s + col] : 0;
            }
        }

        // Load B strip: B[s:s+r, gj] into Bs[0:r, tx]
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

        // Accumulate via DP4A — process r elements in groups of 4
        const int r4 = r / 4;
        for (int q = 0; q < r4; q++) {
            int a_pack = 0, b_pack = 0;
            a_pack |= ((int)As[ty * r + q*4+0] & 0xFF) <<  0;
            a_pack |= ((int)As[ty * r + q*4+1] & 0xFF) <<  8;
            a_pack |= ((int)As[ty * r + q*4+2] & 0xFF) << 16;
            a_pack |= ((int)As[ty * r + q*4+3] & 0xFF) << 24;
            b_pack |= ((int)Bs[(q*4+0) * TN + tx] & 0xFF) <<  0;
            b_pack |= ((int)Bs[(q*4+1) * TN + tx] & 0xFF) <<  8;
            b_pack |= ((int)Bs[(q*4+2) * TN + tx] & 0xFF) << 16;
            b_pack |= ((int)Bs[(q*4+3) * TN + tx] & 0xFF) << 24;
            acc = __dp4a(a_pack, b_pack, acc);
        }
        // Tail if r not multiple of 4 (shouldn't happen with valid r)
        for (int q = r4 * 4; q < r; q++) {
            acc += (int32_t)As[ty * r + q] * (int32_t)Bs[q * TN + tx];
        }

        // --- Warp XOR reduction via butterfly shuffle (sm_75+, ~5 cycles) ---
        uint32_t wx = (uint32_t)acc;
        wx ^= __shfl_xor_sync(0xffffffff, wx, 16);
        wx ^= __shfl_xor_sync(0xffffffff, wx,  8);
        wx ^= __shfl_xor_sync(0xffffffff, wx,  4);
        wx ^= __shfl_xor_sync(0xffffffff, wx,  2);
        wx ^= __shfl_xor_sync(0xffffffff, wx,  1);
        if (lane == 0) warp_xors[warp] = wx;
        __syncthreads();

        // Thread 0 combines WARPS_PER_BLOCK warp results and updates M
        if (tid == 0) {
            uint32_t X = 0;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; w++) X ^= warp_xors[w];
            int slot = ell & 15;
            M[slot] = rol32(M[slot], 13) ^ X;
        }
        __syncthreads();
    }

    // Write C output
    if (gi < m && gj < n) C[gi * n + gj] = acc;

    // Write M state
    if (tid == 0) {
        int num_tiles_n = (n + TN - 1) / TN;
        int base = (tile_i * num_tiles_n + tile_j) * 16;
        #pragma unroll
        for (int q = 0; q < 16; q++) M_out[base + q] = M[q];
    }
}
