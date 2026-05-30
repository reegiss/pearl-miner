/*
 * Pearl PoUW — Tiled INT8 MatMul with M-state accumulation
 *
 * Each thread block computes one output tile (i, j) of size TM×TN.
 * After every full depth step ℓ the XOR of all INT32 accumulators
 * is mixed into the 512-bit state M[16].
 *
 * Outputs:
 *   c_out  — full C' matrix in INT32, row-major
 *   m_out  — M[16] per tile, shape (num_tiles_m, num_tiles_n, 16)
 */

#include <stdint.h>

#define TM 16
#define TN 16

__device__ __forceinline__ uint32_t rol32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

extern "C" __global__ void tiled_matmul(
    const int8_t* __restrict__ A,   // m × k, row-major
    const int8_t* __restrict__ B,   // k × n, row-major
    int32_t*      __restrict__ C,   // m × n, row-major  (output)
    uint32_t*     __restrict__ M_out, // (m/TM, n/TN, 16) (output)
    int m, int n, int k, int r
) {
    // Tile coordinates
    const int tile_i = blockIdx.y;  // row tile index
    const int tile_j = blockIdx.x;  // col tile index
    const int ty     = threadIdx.y; // row within tile [0, TM)
    const int tx     = threadIdx.x; // col within tile [0, TN)

    // Global matrix offsets for this tile
    const int gi = tile_i * TM + ty;
    const int gj = tile_j * TN + tx;

    // Accumulator for this output element
    int32_t acc = 0;

    // Shared memory for A strip (TM × r) and B strip (r × TN)
    __shared__ int8_t As[TM][32]; // max r=32 per depth step slice
    __shared__ int8_t Bs[32][TN];

    // M state — 16 × int32, maintained by thread 0
    __shared__ uint32_t M[16];
    if (ty == 0 && tx == 0) {
        #pragma unroll
        for (int q = 0; q < 16; q++) M[q] = 0;
    }

    // Shared scratch for XOR reduction (TM × TN elements)
    __shared__ int32_t xor_scratch[TM * TN];

    __syncthreads();

    int num_steps = k / r; // full depth steps only (partial ignored for M)

    for (int ell = 0; ell < num_steps; ell++) {
        int s = ell * r; // start column in k dimension

        // --- Load A strip: A[gi, s:s+r] ---
        // Each thread loads one element of the strip it needs.
        // We need TM rows × r cols. Each thread (ty, tx) loads A[gi, s + tx*(r/TN) + ...]
        // Simple approach: stride over r using all TM×TN threads
        {
            int total = TM * r;
            int tid   = ty * TN + tx;
            for (int idx = tid; idx < total; idx += TM * TN) {
                int row = idx / r;
                int col = idx % r;
                int gi_row = tile_i * TM + row;
                As[row][col] = (gi_row < m && (s + col) < k) ? A[gi_row * k + s + col] : 0;
            }
        }

        // --- Load B strip: B[s:s+r, gj] ---
        {
            int total = r * TN;
            int tid   = ty * TN + tx;
            for (int idx = tid; idx < total; idx += TM * TN) {
                int row = idx / TN;
                int col = idx % TN;
                int gj_col = tile_j * TN + col;
                Bs[row][col] = ((s + row) < k && gj_col < n) ? B[(s + row) * n + gj_col] : 0;
            }
        }

        __syncthreads();

        // --- Accumulate using DP4A (4 INT8s per instruction) ---
        // Process r elements in groups of 4
        int r4 = r / 4;
        for (int q = 0; q < r4; q++) {
            // Pack 4 consecutive A values into int32
            int a_packed = 0;
            a_packed |= ((int)(As[ty][q*4+0]) & 0xFF) <<  0;
            a_packed |= ((int)(As[ty][q*4+1]) & 0xFF) <<  8;
            a_packed |= ((int)(As[ty][q*4+2]) & 0xFF) << 16;
            a_packed |= ((int)(As[ty][q*4+3]) & 0xFF) << 24;

            int b_packed = 0;
            b_packed |= ((int)(Bs[q*4+0][tx]) & 0xFF) <<  0;
            b_packed |= ((int)(Bs[q*4+1][tx]) & 0xFF) <<  8;
            b_packed |= ((int)(Bs[q*4+2][tx]) & 0xFF) << 16;
            b_packed |= ((int)(Bs[q*4+3][tx]) & 0xFF) << 24;

            acc = __dp4a(a_packed, b_packed, acc);
        }

        // Handle remainder if r % 4 != 0
        for (int q = r4 * 4; q < r; q++) {
            acc += (int32_t)As[ty][q] * (int32_t)Bs[q][tx];
        }

        // --- XOR reduction across tile ---
        // Write this thread's accumulator to scratch
        xor_scratch[ty * TN + tx] = acc;
        __syncthreads();

        // Parallel XOR reduction — thread 0 computes final XOR
        if (ty == 0 && tx == 0) {
            uint32_t X = 0;
            for (int idx = 0; idx < TM * TN; idx++) {
                X ^= (uint32_t)xor_scratch[idx];
            }
            // M[ℓ mod 16] = rotate_left(M[ℓ mod 16], 13) ^ X
            int slot = ell & 15;
            M[slot] = rol32(M[slot], 13) ^ X;
        }

        __syncthreads();
    }

    // --- Write C' output ---
    if (gi < m && gj < n) {
        C[gi * n + gj] = acc;
    }

    // --- Write M state output ---
    if (ty == 0 && tx == 0) {
        int num_tiles_n = (n + TN - 1) / TN;
        int m_base = (tile_i * num_tiles_n + tile_j) * 16;
        #pragma unroll
        for (int q = 0; q < 16; q++) {
            M_out[m_base + q] = M[q];
        }
    }
}
