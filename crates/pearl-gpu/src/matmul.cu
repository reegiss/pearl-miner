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
__device__ __forceinline__ uint32_t ror32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

/* ------------------------------------------------------------------ */
/*  BLAKE3 compress (64-byte message, keyed hash, single block/root)   */
/* ------------------------------------------------------------------ */

__constant__ uint32_t BLAKE3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

// flags: CHUNK_START|CHUNK_END|ROOT|KEYED_HASH = 1|2|8|16 = 27
#define BLAKE3_DOMAIN 27u

__constant__ uint8_t BLAKE3_MSG_SCHED[7][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    { 2, 6, 3,10, 7, 0, 4,13, 1,11,12, 5, 9,14,15, 8},
    { 3, 4,10,12,13, 2, 7,14, 6, 5, 9, 0,11,15, 8, 1},
    {10, 7,12, 9,14, 3,13,15, 4, 0,11, 2, 5, 8, 1, 6},
    {12,13, 9,11,15,10,14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    { 9,14,11, 5, 8,12,15, 1,13, 3, 0,10, 2, 6, 4, 7},
    {11,15, 5, 0, 1, 9, 8, 6,14,10, 2,12, 3, 4, 7,13},
};

__device__ __forceinline__ void blake3_g(
    uint32_t* v, int a, int b, int c, int d, uint32_t x, uint32_t y
) {
    v[a] += v[b] + x;  v[d] = ror32(v[d] ^ v[a], 16);
    v[c] += v[d];      v[b] = ror32(v[b] ^ v[c], 12);
    v[a] += v[b] + y;  v[d] = ror32(v[d] ^ v[a],  8);
    v[c] += v[d];      v[b] = ror32(v[b] ^ v[c],  7);
}

// key[8]: sA as 8 LE-u32 words  msg[16]: M-state u32[16] (global ptr)  out[8]: hash u32[8]
__device__ void blake3_hash64(const uint32_t* key, const uint32_t* msg, uint32_t* out) {
    // Pre-load message into registers: avoids strided global-memory accesses
    // inside the 7-round loop (each blake3_g call would otherwise re-fetch from global).
    uint32_t m[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) m[i] = msg[i];

    uint32_t v[16];
    v[0]=key[0]; v[1]=key[1]; v[2]=key[2]; v[3]=key[3];
    v[4]=key[4]; v[5]=key[5]; v[6]=key[6]; v[7]=key[7];
    v[8]=BLAKE3_IV[0]; v[9]=BLAKE3_IV[1]; v[10]=BLAKE3_IV[2]; v[11]=BLAKE3_IV[3];
    v[12]=0; v[13]=0; v[14]=64; v[15]=BLAKE3_DOMAIN;

    #pragma unroll
    for (int r = 0; r < 7; r++) {
        const uint8_t* s = BLAKE3_MSG_SCHED[r];
        blake3_g(v, 0, 4,  8, 12, m[s[ 0]], m[s[ 1]]);
        blake3_g(v, 1, 5,  9, 13, m[s[ 2]], m[s[ 3]]);
        blake3_g(v, 2, 6, 10, 14, m[s[ 4]], m[s[ 5]]);
        blake3_g(v, 3, 7, 11, 15, m[s[ 6]], m[s[ 7]]);
        blake3_g(v, 0, 5, 10, 15, m[s[ 8]], m[s[ 9]]);
        blake3_g(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        blake3_g(v, 2, 7,  8, 13, m[s[12]], m[s[13]]);
        blake3_g(v, 3, 4,  9, 14, m[s[14]], m[s[15]]);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) out[i] = v[i] ^ v[i + 8];
}

// Both hash and threshold stored as 8 LE-u32 words. Compare as uint256 big-endian.
__device__ __forceinline__ bool hash_le_threshold(const uint32_t* h, const uint32_t* t) {
    for (int i = 7; i >= 0; i--) {
        if (h[i] < t[i]) return true;
        if (h[i] > t[i]) return false;
    }
    return true;
}

/* ------------------------------------------------------------------ */
/*  Kernel D: BLAKE3 difficulty check on GPU                           */
/*  Reads M_out, checks threshold, writes winning tiles to d_found.   */
/*                                                                     */
/*  d_found layout per entry (FOUND_STRIDE = 26 u32):                  */
/*    [0]    tile_i                                                     */
/*    [1]    tile_j                                                     */
/*    [2..17] m_state[16]                                              */
/*    [18..25] hash_words[8]                                           */
/* ------------------------------------------------------------------ */

#define FOUND_STRIDE 26
#define MAX_FOUND    256

extern "C" __global__ void blake3_check(
    const uint32_t* __restrict__ M_out,
    const uint32_t* __restrict__ sa_key,    // 8 u32
    const uint32_t* __restrict__ threshold, // 8 u32
    uint32_t*       __restrict__ d_found,
    uint32_t*       __restrict__ d_found_count,
    int num_tiles,
    int num_tiles_n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tiles) return;

    const uint32_t* m = M_out + idx * 16;
    uint32_t hash_out[8];
    blake3_hash64(sa_key, m, hash_out);

    if (!hash_le_threshold(hash_out, threshold)) return;

    int pos = (int)atomicAdd(d_found_count, 1u);
    if (pos >= MAX_FOUND) return;

    int base = pos * FOUND_STRIDE;
    d_found[base + 0] = (uint32_t)(idx / num_tiles_n);   // tile_i
    d_found[base + 1] = (uint32_t)(idx % num_tiles_n);   // tile_j
    #pragma unroll
    for (int q = 0; q < 16; q++) d_found[base + 2  + q] = m[q];
    #pragma unroll
    for (int q = 0; q < 8;  q++) d_found[base + 18 + q] = hash_out[q];
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
/*  Kernel B: WMMA INT8 tensor cores                                   */
/*    sm_80+ (Ampere/Ada): m16n8k32 — native Ampere tensor core        */
/*    sm_72–79 (Turing):   m16n16k16 — Turing tensor core              */
/* ------------------------------------------------------------------ */

#if __CUDA_ARCH__ >= 800

#include <mma.h>
using namespace nvcuda::wmma;

extern "C" __global__ void tiled_matmul_wmma(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t*      __restrict__ C,    // unused — ABI compat
    uint32_t*     __restrict__ M_out,
    int m, int n, int k, int r
) {
    /*
     * blockDim = (32, 8) — 8 warps, each owns one 16×16 output tile.
     * gridDim  = (n/16, (m+127)/128)
     * smem     = 9 × r × 16 bytes
     *
     * Two m16n8k32 accumulators per warp cover the full 16×16 output tile:
     *   acc_l → cols [0..7], acc_r → cols [8..15]
     * For r=32: sub_steps = r/32 = 1 (one pass, no inner loop).
     */
    extern __shared__ int32_t smem_wmma[];

    const int warp_id    = threadIdx.y;
    const int lane       = threadIdx.x;
    const int tid        = warp_id * 32 + lane;

    const int tile_i     = blockIdx.y * 8 + warp_id;
    const int tile_j     = blockIdx.x;
    const bool active    = (tile_i * 16 < m && tile_j * 16 < n);
    const int  a_row_base = tile_i * 16;
    const int  b_col_base = tile_j * 16;

    int8_t* Bs  = (int8_t*)smem_wmma;
    int8_t* wAs = Bs + r * 16 + warp_id * 16 * r;

    fragment<accumulator, 16, 8, 32, int32_t> acc_l, acc_r;
    fill_fragment(acc_l, 0);
    fill_fragment(acc_r, 0);

    uint32_t M[16];
    #pragma unroll
    for (int q = 0; q < 16; q++) M[q] = 0;

    const int num_steps = k / r;
    const int sub_steps = r / 32;   // = 1 for default r=32

    for (int ell = 0; ell < num_steps; ell++) {
        const int s = ell * r;

        for (int idx = tid; idx < r * 16; idx += 256) {
            int row = idx / 16, col = idx % 16;
            int gcol = b_col_base + col;
            Bs[row * 16 + col] = ((s + row) < k && gcol < n)
                ? B[(s + row) * n + gcol] : (int8_t)0;
        }
        __syncthreads();

        if (active) {
            for (int idx = lane; idx < 16 * r; idx += 32) {
                int row = idx / r, col = idx % r;
                int grow = a_row_base + row;
                wAs[idx] = (grow < m && (s + col) < k)
                    ? A[grow * k + s + col] : (int8_t)0;
            }
            __syncwarp();

            for (int sub = 0; sub < sub_steps; sub++) {
                fragment<matrix_a, 16, 8, 32, int8_t, row_major> a_frag;
                fragment<matrix_b, 16, 8, 32, int8_t, row_major> b_frag_l, b_frag_r;

                load_matrix_sync(a_frag,   wAs + sub * 32,           r);
                load_matrix_sync(b_frag_l, Bs  + sub * 32 * 16,     16);
                load_matrix_sync(b_frag_r, Bs  + sub * 32 * 16 + 8, 16);

                mma_sync(acc_l, a_frag, b_frag_l, acc_l);
                mma_sync(acc_r, a_frag, b_frag_r, acc_r);
            }

            uint32_t local_xor = 0;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                local_xor ^= (uint32_t)acc_l.x[i];
                local_xor ^= (uint32_t)acc_r.x[i];
            }
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor, 16);
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  8);
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  4);
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  2);
            local_xor ^= __shfl_xor_sync(0xffffffff, local_xor,  1);
            if (lane == 0)
                M[ell & 15] = rol32(M[ell & 15], 13) ^ local_xor;
        }

        __syncthreads();
    }

    if (!active) return;

    if (lane == 0) {
        int num_tiles_n = (n + 15) / 16;
        int base = (tile_i * num_tiles_n + tile_j) * 16;
        #pragma unroll
        for (int q = 0; q < 16; q++) M_out[base + q] = M[q];
    }
}

#elif __CUDA_ARCH__ >= 720

#include <mma.h>
using namespace nvcuda::wmma;

extern "C" __global__ void tiled_matmul_wmma(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    int32_t*      __restrict__ C,
    uint32_t*     __restrict__ M_out,
    int m, int n, int k, int r
) {
    extern __shared__ int32_t smem_wmma[];

    const int warp_id    = threadIdx.y;
    const int lane       = threadIdx.x;
    const int tid        = warp_id * 32 + lane;

    const int tile_i     = blockIdx.y * 8 + warp_id;
    const int tile_j     = blockIdx.x;
    const bool active    = (tile_i * 16 < m && tile_j * 16 < n);
    const int  a_row_base = tile_i * 16;
    const int  b_col_base = tile_j * 16;

    int8_t* Bs  = (int8_t*)smem_wmma;
    int8_t* wAs = Bs + r * 16 + warp_id * 16 * r;

    fragment<accumulator, 16, 16, 16, int32_t> acc;
    fill_fragment(acc, 0);

    uint32_t M[16];
    #pragma unroll
    for (int q = 0; q < 16; q++) M[q] = 0;

    const int num_steps = k / r;

    for (int ell = 0; ell < num_steps; ell++) {
        const int s = ell * r;

        for (int idx = tid; idx < r * 16; idx += 256) {
            int row = idx / 16, col = idx % 16;
            int gcol = b_col_base + col;
            Bs[row * 16 + col] = ((s + row) < k && gcol < n)
                ? B[(s + row) * n + gcol] : (int8_t)0;
        }
        __syncthreads();

        if (active) {
            for (int idx = lane; idx < 16 * r; idx += 32) {
                int row = idx / r, col = idx % r;
                int grow = a_row_base + row;
                wAs[idx] = (grow < m && (s + col) < k)
                    ? A[grow * k + s + col] : (int8_t)0;
            }
            __syncwarp();

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

        __syncthreads();
    }

    if (!active) return;

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
