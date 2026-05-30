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
    const uint64_t row_mul = 0x9e3779b97f4a7c15ULL;
    const uint64_t col_mul = 0xd1b54a32d192ed03ULL;
    return splitmix64(base ^ (row * row_mul) ^ (col * col_mul));
}

__device__ __forceinline__ int8_t compute_noisy_a(uint64_t job_seed, uint64_t sa_seed, int row, int col, int r) {
    const uint64_t col_mul = 0xd1b54a32d192ed03ULL;
    const uint64_t row_mul = 0x9e3779b97f4a7c15ULL;
    const uint64_t global_col_part = (uint64_t)col * col_mul;
    const uint64_t row_part = (uint64_t)row * row_mul;

    int8_t a_val = (int8_t)((splitmix64(job_seed ^ row_part ^ global_col_part) & 0x7F) - 64);
    
    uint64_t er_s = splitmix64((sa_seed ^ 0x100ULL) ^ global_col_part);
    int pos_col = (int)(er_s & (r - 1));
    er_s = splitmix64(er_s);
    int neg_col = (int)(er_s & (r - 1));
    if (neg_col == pos_col) neg_col = (neg_col + 1) & (r - 1);

    const uint64_t base_el = (sa_seed ^ 0x200ULL);
    int8_t el_pos = (int8_t)((splitmix64(base_el ^ row_part ^ ((uint64_t)pos_col * col_mul)) & 0x3F) - 32);
    int8_t el_neg = (int8_t)((splitmix64(base_el ^ row_part ^ ((uint64_t)neg_col * col_mul)) & 0x3F) - 32);
    
    int val = (int)a_val + (int)el_pos - (int)el_neg;
    if (val > 127) val = 127;
    if (val < -127) val = -127;
    return (int8_t)val;
}

/* ------------------------------------------------------------------ */
/*  Kernel 0: generate A' = A + EL·ER into global memory              */
/*  blockDim=(32,16) — one thread per element of the m×k A' matrix.   */
/* ------------------------------------------------------------------ */

extern "C" __global__ void generate_a_prime(
    int8_t*  __restrict__ A_prime,
    uint64_t job_seed, uint64_t sa_seed,
    int m, int k, int r
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= k) return;
    A_prime[row * k + col] = compute_noisy_a(job_seed, sa_seed, row, col, r);
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
                int gcol = s + col;
                As[idx] = (grow < m && gcol < k)
                    ? __ldg(&A[grow * k + gcol]) : (int8_t)0;
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
/*  cp.async helpers — sm_80+ (Ampere) only                           */
/*  cp.async.ca writes directly to shared memory, bypassing L1.       */
/* ------------------------------------------------------------------ */

#if __CUDA_ARCH__ >= 800

__device__ __forceinline__
void cp_async16(void* dst, const void* src) {
    uint32_t smem_ptr = __cvta_generic_to_shared(dst);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_ptr), "l"((const char*)src) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// Wait until all pending cp.async groups are done.
__device__ __forceinline__ void cp_async_wait0() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
}

#endif  // __CUDA_ARCH__ >= 800

/* ------------------------------------------------------------------ */
/*  Kernel B: WMMA INT8 tensor cores (sm_72+)                         */
/*                                                                     */
/*  Uses m16n16k16 INT8 for all targets.                               */
/*  sm_80+: B-strip is double-buffered via cp.async so B[ell+1] is    */
/*  prefetched while tensor cores consume B[ell] (hides ~500-cycle    */
/*  global-mem latency behind WMMA + A-strip load).                   */
/*                                                                     */
/*  Smem layout:                                                       */
/*    sm_80+: Bs[0](r*16) | Bs[1](r*16) | wAs[8 warps](8*16*r)       */
/*            = 10 * r * 16 bytes                                      */
/*    sm_72+: Bs(r*16)    |              | wAs[8 warps](8*16*r)       */
/*            =  9 * r * 16 bytes                                      */
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
    extern __shared__ int32_t smem_wmma[];

    const int warp_id    = threadIdx.y;
    const int lane       = threadIdx.x;
    const int tid        = warp_id * 32 + lane;

    const int tile_i      = blockIdx.y * 8 + warp_id;
    const int tile_j      = blockIdx.x;
    const bool active     = (tile_i * 16 < m && tile_j * 16 < n);
    const int  a_row_base = tile_i * 16;
    const int  b_col_base = tile_j * 16;

    // Smem pointers — layout differs between sm_80+ and sm_72.
#if __CUDA_ARCH__ >= 800
    int8_t* Bs_buf0 = (int8_t*)smem_wmma;
    int8_t* Bs_buf1 = Bs_buf0 + r * 16;
    int8_t* wAs     = Bs_buf1 + r * 16 + warp_id * 16 * r;
#else
    int8_t* Bs  = (int8_t*)smem_wmma;
    int8_t* wAs = Bs + r * 16 + warp_id * 16 * r;
#endif

    fragment<accumulator, 16, 16, 16, int32_t> acc;
    fill_fragment(acc, 0);

    uint32_t M[16];
    #pragma unroll
    for (int q = 0; q < 16; q++) M[q] = 0;

    const int  num_steps  = k / r;
    const bool b_full_col = (b_col_base + 15 < n);
    const int  valid_rows = active ? min(16, m - a_row_base) : 0;
    const int  r4         = r / 16;

#if __CUDA_ARCH__ >= 800
    /* ---- sm_80+ double-buffer path --------------------------------- */
    int8_t* Bs_cur = Bs_buf0;
    int8_t* Bs_nxt = Bs_buf1;

    // Pre-load B[0] → Bs_cur before the main loop.
    if (tid < r) {
        if (b_full_col) {
            cp_async16(Bs_cur + tid * 16, B + tid * n + b_col_base);
        } else {
            for (int c = 0; c < 16; c++)
                Bs_cur[tid * 16 + c] = (b_col_base + c < n)
                    ? __ldg(&B[tid * n + b_col_base + c]) : (int8_t)0;
        }
    }
    cp_async_commit();
    cp_async_wait0();
    __syncthreads();

    for (int ell = 0; ell < num_steps; ell++) {
        const int s = ell * r;

        // Prefetch B[ell+1] → Bs_nxt while tensor cores consume B[ell].
        // For the last iteration no prefetch is issued; commit creates an
        // empty group that cp_async_wait0() clears immediately.
        if (ell + 1 < num_steps && tid < r) {
            const int s1 = s + r;
            if (b_full_col) {
                cp_async16(Bs_nxt + tid * 16,
                           B + (s1 + tid) * n + b_col_base);
            } else {
                for (int c = 0; c < 16; c++)
                    Bs_nxt[tid * 16 + c] = (b_col_base + c < n)
                        ? __ldg(&B[(s1 + tid) * n + b_col_base + c]) : (int8_t)0;
            }
        }
        cp_async_commit();  // all threads commit (empty for warps 1-7)

        if (active) {
            // A-strip: int4 loads [concurrent with B prefetch above]
            for (int q = lane; q < 16 * r4; q += 32) {
                const int row = q / r4, chunk = q % r4;
                int4* dst = (int4*)wAs + q;
                *dst = (row < valid_rows)
                    ? __ldg((const int4*)(A + (a_row_base + row) * k + s) + chunk)
                    : make_int4(0, 0, 0, 0);
            }
            __syncwarp();

            const int sub_steps = r / 16;
            for (int sub = 0; sub < sub_steps; sub++) {
                fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
                fragment<matrix_b, 16, 16, 16, int8_t, row_major> b_frag;
                load_matrix_sync(a_frag, wAs     + sub * 16,       r);
                load_matrix_sync(b_frag, Bs_cur  + sub * 16 * 16, 16);
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

        // Drain the prefetch, sync, then swap ping-pong buffers.
        cp_async_wait0();
        __syncthreads();
        int8_t* tmp = Bs_cur; Bs_cur = Bs_nxt; Bs_nxt = tmp;
    }

#else
    /* ---- sm_72–79 single-buffer path ------------------------------- */
    for (int ell = 0; ell < num_steps; ell++) {
        const int s = ell * r;

        if (tid < r) {
            if (b_full_col) {
                *((int4*)(Bs + tid * 16)) =
                    __ldg((const int4*)(B + (s + tid) * n + b_col_base));
            } else {
                for (int c = 0; c < 16; c++)
                    Bs[tid * 16 + c] = (b_col_base + c < n)
                        ? __ldg(&B[(s + tid) * n + b_col_base + c]) : (int8_t)0;
            }
        }
        __syncthreads();

        if (active) {
            for (int q = lane; q < 16 * r4; q += 32) {
                const int row = q / r4, chunk = q % r4;
                int4* dst = (int4*)wAs + q;
                *dst = (row < valid_rows)
                    ? __ldg((const int4*)(A + (a_row_base + row) * k + s) + chunk)
                    : make_int4(0, 0, 0, 0);
            }
            __syncwarp();

            const int sub_steps = r / 16;
            for (int sub = 0; sub < sub_steps; sub++) {
                fragment<matrix_a, 16, 16, 16, int8_t, row_major> a_frag;
                fragment<matrix_b, 16, 16, 16, int8_t, row_major> b_frag;
                load_matrix_sync(a_frag, wAs + sub * 16,      r);
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
#endif  // __CUDA_ARCH__ >= 800

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

#endif  // __CUDA_ARCH__ >= 720
