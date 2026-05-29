#include <stdint.h>
#include <cuda_runtime.h>

// ============================================================
// BLAKE3 — single 64-byte block, keyed hash
// Spec: https://github.com/BLAKE3-team/BLAKE3-specs/raw/master/blake3.pdf
// ============================================================

static __constant__ uint32_t BLAKE3_IV[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

// flags for single-block keyed hash (CHUNK_START | CHUNK_END | ROOT | KEYED_HASH)
#define B3_FLAGS 27u

static __constant__ uint8_t MSG_PERM[16] = {
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8
};

__device__ __forceinline__ uint32_t rotr32(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ void b3g(uint32_t* s, int a, int b, int c, int d,
                                    uint32_t x, uint32_t y) {
    s[a] += s[b] + x; s[d] = rotr32(s[d] ^ s[a], 16);
    s[c] += s[d];     s[b] = rotr32(s[b] ^ s[c], 12);
    s[a] += s[b] + y; s[d] = rotr32(s[d] ^ s[a],  8);
    s[c] += s[d];     s[b] = rotr32(s[b] ^ s[c],  7);
}

__device__ void b3_round(uint32_t* s, const uint32_t* m) {
    b3g(s, 0, 4,  8, 12, m[0],  m[1]);
    b3g(s, 1, 5,  9, 13, m[2],  m[3]);
    b3g(s, 2, 6, 10, 14, m[4],  m[5]);
    b3g(s, 3, 7, 11, 15, m[6],  m[7]);
    b3g(s, 0, 5, 10, 15, m[8],  m[9]);
    b3g(s, 1, 6, 11, 12, m[10], m[11]);
    b3g(s, 2, 7,  8, 13, m[12], m[13]);
    b3g(s, 3, 4,  9, 14, m[14], m[15]);
}

// BLAKE3 keyed hash: key[32] + msg_i32[16] → out[32]
// Matches Rust: blake3::keyed_hash(&s_a, &m_bytes) where m_bytes = M[16] as LE i32s.
__device__ void blake3_keyed_hash(
    const uint8_t* __restrict__ key,
    const int32_t* __restrict__ msg_i32,
    uint8_t*       __restrict__ out
) {
    // Parse 32-byte key as 8 LE u32
    uint32_t kw[8];
    for (int i = 0; i < 8; i++) {
        kw[i] = ((uint32_t)key[4*i])
              | ((uint32_t)key[4*i+1] << 8)
              | ((uint32_t)key[4*i+2] << 16)
              | ((uint32_t)key[4*i+3] << 24);
    }

    // Message words (CUDA is little-endian, same as Rust to_le_bytes())
    uint32_t blk[16];
    for (int i = 0; i < 16; i++) blk[i] = (uint32_t)msg_i32[i];

    // Initial state: [key_words | IV[0..3] | counter=0 | block_len=64 | flags]
    uint32_t s[16] = {
        kw[0], kw[1], kw[2], kw[3],
        kw[4], kw[5], kw[6], kw[7],
        BLAKE3_IV[0], BLAKE3_IV[1], BLAKE3_IV[2], BLAKE3_IV[3],
        0u, 0u, 64u, B3_FLAGS
    };

    // 7 rounds with permutation between each (no permute after round 7)
    for (int rnd = 0; rnd < 7; rnd++) {
        b3_round(s, blk);
        if (rnd < 6) {
            uint32_t tmp[16];
            for (int i = 0; i < 16; i++) tmp[i] = blk[MSG_PERM[i]];
            for (int i = 0; i < 16; i++) blk[i] = tmp[i];
        }
    }

    // Output: first 8 words = s[0..7] ^ s[8..15]
    for (int i = 0; i < 8; i++) {
        uint32_t w = s[i] ^ s[i + 8];
        out[4*i]   = (uint8_t)(w);
        out[4*i+1] = (uint8_t)(w >> 8);
        out[4*i+2] = (uint8_t)(w >> 16);
        out[4*i+3] = (uint8_t)(w >> 24);
    }
}

// Compare two 32-byte LE uint256: a <= b?
__device__ __forceinline__ bool le256_lte(const uint8_t* a, const uint8_t* b) {
    for (int i = 31; i >= 0; i--) {
        if (a[i] < b[i]) return true;
        if (a[i] > b[i]) return false;
    }
    return true;
}

// ============================================================
// Found-tile output structure (must match Rust #[repr(C)])
// ============================================================

typedef struct __align__(4) {
    uint32_t tile_i;
    uint32_t tile_j;
    int32_t  m_state[16];
    uint8_t  final_hash[32];
} FoundTileGpu;  // 4 + 4 + 64 + 32 = 104 bytes

// ============================================================
// Mining kernel
//
// Grid:  (ceil(m/tm), ceil(n/tn), 1)
// Block: (tm, tn, 1)    — requires tm*tn <= 1024
//
// Dynamic shared memory: (tm*tn + 16) * sizeof(int32_t) bytes
//   shmem[0 .. tm*tn]       : Cblk (current tile accumulator, for XOR)
//   shmem[tm*tn .. tm*tn+16]: M[16] state
//
// A[m×k] row-major:    A[i][l]  = A[i*k + l]
// B[k×n] col-major:    B[l][j]  = B[j*k + l]   (B^T stored row-major)
// C[m×n] row-major:    C[i][j]  = C[i*n + j]
// ============================================================

extern "C" __global__ void tiled_matmul_mine(
    const int8_t*  __restrict__ A,
    const int8_t*  __restrict__ B,
    int32_t*                    C_out,
    FoundTileGpu*               found_buf,
    int32_t*                    found_count,   // atomic counter
    int32_t                     max_found,
    const uint8_t* __restrict__ s_a,           // 32-byte key
    const uint8_t* __restrict__ threshold,     // 32-byte LE uint256
    uint32_t m, uint32_t n, uint32_t k, uint32_t r,
    uint32_t tm, uint32_t tn
) {
    const uint32_t ti  = blockIdx.x;
    const uint32_t tj  = blockIdx.y;
    const uint32_t row = threadIdx.x;
    const uint32_t col = threadIdx.y;
    const uint32_t tid = row * tn + col;
    const uint32_t tile_sz = tm * tn;

    const uint32_t gi = ti * tm + row;
    const uint32_t gj = tj * tn + col;

    if (gi >= m || gj >= n) return;

    extern __shared__ int32_t shmem[];
    int32_t* Cblk = shmem;
    int32_t* M    = shmem + tile_sz;

    Cblk[tid] = 0;
    if (tid < 16) M[tid] = 0;
    __syncthreads();

    int32_t acc = 0;
    const uint32_t num_steps = k / r;

    for (uint32_t step = 0; step < num_steps; step++) {
        const uint32_t base = step * r;

        // Accumulate r depth steps for this output element
        int32_t partial = 0;
        for (uint32_t d = 0; d < r; d++) {
            partial += (int32_t)A[gi * k + base + d]
                     * (int32_t)B[gj * k + base + d];  // col-major B
        }
        acc += partial;

        // Write to shared memory for XOR reduction
        Cblk[tid] = acc;
        __syncthreads();

        // Thread 0: XOR all Cblk elements, update M[step % 16]
        if (tid == 0) {
            int32_t X = 0;
            for (uint32_t t = 0; t < tile_sz; t++) X ^= Cblk[t];
            const uint32_t idx = step % 16;
            uint32_t mv = (uint32_t)M[idx];
            mv = (mv << 13) | (mv >> 19);  // rotate_left(mv, 13)
            M[idx] = (int32_t)(mv ^ (uint32_t)X);
        }
        __syncthreads();
    }

    // Write noisy product
    C_out[gi * n + gj] = acc;

    // Difficulty check — thread 0 only
    if (tid == 0) {
        uint8_t hash_out[32];
        blake3_keyed_hash(s_a, M, hash_out);

        if (le256_lte(hash_out, threshold)) {
            int32_t idx = atomicAdd(found_count, 1);
            if (idx < max_found) {
                found_buf[idx].tile_i = ti;
                found_buf[idx].tile_j = tj;
                for (int i = 0; i < 16; i++) found_buf[idx].m_state[i] = M[i];
                for (int i = 0; i < 32; i++) found_buf[idx].final_hash[i] = hash_out[i];
            }
        }
    }
}
