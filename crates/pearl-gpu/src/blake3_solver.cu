/*
 * Pearl PoUW — Standalone BLAKE3 pool challenge solver
 *
 * Compiled as a SEPARATE PTX module from matmul.cu to avoid register
 * pressure interference with the WMMA tensor-core kernel during JIT compilation.
 *
 * Protocol: BLAKE3(seed_32bytes ++ nonce_le_8bytes), unkeyed single-chunk.
 * Leading-zero check: difficulty bits must be zero from hash[0] word onwards.
 */

#include <stdint.h>

__constant__ uint32_t BLAKE3_IV_S[8] = {
    0x6A09E667u, 0xBB67AE85u, 0x3C6EF372u, 0xA54FF53Au,
    0x510E527Fu, 0x9B05688Cu, 0x1F83D9ABu, 0x5BE0CD19u
};

// CHUNK_START|CHUNK_END|ROOT — unkeyed single chunk
#define FLAGS_UNKEYED 11u

__constant__ uint8_t MSG_SCHED[7][16] = {
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15},
    { 2, 6, 3,10, 7, 0, 4,13, 1,11,12, 5, 9,14,15, 8},
    { 3, 4,10,12,13, 2, 7,14, 6, 5, 9, 0,11,15, 8, 1},
    {10, 7,12, 9,14, 3,13,15, 4, 0,11, 2, 5, 8, 1, 6},
    {12,13, 9,11,15,10,14, 8, 7, 2, 5, 3, 0, 1, 6, 4},
    { 9,14,11, 5, 8,12,15, 1,13, 3, 0,10, 2, 6, 4, 7},
    {11,15, 5, 0, 1, 9, 8, 6,14,10, 2,12, 3, 4, 7,13},
};

__device__ __forceinline__ uint32_t ror32s(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

__device__ __forceinline__ void g_mix(
    uint32_t* v, int a, int b, int c, int d, uint32_t x, uint32_t y
) {
    v[a] += v[b] + x;  v[d] = ror32s(v[d] ^ v[a], 16);
    v[c] += v[d];      v[b] = ror32s(v[b] ^ v[c], 12);
    v[a] += v[b] + y;  v[d] = ror32s(v[d] ^ v[a],  8);
    v[c] += v[d];      v[b] = ror32s(v[b] ^ v[c],  7);
}

extern "C" __global__ void solve_pool_challenge_real(
    const uint32_t* __restrict__ d_seed,
    uint64_t                     base_nonce,
    uint32_t                     difficulty,
    uint32_t*       __restrict__ d_nonce,
    uint32_t*       __restrict__ d_found
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t nonce = base_nonce + (uint64_t)tid;

    uint32_t m[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) m[i] = d_seed[i];
    m[8]  = (uint32_t)(nonce & 0xFFFFFFFFu);
    m[9]  = (uint32_t)(nonce >> 32);
    #pragma unroll
    for (int i = 10; i < 16; i++) m[i] = 0;

    uint32_t v[16];
    #pragma unroll
    for (int i = 0; i < 8; i++) v[i]     = BLAKE3_IV_S[i];
    #pragma unroll
    for (int i = 0; i < 4; i++) v[8 + i] = BLAKE3_IV_S[i];
    v[12] = 0; v[13] = 0; v[14] = 40; v[15] = FLAGS_UNKEYED;

    #pragma unroll
    for (int r = 0; r < 7; r++) {
        const uint8_t* s = MSG_SCHED[r];
        g_mix(v, 0, 4,  8, 12, m[s[ 0]], m[s[ 1]]);
        g_mix(v, 1, 5,  9, 13, m[s[ 2]], m[s[ 3]]);
        g_mix(v, 2, 6, 10, 14, m[s[ 4]], m[s[ 5]]);
        g_mix(v, 3, 7, 11, 15, m[s[ 6]], m[s[ 7]]);
        g_mix(v, 0, 5, 10, 15, m[s[ 8]], m[s[ 9]]);
        g_mix(v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g_mix(v, 2, 7,  8, 13, m[s[12]], m[s[13]]);
        g_mix(v, 3, 4,  9, 14, m[s[14]], m[s[15]]);
    }

    uint32_t hash[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) hash[i] = v[i] ^ v[i + 8];

    uint32_t full_words = difficulty >> 5;
    uint32_t rem_bits   = difficulty & 31u;

    bool meets = true;
    for (uint32_t i = 0; i < full_words && meets; i++) {
        if (hash[i] != 0u) meets = false;
    }
    if (meets && rem_bits > 0u) {
        uint32_t hw = hash[full_words];
        uint32_t w  = ((hw & 0xFF000000u) >> 24)
                    | ((hw & 0x00FF0000u) >>  8)
                    | ((hw & 0x0000FF00u) <<  8)
                    | ((hw & 0x000000FFu) << 24);
        if (w & ~(0xFFFFFFFFu >> rem_bits)) meets = false;
    }

    if (meets) {
        if (atomicCAS(d_found, 0u, 1u) == 0u) {
            d_nonce[0] = (uint32_t)(nonce & 0xFFFFFFFFu);
            d_nonce[1] = (uint32_t)(nonce >> 32);
        }
    }
}
