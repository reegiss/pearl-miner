/// Mining parameters derived from a pool challenge.
#[derive(Clone, Debug)]
pub struct MiningParams {
    /// Blockchain state σ — 32-byte seed from pool challenge.
    pub sigma: [u8; 32],
    /// Difficulty target b (bits).
    pub difficulty: u32,
    /// Noise rank r ∈ {32,64,128,256,512,1024}.
    pub r: usize,
    /// Common dimension k; 16r ≤ k ≤ 4r², 64|k.
    pub k: usize,
    /// Output tile height tm.
    pub tm: usize,
    /// Output tile width tn.
    pub tn: usize,
    /// Matrix A rows m.
    pub m: usize,
    /// Matrix B columns n.
    pub n: usize,
}

/// A tile that satisfied the block-opening condition.
#[derive(Clone, Debug)]
pub struct FoundBlock {
    pub tile_i: usize,
    pub tile_j: usize,
    /// M[16] hash state that passed the BLAKE3 difficulty check.
    pub m_state: [u32; 16],
    /// BLAKE3(M, key=sA) that was ≤ threshold.
    pub hash: [u8; 32],
}

/// Commitment hashes derived from A, B, μ, σ.
#[derive(Clone, Debug)]
pub struct Commitments {
    pub kappa: [u8; 32],
    pub h_a: [u8; 32],
    pub h_b: [u8; 32],
    pub s_b: [u8; 32],
    pub s_a: [u8; 32],
}
