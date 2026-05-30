use pearl_types::{Commitments, MiningParams};

/// Computed once per challenge (B and σ fixed).
/// sB does not depend on A, so this can be reused across many A matrices.
pub struct ChallengeCommitment {
    pub kappa: [u8; 32],
    pub h_b:   [u8; 32],
    pub s_b:   [u8; 32],
}

/// Compute the challenge-level commitment from B (column-major) and params.
/// Call once per challenge; reuse for every job in that challenge.
pub fn compute_challenge(b_col_major: &[i8], params: &MiningParams) -> ChallengeCommitment {
    let mu    = encode_mu(params);
    let kappa = blake3_hash(&[&params.sigma, mu.as_slice()]);
    let h_b   = keyed_hash_i8(b_col_major, &kappa);
    let s_b   = blake3_hash(&[&kappa, &h_b]);
    ChallengeCommitment { kappa, h_b, s_b }
}

/// Compute sA from A (row-major) given the pre-computed challenge commitment.
/// Call once per job.
pub fn compute_sa(a_row_major: &[i8], cc: &ChallengeCommitment) -> ([u8; 32], Commitments) {
    let h_a = keyed_hash_i8(a_row_major, &cc.kappa);
    let s_a = blake3_hash(&[&cc.s_b, &h_a]);
    let commitments = Commitments {
        kappa: cc.kappa,
        h_a,
        h_b:   cc.h_b,
        s_b:   cc.s_b,
        s_a,
    };
    (s_a, commitments)
}

/// Full computation (kept for compatibility).
pub fn compute(
    a_row_major: &[i8],
    b_col_major: &[i8],
    params: &MiningParams,
) -> Commitments {
    let cc = compute_challenge(b_col_major, params);
    compute_sa(a_row_major, &cc).1
}

fn keyed_hash_i8(data: &[i8], key: &[u8; 32]) -> [u8; 32] {
    let bytes = i8_as_u8(data);
    *blake3::keyed_hash(key, bytes).as_bytes()
}

fn blake3_hash(parts: &[&[u8]]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    for p in parts { h.update(p); }
    *h.finalize().as_bytes()
}

fn i8_as_u8(s: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len()) }
}

fn encode_mu(p: &MiningParams) -> [u8; 32] {
    let mut mu = [0u8; 32];
    mu[0..8].copy_from_slice(&(p.r as u64).to_le_bytes());
    mu[8..16].copy_from_slice(&(p.k as u64).to_le_bytes());
    mu[16..20].copy_from_slice(&(p.tm as u32).to_le_bytes());
    mu[20..24].copy_from_slice(&(p.tn as u32).to_le_bytes());
    mu[24..28].copy_from_slice(&(p.m as u32).to_le_bytes());
    mu[28..32].copy_from_slice(&(p.n as u32).to_le_bytes());
    mu
}
