use pearl_types::{Commitments, MiningParams};

/// Compute commitment hashes for matrices A (row-major) and B (column-major).
///
/// κ  = BLAKE3(σ ‖ μ)          where μ encodes r, k, tm, tn
/// HA = BLAKE3(A_flat,  key=κ)
/// HB = BLAKE3(BT_flat, key=κ)
/// sB = BLAKE3(κ ‖ HB)
/// sA = BLAKE3(sB ‖ HA)
pub fn compute(
    a_row_major: &[i8],
    b_col_major: &[i8],
    params: &MiningParams,
) -> Commitments {
    // μ — deterministic encoding of mining config
    let mu = encode_mu(params);

    // κ = BLAKE3(σ ‖ μ)
    let mut kappa_input = Vec::with_capacity(32 + mu.len());
    kappa_input.extend_from_slice(&params.sigma);
    kappa_input.extend_from_slice(&mu);
    let kappa: [u8; 32] = *blake3::hash(&kappa_input).as_bytes();

    // HA = BLAKE3(A_flat, key=κ)
    let h_a = keyed_hash_i8(a_row_major, &kappa);

    // HB = BLAKE3(BT_flat, key=κ)
    let h_b = keyed_hash_i8(b_col_major, &kappa);

    // sB = BLAKE3(κ ‖ HB)
    let mut sb_input = [0u8; 64];
    sb_input[..32].copy_from_slice(&kappa);
    sb_input[32..].copy_from_slice(&h_b);
    let s_b: [u8; 32] = *blake3::hash(&sb_input).as_bytes();

    // sA = BLAKE3(sB ‖ HA)
    let mut sa_input = [0u8; 64];
    sa_input[..32].copy_from_slice(&s_b);
    sa_input[32..].copy_from_slice(&h_a);
    let s_a: [u8; 32] = *blake3::hash(&sa_input).as_bytes();

    Commitments { kappa, h_a, h_b, s_b, s_a }
}

fn keyed_hash_i8(data: &[i8], key: &[u8; 32]) -> [u8; 32] {
    let bytes: &[u8] = bytemuck_i8_to_u8(data);
    *blake3::keyed_hash(key, bytes).as_bytes()
}

fn bytemuck_i8_to_u8(s: &[i8]) -> &[u8] {
    // SAFETY: i8 and u8 have same size/alignment; reinterpreting is valid.
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
