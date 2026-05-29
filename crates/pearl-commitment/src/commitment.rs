use pearl_types::{Commitments, MiningConfig};

pub fn compute(a: &[i8], b: &[i8], config: &MiningConfig) -> Commitments {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&config.sigma);
    hasher.update(&config.mu);
    let kappa: [u8; 32] = *hasher.finalize().as_bytes();

    let a_bytes: Vec<u8> = a.iter().map(|&x| x as u8).collect();
    let ha: [u8; 32] = *blake3::keyed_hash(&kappa, &a_bytes).as_bytes();

    let b_bytes: Vec<u8> = b.iter().map(|&x| x as u8).collect();
    let hb: [u8; 32] = *blake3::keyed_hash(&kappa, &b_bytes).as_bytes();

    let mut sb_input = [0u8; 64];
    sb_input[..32].copy_from_slice(&kappa);
    sb_input[32..].copy_from_slice(&hb);
    let s_b: [u8; 32] = *blake3::hash(&sb_input).as_bytes();

    let mut sa_input = [0u8; 64];
    sa_input[..32].copy_from_slice(&s_b);
    sa_input[32..].copy_from_slice(&ha);
    let s_a: [u8; 32] = *blake3::hash(&sa_input).as_bytes();

    Commitments { kappa, ha, hb, s_a, s_b }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pearl_types::{MatrixParams, MiningConfig};

    fn cfg() -> MiningConfig {
        MiningConfig {
            params: MatrixParams { m: 4, n: 4, k: 64, r: 32, tm: 2, tn: 2 },
            difficulty_bits: 10.0,
            sigma: b"chain-state".to_vec(),
            mu: b"miner-cfg".to_vec(),
        }
    }

    #[test]
    fn test_compute_is_deterministic() {
        let a = vec![1i8; 4 * 64];
        let b = vec![2i8; 64 * 4];
        assert_eq!(compute(&a, &b, &cfg()), compute(&a, &b, &cfg()));
    }

    #[test]
    fn test_ha_depends_on_a_not_b() {
        let b = vec![3i8; 64 * 4];
        let c1 = compute(&vec![1i8; 4 * 64], &b, &cfg());
        let c2 = compute(&vec![2i8; 4 * 64], &b, &cfg());
        assert_ne!(c1.ha, c2.ha);
        assert_eq!(c1.hb, c2.hb);
    }

    #[test]
    fn test_hb_depends_on_b_not_a() {
        let a = vec![1i8; 4 * 64];
        let c1 = compute(&a, &vec![2i8; 64 * 4], &cfg());
        let c2 = compute(&a, &vec![3i8; 64 * 4], &cfg());
        assert_ne!(c1.hb, c2.hb);
        assert_eq!(c1.ha, c2.ha);
    }

    #[test]
    fn test_sa_changes_when_ha_changes() {
        let b = vec![0i8; 64 * 4];
        let c1 = compute(&vec![1i8; 4 * 64], &b, &cfg());
        let c2 = compute(&vec![2i8; 4 * 64], &b, &cfg());
        assert_ne!(c1.s_a, c2.s_a);
    }

    #[test]
    fn test_sa_changes_when_hb_changes() {
        let a = vec![0i8; 4 * 64];
        let c1 = compute(&a, &vec![1i8; 64 * 4], &cfg());
        let c2 = compute(&a, &vec![2i8; 64 * 4], &cfg());
        assert_ne!(c1.s_a, c2.s_a);
    }
}
