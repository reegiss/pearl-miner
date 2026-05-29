use num_bigint::BigUint;
use num_traits::One;
use sha2::{Digest, Sha256};

use pearl_types::{BlockCertificate, MiningConfig};

use crate::serial::PearlBlock;

pub fn block_identity(block: &PearlBlock) -> [u8; 32] {
    // pouw_meta = SHA256(final_hash) — stub until pearl-proof is added
    let pouw_meta = Sha256::digest(&block.certificate.tile.final_hash);

    let mut header = Vec::with_capacity(4 + 32 + 32 + 8 + 4 + 32);
    header.extend_from_slice(&block.version.to_le_bytes());
    header.extend_from_slice(&block.prev_hash);
    header.extend_from_slice(&block.tx_root);
    header.extend_from_slice(&block.timestamp.to_le_bytes());
    header.extend_from_slice(&block.n_bits.to_le_bytes());
    header.extend_from_slice(&pouw_meta);

    let first: [u8; 32] = Sha256::digest(&header).into();
    let second: [u8; 32] = Sha256::digest(first).into();
    second
}

pub fn validate_certificate(cert: &BlockCertificate, config: &MiningConfig) -> bool {
    let mut m_bytes = [0u8; 64];
    for (i, &v) in cert.tile.m_state.iter().enumerate() {
        m_bytes[i * 4..(i + 1) * 4].copy_from_slice(&v.to_le_bytes());
    }

    let hash = blake3::keyed_hash(&cert.commitments.s_a, &m_bytes);
    let hash_val = BigUint::from_bytes_le(hash.as_bytes());
    let threshold = difficulty_threshold(
        config.difficulty_bits,
        config.params.r,
        config.params.tm,
        config.params.tn,
    );
    hash_val <= threshold
}

/// threshold = floor(2^(256 - floor(b))) · r · tm · tn
fn difficulty_threshold(b: f64, r: u32, tm: u32, tn: u32) -> BigUint {
    let b_int = b.floor() as u32;
    let shift = 256u32.saturating_sub(b_int);
    (BigUint::one() << shift) * r * tm * tn
}

#[cfg(test)]
mod tests {
    use super::*;
    use pearl_types::{BlockCertificate, Commitments, FoundTile, MatrixParams, MiningConfig};

    use crate::serial::PearlBlock;

    fn test_cert(s_a: [u8; 32]) -> BlockCertificate {
        BlockCertificate {
            commitments: Commitments {
                kappa: [0u8; 32], ha: [0u8; 32], hb: [0u8; 32],
                s_a, s_b: [0u8; 32],
            },
            tile: FoundTile {
                tile_i: 0, tile_j: 0,
                m_state: [99i32; 16],
                final_hash: [0u8; 32],
            },
            merkle_proof_a: vec![],
            merkle_proof_b: vec![],
        }
    }

    fn cfg(b: f64) -> MiningConfig {
        MiningConfig {
            params: MatrixParams { m: 4, n: 4, k: 64, r: 32, tm: 2, tn: 2 },
            difficulty_bits: b,
            sigma: vec![], mu: vec![],
        }
    }

    fn test_block(cert: BlockCertificate) -> PearlBlock {
        PearlBlock {
            version: 1,
            prev_hash: [1u8; 32],
            tx_root: [2u8; 32],
            timestamp: 12345,
            n_bits: 0x1b00ffff,
            certificate: cert,
            transactions: vec![],
        }
    }

    #[test]
    fn test_validate_certificate_passes_with_zero_difficulty() {
        let cert = test_cert([1u8; 32]);
        assert!(validate_certificate(&cert, &cfg(0.0)));
    }

    #[test]
    fn test_validate_certificate_fails_with_max_difficulty() {
        let cert = test_cert([1u8; 32]);
        assert!(!validate_certificate(&cert, &cfg(256.0)));
    }

    #[test]
    fn test_block_identity_is_deterministic() {
        let block = test_block(test_cert([0u8; 32]));
        assert_eq!(block_identity(&block), block_identity(&block));
    }

    #[test]
    fn test_block_identity_changes_with_prev_hash() {
        let b1 = test_block(test_cert([0u8; 32]));
        let mut b2 = b1.clone();
        b2.prev_hash = [0xffu8; 32];
        assert_ne!(block_identity(&b1), block_identity(&b2));
    }

    #[test]
    fn test_block_identity_changes_with_timestamp() {
        let b1 = test_block(test_cert([0u8; 32]));
        let mut b2 = b1.clone();
        b2.timestamp = b1.timestamp + 1;
        assert_ne!(block_identity(&b1), block_identity(&b2));
    }
}
