use pearl_block::PearlBlock;
use pearl_commitment::MerkleTree;
use pearl_types::{BlockCertificate, Commitments, FoundTile, MiningConfig};

/// Add noise EL·ER to `base` element-wise (saturating i8 add).
/// base: rows×cols, EL: rows×rank, ER: rank×cols — all row-major flat arrays.
pub fn apply_noise(base: &[i8], el: &[i8], er: &[i8], rows: usize, cols: usize, rank: usize) -> Vec<i8> {
    let mut result = base.to_vec();
    for i in 0..rows {
        for j in 0..cols {
            let mut e: i32 = 0;
            for p in 0..rank {
                e += el[i * rank + p] as i32 * er[p * cols + j] as i32;
            }
            result[i * cols + j] = base[i * cols + j].saturating_add(e as i8);
        }
    }
    result
}

/// Build a PearlBlock from a winning tile.
/// prev_hash and tx_root are zero-filled until P2P is integrated.
pub fn assemble_block(
    tile: &FoundTile,
    a: &[i8],
    b: &[i8],
    commitments: &Commitments,
    config: &MiningConfig,
) -> PearlBlock {
    let k = config.params.k as usize;
    let m = config.params.m as usize;

    let proof_a = MerkleTree::from_rows(a, k).proof(tile.tile_i as usize);
    let proof_b = MerkleTree::from_rows(b, m).proof(tile.tile_j as usize);

    PearlBlock {
        version:     1,
        prev_hash:   [0u8; 32],
        tx_root:     [0u8; 32],
        timestamp:   std::time::SystemTime::now()
                         .duration_since(std::time::UNIX_EPOCH)
                         .unwrap_or_default()
                         .as_secs(),
        n_bits:      0x1b00ffff,
        certificate: BlockCertificate {
            commitments: commitments.clone(),
            tile:        tile.clone(),
            merkle_proof_a: proof_a,
            merkle_proof_b: proof_b,
        },
        transactions: vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use pearl_types::{Commitments, FoundTile, MatrixParams, MiningConfig};

    pub fn dummy_config() -> Arc<MiningConfig> {
        Arc::new(MiningConfig {
            params: MatrixParams { m: 4, n: 4, k: 64, r: 32, tm: 2, tn: 2 },
            difficulty_bits: 0.0,
            sigma: b"test".to_vec(),
            mu:    b"test".to_vec(),
        })
    }

    pub fn dummy_commitments() -> Arc<Commitments> {
        Arc::new(Commitments {
            kappa: [0u8; 32], ha: [1u8; 32], hb: [2u8; 32],
            s_a:   [3u8; 32], s_b: [4u8; 32],
        })
    }

    #[test]
    fn test_apply_noise_adds_el_er() {
        // EL: 2×2, all zeros; ER: 2×4, all zeros → noise is zero → result == base
        let base  = vec![1i8, 2, 3, 4, 5, 6, 7, 8]; // 2 rows × 4 cols
        let el    = vec![0i8; 2 * 2];
        let er    = vec![0i8; 2 * 4];
        let result = apply_noise(&base, &el, &er, 2, 4, 2);
        assert_eq!(result, base);
    }

    #[test]
    fn test_assemble_block_fields() {
        let a: Arc<[i8]> = vec![1i8; 4 * 64].into();
        let b: Arc<[i8]> = vec![2i8; 64 * 4].into();
        let config = dummy_config();
        let tile = FoundTile {
            tile_i: 1, tile_j: 2,
            m_state: [7i32; 16],
            final_hash: [0xAAu8; 32],
        };
        let commitments = dummy_commitments();

        let block = assemble_block(&tile, &a, &b, &commitments, &config);

        assert_eq!(block.version, 1);
        assert_eq!(block.certificate.tile, tile);
        assert_eq!(block.certificate.commitments, *commitments);
        assert!(!block.certificate.merkle_proof_a.is_empty());
        assert!(!block.certificate.merkle_proof_b.is_empty());
    }
}
