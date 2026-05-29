use pearl_types::{BlockCertificate, Commitments, FoundTile};

#[derive(Debug, Clone, PartialEq)]
pub struct PearlBlock {
    pub version: u32,
    pub prev_hash: [u8; 32],
    pub tx_root: [u8; 32],
    pub timestamp: u64,
    pub n_bits: u32,
    pub certificate: BlockCertificate,
    pub transactions: Vec<Vec<u8>>,
}

pub fn serialize(block: &PearlBlock) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&block.version.to_le_bytes());
    buf.extend_from_slice(&block.prev_hash);
    buf.extend_from_slice(&block.tx_root);
    buf.extend_from_slice(&block.timestamp.to_le_bytes());
    buf.extend_from_slice(&block.n_bits.to_le_bytes());

    let c = &block.certificate;
    buf.extend_from_slice(&c.commitments.kappa);
    buf.extend_from_slice(&c.commitments.ha);
    buf.extend_from_slice(&c.commitments.hb);
    buf.extend_from_slice(&c.commitments.s_a);
    buf.extend_from_slice(&c.commitments.s_b);
    buf.extend_from_slice(&c.tile.tile_i.to_le_bytes());
    buf.extend_from_slice(&c.tile.tile_j.to_le_bytes());
    for &v in &c.tile.m_state {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf.extend_from_slice(&c.tile.final_hash);
    buf.extend_from_slice(&(c.merkle_proof_a.len() as u32).to_le_bytes());
    for h in &c.merkle_proof_a {
        buf.extend_from_slice(h);
    }
    buf.extend_from_slice(&(c.merkle_proof_b.len() as u32).to_le_bytes());
    for h in &c.merkle_proof_b {
        buf.extend_from_slice(h);
    }
    buf.extend_from_slice(&(block.transactions.len() as u32).to_le_bytes());
    for tx in &block.transactions {
        buf.extend_from_slice(&(tx.len() as u32).to_le_bytes());
        buf.extend_from_slice(tx);
    }
    buf
}

pub fn deserialize(bytes: &[u8]) -> Result<PearlBlock, String> {
    let mut pos = 0usize;

    macro_rules! take {
        ($n:expr) => {{
            if pos + $n > bytes.len() {
                return Err(format!("unexpected end of data at offset {pos}"));
            }
            let s = &bytes[pos..pos + $n];
            pos += $n;
            s
        }};
    }
    macro_rules! r32 {
        () => {
            u32::from_le_bytes(take!(4).try_into().unwrap())
        };
    }
    macro_rules! r64 {
        () => {
            u64::from_le_bytes(take!(8).try_into().unwrap())
        };
    }
    macro_rules! rh {
        () => {
            <[u8; 32]>::try_from(take!(32)).unwrap()
        };
    }

    let version = r32!();
    let prev_hash = rh!();
    let tx_root = rh!();
    let timestamp = r64!();
    let n_bits = r32!();

    let commitments = Commitments {
        kappa: rh!(),
        ha: rh!(),
        hb: rh!(),
        s_a: rh!(),
        s_b: rh!(),
    };

    let tile_i = r32!();
    let tile_j = r32!();
    let mut m_state = [0i32; 16];
    for v in &mut m_state {
        *v = i32::from_le_bytes(take!(4).try_into().unwrap());
    }
    let final_hash = rh!();
    let tile = FoundTile { tile_i, tile_j, m_state, final_hash };

    let pa_len = r32!() as usize;
    let mut merkle_proof_a = Vec::with_capacity(pa_len);
    for _ in 0..pa_len {
        merkle_proof_a.push(rh!());
    }

    let pb_len = r32!() as usize;
    let mut merkle_proof_b = Vec::with_capacity(pb_len);
    for _ in 0..pb_len {
        merkle_proof_b.push(rh!());
    }

    let certificate = BlockCertificate { commitments, tile, merkle_proof_a, merkle_proof_b };

    let tx_count = r32!() as usize;
    let mut transactions = Vec::with_capacity(tx_count);
    for _ in 0..tx_count {
        let tx_len = r32!() as usize;
        transactions.push(take!(tx_len).to_vec());
    }

    Ok(PearlBlock { version, prev_hash, tx_root, timestamp, n_bits, certificate, transactions })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pearl_types::{BlockCertificate, Commitments, FoundTile};

    fn test_cert() -> BlockCertificate {
        BlockCertificate {
            commitments: Commitments {
                kappa: [0u8; 32], ha: [1u8; 32], hb: [2u8; 32],
                s_a: [3u8; 32], s_b: [4u8; 32],
            },
            tile: FoundTile {
                tile_i: 7, tile_j: 13,
                m_state: [42i32; 16],
                final_hash: [5u8; 32],
            },
            merkle_proof_a: vec![[6u8; 32], [7u8; 32]],
            merkle_proof_b: vec![[8u8; 32]],
        }
    }

    #[test]
    fn test_roundtrip_empty_transactions() {
        let block = PearlBlock {
            version: 1,
            prev_hash: [0xABu8; 32],
            tx_root: [0xCDu8; 32],
            timestamp: 1_700_000_000u64,
            n_bits: 0x1b00ffff,
            certificate: test_cert(),
            transactions: vec![],
        };
        let bytes = serialize(&block);
        let back = deserialize(&bytes).expect("deserialize failed");
        assert_eq!(back.version, block.version);
        assert_eq!(back.prev_hash, block.prev_hash);
        assert_eq!(back.tx_root, block.tx_root);
        assert_eq!(back.timestamp, block.timestamp);
        assert_eq!(back.n_bits, block.n_bits);
        assert_eq!(back.certificate, block.certificate);
        assert_eq!(back.transactions, block.transactions);
    }

    #[test]
    fn test_roundtrip_with_transactions() {
        let block = PearlBlock {
            version: 2,
            prev_hash: [1u8; 32],
            tx_root: [2u8; 32],
            timestamp: 9999,
            n_bits: 0xdeadbeef,
            certificate: test_cert(),
            transactions: vec![vec![0x01, 0x02, 0x03], vec![0xff; 100]],
        };
        let bytes = serialize(&block);
        let back = deserialize(&bytes).unwrap();
        assert_eq!(back.transactions, block.transactions);
    }

    #[test]
    fn test_deserialize_truncated_returns_error() {
        let block = PearlBlock {
            version: 1,
            prev_hash: [0u8; 32],
            tx_root: [0u8; 32],
            timestamp: 0,
            n_bits: 0,
            certificate: test_cert(),
            transactions: vec![],
        };
        let mut bytes = serialize(&block);
        bytes.truncate(bytes.len() / 2);
        assert!(deserialize(&bytes).is_err());
    }
}
