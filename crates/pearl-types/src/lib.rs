#[derive(Debug, Clone, PartialEq)]
pub struct MatrixParams {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub r: u32,
    pub tm: u32,
    pub tn: u32,
}

#[derive(Debug, Clone)]
pub struct MiningConfig {
    pub params: MatrixParams,
    pub difficulty_bits: f64,
    pub sigma: Vec<u8>,
    pub mu: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Commitments {
    pub kappa: [u8; 32],
    pub ha: [u8; 32],
    pub hb: [u8; 32],
    pub s_a: [u8; 32],
    pub s_b: [u8; 32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct FoundTile {
    pub tile_i: u32,
    pub tile_j: u32,
    pub m_state: [i32; 16],
    pub final_hash: [u8; 32],
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockCertificate {
    pub commitments: Commitments,
    pub tile: FoundTile,
    pub merkle_proof_a: Vec<[u8; 32]>,
    pub merkle_proof_b: Vec<[u8; 32]>,
}
