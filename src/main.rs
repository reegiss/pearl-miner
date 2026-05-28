mod pipeline;

use anyhow::Result;
use pearl_block::{block_identity, serialize, validate_certificate, PearlBlock};
use pearl_commitment::{compute as commitment_compute, MerkleTree};
use pearl_gpu::GpuMiner;
use pearl_noise::{generate_e, generate_f};
use pearl_peel::recover;
use pearl_types::{BlockCertificate, MiningConfig};

fn apply_noise(base: &[i8], el: &[i8], er: &[i8], rows: usize, cols: usize, rank: usize) -> Vec<i8> {
    // Compute base + EL·ER element-wise, clamped to i8
    // EL: rows×rank, ER: rank×cols → E: rows×cols
    // E[i][j] = sum_p EL[i*rank+p] * ER[p*cols+j]
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

fn run_pipeline(config: &MiningConfig, a: &[i8], b: &[i8], gpu: &GpuMiner) -> Result<()> {
    let p = &config.params;
    let (m, n, k, r) = (p.m as usize, p.n as usize, p.k as usize, p.r as usize);

    // 1. Commitment hash
    let commitments = commitment_compute(a, b, config);
    eprintln!("[commitment] sA={}", hex(&commitments.s_a));

    // 2. Noise generation
    let (el, er) = generate_e(p.m, p.k, p.r, &commitments.s_a);
    let (fl, fr) = generate_f(p.k, p.n, p.r, &commitments.s_b);

    // 3. Noisy matrices: A' = A + E, B' = B + F
    let a_noisy = apply_noise(a, &el, &er, m, k, r);
    let b_noisy = apply_noise(b, &fl, &fr, k, n, r);

    // 4. GPU: tiled INT8 matmul + difficulty check
    let (found_tiles, ab_noisy) = gpu.mine(&a_noisy, &b_noisy, &commitments, config)?;

    // 5. Clean product recovery (always — AI workload needs A·B)
    let ab_clean = recover(&ab_noisy, a, &b_noisy, &el, &er, &fl, &fr, &config.params);
    eprintln!("[peel] recovered {}×{} clean product", m, n);
    let _ = ab_clean; // delivered to AI workload in production

    // 6. If a block was found, assemble and report
    if found_tiles.is_empty() {
        eprintln!("[mine] no block found for this (A, B) pair");
        return Ok(());
    }

    eprintln!("[mine] {} winning tile(s) found!", found_tiles.len());

    let tile = &found_tiles[0];
    eprintln!("[mine] winning tile: ({}, {})  hash={}",
        tile.tile_i, tile.tile_j, hex(&tile.final_hash));

    // Verify on CPU
    let cert = BlockCertificate {
        commitments: commitments.clone(),
        tile: tile.clone(),
        merkle_proof_a: MerkleTree::from_rows(a, k).proof(tile.tile_i as usize),
        merkle_proof_b: MerkleTree::from_rows(b, m).proof(tile.tile_j as usize),
    };
    assert!(validate_certificate(&cert, config), "certificate validation failed");

    let block = PearlBlock {
        version:      1,
        prev_hash:    [0u8; 32], // would come from P2P in production
        tx_root:      [0u8; 32],
        timestamp:    std::time::SystemTime::now()
                          .duration_since(std::time::UNIX_EPOCH)
                          .unwrap_or_default()
                          .as_secs(),
        n_bits:       0x1b00ffff,
        certificate:  cert,
        transactions: vec![],
    };

    let identity = block_identity(&block);
    let bytes    = serialize(&block);
    eprintln!("[block] identity={}  size={} bytes", hex(&identity), bytes.len());

    Ok(())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn main() -> Result<()> {
    let config = MiningConfig {
        params: pearl_types::MatrixParams {
            m: 32, n: 32, k: 512, r: 32,
            tm: 4, tn: 4,
        },
        difficulty_bits: 1.0, // very low: almost every tile wins (for testing)
        sigma: b"pearl-genesis-block".to_vec(),
        mu:    b"miner-pubkey-placeholder".to_vec(),
    };

    eprintln!("[init] m={} n={} k={} r={} tm={} tn={} difficulty={}",
        config.params.m, config.params.n, config.params.k,
        config.params.r, config.params.tm, config.params.tn,
        config.difficulty_bits);

    let gpu = GpuMiner::new(0)?;
    eprintln!("[init] GPU 0 ready");

    // Generate test matrices (in production these come from AI workload)
    let p = &config.params;
    let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);
    let a: Vec<i8> = (0..m * k).map(|i| ((i * 7 + 3) % 128) as i8 - 64).collect();
    let b: Vec<i8> = (0..k * n).map(|i| ((i * 11 + 5) % 128) as i8 - 64).collect();
    eprintln!("[data] generated {}×{} A and {}×{} B (INT8)", m, k, k, n);

    run_pipeline(&config, &a, &b, &gpu)
}
