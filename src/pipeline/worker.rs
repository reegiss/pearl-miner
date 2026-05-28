use std::sync::Arc;
use anyhow::Result;
use tokio::sync::{mpsc, watch};
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

pub async fn preprocessor_loop(
    raw_slot:    Arc<crate::pipeline::slot::LastSlot<crate::pipeline::RawJob>>,
    prepared_tx: watch::Sender<Option<Arc<crate::pipeline::PreparedJob>>>,
) {
    use crate::pipeline::PreparedJob;
    use std::sync::Mutex;

    loop {
        if prepared_tx.receiver_count() == 0 {
            break; // all GPU workers have exited
        }

        let job = raw_slot.take().await;

        let a   = Arc::clone(&job.a);
        let b   = Arc::clone(&job.b);
        let cfg = Arc::clone(&job.config);
        let result_tx_raw = job.result_tx;

        let preprocess_result = tokio::task::spawn_blocking(move || {
            let p = &cfg.params;
            let (m, n, k, r) = (p.m as usize, p.n as usize, p.k as usize, p.r as usize);
            let commitments  = pearl_commitment::compute(&a, &b, &cfg);
            let (el, er)     = pearl_noise::generate_e(p.m, p.k, p.r, &commitments.s_a);
            let (fl, fr)     = pearl_noise::generate_f(p.k, p.n, p.r, &commitments.s_b);
            let a_noisy      = apply_noise(&a, &el, &er, m, k, r);
            let b_noisy      = apply_noise(&b, &fl, &fr, k, n, r);
            PreparedJob {
                a:           Arc::clone(&a),
                b:           Arc::clone(&b),
                a_noisy,
                b_noisy,
                el, er, fl, fr,
                commitments: Arc::new(commitments),
                config:      cfg,
                result_tx:   Arc::new(Mutex::new(Some(result_tx_raw))),
            }
        }).await;

        match preprocess_result {
            Ok(prepared) => {
                let _ = prepared_tx.send_replace(Some(Arc::new(prepared)));
            }
            Err(_panic) => {
                // result_tx_raw moved into spawn_blocking and dropped on panic;
                // oneshot closes → caller gets WorkerPanic via map_err
            }
        }
    }
}

pub async fn gpu_worker_loop<F>(
    mut prepared_rx: watch::Receiver<Option<Arc<crate::pipeline::PreparedJob>>>,
    block_tx: mpsc::Sender<PearlBlock>,
    mine_fn: F,
)
where
    F: FnMut(&[i8], &[i8], &Commitments, &MiningConfig)
         -> Result<(Vec<FoundTile>, Vec<i32>)>
         + Send + 'static,
{
    use crate::pipeline::PipelineError;
    let mine_fn = Arc::new(std::sync::Mutex::new(mine_fn));

    loop {
        if prepared_rx.changed().await.is_err() {
            break; // sender dropped — pipeline shut down
        }

        let job = {
            let borrow = prepared_rx.borrow_and_update();
            match borrow.as_ref() {
                Some(j) => Arc::clone(j),
                None    => continue,
            }
        };

        // Clone everything needed inside spawn_blocking
        let a_noisy     = job.a_noisy.clone();
        let b_noisy     = job.b_noisy.clone();
        let el          = job.el.clone();
        let er          = job.er.clone();
        let fl          = job.fl.clone();
        let fr          = job.fr.clone();
        let a_in        = Arc::clone(&job.a);
        let commitments = Arc::clone(&job.commitments);
        let config      = Arc::clone(&job.config);
        let params      = config.params.clone();
        let mine_fn2    = Arc::clone(&mine_fn);
        // Keep clones in async context for assemble_block after spawn_blocking
        let a_for_block   = Arc::clone(&job.a);
        let b_for_block   = Arc::clone(&job.b);
        let comm_for_block = Arc::clone(&job.commitments);
        let cfg_for_block  = Arc::clone(&job.config);
        let result_tx     = Arc::clone(&job.result_tx);

        let mine_result = tokio::task::spawn_blocking(move || {
            let mut f = mine_fn2.lock().unwrap();
            let (found_tiles, ab_noisy) = f(&a_noisy, &b_noisy, &commitments, &config)?;
            let clean_ab = pearl_peel::recover(
                &ab_noisy, &a_in, &b_noisy,
                &el, &er, &fl, &fr, &params,
            );
            Ok::<(Vec<FoundTile>, Vec<i32>), anyhow::Error>((found_tiles, clean_ab))
        }).await;

        match mine_result {
            Ok(Ok((found_tiles, clean_ab))) => {
                // First worker to take result_tx wins; others see None and skip
                if let Some(tx) = result_tx.lock().unwrap().take() {
                    let _ = tx.send(Ok(clean_ab));
                }
                // Send blocks in async context (cannot .await inside spawn_blocking)
                for tile in &found_tiles {
                    let block = assemble_block(
                        tile, &a_for_block, &b_for_block,
                        &comm_for_block, &cfg_for_block,
                    );
                    let _ = block_tx.send(block).await;
                }
            }
            Ok(Err(_)) | Err(_) => {
                if let Some(tx) = result_tx.lock().unwrap().take() {
                    let _ = tx.send(Err(PipelineError::WorkerPanic));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::watch;
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

    #[tokio::test]
    async fn test_preprocessor_dispatches_prepared_job() {
        use crate::pipeline::slot::LastSlot;
        use crate::pipeline::RawJob;

        let config = dummy_config();
        let raw_slot = Arc::new(LastSlot::<RawJob>::new());
        let (prepared_tx, mut prepared_rx) =
            watch::channel(None::<Arc<crate::pipeline::PreparedJob>>);

        let slot2 = Arc::clone(&raw_slot);
        tokio::spawn(preprocessor_loop(slot2, prepared_tx));

        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        let job = RawJob {
            a:         vec![1i8; 4 * 64].into(),
            b:         vec![2i8; 64 * 4].into(),
            config:    Arc::clone(&config),
            result_tx,
        };
        raw_slot.put(job);

        tokio::time::timeout(
            std::time::Duration::from_millis(500),
            prepared_rx.changed(),
        ).await.expect("preprocessor timed out").unwrap();

        let prepared = prepared_rx.borrow().clone()
            .expect("expected Some(PreparedJob)");
        assert_eq!(prepared.a_noisy.len(), 4 * 64);
        assert_eq!(prepared.b_noisy.len(), 64 * 4);
    }

    use crate::pipeline::{PipelineError, PreparedJob};
    use std::sync::Mutex;

    fn make_prepared_job(
        config: Arc<MiningConfig>,
        result_tx: tokio::sync::oneshot::Sender<Result<Vec<i32>, PipelineError>>,
    ) -> Arc<PreparedJob> {
        let (m, k, n, r) = (4usize, 64usize, 4usize, 32usize);
        let a: Arc<[i8]> = vec![1i8; m * k].into();
        let b: Arc<[i8]> = vec![2i8; k * n].into();
        Arc::new(PreparedJob {
            a:       Arc::clone(&a),
            b:       Arc::clone(&b),
            a_noisy: a.to_vec(),
            b_noisy: b.to_vec(),
            el: vec![0i8; m * r],
            er: vec![0i8; r * k],
            fl: vec![0i8; k * r],
            fr: vec![0i8; r * n],
            commitments: dummy_commitments(),
            config,
            result_tx: Arc::new(Mutex::new(Some(result_tx))),
        })
    }

    #[tokio::test]
    async fn test_worker_delivers_clean_product() {
        let config = dummy_config();
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let job = make_prepared_job(Arc::clone(&config), result_tx);

        let (prepared_tx, prepared_rx) = watch::channel(None::<Arc<PreparedJob>>);
        let (block_tx, _block_rx) = mpsc::channel(8);

        let mock_mine = move |_a: &[i8], _b: &[i8], _c: &Commitments, cfg: &MiningConfig| {
            let sz = (cfg.params.m * cfg.params.n) as usize;
            Ok::<(Vec<FoundTile>, Vec<i32>), anyhow::Error>((vec![], vec![42i32; sz]))
        };

        tokio::spawn(gpu_worker_loop(prepared_rx, block_tx, mock_mine));
        prepared_tx.send_replace(Some(job));

        let result = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            result_rx,
        ).await.expect("timed out").unwrap();

        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 4 * 4);
    }

    #[tokio::test]
    async fn test_worker_emits_block_when_tile_found() {
        let config = dummy_config();
        let (result_tx, _result_rx) = tokio::sync::oneshot::channel();
        let job = make_prepared_job(Arc::clone(&config), result_tx);

        let (prepared_tx, prepared_rx) = watch::channel(None::<Arc<PreparedJob>>);
        let (block_tx, mut block_rx) = mpsc::channel(8);

        let mock_mine = move |_a: &[i8], _b: &[i8], _c: &Commitments, cfg: &MiningConfig| {
            let tile = FoundTile {
                tile_i: 0, tile_j: 0,
                m_state: [1i32; 16],
                final_hash: [0u8; 32],
            };
            let sz = (cfg.params.m * cfg.params.n) as usize;
            Ok::<(Vec<FoundTile>, Vec<i32>), anyhow::Error>((vec![tile], vec![0i32; sz]))
        };

        tokio::spawn(gpu_worker_loop(prepared_rx, block_tx, mock_mine));
        prepared_tx.send_replace(Some(job));

        let block = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            block_rx.recv(),
        ).await.expect("timed out").expect("channel closed");
        assert_eq!(block.certificate.tile.tile_i, 0);
    }

    #[tokio::test]
    async fn test_two_workers_only_one_delivers_result() {
        let config = dummy_config();
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let job = make_prepared_job(Arc::clone(&config), result_tx);

        let (prepared_tx, prepared_rx1) = watch::channel(None::<Arc<PreparedJob>>);
        let prepared_rx2 = prepared_tx.subscribe();
        let (block_tx1, _) = mpsc::channel(8);
        let (block_tx2, _) = mpsc::channel(8);

        let mock_mine = move |_a: &[i8], _b: &[i8], _c: &Commitments, cfg: &MiningConfig| {
            let sz = (cfg.params.m * cfg.params.n) as usize;
            Ok::<(Vec<FoundTile>, Vec<i32>), anyhow::Error>((vec![], vec![0i32; sz]))
        };

        tokio::spawn(gpu_worker_loop(prepared_rx1, block_tx1, mock_mine));
        tokio::spawn(gpu_worker_loop(prepared_rx2, block_tx2, mock_mine));
        prepared_tx.send_replace(Some(job));

        let result = tokio::time::timeout(
            std::time::Duration::from_millis(500),
            result_rx,
        ).await.expect("timed out").unwrap();
        assert!(result.is_ok(), "expected Ok, got {:?}", result);
    }
}
