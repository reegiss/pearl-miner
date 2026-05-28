pub mod slot;
pub mod jobs;
pub mod worker;

use std::sync::Arc;
use tokio::sync::{mpsc, watch};
use tokio::task::JoinHandle;
use pearl_block::PearlBlock;
use pearl_types::MiningConfig;

use slot::LastSlot;
pub use jobs::{PreparedJob, RawJob};

#[derive(Debug)]
pub enum PipelineError {
    /// Job displaced by a newer submission before preprocessing began.
    Dropped,
    /// A worker task panicked or all GPU workers exited.
    WorkerPanic,
}

/// Clone-able submit handle. Multiple callers may call submit() concurrently.
#[derive(Clone)]
pub struct PipelineHandle {
    pub(crate) raw_slot: Arc<LastSlot<RawJob>>,
    pub(crate) config:   Arc<MiningConfig>,
}

impl PipelineHandle {
    /// Submit (A, B) for mining. Returns clean A·B when any GPU finishes.
    ///
    /// # Errors
    ///
    /// Returns `Err(PipelineError::Dropped)` if a newer submission displaced
    /// this one before the preprocessor started working on it.
    /// Returns `Err(PipelineError::WorkerPanic)` if all GPU workers exited.
    pub async fn submit(
        &self,
        a: Arc<[i8]>,
        b: Arc<[i8]>,
    ) -> Result<Vec<i32>, PipelineError> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let job = RawJob {
            a,
            b,
            config: Arc::clone(&self.config),
            result_tx,
        };
        if let Some(displaced) = self.raw_slot.put(job) {
            let _ = displaced.result_tx.send(Err(PipelineError::Dropped));
        }
        result_rx.await.map_err(|_| PipelineError::WorkerPanic)?
    }
}

/// Receives found blocks. Not Clone — single consumer expected.
pub struct BlockReceiver {
    pub(crate) rx: mpsc::Receiver<PearlBlock>,
}

impl BlockReceiver {
    /// Returns `None` when the pipeline shuts down.
    pub async fn next(&mut self) -> Option<PearlBlock> {
        self.rx.recv().await
    }
}

/// Owns all worker [`JoinHandle`]s. Drop to shut down.
pub struct MiningPipeline {
    _handles: Vec<JoinHandle<()>>,
}

impl MiningPipeline {
    /// Spawns preprocessor + one GPU worker per device_id.
    pub fn start(
        config: Arc<MiningConfig>,
        device_ids: &[u32],
    ) -> (MiningPipeline, PipelineHandle, BlockReceiver) {
        let raw_slot = Arc::new(slot::LastSlot::<RawJob>::new());
        let (prepared_tx, _) = watch::channel(None::<Arc<PreparedJob>>);
        let (block_tx, block_rx) = mpsc::channel::<PearlBlock>(64);

        let mut handles: Vec<JoinHandle<()>> = Vec::new();

        // Spawn one GPU worker per device
        for &device_id in device_ids {
            let prepared_rx = prepared_tx.subscribe();
            let btx         = block_tx.clone();

            handles.push(tokio::spawn(async move {
                let gpu_result = tokio::task::spawn_blocking(
                    move || pearl_gpu::GpuMiner::new(device_id)
                ).await;

                match gpu_result {
                    Ok(Ok(gpu)) => {
                        let mine_fn = move |a: &[i8],
                                            b: &[i8],
                                            c: &pearl_types::Commitments,
                                            cfg: &pearl_types::MiningConfig| {
                            gpu.mine(a, b, c, cfg)
                        };
                        worker::gpu_worker_loop(prepared_rx, btx, mine_fn).await;
                    }
                    _ => {}
                }
            }));
        }

        // Spawn the preprocessor after GPU workers so receiver_count > 0 at start
        let slot_clone = Arc::clone(&raw_slot);
        handles.push(tokio::spawn(worker::preprocessor_loop(slot_clone, prepared_tx)));

        let handle = PipelineHandle { raw_slot, config };

        (MiningPipeline { _handles: handles }, handle, BlockReceiver { rx: block_rx })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_handle_is_clone() {
        fn assert_clone<T: Clone>() {}
        assert_clone::<PipelineHandle>();
    }

    use std::time::Duration;
    use tokio::time::timeout;
    use pearl_types::MatrixParams;

    fn test_config() -> Arc<MiningConfig> {
        Arc::new(MiningConfig {
            params: MatrixParams { m: 4, n: 4, k: 64, r: 32, tm: 2, tn: 2 },
            difficulty_bits: 256.0,
            sigma: b"test-sigma".to_vec(),
            mu:    b"test-mu".to_vec(),
        })
    }

    #[tokio::test]
    async fn test_drop_oldest_signals_dropped() {
        let raw_slot = Arc::new(slot::LastSlot::<RawJob>::new());
        let (result_tx1, result_rx1) =
            tokio::sync::oneshot::channel::<Result<Vec<i32>, PipelineError>>();
        let (result_tx2, _result_rx2) =
            tokio::sync::oneshot::channel::<Result<Vec<i32>, PipelineError>>();

        let config = test_config();
        let a: Arc<[i8]> = vec![0i8; 4 * 64].into();
        let b: Arc<[i8]> = vec![0i8; 64 * 4].into();

        raw_slot.put(RawJob {
            a: Arc::clone(&a), b: Arc::clone(&b),
            config: Arc::clone(&config), result_tx: result_tx1,
        });
        if let Some(displaced) = raw_slot.put(RawJob {
            a, b, config, result_tx: result_tx2,
        }) {
            let _ = displaced.result_tx.send(Err(PipelineError::Dropped));
        }

        let result = result_rx1.await.unwrap();
        assert!(matches!(result, Err(PipelineError::Dropped)));
    }

    #[tokio::test]
    async fn test_pipeline_start_compiles_and_drops_cleanly() {
        let (pipeline, handle, _blocks) =
            MiningPipeline::start(test_config(), &[]);
        let _handle2 = handle.clone();
        drop(pipeline);
    }
}
