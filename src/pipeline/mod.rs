pub mod slot;
pub mod jobs;
pub mod worker;

use std::sync::Arc;
use tokio::sync::mpsc;
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
        todo!("implemented in Task 8")
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
}
