use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;
use pearl_types::{Commitments, MiningConfig};

use crate::pipeline::PipelineError;

/// Lives from submit() until the preprocessor calls LastSlot::take().
pub struct RawJob {
    pub a:         Arc<[i8]>,
    pub b:         Arc<[i8]>,
    pub config:    Arc<MiningConfig>,
    pub result_tx: oneshot::Sender<Result<Vec<i32>, PipelineError>>,
}

/// Lives from end-of-preprocessing until all GPU workers finish with it.
/// Shared across workers via Arc.
pub struct PreparedJob {
    pub a:       Arc<[i8]>,
    pub b:       Arc<[i8]>,
    pub a_noisy: Vec<i8>,
    pub b_noisy: Vec<i8>,
    pub el:      Vec<i8>,
    pub er:      Vec<i8>,
    pub fl:      Vec<i8>,
    pub fr:      Vec<i8>,
    pub commitments: Arc<Commitments>,
    pub config:      Arc<MiningConfig>,
    /// First GPU worker to take the sender wins; others see None.
    pub result_tx: Arc<Mutex<Option<oneshot::Sender<Result<Vec<i32>, PipelineError>>>>>,
}
