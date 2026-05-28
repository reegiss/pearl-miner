pub mod slot;
pub mod jobs;
pub mod worker;

#[derive(Debug)]
pub enum PipelineError {
    Dropped,
    WorkerPanic,
}

pub use jobs::{PreparedJob, RawJob};
