mod pipeline;

use std::sync::Arc;
use anyhow::Result;
use pearl_types::{MatrixParams, MiningConfig};
use pipeline::{MiningPipeline, PipelineError};

#[tokio::main]
async fn main() -> Result<()> {
    let config = Arc::new(MiningConfig {
        params: MatrixParams {
            m: 32, n: 32, k: 512, r: 32,
            tm: 4,  tn: 4,
        },
        difficulty_bits: 1.0,
        sigma: b"pearl-genesis-block".to_vec(),
        mu:    b"miner-pubkey-placeholder".to_vec(),
    });

    let n_devices = pearl_gpu::device_count();
    if n_devices == 0 {
        eprintln!("[init] no CUDA devices found — exiting");
        return Ok(());
    }
    let device_ids: Vec<u32> = (0..n_devices as u32).collect();
    eprintln!("[init] detected {} CUDA device(s): {:?}", n_devices, device_ids);

    let (_pipeline, handle, mut blocks) =
        MiningPipeline::start(Arc::clone(&config), &device_ids);

    // Print found blocks in a background task
    tokio::spawn(async move {
        while let Some(block) = blocks.next().await {
            let id    = pearl_block::block_identity(&block);
            let bytes = pearl_block::serialize(&block);
            eprintln!("[block] found! identity={}  size={} bytes", hex(&id), bytes.len());
        }
        eprintln!("[block] block receiver closed");
    });

    // Mining loop — submit the same synthetic (A, B) pair repeatedly.
    // In production, A and B come from the AI inference workload.
    let p = &config.params;
    let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);
    let a: Arc<[i8]> = (0..m * k)
        .map(|i| ((i * 7 + 3) % 128) as i8 - 64)
        .collect::<Vec<_>>()
        .into();
    let b: Arc<[i8]> = (0..k * n)
        .map(|i| ((i * 11 + 5) % 128) as i8 - 64)
        .collect::<Vec<_>>()
        .into();
    eprintln!("[data] generated {}×{} A and {}×{} B (INT8)", m, k, k, n);

    loop {
        match handle.submit(Arc::clone(&a), Arc::clone(&b)).await {
            Ok(clean_ab) => {
                eprintln!("[peel] recovered {}×{} clean product ({} elements)",
                    m, n, clean_ab.len());
            }
            Err(PipelineError::Dropped) => {
                eprintln!("[submit] job dropped — superseded by newer submission");
            }
            Err(PipelineError::WorkerPanic) => {
                eprintln!("[submit] all GPU workers exited — shutting down");
                break;
            }
        }
    }

    Ok(())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
