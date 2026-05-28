mod pipeline;
mod args;
mod pool;

use std::sync::Arc;
use anyhow::Result;
use clap::Parser;
use pearl_types::{MatrixParams, MiningConfig};
use pipeline::{MiningPipeline, PipelineError};
use args::Args;
use pool::{pool_task, PoolChallenge};
use tokio::sync::mpsc;

fn make_config(challenge: &PoolChallenge, mu: &[u8]) -> Arc<MiningConfig> {
    Arc::new(MiningConfig {
        params: MatrixParams { m: 32, n: 32, k: 512, r: 32, tm: 4, tn: 4 },
        difficulty_bits: challenge.difficulty,
        sigma: challenge.seed.to_vec(),
        mu: mu.to_vec(),
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let mu = args.wallet_bytes()
        .map_err(|e| anyhow::anyhow!("{}", e))?;
    args.validate_pool()
        .map_err(|e| anyhow::anyhow!("{}", e))?;

    eprintln!("[init] wallet: {}", args.wallet);
    eprintln!("[init] pool:   {}", args.pool);

    let n_devices = pearl_gpu::device_count();
    if n_devices == 0 {
        eprintln!("[init] no CUDA devices found — exiting");
        return Ok(());
    }
    let device_ids: Vec<u32> = (0..n_devices as u32).collect();
    eprintln!("[init] detected {} CUDA device(s): {:?}", n_devices, device_ids);

    // Pool channels
    let (challenge_tx, mut challenge_rx) = tokio::sync::watch::channel(None::<PoolChallenge>);
    let (block_tx, block_rx)             = mpsc::channel::<pearl_block::PearlBlock>(32);

    tokio::spawn(pool_task(
        args.pool.clone(),
        args.wallet.clone(),
        challenge_tx,
        block_rx,
    ));

    // Wait for the first challenge before starting the pipeline
    eprintln!("[pool] waiting for first challenge...");
    challenge_rx.changed().await
        .map_err(|_| anyhow::anyhow!("pool task exited before first challenge"))?;
    let first = challenge_rx.borrow().clone().unwrap();
    eprintln!("[pool] first challenge — seed={} difficulty={}",
        hex(&first.seed), first.difficulty);

    // Synthetic matrices (in production these come from the AI workload)
    let p = MatrixParams { m: 32, n: 32, k: 512, r: 32, tm: 4, tn: 4 };
    let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);
    let a: Arc<[i8]> = (0..m * k)
        .map(|i| ((i * 7 + 3) % 128) as i8 - 64)
        .collect::<Vec<_>>().into();
    let b: Arc<[i8]> = (0..k * n)
        .map(|i| ((i * 11 + 5) % 128) as i8 - 64)
        .collect::<Vec<_>>().into();
    eprintln!("[data] generated {}×{} A and {}×{} B (INT8)", m, k, k, n);

    let config = make_config(&first, &mu);
    let (mut pipeline, mut handle, mut blocks) =
        MiningPipeline::start(Arc::clone(&config), &device_ids);

    loop {
        tokio::select! {
            // New challenge → restart pipeline with updated sigma/difficulty
            Ok(_) = challenge_rx.changed() => {
                let ch = challenge_rx.borrow().clone().unwrap();
                eprintln!("[pool] new challenge — seed={} difficulty={}",
                    hex(&ch.seed), ch.difficulty);
                let new_config = make_config(&ch, &mu);
                drop(pipeline);
                (pipeline, handle, blocks) =
                    MiningPipeline::start(Arc::clone(&new_config), &device_ids);
            }

            // Block found → forward to pool_task for submission
            Some(block) = blocks.next() => {
                let _ = block_tx.send(block).await;
            }

            // Mining pipeline result (clean A·B)
            result = handle.submit(Arc::clone(&a), Arc::clone(&b)) => {
                match result {
                    Ok(clean_ab) => {
                        eprintln!("[peel] recovered {}×{} product ({} elements)",
                            m, n, clean_ab.len());
                    }
                    Err(PipelineError::Dropped)     => {}
                    Err(PipelineError::WorkerPanic) => {
                        eprintln!("[submit] GPU workers exited — shutting down");
                        break;
                    }
                }
            }
        }
    }

    Ok(())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
