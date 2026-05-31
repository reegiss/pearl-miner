mod miner;
mod pool;

use clap::Parser;
use pearl_gpu::GpuMiner;
use pearl_types::MiningParams;
use std::sync::Arc;
use tokio::sync::{mpsc, watch};

#[derive(Parser)]
#[command(name = "pearl-miner", version, about = "Pearl Miner — Proof-of-Useful-Work")]
struct Cli {
    #[arg(short, long, help = "Wallet address for mining rewards")]
    wallet: String,

    #[arg(short, long, help = "Pool URL (e.g. us1.alphapool.tech:5566)")]
    pool: String,

    #[arg(long, default_value = "x", help = "Pool password")]
    password: String,

    #[arg(
        long,
        value_delimiter = ',',
        help = "GPU device indices to use (default: all). E.g. --gpu 0,1,2"
    )]
    gpu: Vec<usize>,
}

#[tokio::main]
async fn main() {
    println!("Pearl Miner v{}", env!("CARGO_PKG_VERSION"));
    println!("Copyright (c) 2026 Pearl Protocol");
    println!();

    let args = Cli::parse();

    let device_ids: Vec<usize> = if args.gpu.is_empty() {
        let count = pearl_gpu::device_count();
        if count == 0 {
            eprintln!("No CUDA devices found.");
            std::process::exit(1);
        }
        (0..count).collect()
    } else {
        args.gpu.clone()
    };

    let miner = match miner::Miner::new(&device_ids[..]) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to initialize GPU(s): {e}");
            std::process::exit(1);
        }
    };

    // Pool GPU uses the LAST device to avoid contention with mining GPU 0.
    // If only 1 GPU available, they share device 0.
    let pool_device = *device_ids.last().unwrap();
    let pool_gpu = match GpuMiner::new(pool_device, &pearl_types::MiningParams {
        sigma: [0u8; 32], difficulty: 32,
        r: 256, k: 4096, tm: 16, tn: 16, m: 8192, n: 8192,
    }) {
        Ok(g) => Arc::new(g),
        Err(e) => {
            eprintln!("Failed to init pool GPU: {e}");
            std::process::exit(1);
        }
    };

    // challenge_tx carries (sigma_hex, pouw_difficulty, job_id) from mining.notify
    let (challenge_tx, challenge_rx) = watch::channel(None::<(String, u32, String)>);
    // params_tx carries pool-assigned matrix dimensions
    let (params_tx, params_rx)       = watch::channel(None::<MiningParams>);
    let (submit_tx, submit_rx)       = mpsc::channel(32);

    let pool_addr   = args.pool.clone();
    let pool_wallet = args.wallet.clone();
    let pool_pass   = args.password.clone();
    let n_gpus = device_ids.len();
    tokio::spawn(async move {
        pool::run(&pool_addr, &pool_wallet, &pool_pass, n_gpus, pool_gpu, challenge_tx, params_tx, submit_rx).await;
    });

    miner.run(args.wallet, challenge_rx, params_rx, submit_tx).await;
}
