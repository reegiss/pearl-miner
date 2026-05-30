mod miner;
mod pool;

use clap::Parser;
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

    // Resolve device list: use all GPUs if none specified
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

    let (challenge_tx, challenge_rx) = watch::channel(None);
    let (submit_tx, submit_rx)       = mpsc::channel(32);

    let pool_addr   = args.pool.clone();
    let pool_wallet = args.wallet.clone();
    let pool_pass   = args.password.clone();
    tokio::spawn(async move {
        pool::run(&pool_addr, &pool_wallet, &pool_pass, challenge_tx, submit_rx).await;
    });

    miner.run(args.wallet, challenge_rx, submit_tx).await;
}
