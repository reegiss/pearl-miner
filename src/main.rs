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

    #[arg(long, default_value = "0", help = "CUDA device index")]
    gpu: usize,
}

#[tokio::main]
async fn main() {
    println!("Pearl Miner v{}", env!("CARGO_PKG_VERSION"));
    println!("Copyright (c) 2026 Pearl Protocol");
    println!();

    let args = Cli::parse();

    let miner = match miner::Miner::new(args.gpu) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Failed to initialize GPU {}: {e}", args.gpu);
            std::process::exit(1);
        }
    };

    // challenge: pool → miner
    let (challenge_tx, challenge_rx) = watch::channel(None);
    // submit: miner → pool
    let (submit_tx, submit_rx) = mpsc::channel(32);

    let pool_addr   = args.pool.clone();
    let pool_wallet = args.wallet.clone();
    let pool_pass   = args.password.clone();
    tokio::spawn(async move {
        pool::run(&pool_addr, &pool_wallet, &pool_pass, challenge_tx, submit_rx).await;
    });

    miner.run(args.wallet, challenge_rx, submit_tx).await;
}
