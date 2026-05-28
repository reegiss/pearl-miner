# Pearl Miner — CLI Arguments

**Date:** 2026-05-28  
**Status:** Approved  
**Scope:** `src/args.rs` + `src/main.rs` — argument parsing, wallet/pool validation, integration with `MiningConfig`.

---

## 1. Overview

The miner currently has all parameters hardcoded in `main.rs`. This design adds a proper CLI interface so operators can configure the wallet address and mining pool at startup without recompiling.

```
pearl-miner --wallet prl1p0jfkajyjgeu9eszntdfztcxyfjk5lh9mmvna69hd5ny83l8laqtq45rjx4 \
            --pool us1.alphapool.tech:5566
```

---

## 2. Dependencies

Add to root `Cargo.toml`:

```toml
clap   = { version = "4", features = ["derive"] }
bech32 = "0.11"
```

- **clap** — derive-based CLI parsing; generates `--help`, `--version`, and error messages automatically.
- **bech32** — decodes and validates bech32m wallet addresses.

---

## 3. File Layout

```
src/
├── args.rs    — new: Args struct, wallet_bytes(), validate_pool()
└── main.rs    — modified: calls Args::parse(), passes wallet bytes as mu
```

---

## 4. `src/args.rs`

```rust
#[derive(clap::Parser)]
#[command(name = "pearl-miner", version, about = "Pearl PoUW miner")]
pub struct Args {
    /// Endereço Taproot da carteira Pearl (bech32m, prefixo prl1)
    #[arg(long)]
    pub wallet: String,

    /// Endereço da pool de mineração (host:porta, ex: us1.alphapool.tech:5566)
    #[arg(long)]
    pub pool: String,
}

impl Args {
    /// Decode and validate the wallet address.
    /// Returns the witness-program bytes to use as MiningConfig.mu.
    pub fn wallet_bytes(&self) -> Result<Vec<u8>, String>;

    /// Validate the pool address (must be non-empty and contain ':').
    pub fn validate_pool(&self) -> Result<(), String>;
}
```

### Wallet validation (`wallet_bytes`)

1. Decode with `bech32::decode(&self.wallet)` — returns `(hrp, data, variant)`.
2. Check `variant == bech32::Variant::Bech32m` — Pearl uses bech32m, not legacy bech32.
3. Check `hrp == "prl"` — rejects addresses from other networks (e.g. `bc` for Bitcoin).
4. Convert the 5-bit groups to 8-bit bytes via `bech32::convert_bits(&data, 5, 8, false)`.
5. Return the raw witness-program bytes as `Vec<u8>`.

These bytes become `MiningConfig.mu` so the commitment hash commits to the miner's actual key, not to a string encoding.

**Reference wallet for tests:** `prl1p0jfkajyjgeu9eszntdfztcxyfjk5lh9mmvna69hd5ny83l8laqtq45rjx4`

### Pool validation (`validate_pool`)

Pool addresses use the `host:porta` format without a URL scheme (e.g. `us1.alphapool.tech:5566`).

- Returns `Err` if the string is empty.
- Returns `Err` if the string does not contain `':'` (missing port separator).
- No DNS resolution or connection attempt — actual protocol handling belongs to the future P2P module.

**Reference pool for tests:** `us1.alphapool.tech:5566`

---

## 5. Integration in `main.rs`

Args are parsed and validated before GPU detection, so the miner fails fast with a clear message if the configuration is wrong.

```rust
let args = Args::parse();
let mu = args.wallet_bytes()
    .map_err(|e| anyhow::anyhow!("{}", e))?;
args.validate_pool()
    .map_err(|e| anyhow::anyhow!("{}", e))?;

eprintln!("[init] wallet: {}", args.wallet);
eprintln!("[init] pool:   {}", args.pool);

let config = Arc::new(MiningConfig {
    params: MatrixParams { m: 32, n: 32, k: 512, r: 32, tm: 4, tn: 4 },
    difficulty_bits: 1.0,
    sigma: b"pearl-genesis-block".to_vec(),  // replaced by P2P later
    mu,
});
```

---

## 6. Error Messages

| Situation | Message |
|---|---|
| `--wallet` missing | `error: the following required arguments were not provided: --wallet <WALLET>` (clap) |
| Wrong network prefix | `error: endereço da carteira inválido: HRP deve ser 'prl', encontrado '<hrp>'` |
| Not bech32m variant | `error: endereço da carteira inválido: deve ser bech32m (não bech32 legado)` |
| Bad checksum | `error: endereço da carteira inválido: checksum bech32m incorreto` |
| Pool empty | `error: endereço da pool inválido: não pode ser vazio` |
| Pool sem porta | `error: endereço da pool inválido: formato esperado host:porta (ex: us1.alphapool.tech:5566)` |

---

## 7. Help Output

```
Usage: pearl-miner --wallet <WALLET> --pool <POOL>

Options:
      --wallet <WALLET>  Endereço Taproot da carteira Pearl (bech32m, prefixo prl1)
      --pool <POOL>      Endereço da pool (host:porta, ex: us1.alphapool.tech:5566)
  -h, --help             Print help
  -V, --version          Print version
```

---

## 8. Testing

- `wallet_bytes()` with `prl1p0jfkajyjgeu9eszntdfztcxyfjk5lh9mmvna69hd5ny83l8laqtq45rjx4` returns 32 bytes.
- `wallet_bytes()` with wrong HRP (`bc1q...`) returns an error mentioning `'prl'`.
- `wallet_bytes()` with a legacy bech32 (not bech32m) address returns an error.
- `wallet_bytes()` with a corrupted checksum returns an error.
- `validate_pool()` accepts `us1.alphapool.tech:5566`.
- `validate_pool()` rejects `""` (empty).
- `validate_pool()` rejects `"us1.alphapool.tech"` (no port separator).

---

## 9. Out of Scope

- Pool authentication (username/password/worker name)
- Config file (TOML)
- GPU selection flags (`--gpus 0,1`)
- Log level flag
- Actual pool connection/protocol
