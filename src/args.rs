use bech32::primitives::decode::CheckedHrpstring;
use bech32::Bech32m;

const PEARL_HRP: &str = "prl";

#[derive(clap::Parser, Debug)]
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
    pub fn wallet_bytes(&self) -> Result<Vec<u8>, String> {
        let checked = CheckedHrpstring::new::<Bech32m>(&self.wallet)
            .map_err(|e| format!("endereço da carteira inválido: {e}"))?;

        let hrp_val = checked.hrp();
        let hrp = hrp_val.as_str();
        if hrp != PEARL_HRP {
            return Err(format!(
                "endereço da carteira inválido: HRP deve ser '{}', encontrado '{}'",
                PEARL_HRP, hrp
            ));
        }

        Ok(checked.byte_iter().collect())
    }

    /// Validate the pool address (must be non-empty and contain ':').
    pub fn validate_pool(&self) -> Result<(), String> {
        if self.pool.is_empty() {
            return Err("endereço da pool inválido: não pode ser vazio".to_string());
        }
        if !self.pool.contains(':') {
            return Err(
                "endereço da pool inválido: formato esperado host:porta (ex: us1.alphapool.tech:5566)"
                    .to_string(),
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_WALLET: &str =
        "prl1p0jfkajyjgeu9eszntdfztcxyfjk5lh9mmvna69hd5ny83l8laqtq45rjx4";
    const VALID_POOL: &str = "us1.alphapool.tech:5566";

    fn args(wallet: &str, pool: &str) -> Args {
        Args { wallet: wallet.to_string(), pool: pool.to_string() }
    }

    #[test]
    fn test_valid_wallet_returns_bytes() {
        let bytes = args(VALID_WALLET, VALID_POOL).wallet_bytes().unwrap();
        assert_eq!(bytes.len(), 33);
    }

    #[test]
    fn test_wrong_hrp_is_rejected() {
        let btc = "bc1p5d7rjq7g6rdk2yhzks9smlaqtedr4dekq08ge8ztwac72sfr9rusxg3297";
        let err = args(btc, VALID_POOL).wallet_bytes().unwrap_err();
        assert!(err.contains("prl"), "error should mention 'prl': {err}");
    }

    #[test]
    fn test_legacy_bech32_is_rejected() {
        let legacy = "bc1qw508d6qejxtdg4y5r3zarvary0c5xw7kv8f3t4";
        let err = args(legacy, VALID_POOL).wallet_bytes().unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn test_corrupted_checksum_is_rejected() {
        let bad = "prl1p0jfkajyjgeu9eszntdfztcxyfjk5lh9mmvna69hd5ny83l8laqtq45rjx5";
        let err = args(bad, VALID_POOL).wallet_bytes().unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn test_valid_pool_is_accepted() {
        assert!(args(VALID_WALLET, VALID_POOL).validate_pool().is_ok());
    }

    #[test]
    fn test_empty_pool_is_rejected() {
        let err = args(VALID_WALLET, "").validate_pool().unwrap_err();
        assert!(!err.is_empty());
    }

    #[test]
    fn test_pool_without_port_is_rejected() {
        let err = args(VALID_WALLET, "us1.alphapool.tech").validate_pool().unwrap_err();
        assert!(err.contains(':'), "error should mention ':': {err}");
    }
}
