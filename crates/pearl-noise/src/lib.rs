fn prng(seed: &[u8; 32], tag: u8, idx: u64) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new_keyed(seed);
    hasher.update(&[tag]);
    hasher.update(&idx.to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn gen_el(m: u32, r: u32, seed: &[u8; 32]) -> Vec<i8> {
    let size = (m * r) as usize;
    let mut el = Vec::with_capacity(size);
    for idx in 0..size as u64 {
        let h = prng(seed, 0, idx);
        let val = u32::from_le_bytes([h[0], h[1], h[2], h[3]]);
        el.push((val % 64) as i8 - 32);
    }
    el
}

fn gen_er(r: u32, k: u32, seed: &[u8; 32]) -> Vec<i8> {
    let mut er = vec![0i8; (r * k) as usize];
    for col in 0..k as u64 {
        let h = prng(seed, 1, col);
        let p1 = u32::from_le_bytes([h[0], h[1], h[2], h[3]]) as usize % r as usize;
        let mut p2 = u32::from_le_bytes([h[4], h[5], h[6], h[7]]) as usize % r as usize;
        if p2 == p1 {
            p2 = (p1 + 1) % r as usize;
        }
        er[p1 * k as usize + col as usize] = 1;
        er[p2 * k as usize + col as usize] = -1;
    }
    er
}

/// Returns (EL: m×r, ER: r×k).
pub fn generate_e(m: u32, k: u32, r: u32, seed: &[u8; 32]) -> (Vec<i8>, Vec<i8>) {
    (gen_el(m, r, seed), gen_er(r, k, seed))
}

fn gen_fl(k: u32, r: u32, seed: &[u8; 32]) -> Vec<i8> {
    let mut fl = vec![0i8; (k * r) as usize];
    for row in 0..k as u64 {
        let h = prng(seed, 2, row);
        let p1 = u32::from_le_bytes([h[0], h[1], h[2], h[3]]) as usize % r as usize;
        let mut p2 = u32::from_le_bytes([h[4], h[5], h[6], h[7]]) as usize % r as usize;
        if p2 == p1 {
            p2 = (p1 + 1) % r as usize;
        }
        fl[row as usize * r as usize + p1] = 1;
        fl[row as usize * r as usize + p2] = -1;
    }
    fl
}

fn gen_fr(r: u32, n: u32, seed: &[u8; 32]) -> Vec<i8> {
    let size = (r * n) as usize;
    let mut fr = Vec::with_capacity(size);
    for idx in 0..size as u64 {
        let h = prng(seed, 3, idx);
        let val = u32::from_le_bytes([h[0], h[1], h[2], h[3]]);
        fr.push((val % 64) as i8 - 32);
    }
    fr
}

/// Returns (FL: k×r, FR: r×n).
pub fn generate_f(k: u32, n: u32, r: u32, seed: &[u8; 32]) -> (Vec<i8>, Vec<i8>) {
    (gen_fl(k, r, seed), gen_fr(r, n, seed))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: [u8; 32] = [42u8; 32];

    #[test]
    fn test_el_shape_and_range() {
        let (el, _) = generate_e(4, 64, 32, &SEED);
        assert_eq!(el.len(), 4 * 32);
        for &v in &el {
            assert!(v >= -32 && v <= 31, "EL entry {} out of [-32,31]", v);
        }
    }

    #[test]
    fn test_er_shape_and_column_structure() {
        let (r, k) = (32u32, 64u32);
        let (_, er) = generate_e(4, k, r, &SEED);
        assert_eq!(er.len(), (r * k) as usize);

        for col in 0..k as usize {
            let col_vals: Vec<i8> = (0..r as usize)
                .map(|row| er[row * k as usize + col])
                .collect();
            let ones = col_vals.iter().filter(|&&v| v == 1).count();
            let neg_ones = col_vals.iter().filter(|&&v| v == -1).count();
            let zeros = col_vals.iter().filter(|&&v| v == 0).count();
            assert_eq!(ones, 1, "col {col}: expected 1 +1, got {ones}");
            assert_eq!(neg_ones, 1, "col {col}: expected 1 -1, got {neg_ones}");
            assert_eq!(zeros, (r - 2) as usize,
                "col {col}: expected {} zeros, got {zeros}", r - 2);
        }
    }

    #[test]
    fn test_generate_e_is_deterministic() {
        let (el1, er1) = generate_e(4, 64, 32, &SEED);
        let (el2, er2) = generate_e(4, 64, 32, &SEED);
        assert_eq!(el1, el2);
        assert_eq!(er1, er2);
    }

    #[test]
    fn test_different_seeds_give_different_el() {
        let (el1, _) = generate_e(4, 64, 32, &[1u8; 32]);
        let (el2, _) = generate_e(4, 64, 32, &[2u8; 32]);
        assert_ne!(el1, el2);
    }

    #[test]
    fn test_fl_shape_and_row_structure() {
        let (k, n, r) = (64u32, 4u32, 32u32);
        let (fl, _) = generate_f(k, n, r, &SEED);
        assert_eq!(fl.len(), (k * r) as usize);

        for row in 0..k as usize {
            let row_vals: Vec<i8> = (0..r as usize)
                .map(|col| fl[row * r as usize + col])
                .collect();
            let ones = row_vals.iter().filter(|&&v| v == 1).count();
            let neg_ones = row_vals.iter().filter(|&&v| v == -1).count();
            assert_eq!(ones, 1, "row {row}: expected 1 +1");
            assert_eq!(neg_ones, 1, "row {row}: expected 1 -1");
        }
    }

    #[test]
    fn test_fr_shape_and_range() {
        let (k, n, r) = (64u32, 4u32, 32u32);
        let (_, fr) = generate_f(k, n, r, &SEED);
        assert_eq!(fr.len(), (r * n) as usize);
        for &v in &fr {
            assert!(v >= -32 && v <= 31, "FR entry {} out of [-32,31]", v);
        }
    }

    #[test]
    fn test_generate_f_is_deterministic() {
        let (fl1, fr1) = generate_f(64, 4, 32, &SEED);
        let (fl2, fr2) = generate_f(64, 4, 32, &SEED);
        assert_eq!(fl1, fl2);
        assert_eq!(fr1, fr2);
    }
}
