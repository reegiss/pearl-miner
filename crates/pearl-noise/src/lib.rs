use rayon::prelude::*;

/// Noise for A: EL (dense m×r) + ER encoded as sparse column pairs.
/// Each column j of ER has one +1 at row `pos[j]` and one -1 at row `neg[j]`.
pub struct ENoise {
    pub el:   Vec<i8>,           // m × r, row-major
    pub er_cols: Vec<(u16, u16)>,// k entries: (pos_row, neg_row) per column of ER
}

/// Noise for B: FL encoded as sparse row pairs + FR (dense r×k).
/// Each row i of FL has one +1 at col `pos[i]` and one -1 at col `neg[i]`.
pub struct FNoise {
    pub fl_rows: Vec<(u16, u16)>,// n entries: (pos_col, neg_col) per row of FL
    pub fr:   Vec<i8>,           // r × k, row-major
}

/// Generate E noise from sA.
pub fn generate_e(m: usize, k: usize, r: usize, seed: &[u8; 32]) -> ENoise {
    let el      = gen_dense(m, r, seed, 0);
    let er_cols = gen_sparse_cols(k, r, seed, 1);
    ENoise { el, er_cols }
}

/// Generate F noise from sB.
pub fn generate_f(n: usize, k: usize, r: usize, seed: &[u8; 32]) -> FNoise {
    let fl_rows = gen_sparse_rows(n, r, seed, 0);
    let fr      = gen_dense(r, k, seed, 1);
    FNoise { fl_rows, fr }
}

/// A' = A + EL·ER  in O(m × k × 2) — exploits ER sparsity.
pub fn apply_e(a: &[i8], noise: &ENoise, m: usize, k: usize, r: usize) -> Vec<i8> {
    let mut out = a.to_vec();
    out.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
        let el_row = &noise.el[i * r..(i + 1) * r];
        for (j, &(pos, neg)) in noise.er_cols.iter().enumerate() {
            let e = el_row[pos as usize] as i16 - el_row[neg as usize] as i16;
            let v = row[j] as i16 + e;
            row[j] = v.clamp(-127, 127) as i8;
        }
    });
    out
}

/// B' = B + FL·FR  in O(n × k × 2) — exploits FL sparsity.
pub fn apply_f(b: &[i8], noise: &FNoise, n: usize, k: usize) -> Vec<i8> {
    let mut out = b.to_vec();
    out.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
        let (pos, neg) = noise.fl_rows[i];
        let fr_pos = &noise.fr[pos as usize * k..(pos as usize + 1) * k];
        let fr_neg = &noise.fr[neg as usize * k..(neg as usize + 1) * k];
        for j in 0..k {
            let f = fr_pos[j] as i16 - fr_neg[j] as i16;
            let v = row[j] as i16 + f;
            row[j] = v.clamp(-127, 127) as i8;
        }
    });
    out
}

// --- internal generators ---

fn gen_dense(rows: usize, cols: usize, seed: &[u8; 32], domain: u8) -> Vec<i8> {
    let mut out = vec![0i8; rows * cols];
    out.par_chunks_mut(cols).enumerate().for_each(|(i, row_slice)| {
        for j in 0..cols {
            row_slice[j] = prng_i6(seed, domain, i as u64, j as u64);
        }
    });
    out
}

fn gen_sparse_cols(k: usize, r: usize, seed: &[u8; 32], domain: u8) -> Vec<(u16, u16)> {
    (0..k).into_par_iter().map(|j| {
        let pos = prng_range(seed, domain, j as u64, 0, r) as u16;
        let mut neg = prng_range(seed, domain, j as u64, 1, r) as u16;
        if neg == pos { neg = (neg + 1) % r as u16; }
        (pos, neg)
    }).collect()
}

fn gen_sparse_rows(n: usize, r: usize, seed: &[u8; 32], domain: u8) -> Vec<(u16, u16)> {
    (0..n).into_par_iter().map(|i| {
        let pos = prng_range(seed, domain, i as u64, 0, r) as u16;
        let mut neg = prng_range(seed, domain, i as u64, 1, r) as u16;
        if neg == pos { neg = (neg + 1) % r as u16; }
        (pos, neg)
    }).collect()
}

fn prng_i6(seed: &[u8; 32], domain: u8, row: u64, col: u64) -> i8 {
    let raw = prng_u64(seed, domain, row, col) as u8;
    ((raw & 0x3F) as i8) - 32
}

fn prng_range(seed: &[u8; 32], domain: u8, idx: u64, sub: u8, range: usize) -> usize {
    (prng_u64(seed, domain, idx, sub as u64) as usize) % range
}

fn prng_u64(seed: &[u8; 32], domain: u8, a: u64, b: u64) -> u64 {
    let mut msg = [0u8; 18];
    msg[0] = domain;
    msg[1..9].copy_from_slice(&a.to_le_bytes());
    msg[9..17].copy_from_slice(&b.to_le_bytes());
    let h = blake3::keyed_hash(seed, &msg);
    u64::from_le_bytes(h.as_bytes()[..8].try_into().unwrap())
}
