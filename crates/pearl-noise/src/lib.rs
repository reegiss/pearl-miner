/// Generate EL (m×r) and ER (r×k) from seed sA.
///
/// EL: uniform INT8 in [-32, 31] — 6-bit signed values
/// ER: each column has exactly one +1 and one -1 at distinct random rows
pub fn generate_e(m: usize, k: usize, r: usize, seed: &[u8; 32]) -> (Vec<i8>, Vec<i8>) {
    let el = generate_el(m, r, seed, 0);
    let er = generate_er(r, k, seed, 1);
    (el, er)
}

/// Generate FL (n×r) and FR (r×k) from seed sB.
///
/// FL has ER^T distribution: each row has one +1 and one -1 at distinct random cols.
/// FR has EL^T distribution: uniform INT8 in [-32, 31].
pub fn generate_f(n: usize, k: usize, r: usize, seed: &[u8; 32]) -> (Vec<i8>, Vec<i8>) {
    // FL (n×r): distribution of ER^T — each row has one +1 and one -1
    let fl = generate_fl(n, r, seed, 0);
    // FR (r×k): distribution of EL^T — uniform in [-32, 31] per spec
    let fr = generate_el(r, k, seed, 1); // same distribution as EL
    (fl, fr)
}

/// Apply noise: out = matrix + EL·ER  (in-place, result in out).
///
/// matrix: m×k row-major i8
/// el:     m×r row-major i8
/// er:     r×k row-major i8  (sparse: one +1, one -1 per column)
/// Returns A' = A + EL·ER as m×k i8, clamped to [-127, 127].
pub fn apply_noise(matrix: &[i8], el: &[i8], er: &[i8], rows: usize, k: usize, r: usize) -> Vec<i8> {
    let mut out = matrix.to_vec();
    // For each output element (i, j): out[i*k+j] += sum_s(EL[i,s] * ER[s,j])
    for i in 0..rows {
        for s in 0..r {
            let el_val = el[i * r + s] as i32;
            if el_val == 0 { continue; }
            for j in 0..k {
                let er_val = er[s * k + j] as i32;
                let idx = i * k + j;
                out[idx] = (out[idx] as i32 + el_val * er_val).clamp(-127, 127) as i8;
            }
        }
    }
    out
}

// --- internal generators ---

fn generate_el(rows: usize, cols: usize, seed: &[u8; 32], domain: u8) -> Vec<i8> {
    let mut out = vec![0i8; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            let val = prng_i6(seed, domain, i as u64, j as u64);
            out[i * cols + j] = val;
        }
    }
    out
}

fn generate_er(r: usize, k: usize, seed: &[u8; 32], domain: u8) -> Vec<i8> {
    let mut out = vec![0i8; r * k];
    for j in 0..k {
        // Pick two distinct rows for +1 and -1
        let row_pos = prng_range(seed, domain, j as u64, 0, r);
        let mut row_neg = prng_range(seed, domain, j as u64, 1, r);
        if row_neg == row_pos {
            row_neg = (row_neg + 1) % r;
        }
        out[row_pos * k + j] = 1;
        out[row_neg * k + j] = -1;
    }
    out
}

fn generate_fl(n: usize, r: usize, seed: &[u8; 32], domain: u8) -> Vec<i8> {
    // FL (n×r): each row has one +1 and one -1 — this is the ER^T distribution
    let mut out = vec![0i8; n * r];
    for i in 0..n {
        let col_pos = prng_range(seed, domain, i as u64, 0, r);
        let mut col_neg = prng_range(seed, domain, i as u64, 1, r);
        if col_neg == col_pos {
            col_neg = (col_neg + 1) % r;
        }
        out[i * r + col_pos] = 1;
        out[i * r + col_neg] = -1;
    }
    out
}

/// BLAKE3-based PRNG: returns uniform i8 in [-32, 31] (6-bit signed).
fn prng_i6(seed: &[u8; 32], domain: u8, row: u64, col: u64) -> i8 {
    let raw = prng_byte(seed, domain, row, col);
    // Map 0..63 → -32..31
    ((raw & 0x3F) as i8) - 32
}

/// BLAKE3-based PRNG: returns uniform value in 0..range.
fn prng_range(seed: &[u8; 32], domain: u8, idx: u64, sub: u8, range: usize) -> usize {
    let raw = prng_u64(seed, domain, idx, sub as u64);
    (raw as usize) % range
}

fn prng_byte(seed: &[u8; 32], domain: u8, a: u64, b: u64) -> u8 {
    prng_u64(seed, domain, a, b) as u8
}

fn prng_u64(seed: &[u8; 32], domain: u8, a: u64, b: u64) -> u64 {
    let mut msg = [0u8; 18];
    msg[0] = domain;
    msg[1..9].copy_from_slice(&a.to_le_bytes());
    msg[9..17].copy_from_slice(&b.to_le_bytes());
    let h = blake3::keyed_hash(seed, &msg);
    u64::from_le_bytes(h.as_bytes()[..8].try_into().unwrap())
}
