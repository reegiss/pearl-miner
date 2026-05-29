use pearl_types::MatrixParams;

fn matmul_i32(a: &[i32], b: &[i32], m: usize, k: usize, n: usize) -> Vec<i32> {
    let mut c = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            for l in 0..k {
                c[i * n + j] += a[i * k + l] * b[l * n + j];
            }
        }
    }
    c
}

fn cast_i32(v: &[i8]) -> Vec<i32> {
    v.iter().map(|&x| x as i32).collect()
}

/// Recovers the clean matrix product A·B from the noisy computation.
///
/// Formula: A·B = A'·B' − (A·FL)·FR − EL·(ER·B')
pub fn recover(
    ab_noisy: &[i32],
    a: &[i8],
    b_noisy: &[i8],
    el: &[i8],
    er: &[i8],
    fl: &[i8],
    fr: &[i8],
    params: &MatrixParams,
) -> Vec<i32> {
    let m = params.m as usize;
    let n = params.n as usize;
    let k = params.k as usize;
    let r = params.r as usize;

    let a_i32 = cast_i32(a);
    let fl_i32 = cast_i32(fl);
    let fr_i32 = cast_i32(fr);
    let el_i32 = cast_i32(el);
    let er_i32 = cast_i32(er);
    let bn_i32 = cast_i32(b_noisy);

    let a_fl = matmul_i32(&a_i32, &fl_i32, m, k, r);
    let term1 = matmul_i32(&a_fl, &fr_i32, m, r, n);

    let er_b = matmul_i32(&er_i32, &bn_i32, r, k, n);
    let term2 = matmul_i32(&el_i32, &er_b, m, r, n);

    ab_noisy.iter()
        .zip(term1.iter())
        .zip(term2.iter())
        .map(|((&c, &t1), &t2)| c - t1 - t2)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pearl_noise::{generate_e, generate_f};
    use pearl_types::MatrixParams;

    fn naive_matmul(a: &[i32], b: &[i32], m: usize, k: usize, n: usize) -> Vec<i32> {
        let mut c = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    c[i * n + j] += a[i * k + l] * b[l * n + j];
                }
            }
        }
        c
    }

    fn as_i32(v: &[i8]) -> Vec<i32> {
        v.iter().map(|&x| x as i32).collect()
    }

    #[test]
    fn test_recover_matches_direct_matmul() {
        let params = MatrixParams { m: 4, n: 4, k: 64, r: 32, tm: 2, tn: 2 };
        let (m, n, k, r) = (4usize, 4usize, 64usize, 32usize);

        let a: Vec<i8> = (0..m * k).map(|i| (i % 64) as i8 - 32).collect();
        let b: Vec<i8> = (0..k * n).map(|i| (i % 32) as i8 - 16).collect();

        let (el, er) = generate_e(params.m, params.k, params.r, &[1u8; 32]);
        let (fl, fr) = generate_f(params.k, params.n, params.r, &[2u8; 32]);

        let e = naive_matmul(&as_i32(&el), &as_i32(&er), m, r, k);
        let f = naive_matmul(&as_i32(&fl), &as_i32(&fr), k, r, n);

        let a_noisy: Vec<i8> = a.iter().zip(e.iter()).map(|(&a, &e)| a + e as i8).collect();
        let b_noisy: Vec<i8> = b.iter().zip(f.iter()).map(|(&b, &f)| b + f as i8).collect();

        let ab_noisy = naive_matmul(&as_i32(&a_noisy), &as_i32(&b_noisy), m, k, n);

        let recovered = recover(&ab_noisy, &a, &b_noisy, &el, &er, &fl, &fr, &params);
        let direct = naive_matmul(&as_i32(&a), &as_i32(&b), m, k, n);

        assert_eq!(recovered, direct, "recovered product does not match direct A·B");
    }
}
