pub struct MerkleTree {
    nodes: Vec<[u8; 32]>,
    padded_size: usize,
}

fn hash_leaf(data: &[u8]) -> [u8; 32] {
    *blake3::hash(data).as_bytes()
}

fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut input = [0u8; 64];
    input[..32].copy_from_slice(left);
    input[32..].copy_from_slice(right);
    *blake3::hash(&input).as_bytes()
}

impl MerkleTree {
    pub fn from_rows(data: &[i8], row_len: usize) -> Self {
        let row_count = if row_len == 0 { 0 } else { data.len() / row_len };
        let padded_size = row_count.next_power_of_two().max(1);
        let total = 2 * padded_size;
        let mut nodes = vec![[0u8; 32]; total];

        for i in 0..row_count {
            let bytes: Vec<u8> = data[i * row_len..(i + 1) * row_len]
                .iter().map(|&x| x as u8).collect();
            nodes[padded_size + i] = hash_leaf(&bytes);
        }

        for i in (1..padded_size).rev() {
            nodes[i] = hash_pair(&nodes[2 * i], &nodes[2 * i + 1]);
        }

        MerkleTree { nodes, padded_size }
    }

    pub fn root(&self) -> [u8; 32] {
        self.nodes[1]
    }

    pub fn proof(&self, idx: usize) -> Vec<[u8; 32]> {
        let mut path = Vec::new();
        let mut i = self.padded_size + idx;
        while i > 1 {
            path.push(self.nodes[i ^ 1]);
            i /= 2;
        }
        path
    }

    pub fn verify_proof(&self, idx: usize, leaf_bytes: &[u8], proof: &[[u8; 32]]) -> bool {
        let mut current = hash_leaf(leaf_bytes);
        let mut i = self.padded_size + idx;
        for sibling in proof {
            let (left, right) = if i % 2 == 0 {
                (&current, sibling)
            } else {
                (sibling, &current)
            };
            current = hash_pair(left, right);
            i /= 2;
        }
        current == self.root()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row_bytes(row: &[i8]) -> Vec<u8> {
        row.iter().map(|&x| x as u8).collect()
    }

    #[test]
    fn test_root_is_deterministic() {
        let data = vec![1i8; 4 * 8];
        let t1 = MerkleTree::from_rows(&data, 8);
        let t2 = MerkleTree::from_rows(&data, 8);
        assert_eq!(t1.root(), t2.root());
    }

    #[test]
    fn test_root_changes_with_data() {
        let t1 = MerkleTree::from_rows(&vec![1i8; 4 * 8], 8);
        let t2 = MerkleTree::from_rows(&vec![2i8; 4 * 8], 8);
        assert_ne!(t1.root(), t2.root());
    }

    #[test]
    fn test_proof_verifies_row0() {
        let data: Vec<i8> = (0..16).map(|i| i as i8).collect();
        let tree = MerkleTree::from_rows(&data, 8);
        let proof = tree.proof(0);
        let leaf = row_bytes(&data[0..8]);
        assert!(tree.verify_proof(0, &leaf, &proof));
    }

    #[test]
    fn test_proof_verifies_row1() {
        let data: Vec<i8> = (0..16).map(|i| i as i8).collect();
        let tree = MerkleTree::from_rows(&data, 8);
        let proof = tree.proof(1);
        let leaf = row_bytes(&data[8..16]);
        assert!(tree.verify_proof(1, &leaf, &proof));
    }

    #[test]
    fn test_wrong_leaf_fails_verification() {
        let data: Vec<i8> = (0..16).map(|i| i as i8).collect();
        let tree = MerkleTree::from_rows(&data, 8);
        let proof = tree.proof(0);
        let wrong_leaf = vec![0xffu8; 8];
        assert!(!tree.verify_proof(0, &wrong_leaf, &proof));
    }

    #[test]
    fn test_non_power_of_two_row_count() {
        let data: Vec<i8> = (0..24).map(|i| i as i8).collect();
        let tree = MerkleTree::from_rows(&data, 8);
        for idx in 0..3 {
            let proof = tree.proof(idx);
            let leaf = row_bytes(&data[idx * 8..(idx + 1) * 8]);
            assert!(tree.verify_proof(idx, &leaf, &proof), "row {} failed", idx);
        }
    }
}
