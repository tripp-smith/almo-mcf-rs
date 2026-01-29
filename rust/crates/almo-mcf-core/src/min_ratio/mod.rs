use crate::trees::{LowStretchTree, TreeError};

#[derive(Debug, Clone)]
pub struct CycleCandidate {
    pub ratio: f64,
    pub numerator: f64,
    pub denominator: f64,
    pub cycle_edges: Vec<(usize, i8)>,
}

#[derive(Debug, Clone)]
pub struct MinRatioOracle {
    pub seed: u64,
    pub rebuild_every: usize,
    pub last_rebuild: usize,
    pub tree: Option<LowStretchTree>,
}

impl MinRatioOracle {
    pub fn new(seed: u64, rebuild_every: usize) -> Self {
        Self {
            seed,
            rebuild_every,
            last_rebuild: 0,
            tree: None,
        }
    }

    pub fn rebuild_tree(
        &mut self,
        iter: usize,
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        lengths: &[f64],
    ) -> Result<(), TreeError> {
        self.tree = Some(LowStretchTree::build(
            node_count,
            tails,
            heads,
            lengths,
            self.seed ^ iter as u64,
        )?);
        self.last_rebuild = iter;
        Ok(())
    }

    pub fn best_cycle(
        &mut self,
        iter: usize,
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        gradients: &[f64],
        lengths: &[f64],
    ) -> Result<Option<CycleCandidate>, TreeError> {
        if self.tree.is_none() || iter == 0 || (iter - self.last_rebuild) >= self.rebuild_every {
            self.rebuild_tree(iter, node_count, tails, heads, lengths)?;
        }
        let tree = self.tree.as_ref().expect("tree should exist after rebuild");

        let mut best: Option<CycleCandidate> = None;
        for edge_id in 0..tails.len() {
            if tree.tree_edges[edge_id] {
                continue;
            }
            let tail = tails[edge_id] as usize;
            let head = heads[edge_id] as usize;
            let Some(path_edges) = tree.path_edges(head, tail, tails, heads) else {
                continue;
            };
            let mut numerator = gradients[edge_id];
            let mut denominator = lengths[edge_id].abs();
            let mut cycle_edges = Vec::with_capacity(path_edges.len() + 1);
            cycle_edges.push((edge_id, 1));

            for (path_edge, dir) in path_edges {
                let grad = gradients[path_edge];
                numerator += (dir as f64) * grad;
                denominator += lengths[path_edge].abs();
                cycle_edges.push((path_edge, dir));
            }

            if denominator <= 0.0 {
                continue;
            }
            let ratio = numerator / denominator;
            let candidate = CycleCandidate {
                ratio,
                numerator,
                denominator,
                cycle_edges,
            };
            if best.as_ref().map(|best| ratio < best.ratio).unwrap_or(true) {
                best = Some(candidate);
            }
        }

        Ok(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct TestRng(u64);

    impl TestRng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.0
        }

        fn next_usize(&mut self, max: usize) -> usize {
            (self.next_u64() as usize) % max
        }
    }

    #[test]
    fn cycle_scoring_matches_hand_graph() {
        let tails = vec![0, 1, 0];
        let heads = vec![1, 2, 2];
        let gradients = vec![1.0, -2.0, 3.0];
        let lengths = vec![2.0, 1.0, 4.0];
        let mut oracle = MinRatioOracle::new(5, 1);
        let best = oracle
            .best_cycle(0, 3, &tails, &heads, &gradients, &lengths)
            .unwrap()
            .unwrap();
        assert!(best.denominator > 0.0);
        let mut expected_numerator = 0.0;
        let mut expected_denominator = 0.0;
        for (edge_id, dir) in &best.cycle_edges {
            expected_numerator += (*dir as f64) * gradients[*edge_id];
            expected_denominator += lengths[*edge_id].abs();
        }
        assert!((best.numerator - expected_numerator).abs() < 1e-9);
        assert!((best.denominator - expected_denominator).abs() < 1e-9);
    }

    #[test]
    fn returns_circulation_edges() {
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let gradients = vec![1.0, 1.0, 1.0, -4.0];
        let lengths = vec![1.0, 1.0, 1.0, 2.0];
        let mut oracle = MinRatioOracle::new(11, 2);
        let best = oracle
            .best_cycle(0, 3, &tails, &heads, &gradients, &lengths)
            .unwrap()
            .unwrap();
        let mut incidence = [0_i32; 3];
        for (edge_id, dir) in best.cycle_edges {
            let tail = tails[edge_id] as usize;
            let head = heads[edge_id] as usize;
            let sign = dir as i32;
            incidence[tail] -= sign;
            incidence[head] += sign;
        }
        assert!(incidence.iter().all(|&v| v == 0));
    }

    #[test]
    fn random_cycles_form_circulations() {
        let mut rng = TestRng::new(123);
        for _ in 0..20 {
            let node_count = 6;
            let mut tails = Vec::new();
            let mut heads = Vec::new();
            let mut gradients = Vec::new();
            let mut lengths = Vec::new();
            for _ in 0..12 {
                let tail = rng.next_usize(node_count);
                let mut head = rng.next_usize(node_count);
                if head == tail {
                    head = (head + 1) % node_count;
                }
                tails.push(tail as u32);
                heads.push(head as u32);
                gradients.push((rng.next_u64() % 11) as f64 - 5.0);
                lengths.push((rng.next_u64() % 5) as f64 + 1.0);
            }
            let mut oracle = MinRatioOracle::new(rng.next_u64(), 1);
            let Some(best) = oracle
                .best_cycle(0, node_count, &tails, &heads, &gradients, &lengths)
                .unwrap()
            else {
                continue;
            };
            let mut incidence = vec![0_i32; node_count];
            for (edge_id, dir) in best.cycle_edges {
                let tail = tails[edge_id] as usize;
                let head = heads[edge_id] as usize;
                let sign = dir as i32;
                incidence[tail] -= sign;
                incidence[head] += sign;
            }
            assert!(incidence.iter().all(|&v| v == 0));
        }
    }
}
