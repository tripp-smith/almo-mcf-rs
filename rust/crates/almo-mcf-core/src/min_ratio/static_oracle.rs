use crate::min_ratio::{best_cycle_over_edges, CycleCandidate};
use crate::trees::{LowStretchTree, TreeError};

#[derive(Debug, Clone)]
pub struct StaticOracle {
    pub seed: u64,
}

impl StaticOracle {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    pub fn best_cycle(
        &self,
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        gradients: &[f64],
        lengths: &[f64],
    ) -> Result<Option<CycleCandidate>, TreeError> {
        let tree = LowStretchTree::build(node_count, tails, heads, lengths, self.seed)?;
        Ok(best_cycle_over_edges(
            &tree, tails, heads, gradients, lengths,
        ))
    }

    pub fn approximate_best_cycle(
        &self,
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        gradients: &[f64],
        lengths: &[f64],
        sample_stride: usize,
    ) -> Result<Option<CycleCandidate>, TreeError> {
        let tree = LowStretchTree::build(node_count, tails, heads, lengths, self.seed)?;
        let stride = sample_stride.max(1);
        let mut best: Option<CycleCandidate> = None;
        for edge_id in (0..tails.len()).step_by(stride) {
            let Some(candidate) =
                super::score_edge_cycle(&tree, edge_id, tails, heads, gradients, lengths)
            else {
                continue;
            };
            best = Some(match best {
                Some(current) => super::select_better_candidate(current, candidate),
                None => candidate,
            });
        }
        Ok(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_oracle_finds_cycle() {
        let tails = vec![0, 1, 2];
        let heads = vec![1, 2, 0];
        let gradients = vec![-1.0, -2.0, -3.0];
        let lengths = vec![1.0, 1.0, 1.0];
        let oracle = StaticOracle::new(7);
        let best = oracle
            .best_cycle(3, &tails, &heads, &gradients, &lengths)
            .unwrap()
            .unwrap();
        assert!(best.ratio < 0.0);
    }
}
