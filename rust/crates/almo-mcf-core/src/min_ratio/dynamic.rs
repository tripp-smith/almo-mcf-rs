use crate::min_ratio::{CycleCandidate, MinRatioOracle, TreeError};

#[derive(Debug, Clone)]
pub struct TreeChainLevel {
    pub oracle: MinRatioOracle,
    pub instability_budget: usize,
    pub max_instability: usize,
    pub last_rebuild: usize,
}

impl TreeChainLevel {
    pub fn new(seed: u64, rebuild_every: usize, max_instability: usize) -> Self {
        Self {
            oracle: MinRatioOracle::new(seed, rebuild_every),
            instability_budget: 0,
            max_instability,
            last_rebuild: 0,
        }
    }

    pub fn record_instability(&mut self, iter: usize, amount: usize) {
        self.instability_budget = self.instability_budget.saturating_add(amount);
        if self.instability_budget >= self.max_instability {
            self.instability_budget = 0;
            self.last_rebuild = iter;
        }
    }
}

#[derive(Debug, Clone)]
pub struct TreeChainHierarchy {
    pub seed: u64,
    pub levels: Vec<TreeChainLevel>,
    pub rebuild_every: usize,
}

impl TreeChainHierarchy {
    pub fn new(seed: u64, levels: usize, rebuild_every: usize, max_instability: usize) -> Self {
        let mut chain = Vec::with_capacity(levels);
        for level in 0..levels {
            chain.push(TreeChainLevel::new(
                seed ^ (0x9e37_79b9_7f4a_7c15u64.wrapping_mul(level as u64 + 1)),
                rebuild_every,
                max_instability,
            ));
        }
        Self {
            seed,
            levels: chain,
            rebuild_every,
        }
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
        let mut best: Option<CycleCandidate> = None;
        for level in &mut self.levels {
            let candidate = level
                .oracle
                .best_cycle(iter, node_count, tails, heads, gradients, lengths)?;
            if let Some(candidate) = candidate {
                if best
                    .as_ref()
                    .map(|best| candidate.ratio < best.ratio)
                    .unwrap_or(true)
                {
                    best = Some(candidate);
                }
            }
        }
        Ok(best)
    }

    pub fn record_instability(&mut self, iter: usize, amount: usize) {
        for level in &mut self.levels {
            level.record_instability(iter, amount);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::min_ratio::MinRatioOracle;

    #[test]
    fn hierarchy_matches_fallback_cycle() {
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let gradients = vec![1.0, -2.0, 0.5, -3.0];
        let lengths = vec![1.0, 2.0, 1.0, 3.0];
        let mut fallback = MinRatioOracle::new(13, 1);
        let mut hierarchy = TreeChainHierarchy::new(13, 2, 1, 5);
        let fallback_best = fallback
            .best_cycle(0, 3, &tails, &heads, &gradients, &lengths)
            .unwrap()
            .unwrap();
        let hierarchy_best = hierarchy
            .best_cycle(0, 3, &tails, &heads, &gradients, &lengths)
            .unwrap()
            .unwrap();
        assert!((fallback_best.ratio - hierarchy_best.ratio).abs() < 1e-9);
        let mut fallback_edges = fallback_best.cycle_edges.clone();
        let mut hierarchy_edges = hierarchy_best.cycle_edges.clone();
        fallback_edges.sort_unstable();
        hierarchy_edges.sort_unstable();
        assert_eq!(fallback_edges, hierarchy_edges);
    }

    #[test]
    fn hierarchy_is_deterministic_under_seed() {
        let tails = vec![0, 1, 2, 0, 1];
        let heads = vec![1, 2, 0, 2, 3];
        let gradients = vec![0.5, -1.5, -2.0, 1.0, 0.25];
        let lengths = vec![1.0, 1.0, 2.0, 2.0, 3.0];
        let mut h1 = TreeChainHierarchy::new(21, 3, 2, 4);
        let mut h2 = TreeChainHierarchy::new(21, 3, 2, 4);
        let best1 = h1
            .best_cycle(0, 4, &tails, &heads, &gradients, &lengths)
            .unwrap()
            .unwrap();
        let best2 = h2
            .best_cycle(0, 4, &tails, &heads, &gradients, &lengths)
            .unwrap()
            .unwrap();
        assert_eq!(best1.cycle_edges, best2.cycle_edges);
        assert!((best1.ratio - best2.ratio).abs() < 1e-12);
    }

    #[test]
    fn instability_budget_triggers_rebuild_tracking() {
        let mut hierarchy = TreeChainHierarchy::new(9, 2, 3, 2);
        hierarchy.record_instability(1, 1);
        assert_eq!(hierarchy.levels[0].last_rebuild, 0);
        hierarchy.record_instability(2, 1);
        assert_eq!(hierarchy.levels[0].last_rebuild, 2);
    }
}
