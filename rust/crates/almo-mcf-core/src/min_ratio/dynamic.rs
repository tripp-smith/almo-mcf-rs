use crate::min_ratio::{CycleCandidate, MinRatioOracle, OracleQuery, TreeError};

#[derive(Debug, Clone)]
pub struct TreeChainLevel {
    pub oracle: MinRatioOracle,
    pub instability_budget: usize,
    pub max_instability: usize,
    pub last_rebuild: usize,
    pub approx_factor: f64,
    pub needs_rebuild: bool,
}

impl TreeChainLevel {
    pub fn new(
        seed: u64,
        rebuild_every: usize,
        max_instability: usize,
        approx_factor: f64,
    ) -> Self {
        Self {
            oracle: MinRatioOracle::new(seed, rebuild_every),
            instability_budget: 0,
            max_instability,
            last_rebuild: 0,
            approx_factor,
            needs_rebuild: false,
        }
    }

    pub fn record_instability(&mut self, amount: usize) {
        self.instability_budget = self.instability_budget.saturating_add(amount);
        if self.instability_budget >= self.max_instability {
            self.instability_budget = 0;
            self.needs_rebuild = true;
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
        Self::new_with_approx(seed, levels, rebuild_every, max_instability, 0.0)
    }

    pub fn new_with_approx(
        seed: u64,
        levels: usize,
        rebuild_every: usize,
        max_instability: usize,
        approx_factor: f64,
    ) -> Self {
        let mut chain = Vec::with_capacity(levels);
        for level in 0..levels {
            chain.push(TreeChainLevel::new(
                seed ^ (0x9e37_79b9_7f4a_7c15u64.wrapping_mul(level as u64 + 1)),
                rebuild_every,
                max_instability,
                approx_factor,
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
        let mut best_score: Option<f64> = None;
        let query = OracleQuery {
            iter,
            node_count,
            tails,
            heads,
            gradients,
            lengths,
        };
        for level in &mut self.levels {
            let candidate = level
                .oracle
                .best_cycle_with_rebuild(query, level.needs_rebuild)?;
            if level.needs_rebuild {
                level.needs_rebuild = false;
                level.last_rebuild = iter;
            }
            if let Some(candidate) = candidate {
                let score = candidate.ratio / (1.0 + level.approx_factor);
                if best_score.map(|best| score < best).unwrap_or(true) {
                    best = Some(candidate);
                    best_score = Some(score);
                }
            }
        }
        Ok(best)
    }

    pub fn record_instability(&mut self, amount: usize) {
        for level in &mut self.levels {
            level.record_instability(amount);
        }
    }
}

#[derive(Debug, Clone)]
pub struct FullDynamicOracle {
    hierarchy: TreeChainHierarchy,
    spanner: crate::spanner::DynamicSpanner,
    last_gradients: Vec<f64>,
    last_lengths: Vec<f64>,
    last_edge_count: usize,
    node_count: usize,
    gradient_change: f64,
    length_factor: f64,
}

impl FullDynamicOracle {
    pub fn new(
        seed: u64,
        levels: usize,
        rebuild_every: usize,
        max_instability: usize,
        approx_factor: f64,
    ) -> Self {
        Self {
            hierarchy: TreeChainHierarchy::new_with_approx(
                seed,
                levels,
                rebuild_every,
                max_instability,
                approx_factor,
            ),
            spanner: crate::spanner::DynamicSpanner::new(0),
            last_gradients: Vec::new(),
            last_lengths: Vec::new(),
            last_edge_count: 0,
            node_count: 0,
            gradient_change: 0.5,
            length_factor: 1.25,
        }
    }

    fn rebuild_spanner(&mut self, node_count: usize, tails: &[u32], heads: &[u32]) {
        let mut spanner = crate::spanner::DynamicSpanner::new(node_count);
        for (edge_id, (&tail, &head)) in tails.iter().zip(heads.iter()).enumerate() {
            let spanner_edge = spanner.insert_edge(tail as usize, head as usize);
            spanner.set_embedding(
                edge_id,
                vec![crate::spanner::EmbeddingStep::new(spanner_edge, 1)],
            );
        }
        self.spanner = spanner;
        self.last_edge_count = tails.len();
        self.node_count = node_count;
    }

    fn ensure_spanner(&mut self, node_count: usize, tails: &[u32], heads: &[u32]) {
        if self.node_count != node_count || self.last_edge_count != tails.len() {
            self.rebuild_spanner(node_count, tails, heads);
            return;
        }
        if self.spanner.node_count != node_count {
            self.rebuild_spanner(node_count, tails, heads);
        }
    }

    fn record_updates(&mut self, gradients: &[f64], lengths: &[f64]) {
        if self.last_gradients.is_empty() || self.last_lengths.is_empty() {
            self.last_gradients = gradients.to_vec();
            self.last_lengths = lengths.to_vec();
            return;
        }
        let mut instability = 0_usize;
        for ((&prev_g, &prev_l), (&g, &l)) in self
            .last_gradients
            .iter()
            .zip(self.last_lengths.iter())
            .zip(gradients.iter().zip(lengths.iter()))
        {
            if (g - prev_g).abs() > self.gradient_change {
                instability += 1;
            }
            if prev_l > 0.0 && l > 0.0 {
                let ratio = if l > prev_l { l / prev_l } else { prev_l / l };
                if ratio > self.length_factor {
                    instability += 1;
                }
            }
        }
        if instability > 0 {
            self.hierarchy.record_instability(instability);
        }
        self.last_gradients = gradients.to_vec();
        self.last_lengths = lengths.to_vec();
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
        self.ensure_spanner(node_count, tails, heads);
        self.record_updates(gradients, lengths);
        let candidate = self
            .hierarchy
            .best_cycle(iter, node_count, tails, heads, gradients, lengths)?;
        if let Some(candidate) = candidate.as_ref() {
            let mut valid = true;
            for (edge_id, _) in &candidate.cycle_edges {
                if !self.spanner.embedding_valid(*edge_id) {
                    valid = false;
                    break;
                }
            }
            if !valid {
                self.rebuild_spanner(node_count, tails, heads);
            }
        }
        Ok(candidate)
    }

    pub fn edge_embedding(&self, edge_id: usize) -> Option<Vec<(usize, i8)>> {
        self.spanner
            .embedding_steps(edge_id)
            .map(|steps| steps.iter().map(|step| (step.edge, step.dir)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::min_ratio::{MinRatioOracle, OracleQuery};
    use crate::spanner::DynamicSpanner;

    #[test]
    fn hierarchy_matches_fallback_cycle() {
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let gradients = vec![1.0, -2.0, 0.5, -3.0];
        let lengths = vec![1.0, 2.0, 1.0, 3.0];
        let mut fallback = MinRatioOracle::new(13, 1);
        let mut hierarchy = TreeChainHierarchy::new(13, 2, 1, 5);
        let fallback_best = fallback
            .best_cycle(OracleQuery {
                iter: 0,
                node_count: 3,
                tails: &tails,
                heads: &heads,
                gradients: &gradients,
                lengths: &lengths,
            })
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
        hierarchy.record_instability(1);
        assert_eq!(hierarchy.levels[0].last_rebuild, 0);
        hierarchy.record_instability(1);
        assert!(hierarchy.levels[0].needs_rebuild);
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let gradients = vec![1.0, -2.0, 0.5, -3.0];
        let lengths = vec![1.0, 2.0, 1.0, 3.0];
        hierarchy
            .best_cycle(2, 3, &tails, &heads, &gradients, &lengths)
            .unwrap();
        assert_eq!(hierarchy.levels[0].last_rebuild, 2);
    }

    #[test]
    fn dynamic_oracle_matches_fallback_on_small_graph() {
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let gradients = vec![1.0, -2.0, 0.5, -3.0];
        let lengths = vec![1.0, 2.0, 1.0, 3.0];
        let mut fallback = MinRatioOracle::new(17, 1);
        let mut dynamic = FullDynamicOracle::new(17, 2, 1, 5, 0.0);
        let fallback_best = fallback
            .best_cycle(OracleQuery {
                iter: 0,
                node_count: 3,
                tails: &tails,
                heads: &heads,
                gradients: &gradients,
                lengths: &lengths,
            })
            .unwrap()
            .unwrap();
        let dynamic_best = dynamic
            .best_cycle(0, 3, &tails, &heads, &gradients, &lengths)
            .unwrap()
            .unwrap();
        assert!((fallback_best.ratio - dynamic_best.ratio).abs() < 1e-9);
        let mut fallback_edges = fallback_best.cycle_edges.clone();
        let mut dynamic_edges = dynamic_best.cycle_edges.clone();
        fallback_edges.sort_unstable();
        dynamic_edges.sort_unstable();
        assert_eq!(fallback_edges, dynamic_edges);
    }

    #[test]
    fn dynamic_oracle_tracks_updates_and_stays_stable() {
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let gradients = vec![1.0, -2.0, 0.5, -3.0];
        let lengths = vec![1.0, 2.0, 1.0, 3.0];
        let mut dynamic = FullDynamicOracle::new(31, 3, 2, 2, 0.1);
        let first = dynamic
            .best_cycle(0, 3, &tails, &heads, &gradients, &lengths)
            .unwrap()
            .unwrap();
        let updated_gradients = vec![1.6, -2.4, 0.4, -2.8];
        let updated_lengths = vec![1.4, 2.8, 1.1, 2.9];
        let second = dynamic
            .best_cycle(1, 3, &tails, &heads, &updated_gradients, &updated_lengths)
            .unwrap()
            .unwrap();
        assert!(!first.cycle_edges.is_empty());
        assert!(!second.cycle_edges.is_empty());
    }

    #[test]
    fn dynamic_oracle_rebuilds_invalid_embeddings() {
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let gradients = vec![1.0, -2.0, 0.5, -3.0];
        let lengths = vec![1.0, 2.0, 1.0, 3.0];
        let mut dynamic = FullDynamicOracle::new(41, 2, 1, 5, 0.0);
        dynamic
            .best_cycle(0, 3, &tails, &heads, &gradients, &lengths)
            .unwrap();
        let mut spanner = DynamicSpanner::new(3);
        let edge = spanner.insert_edge(0, 1);
        spanner.set_embedding(0, vec![crate::spanner::EmbeddingStep::new(edge, 1)]);
        dynamic.spanner = spanner;
        assert!(!dynamic.spanner.embedding_valid(2));
        dynamic
            .best_cycle(1, 3, &tails, &heads, &gradients, &lengths)
            .unwrap();
        assert!(dynamic.spanner.embedding_valid(2));
    }

    #[test]
    fn dynamic_oracle_exposes_edge_embeddings() {
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let gradients = vec![1.0, -2.0, 0.5, -3.0];
        let lengths = vec![1.0, 2.0, 1.0, 3.0];
        let mut dynamic = FullDynamicOracle::new(51, 2, 1, 5, 0.0);
        dynamic
            .best_cycle(0, 3, &tails, &heads, &gradients, &lengths)
            .unwrap();
        let embedding = dynamic.edge_embedding(1).expect("embedding should exist");
        assert_eq!(embedding, vec![(1, 1)]);
    }
}
