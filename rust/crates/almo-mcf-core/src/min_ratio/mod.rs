use crate::spanner::{DynamicSpanner, EdgeValues};
use crate::trees::{LowStretchTree, TreeBuildMode, TreeError};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
pub mod branching_tree_chain;
pub mod dynamic;
pub mod hsfc;
pub mod oracle;
pub mod static_oracle;

pub use oracle::{DynamicUpdateOracle, OracleError, SparseFlowDelta};

#[derive(Debug, Clone, Copy)]
pub struct OracleQuery<'a> {
    pub iter: usize,
    pub node_count: usize,
    pub tails: &'a [u32],
    pub heads: &'a [u32],
    pub gradients: &'a [f64],
    pub lengths: &'a [f64],
}

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
    /// Deterministic mode disables randomized choices in tree and cycle sampling.
    pub deterministic: bool,
    /// Optional deterministic seed used only for stable tie-breaking.
    pub deterministic_seed: Option<u64>,
    pub rebuild_every: usize,
    pub last_rebuild: usize,
    pub tree: Option<LowStretchTree>,
    pub spanner: Option<DynamicSpanner>,
    pub use_spanner_embeddings: bool,
    pub stability_window: usize,
    pub ratio_tolerance: f64,
    pub stable_iters: usize,
    pub last_ratio: Option<f64>,
    pub early_stop_kappa: Option<f64>,
    pub approx_stats: Option<ApproximationStats>,
    pub cost_stats: OracleCostStats,
}

impl MinRatioOracle {
    pub fn new(seed: u64, rebuild_every: usize) -> Self {
        Self {
            seed,
            deterministic: false,
            deterministic_seed: None,
            rebuild_every,
            last_rebuild: 0,
            tree: None,
            spanner: None,
            use_spanner_embeddings: true,
            stability_window: 0,
            ratio_tolerance: 0.0,
            stable_iters: 0,
            last_ratio: None,
            early_stop_kappa: None,
            approx_stats: None,
            cost_stats: OracleCostStats::default(),
        }
    }

    pub fn new_with_mode(
        seed: u64,
        rebuild_every: usize,
        deterministic: bool,
        deterministic_seed: Option<u64>,
    ) -> Self {
        Self {
            deterministic,
            deterministic_seed,
            ..Self::new(seed, rebuild_every)
        }
    }

    pub fn new_with_stability(
        seed: u64,
        rebuild_every: usize,
        stability_window: usize,
        ratio_tolerance: f64,
    ) -> Self {
        Self {
            stability_window,
            ratio_tolerance,
            ..Self::new(seed, rebuild_every)
        }
    }

    pub fn rebuild_tree(
        &mut self,
        iter: usize,
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        gradients: &[f64],
        lengths: &[f64],
    ) -> Result<(), TreeError> {
        let tree = if self.deterministic {
            let seed = self.deterministic_seed.unwrap_or(0);
            LowStretchTree::build_with_mode(
                node_count,
                tails,
                heads,
                lengths,
                TreeBuildMode::Deterministic,
                seed,
            )?
        } else {
            LowStretchTree::build_with_mode(
                node_count,
                tails,
                heads,
                lengths,
                TreeBuildMode::Randomized,
                self.seed ^ iter as u64,
            )?
        };
        self.tree = Some(tree);
        if self.use_spanner_embeddings {
            let mut spanner = DynamicSpanner::new(node_count);
            for (edge_id, (&tail, &head)) in tails.iter().zip(heads.iter()).enumerate() {
                let gradient = gradients.get(edge_id).copied().unwrap_or(0.0);
                spanner.insert_edge_with_values(
                    tail as usize,
                    head as usize,
                    lengths[edge_id],
                    gradient,
                );
            }
            self.spanner = Some(spanner);
        }
        self.last_rebuild = iter;
        self.stable_iters = 0;
        self.last_ratio = None;
        self.approx_stats = None;
        Ok(())
    }

    fn should_rebuild(&self, iter: usize) -> bool {
        self.tree.is_none()
            || (self.use_spanner_embeddings && self.spanner.is_none())
            || iter == 0
            || (iter - self.last_rebuild) >= self.rebuild_every
            || (self.stability_window > 0 && self.stable_iters >= self.stability_window)
    }

    fn update_stability(&mut self, candidate: Option<&CycleCandidate>) {
        if self.stability_window == 0 {
            return;
        }
        match candidate {
            Some(best) => {
                if let Some(last) = self.last_ratio {
                    if (best.ratio - last).abs() <= self.ratio_tolerance {
                        self.stable_iters = self.stable_iters.saturating_add(1);
                    } else {
                        self.stable_iters = 1;
                    }
                } else {
                    self.stable_iters = 1;
                }
                self.last_ratio = Some(best.ratio);
            }
            None => {
                self.stable_iters = self.stable_iters.saturating_add(1);
                self.last_ratio = None;
            }
        }
    }

    pub fn best_cycle(
        &mut self,
        query: OracleQuery<'_>,
    ) -> Result<Option<CycleCandidate>, TreeError> {
        self.best_cycle_with_rebuild(query, false)
    }

    pub fn best_cycle_with_rebuild(
        &mut self,
        query: OracleQuery<'_>,
        force_rebuild: bool,
    ) -> Result<Option<CycleCandidate>, TreeError> {
        if force_rebuild || self.should_rebuild(query.iter) {
            self.rebuild_tree(
                query.iter,
                query.node_count,
                query.tails,
                query.heads,
                query.gradients,
                query.lengths,
            )?;
        }
        let best = if self.use_spanner_embeddings {
            self.best_cycle_over_embeddings(query)
        } else {
            let tree = self.tree.as_ref().expect("tree should exist after rebuild");
            best_cycle_over_edges(
                tree,
                query.tails,
                query.heads,
                query.gradients,
                query.lengths,
            )
        };

        self.update_stability(best.as_ref());
        let best = if let (Some(kappa), Some(candidate)) = (self.early_stop_kappa, best.as_ref()) {
            if candidate.ratio >= -kappa {
                None
            } else {
                best
            }
        } else {
            best
        };
        Ok(best)
    }

    pub fn enable_spanner_embeddings(&mut self, enabled: bool) {
        self.use_spanner_embeddings = enabled;
        if !enabled {
            self.spanner = None;
        }
    }

    pub fn set_early_stop_kappa(&mut self, kappa: Option<f64>) {
        self.early_stop_kappa = kappa;
    }

    pub fn approx_stats(&self) -> Option<&ApproximationStats> {
        self.approx_stats.as_ref()
    }

    pub fn apply_updates(
        &mut self,
        updates: &[(usize, f64, f64)],
        gradient_threshold: f64,
        length_factor: f64,
    ) -> usize {
        self.cost_stats.update_batches = self.cost_stats.update_batches.saturating_add(1);
        if let Some(spanner) = self.spanner.as_mut() {
            let instability =
                spanner.batch_update_edges(updates, gradient_threshold, length_factor);
            self.cost_stats.total_updates =
                self.cost_stats.total_updates.saturating_add(updates.len());
            instability
        } else {
            0
        }
    }

    fn best_cycle_over_embeddings(&mut self, query: OracleQuery<'_>) -> Option<CycleCandidate> {
        let tree = self.tree.as_ref()?;
        let spanner = self.spanner.as_mut()?;
        let mut stats = ApproximationStats::default();
        let mut best: Option<CycleCandidate> = None;
        let edge_count = query.tails.len();
        for edge_id in 0..edge_count {
            if let Some(values) = spanner.edge_values(edge_id) {
                if (values.length - query.lengths[edge_id]).abs() > 0.0
                    || (values.gradient - query.gradients[edge_id]).abs() > 0.0
                {
                    spanner.update_edge_values(
                        edge_id,
                        query.lengths[edge_id],
                        query.gradients[edge_id],
                    );
                }
            }
            if tree.tree_edges.get(edge_id).copied().unwrap_or(false) {
                continue;
            }
            let tail = query.tails[edge_id] as usize;
            let head = query.heads[edge_id] as usize;
            let steps = spanner
                .embedding_steps(edge_id)
                .map(|steps| steps.to_vec())
                .or_else(|| spanner.embed_edge_with_bfs(edge_id, head, tail));
            let Some(steps) = steps else {
                if let Some(fallback) = score_edge_cycle(
                    tree,
                    edge_id,
                    query.tails,
                    query.heads,
                    query.gradients,
                    query.lengths,
                ) {
                    best = Some(match best {
                        Some(current) => select_better_candidate(current, fallback),
                        None => fallback,
                    });
                    stats.candidate_count += 1;
                }
                continue;
            };
            let mut cycle_edges = Vec::with_capacity(steps.len() + 1);
            let mut numerator = query.gradients[edge_id];
            let mut denominator = query.lengths[edge_id].abs();
            cycle_edges.push((edge_id, 1));
            for step in &steps {
                let edge = spanner.edge_values(step.edge)?;
                if !edge.active {
                    continue;
                }
                let dir = step.dir as f64;
                numerator += dir * edge.gradient;
                denominator += edge.length.abs();
                cycle_edges.push((step.edge, step.dir));
                stats.update_errors(step.edge, edge, query);
            }
            if denominator <= 0.0 {
                continue;
            }
            let candidate = CycleCandidate {
                ratio: numerator / denominator,
                numerator,
                denominator,
                cycle_edges,
            };
            best = Some(match best {
                Some(current) => select_better_candidate(current, candidate),
                None => candidate,
            });
            stats.candidate_count += 1;
        }
        self.approx_stats = Some(stats);
        self.cost_stats.query_count = self.cost_stats.query_count.saturating_add(1);
        self.cost_stats.total_candidates = self
            .cost_stats
            .total_candidates
            .saturating_add(self.approx_stats.as_ref().map_or(0, |s| s.candidate_count));
        let tree_best = best_cycle_over_edges(
            tree,
            query.tails,
            query.heads,
            query.gradients,
            query.lengths,
        );
        match (best, tree_best) {
            (Some(left), Some(right)) => Some(select_better_candidate(left, right)),
            (Some(left), None) => Some(left),
            (None, Some(right)) => Some(right),
            (None, None) => None,
        }
    }
}

fn cycle_candidate_key(candidate: &CycleCandidate) -> (f64, usize) {
    let lead_edge = candidate
        .cycle_edges
        .first()
        .map(|(edge_id, _)| *edge_id)
        .unwrap_or(usize::MAX);
    (candidate.ratio, lead_edge)
}

pub(crate) fn select_better_candidate(
    left: CycleCandidate,
    right: CycleCandidate,
) -> CycleCandidate {
    const EPS: f64 = 1e-12;
    let left_key = cycle_candidate_key(&left);
    let right_key = cycle_candidate_key(&right);
    if left_key.0 + EPS < right_key.0 {
        left
    } else if (left_key.0 - right_key.0).abs() <= EPS {
        if left_key.1 <= right_key.1 {
            left
        } else {
            right
        }
    } else {
        right
    }
}

pub(crate) fn score_edge_cycle(
    tree: &LowStretchTree,
    edge_id: usize,
    tails: &[u32],
    heads: &[u32],
    gradients: &[f64],
    lengths: &[f64],
) -> Option<CycleCandidate> {
    if tree.tree_edges[edge_id] {
        return None;
    }
    let tail = tails[edge_id] as usize;
    let head = heads[edge_id] as usize;
    let path_edges = tree.path_edges(head, tail, tails, heads)?;

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
        return None;
    }
    let ratio = numerator / denominator;
    Some(CycleCandidate {
        ratio,
        numerator,
        denominator,
        cycle_edges,
    })
}

#[derive(Debug, Default, Clone)]
pub struct ApproximationStats {
    pub max_gradient_error: f64,
    pub max_length_ratio: f64,
    pub candidate_count: usize,
}

impl ApproximationStats {
    fn update_errors(&mut self, edge_id: usize, edge: EdgeValues, query: OracleQuery<'_>) {
        let query_gradient = query.gradients.get(edge_id).copied().unwrap_or(0.0);
        let query_length = query.lengths.get(edge_id).copied().unwrap_or(0.0).abs();
        let grad_error = (edge.gradient - query_gradient).abs();
        if grad_error > self.max_gradient_error {
            self.max_gradient_error = grad_error;
        }
        if query_length > 0.0 && edge.length > 0.0 {
            let ratio = if edge.length > query_length {
                edge.length / query_length
            } else {
                query_length / edge.length
            };
            if ratio > self.max_length_ratio {
                self.max_length_ratio = ratio;
            }
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct OracleCostStats {
    pub query_count: usize,
    pub update_batches: usize,
    pub total_updates: usize,
    pub total_candidates: usize,
}

#[cfg(feature = "parallel")]
pub(crate) fn best_cycle_over_edges(
    tree: &LowStretchTree,
    tails: &[u32],
    heads: &[u32],
    gradients: &[f64],
    lengths: &[f64],
) -> Option<CycleCandidate> {
    (0..tails.len())
        .into_par_iter()
        .filter_map(|edge_id| score_edge_cycle(tree, edge_id, tails, heads, gradients, lengths))
        .reduce_with(select_better_candidate)
}

#[cfg(not(feature = "parallel"))]
pub(crate) fn best_cycle_over_edges(
    tree: &LowStretchTree,
    tails: &[u32],
    heads: &[u32],
    gradients: &[f64],
    lengths: &[f64],
) -> Option<CycleCandidate> {
    let mut best: Option<CycleCandidate> = None;
    for edge_id in 0..tails.len() {
        let Some(candidate) = score_edge_cycle(tree, edge_id, tails, heads, gradients, lengths)
        else {
            continue;
        };
        best = Some(match best {
            Some(current) => select_better_candidate(current, candidate),
            None => candidate,
        });
    }
    best
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
    fn deterministic_candidate_tie_breaker() {
        let left = CycleCandidate {
            ratio: -1.0,
            numerator: -2.0,
            denominator: 2.0,
            cycle_edges: vec![(3, 1)],
        };
        let right = CycleCandidate {
            ratio: -1.0,
            numerator: -4.0,
            denominator: 4.0,
            cycle_edges: vec![(7, 1)],
        };
        let chosen = select_better_candidate(left.clone(), right.clone());
        assert_eq!(chosen.cycle_edges[0].0, 3);

        let chosen = select_better_candidate(right, left);
        assert_eq!(chosen.cycle_edges[0].0, 3);
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
                .best_cycle(OracleQuery {
                    iter: 0,
                    node_count,
                    tails: &tails,
                    heads: &heads,
                    gradients: &gradients,
                    lengths: &lengths,
                })
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

    #[test]
    fn finds_negative_ratio_cycle_when_available() {
        let tails = vec![0, 1, 2];
        let heads = vec![1, 2, 0];
        let gradients = vec![-2.0, -3.0, -1.0];
        let lengths = vec![1.0, 1.0, 1.0];
        let mut oracle = MinRatioOracle::new(3, 10);
        let best = oracle
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
        assert!(
            best.ratio < 0.0,
            "expected negative ratio, got {}",
            best.ratio
        );
    }

    #[test]
    fn deterministic_mode_rebuilds_with_stable_cycle() {
        let tails = vec![0, 1, 0, 2];
        let heads = vec![1, 2, 2, 0];
        let gradients = vec![1.0, -3.0, 2.0, -1.0];
        let lengths = vec![2.0, 1.0, 4.0, 3.0];
        let mut oracle = MinRatioOracle::new_with_mode(17, 1, true, None);
        let best_first = oracle
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
        let best_second = oracle
            .best_cycle(OracleQuery {
                iter: 1,
                node_count: 3,
                tails: &tails,
                heads: &heads,
                gradients: &gradients,
                lengths: &lengths,
            })
            .unwrap()
            .unwrap();
        assert_eq!(best_first.cycle_edges, best_second.cycle_edges);
    }

    #[test]
    fn stability_trigger_forces_rebuild() {
        let tails = vec![0, 1, 2];
        let heads = vec![1, 2, 0];
        let gradients = vec![1.0, -1.0, 0.5];
        let lengths = vec![1.0, 1.0, 1.0];
        let mut oracle = MinRatioOracle::new_with_stability(7, 100, 1, 1e-12);
        oracle
            .best_cycle(OracleQuery {
                iter: 0,
                node_count: 3,
                tails: &tails,
                heads: &heads,
                gradients: &gradients,
                lengths: &lengths,
            })
            .unwrap();
        assert_eq!(oracle.last_rebuild, 0);
        oracle
            .best_cycle(OracleQuery {
                iter: 1,
                node_count: 3,
                tails: &tails,
                heads: &heads,
                gradients: &gradients,
                lengths: &lengths,
            })
            .unwrap();
        assert_eq!(oracle.last_rebuild, 1);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_best_cycle_matches_serial_scoring() {
        let mut rng = TestRng::new(17);
        let node_count = 12;
        let edge_count = 40;
        let mut tails = Vec::with_capacity(edge_count);
        let mut heads = Vec::with_capacity(edge_count);
        let mut gradients = Vec::with_capacity(edge_count);
        let mut lengths = Vec::with_capacity(edge_count);
        for _ in 0..edge_count {
            let tail = rng.next_usize(node_count) as u32;
            let mut head = rng.next_usize(node_count) as u32;
            if head == tail {
                head = (head + 1) % node_count as u32;
            }
            tails.push(tail);
            heads.push(head);
            gradients.push((rng.next_u64() % 13) as f64 - 6.0);
            lengths.push((rng.next_u64() % 9) as f64 + 1.0);
        }

        let mut oracle = MinRatioOracle::new(9, 1);
        oracle.enable_spanner_embeddings(false);
        let best_parallel = oracle
            .best_cycle(OracleQuery {
                iter: 0,
                node_count,
                tails: &tails,
                heads: &heads,
                gradients: &gradients,
                lengths: &lengths,
            })
            .unwrap()
            .unwrap();

        let tree = oracle.tree.as_ref().expect("tree should be built");
        let mut best_serial: Option<CycleCandidate> = None;
        for edge_id in 0..edge_count {
            let Some(candidate) =
                score_edge_cycle(tree, edge_id, &tails, &heads, &gradients, &lengths)
            else {
                continue;
            };
            best_serial = Some(match best_serial {
                Some(current) => select_better_candidate(current, candidate),
                None => candidate,
            });
        }
        let best_serial = best_serial.expect("should find a candidate");
        assert_eq!(
            cycle_candidate_key(&best_parallel),
            cycle_candidate_key(&best_serial)
        );
    }
}
