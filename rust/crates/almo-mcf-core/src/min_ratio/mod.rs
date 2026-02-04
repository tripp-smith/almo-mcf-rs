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
    pub stability_window: usize,
    pub ratio_tolerance: f64,
    pub stable_iters: usize,
    pub last_ratio: Option<f64>,
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
            stability_window: 0,
            ratio_tolerance: 0.0,
            stable_iters: 0,
            last_ratio: None,
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
        self.last_rebuild = iter;
        self.stable_iters = 0;
        self.last_ratio = None;
        Ok(())
    }

    fn should_rebuild(&self, iter: usize) -> bool {
        self.tree.is_none()
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
                query.lengths,
            )?;
        }
        let tree = self.tree.as_ref().expect("tree should exist after rebuild");
        let best = best_cycle_over_edges(
            tree,
            query.tails,
            query.heads,
            query.gradients,
            query.lengths,
        );

        self.update_stability(best.as_ref());
        Ok(best)
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
