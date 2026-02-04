use crate::min_ratio::{
    best_cycle_over_edges, CycleCandidate, MinRatioOracle, OracleQuery, TreeError,
};
use crate::spanner::DynamicSpanner;
use crate::trees::dynamic::{DynamicTreeChain, DynamicTreeChainConfig};
use crate::trees::TreeBuildMode;

#[derive(Debug, Clone)]
pub struct DynamicOracle {
    pub tree_chain: DynamicTreeChain,
    pub spanner: DynamicSpanner,
    pub deterministic: bool,
    pub max_levels: usize,
    pub trees_per_level: usize,
    pub spanner_degree: usize,
    last_node_count: usize,
    last_edge_count: usize,
    last_lengths: Vec<f64>,
}

impl DynamicOracle {
    pub fn new(seed: u64, deterministic: bool) -> Self {
        let config = if deterministic {
            DynamicTreeChainConfig::deterministic(3, 1)
        } else {
            DynamicTreeChainConfig::randomized(seed, 3, 1)
        };
        let tree_chain = DynamicTreeChain::build(1, &[], &[], &[], config).unwrap_or_else(|_| {
            DynamicTreeChain {
                base_node_count: 1,
                levels: Vec::new(),
            }
        });
        Self {
            tree_chain,
            spanner: DynamicSpanner::new(1),
            deterministic,
            max_levels: 3,
            trees_per_level: 1,
            spanner_degree: 4,
            last_node_count: 0,
            last_edge_count: 0,
            last_lengths: Vec::new(),
        }
    }

    fn rebuild(&mut self, node_count: usize, query: &OracleQuery<'_>) -> Result<(), TreeError> {
        let config = DynamicTreeChainConfig {
            seed: if self.deterministic {
                0
            } else {
                query.iter as u64
            },
            max_levels: self.max_levels,
            trees_per_level: self.trees_per_level,
            build_mode: if self.deterministic {
                TreeBuildMode::Deterministic
            } else {
                TreeBuildMode::Randomized
            },
        };
        self.tree_chain.rebuild_from_graph(
            node_count,
            query.tails,
            query.heads,
            query.lengths,
            config,
        )?;
        self.spanner = DynamicSpanner::build_sparse_subgraph(
            node_count,
            query.tails,
            query.heads,
            query.lengths,
            query.gradients,
            self.spanner_degree,
            self.deterministic,
        );
        self.last_node_count = node_count;
        self.last_edge_count = query.tails.len();
        self.last_lengths = query.lengths.to_vec();
        Ok(())
    }

    pub fn find_approx_min_ratio_cycle(
        &mut self,
        query: OracleQuery<'_>,
    ) -> Result<Option<CycleCandidate>, TreeError> {
        if self.tree_chain.levels.is_empty()
            || self.last_node_count != query.node_count
            || self.last_edge_count != query.tails.len()
            || self.last_lengths.len() != query.lengths.len()
            || self
                .last_lengths
                .iter()
                .zip(query.lengths.iter())
                .any(|(a, b)| (a - b).abs() > 1e-12)
        {
            self.rebuild(query.node_count, &query)?;
        }
        let Some(level) = self.tree_chain.levels.first() else {
            return Ok(None);
        };
        let Some(tree) = level.trees.first() else {
            return Ok(None);
        };
        let best = best_cycle_over_edges(
            tree,
            query.tails,
            query.heads,
            query.gradients,
            query.lengths,
        );
        Ok(best)
    }
}

#[derive(Debug, Clone)]
pub enum OracleEngine {
    Static(MinRatioOracle),
    Dynamic(DynamicOracle),
}

impl OracleEngine {
    pub fn new(seed: u64, rebuild_every: usize, deterministic: bool, use_dynamic: bool) -> Self {
        if use_dynamic {
            OracleEngine::Dynamic(DynamicOracle::new(seed, deterministic))
        } else {
            OracleEngine::Static(MinRatioOracle::new_with_mode(
                seed,
                rebuild_every.max(1),
                deterministic,
            ))
        }
    }

    pub fn best_cycle(
        &mut self,
        query: OracleQuery<'_>,
        use_dynamic: bool,
    ) -> Result<Option<CycleCandidate>, TreeError> {
        match self {
            OracleEngine::Static(oracle) => oracle.best_cycle(query),
            OracleEngine::Dynamic(oracle) => {
                if use_dynamic {
                    oracle.find_approx_min_ratio_cycle(query)
                } else {
                    let mut fallback = MinRatioOracle::new_with_mode(0, 25, oracle.deterministic);
                    fallback.best_cycle(query)
                }
            }
        }
    }
}
