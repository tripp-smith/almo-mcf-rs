use crate::data_structures::decremental_spanner::DecrementalSpanner as HierDecrementalSpanner;
use crate::graph::{Graph, NodeId};
use crate::min_ratio::{
    score_edge_cycle, select_better_candidate, CycleCandidate, MinRatioOracle, OracleQuery,
    TreeError,
};
use crate::spanner::DynamicSpanner;
use crate::trees::dynamic::{DynamicTreeChain, DynamicTreeChainConfig};
use crate::trees::TreeBuildMode;
use rayon::prelude::*;
use std::fmt;

#[derive(Debug, Clone)]
pub struct SparseFlowDelta {
    pub edge_deltas: Vec<(usize, f64)>,
}

impl SparseFlowDelta {
    pub fn new(edge_deltas: Vec<(usize, f64)>) -> Self {
        Self { edge_deltas }
    }

    pub fn total_abs_flow(&self) -> f64 {
        self.edge_deltas.iter().map(|(_, delta)| delta.abs()).sum()
    }
}

#[derive(Debug, Clone)]
pub enum OracleError {
    RebuildRequired {
        level: Option<usize>,
        reason: String,
    },
    InvalidEdge {
        edge_idx: usize,
    },
    Other(String),
}

impl fmt::Display for OracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OracleError::RebuildRequired { level, reason } => {
                write!(f, "rebuild required at {level:?}: {reason}")
            }
            OracleError::InvalidEdge { edge_idx } => {
                write!(f, "invalid edge index {edge_idx}")
            }
            OracleError::Other(message) => write!(f, "{message}"),
        }
    }
}

pub trait DynamicUpdateOracle {
    fn update_gradient(&mut self, edge_idx: usize, new_g: f64) -> Result<(), OracleError>;
    fn update_length(&mut self, edge_idx: usize, new_ell: f64) -> Result<(), OracleError>;
    fn batch_update_gradient(&mut self, updates: &[(usize, f64)]) -> Result<(), OracleError>;
    fn batch_update_lengths(&mut self, updates: &[(usize, f64)]) -> Result<(), OracleError>;
    fn update_many(
        &mut self,
        g_updates: &[(usize, f64)],
        ell_updates: &[(usize, f64)],
    ) -> Result<(), OracleError>;
    fn notify_flow_change(&mut self, flow_delta: &SparseFlowDelta) -> Result<(), OracleError>;
}

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
    last_gradients: Vec<f64>,
    last_lengths: Vec<f64>,
    decremental_spanner: HierDecrementalSpanner,
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
            last_gradients: Vec::new(),
            last_lengths: Vec::new(),
            decremental_spanner: HierDecrementalSpanner::default(),
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
        self.last_gradients = query.gradients.to_vec();
        self.last_lengths = query.lengths.to_vec();

        let mut graph = Graph::new(node_count);
        for edge_id in 0..query.tails.len() {
            let _ = graph.add_edge(
                NodeId(query.tails[edge_id] as usize),
                NodeId(query.heads[edge_id] as usize),
                0.0,
                1.0,
                query.lengths[edge_id],
            );
        }
        self.decremental_spanner
            .initialize(&graph, 2.0, 2.0, self.max_levels);
        Ok(())
    }

    pub fn find_approx_min_ratio_cycle(
        &mut self,
        query: OracleQuery<'_>,
    ) -> Result<Option<CycleCandidate>, TreeError> {
        self.decremental_spanner.update(query.iter, &[], &[]);
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
        let edge_count = query.tails.len();
        if edge_count == 0 {
            return Ok(None);
        }
        let candidates: Vec<usize> = (0..edge_count).collect();
        let threads = rayon::current_num_threads().max(1);
        let chunk_size = (edge_count / threads).max(64);
        let best = candidates
            .par_chunks(chunk_size)
            .filter_map(|chunk| {
                let mut local_best: Option<CycleCandidate> = None;
                for &edge_id in chunk {
                    let Some(candidate) = score_edge_cycle(
                        tree,
                        edge_id,
                        query.tails,
                        query.heads,
                        query.gradients,
                        query.lengths,
                    ) else {
                        continue;
                    };
                    local_best = Some(match local_best {
                        Some(current) => select_better_candidate(current, candidate),
                        None => candidate,
                    });
                }
                local_best
            })
            .reduce_with(select_better_candidate);
        Ok(best)
    }
}

impl DynamicUpdateOracle for DynamicOracle {
    fn update_gradient(&mut self, edge_idx: usize, new_g: f64) -> Result<(), OracleError> {
        if edge_idx >= self.last_edge_count {
            return Err(OracleError::InvalidEdge { edge_idx });
        }
        if self.last_gradients.len() != self.last_edge_count {
            self.last_gradients.resize(self.last_edge_count, 0.0);
        }
        if let Some(gradient) = self.last_gradients.get_mut(edge_idx) {
            *gradient = new_g;
        }
        Ok(())
    }

    fn update_length(&mut self, edge_idx: usize, new_ell: f64) -> Result<(), OracleError> {
        if edge_idx >= self.last_edge_count {
            return Err(OracleError::InvalidEdge { edge_idx });
        }
        if self.last_lengths.len() != self.last_edge_count {
            self.last_lengths.resize(self.last_edge_count, 0.0);
        }
        if let Some(length) = self.last_lengths.get_mut(edge_idx) {
            *length = new_ell;
        }
        Ok(())
    }

    fn batch_update_gradient(&mut self, updates: &[(usize, f64)]) -> Result<(), OracleError> {
        for &(edge_idx, new_g) in updates {
            self.update_gradient(edge_idx, new_g)?;
        }
        Ok(())
    }

    fn batch_update_lengths(&mut self, updates: &[(usize, f64)]) -> Result<(), OracleError> {
        for &(edge_idx, new_ell) in updates {
            self.update_length(edge_idx, new_ell)?;
        }
        Ok(())
    }

    fn update_many(
        &mut self,
        g_updates: &[(usize, f64)],
        ell_updates: &[(usize, f64)],
    ) -> Result<(), OracleError> {
        self.batch_update_gradient(g_updates)?;
        self.batch_update_lengths(ell_updates)?;
        Ok(())
    }

    fn notify_flow_change(&mut self, _flow_delta: &SparseFlowDelta) -> Result<(), OracleError> {
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub enum OracleEngine {
    Static(Box<MinRatioOracle>),
    Dynamic(Box<DynamicOracle>),
}

impl OracleEngine {
    pub fn new(seed: u64, rebuild_every: usize, deterministic: bool, use_dynamic: bool) -> Self {
        if use_dynamic {
            OracleEngine::Dynamic(Box::new(DynamicOracle::new(seed, deterministic)))
        } else {
            OracleEngine::Static(Box::new(MinRatioOracle::new_with_mode(
                seed,
                rebuild_every.max(1),
                deterministic,
                None,
            )))
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
                    let mut fallback =
                        MinRatioOracle::new_with_mode(0, 25, oracle.deterministic, None);
                    fallback.best_cycle(query)
                }
            }
        }
    }
}
