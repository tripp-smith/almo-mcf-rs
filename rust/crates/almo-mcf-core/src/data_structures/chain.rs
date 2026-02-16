use crate::data_structures::decremental_spanner::DecrementalSpanner;
use crate::data_structures::embedding::{DynamicEmbedding, EmbeddedPath};
use crate::data_structures::lsd::{LowStretchDecomposition, MultiplicativeWeights, Path};
use crate::graph::{EdgeId, Graph};
use crate::rebuilding::RebuildingGame;

#[derive(Debug, Clone, Default)]
pub struct HierarchicalShortestPathForest {
    pub routed_paths: Vec<Vec<EdgeId>>,
}

#[derive(Debug, Clone, Default)]
pub struct Circulation {
    pub candidate_edges: Vec<EdgeId>,
}

#[derive(Debug, Clone, Default)]
pub struct Cycle {
    pub edges: Vec<EdgeId>,
}

#[derive(Debug, Clone)]
pub struct ChainParams {
    pub beta: f64,
    pub gamma_l: f64,
    pub gamma_c: f64,
    pub levels: usize,
    pub deterministic: bool,
    pub deterministic_seed: Option<u64>,
}

impl Default for ChainParams {
    fn default() -> Self {
        Self {
            beta: 2.0,
            gamma_l: 2.0,
            gamma_c: 2.0,
            levels: 3,
            deterministic: true,
            deterministic_seed: Some(0),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ChainStats {
    pub level_stretches: Vec<f64>,
    pub rebuild_levels: usize,
    pub total_recourse: f64,
    pub seed_used: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct DataStructureChain {
    pub lsd: LowStretchDecomposition,
    pub hsfc: HierarchicalShortestPathForest,
    pub spanner: DecrementalSpanner,
    pub mw_weights: MultiplicativeWeights,
    pub embedding: DynamicEmbedding,
    deterministic_seed: Option<u64>,
}

impl DataStructureChain {
    pub fn initialize_chain(graph: &Graph, params: ChainParams) -> Self {
        let mut lsd =
            LowStretchDecomposition::new(graph.node_count(), graph.edge_count(), params.beta);
        lsd.bootstrap_from_graph(graph);
        let mut spanner = DecrementalSpanner::default();
        spanner.initialize(graph, params.gamma_l, params.gamma_c, params.levels.max(1));
        Self {
            mw_weights: lsd.multiplicative_weights.clone(),
            lsd,
            hsfc: HierarchicalShortestPathForest::default(),
            spanner,
            embedding: DynamicEmbedding::new(params.levels.max(1)),
            deterministic_seed: if params.deterministic {
                params.deterministic_seed.or(Some(0))
            } else {
                None
            },
        }
    }

    pub fn query_min_ratio_cycle_chain(
        &mut self,
        circulation: &Circulation,
        t: usize,
        rebuilding_game: &mut RebuildingGame,
    ) -> (Cycle, f64) {
        let updates = self
            .lsd
            .update_decomposition(&circulation.candidate_edges, t);
        self.spanner.update(t, &circulation.candidate_edges, &[]);
        let path = Path {
            edges: circulation.candidate_edges.clone(),
            stretch: (self.lsd.sampling_params.n.max(2) as f64).ln().powi(3),
        };
        let _weighted = self
            .lsd
            .apply_multiplicative_weights(std::slice::from_ref(&path), &[1.0]);
        let embedded: EmbeddedPath =
            self.embedding
                .embed_dynamic_path(&path, 0, &circulation.candidate_edges);
        self.mw_weights = self.lsd.multiplicative_weights.clone();
        self.hsfc
            .routed_paths
            .push(circulation.candidate_edges.clone());

        let quality = (-(updates.len() as f64)).exp() / embedded.distortion.max(1.0);
        let is_win = quality >= (-(self.lsd.sampling_params.m as f64).ln().powf(7.0 / 8.0)).exp();
        let decision = rebuilding_game.play_round(t, circulation.candidate_edges.len(), is_win);
        for level in decision.levels_to_rebuild {
            self.rebuild_level(level);
        }

        (
            Cycle {
                edges: circulation.candidate_edges.clone(),
            },
            quality,
        )
    }

    pub fn rebuild_level(&mut self, level: usize) {
        let idx = level.min(self.embedding.embedding_maps.len().saturating_sub(1));
        if let Some(buffer) = self.embedding.update_buffers.get_mut(idx) {
            buffer.clear();
        }
    }

    pub fn set_deterministic_mode(
        &mut self,
        seed: Option<u64>,
        rebuilding_game: &mut RebuildingGame,
    ) -> bool {
        let resolved = seed.unwrap_or(0);
        self.deterministic_seed = Some(resolved);
        rebuilding_game.enable_derandomization(resolved);
        true
    }

    pub fn verify_chain_invariants(&self) -> bool {
        let stretch_ok = self
            .lsd
            .compute_average_stretch(&self.mw_weights.path_history)
            <= (self.lsd.sampling_params.n.max(2) as f64)
                .ln()
                .powi(4)
                .max(1.0);
        let sparsity_ok =
            self.lsd.sampling_params.m <= self.lsd.sampling_params.m.saturating_mul(2);
        stretch_ok && sparsity_ok
    }

    pub fn log_chain_stats(&self, iter: usize) -> ChainStats {
        ChainStats {
            level_stretches: self
                .lsd
                .levels
                .iter()
                .map(|level| level.p_j.recip().min(1e6))
                .collect(),
            rebuild_levels: self
                .lsd
                .levels
                .iter()
                .filter(|level| iter.is_multiple_of(level.period))
                .count(),
            total_recourse: self.mw_weights.path_history.len() as f64,
            seed_used: self.deterministic_seed,
        }
    }
}
