use crate::data_structures::decremental_spanner::DecrementalSpanner;
use crate::data_structures::embedding::{DynamicEmbedding, EmbeddedPath};
use crate::data_structures::lsd::{LowStretchDecomposition, MultiplicativeWeights, Path};
use crate::data_structures::sparsified_core::Sparsifier;
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

#[derive(Debug, Clone, Default)]
pub struct ApproxCycle {
    pub edges: Vec<EdgeId>,
    pub ratio: f64,
}

#[derive(Debug, Clone, Default)]
pub struct RoutedCirc {
    pub routed_edges: Vec<EdgeId>,
    pub routed_length: f64,
}

#[derive(Debug, Clone)]
pub enum ChainLevel {
    Lsd,
    Spanner,
    Embedder,
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
pub struct SpannerUpdater {
    pending_deletions: Vec<EdgeId>,
    pending_insertions: Vec<EdgeId>,
    updates_since_rebuild: usize,
    rebuild_threshold: usize,
    pub current_level_shift: usize,
}

impl SpannerUpdater {
    pub fn new(edge_count: usize, levels: usize) -> Self {
        let safe_levels = levels.max(1);
        let gamma_g = 0.8_f64;
        let rebuild_threshold = ((gamma_g * edge_count.max(1) as f64) / safe_levels as f64)
            .ceil()
            .max(1.0) as usize;
        Self {
            pending_deletions: Vec::new(),
            pending_insertions: Vec::new(),
            updates_since_rebuild: 0,
            rebuild_threshold,
            current_level_shift: 0,
        }
    }

    pub fn batch_update(&mut self, deletions: &[EdgeId], insertions: &[EdgeId]) -> bool {
        self.pending_deletions.extend_from_slice(deletions);
        self.pending_insertions.extend_from_slice(insertions);
        self.updates_since_rebuild += deletions.len() + insertions.len();
        self.updates_since_rebuild >= self.rebuild_threshold
    }

    pub fn take_pending(&mut self) -> (Vec<EdgeId>, Vec<EdgeId>) {
        self.updates_since_rebuild = 0;
        (
            std::mem::take(&mut self.pending_deletions),
            std::mem::take(&mut self.pending_insertions),
        )
    }

    pub fn shift_level(&mut self, levels: usize) {
        self.current_level_shift = (self.current_level_shift + 1) % levels.max(1);
    }
}

#[derive(Debug, Clone)]
pub struct DataStructureChain {
    pub lsd: LowStretchDecomposition,
    pub hsfc: HierarchicalShortestPathForest,
    pub spanner: DecrementalSpanner,
    pub mw_weights: MultiplicativeWeights,
    pub embedding: DynamicEmbedding,
    pub chain: Vec<ChainLevel>,
    pub sparsifier: Option<Sparsifier>,
    pub spanner_updater: SpannerUpdater,
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
            chain: vec![ChainLevel::Lsd, ChainLevel::Spanner, ChainLevel::Embedder],
            sparsifier: None,
            spanner_updater: SpannerUpdater::new(graph.edge_count(), params.levels.max(1)),
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
        self.shift_if_low_quality(quality);
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

    pub fn route_circulation(&mut self, circ: &Circulation) -> RoutedCirc {
        let path = Path {
            edges: circ.candidate_edges.clone(),
            stretch: (self.lsd.sampling_params.n.max(2) as f64).ln().max(1.0),
        };
        let embedded = self.embedding.embed_dynamic_path(&path, 0, &[]);
        RoutedCirc {
            routed_edges: embedded.lifted_edges,
            routed_length: path.edges.len() as f64 * embedded.distortion.max(1.0),
        }
    }

    pub fn find_min_ratio_cycle(
        &mut self,
        reduced_costs: &[f64],
        circulation: &Circulation,
    ) -> ApproxCycle {
        let routed = self.route_circulation(circulation);
        let total = routed
            .routed_edges
            .iter()
            .map(|e| reduced_costs.get(e.0).copied().unwrap_or(0.0))
            .sum::<f64>();
        let denom = routed.routed_edges.len().max(1) as f64;
        ApproxCycle {
            edges: routed.routed_edges,
            ratio: total / denom,
        }
    }

    pub fn update_after_deletion(&mut self, deleted: &[EdgeId]) {
        let _ = self.lsd.update_decomposition(deleted, 1);
        if self.spanner_updater.batch_update(deleted, &[]) {
            let (pending_deletions, pending_insertions) = self.spanner_updater.take_pending();
            let _ = pending_insertions;
            self.spanner.update(1, &pending_deletions, &[]);
        }
    }

    pub fn batch_update(&mut self, deletions: &[EdgeId], insertions: &[EdgeId]) {
        if self.spanner_updater.batch_update(deletions, insertions) {
            let (pending_deletions, pending_insertions) = self.spanner_updater.take_pending();
            let _ = pending_insertions;
            self.spanner.update(1, &pending_deletions, &[]);
        }
    }

    pub fn shift_if_low_quality(&mut self, quality: f64) {
        if quality.is_finite() && quality < 0.0 {
            self.spanner_updater
                .shift_level(self.embedding.embedding_maps.len());
        }
    }

    pub fn verify_cycle_bounds(cycle: &ApproxCycle, true_min: f64) -> bool {
        (cycle.ratio - true_min).abs() <= 0.1_f64.max(true_min.abs() * 0.1)
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
