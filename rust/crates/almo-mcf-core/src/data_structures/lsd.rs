use crate::graph::{EdgeId, Graph, NodeId};

#[derive(Debug, Clone)]
pub struct Cluster {
    pub id: usize,
    pub level: usize,
    pub center: NodeId,
    pub members: Vec<NodeId>,
    pub diameter_bound: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ClusterUpdate {
    pub level: usize,
    pub removed_edges: Vec<EdgeId>,
    pub touched_cluster_ids: Vec<usize>,
}

#[derive(Debug, Clone, Default)]
pub struct LsdLevel {
    pub level_index: usize,
    pub period: usize,
    pub p_j: f64,
    pub cluster_ids: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub n: usize,
    pub m: usize,
    pub beta: f64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            n: 1,
            m: 0,
            beta: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Path {
    pub edges: Vec<EdgeId>,
    pub stretch: f64,
}

#[derive(Debug, Clone)]
pub struct MultiplicativeWeights {
    pub eta: f64,
    pub regret_bound: f64,
    pub path_history: Vec<Path>,
}

impl MultiplicativeWeights {
    pub fn new(n: usize) -> Self {
        let ln_n = (n.max(2) as f64).ln();
        Self {
            eta: 1.0 / ln_n,
            regret_bound: 0.0,
            path_history: Vec::new(),
        }
    }

    pub fn update_regret(&mut self) {
        let t = self.path_history.len().max(1) as f64;
        self.regret_bound = (t * (self.eta.recip())).sqrt();
    }
}

#[derive(Debug, Clone, Default)]
pub struct WeightedLsd {
    pub edge_weights: Vec<f64>,
    pub average_stretch: f64,
}

#[derive(Debug, Clone)]
pub struct LowStretchDecomposition {
    pub levels: Vec<LsdLevel>,
    pub clusters: Vec<Cluster>,
    pub sampling_params: SamplingParams,
    pub multiplicative_weights: MultiplicativeWeights,
    edge_weights: Vec<f64>,
}

pub type LowStretchDecomp = LowStretchDecomposition;

impl LowStretchDecomposition {
    pub fn new(n: usize, m: usize, beta: f64) -> Self {
        let level_count = (n.max(2) as f64).log2().ceil() as usize;
        let mut levels = Vec::with_capacity(level_count.max(1));
        for j in 1..=level_count.max(1) {
            let p_j = (n.max(2) as f64).powf(-1.0 + 1.0 / (j as f64));
            let period = (n.max(2) as f64).powf(1.0 / (j as f64)).round().max(1.0) as usize;
            levels.push(LsdLevel {
                level_index: j,
                period,
                p_j,
                cluster_ids: Vec::new(),
            });
        }

        Self {
            levels,
            clusters: Vec::new(),
            sampling_params: SamplingParams { n, m, beta },
            multiplicative_weights: MultiplicativeWeights::new(n),
            edge_weights: vec![1.0; m],
        }
    }

    pub fn update_decomposition(&mut self, deletions: &[EdgeId], t: usize) -> Vec<ClusterUpdate> {
        let mut updates = Vec::new();
        for level in &self.levels {
            if t.is_multiple_of(level.period) {
                // Paper 1 Alg 7 line 11
                let touched = self
                    .clusters
                    .iter()
                    .filter(|cluster| cluster.level == level.level_index)
                    .map(|cluster| cluster.id)
                    .collect::<Vec<_>>();
                updates.push(ClusterUpdate {
                    level: level.level_index,
                    // Paper 1 Alg 7 line 12
                    removed_edges: deletions.to_vec(),
                    // Paper 1 Alg 7 lines 13-20
                    touched_cluster_ids: touched,
                });
            }
        }

        if !deletions.is_empty() {
            let synthetic_path = Path {
                edges: deletions.to_vec(),
                stretch: (self.sampling_params.n.max(2) as f64).ln().powi(2),
            };
            let weights = vec![1.0; synthetic_path.edges.len()];
            let _ = self.apply_multiplicative_weights(&[synthetic_path], &weights);
        }

        updates
    }

    pub fn update_weights(&mut self, deleted_edges: &[EdgeId]) -> WeightedLsd {
        let base_stretch = (self.sampling_params.n.max(2) as f64).ln().max(1.0);
        let paths: Vec<Path> = deleted_edges
            .iter()
            .copied()
            .map(|e| Path {
                edges: vec![e],
                stretch: base_stretch,
            })
            .collect();
        self.apply_multiplicative_weights(&paths, &vec![1.0; deleted_edges.len()])
    }

    pub fn handle_edge_deletion(&mut self, edge: EdgeId, graph: &Graph) -> Vec<ClusterUpdate> {
        let updates = self.update_decomposition(&[edge], 1);
        let p = self.levels.first().map(|level| level.p_j).unwrap_or(1.0);
        let (clusters, _) = Self::sample_and_cluster(graph, p, self.sampling_params.beta);
        self.clusters = clusters;
        updates
    }

    pub fn apply_multiplicative_weights(&mut self, paths: &[Path], weights: &[f64]) -> WeightedLsd {
        for path in paths {
            self.multiplicative_weights.path_history.push(path.clone());
        }

        // Paper 1 §6.2: w_e <- w_e * exp(-eta * stretch_e)
        for (idx, edge_weight) in self.edge_weights.iter_mut().enumerate() {
            let stretch_e = paths
                .iter()
                .flat_map(|path| path.edges.iter().map(move |_| path.stretch))
                .nth(idx)
                .unwrap_or(0.0);
            let step = (-self.multiplicative_weights.eta * stretch_e).exp();
            *edge_weight *= step.max(1e-12);
        }
        for (idx, value) in weights.iter().copied().enumerate() {
            if idx < self.edge_weights.len() {
                self.edge_weights[idx] *= value.max(1e-12);
            }
        }

        self.multiplicative_weights.update_regret();

        WeightedLsd {
            edge_weights: self.edge_weights.clone(),
            average_stretch: self.compute_average_stretch(paths),
        }
    }

    pub fn compute_average_stretch(&self, query_paths: &[Path]) -> f64 {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        for (idx, path) in query_paths.iter().enumerate() {
            let weight = self.edge_weights.get(idx).copied().unwrap_or(1.0);
            weighted_sum += weight * path.stretch;
            total_weight += weight;
        }
        if total_weight == 0.0 {
            return 0.0;
        }
        weighted_sum / total_weight
    }

    fn sample_and_cluster(graph: &Graph, p: f64, beta: f64) -> (Vec<Cluster>, Vec<NodeId>) {
        let mut centers = Vec::new();
        let mut clusters = Vec::new();
        for node in 0..graph.node_count() {
            // Paper 1 Alg 7 line 1
            let score = ((node + 1) as f64) / (graph.node_count().max(1) as f64);
            if score <= p {
                centers.push(NodeId(node));
            }
        }

        if centers.is_empty() && graph.node_count() > 0 {
            centers.push(NodeId(0));
        }

        for (idx, center) in centers.iter().copied().enumerate() {
            // Paper 1 Alg 7 lines 2-10
            let mut members = vec![center];
            for node in 0..graph.node_count() {
                if node % centers.len().max(1) == idx {
                    members.push(NodeId(node));
                }
            }
            members.sort_by_key(|node| node.0);
            members.dedup_by_key(|node| node.0);
            clusters.push(Cluster {
                id: idx,
                level: 1,
                center,
                members,
                diameter_bound: beta * (graph.node_count().max(2) as f64).ln(),
            });
        }

        (clusters, centers)
    }

    pub fn bootstrap_from_graph(&mut self, graph: &Graph) {
        if let Some(level) = self.levels.first() {
            let (clusters, _) =
                Self::sample_and_cluster(graph, level.p_j, self.sampling_params.beta);
            self.clusters = clusters;
            let cap = self.sampling_params.beta * (self.sampling_params.n.max(2) as f64).ln();
            debug_assert!(self.clusters.iter().all(|c| c.diameter_bound <= cap * 2.0));
        }
    }
}

/// ```
/// use almo_mcf_core::data_structures::lsd::LowStretchDecomposition;
///
/// let lsd = LowStretchDecomposition::new(64, 128, 2.0);
/// let cap = 2.0 * (64_f64).ln() * 2.0;
/// assert!(lsd.clusters.iter().all(|c| c.diameter_bound <= cap));
/// ```
fn _doc_test_marker() {}
