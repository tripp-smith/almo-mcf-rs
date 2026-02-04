use crate::McfOptions;

pub type SolverOptions = McfOptions;

#[derive(Debug, Clone)]
pub struct SpannerConfig {
    pub stretch: f64,
    pub size_bound: usize,
    pub congestion_bound: f64,
    pub path_length_bound: f64,
    pub recourse_per_batch: f64,
    pub epsilon: f64,
    pub l: usize,
    pub level_count: usize,
    /// When true, enforce deterministic sparsification and clustering.
    pub deterministic: bool,
    /// Optional deterministic seed for tie-breaking hashes.
    pub deterministic_seed: Option<u64>,
}

impl SpannerConfig {
    pub fn from_graph(node_count: usize, edge_count: usize, opts: Option<&SolverOptions>) -> Self {
        let m = edge_count.max(2) as f64;
        let n = node_count.max(2) as f64;
        let log_m = m.ln();
        let log_n = n.ln();
        let l = log_m.powf(0.25).floor().max(1.0) as usize;
        let epsilon = opts
            .map(|options| options.approx_factor.abs().max(1e-6))
            .unwrap_or_else(|| 1.0 / (log_n * log_n).max(1.0));
        let level_count = ((log_n.sqrt()).ceil() as usize).max(1);
        let stretch = (log_n.powi(2).max(1.0)) * (1.0 + epsilon);
        let size_bound = Self::size_bound_with_factor::<10>(node_count, log_n);
        let congestion_bound = log_n.max(1.0);
        let path_length_bound = log_n.max(1.0);
        let recourse_per_batch = n.powf(1.0 / (l as f64));
        let deterministic = opts.map(|options| options.deterministic).unwrap_or(false);
        let deterministic_seed = opts.and_then(|options| options.deterministic_seed);
        Self {
            stretch,
            size_bound: size_bound.max(1),
            congestion_bound,
            path_length_bound,
            recourse_per_batch,
            epsilon,
            l,
            level_count,
            deterministic,
            deterministic_seed,
        }
    }

    pub fn recourse_limit(&self, node_count: usize) -> usize {
        let n = node_count.max(2) as f64;
        n.powf(1.0 / (self.l as f64)).ceil() as usize
    }

    pub fn size_bound_with_factor<const MULT: usize>(node_count: usize, log_n: f64) -> usize {
        let n = node_count.max(1) as f64;
        (MULT as f64 * n * log_n.max(1.0)).ceil().max(1.0) as usize
    }
}
