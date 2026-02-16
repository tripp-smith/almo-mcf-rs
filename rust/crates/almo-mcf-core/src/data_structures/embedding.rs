use crate::data_structures::lsd::Path;
use crate::graph::EdgeId;

#[derive(Debug, Clone, Default)]
pub struct EmbeddingMap {
    pub mapped_edges: Vec<EdgeId>,
    pub distortion: f64,
}

#[derive(Debug, Clone, Default)]
pub struct EmbeddedPath {
    pub lifted_edges: Vec<EdgeId>,
    pub distortion: f64,
}

#[derive(Debug, Clone, Default)]
pub struct DynamicEmbedding {
    pub embedding_maps: Vec<EmbeddingMap>,
    pub update_buffers: Vec<Vec<EdgeId>>,
    pub distortion_bounds: Vec<f64>,
    total_reembed_work: f64,
}

impl DynamicEmbedding {
    pub fn new(levels: usize) -> Self {
        let count = levels.max(1);
        Self {
            embedding_maps: vec![EmbeddingMap::default(); count],
            update_buffers: vec![Vec::new(); count],
            distortion_bounds: vec![1.0; count],
            total_reembed_work: 0.0,
        }
    }

    pub fn embed_dynamic_path(
        &mut self,
        path: &Path,
        level: usize,
        deletions: &[EdgeId],
    ) -> EmbeddedPath {
        let idx = level.min(self.embedding_maps.len().saturating_sub(1));
        let mut lifted = Vec::new();
        for edge in &path.edges {
            if !deletions.iter().any(|d| d.0 == edge.0) {
                lifted.push(*edge);
            }
        }
        // Paper 1 §2.7 notation: \hat{Π}(u,v) lifted embedding represented by `lifted`.
        let base = (path.stretch.max(1.0)).ln().max(1.0);
        let distortion = base + (deletions.len() as f64 + 1.0).ln();
        self.embedding_maps[idx] = EmbeddingMap {
            mapped_edges: lifted.clone(),
            distortion,
        };
        self.update_buffers[idx].extend_from_slice(deletions);
        self.distortion_bounds[idx] = self.distortion_bounds[idx].max(distortion);
        self.total_reembed_work += deletions.len() as f64 * base;

        EmbeddedPath {
            lifted_edges: lifted,
            distortion,
        }
    }

    pub fn amortize_update_cost(&self, total_updates: usize) -> f64 {
        self.total_reembed_work / total_updates.max(1) as f64
    }
}
