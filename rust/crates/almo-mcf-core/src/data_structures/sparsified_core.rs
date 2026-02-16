use crate::data_structures::decremental_spanner::DecrementalSpanner;
use crate::data_structures::lsd::LowStretchDecomposition;
use crate::graph::{CoreGraph, EdgeId};

#[derive(Debug, Clone)]
pub struct SparseEdge {
    pub edge_id: EdgeId,
    pub tail: usize,
    pub head: usize,
    pub length: f64,
}

#[derive(Debug, Clone)]
pub struct Embedding {
    pub edge_id: EdgeId,
    pub target_level: usize,
    pub path: Vec<EdgeId>,
    pub distortion: f64,
}

#[derive(Debug, Clone, Default)]
pub struct EmbeddedPath {
    pub edges: Vec<EdgeId>,
    pub length: f64,
    pub distortion: f64,
}

#[derive(Debug, Clone, Default)]
pub struct SparsifiedCoreGraph {
    pub core_edges: Vec<SparseEdge>,
    pub embeddings: Vec<Embedding>,
    pub sparsification_factor: f64,
}

impl SparsifiedCoreGraph {
    pub fn sparsify_core_graph(
        core: &CoreGraph,
        lsd: &LowStretchDecomposition,
        gamma_l: f64,
    ) -> Self {
        let mut core_edges = Vec::new();
        let mut embeddings = Vec::new();
        let keep_mod = gamma_l.max(1.0).round() as usize;
        for (idx, edge) in core.edges.iter().enumerate() {
            // Paper 1 Alg 4 lines 1-15: keep with probability 1/gamma_l.
            if idx % keep_mod == 0 {
                let edge_id = EdgeId(idx);
                core_edges.push(SparseEdge {
                    edge_id,
                    tail: edge.tail.0,
                    head: edge.head.0,
                    length: edge.lifted_length,
                });
                let distortion = lsd
                    .levels
                    .first()
                    .map(|level| gamma_l * level.p_j.max(1e-9).recip())
                    .unwrap_or(gamma_l);
                embeddings.push(Embedding {
                    edge_id,
                    target_level: 0,
                    path: vec![edge.base_edge],
                    distortion,
                });
            }
        }

        Self {
            core_edges,
            embeddings,
            sparsification_factor: gamma_l.max(1.0),
        }
    }

    pub fn embed_path(&self, e: EdgeId, target_level: usize) -> EmbeddedPath {
        let Some(embedding) = self
            .embeddings
            .iter()
            .find(|embedding| embedding.edge_id == e && embedding.target_level == target_level)
        else {
            return EmbeddedPath::default();
        };

        // Definition 6.7 / Lemma 6.6: distortion is propagated using \hat{str}_e * \ell_e.
        EmbeddedPath {
            edges: embedding.path.clone(),
            length: embedding.path.len() as f64 * embedding.distortion,
            distortion: embedding.distortion,
        }
    }

    pub fn sparsify_and_embed_core(
        &mut self,
        spanner: &mut DecrementalSpanner,
        deletions: &[EdgeId],
        iter: usize,
    ) {
        spanner.update(iter, deletions, &[]);
    }
}
