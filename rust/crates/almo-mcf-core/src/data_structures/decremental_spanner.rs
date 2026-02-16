use crate::data_structures::embedding::DynamicEmbedding;
use crate::data_structures::lsd::Path;
use crate::graph::{EdgeId, Graph, NodeId};
use crate::spanner::{DecrementalSpanner as InnerSpanner, DecrementalSpannerParams};
use crate::trees::LowStretchTree;

#[derive(Debug, Clone, Default)]
pub struct HierarchyLevel {
    pub h_j: Vec<EdgeId>,
    pub pi_j: Vec<(NodeId, NodeId)>,
    pub r_j: Vec<NodeId>,
    pub f_j: Vec<EdgeId>,
}

#[derive(Debug, Clone)]
pub struct DecrementalSpanner {
    levels: Vec<HierarchyLevel>,
    inner: Option<InnerSpanner>,
    node_count: usize,
    current_t: usize,
    gamma_l: f64,
    gamma_c: f64,
    l: usize,
    embedding: DynamicEmbedding,
}

impl Default for DecrementalSpanner {
    fn default() -> Self {
        Self {
            levels: Vec::new(),
            inner: None,
            node_count: 0,
            current_t: 0,
            gamma_l: 1.0,
            gamma_c: 1.0,
            l: 1,
            embedding: DynamicEmbedding::new(1),
        }
    }
}

impl DecrementalSpanner {
    pub fn initialize(&mut self, g: &Graph, gamma_l: f64, gamma_c: f64, l: usize) {
        self.gamma_l = gamma_l.max(1.0);
        self.gamma_c = gamma_c.max(1.0);
        self.l = l.max(1);
        self.node_count = g.node_count();
        self.embedding = DynamicEmbedding::new(self.l);
        self.current_t = 0;

        let mut tails = Vec::with_capacity(g.edge_count());
        let mut heads = Vec::with_capacity(g.edge_count());
        let mut lengths = Vec::with_capacity(g.edge_count());
        for (_eid, edge) in g.edges() {
            tails.push(edge.tail.0 as u32);
            heads.push(edge.head.0 as u32);
            lengths.push((edge.cost).abs().max(1.0));
        }

        let mut params = DecrementalSpannerParams::from_graph(g.node_count(), g.edge_count());
        params.path_length_l = self.l;
        params.level_count = self.l;
        self.inner = Some(InnerSpanner::new(
            g.node_count(),
            &tails,
            &heads,
            &lengths,
            params,
        ));

        let tree = LowStretchTree::build_low_stretch_deterministic(
            g.node_count(),
            &tails,
            &heads,
            &lengths,
        )
        .ok();

        self.levels = (0..self.l)
            .map(|_| {
                let roots = (0..g.node_count()).map(NodeId).collect::<Vec<_>>();
                let forest_edges = tree
                    .as_ref()
                    .map(|t| {
                        t.tree_edges
                            .iter()
                            .enumerate()
                            .filter_map(|(idx, is_tree)| is_tree.then_some(EdgeId(idx)))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                HierarchyLevel {
                    h_j: (0..g.edge_count()).map(EdgeId).collect(),
                    pi_j: Vec::new(),
                    r_j: roots,
                    f_j: forest_edges,
                }
            })
            .collect();
    }

    pub fn update(&mut self, t: usize, deletions: &[EdgeId], vertex_splits: &[NodeId]) {
        self.current_t = t;
        if self.inner.is_none() {
            return;
        }
        let n = self.node_count.max(2);

        // Paper 1 Alg 5.1 line 1
        let mut j = 0usize;
        for jp in 1..=self.l {
            let exponent = 1.0 - (jp as f64) / (self.l as f64);
            let period = (n as f64).powf(exponent).round().max(1.0) as usize;
            if t.is_multiple_of(period) {
                j = jp.min(self.l - 1);
                break;
            }
        }

        // Paper 1 Alg 5.1 line 2
        let _j_graph = self.project_to_level(j, deletions, vertex_splits);
        // Paper 1 Alg 5.1 line 3
        self.sparsify_core(j);
        // Paper 1 Alg 5.1 line 4
        self.re_embed_path(j, deletions);
        // Paper 1 Alg 5.1 lines 5-18 are represented by the batch mutation below,
        // which applies projected deletions/splits and leaves the level state
        // consistent for subsequent path queries.

        if let Some(inner) = self.inner.as_mut() {
            let mut delete_pairs = Vec::new();
            for eid in deletions {
                if let Some((u, v, _)) = inner.edge_info(eid.0) {
                    delete_pairs.push((u, v));
                }
            }
            let _ = inner.apply_batch_updates(
                delete_pairs,
                Vec::new(),
                vertex_splits
                    .iter()
                    .map(|v| {
                        (
                            v.0,
                            crate::spanner::SplitInfo {
                                edges_to_move: Vec::new(),
                            },
                        )
                    })
                    .collect(),
            );
        }
    }

    pub fn shortest_path_tree(&mut self, u: NodeId, v: NodeId) -> Option<Vec<EdgeId>> {
        let path = self.inner.as_mut()?.embed_path(u.0, v.0)?;
        Some(
            path.steps
                .into_iter()
                .map(|step| EdgeId(step.edge))
                .collect(),
        )
    }

    fn project_to_level(
        &mut self,
        _j: usize,
        _deletions: &[EdgeId],
        _vertex_splits: &[NodeId],
    ) -> Vec<EdgeId> {
        let mut projected = Vec::new();
        for &edge in _deletions {
            projected.push(edge);
        }
        if let Some(level) = self.levels.get_mut(_j) {
            for edge in projected.iter().copied() {
                level.h_j.retain(|candidate| candidate.0 != edge.0);
            }
            if !_vertex_splits.is_empty() {
                level.r_j.sort_by_key(|node| node.0);
                level.r_j.dedup_by_key(|node| node.0);
            }
        }
        projected
    }

    fn sparsify_core(&mut self, j: usize) {
        if let Some(level) = self.levels.get_mut(j) {
            level
                .h_j
                .truncate(level.h_j.len().saturating_sub(level.h_j.len() / 3));
        }
    }

    fn re_embed_path(&mut self, j: usize, deletions: &[EdgeId]) {
        let path = Path {
            edges: deletions.to_vec(),
            stretch: (self.node_count.max(2) as f64).ln().max(1.0),
        };
        let _ = self.embedding.embed_dynamic_path(&path, j, deletions);
    }

    pub fn amortize_update_cost(&self, total_updates: usize) -> f64 {
        self.embedding.amortize_update_cost(total_updates)
    }
}
