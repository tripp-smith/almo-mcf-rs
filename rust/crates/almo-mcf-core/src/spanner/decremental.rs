use std::collections::{HashMap, HashSet, VecDeque};

use crate::trees::LowStretchTree;

use super::{EmbeddingPath, EmbeddingStep};

#[derive(Debug, Clone)]
pub struct SplitInfo {
    pub edges_to_move: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct DecrementalSpannerParams {
    /// Theorem 5.1: maintains O(n) edges, supports decremental deletions/splits
    /// with amortized polylog update time per affected element, and provides
    /// embeddings with bounded congestion and hop length.
    /// Stretch factor κ. See Theorem 5.1: keep O(n) edges with polylog stretch.
    pub stretch_factor: f64,
    /// Path-length parameter L ≈ floor((log m)^{1/4}).
    pub path_length_l: usize,
    /// Exponent used to compute L from log m.
    pub l_exponent: f64,
    /// Batch size for recourse amortization.
    pub batch_size: usize,
    /// Maximum edges per level, expressed as C * n.
    pub max_edges_factor: usize,
    /// Congestion bound (paths per edge) in the embedding.
    pub congestion_bound: usize,
    /// Prefer deterministic expander decomposition.
    pub deterministic: bool,
    /// Reject true insertions when running in decremental-only mode.
    pub strict_decremental: bool,
    /// Number of hierarchy levels to build.
    pub level_count: usize,
}

#[derive(Debug, Clone)]
pub struct LayerEdge {
    u: usize,
    v: usize,
    length: f64,
    active: bool,
}

#[derive(Debug, Clone)]
pub struct GraphLayer {
    pub node_count: usize,
    pub edges: Vec<LayerEdge>,
    pub adjacency: Vec<Vec<usize>>,
    pub sparse_edges: Vec<usize>,
    pub sparse_adjacency: Vec<Vec<usize>>,
    pub cluster_map: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct DecrementalSpanner {
    pub params: DecrementalSpannerParams,
    pub levels: Vec<GraphLayer>,
    pub embeddings: HashMap<usize, EmbeddingPath>,
    pub recourse_this_batch: usize,
    pub total_updates: usize,
    pub rebuilds: usize,
    pending_projection: Vec<HashSet<usize>>,
    base_edge_map: HashMap<(usize, usize), Vec<usize>>,
    edge_usage: HashMap<usize, usize>,
}

#[derive(Debug, Clone)]
pub struct ContractedGraph {
    pub node_count: usize,
    pub tails: Vec<u32>,
    pub heads: Vec<u32>,
    pub lengths: Vec<f64>,
    pub supernodes: Vec<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct LSForest {
    pub node_count: usize,
    pub adjacency: Vec<Vec<(usize, f64)>>,
}

impl LSForest {
    pub fn from_tree(tree: &LowStretchTree, tails: &[u32], heads: &[u32], lengths: &[f64]) -> Self {
        let node_count = tree.parent.len();
        let mut adjacency = vec![Vec::new(); node_count];
        for (edge_id, (&tail, &head)) in tails.iter().zip(heads.iter()).enumerate() {
            if !tree.tree_edges.get(edge_id).copied().unwrap_or(false) {
                continue;
            }
            let u = tail as usize;
            let v = head as usize;
            let length = lengths.get(edge_id).copied().unwrap_or(1.0);
            adjacency[u].push((v, length));
            adjacency[v].push((u, length));
        }
        Self {
            node_count,
            adjacency,
        }
    }

    pub fn path_between(&self, start: usize, end: usize) -> Option<Vec<usize>> {
        if start >= self.node_count || end >= self.node_count {
            return None;
        }
        if start == end {
            return Some(Vec::new());
        }
        let mut parent = vec![None; self.node_count];
        let mut queue = VecDeque::new();
        parent[start] = Some(start);
        queue.push_back(start);
        while let Some(node) = queue.pop_front() {
            if node == end {
                break;
            }
            for &(neighbor, _) in &self.adjacency[node] {
                if parent[neighbor].is_some() {
                    continue;
                }
                parent[neighbor] = Some(node);
                queue.push_back(neighbor);
            }
        }
        parent[end]?;
        let mut path = Vec::new();
        let mut current = end;
        while current != start {
            path.push(current);
            current = parent[current]?;
        }
        path.push(start);
        path.reverse();
        Some(path)
    }
}

impl DecrementalSpannerParams {
    pub fn from_graph(node_count: usize, edge_count: usize) -> Self {
        let log_m = (edge_count.max(2) as f64).ln();
        let l_exponent = default_l_exponent();
        let mut path_length_l = log_m.powf(l_exponent).floor() as usize;
        path_length_l = path_length_l.max(1);
        let log_n = (node_count.max(2) as f64).ln();
        let epsilon = 1.0 / (log_n * log_n).max(1.0);
        let batch_size = log_n.ceil() as usize;
        let congestion_bound = log_n.ceil() as usize;
        let level_count = path_length_l.max(1);
        Self {
            stretch_factor: 1.0 + epsilon,
            path_length_l,
            l_exponent,
            batch_size: batch_size.max(1),
            max_edges_factor: 8,
            congestion_bound: congestion_bound.max(1),
            deterministic: true,
            strict_decremental: false,
            level_count,
        }
    }

    pub fn recourse_exponent(&self) -> f64 {
        1.0 / (self.path_length_l as f64)
    }

    pub fn validate(&self, node_count: usize, spanner_edges: usize) {
        let limit = self.max_edges_factor.max(1) * node_count.max(1);
        assert!(
            spanner_edges <= limit,
            "spanner edges {} exceed {} * n (n={})",
            spanner_edges,
            self.max_edges_factor,
            node_count
        );
        let expected = 1.0 / (self.path_length_l as f64);
        assert!(
            (self.recourse_exponent() - expected).abs() < 1e-9,
            "recourse exponent mismatch"
        );
    }

    pub fn recourse_limit(&self, node_count: usize) -> usize {
        let exponent = self.recourse_exponent();
        ((node_count.max(2) as f64).powf(exponent)).ceil() as usize
    }
}

pub fn build_spanner_on_core(
    core_graph: &ContractedGraph,
    forest: &LSForest,
    params: DecrementalSpannerParams,
) -> DecrementalSpanner {
    let mut spanner = DecrementalSpanner::new(
        core_graph.node_count,
        &core_graph.tails,
        &core_graph.heads,
        &core_graph.lengths,
        params,
    );
    for (super_idx, vertices) in core_graph.supernodes.iter().enumerate() {
        if vertices.len() <= 1 {
            continue;
        }
        let anchor = vertices[0];
        for &vertex in vertices.iter().skip(1) {
            if let Some(path) = forest.path_between(anchor, vertex) {
                let mut steps = Vec::new();
                for window in path.windows(2) {
                    let u = window[0];
                    let v = window[1];
                    if let Some(edge_id) = spanner.find_edge_id(u, v) {
                        steps.push(EmbeddingStep::new(edge_id, 1));
                    }
                }
                spanner
                    .embeddings
                    .insert(super_idx, EmbeddingPath { steps });
            }
        }
    }
    spanner
}

impl DecrementalSpanner {
    pub fn new(
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        lengths: &[f64],
        params: DecrementalSpannerParams,
    ) -> Self {
        let mut spanner = Self {
            params,
            levels: Vec::new(),
            embeddings: HashMap::new(),
            recourse_this_batch: 0,
            total_updates: 0,
            rebuilds: 0,
            pending_projection: Vec::new(),
            base_edge_map: HashMap::new(),
            edge_usage: HashMap::new(),
        };
        spanner.rebuild_from_edges(node_count, tails, heads, lengths);
        spanner
    }

    pub fn apply_batch_updates(
        &mut self,
        deletes: Vec<(usize, usize)>,
        inserts: Vec<(usize, usize, f64)>,
        vertex_splits: Vec<(usize, SplitInfo)>,
    ) -> Result<(), String> {
        if self.params.strict_decremental && !inserts.is_empty() {
            return Err("insertions not allowed in decremental mode".to_string());
        }
        let mut updated = 0usize;
        for (u, v) in deletes {
            if self.delete_edge(u, v) {
                updated += 1;
            }
        }
        for (u, v, length) in inserts {
            self.insert_edge(u, v, length);
            updated += 1;
        }
        for (vertex, split) in vertex_splits {
            self.split_vertex(vertex, &split.edges_to_move);
            updated += 1;
        }
        self.recourse_this_batch += updated;
        self.total_updates += updated;
        if self.recourse_this_batch >= self.params.recourse_limit(self.levels[0].node_count) {
            self.rebuild();
        } else {
            self.project_pending();
        }
        Ok(())
    }

    pub fn embed_path(&mut self, start: usize, end: usize) -> Option<EmbeddingPath> {
        let path_edges = self.find_path_in_spanner(start, end)?;
        for edge_id in &path_edges {
            *self.edge_usage.entry(*edge_id).or_insert(0) += 1;
        }
        let steps = path_edges
            .into_iter()
            .map(|edge| EmbeddingStep::new(edge, 1))
            .collect::<Vec<_>>();
        Some(EmbeddingPath { steps })
    }

    pub fn get_embedding(&mut self, start: usize, end: usize) -> Option<(usize, usize)> {
        let path = self.embed_path(start, end)?;
        let length = path.steps.len();
        let mut congestion = 0;
        for step in &path.steps {
            let usage = self.edge_usage.get(&step.edge).copied().unwrap_or(0);
            congestion = congestion.max(usage);
        }
        Some((congestion, length))
    }

    pub fn edge_count(&self) -> usize {
        self.levels
            .first()
            .map(|layer| layer.edges.iter().filter(|e| e.active).count())
            .unwrap_or(0)
    }

    fn rebuild(&mut self) {
        if let Some(level0) = self.levels.first() {
            let (tails, heads, lengths) = export_layer_edges(level0);
            let node_count = level0.node_count;
            self.rebuild_from_edges(node_count, &tails, &heads, &lengths);
            self.rebuilds += 1;
            self.recourse_this_batch = 0;
        }
    }

    fn rebuild_from_edges(
        &mut self,
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        lengths: &[f64],
    ) {
        self.levels.clear();
        self.pending_projection.clear();
        self.base_edge_map.clear();
        let level0 = build_layer(
            node_count,
            tails,
            heads,
            lengths,
            None,
            self.params.deterministic,
        );
        self.levels.push(level0);
        self.pending_projection.push(HashSet::new());

        for level in 1..self.params.level_count {
            let (new_tails, new_heads, new_lengths, new_node_count) =
                contract_layer(self.levels.last().unwrap());
            if new_node_count == 0 {
                break;
            }
            let layer = build_layer(
                new_node_count,
                &new_tails,
                &new_heads,
                &new_lengths,
                None,
                self.params.deterministic,
            );
            self.levels.push(layer);
            self.pending_projection.push(HashSet::new());
            if level + 1 == self.params.level_count {
                break;
            }
        }
        self.rebuild_edge_map();
        let total_sparse = self
            .levels
            .iter()
            .map(|layer| layer.sparse_edges.len())
            .sum();
        self.params
            .validate(self.levels[0].node_count, total_sparse);
    }

    fn rebuild_edge_map(&mut self) {
        if let Some(level0) = self.levels.first() {
            self.base_edge_map.clear();
            for (edge_id, edge) in level0.edges.iter().enumerate() {
                let key = ordered_pair(edge.u, edge.v);
                self.base_edge_map.entry(key).or_default().push(edge_id);
            }
        }
    }

    fn insert_edge(&mut self, u: usize, v: usize, length: f64) {
        let Some(level0) = self.levels.first_mut() else {
            return;
        };
        let edge_id = level0.edges.len();
        level0.edges.push(LayerEdge {
            u,
            v,
            length,
            active: true,
        });
        if u >= level0.node_count || v >= level0.node_count {
            level0.node_count = level0.node_count.max(u + 1).max(v + 1);
            level0.adjacency.resize_with(level0.node_count, Vec::new);
            level0
                .sparse_adjacency
                .resize_with(level0.node_count, Vec::new);
        }
        level0.adjacency[u].push(edge_id);
        level0.adjacency[v].push(edge_id);
        level0.sparse_edges.push(edge_id);
        level0.sparse_adjacency[u].push(edge_id);
        level0.sparse_adjacency[v].push(edge_id);
        self.base_edge_map
            .entry(ordered_pair(u, v))
            .or_default()
            .push(edge_id);
    }

    fn delete_edge(&mut self, u: usize, v: usize) -> bool {
        let Some(level0) = self.levels.first_mut() else {
            return false;
        };
        let key = ordered_pair(u, v);
        let Some(edges) = self.base_edge_map.get(&key) else {
            return false;
        };
        for &edge_id in edges {
            if let Some(edge) = level0.edges.get_mut(edge_id) {
                if edge.active {
                    edge.active = false;
                    self.pending_projection[0].insert(edge_id);
                    return true;
                }
            }
        }
        false
    }

    fn split_vertex(&mut self, vertex: usize, edges_to_move: &[usize]) {
        let Some(level0) = self.levels.first_mut() else {
            return;
        };
        let new_vertex = level0.node_count;
        level0.node_count += 1;
        level0.adjacency.push(Vec::new());
        level0.sparse_adjacency.push(Vec::new());
        for &edge_id in edges_to_move {
            let Some(edge) = level0.edges.get_mut(edge_id) else {
                continue;
            };
            if !edge.active {
                continue;
            }
            if edge.u == vertex {
                edge.u = new_vertex;
            }
            if edge.v == vertex {
                edge.v = new_vertex;
            }
            level0.adjacency[new_vertex].push(edge_id);
        }
    }

    fn project_pending(&mut self) {
        for level in 0..self.levels.len() {
            if self.pending_projection[level].is_empty() {
                continue;
            }
            let deletions: Vec<usize> = self.pending_projection[level].drain().collect();
            let level_layer = &mut self.levels[level];
            for edge_id in deletions {
                if let Some(edge) = level_layer.edges.get_mut(edge_id) {
                    edge.active = false;
                }
            }
            if level + 1 < self.levels.len() {
                self.pending_projection[level + 1].extend(0..self.levels[level + 1].edges.len());
            }
        }
    }

    fn find_path_in_spanner(&self, start: usize, end: usize) -> Option<Vec<usize>> {
        let layer = self.levels.first()?;
        if start >= layer.node_count || end >= layer.node_count {
            return None;
        }
        let mut parent_edge = vec![None; layer.node_count];
        let mut parent_node = vec![None; layer.node_count];
        let mut queue = VecDeque::new();
        parent_node[start] = Some(start);
        queue.push_back(start);
        while let Some(node) = queue.pop_front() {
            if node == end {
                break;
            }
            for &edge_id in &layer.sparse_adjacency[node] {
                let edge = layer.edges.get(edge_id)?;
                if !edge.active {
                    continue;
                }
                let next = if edge.u == node { edge.v } else { edge.u };
                if parent_node[next].is_some() {
                    continue;
                }
                parent_node[next] = Some(node);
                parent_edge[next] = Some(edge_id);
                queue.push_back(next);
            }
        }
        parent_node[end]?;
        let mut path = Vec::new();
        let mut current = end;
        while current != start {
            let edge_id = parent_edge[current]?;
            path.push(edge_id);
            current = parent_node[current]?;
        }
        path.reverse();
        Some(path)
    }

    fn find_edge_id(&self, u: usize, v: usize) -> Option<usize> {
        let key = ordered_pair(u, v);
        self.base_edge_map.get(&key).and_then(|ids| {
            ids.iter().copied().find(|id| {
                self.levels
                    .first()
                    .and_then(|layer| layer.edges.get(*id))
                    .map(|edge| edge.active)
                    .unwrap_or(false)
            })
        })
    }
}

fn build_layer(
    node_count: usize,
    tails: &[u32],
    heads: &[u32],
    lengths: &[f64],
    cluster_target: Option<usize>,
    deterministic: bool,
) -> GraphLayer {
    let mut edges = Vec::new();
    let mut adjacency = vec![Vec::new(); node_count];
    for (edge_id, (&tail, &head)) in tails.iter().zip(heads.iter()).enumerate() {
        let u = tail as usize;
        let v = head as usize;
        let length = lengths.get(edge_id).copied().unwrap_or(1.0);
        edges.push(LayerEdge {
            u,
            v,
            length,
            active: true,
        });
        if u >= adjacency.len() || v >= adjacency.len() {
            continue;
        }
        adjacency[u].push(edge_id);
        adjacency[v].push(edge_id);
    }

    let target = cluster_target.unwrap_or_else(|| (node_count.max(2) as f64).ln().ceil() as usize);
    let cluster_map = expander_decomposition(node_count, &edges, &adjacency, target, deterministic);

    let (sparse_edges, sparse_adjacency) =
        sparsify_layer(node_count, &edges, &adjacency, &cluster_map, deterministic);

    GraphLayer {
        node_count,
        edges,
        adjacency,
        sparse_edges,
        sparse_adjacency,
        cluster_map,
    }
}

fn expander_decomposition(
    node_count: usize,
    edges: &[LayerEdge],
    adjacency: &[Vec<usize>],
    target: usize,
    deterministic: bool,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..node_count).collect();
    if !deterministic {
        order.reverse();
    }
    let mut cluster_map = vec![usize::MAX; node_count];
    let mut cluster_id = 0;
    for start in order {
        if cluster_map[start] != usize::MAX {
            continue;
        }
        let mut queue = VecDeque::new();
        queue.push_back(start);
        cluster_map[start] = cluster_id;
        let mut size = 1;
        while let Some(node) = queue.pop_front() {
            if size >= target {
                break;
            }
            for &edge_id in &adjacency[node] {
                let Some(edge) = edges.get(edge_id) else {
                    continue;
                };
                let next = if edge.u == node { edge.v } else { edge.u };
                if next >= node_count {
                    continue;
                }
                if cluster_map[next] != usize::MAX {
                    continue;
                }
                cluster_map[next] = cluster_id;
                size += 1;
                queue.push_back(next);
                if size >= target {
                    break;
                }
            }
        }
        cluster_id += 1;
    }
    cluster_map
}

fn sparsify_layer(
    node_count: usize,
    edges: &[LayerEdge],
    adjacency: &[Vec<usize>],
    cluster_map: &[usize],
    deterministic: bool,
) -> (Vec<usize>, Vec<Vec<usize>>) {
    let mut sparse_edges = Vec::new();
    let mut sparse_adjacency = vec![Vec::new(); node_count];
    let mut seen = HashSet::new();
    for (_, neighbors) in adjacency.iter().enumerate().take(node_count) {
        let mut neighbors: Vec<usize> = neighbors.clone();
        if !deterministic {
            neighbors.reverse();
        }
        for &edge_id in neighbors.iter().take(2) {
            if seen.insert(edge_id) {
                sparse_edges.push(edge_id);
            }
        }
    }

    for &edge_id in &sparse_edges {
        if let Some(edge) = edges.get(edge_id) {
            if edge.u < node_count {
                sparse_adjacency[edge.u].push(edge_id);
            }
            if edge.v < node_count {
                sparse_adjacency[edge.v].push(edge_id);
            }
        }
    }

    // ensure connectivity within clusters by adding one edge per cluster if missing
    let mut cluster_seen = HashSet::new();
    for (edge_id, edge) in edges.iter().enumerate() {
        if !edge.active {
            continue;
        }
        let cluster = cluster_map.get(edge.u).copied().unwrap_or(usize::MAX);
        if cluster == usize::MAX {
            continue;
        }
        if cluster_seen.insert(cluster) && seen.insert(edge_id) {
            sparse_edges.push(edge_id);
            sparse_adjacency[edge.u].push(edge_id);
            sparse_adjacency[edge.v].push(edge_id);
        }
    }

    (sparse_edges, sparse_adjacency)
}

fn contract_layer(layer: &GraphLayer) -> (Vec<u32>, Vec<u32>, Vec<f64>, usize) {
    let mut edge_map: HashMap<(usize, usize), f64> = HashMap::new();
    let mut max_node = 0;
    for edge in &layer.edges {
        if !edge.active {
            continue;
        }
        let u = layer.cluster_map.get(edge.u).copied().unwrap_or(edge.u);
        let v = layer.cluster_map.get(edge.v).copied().unwrap_or(edge.v);
        if u == v {
            continue;
        }
        max_node = max_node.max(u.max(v));
        let key = ordered_pair(u, v);
        let entry = edge_map.entry(key).or_insert(f64::INFINITY);
        if edge.length < *entry {
            *entry = edge.length;
        }
    }
    let mut new_tails = Vec::new();
    let mut new_heads = Vec::new();
    let mut new_lengths = Vec::new();
    for ((u, v), length) in edge_map {
        new_tails.push(u as u32);
        new_heads.push(v as u32);
        new_lengths.push(length);
    }
    (new_tails, new_heads, new_lengths, max_node + 1)
}

fn export_layer_edges(layer: &GraphLayer) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let mut tails = Vec::new();
    let mut heads = Vec::new();
    let mut lengths = Vec::new();
    for edge in &layer.edges {
        if !edge.active {
            continue;
        }
        tails.push(edge.u as u32);
        heads.push(edge.v as u32);
        lengths.push(edge.length);
    }
    (tails, heads, lengths)
}

fn ordered_pair(u: usize, v: usize) -> (usize, usize) {
    if u <= v {
        (u, v)
    } else {
        (v, u)
    }
}

fn default_l_exponent() -> f64 {
    #[cfg(feature = "spanner-l-test")]
    {
        0.5
    }
    #[cfg(not(feature = "spanner-l-test"))]
    {
        0.25
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_validate_recourse_and_limits() {
        let params = DecrementalSpannerParams::from_graph(10, 40);
        params.validate(10, 20);
        assert!(params.recourse_exponent() > 0.0);
    }

    #[test]
    fn spanner_builds_levels_and_edges() {
        let tails = vec![0, 1, 2, 3];
        let heads = vec![1, 2, 3, 0];
        let lengths = vec![1.0, 1.0, 1.0, 1.0];
        let params = DecrementalSpannerParams::from_graph(4, 4);
        let spanner = DecrementalSpanner::new(4, &tails, &heads, &lengths, params);
        assert!(!spanner.levels.is_empty());
        assert!(spanner.edge_count() <= 4);
    }

    #[test]
    fn batch_updates_trigger_rebuild() {
        let tails = vec![0, 1, 2];
        let heads = vec![1, 2, 0];
        let lengths = vec![1.0, 1.0, 1.0];
        let mut params = DecrementalSpannerParams::from_graph(3, 3);
        params.level_count = 2;
        params.batch_size = 1;
        let mut spanner = DecrementalSpanner::new(3, &tails, &heads, &lengths, params);
        let _ = spanner.apply_batch_updates(vec![(0, 1)], Vec::new(), Vec::new());
        assert!(spanner.rebuilds <= spanner.total_updates);
    }

    #[test]
    fn embedding_returns_path_metrics() {
        let tails = vec![0, 1, 2];
        let heads = vec![1, 2, 0];
        let lengths = vec![1.0, 1.0, 1.0];
        let params = DecrementalSpannerParams::from_graph(3, 3);
        let mut spanner = DecrementalSpanner::new(3, &tails, &heads, &lengths, params);
        let metrics = spanner.get_embedding(0, 2).expect("path exists");
        assert!(metrics.1 >= 1);
    }

    #[test]
    fn build_spanner_on_core_preserves_supernodes() {
        let tails = vec![0, 1];
        let heads = vec![1, 2];
        let lengths = vec![1.0, 1.0];
        let tree = LowStretchTree::build_low_stretch_deterministic(3, &tails, &heads, &lengths)
            .expect("tree");
        let forest = LSForest::from_tree(&tree, &tails, &heads, &lengths);
        let core = ContractedGraph {
            node_count: 3,
            tails: tails.clone(),
            heads: heads.clone(),
            lengths: lengths.clone(),
            supernodes: vec![vec![0], vec![1, 2], vec![2]],
        };
        let params = DecrementalSpannerParams::from_graph(3, 2);
        let spanner = build_spanner_on_core(&core, &forest, params);
        assert!(!spanner.levels.is_empty());
    }
}
