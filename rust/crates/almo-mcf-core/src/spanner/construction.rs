use std::collections::{HashMap, HashSet, VecDeque};

use crate::graph::undirected::UndirectedView;
use crate::graph::{EdgeId, Graph, NodeId};
use crate::spanner::params::SpannerConfig;
use crate::spanner::{EmbeddingPath, EmbeddingStep};

#[derive(Debug, Clone)]
pub struct LevelEdge {
    pub u: usize,
    pub v: usize,
    pub active: bool,
}

#[derive(Debug, Clone)]
pub struct LevelGraph {
    pub node_count: usize,
    pub edges: Vec<LevelEdge>,
    pub adjacency: Vec<Vec<usize>>,
    pub sparse_edges: Vec<usize>,
    pub sparse_adjacency: Vec<Vec<usize>>,
    pub cluster_map: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Spanner {
    pub config: SpannerConfig,
    pub levels: Vec<LevelGraph>,
    pub embeddings: HashMap<EdgeId, EmbeddingPath>,
    pub recourse_this_batch: usize,
    pub total_updates: usize,
    pub rebuilds: usize,
    pub(crate) base_edge_map: HashMap<(usize, usize), Vec<usize>>,
    pub(crate) pending_levels: Vec<HashSet<usize>>,
    pub(crate) edge_usage: HashMap<usize, usize>,
    pub(crate) original_edges: HashMap<EdgeId, (usize, usize)>,
}

impl Spanner {
    pub fn edge_count(&self) -> usize {
        self.levels
            .first()
            .map(|layer| {
                layer
                    .sparse_edges
                    .iter()
                    .filter(|&&edge_id| layer.edges.get(edge_id).map(|e| e.active).unwrap_or(false))
                    .count()
            })
            .unwrap_or(0)
    }

    pub fn sparse_edge_ids(&self) -> Option<&[usize]> {
        self.levels
            .first()
            .map(|layer| layer.sparse_edges.as_slice())
    }

    pub fn edge_info(&self, edge_id: usize) -> Option<(usize, usize)> {
        let layer = self.levels.first()?;
        let edge = layer.edges.get(edge_id)?;
        if !edge.active {
            return None;
        }
        Some((edge.u, edge.v))
    }

    pub fn query_embedding_steps(&mut self, edge_id: EdgeId) -> Option<Vec<EmbeddingStep>> {
        if let Some(path) = self.embeddings.get(&edge_id) {
            return Some(path.steps.clone());
        }
        let (u, v) = self.edge_endpoints(edge_id)?;
        let steps = self.embed_edge_with_bfs(edge_id, u, v)?;
        Some(steps)
    }

    pub fn edge_endpoints(&self, edge_id: EdgeId) -> Option<(usize, usize)> {
        self.original_edges.get(&edge_id).copied()
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

    pub fn get_embedding_metrics(&mut self, start: usize, end: usize) -> Option<(usize, usize)> {
        let path = self.embed_path(start, end)?;
        let length = path.steps.len();
        let mut congestion = 0;
        for step in &path.steps {
            let usage = self.edge_usage.get(&step.edge).copied().unwrap_or(0);
            congestion = congestion.max(usage);
        }
        Some((congestion, length))
    }

    pub fn embed_edge_with_bfs(
        &mut self,
        original_edge: EdgeId,
        start: usize,
        end: usize,
    ) -> Option<Vec<EmbeddingStep>> {
        let layer = self.levels.first()?;
        if start >= layer.node_count || end >= layer.node_count {
            return None;
        }
        if start == end {
            return None;
        }
        let mut visited = vec![false; layer.node_count];
        let mut parent: Vec<Option<(usize, usize)>> = vec![None; layer.node_count];
        let mut queue = VecDeque::new();
        visited[start] = true;
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
                if visited[next] {
                    continue;
                }
                visited[next] = true;
                parent[next] = Some((node, edge_id));
                queue.push_back(next);
            }
        }
        if !visited[end] {
            return None;
        }
        let mut steps_rev: Vec<EmbeddingStep> = Vec::new();
        let mut curr = end;
        while curr != start {
            let (prev, edge_id) = parent[curr]?;
            let edge = layer.edges.get(edge_id)?;
            let dir = if edge.u == prev && edge.v == curr {
                1
            } else {
                -1
            };
            steps_rev.push(EmbeddingStep::new(edge_id, dir));
            curr = prev;
        }
        steps_rev.reverse();
        self.embeddings.insert(
            original_edge,
            EmbeddingPath {
                steps: steps_rev.clone(),
            },
        );
        Some(steps_rev)
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

    pub(crate) fn rebuild_edge_map(&mut self) {
        if let Some(level0) = self.levels.first() {
            self.base_edge_map.clear();
            for (edge_id, edge) in level0.edges.iter().enumerate() {
                let key = ordered_pair(edge.u, edge.v);
                self.base_edge_map.entry(key).or_default().push(edge_id);
            }
        }
    }
}

pub fn build_spanner(graph: &Graph, config: &SpannerConfig) -> Spanner {
    let node_count = graph.node_count();
    let (edges, adjacency) = build_base_layer(graph);
    let mut spanner = Spanner {
        config: config.clone(),
        levels: Vec::new(),
        embeddings: HashMap::new(),
        recourse_this_batch: 0,
        total_updates: 0,
        rebuilds: 0,
        base_edge_map: HashMap::new(),
        pending_levels: Vec::new(),
        edge_usage: HashMap::new(),
        original_edges: HashMap::new(),
    };
    let mut level0 = build_level(node_count, edges, adjacency, config);
    spanner.levels.push(level0);
    spanner.pending_levels.push(HashSet::new());

    for _ in 1..config.level_count {
        let (next_edges, next_adj, next_nodes) = contract_layer(spanner.levels.last().unwrap());
        if next_nodes <= 1 {
            break;
        }
        level0 = build_level(next_nodes, next_edges, next_adj, config);
        spanner.levels.push(level0);
        spanner.pending_levels.push(HashSet::new());
    }

    spanner.rebuild_edge_map();
    spanner.populate_embeddings(graph);
    spanner
}

impl Spanner {
    fn populate_embeddings(&mut self, graph: &Graph) {
        for (edge_id, edge) in graph.edges() {
            self.original_edges
                .insert(edge_id, (edge.tail.0, edge.head.0));
            let start = edge.tail.0;
            let end = edge.head.0;
            if let Some(path) = self.embed_edge_with_bfs(edge_id, start, end) {
                let _ = path;
            } else {
                self.ensure_direct_edge(start, end, edge_id);
            }
        }
    }

    pub(crate) fn ensure_direct_edge(&mut self, u: usize, v: usize, edge_id: EdgeId) {
        let Some(level0) = self.levels.first_mut() else {
            return;
        };
        let key = ordered_pair(u, v);
        let existing = self
            .base_edge_map
            .get(&key)
            .and_then(|ids| {
                ids.iter()
                    .find(|&&id| level0.edges.get(id).map(|e| e.active).unwrap_or(false))
            })
            .copied();
        let edge_ref = if let Some(existing) = existing {
            existing
        } else {
            let new_id = level0.edges.len();
            level0.edges.push(LevelEdge { u, v, active: true });
            level0.adjacency[u].push(new_id);
            level0.adjacency[v].push(new_id);
            level0.sparse_edges.push(new_id);
            level0.sparse_adjacency[u].push(new_id);
            level0.sparse_adjacency[v].push(new_id);
            self.base_edge_map.entry(key).or_default().push(new_id);
            new_id
        };
        self.embeddings.insert(
            edge_id,
            EmbeddingPath {
                steps: vec![EmbeddingStep::new(edge_ref, 1)],
            },
        );
    }
}

fn build_base_layer(graph: &Graph) -> (Vec<LevelEdge>, Vec<Vec<usize>>) {
    let node_count = graph.node_count();
    let mut edges = Vec::new();
    let mut adjacency = vec![Vec::new(); node_count];
    let mut seen = HashSet::new();
    let view = UndirectedView::new(graph);
    for node in 0..node_count {
        let Ok(neighbors) = view.neighbors(NodeId(node)) else {
            continue;
        };
        for neighbor in neighbors {
            let u = neighbor.from.0;
            let v = neighbor.to.0;
            let key = ordered_pair(u, v);
            if !seen.insert(key) {
                continue;
            }
            let edge_id = edges.len();
            edges.push(LevelEdge { u, v, active: true });
            adjacency[u].push(edge_id);
            adjacency[v].push(edge_id);
        }
    }
    (edges, adjacency)
}

pub(crate) fn build_level(
    node_count: usize,
    edges: Vec<LevelEdge>,
    mut adjacency: Vec<Vec<usize>>,
    config: &SpannerConfig,
) -> LevelGraph {
    if config.deterministic {
        sort_adjacency(&mut adjacency, &edges, config.deterministic_seed);
    }
    let target = ((node_count.max(2) as f64).ln().ceil() as usize).max(1);
    let cluster_map = greedy_cluster(node_count, &edges, &adjacency, target, config.deterministic);
    let (sparse_edges, sparse_adjacency) = sparsify_level(
        node_count,
        &edges,
        &adjacency,
        &cluster_map,
        config.size_bound,
        config.deterministic,
        config.deterministic_seed,
    );
    LevelGraph {
        node_count,
        edges,
        adjacency,
        sparse_edges,
        sparse_adjacency,
        cluster_map,
    }
}

pub(crate) fn greedy_cluster(
    node_count: usize,
    edges: &[LevelEdge],
    adjacency: &[Vec<usize>],
    target: usize,
    deterministic: bool,
) -> Vec<usize> {
    let mut degree: Vec<(usize, usize)> = adjacency
        .iter()
        .enumerate()
        .map(|(idx, neighbors)| (idx, neighbors.len()))
        .collect();
    if deterministic {
        degree.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    } else {
        degree.sort_by(|a, b| b.1.cmp(&a.1));
    }
    let mut cluster_map = vec![usize::MAX; node_count];
    let mut cluster_id = 0;
    for (start, _) in degree {
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
                if next >= node_count || cluster_map[next] != usize::MAX {
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

pub(crate) fn sparsify_level(
    node_count: usize,
    edges: &[LevelEdge],
    adjacency: &[Vec<usize>],
    cluster_map: &[usize],
    size_bound: usize,
    deterministic: bool,
    deterministic_seed: Option<u64>,
) -> (Vec<usize>, Vec<Vec<usize>>) {
    let mut sparse_edges = Vec::new();
    let mut sparse_adjacency = vec![Vec::new(); node_count];
    let mut seen = HashSet::new();

    for neighbors in adjacency.iter().take(node_count) {
        let mut ordered = neighbors.to_vec();
        if deterministic {
            ordered
                .sort_by_key(|&edge_id| deterministic_edge_key(edge_id, edges, deterministic_seed));
        }
        for &edge_id in ordered.iter().take(2) {
            if sparse_edges.len() >= size_bound {
                break;
            }
            if seen.insert(edge_id) {
                sparse_edges.push(edge_id);
            }
        }
    }

    let mut cluster_pairs = HashSet::new();
    let mut edge_ids: Vec<usize> = (0..edges.len()).collect();
    if deterministic {
        edge_ids.sort_by_key(|&edge_id| deterministic_edge_key(edge_id, edges, deterministic_seed));
    }
    for edge_id in edge_ids {
        let edge = &edges[edge_id];
        if sparse_edges.len() >= size_bound {
            break;
        }
        if !edge.active {
            continue;
        }
        let cu = cluster_map.get(edge.u).copied().unwrap_or(edge.u);
        let cv = cluster_map.get(edge.v).copied().unwrap_or(edge.v);
        if cu == cv {
            continue;
        }
        let key = ordered_pair(cu, cv);
        if cluster_pairs.insert(key) && seen.insert(edge_id) {
            sparse_edges.push(edge_id);
        }
    }

    let mut cluster_seen = HashSet::new();
    let mut edge_ids: Vec<usize> = (0..edges.len()).collect();
    if deterministic {
        edge_ids.sort_by_key(|&edge_id| deterministic_edge_key(edge_id, edges, deterministic_seed));
    }
    for edge_id in edge_ids {
        let edge = &edges[edge_id];
        if sparse_edges.len() >= size_bound {
            break;
        }
        if !edge.active {
            continue;
        }
        let cluster = cluster_map.get(edge.u).copied().unwrap_or(usize::MAX);
        if cluster == usize::MAX {
            continue;
        }
        if cluster_seen.insert(cluster) && seen.insert(edge_id) {
            sparse_edges.push(edge_id);
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
    if deterministic {
        for neighbors in sparse_adjacency.iter_mut() {
            neighbors
                .sort_by_key(|&edge_id| deterministic_edge_key(edge_id, edges, deterministic_seed));
        }
    }

    (sparse_edges, sparse_adjacency)
}

pub(crate) fn contract_layer(layer: &LevelGraph) -> (Vec<LevelEdge>, Vec<Vec<usize>>, usize) {
    let mut edge_map: HashMap<(usize, usize), usize> = HashMap::new();
    let mut max_node = 0;
    let mut edges = Vec::new();
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
        if edge_map.contains_key(&key) {
            continue;
        }
        let edge_id = edges.len();
        edge_map.insert(key, edge_id);
        edges.push(LevelEdge { u, v, active: true });
    }
    let node_count = max_node + 1;
    let mut adjacency = vec![Vec::new(); node_count.max(1)];
    for (edge_id, edge) in edges.iter().enumerate() {
        if edge.u < node_count {
            adjacency[edge.u].push(edge_id);
        }
        if edge.v < node_count {
            adjacency[edge.v].push(edge_id);
        }
    }
    (edges, adjacency, node_count.max(1))
}

fn ordered_pair(u: usize, v: usize) -> (usize, usize) {
    if u <= v {
        (u, v)
    } else {
        (v, u)
    }
}

fn sort_adjacency(
    adjacency: &mut [Vec<usize>],
    edges: &[LevelEdge],
    deterministic_seed: Option<u64>,
) {
    for neighbors in adjacency.iter_mut() {
        neighbors
            .sort_by_key(|&edge_id| deterministic_edge_key(edge_id, edges, deterministic_seed));
    }
}

fn deterministic_edge_key(
    edge_id: usize,
    edges: &[LevelEdge],
    deterministic_seed: Option<u64>,
) -> (usize, usize, u64) {
    let edge = &edges[edge_id];
    let (u, v) = ordered_pair(edge.u, edge.v);
    let seed = deterministic_seed.unwrap_or(0);
    (u, v, splitmix64(seed ^ edge_id as u64))
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
