use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::graph::{Edge, EdgeId, NodeId};
use crate::spanner::construction::{build_level, contract_layer, greedy_cluster, sparsify_level};
use crate::spanner::construction::{LevelEdge, Spanner};
use crate::spanner::{DeterministicDynamicSpanner, DynamicSpanner, EmbeddingStep};
use crate::trees::{LowStretchTree, TreeBuildMode};

#[derive(Debug, Clone)]
pub struct InstabilityTracker {
    thresholds: Vec<f64>,
    accumulated: Vec<f64>,
}

impl InstabilityTracker {
    pub fn new(levels: usize, threshold: f64) -> Self {
        let count = levels.max(1);
        Self {
            thresholds: vec![threshold.max(0.0); count],
            accumulated: vec![0.0; count],
        }
    }

    pub fn record(&mut self, level: usize, amount: f64) {
        if level >= self.accumulated.len() {
            return;
        }
        self.accumulated[level] += amount;
    }

    pub fn exceeds_threshold(&self, level: usize) -> bool {
        self.accumulated.get(level).copied().unwrap_or(0.0)
            >= self.thresholds.get(level).copied().unwrap_or(f64::INFINITY)
    }

    pub fn set_threshold(&mut self, level: usize, threshold: f64) {
        if level >= self.thresholds.len() {
            return;
        }
        self.thresholds[level] = threshold.max(0.0);
    }

    pub fn should_rebuild_level(&self, level: usize, policy: &RebuildPolicy, iter: usize) -> bool {
        match policy {
            RebuildPolicy::Periodic { every } => *every > 0 && iter.is_multiple_of(*every),
            RebuildPolicy::Instability { threshold } => {
                self.accumulated.get(level).copied().unwrap_or(0.0) >= *threshold
            }
            RebuildPolicy::Hybrid { every, threshold } => {
                (*every > 0 && iter.is_multiple_of(*every))
                    || self.accumulated.get(level).copied().unwrap_or(0.0) >= *threshold
            }
        }
    }

    pub fn reset_level(&mut self, level: usize) {
        if level < self.accumulated.len() {
            self.accumulated[level] = 0.0;
        }
    }

    pub fn accumulated(&self) -> &[f64] {
        &self.accumulated
    }
}

impl Default for InstabilityTracker {
    fn default() -> Self {
        Self::new(1, 0.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum RebuildPolicy {
    Periodic { every: usize },
    Instability { threshold: f64 },
    Hybrid { every: usize, threshold: f64 },
}

#[derive(Debug, Clone, Copy)]
pub struct SpannerUpdates {
    pub changed_edges: usize,
    pub recourse: usize,
}

impl Spanner {
    pub fn batch_update(&mut self, inserts: &[Edge], deletes: &[Edge]) -> SpannerUpdates {
        let mut updated = 0usize;
        for edge in deletes {
            if self.delete_edge(edge.tail.0, edge.head.0) {
                updated += 1;
            }
        }
        for edge in inserts {
            self.insert_edge(edge.tail.0, edge.head.0);
            updated += 1;
        }
        self.recourse_this_batch += updated;
        self.total_updates += updated;
        let limit = self
            .config
            .recourse_limit(self.levels.first().map(|l| l.node_count).unwrap_or(1));
        if self.recourse_this_batch >= limit {
            self.rebuild_from_level0();
            self.rebuilds += 1;
            self.recourse_this_batch = 0;
        } else {
            self.process_pending();
        }
        SpannerUpdates {
            changed_edges: updated,
            recourse: self.recourse_this_batch,
        }
    }

    pub fn split_vertex(&mut self, vertex: NodeId, new_vertex: NodeId) -> bool {
        let new_id = new_vertex.0;
        let old_id = vertex.0;
        let mut changed = false;
        for level in &mut self.levels {
            if old_id >= level.node_count {
                continue;
            }
            if new_id >= level.node_count {
                level.node_count = new_id + 1;
                level.adjacency.resize_with(level.node_count, Vec::new);
                level
                    .sparse_adjacency
                    .resize_with(level.node_count, Vec::new);
                let cluster = level.cluster_map.get(old_id).copied().unwrap_or(0);
                while level.cluster_map.len() < level.node_count {
                    level.cluster_map.push(cluster);
                }
                changed = true;
            }
        }
        if changed {
            self.recourse_this_batch += 1;
        }
        changed
    }

    pub fn query_embedding(&mut self, edge_id: EdgeId) -> Vec<EdgeId> {
        self.query_embedding_steps(edge_id)
            .unwrap_or_default()
            .into_iter()
            .map(|step| EdgeId(step.edge))
            .collect()
    }

    fn insert_edge(&mut self, u: usize, v: usize) {
        let Some(level0) = self.levels.first_mut() else {
            return;
        };
        let edge_id = level0.edges.len();
        level0.edges.push(LevelEdge { u, v, active: true });
        if u >= level0.node_count || v >= level0.node_count {
            level0.node_count = level0.node_count.max(u + 1).max(v + 1);
            level0.adjacency.resize_with(level0.node_count, Vec::new);
            level0
                .sparse_adjacency
                .resize_with(level0.node_count, Vec::new);
            while level0.cluster_map.len() < level0.node_count {
                level0.cluster_map.push(level0.cluster_map.len());
            }
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
        self.pending_levels[0].insert(edge_id);
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
                    self.pending_levels[0].insert(edge_id);
                    return true;
                }
            }
        }
        false
    }

    fn process_pending(&mut self) {
        let mut heap = BinaryHeap::new();
        for level in 0..self.pending_levels.len() {
            if !self.pending_levels[level].is_empty() {
                heap.push(Reverse(level));
            }
        }
        while let Some(Reverse(level)) = heap.pop() {
            if self.pending_levels[level].is_empty() {
                continue;
            }
            let node_count = self.levels[level].node_count;
            let target = ((node_count.max(2) as f64).ln().ceil() as usize).max(1);
            let new_cluster = greedy_cluster(
                node_count,
                &self.levels[level].edges,
                &self.levels[level].adjacency,
                target,
                self.config.deterministic,
            );
            let size_bound = self.level_size_bound(node_count);
            let (new_sparse, new_sparse_adj) = sparsify_level(
                node_count,
                &self.levels[level].edges,
                &self.levels[level].adjacency,
                &new_cluster,
                size_bound,
                self.config.deterministic,
                self.config.deterministic_seed,
            );
            let changed = count_edge_changes(&self.levels[level].sparse_edges, &new_sparse);
            self.recourse_this_batch = self.recourse_this_batch.saturating_add(changed);
            self.levels[level].cluster_map = new_cluster;
            self.levels[level].sparse_edges = new_sparse;
            self.levels[level].sparse_adjacency = new_sparse_adj;
            self.pending_levels[level].clear();
            if level + 1 < self.levels.len() {
                self.pending_levels[level + 1].extend(0..self.levels[level + 1].edges.len());
                heap.push(Reverse(level + 1));
            }
        }
    }

    fn level_size_bound(&self, node_count: usize) -> usize {
        let base = self.levels.first().map(|l| l.node_count).unwrap_or(1);
        let ratio = (node_count as f64) / (base as f64);
        ((self.config.size_bound as f64) * ratio)
            .ceil()
            .max(node_count as f64) as usize
    }

    fn rebuild_from_level0(&mut self) {
        let Some(level0) = self.levels.first() else {
            return;
        };
        let edges = level0.edges.clone();
        let adjacency = level0.adjacency.clone();
        let node_count = level0.node_count;
        self.levels.clear();
        self.pending_levels.clear();
        let level = build_level(node_count, edges, adjacency, &self.config);
        self.levels.push(level);
        self.pending_levels.push(std::collections::HashSet::new());
        for _ in 1..self.config.level_count {
            let (next_edges, next_adj, next_nodes) = contract_layer(self.levels.last().unwrap());
            if next_nodes <= 1 {
                break;
            }
            let level = build_level(next_nodes, next_edges, next_adj, &self.config);
            self.levels.push(level);
            self.pending_levels.push(std::collections::HashSet::new());
        }
        self.rebuild_edge_map();
        self.embeddings.clear();
        let originals: Vec<(EdgeId, usize, usize)> = self
            .original_edges
            .iter()
            .map(|(&edge_id, &(u, v))| (edge_id, u, v))
            .collect();
        for (edge_id, u, v) in originals {
            if self.embed_edge_with_bfs(edge_id, u, v).is_none() {
                self.ensure_direct_edge(u, v, edge_id);
            }
        }
    }
}

fn count_edge_changes(old_edges: &[usize], new_edges: &[usize]) -> usize {
    let old: std::collections::HashSet<_> = old_edges.iter().copied().collect();
    let new: std::collections::HashSet<_> = new_edges.iter().copied().collect();
    let removed = old.difference(&new).count();
    let added = new.difference(&old).count();
    removed + added
}

fn ordered_pair(u: usize, v: usize) -> (usize, usize) {
    if u <= v {
        (u, v)
    } else {
        (v, u)
    }
}

impl DynamicSpanner {
    pub fn build_sparse_subgraph(
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        lengths: &[f64],
        gradients: &[f64],
        max_extra_per_node: usize,
        deterministic: bool,
    ) -> Self {
        let build_mode = if deterministic {
            TreeBuildMode::Deterministic
        } else {
            TreeBuildMode::Randomized
        };
        let tree =
            LowStretchTree::build_with_mode(node_count, tails, heads, lengths, build_mode, 0)
                .unwrap_or_else(|_| {
                    LowStretchTree::build(node_count, tails, heads, lengths, 0).unwrap()
                });
        let mut spanner = DynamicSpanner::new(node_count);
        let mut edge_to_spanner = vec![None; tails.len()];
        for (edge_id, (&tail, &head)) in tails.iter().zip(heads.iter()).enumerate() {
            if tree.tree_edges.get(edge_id).copied().unwrap_or(false) {
                let spanner_edge = spanner.insert_edge_with_values(
                    tail as usize,
                    head as usize,
                    lengths[edge_id].abs(),
                    gradients[edge_id],
                );
                edge_to_spanner[edge_id] = Some(spanner_edge);
                spanner.set_embedding(edge_id, vec![EmbeddingStep::new(spanner_edge, 1)]);
            }
        }

        let mut extras: Vec<(f64, usize)> = (0..tails.len())
            .filter(|&edge_id| edge_to_spanner[edge_id].is_none())
            .map(|edge_id| (lengths[edge_id].abs(), edge_id))
            .collect();
        extras.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut degree = vec![0usize; node_count];
        for (edge_id, (&tail, &head)) in tails.iter().zip(heads.iter()).enumerate() {
            if edge_to_spanner[edge_id].is_some() {
                degree[tail as usize] += 1;
                degree[head as usize] += 1;
            }
        }
        for (_, edge_id) in extras {
            let u = tails[edge_id] as usize;
            let v = heads[edge_id] as usize;
            if degree[u] >= max_extra_per_node && degree[v] >= max_extra_per_node {
                continue;
            }
            let spanner_edge =
                spanner.insert_edge_with_values(u, v, lengths[edge_id].abs(), gradients[edge_id]);
            edge_to_spanner[edge_id] = Some(spanner_edge);
            spanner.set_embedding(edge_id, vec![EmbeddingStep::new(spanner_edge, 1)]);
            degree[u] += 1;
            degree[v] += 1;
        }

        for edge_id in 0..tails.len() {
            if edge_to_spanner[edge_id].is_some() {
                continue;
            }
            let start = tails[edge_id] as usize;
            let end = heads[edge_id] as usize;
            if spanner.embed_edge_with_bfs(edge_id, start, end).is_none() {
                let spanner_edge = spanner.insert_edge_with_values(
                    start,
                    end,
                    lengths[edge_id].abs(),
                    gradients[edge_id],
                );
                spanner.set_embedding(edge_id, vec![EmbeddingStep::new(spanner_edge, 1)]);
            }
        }

        spanner
    }

    pub fn query_path(&self, start: usize, end: usize) -> Option<Vec<EmbeddingStep>> {
        if start >= self.node_count || end >= self.node_count || start == end {
            return None;
        }
        let mut visited = vec![false; self.node_count];
        let mut parent: Vec<Option<(usize, usize)>> = vec![None; self.node_count];
        let mut queue = std::collections::VecDeque::new();
        visited[start] = true;
        queue.push_back(start);
        while let Some(node) = queue.pop_front() {
            if node == end {
                break;
            }
            let Some(adjacent) = self.adjacency.get(node) else {
                continue;
            };
            for &edge_id in adjacent {
                let Some(edge) = self.edges.get(edge_id) else {
                    continue;
                };
                if !edge.active {
                    continue;
                }
                let next = if edge.u == node {
                    edge.v
                } else if edge.v == node {
                    edge.u
                } else {
                    continue;
                };
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
            let edge = self.edges.get(edge_id)?;
            let dir = if edge.u == prev && edge.v == curr {
                1
            } else {
                -1
            };
            steps_rev.push(EmbeddingStep::new(edge_id, dir));
            curr = prev;
        }
        steps_rev.reverse();
        Some(steps_rev)
    }

    pub fn get_spanner_edges(&self) -> Vec<(usize, usize)> {
        self.edges
            .iter()
            .filter(|edge| edge.active)
            .map(|edge| (edge.u, edge.v))
            .collect()
    }
}

impl DeterministicDynamicSpanner {
    pub fn update_edge(&mut self, edge_id: usize, length: f64, gradient: f64) -> bool {
        self.update_edge_values(edge_id, length, gradient)
    }

    pub fn query_path(&mut self, start: usize, end: usize) -> Option<Vec<EmbeddingStep>> {
        self.spanner.query_path(start, end)
    }

    pub fn get_spanner_edges(&self) -> Vec<(usize, usize)> {
        self.spanner.get_spanner_edges()
    }

    pub fn vertex_split_recursive(&mut self, vertex: usize, depth: usize) -> usize {
        let new_vertex = self.spanner.split_vertex(vertex);
        if depth > 0 {
            self.pending_updates = self.pending_updates.saturating_add(1);
            self.maintain();
        }
        new_vertex
    }

    pub fn sparsify_edges_with_embeddings(&mut self, max_edges: usize) {
        if self.spanner.edge_count <= max_edges {
            return;
        }
        let mut active_edges: Vec<usize> = self
            .spanner
            .edges
            .iter()
            .enumerate()
            .filter_map(|(edge_id, edge)| edge.active.then_some(edge_id))
            .collect();
        active_edges.sort_unstable();
        let excess = active_edges.len().saturating_sub(max_edges);
        for edge_id in active_edges.into_iter().rev().take(excess) {
            if let Some((u, v)) = self.spanner.edge_endpoints(edge_id) {
                if let Some(path) = self.spanner.query_path(u, v) {
                    self.spanner.set_embedding(edge_id, path);
                }
            }
            self.spanner.delete_edge(edge_id);
        }
    }
}
