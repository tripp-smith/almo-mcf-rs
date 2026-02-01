use std::collections::HashMap;
use std::collections::VecDeque;

use crate::trees::LowStretchTree;

pub mod oracle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EmbeddingStep {
    pub edge: usize,
    pub dir: i8,
}

impl EmbeddingStep {
    pub fn new(edge: usize, dir: i8) -> Self {
        Self { edge, dir }
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingPath {
    pub steps: Vec<EmbeddingStep>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmbeddingMetrics {
    pub total_length: f64,
    pub total_gradient: f64,
}

#[derive(Debug, Clone)]
struct SpannerEdge {
    u: usize,
    v: usize,
    active: bool,
    length: f64,
    gradient: f64,
}

#[derive(Debug, Default, Clone)]
pub struct DynamicSpanner {
    pub node_count: usize,
    pub edge_count: usize,
    edges: Vec<SpannerEdge>,
    adjacency: Vec<Vec<usize>>,
    embeddings: HashMap<usize, EmbeddingPath>,
}

#[derive(Debug, Clone)]
pub struct SpannerLevel {
    pub spanner: DynamicSpanner,
    pub tree: LowStretchTree,
}

#[derive(Debug, Clone)]
pub struct SpannerHierarchy {
    pub levels: Vec<SpannerLevel>,
    pub rebuild_every: usize,
    pub instability_budget: usize,
    pub step_count: usize,
    pub instability: usize,
}

#[derive(Debug, Clone)]
pub struct SpannerMaintenance {
    pub spanner: DynamicSpanner,
    pub max_edges: usize,
    pub rebuild_every: usize,
    pending_updates: usize,
}

#[derive(Debug, Clone)]
pub struct RecursiveHierarchy {
    pub levels: Vec<SpannerLevel>,
    pub vertex_maps: Vec<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct DeterministicVertexDecomposition {
    pub node_count: usize,
    pub cluster_target: usize,
    pub rebuild_every: usize,
    pub map: Vec<usize>,
    pending_updates: usize,
}

#[derive(Debug, Clone)]
struct OriginalEdge {
    u: usize,
    v: usize,
    active: bool,
    length: f64,
    gradient: f64,
}

#[derive(Debug, Clone)]
pub struct DeterministicDynamicSpanner {
    pub node_count: usize,
    pub stretch_bound: f64,
    pub max_edges: usize,
    pub rebuild_every: usize,
    pending_updates: usize,
    edges: Vec<OriginalEdge>,
    edge_to_spanner: Vec<Option<usize>>,
    spanner: DynamicSpanner,
}

impl SpannerHierarchy {
    pub fn build_recursive(params: SpannerBuildParams<'_>) -> Option<Self> {
        let SpannerBuildParams {
            node_count,
            tails,
            heads,
            lengths,
            seed,
            levels,
            rebuild_every,
            instability_budget,
        } = params;
        if node_count == 0 || tails.len() != heads.len() || tails.len() != lengths.len() {
            return None;
        }
        let mut hierarchy = Vec::new();
        let level_count = levels.max(1);
        for level in 0..level_count {
            let tree = LowStretchTree::build_low_stretch(
                node_count,
                tails,
                heads,
                lengths,
                seed + level as u64,
            )
            .ok()?;
            let mut spanner = DynamicSpanner::new(node_count);
            for (edge_id, (&tail, &head)) in tails.iter().zip(heads.iter()).enumerate() {
                spanner.insert_edge_with_values(
                    tail as usize,
                    head as usize,
                    lengths[edge_id],
                    0.0,
                );
            }
            hierarchy.push(SpannerLevel { spanner, tree });
        }
        Some(Self {
            levels: hierarchy,
            rebuild_every: rebuild_every.max(1),
            instability_budget,
            step_count: 0,
            instability: 0,
        })
    }

    pub fn apply_updates(
        &mut self,
        updates: &[(usize, f64, f64)],
        gradient_threshold: f64,
        length_factor: f64,
    ) {
        for level in &mut self.levels {
            let instability =
                level
                    .spanner
                    .batch_update_edges(updates, gradient_threshold, length_factor);
            self.instability += instability;
        }
    }

    pub fn tick(&mut self, seed: u64, rebuild_levels: usize) -> bool {
        self.step_count += 1;
        if self.should_rebuild() {
            let rebuilt = self.rebuild(seed, rebuild_levels);
            return rebuilt;
        }
        false
    }

    pub fn should_rebuild(&self) -> bool {
        if self.instability_budget > 0 && self.instability >= self.instability_budget {
            return true;
        }
        self.step_count.is_multiple_of(self.rebuild_every)
    }

    pub fn rebuild(&mut self, seed: u64, rebuild_levels: usize) -> bool {
        let Some(level) = self.levels.first() else {
            return false;
        };
        let node_count = level.spanner.node_count;
        let mut tails = Vec::new();
        let mut heads = Vec::new();
        let mut lengths = Vec::new();
        for edge_id in 0..level.spanner.edges.len() {
            if let Some((u, v)) = level.spanner.edge_endpoints(edge_id) {
                tails.push(u as u32);
                heads.push(v as u32);
                lengths.push(level.spanner.edges[edge_id].length);
            }
        }
        if let Some(new_hierarchy) = SpannerHierarchy::build_recursive(SpannerBuildParams {
            node_count,
            tails: &tails,
            heads: &heads,
            lengths: &lengths,
            seed,
            levels: rebuild_levels,
            rebuild_every: self.rebuild_every,
            instability_budget: self.instability_budget,
        }) {
            self.levels = new_hierarchy.levels;
            self.instability = 0;
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SpannerBuildParams<'a> {
    pub node_count: usize,
    pub tails: &'a [u32],
    pub heads: &'a [u32],
    pub lengths: &'a [f64],
    pub seed: u64,
    pub levels: usize,
    pub rebuild_every: usize,
    pub instability_budget: usize,
}

impl DynamicSpanner {
    pub fn new(node_count: usize) -> Self {
        Self {
            node_count,
            edge_count: 0,
            edges: Vec::new(),
            adjacency: vec![Vec::new(); node_count],
            embeddings: HashMap::new(),
        }
    }

    pub fn insert_edge(&mut self, u: usize, v: usize) -> usize {
        self.insert_edge_with_values(u, v, 1.0, 0.0)
    }

    pub fn insert_edge_with_values(
        &mut self,
        u: usize,
        v: usize,
        length: f64,
        gradient: f64,
    ) -> usize {
        let edge_id = self.edges.len();
        self.edges.push(SpannerEdge {
            u,
            v,
            active: true,
            length,
            gradient,
        });
        if u >= self.adjacency.len() || v >= self.adjacency.len() {
            let new_len = usize::max(u, v) + 1;
            self.adjacency.resize_with(new_len, Vec::new);
            self.node_count = new_len;
        }
        self.adjacency[u].push(edge_id);
        self.adjacency[v].push(edge_id);
        self.edge_count += 1;
        edge_id
    }

    pub fn delete_edge(&mut self, edge_id: usize) -> bool {
        let Some(edge) = self.edges.get_mut(edge_id) else {
            return false;
        };
        if !edge.active {
            return false;
        }
        edge.active = false;
        self.edge_count = self.edge_count.saturating_sub(1);
        if let Some(list) = self.adjacency.get_mut(edge.u) {
            list.retain(|&id| id != edge_id);
        }
        if let Some(list) = self.adjacency.get_mut(edge.v) {
            list.retain(|&id| id != edge_id);
        }
        true
    }

    pub fn split_vertex(&mut self, vertex: usize) -> usize {
        if vertex >= self.node_count {
            self.node_count = vertex + 1;
            self.adjacency.resize_with(self.node_count, Vec::new);
        }
        let new_vertex = self.node_count;
        self.node_count += 1;
        self.adjacency.push(Vec::new());
        new_vertex
    }

    pub fn split_vertex_with_edges(&mut self, vertex: usize, edges_to_move: &[usize]) -> usize {
        let new_vertex = self.split_vertex(vertex);
        for &edge_id in edges_to_move {
            let Some(edge) = self.edges.get_mut(edge_id) else {
                continue;
            };
            if !edge.active {
                continue;
            }
            let mut moved = false;
            if edge.u == vertex {
                edge.u = new_vertex;
                moved = true;
            }
            if edge.v == vertex {
                edge.v = new_vertex;
                moved = true;
            }
            if moved {
                if let Some(list) = self.adjacency.get_mut(vertex) {
                    list.retain(|&id| id != edge_id);
                }
                self.adjacency[new_vertex].push(edge_id);
            }
        }
        new_vertex
    }

    pub fn update_edge_values(&mut self, edge_id: usize, length: f64, gradient: f64) -> bool {
        let Some(edge) = self.edges.get_mut(edge_id) else {
            return false;
        };
        if !edge.active {
            return false;
        }
        edge.length = length;
        edge.gradient = gradient;
        true
    }

    pub fn apply_edge_update(
        &mut self,
        edge_id: usize,
        length: f64,
        gradient: f64,
        gradient_threshold: f64,
        length_factor: f64,
    ) -> Option<bool> {
        let edge = self.edges.get(edge_id)?;
        if !edge.active {
            return None;
        }
        let mut significant = false;
        if (gradient - edge.gradient).abs() > gradient_threshold {
            significant = true;
        }
        if edge.length > 0.0 && length > 0.0 {
            let ratio = if length > edge.length {
                length / edge.length
            } else {
                edge.length / length
            };
            if ratio > length_factor {
                significant = true;
            }
        }
        self.update_edge_values(edge_id, length, gradient);
        Some(significant)
    }

    pub fn batch_update_edges(
        &mut self,
        updates: &[(usize, f64, f64)],
        gradient_threshold: f64,
        length_factor: f64,
    ) -> usize {
        let mut instability = 0;
        for &(edge_id, length, gradient) in updates {
            if self
                .apply_edge_update(edge_id, length, gradient, gradient_threshold, length_factor)
                .unwrap_or(false)
            {
                instability += 1;
            }
        }
        instability
    }

    pub fn set_embedding(&mut self, original_edge: usize, path_edges: Vec<EmbeddingStep>) {
        self.embeddings
            .insert(original_edge, EmbeddingPath { steps: path_edges });
    }

    pub fn embed_edge_with_bfs(
        &mut self,
        original_edge: usize,
        start: usize,
        end: usize,
    ) -> Option<Vec<EmbeddingStep>> {
        if start >= self.node_count || end >= self.node_count {
            return None;
        }
        if start == end {
            return None;
        }
        let mut visited = vec![false; self.node_count];
        let mut parent: Vec<Option<(usize, usize)>> = vec![None; self.node_count];
        let mut queue = VecDeque::new();
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
        self.set_embedding(original_edge, steps_rev.clone());
        Some(steps_rev)
    }

    pub fn embedding_path(&self, original_edge: usize) -> Option<&EmbeddingPath> {
        self.embeddings.get(&original_edge)
    }

    pub fn embedding_steps(&self, original_edge: usize) -> Option<&[EmbeddingStep]> {
        self.embeddings
            .get(&original_edge)
            .map(|path| path.steps.as_slice())
    }

    pub fn edge_endpoints(&self, edge_id: usize) -> Option<(usize, usize)> {
        self.edges.get(edge_id).and_then(|edge| {
            if edge.active {
                Some((edge.u, edge.v))
            } else {
                None
            }
        })
    }

    pub fn embedding_valid(&self, original_edge: usize) -> bool {
        let Some(path) = self.embeddings.get(&original_edge) else {
            return false;
        };
        if path.steps.is_empty() {
            return false;
        }
        let mut prev_end: Option<usize> = None;
        for step in &path.steps {
            if step.dir != 1 && step.dir != -1 {
                return false;
            };
            let Some((u, v)) = self.oriented_endpoints(step.edge, step.dir) else {
                return false;
            };
            if let Some(prev) = prev_end {
                if prev != u {
                    return false;
                }
            }
            prev_end = Some(v);
        }
        true
    }

    pub fn embedding_endpoints(&self, original_edge: usize) -> Option<(usize, usize)> {
        let path = self.embeddings.get(&original_edge)?;
        let mut start: Option<usize> = None;
        let mut current: Option<usize> = None;
        for step in &path.steps {
            let (u, v) = self.oriented_endpoints(step.edge, step.dir)?;
            if let Some(prev) = current {
                if prev != u {
                    return None;
                }
            } else {
                start = Some(u);
            }
            current = Some(v);
        }
        match (start, current) {
            (Some(u), Some(v)) => Some((u, v)),
            _ => None,
        }
    }

    pub fn embedding_metrics(&self, original_edge: usize) -> Option<EmbeddingMetrics> {
        let path = self.embeddings.get(&original_edge)?;
        if path.steps.is_empty() {
            return None;
        }
        let mut total_length = 0.0;
        let mut total_gradient = 0.0;
        let mut current: Option<usize> = None;
        for step in &path.steps {
            let (u, v) = self.oriented_endpoints(step.edge, step.dir)?;
            if let Some(prev) = current {
                if prev != u {
                    return None;
                }
            }
            let edge = self.edges.get(step.edge)?;
            if !edge.active {
                return None;
            }
            total_length += edge.length;
            total_gradient += (step.dir as f64) * edge.gradient;
            current = Some(v);
        }
        Some(EmbeddingMetrics {
            total_length,
            total_gradient,
        })
    }

    pub fn embedding_ratio(&self, original_edge: usize) -> Option<f64> {
        let metrics = self.embedding_metrics(original_edge)?;
        if metrics.total_length <= 0.0 {
            return None;
        }
        Some(metrics.total_gradient / metrics.total_length)
    }

    fn oriented_endpoints(&self, edge_id: usize, dir: i8) -> Option<(usize, usize)> {
        let (u, v) = self.edge_endpoints(edge_id)?;
        if dir >= 0 {
            Some((u, v))
        } else {
            Some((v, u))
        }
    }
}

impl DeterministicDynamicSpanner {
    pub fn new(
        node_count: usize,
        stretch_bound: f64,
        max_edges: usize,
        rebuild_every: usize,
    ) -> Self {
        Self {
            node_count,
            stretch_bound: stretch_bound.max(1.0),
            max_edges: max_edges.max(1),
            rebuild_every: rebuild_every.max(1),
            pending_updates: 0,
            edges: Vec::new(),
            edge_to_spanner: Vec::new(),
            spanner: DynamicSpanner::new(node_count),
        }
    }

    pub fn sync_snapshot(
        &mut self,
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        lengths: &[f64],
        gradients: &[f64],
    ) {
        self.node_count = node_count;
        self.max_edges = self.max_edges.max(node_count.saturating_mul(8).max(1));
        self.rebuild_every = self
            .rebuild_every
            .max(((node_count as f64).ln().ceil() as usize).max(1));
        self.edges = tails
            .iter()
            .zip(heads.iter())
            .zip(lengths.iter().zip(gradients.iter()))
            .map(|((&tail, &head), (&length, &gradient))| OriginalEdge {
                u: tail as usize,
                v: head as usize,
                active: true,
                length,
                gradient,
            })
            .collect();
        self.edge_to_spanner = vec![None; self.edges.len()];
        self.pending_updates = 0;
        self.rebuild();
    }

    pub fn insert_edge_with_values(
        &mut self,
        u: usize,
        v: usize,
        length: f64,
        gradient: f64,
    ) -> usize {
        let edge_id = self.edges.len();
        self.edges.push(OriginalEdge {
            u,
            v,
            active: true,
            length,
            gradient,
        });
        if u >= self.node_count || v >= self.node_count {
            self.node_count = usize::max(u, v) + 1;
        }
        self.edge_to_spanner.push(None);
        self.pending_updates += 1;
        self.maintain();
        edge_id
    }

    pub fn delete_edge(&mut self, edge_id: usize) -> bool {
        let Some(edge) = self.edges.get_mut(edge_id) else {
            return false;
        };
        if !edge.active {
            return false;
        }
        edge.active = false;
        self.pending_updates += 1;
        self.maintain();
        true
    }

    pub fn update_edge_values(&mut self, edge_id: usize, length: f64, gradient: f64) -> bool {
        let Some(edge) = self.edges.get_mut(edge_id) else {
            return false;
        };
        if !edge.active {
            return false;
        }
        edge.length = length;
        edge.gradient = gradient;
        if let Some(spanner_edge) = self.edge_to_spanner.get(edge_id).and_then(|id| *id) {
            self.spanner
                .update_edge_values(spanner_edge, length, gradient);
        }
        self.pending_updates += 1;
        self.maintain();
        true
    }

    pub fn maintain(&mut self) {
        if self.pending_updates >= self.rebuild_every {
            self.rebuild();
            self.pending_updates = 0;
        }
    }

    pub fn embedding_valid(&self, edge_id: usize) -> bool {
        self.spanner.embedding_valid(edge_id)
    }

    pub fn embedding_steps(&self, edge_id: usize) -> Option<&[EmbeddingStep]> {
        self.spanner.embedding_steps(edge_id)
    }

    pub fn embed_edge_with_bfs(
        &mut self,
        edge_id: usize,
        start: usize,
        end: usize,
    ) -> Option<Vec<EmbeddingStep>> {
        self.spanner.embed_edge_with_bfs(edge_id, start, end)
    }

    pub fn embedding_ratio(&self, edge_id: usize) -> Option<f64> {
        let metrics = self.spanner.embedding_metrics(edge_id)?;
        let edge = self.edges.get(edge_id)?;
        if !edge.active || edge.length <= 0.0 {
            return None;
        }
        Some(metrics.total_length / edge.length)
    }

    fn rebuild(&mut self) {
        let active_edges: Vec<(usize, &OriginalEdge)> = self
            .edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| edge.active)
            .collect();
        self.spanner = DynamicSpanner::new(self.node_count);
        self.edge_to_spanner = vec![None; self.edges.len()];
        if active_edges.is_empty() {
            return;
        }

        let mut tails = Vec::with_capacity(active_edges.len());
        let mut heads = Vec::with_capacity(active_edges.len());
        let mut lengths = Vec::with_capacity(active_edges.len());
        let mut active_ids = Vec::with_capacity(active_edges.len());
        for (edge_id, edge) in &active_edges {
            tails.push(edge.u as u32);
            heads.push(edge.v as u32);
            lengths.push(edge.length.abs());
            active_ids.push(*edge_id);
        }

        let Ok(tree) = LowStretchTree::build_low_stretch_deterministic(
            self.node_count,
            &tails,
            &heads,
            &lengths,
        ) else {
            return;
        };

        for (compact_id, &edge_id) in active_ids.iter().enumerate() {
            if !tree.tree_edges.get(compact_id).copied().unwrap_or(false) {
                continue;
            }
            let edge = &self.edges[edge_id];
            let spanner_edge =
                self.spanner
                    .insert_edge_with_values(edge.u, edge.v, edge.length, edge.gradient);
            self.edge_to_spanner[edge_id] = Some(spanner_edge);
        }

        for &edge_id in active_ids.iter() {
            if self.edge_to_spanner[edge_id].is_some() {
                continue;
            }
            let edge = &self.edges[edge_id];
            let mut include_direct = edge.length <= 0.0;
            if let Some(tree_distance) = tree.path_length(edge.u, edge.v) {
                if tree_distance > self.stretch_bound * edge.length.abs() {
                    include_direct = true;
                }
            } else {
                include_direct = true;
            }
            if include_direct {
                let spanner_edge = self.spanner.insert_edge_with_values(
                    edge.u,
                    edge.v,
                    edge.length,
                    edge.gradient,
                );
                self.edge_to_spanner[edge_id] = Some(spanner_edge);
            }
        }

        if self.spanner.edge_count > self.max_edges {
            self.max_edges = self.spanner.edge_count.max(self.max_edges);
        }

        self.spanner.embeddings.clear();
        for (compact_id, &edge_id) in active_ids.iter().enumerate() {
            if let Some(spanner_edge) = self.edge_to_spanner[edge_id] {
                self.spanner
                    .set_embedding(edge_id, vec![EmbeddingStep::new(spanner_edge, 1)]);
                continue;
            }
            let Some(path) = tree.path_edges(
                tails[compact_id] as usize,
                heads[compact_id] as usize,
                &tails,
                &heads,
            ) else {
                let edge = &self.edges[edge_id];
                let spanner_edge = self.spanner.insert_edge_with_values(
                    edge.u,
                    edge.v,
                    edge.length,
                    edge.gradient,
                );
                self.edge_to_spanner[edge_id] = Some(spanner_edge);
                self.spanner
                    .set_embedding(edge_id, vec![EmbeddingStep::new(spanner_edge, 1)]);
                continue;
            };
            let mut steps = Vec::with_capacity(path.len());
            let mut valid = true;
            for (compact_edge_id, dir) in path {
                let original_id = active_ids[compact_edge_id];
                let Some(spanner_edge) = self.edge_to_spanner[original_id] else {
                    valid = false;
                    break;
                };
                steps.push(EmbeddingStep::new(spanner_edge, dir));
            }
            if valid {
                self.spanner.set_embedding(edge_id, steps);
            } else {
                let edge = &self.edges[edge_id];
                let spanner_edge = self.spanner.insert_edge_with_values(
                    edge.u,
                    edge.v,
                    edge.length,
                    edge.gradient,
                );
                self.edge_to_spanner[edge_id] = Some(spanner_edge);
                self.spanner
                    .set_embedding(edge_id, vec![EmbeddingStep::new(spanner_edge, 1)]);
            }
        }
    }
}

impl SpannerMaintenance {
    pub fn new(node_count: usize, max_edges: usize, rebuild_every: usize) -> Self {
        Self {
            spanner: DynamicSpanner::new(node_count),
            max_edges: max_edges.max(1),
            rebuild_every: rebuild_every.max(1),
            pending_updates: 0,
        }
    }

    pub fn insert_edge_with_values(
        &mut self,
        u: usize,
        v: usize,
        length: f64,
        gradient: f64,
    ) -> usize {
        let edge_id = self.spanner.insert_edge_with_values(u, v, length, gradient);
        self.pending_updates += 1;
        self.maintain();
        edge_id
    }

    pub fn delete_edge(&mut self, edge_id: usize) -> bool {
        let deleted = self.spanner.delete_edge(edge_id);
        if deleted {
            self.pending_updates += 1;
            self.maintain();
        }
        deleted
    }

    pub fn update_edge_values(&mut self, edge_id: usize, length: f64, gradient: f64) -> bool {
        let updated = self.spanner.update_edge_values(edge_id, length, gradient);
        if updated {
            self.pending_updates += 1;
            self.maintain();
        }
        updated
    }

    pub fn maintain(&mut self) {
        if self.pending_updates == 0 {
            return;
        }
        if self.pending_updates < self.rebuild_every && self.spanner.edge_count <= self.max_edges {
            return;
        }
        self.prune_to_max_edges();
        self.refresh_embeddings();
        self.pending_updates = 0;
    }

    fn prune_to_max_edges(&mut self) {
        if self.spanner.edge_count <= self.max_edges {
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
        let excess = active_edges.len().saturating_sub(self.max_edges);
        for edge_id in active_edges.into_iter().rev().take(excess) {
            self.spanner.delete_edge(edge_id);
        }
    }

    fn refresh_embeddings(&mut self) {
        self.spanner
            .embeddings
            .retain(|edge_id, _| self.spanner.edges.get(*edge_id).is_some());
        for edge_id in 0..self.spanner.edges.len() {
            if self.spanner.edges[edge_id].active {
                self.spanner
                    .set_embedding(edge_id, vec![EmbeddingStep::new(edge_id, 1)]);
            }
        }
    }
}

impl RecursiveHierarchy {
    pub fn build(params: SpannerBuildParams<'_>, cluster_ratio: usize) -> Option<Self> {
        let mut levels = Vec::new();
        let mut vertex_maps = Vec::new();
        let mut node_count = params.node_count;
        let mut tails = params.tails.to_vec();
        let mut heads = params.heads.to_vec();
        let mut lengths = params.lengths.to_vec();
        for level in 0..params.levels.max(1) {
            let tree = LowStretchTree::build_low_stretch_deterministic(
                node_count, &tails, &heads, &lengths,
            )
            .ok()?;
            let mut spanner = DynamicSpanner::new(node_count);
            for (edge_id, (&tail, &head)) in tails.iter().zip(heads.iter()).enumerate() {
                spanner.insert_edge_with_values(
                    tail as usize,
                    head as usize,
                    lengths[edge_id],
                    0.0,
                );
                spanner.set_embedding(edge_id, vec![EmbeddingStep::new(edge_id, 1)]);
            }
            levels.push(SpannerLevel { spanner, tree });
            let map = deterministic_vertex_map(
                node_count,
                &tails,
                &heads,
                &levels.last().unwrap().tree,
                cluster_ratio,
            );
            vertex_maps.push(map.clone());
            if level + 1 == params.levels {
                break;
            }
            let (new_tails, new_heads, new_lengths, new_node_count) =
                compress_edges(&map, &tails, &heads, &lengths);
            tails = new_tails;
            heads = new_heads;
            lengths = new_lengths;
            node_count = new_node_count;
        }
        Some(Self {
            levels,
            vertex_maps,
        })
    }
}

impl DeterministicVertexDecomposition {
    pub fn new(
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        tree: &LowStretchTree,
        cluster_target: usize,
        rebuild_every: usize,
    ) -> Self {
        let map = deterministic_vertex_map(node_count, tails, heads, tree, cluster_target);
        Self {
            node_count,
            cluster_target: cluster_target.max(1),
            rebuild_every: rebuild_every.max(1),
            map,
            pending_updates: 0,
        }
    }

    pub fn record_update(&mut self, updates: usize) {
        self.pending_updates = self.pending_updates.saturating_add(updates);
    }

    pub fn should_rebuild(&self) -> bool {
        self.pending_updates >= self.rebuild_every
    }

    pub fn rebuild(&mut self, tails: &[u32], heads: &[u32], tree: &LowStretchTree) {
        self.map =
            deterministic_vertex_map(self.node_count, tails, heads, tree, self.cluster_target);
        self.pending_updates = 0;
    }

    pub fn default_rebuild_every(node_count: usize) -> usize {
        ((node_count as f64).ln().ceil() as usize).max(1)
    }
}

fn deterministic_vertex_map(
    node_count: usize,
    tails: &[u32],
    heads: &[u32],
    tree: &LowStretchTree,
    cluster_target: usize,
) -> Vec<usize> {
    let target = cluster_target.max(1);
    let mut adjacency = vec![Vec::new(); node_count];
    for edge_id in 0..tails.len() {
        if !tree.tree_edges.get(edge_id).copied().unwrap_or(false) {
            continue;
        }
        let u = tails[edge_id] as usize;
        let v = heads[edge_id] as usize;
        if u >= node_count || v >= node_count {
            continue;
        }
        adjacency[u].push(v);
        adjacency[v].push(u);
    }
    for neighbors in adjacency.iter_mut() {
        neighbors.sort_unstable();
    }

    let mut map = vec![usize::MAX; node_count];
    let mut cluster_id = 0;
    for start in 0..node_count {
        if map[start] != usize::MAX {
            continue;
        }
        let mut queue = VecDeque::new();
        queue.push_back(start);
        map[start] = cluster_id;
        let mut size = 1;
        while let Some(node) = queue.pop_front() {
            for &neighbor in &adjacency[node] {
                if map[neighbor] != usize::MAX {
                    continue;
                }
                if size >= target {
                    continue;
                }
                map[neighbor] = cluster_id;
                size += 1;
                queue.push_back(neighbor);
            }
        }
        cluster_id += 1;
    }
    map
}

fn compress_edges(
    map: &[usize],
    tails: &[u32],
    heads: &[u32],
    lengths: &[f64],
) -> (Vec<u32>, Vec<u32>, Vec<f64>, usize) {
    let mut edge_map: HashMap<(usize, usize), f64> = HashMap::new();
    let mut max_node = 0;
    for (edge_id, (&tail, &head)) in tails.iter().zip(heads.iter()).enumerate() {
        let u = map[tail as usize];
        let v = map[head as usize];
        if u == v {
            continue;
        }
        max_node = max_node.max(u.max(v));
        let key = if u <= v { (u, v) } else { (v, u) };
        let entry = edge_map.entry(key).or_insert(f64::INFINITY);
        if lengths[edge_id] < *entry {
            *entry = lengths[edge_id];
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_paths_use_edges_in_h() {
        let mut spanner = DynamicSpanner::new(3);
        let e0 = spanner.insert_edge(0, 1);
        let e1 = spanner.insert_edge(1, 2);
        spanner.set_embedding(
            10,
            vec![EmbeddingStep::new(e0, 1), EmbeddingStep::new(e1, 1)],
        );
        let embedding = spanner.embedding_path(10).unwrap();
        assert_eq!(embedding.steps.len(), 2);
        assert!(spanner.embedding_valid(10));
    }

    #[test]
    fn delete_edge_invalidates_embedding() {
        let mut spanner = DynamicSpanner::new(3);
        let e0 = spanner.insert_edge(0, 1);
        let e1 = spanner.insert_edge(1, 2);
        spanner.set_embedding(
            11,
            vec![EmbeddingStep::new(e0, 1), EmbeddingStep::new(e1, 1)],
        );
        assert!(spanner.embedding_valid(11));
        spanner.delete_edge(e1);
        assert!(!spanner.embedding_valid(11));
    }

    #[test]
    fn vertex_split_extends_graph() {
        let mut spanner = DynamicSpanner::new(2);
        let new_vertex = spanner.split_vertex(1);
        assert_eq!(new_vertex, 2);
        assert_eq!(spanner.node_count, 3);
        let e0 = spanner.insert_edge(1, new_vertex);
        spanner.set_embedding(12, vec![EmbeddingStep::new(e0, 1)]);
        assert!(spanner.embedding_valid(12));
    }

    #[test]
    fn embedding_endpoints_follow_path_orientation() {
        let mut spanner = DynamicSpanner::new(4);
        let e0 = spanner.insert_edge(0, 1);
        let e1 = spanner.insert_edge(1, 2);
        let e2 = spanner.insert_edge(2, 3);
        spanner.set_embedding(
            20,
            vec![
                EmbeddingStep::new(e0, 1),
                EmbeddingStep::new(e1, 1),
                EmbeddingStep::new(e2, 1),
            ],
        );
        assert_eq!(spanner.embedding_endpoints(20), Some((0, 3)));
    }

    #[test]
    fn bfs_embedding_connects_expected_endpoints() {
        let mut spanner = DynamicSpanner::new(4);
        spanner.insert_edge(0, 1);
        spanner.insert_edge(1, 2);
        spanner.insert_edge(2, 3);
        let steps = spanner
            .embed_edge_with_bfs(21, 0, 3)
            .expect("path should exist");
        assert_eq!(steps.len(), 3);
        assert!(spanner.embedding_valid(21));
        assert_eq!(spanner.embedding_endpoints(21), Some((0, 3)));
    }
    #[test]
    fn embedding_direction_enforces_path_order() {
        let mut spanner = DynamicSpanner::new(3);
        let e0 = spanner.insert_edge(0, 1);
        let e1 = spanner.insert_edge(1, 2);
        spanner.set_embedding(
            13,
            vec![EmbeddingStep::new(e1, -1), EmbeddingStep::new(e0, -1)],
        );
        assert!(spanner.embedding_valid(13));

        spanner.set_embedding(
            14,
            vec![EmbeddingStep::new(e0, 1), EmbeddingStep::new(e1, -1)],
        );
        assert!(!spanner.embedding_valid(14));
    }

    #[test]
    fn bfs_embedding_builds_path_steps() {
        let mut spanner = DynamicSpanner::new(4);
        let e0 = spanner.insert_edge(0, 1);
        let e1 = spanner.insert_edge(1, 2);
        let e2 = spanner.insert_edge(2, 3);
        let steps = spanner
            .embed_edge_with_bfs(21, 0, 3)
            .expect("path should exist");
        assert_eq!(
            steps,
            vec![
                EmbeddingStep::new(e0, 1),
                EmbeddingStep::new(e1, 1),
                EmbeddingStep::new(e2, 1)
            ]
        );
        assert!(spanner.embedding_valid(21));
    }

    #[test]
    fn bfs_embedding_returns_none_when_disconnected() {
        let mut spanner = DynamicSpanner::new(3);
        spanner.insert_edge(0, 1);
        assert!(spanner.embed_edge_with_bfs(22, 0, 2).is_none());
        assert!(!spanner.embedding_valid(22));
    }

    #[test]
    fn embedding_metrics_accumulates_values() {
        let mut spanner = DynamicSpanner::new(3);
        let e0 = spanner.insert_edge_with_values(0, 1, 2.0, 1.5);
        let e1 = spanner.insert_edge_with_values(2, 1, 3.0, -2.0);
        spanner.set_embedding(
            30,
            vec![EmbeddingStep::new(e0, 1), EmbeddingStep::new(e1, -1)],
        );
        let metrics = spanner.embedding_metrics(30).expect("metrics should exist");
        assert!((metrics.total_length - 5.0).abs() < 1e-9);
        assert!((metrics.total_gradient - 3.5).abs() < 1e-9);
        assert!(spanner.embedding_ratio(30).is_some());
    }

    #[test]
    fn split_vertex_reassigns_edges() {
        let mut spanner = DynamicSpanner::new(3);
        let e0 = spanner.insert_edge(0, 1);
        let e1 = spanner.insert_edge(1, 2);
        let new_vertex = spanner.split_vertex_with_edges(1, &[e0]);
        assert_eq!(new_vertex, 3);
        assert_eq!(spanner.edge_endpoints(e0), Some((0, 3)));
        assert_eq!(spanner.edge_endpoints(e1), Some((1, 2)));
    }

    #[test]
    fn batch_update_edges_tracks_instability() {
        let mut spanner = DynamicSpanner::new(2);
        let e0 = spanner.insert_edge_with_values(0, 1, 1.0, 0.0);
        let e1 = spanner.insert_edge_with_values(0, 1, 2.0, 0.5);
        let updates = vec![(e0, 1.1, 0.6), (e1, 2.2, 0.9)];
        let instability = spanner.batch_update_edges(&updates, 0.3, 1.1);
        assert_eq!(instability, 2);
    }

    #[test]
    fn apply_edge_update_respects_thresholds() {
        let mut spanner = DynamicSpanner::new(2);
        let edge_id = spanner.insert_edge_with_values(0, 1, 1.0, 0.1);
        let significant = spanner.apply_edge_update(edge_id, 1.05, 0.15, 0.2, 1.2);
        assert_eq!(significant, Some(false));
        assert!((spanner.edges[edge_id].length - 1.05).abs() < 1e-9);
        assert!((spanner.edges[edge_id].gradient - 0.15).abs() < 1e-9);

        let significant = spanner.apply_edge_update(edge_id, 1.5, 0.6, 0.2, 1.2);
        assert_eq!(significant, Some(true));

        spanner.delete_edge(edge_id);
        assert_eq!(spanner.apply_edge_update(edge_id, 2.0, 0.0, 0.2, 1.2), None);
    }

    #[test]
    fn hierarchy_rebuilds_on_instability() {
        let tails = vec![0, 1, 2];
        let heads = vec![1, 2, 0];
        let lengths = vec![1.0, 1.0, 1.0];
        let mut hierarchy = SpannerHierarchy::build_recursive(SpannerBuildParams {
            node_count: 3,
            tails: &tails,
            heads: &heads,
            lengths: &lengths,
            seed: 3,
            levels: 2,
            rebuild_every: 5,
            instability_budget: 1,
        })
        .unwrap();
        let updates = vec![(0, 2.0, 0.8), (1, 2.5, 0.9)];
        hierarchy.apply_updates(&updates, 0.1, 1.1);
        assert!(hierarchy.should_rebuild());
        assert!(hierarchy.tick(4, 2));
    }

    #[test]
    fn maintenance_prunes_and_refreshes_embeddings() {
        let mut maintenance = SpannerMaintenance::new(4, 2, 2);
        let e0 = maintenance.insert_edge_with_values(0, 1, 1.0, 0.1);
        let _e1 = maintenance.insert_edge_with_values(1, 2, 1.0, 0.2);
        let _e2 = maintenance.insert_edge_with_values(2, 3, 1.0, 0.3);
        maintenance.maintain();
        assert!(maintenance.spanner.edge_count <= 2);
        assert!(maintenance.spanner.embedding_valid(e0));
    }

    #[test]
    fn recursive_hierarchy_reduces_vertices() {
        let tails = vec![0, 1, 2, 3];
        let heads = vec![1, 2, 3, 0];
        let lengths = vec![1.0, 1.0, 1.0, 1.0];
        let hierarchy = RecursiveHierarchy::build(
            SpannerBuildParams {
                node_count: 4,
                tails: &tails,
                heads: &heads,
                lengths: &lengths,
                seed: 5,
                levels: 2,
                rebuild_every: 2,
                instability_budget: 1,
            },
            2,
        )
        .unwrap();
        assert_eq!(hierarchy.levels.len(), 2);
        assert!(hierarchy.vertex_maps[1].iter().max().unwrap() < &4);
    }

    #[test]
    fn deterministic_vertex_map_clusters_tree_edges() {
        let tails = vec![0, 1, 2, 3, 4];
        let heads = vec![1, 2, 3, 4, 0];
        let lengths = vec![1.0; tails.len()];
        let tree =
            LowStretchTree::build_low_stretch_deterministic(5, &tails, &heads, &lengths).unwrap();
        let map = deterministic_vertex_map(5, &tails, &heads, &tree, 2);
        assert_eq!(map.len(), 5);
        let unique_clusters = map
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        assert!(unique_clusters.len() >= 3);
        for edge_id in 0..tails.len() {
            if !tree.tree_edges[edge_id] {
                continue;
            }
            let u = tails[edge_id] as usize;
            let v = heads[edge_id] as usize;
            let cu = map[u];
            let cv = map[v];
            assert!(cu != usize::MAX && cv != usize::MAX);
        }
    }

    #[test]
    fn deterministic_decomposition_rebuilds_on_updates() {
        let tails = vec![0, 1, 2];
        let heads = vec![1, 2, 0];
        let lengths = vec![1.0; 3];
        let tree =
            LowStretchTree::build_low_stretch_deterministic(3, &tails, &heads, &lengths).unwrap();
        let mut decomposition =
            DeterministicVertexDecomposition::new(3, &tails, &heads, &tree, 1, 2);
        assert!(!decomposition.should_rebuild());
        decomposition.record_update(1);
        assert!(!decomposition.should_rebuild());
        decomposition.record_update(1);
        assert!(decomposition.should_rebuild());
        decomposition.rebuild(&tails, &heads, &tree);
        assert!(!decomposition.should_rebuild());
    }
}
