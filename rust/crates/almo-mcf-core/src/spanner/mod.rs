use std::collections::HashMap;
use std::collections::VecDeque;

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
}
