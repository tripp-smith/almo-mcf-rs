use std::collections::HashMap;

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

#[derive(Debug, Clone)]
struct SpannerEdge {
    u: usize,
    v: usize,
    active: bool,
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
        let edge_id = self.edges.len();
        self.edges.push(SpannerEdge { u, v, active: true });
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

    pub fn set_embedding(&mut self, original_edge: usize, path_edges: Vec<EmbeddingStep>) {
        self.embeddings
            .insert(original_edge, EmbeddingPath { steps: path_edges });
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
}
