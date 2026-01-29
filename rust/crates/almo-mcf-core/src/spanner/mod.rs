use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct EmbeddingPath {
    pub edges: Vec<usize>,
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

    pub fn set_embedding(&mut self, original_edge: usize, path_edges: Vec<usize>) {
        self.embeddings
            .insert(original_edge, EmbeddingPath { edges: path_edges });
    }

    pub fn embedding_path(&self, original_edge: usize) -> Option<&EmbeddingPath> {
        self.embeddings.get(&original_edge)
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
        let mut prev: Option<(usize, usize)> = None;
        for edge_id in &path.edges {
            let Some((u, v)) = self.edge_endpoints(*edge_id) else {
                return false;
            };
            if let Some((prev_u, prev_v)) = prev {
                if prev_v != u && prev_v != v && prev_u != u && prev_u != v {
                    return false;
                }
            }
            prev = Some((u, v));
        }
        true
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
        spanner.set_embedding(10, vec![e0, e1]);
        let embedding = spanner.embedding_path(10).unwrap();
        assert_eq!(embedding.edges.len(), 2);
        assert!(spanner.embedding_valid(10));
    }

    #[test]
    fn delete_edge_invalidates_embedding() {
        let mut spanner = DynamicSpanner::new(3);
        let e0 = spanner.insert_edge(0, 1);
        let e1 = spanner.insert_edge(1, 2);
        spanner.set_embedding(11, vec![e0, e1]);
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
        spanner.set_embedding(12, vec![e0]);
        assert!(spanner.embedding_valid(12));
    }
}
