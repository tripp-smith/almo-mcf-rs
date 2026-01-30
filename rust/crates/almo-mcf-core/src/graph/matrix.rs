use crate::graph::{EdgeId, Graph, NodeId};

#[derive(Debug, Clone)]
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl SparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn push(&mut self, row: usize, col: usize, value: f64) {
        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(value);
    }
}

pub fn incidence_matrix(graph: &Graph) -> SparseMatrix {
    let mut matrix = SparseMatrix::new(graph.node_count(), graph.edge_count());
    for (edge_id, edge) in graph.edges() {
        push_incidence_entry(&mut matrix, edge.tail, edge_id, -1.0);
        push_incidence_entry(&mut matrix, edge.head, edge_id, 1.0);
    }
    matrix
}

fn push_incidence_entry(matrix: &mut SparseMatrix, node: NodeId, edge: EdgeId, value: f64) {
    matrix.push(node.0, edge.0, value);
}
