use std::collections::HashSet;

use crate::trees::LowStretchTree;

#[derive(Debug, Clone)]
pub struct DynamicForest {
    pub node_count: usize,
    pub tails: Vec<u32>,
    pub heads: Vec<u32>,
    pub lengths: Vec<f64>,
    pub tree_edges: Vec<bool>,
    pub tree: LowStretchTree,
    deleted_edges: HashSet<usize>,
}

impl DynamicForest {
    pub fn new_from_tree(
        node_count: usize,
        tails: Vec<u32>,
        heads: Vec<u32>,
        lengths: Vec<f64>,
        tree_edges: Vec<bool>,
    ) -> Result<Self, super::TreeError> {
        let tree = LowStretchTree::build_from_tree_edges(
            node_count,
            &tails,
            &heads,
            &lengths,
            tree_edges.clone(),
        )?;
        Ok(Self {
            node_count,
            tails,
            heads,
            lengths,
            tree_edges,
            tree,
            deleted_edges: HashSet::new(),
        })
    }

    pub fn promote_root(&mut self, root: usize) -> bool {
        if root >= self.node_count {
            return false;
        }
        let mut order: Vec<usize> = (0..self.node_count).collect();
        if let Some(pos) = order.iter().position(|&v| v == root) {
            order.swap(0, pos);
        }
        if let Ok(tree) = LowStretchTree::build_from_tree_edges_with_order(
            self.node_count,
            &self.tails,
            &self.heads,
            &self.lengths,
            self.tree_edges.clone(),
            &order,
        ) {
            self.tree = tree;
        }
        true
    }

    pub fn delete_edge(&mut self, edge_id: usize) -> bool {
        if edge_id >= self.tree_edges.len() {
            return false;
        }
        self.deleted_edges.insert(edge_id);
        let was_tree = self.tree_edges[edge_id];
        self.tree_edges[edge_id] = false;
        if was_tree {
            self.rebuild_all();
        }
        true
    }

    pub fn delete_edges(&mut self, edge_ids: &[usize]) -> usize {
        let mut rebuild = false;
        let mut count = 0;
        for &edge_id in edge_ids {
            if edge_id >= self.tree_edges.len() {
                continue;
            }
            if self.deleted_edges.insert(edge_id) {
                count += 1;
            }
            if self.tree_edges[edge_id] {
                self.tree_edges[edge_id] = false;
                rebuild = true;
            }
        }
        if rebuild {
            self.rebuild_all();
        }
        count
    }

    pub fn stretch_overestimate(&self, u: usize, v: usize, length: f64) -> f64 {
        if length <= 0.0 {
            return f64::INFINITY;
        }
        match self.tree.path_length(u, v) {
            Some(distance) => distance / length,
            None => f64::INFINITY,
        }
    }

    pub fn edge_stretch_overestimate(&self, edge_id: usize) -> Option<f64> {
        if edge_id >= self.tails.len() || edge_id >= self.lengths.len() {
            return None;
        }
        let u = self.tails[edge_id] as usize;
        let v = self.heads[edge_id] as usize;
        Some(self.stretch_overestimate(u, v, self.lengths[edge_id]))
    }

    pub fn is_deleted(&self, edge_id: usize) -> bool {
        self.deleted_edges.contains(&edge_id)
    }

    pub fn rebuild_all(&mut self) {
        if let Ok(tree) = LowStretchTree::build_from_tree_edges(
            self.node_count,
            &self.tails,
            &self.heads,
            &self.lengths,
            self.tree_edges.clone(),
        ) {
            self.tree = tree;
        }
    }

    #[allow(dead_code)]
    fn rebuild_component(&mut self, root: usize) {
        let _ = root;
        self.rebuild_all();
    }
}
