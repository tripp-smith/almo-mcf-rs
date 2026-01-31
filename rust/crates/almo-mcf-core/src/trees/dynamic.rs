use crate::trees::{LowStretchTree, TreeError};

#[derive(Debug, Clone)]
pub struct DynamicTree {
    pub node_count: usize,
    pub tails: Vec<u32>,
    pub heads: Vec<u32>,
    pub lengths: Vec<f64>,
    pub active: Vec<bool>,
    pub rebuild_every: usize,
    pub update_budget: usize,
    pub update_count: usize,
    pub tree: LowStretchTree,
}

impl DynamicTree {
    pub fn new(
        node_count: usize,
        tails: Vec<u32>,
        heads: Vec<u32>,
        lengths: Vec<f64>,
        seed: u64,
        rebuild_every: usize,
        update_budget: usize,
    ) -> Result<Self, TreeError> {
        let edge_count = lengths.len();
        let tree = LowStretchTree::build_low_stretch(node_count, &tails, &heads, &lengths, seed)?;
        Ok(Self {
            node_count,
            tails,
            heads,
            lengths,
            active: vec![true; edge_count],
            rebuild_every: rebuild_every.max(1),
            update_budget,
            update_count: 0,
            tree,
        })
    }

    pub fn insert_edge(&mut self, tail: u32, head: u32, length: f64) {
        self.tails.push(tail);
        self.heads.push(head);
        self.lengths.push(length);
        self.active.push(true);
        self.update_count += 1;
    }

    pub fn delete_edge(&mut self, edge_id: usize) -> bool {
        let Some(active) = self.active.get_mut(edge_id) else {
            return false;
        };
        if !*active {
            return false;
        }
        *active = false;
        self.update_count += 1;
        true
    }

    pub fn update_length(&mut self, edge_id: usize, length: f64) -> bool {
        let Some(current) = self.lengths.get_mut(edge_id) else {
            return false;
        };
        *current = length;
        self.update_count += 1;
        true
    }

    pub fn should_rebuild(&self, step: usize) -> bool {
        if self.update_budget > 0 && self.update_count >= self.update_budget {
            return true;
        }
        step % self.rebuild_every == 0
    }

    pub fn rebuild(&mut self, seed: u64) -> Result<(), TreeError> {
        let mut tails = Vec::new();
        let mut heads = Vec::new();
        let mut lengths = Vec::new();
        for idx in 0..self.tails.len() {
            if self.active[idx] {
                tails.push(self.tails[idx]);
                heads.push(self.heads[idx]);
                lengths.push(self.lengths[idx]);
            }
        }
        self.tree =
            LowStretchTree::build_low_stretch(self.node_count, &tails, &heads, &lengths, seed)?;
        self.update_count = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebuild_triggers_after_updates() {
        let tails = vec![0, 1, 2];
        let heads = vec![1, 2, 0];
        let lengths = vec![1.0, 1.0, 1.0];
        let mut tree = DynamicTree::new(3, tails, heads, lengths, 1, 4, 2).unwrap();
        tree.update_length(0, 2.0);
        tree.update_length(1, 3.0);
        assert!(tree.should_rebuild(1));
    }
}
