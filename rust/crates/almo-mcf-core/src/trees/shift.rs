use std::collections::HashSet;

use crate::trees::forest::DynamicForest;
use crate::trees::mwu::deterministic_weighted_tree;
use crate::trees::TreeError;

#[derive(Debug, Clone)]
pub struct ShiftableForestCollection {
    pub forests: Vec<DynamicForest>,
    pub shift_indices: Vec<usize>,
    pub rep_timestamps: Vec<usize>,
    pub lazy_updates: Vec<usize>,
    pending_deletions: Vec<HashSet<usize>>,
    step: usize,
}

impl ShiftableForestCollection {
    pub fn build_deterministic(
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        lengths: &[f64],
        forest_count: usize,
        levels: usize,
    ) -> Result<Self, TreeError> {
        let forest_count = forest_count.max(1);
        let mut forests = Vec::with_capacity(forest_count);
        for idx in 0..forest_count {
            let mut weights = vec![1.0; lengths.len()];
            for (edge_id, weight) in weights.iter_mut().enumerate() {
                let bias = 1.0 + (idx as f64) * 1e-3 + (edge_id as f64) * 1e-9;
                *weight *= bias;
            }
            let tree = deterministic_weighted_tree(node_count, tails, heads, lengths, &weights)?;
            let forest = DynamicForest::new_from_tree(
                node_count,
                tails.to_vec(),
                heads.to_vec(),
                lengths.to_vec(),
                tree.tree_edges.clone(),
            )?;
            forests.push(forest);
        }
        Ok(Self {
            forests,
            shift_indices: vec![0; levels.max(1)],
            rep_timestamps: vec![0; levels.max(1)],
            lazy_updates: vec![0; levels.max(1)],
            pending_deletions: vec![HashSet::new(); forest_count],
            step: 0,
        })
    }

    pub fn mark_deleted(&mut self, edge_id: usize) {
        for pending in &mut self.pending_deletions {
            pending.insert(edge_id);
        }
        for entry in &mut self.lazy_updates {
            *entry += 1;
        }
    }

    pub fn shift(&mut self, level: usize) -> Option<()> {
        if level >= self.shift_indices.len() {
            return None;
        }
        let forest_count = self.forests.len();
        let next = (self.shift_indices[level] + 1) % forest_count;
        self.shift_indices[level] = next;
        if next == 0 {
            self.rep_timestamps[level] = self.step;
        }
        self.apply_pending_to_current(level);
        for higher in (level + 1)..self.shift_indices.len() {
            self.shift_indices[higher] = next;
            self.apply_pending_to_current(higher);
        }
        Some(())
    }

    pub fn current_forest(&mut self, level: usize) -> Option<&DynamicForest> {
        self.apply_pending_to_current(level);
        self.forests.get(self.shift_indices[level])
    }

    pub fn current_forest_mut(&mut self, level: usize) -> Option<&mut DynamicForest> {
        self.apply_pending_to_current(level);
        let idx = self.shift_indices.get(level).copied()?;
        self.forests.get_mut(idx)
    }

    pub fn tick(&mut self) {
        self.step += 1;
    }

    pub fn propagate(&mut self, level: usize) {
        self.apply_pending_to_current(level);
    }

    fn apply_pending_to_current(&mut self, level: usize) {
        let Some(&idx) = self.shift_indices.get(level) else {
            return;
        };
        if let Some(pending) = self.pending_deletions.get_mut(idx) {
            if !pending.is_empty() {
                let deletions: Vec<usize> = pending.drain().collect();
                if let Some(forest) = self.forests.get_mut(idx) {
                    forest.delete_edges(&deletions);
                }
            }
        }
        if let Some(entry) = self.lazy_updates.get_mut(level) {
            *entry = 0;
        }
    }
}
