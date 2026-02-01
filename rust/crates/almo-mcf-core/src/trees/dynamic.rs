use crate::trees::{LowStretchTree, TreeBuildMode, TreeError};

#[derive(Debug, Clone, Copy)]
pub struct DynamicTreeConfig {
    pub seed: u64,
    pub rebuild_every: usize,
    pub update_budget: usize,
    pub length_factor: f64,
    pub build_mode: TreeBuildMode,
}

impl DynamicTreeConfig {
    pub fn randomized(seed: u64, rebuild_every: usize, update_budget: usize) -> Self {
        Self {
            seed,
            rebuild_every,
            update_budget,
            length_factor: 1.25,
            build_mode: TreeBuildMode::Randomized,
        }
    }

    pub fn deterministic(rebuild_every: usize, update_budget: usize) -> Self {
        Self {
            seed: 0,
            rebuild_every,
            update_budget,
            length_factor: 1.25,
            build_mode: TreeBuildMode::Deterministic,
        }
    }
}

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
    pub length_factor: f64,
    pub build_mode: TreeBuildMode,
    pub seed: u64,
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
        Self::new_with_config(
            node_count,
            tails,
            heads,
            lengths,
            DynamicTreeConfig::randomized(seed, rebuild_every, update_budget),
        )
    }

    pub fn new_with_config(
        node_count: usize,
        tails: Vec<u32>,
        heads: Vec<u32>,
        lengths: Vec<f64>,
        config: DynamicTreeConfig,
    ) -> Result<Self, TreeError> {
        let edge_count = lengths.len();
        let tree = LowStretchTree::build_low_stretch_with_mode(
            node_count,
            &tails,
            &heads,
            &lengths,
            config.build_mode,
            config.seed,
        )?;
        Ok(Self {
            node_count,
            tails,
            heads,
            lengths,
            active: vec![true; edge_count],
            rebuild_every: config.rebuild_every.max(1),
            update_budget: config.update_budget,
            update_count: 0,
            length_factor: config.length_factor.max(1.0),
            build_mode: config.build_mode,
            seed: config.seed,
            tree,
        })
    }

    pub fn new_deterministic(
        node_count: usize,
        tails: Vec<u32>,
        heads: Vec<u32>,
        lengths: Vec<f64>,
        rebuild_every: usize,
        update_budget: usize,
    ) -> Result<Self, TreeError> {
        Self::new_with_config(
            node_count,
            tails,
            heads,
            lengths,
            DynamicTreeConfig::deterministic(rebuild_every, update_budget),
        )
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
        self.update_count = self.update_count.saturating_add(1);
        if *current > 0.0 && length > 0.0 {
            let ratio = if length > *current {
                length / *current
            } else {
                *current / length
            };
            if ratio > self.length_factor {
                self.update_count = self.update_count.saturating_add(1);
            }
        }
        *current = length;
        true
    }

    pub fn should_rebuild(&self, step: usize) -> bool {
        if self.update_budget > 0 && self.update_count >= self.update_budget {
            return true;
        }
        step.is_multiple_of(self.rebuild_every)
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
        self.tree = LowStretchTree::build_low_stretch_with_mode(
            self.node_count,
            &tails,
            &heads,
            &lengths,
            self.build_mode,
            seed,
        )?;
        self.update_count = 0;
        Ok(())
    }

    pub fn rebuild_with_step(&mut self, step: usize) -> Result<(), TreeError> {
        let seed = match self.build_mode {
            TreeBuildMode::Randomized => self.seed ^ step as u64,
            TreeBuildMode::Deterministic => 0,
        };
        self.rebuild(seed)
    }

    pub fn force_rebuild(&mut self, step: usize) -> Result<(), TreeError> {
        self.update_count = self.update_budget.max(1);
        self.rebuild_with_step(step)
    }

    pub fn update_from_snapshot(
        &mut self,
        step: usize,
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        lengths: &[f64],
    ) -> Result<bool, TreeError> {
        if node_count != self.node_count
            || tails.len() != self.tails.len()
            || heads.len() != self.heads.len()
            || lengths.len() != self.lengths.len()
            || tails != self.tails.as_slice()
            || heads != self.heads.as_slice()
        {
            self.node_count = node_count;
            self.tails = tails.to_vec();
            self.heads = heads.to_vec();
            self.lengths = lengths.to_vec();
            self.active = vec![true; lengths.len()];
            self.update_count = self.update_budget.max(1);
        } else {
            for (edge_id, &length) in lengths.iter().enumerate() {
                self.update_length(edge_id, length);
            }
        }

        if self.should_rebuild(step) {
            self.rebuild_with_step(step)?;
            return Ok(true);
        }
        Ok(false)
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

    #[test]
    fn deterministic_snapshot_rebuilds() {
        let tails = vec![0, 1, 2];
        let heads = vec![1, 2, 0];
        let lengths = vec![1.0, 1.0, 1.0];
        let mut tree =
            DynamicTree::new_deterministic(3, tails.clone(), heads.clone(), lengths, 5, 1).unwrap();
        let updated_lengths = vec![2.5, 1.0, 1.0];
        let rebuilt = tree
            .update_from_snapshot(2, 3, &tails, &heads, &updated_lengths)
            .unwrap();
        assert!(rebuilt);
        assert_eq!(tree.tree.parent.len(), 3);
    }
}
