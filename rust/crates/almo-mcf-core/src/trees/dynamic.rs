use crate::trees::{LowStretchTree, TreeBuildMode, TreeError};

#[derive(Debug, Clone)]
pub struct TreeChainLevel {
    pub level: usize,
    pub node_count: usize,
    pub trees: Vec<LowStretchTree>,
    pub node_map: Vec<usize>,
    pub chains: Vec<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct DynamicTreeChainConfig {
    pub seed: u64,
    pub max_levels: usize,
    pub trees_per_level: usize,
    pub build_mode: TreeBuildMode,
}

impl DynamicTreeChainConfig {
    pub fn deterministic(max_levels: usize, trees_per_level: usize) -> Self {
        Self {
            seed: 0,
            max_levels: max_levels.max(1),
            trees_per_level: trees_per_level.max(1),
            build_mode: TreeBuildMode::Deterministic,
        }
    }

    pub fn randomized(seed: u64, max_levels: usize, trees_per_level: usize) -> Self {
        Self {
            seed,
            max_levels: max_levels.max(1),
            trees_per_level: trees_per_level.max(1),
            build_mode: TreeBuildMode::Randomized,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DynamicTreeChain {
    pub base_node_count: usize,
    pub levels: Vec<TreeChainLevel>,
}

impl DynamicTreeChain {
    pub fn build(
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        lengths: &[f64],
        config: DynamicTreeChainConfig,
    ) -> Result<Self, TreeError> {
        let mut levels = Vec::new();
        let mut current_node_count = node_count.max(1);
        let mut level_tails = tails.to_vec();
        let mut level_heads = heads.to_vec();
        let mut level_lengths = lengths.to_vec();
        let max_levels = config.max_levels.max(1);
        for level in 0..max_levels {
            let mut trees = Vec::with_capacity(config.trees_per_level);
            for tree_id in 0..config.trees_per_level {
                let seed = config.seed ^ ((level as u64) << 32) ^ (tree_id as u64);
                let tree = LowStretchTree::build_with_mode(
                    current_node_count,
                    &level_tails,
                    &level_heads,
                    &level_lengths,
                    config.build_mode,
                    seed,
                )?;
                trees.push(tree);
            }
            let node_map = build_node_map(current_node_count);
            let chains = build_chains_from_tree(trees.first());
            levels.push(TreeChainLevel {
                level,
                node_count: current_node_count,
                trees,
                node_map: node_map.clone(),
                chains,
            });
            current_node_count = node_map.iter().copied().max().unwrap_or(0) + 1;
            if current_node_count <= 1 {
                break;
            }
            let (next_tails, next_heads, next_lengths) =
                compress_edges_by_map(&node_map, &level_tails, &level_heads, &level_lengths);
            level_tails = next_tails;
            level_heads = next_heads;
            level_lengths = next_lengths;
        }
        Ok(Self {
            base_node_count: node_count,
            levels,
        })
    }

    pub fn rebuild_from_graph(
        &mut self,
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        lengths: &[f64],
        config: DynamicTreeChainConfig,
    ) -> Result<(), TreeError> {
        *self = Self::build(node_count, tails, heads, lengths, config)?;
        Ok(())
    }

    pub fn level_count(&self) -> usize {
        self.levels.len()
    }

    pub fn path_length(&self, level: usize, u: usize, v: usize) -> Option<f64> {
        let level = self.levels.get(level)?;
        level.trees.first()?.path_length(u, v)
    }

    pub fn embedding_path(
        &self,
        level: usize,
        u: usize,
        v: usize,
        tails: &[u32],
        heads: &[u32],
    ) -> Option<Vec<(usize, i8)>> {
        let level = self.levels.get(level)?;
        level.trees.first()?.path_edges(u, v, tails, heads)
    }
}

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

fn polynomial_length_bounds(node_count: usize) -> (f64, f64) {
    let node_count = node_count.max(1) as f64;
    let max_length = node_count.powi(4).max(1.0);
    let min_length = 1.0 / max_length;
    (min_length, max_length)
}

fn clamp_polynomial_length(node_count: usize, length: f64) -> f64 {
    let (min_length, max_length) = polynomial_length_bounds(node_count);
    let length = length.abs();
    let length = if length.is_finite() {
        length
    } else {
        max_length
    };
    length.clamp(min_length, max_length)
}

fn build_node_map(node_count: usize) -> Vec<usize> {
    (0..node_count).map(|node| node / 2).collect()
}

fn compress_edges_by_map(
    map: &[usize],
    tails: &[u32],
    heads: &[u32],
    lengths: &[f64],
) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let mut next_tails = Vec::with_capacity(tails.len());
    let mut next_heads = Vec::with_capacity(heads.len());
    let mut next_lengths = Vec::with_capacity(lengths.len());
    for ((&tail, &head), &length) in tails.iter().zip(heads.iter()).zip(lengths.iter()) {
        let mapped_tail = *map.get(tail as usize).unwrap_or(&0) as u32;
        let mapped_head = *map.get(head as usize).unwrap_or(&0) as u32;
        if mapped_tail == mapped_head {
            continue;
        }
        next_tails.push(mapped_tail);
        next_heads.push(mapped_head);
        next_lengths.push(length);
    }
    (next_tails, next_heads, next_lengths)
}

fn build_chains_from_tree(tree: Option<&LowStretchTree>) -> Vec<Vec<usize>> {
    let Some(tree) = tree else {
        return Vec::new();
    };
    let node_count = tree.parent.len();
    let mut visited = vec![false; node_count];
    let mut chains = Vec::new();
    for start in 0..node_count {
        if visited[start] {
            continue;
        }
        let mut chain = Vec::new();
        let mut current = start;
        while !visited[current] {
            visited[current] = true;
            chain.push(current);
            let parent = tree.parent[current];
            if parent == current {
                break;
            }
            current = parent;
        }
        if !chain.is_empty() {
            chains.push(chain);
        }
    }
    chains
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
        let lengths: Vec<f64> = lengths
            .into_iter()
            .map(|length| clamp_polynomial_length(node_count, length))
            .collect();
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
        self.lengths
            .push(clamp_polynomial_length(self.node_count, length));
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
        let bounded = clamp_polynomial_length(self.node_count, length);
        if *current > 0.0 && bounded > 0.0 {
            let ratio = if bounded > *current {
                bounded / *current
            } else {
                *current / bounded
            };
            if ratio > self.length_factor {
                self.update_count = self.update_count.saturating_add(1);
            }
        }
        *current = bounded;
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
            self.lengths = lengths
                .iter()
                .map(|&length| clamp_polynomial_length(node_count, length))
                .collect();
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

    #[test]
    fn polynomial_bounds_clamp_lengths() {
        let tails = vec![0];
        let heads = vec![1];
        let lengths = vec![1e9];
        let mut tree = DynamicTree::new(2, tails, heads, lengths, 1, 4, 2).unwrap();
        let (min_len, max_len) = polynomial_length_bounds(2);
        assert!((tree.lengths[0] - max_len).abs() < 1e-9);
        tree.update_length(0, 1e-12);
        assert!((tree.lengths[0] - min_len).abs() < 1e-9);
    }

    #[test]
    fn tree_chain_builds_levels() {
        let tails = vec![0, 1, 2, 3];
        let heads = vec![1, 2, 3, 0];
        let lengths = vec![1.0, 2.0, 3.0, 4.0];
        let config = DynamicTreeChainConfig::deterministic(3, 2);
        let chain = DynamicTreeChain::build(4, &tails, &heads, &lengths, config).unwrap();
        assert!(chain.level_count() >= 1);
        assert_eq!(chain.levels[0].node_count, 4);
    }
}
