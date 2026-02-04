use std::collections::HashSet;

use crate::graph::{Graph, NodeId};
use crate::trees::lsst::{build_lsst_deterministic, estimate_average_stretch_deterministic, Tree};
use crate::trees::TreeError;

#[derive(Debug, Clone)]
pub struct Forest {
    pub trees: Vec<Tree>,
}

#[derive(Debug, Clone)]
pub struct UpdateBatch {
    pub deletions: Vec<(NodeId, NodeId)>,
}

impl UpdateBatch {
    pub fn new() -> Self {
        Self {
            deletions: Vec::new(),
        }
    }
}

impl Default for UpdateBatch {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
pub struct Shifter {
    forests: Vec<Forest>,
    index: usize,
    pub lazy_updates: Vec<UpdateBatch>,
    stretch_threshold: f64,
}

impl Shifter {
    pub fn new(forests: Vec<Forest>, stretch_threshold: f64) -> Self {
        let lazy_updates = vec![UpdateBatch::new(); forests.len().max(1)];
        Self {
            forests,
            index: 0,
            lazy_updates,
            stretch_threshold: stretch_threshold.max(1.0),
        }
    }

    pub fn next_forest(&mut self) -> Option<&Forest> {
        if self.forests.is_empty() {
            return None;
        }
        self.index = (self.index + 1) % self.forests.len();
        Some(&self.forests[self.index])
    }

    pub fn current_forest(&self) -> Option<&Forest> {
        self.forests.get(self.index)
    }

    pub fn apply_shift(&mut self, graph: &Graph, updates: &UpdateBatch) -> Result<(), TreeError> {
        if self.forests.is_empty() {
            return Ok(());
        }
        let idx = self.index;
        let forest = self.forests.get_mut(idx).expect("forest exists");
        let mut dirty = false;
        for &(u, v) in &updates.deletions {
            for tree in &mut forest.trees {
                dirty |= tree.delete_edge(u, v);
            }
        }
        if dirty {
            let estimated = estimate_average_stretch_deterministic(graph, &forest.trees, 32)?;
            if estimated > self.stretch_threshold {
                let rebuilt = Forest {
                    trees: build_lsst_deterministic(graph, 1.1)?,
                };
                self.forests[idx] = rebuilt;
            }
        }
        Ok(())
    }
}

pub fn precompute_forests(graph: &Graph, s: usize) -> Result<Vec<Forest>, TreeError> {
    let s = s.max(1);
    let node_count = graph.node_count();
    if node_count == 0 {
        return Err(TreeError::EmptyGraph);
    }

    let groups = partition_vertices(node_count, s);
    let mut forests = Vec::with_capacity(s);
    for group in &groups {
        let subgraph = induced_subgraph(graph, group);
        let trees = build_lsst_deterministic(&subgraph, 1.1)?;
        forests.push(Forest { trees });
    }
    Ok(forests)
}

fn partition_vertices(node_count: usize, s: usize) -> Vec<Vec<usize>> {
    let mut groups = vec![Vec::new(); s];
    for node in 0..node_count {
        groups[node % s].push(node);
    }
    groups
}

fn induced_subgraph(graph: &Graph, nodes: &[usize]) -> Graph {
    let mut subgraph = Graph::new(graph.node_count());
    let node_set: HashSet<usize> = nodes.iter().copied().collect();
    for (_, edge) in graph.edges() {
        let u = edge.tail.0;
        let v = edge.head.0;
        if node_set.contains(&u) && node_set.contains(&v) {
            let _ = subgraph.add_edge(edge.tail, edge.head, edge.lower, edge.upper, edge.cost);
        }
    }
    subgraph
}

pub fn build_deterministic_forests(
    graph: &Graph,
    s: usize,
    per_group: usize,
) -> Result<Vec<Forest>, TreeError> {
    let base = precompute_forests(graph, s)?;
    let mut expanded = Vec::with_capacity(s);
    for (group_idx, forest) in base.into_iter().enumerate() {
        let mut trees = forest.trees;
        while trees.len() < per_group.max(1) {
            let idx = trees.len();
            let gamma = 1.0 + 0.01 * ((idx + group_idx + 1) as f64);
            let more = build_lsst_deterministic(graph, gamma)?;
            trees.extend(more);
        }
        expanded.push(Forest { trees });
    }
    Ok(expanded)
}

pub fn apply_lazy_updates(
    shifter: &mut Shifter,
    graph: &Graph,
    updates: UpdateBatch,
) -> Result<(), TreeError> {
    if shifter.forests.is_empty() {
        return Ok(());
    }
    let idx = shifter.index;
    if let Some(batch) = shifter.lazy_updates.get_mut(idx) {
        batch.deletions.extend(updates.deletions);
    }
    let batch = shifter
        .lazy_updates
        .get(idx)
        .cloned()
        .unwrap_or_else(UpdateBatch::new);
    shifter.apply_shift(graph, &batch)?;
    if let Some(batch) = shifter.lazy_updates.get_mut(idx) {
        batch.deletions.clear();
    }
    Ok(())
}

pub fn stable_partition_updates(
    graph: &Graph,
    updates: &[UpdateBatch],
    max_per_shift: usize,
) -> Vec<UpdateBatch> {
    let mut buckets = Vec::new();
    let mut current = UpdateBatch::new();
    for batch in updates {
        for deletion in &batch.deletions {
            current.deletions.push(*deletion);
            if current.deletions.len() >= max_per_shift.max(1) {
                buckets.push(current);
                current = UpdateBatch::new();
            }
        }
    }
    if !current.deletions.is_empty() {
        buckets.push(current);
    }
    if buckets.is_empty() {
        buckets.push(UpdateBatch::new());
    }
    let _ = graph;
    buckets
}
