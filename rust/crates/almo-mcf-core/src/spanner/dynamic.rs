use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::graph::{Edge, EdgeId, NodeId};
use crate::spanner::construction::{build_level, contract_layer, greedy_cluster, sparsify_level};
use crate::spanner::construction::{LevelEdge, Spanner};

#[derive(Debug, Clone, Copy)]
pub struct SpannerUpdates {
    pub changed_edges: usize,
    pub recourse: usize,
}

impl Spanner {
    pub fn batch_update(&mut self, inserts: &[Edge], deletes: &[Edge]) -> SpannerUpdates {
        let mut updated = 0usize;
        for edge in deletes {
            if self.delete_edge(edge.tail.0, edge.head.0) {
                updated += 1;
            }
        }
        for edge in inserts {
            self.insert_edge(edge.tail.0, edge.head.0);
            updated += 1;
        }
        self.recourse_this_batch += updated;
        self.total_updates += updated;
        let limit = self
            .config
            .recourse_limit(self.levels.first().map(|l| l.node_count).unwrap_or(1));
        if self.recourse_this_batch >= limit {
            self.rebuild_from_level0();
            self.rebuilds += 1;
            self.recourse_this_batch = 0;
        } else {
            self.process_pending();
        }
        SpannerUpdates {
            changed_edges: updated,
            recourse: self.recourse_this_batch,
        }
    }

    pub fn split_vertex(&mut self, vertex: NodeId, new_vertex: NodeId) -> bool {
        let new_id = new_vertex.0;
        let old_id = vertex.0;
        let mut changed = false;
        for level in &mut self.levels {
            if old_id >= level.node_count {
                continue;
            }
            if new_id >= level.node_count {
                level.node_count = new_id + 1;
                level.adjacency.resize_with(level.node_count, Vec::new);
                level
                    .sparse_adjacency
                    .resize_with(level.node_count, Vec::new);
                let cluster = level.cluster_map.get(old_id).copied().unwrap_or(0);
                while level.cluster_map.len() < level.node_count {
                    level.cluster_map.push(cluster);
                }
                changed = true;
            }
        }
        if changed {
            self.recourse_this_batch += 1;
        }
        changed
    }

    pub fn query_embedding(&mut self, edge_id: EdgeId) -> Vec<EdgeId> {
        self.query_embedding_steps(edge_id)
            .unwrap_or_default()
            .into_iter()
            .map(|step| EdgeId(step.edge))
            .collect()
    }

    fn insert_edge(&mut self, u: usize, v: usize) {
        let Some(level0) = self.levels.first_mut() else {
            return;
        };
        let edge_id = level0.edges.len();
        level0.edges.push(LevelEdge { u, v, active: true });
        if u >= level0.node_count || v >= level0.node_count {
            level0.node_count = level0.node_count.max(u + 1).max(v + 1);
            level0.adjacency.resize_with(level0.node_count, Vec::new);
            level0
                .sparse_adjacency
                .resize_with(level0.node_count, Vec::new);
            while level0.cluster_map.len() < level0.node_count {
                level0.cluster_map.push(level0.cluster_map.len());
            }
        }
        level0.adjacency[u].push(edge_id);
        level0.adjacency[v].push(edge_id);
        level0.sparse_edges.push(edge_id);
        level0.sparse_adjacency[u].push(edge_id);
        level0.sparse_adjacency[v].push(edge_id);
        self.base_edge_map
            .entry(ordered_pair(u, v))
            .or_default()
            .push(edge_id);
        self.pending_levels[0].insert(edge_id);
    }

    fn delete_edge(&mut self, u: usize, v: usize) -> bool {
        let Some(level0) = self.levels.first_mut() else {
            return false;
        };
        let key = ordered_pair(u, v);
        let Some(edges) = self.base_edge_map.get(&key) else {
            return false;
        };
        for &edge_id in edges {
            if let Some(edge) = level0.edges.get_mut(edge_id) {
                if edge.active {
                    edge.active = false;
                    self.pending_levels[0].insert(edge_id);
                    return true;
                }
            }
        }
        false
    }

    fn process_pending(&mut self) {
        let mut heap = BinaryHeap::new();
        for level in 0..self.pending_levels.len() {
            if !self.pending_levels[level].is_empty() {
                heap.push(Reverse(level));
            }
        }
        while let Some(Reverse(level)) = heap.pop() {
            if self.pending_levels[level].is_empty() {
                continue;
            }
            let node_count = self.levels[level].node_count;
            let target = ((node_count.max(2) as f64).ln().ceil() as usize).max(1);
            let new_cluster = greedy_cluster(
                node_count,
                &self.levels[level].edges,
                &self.levels[level].adjacency,
                target,
            );
            let size_bound = self.level_size_bound(node_count);
            let (new_sparse, new_sparse_adj) = sparsify_level(
                node_count,
                &self.levels[level].edges,
                &self.levels[level].adjacency,
                &new_cluster,
                size_bound,
            );
            let changed = count_edge_changes(&self.levels[level].sparse_edges, &new_sparse);
            self.recourse_this_batch = self.recourse_this_batch.saturating_add(changed);
            self.levels[level].cluster_map = new_cluster;
            self.levels[level].sparse_edges = new_sparse;
            self.levels[level].sparse_adjacency = new_sparse_adj;
            self.pending_levels[level].clear();
            if level + 1 < self.levels.len() {
                self.pending_levels[level + 1].extend(0..self.levels[level + 1].edges.len());
                heap.push(Reverse(level + 1));
            }
        }
    }

    fn level_size_bound(&self, node_count: usize) -> usize {
        let base = self.levels.first().map(|l| l.node_count).unwrap_or(1);
        let ratio = (node_count as f64) / (base as f64);
        ((self.config.size_bound as f64) * ratio)
            .ceil()
            .max(node_count as f64) as usize
    }

    fn rebuild_from_level0(&mut self) {
        let Some(level0) = self.levels.first() else {
            return;
        };
        let edges = level0.edges.clone();
        let adjacency = level0.adjacency.clone();
        let node_count = level0.node_count;
        self.levels.clear();
        self.pending_levels.clear();
        let level = build_level(node_count, edges, adjacency, &self.config);
        self.levels.push(level);
        self.pending_levels.push(std::collections::HashSet::new());
        for _ in 1..self.config.level_count {
            let (next_edges, next_adj, next_nodes) = contract_layer(self.levels.last().unwrap());
            if next_nodes <= 1 {
                break;
            }
            let level = build_level(next_nodes, next_edges, next_adj, &self.config);
            self.levels.push(level);
            self.pending_levels.push(std::collections::HashSet::new());
        }
        self.rebuild_edge_map();
        self.embeddings.clear();
        let originals: Vec<(EdgeId, usize, usize)> = self
            .original_edges
            .iter()
            .map(|(&edge_id, &(u, v))| (edge_id, u, v))
            .collect();
        for (edge_id, u, v) in originals {
            if self.embed_edge_with_bfs(edge_id, u, v).is_none() {
                self.ensure_direct_edge(u, v, edge_id);
            }
        }
    }
}

fn count_edge_changes(old_edges: &[usize], new_edges: &[usize]) -> usize {
    let old: std::collections::HashSet<_> = old_edges.iter().copied().collect();
    let new: std::collections::HashSet<_> = new_edges.iter().copied().collect();
    let removed = old.difference(&new).count();
    let added = new.difference(&old).count();
    removed + added
}

fn ordered_pair(u: usize, v: usize) -> (usize, usize) {
    if u <= v {
        (u, v)
    } else {
        (v, u)
    }
}
