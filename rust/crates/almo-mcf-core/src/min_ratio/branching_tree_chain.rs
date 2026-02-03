use crate::spanner::decremental::{ContractedGraph, DecrementalSpannerParams, LSForest};
use crate::spanner::{build_spanner_on_core, DecrementalSpanner};
use crate::trees::{LowStretchTree, TreeBuildMode, TreeError};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct LevelData {
    pub forest: LSForest,
    pub core: ContractedGraph,
    pub spanner: DecrementalSpanner,
    pub vertex_map: Vec<usize>,
    pub tails: Vec<u32>,
    pub heads: Vec<u32>,
    pub lengths: Vec<f64>,
    pub overstretches: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct BranchingTreeChain {
    pub levels: Vec<LevelData>,
    pub reduction_factor: f64,
}

#[derive(Debug, Clone)]
pub struct Cycle {
    pub edges: Vec<(usize, i8)>,
}

#[derive(Debug, Clone)]
pub struct Circulation {
    pub tree_paths: Vec<Vec<(usize, i8)>>,
    pub off_tree_edges: Vec<(usize, i8)>,
    pub flow_value: f64,
}

#[derive(Debug, Clone)]
pub struct CycleMetrics {
    pub tilde_g: f64,
    pub tilde_length: f64,
}

#[derive(Debug, Clone)]
pub struct CycleCandidate {
    pub cycle: Cycle,
    pub ratio: f64,
    pub metrics: CycleMetrics,
}

impl BranchingTreeChain {
    pub fn default_level_count(node_count: usize) -> usize {
        if node_count <= 1 {
            return 1;
        }
        let log_n = (node_count as f64).ln();
        let log_inv = (1.0_f64 / 0.875_f64).ln();
        let levels = (log_n / log_inv).ceil() as usize;
        levels.max(1)
    }

    pub fn build(
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        lengths: &[f64],
        level_count: Option<usize>,
        reduction_factor: f64,
        deterministic: bool,
    ) -> Result<(Self, Vec<f64>), TreeError> {
        if node_count == 0 {
            return Err(TreeError::EmptyGraph);
        }
        if tails.len() != heads.len() || tails.len() != lengths.len() {
            return Err(TreeError::MissingEdgeLengths);
        }
        let mut levels = Vec::new();
        let mut current_nodes = node_count;
        let mut current_tails = tails.to_vec();
        let mut current_heads = heads.to_vec();
        let mut current_lengths = lengths.to_vec();
        let mut logs = Vec::new();
        let target_levels = level_count.unwrap_or_else(|| Self::default_level_count(node_count));
        let build_mode = if deterministic {
            TreeBuildMode::Deterministic
        } else {
            TreeBuildMode::Randomized
        };

        for _ in 0..target_levels.max(1) {
            let start = Instant::now();
            let tree = LowStretchTree::build_low_stretch_with_mode(
                current_nodes,
                &current_tails,
                &current_heads,
                &current_lengths,
                build_mode,
                0,
            )?;
            let overstretches =
                compute_overstretches(&tree, &current_tails, &current_heads, &current_lengths);
            let forest =
                LSForest::from_tree(&tree, &current_tails, &current_heads, &current_lengths);
            let (core, vertex_map) = contract_graph(
                current_nodes,
                &current_tails,
                &current_heads,
                &current_lengths,
                &tree,
            );
            let params = DecrementalSpannerParams::from_graph(core.node_count, core.tails.len());
            let spanner = build_spanner_on_core(&core, &forest, params);

            logs.push(start.elapsed().as_secs_f64());
            levels.push(LevelData {
                forest,
                core,
                spanner,
                vertex_map,
                tails: current_tails.clone(),
                heads: current_heads.clone(),
                lengths: current_lengths.clone(),
                overstretches,
            });

            let next_node_count = levels
                .last()
                .map(|level| level.core.node_count)
                .unwrap_or(0);
            if next_node_count <= 1 {
                break;
            }
            if !levels.is_empty() && levels.len() > 1 {
                let prev = levels[levels.len() - 2].core.node_count;
                let expected = (prev as f64 * reduction_factor).ceil() as usize;
                assert!(
                    next_node_count <= expected || next_node_count <= 1,
                    "hierarchy reduction failed: {} -> {}",
                    prev,
                    next_node_count
                );
            }
            current_nodes = next_node_count;
            current_tails = levels
                .last()
                .map(|level| level.core.tails.clone())
                .unwrap_or_default();
            current_heads = levels
                .last()
                .map(|level| level.core.heads.clone())
                .unwrap_or_default();
            current_lengths = levels
                .last()
                .map(|level| level.core.lengths.clone())
                .unwrap_or_default();
        }

        Ok((
            Self {
                levels,
                reduction_factor,
            },
            logs,
        ))
    }

    pub fn update_level_zero_lengths(&mut self, lengths: &[f64]) {
        if let Some(level) = self.levels.first_mut() {
            if lengths.len() == level.lengths.len() {
                level.lengths.clone_from_slice(lengths);
            }
        }
    }

    pub fn extract_fundamental_cycle(
        &mut self,
        level_index: usize,
        off_tree_edge: usize,
    ) -> Option<Cycle> {
        let level = self.levels.get(level_index)?;
        if off_tree_edge >= level.tails.len() {
            return None;
        }
        if level.tails.len() != level.heads.len() {
            return None;
        }
        let start = level.tails[off_tree_edge] as usize;
        let end = level.heads[off_tree_edge] as usize;
        let path_edges = level
            .forest
            .path_between(start, end)
            .unwrap_or_default();
        let mut cycle_edges = Vec::new();
        if path_edges.len() > 4 {
            if let Some(spanner_cycle) = self.extract_spanner_path(level_index, start, end) {
                cycle_edges.extend(spanner_cycle);
            }
        } else if path_edges.len() >= 2 {
            for window in path_edges.windows(2) {
                if let Some(edge_id) =
                    find_edge_id(&level.tails, &level.heads, window[0], window[1])
                {
                    cycle_edges.push((edge_id, 1));
                }
            }
        }
        cycle_edges.push((off_tree_edge, 1));
        if cycle_edges.is_empty() {
            None
        } else {
            Some(Cycle { edges: cycle_edges })
        }
    }

    fn extract_spanner_path(
        &mut self,
        level_index: usize,
        start: usize,
        end: usize,
    ) -> Option<Vec<(usize, i8)>> {
        let level = self.levels.get_mut(level_index)?;
        let path = level.spanner.embed_path(start, end)?;
        Some(
            path.steps
                .iter()
                .map(|step| (step.edge, step.dir))
                .collect(),
        )
    }

    pub fn batch_extract(&mut self, level_index: usize, edges: &[usize]) -> Vec<Cycle> {
        let mut cycles = Vec::new();
        for &edge_id in edges {
            if let Some(cycle) = self.extract_fundamental_cycle(level_index, edge_id) {
                cycles.push(cycle);
            }
        }
        cycles
    }

    pub fn compute_cycle_metrics(
        &self,
        level_index: usize,
        cycle: &Cycle,
        gradients: &[f64],
        lengths: &[f64],
    ) -> Option<CycleMetrics> {
        let level = self.levels.get(level_index)?;
        if cycle.edges.is_empty() {
            return None;
        }
        let mut tilde_g = 0.0;
        let mut tilde_length = 0.0;
        let mut c_g = 0.0;
        let mut c_l = 0.0;
        for &(edge_id, dir) in &cycle.edges {
            let g = gradients.get(edge_id)?;
            let l = lengths.get(edge_id)?.abs();
            let stretch = level
                .overstretches
                .get(edge_id)
                .copied()
                .unwrap_or(1.0)
                .max(1.0);
            let signed = (dir as f64) * g;
            let y = signed - c_g;
            let t = tilde_g + y;
            c_g = (t - tilde_g) - y;
            tilde_g = t;
            let over = l * stretch;
            let y_l = over - c_l;
            let t_l = tilde_length + y_l;
            c_l = (t_l - tilde_length) - y_l;
            tilde_length = t_l;
        }
        if !tilde_length.is_finite() || tilde_length <= 0.0 || !tilde_g.is_finite() {
            return None;
        }
        Some(CycleMetrics {
            tilde_g,
            tilde_length,
        })
    }

    pub fn find_max_ratio_cycle(
        &mut self,
        gradients: &[f64],
        lengths: &[f64],
        max_edges_per_level: usize,
    ) -> Option<CycleCandidate> {
        let mut best: Option<CycleCandidate> = None;
        for level_index in 0..self.levels.len() {
            let mut candidates = Vec::new();
            let mut count = 0;
            let sparse_edges = self
                .levels
                .get(level_index)
                .and_then(|level| level.spanner.sparse_edge_ids())
                .map(|edges| edges.to_vec())
                .unwrap_or_default();
            let fallback_edges = self
                .levels
                .get(level_index)
                .map(|level| (0..level.tails.len()).collect::<Vec<_>>())
                .unwrap_or_default();
            let edge_candidates = if sparse_edges.is_empty() {
                fallback_edges
            } else {
                sparse_edges
            };
            for edge_id in edge_candidates {
                if count >= max_edges_per_level {
                    break;
                }
                let Some(level) = self.levels.get(level_index) else {
                    break;
                };
                if level.spanner.edge_info(edge_id).is_none() {
                    continue;
                }
                if let Some(cycle) = self.extract_fundamental_cycle(level_index, edge_id) {
                    candidates.push(cycle);
                    count += 1;
                }
            }
            if candidates.is_empty() {
                continue;
            }
            for cycle in candidates {
                let metrics =
                    self.compute_cycle_metrics(level_index, &cycle, gradients, lengths)?;
                if metrics.tilde_g >= 0.0 {
                    continue;
                }
                let ratio = -metrics.tilde_g / metrics.tilde_length;
                let candidate = CycleCandidate {
                    cycle,
                    ratio,
                    metrics,
                };
                best = Some(match best {
                    Some(prev) => {
                        if candidate.ratio > prev.ratio {
                            candidate
                        } else {
                            prev
                        }
                    }
                    None => candidate,
                });
            }
        }
        if best.is_some() {
            return best;
        }
        let level_index = 0;
        let level = self.levels.get(level_index)?;
        let (cycle_edges, _ratio) = exact_best_cycle(
            level.tails.len().max(1),
            &level.tails,
            &level.heads,
            gradients,
            lengths,
        )?;
        let cycle = Cycle { edges: cycle_edges };
        let metrics = self
            .compute_cycle_metrics(level_index, &cycle, gradients, lengths)
            .unwrap_or_else(|| {
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                for &(edge_id, dir) in &cycle.edges {
                    numerator += (dir as f64) * gradients[edge_id];
                    denominator += lengths[edge_id].abs();
                }
                CycleMetrics {
                    tilde_g: numerator,
                    tilde_length: denominator.max(1.0),
                }
            });
        Some(CycleCandidate {
            cycle,
            ratio: -metrics.tilde_g / metrics.tilde_length,
            metrics,
        })
    }

    pub fn decompose_to_circulation(
        &self,
        cycle: &Cycle,
        gradients: &[f64],
        lengths: &[f64],
    ) -> Option<Circulation> {
        if cycle.edges.is_empty() {
            return None;
        }
        let mut min_residual = f64::INFINITY;
        for &(edge_id, _) in &cycle.edges {
            let residual = lengths.get(edge_id)?.abs();
            min_residual = min_residual.min(residual);
        }
        if !min_residual.is_finite() {
            return None;
        }
        let mut tree_paths = Vec::new();
        let mut off_tree_edges = Vec::new();
        let level = self.levels.first()?;
        for &(edge_id, dir) in &cycle.edges {
            if edge_id >= level.tails.len() {
                continue;
            }
            if level
                .forest
                .path_between(level.tails[edge_id] as usize, level.heads[edge_id] as usize)
                .is_some()
            {
                tree_paths.push(vec![(edge_id, dir)]);
            } else {
                off_tree_edges.push((edge_id, dir));
            }
        }
        if tree_paths.is_empty() {
            tree_paths.push(cycle.edges.clone());
        }
        let _ = gradients;
        Some(Circulation {
            tree_paths,
            off_tree_edges,
            flow_value: min_residual.max(0.0),
        })
    }
}

fn compute_overstretches(
    tree: &LowStretchTree,
    tails: &[u32],
    heads: &[u32],
    lengths: &[f64],
) -> Vec<f64> {
    let mut overstretches = vec![1.0; lengths.len()];
    for (edge_id, overstretch) in overstretches.iter_mut().enumerate() {
        if let Some(stretch) = tree.edge_stretch(edge_id, tails, heads, lengths) {
            *overstretch = stretch.max(1.0);
        }
    }
    overstretches
}

fn contract_graph(
    node_count: usize,
    tails: &[u32],
    heads: &[u32],
    lengths: &[f64],
    tree: &LowStretchTree,
) -> (ContractedGraph, Vec<usize>) {
    let mut component_map = vec![usize::MAX; node_count];
    let mut supernodes: Vec<Vec<usize>> = Vec::new();
    let mut next_id = 0;
    for node in 0..node_count {
        let root = tree.root[node];
        let comp = component_map[root];
        let assigned = if comp == usize::MAX {
            component_map[root] = next_id;
            supernodes.push(vec![node]);
            next_id += 1;
            next_id - 1
        } else {
            comp
        };
        if assigned < supernodes.len() && node != root {
            supernodes[assigned].push(node);
        }
        component_map[node] = assigned;
    }
    let mut new_tails = Vec::new();
    let mut new_heads = Vec::new();
    let mut new_lengths = Vec::new();
    for edge_id in 0..tails.len() {
        let u = tails[edge_id] as usize;
        let v = heads[edge_id] as usize;
        let cu = component_map[u];
        let cv = component_map[v];
        if cu == cv {
            continue;
        }
        new_tails.push(cu as u32);
        new_heads.push(cv as u32);
        new_lengths.push(lengths[edge_id]);
    }
    (
        ContractedGraph {
            node_count: next_id.max(1),
            tails: new_tails,
            heads: new_heads,
            lengths: new_lengths,
            supernodes,
        },
        component_map,
    )
}

fn find_edge_id(tails: &[u32], heads: &[u32], u: usize, v: usize) -> Option<usize> {
    for (edge_id, (&tail, &head)) in tails.iter().zip(heads.iter()).enumerate() {
        if (tail as usize == u && head as usize == v) || (tail as usize == v && head as usize == u)
        {
            return Some(edge_id);
        }
    }
    None
}

pub fn exact_best_cycle(
    node_count: usize,
    tails: &[u32],
    heads: &[u32],
    gradients: &[f64],
    lengths: &[f64],
) -> Option<(Vec<(usize, i8)>, f64)> {
    if node_count == 0 {
        return None;
    }
    let mut best_ratio = f64::INFINITY;
    let mut best_cycle = None;
    for edge_id in 0..tails.len() {
        if gradients[edge_id] >= 0.0 {
            continue;
        }
        let tail = tails[edge_id] as usize;
        let head = heads[edge_id] as usize;
        let Some(path) = shortest_path_edges(node_count, tails, heads, lengths, head, tail) else {
            continue;
        };
        let mut cycle_edges = path.clone();
        cycle_edges.push((edge_id, 1));
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for &(eid, dir) in &cycle_edges {
            numerator += (dir as f64) * gradients[eid];
            denominator += lengths[eid].abs();
        }
        if denominator > 0.0 {
            let ratio = numerator / denominator;
            if ratio < best_ratio {
                best_ratio = ratio;
                best_cycle = Some(cycle_edges);
            }
        }
    }
    best_cycle.map(|cycle| (cycle, best_ratio))
}

fn shortest_path_edges(
    node_count: usize,
    tails: &[u32],
    heads: &[u32],
    lengths: &[f64],
    start: usize,
    end: usize,
) -> Option<Vec<(usize, i8)>> {
    if start == end {
        return Some(Vec::new());
    }
    let mut dist = vec![f64::INFINITY; node_count];
    let mut prev = vec![None; node_count];
    dist[start] = 0.0;
    for _ in 0..node_count.saturating_sub(1) {
        let mut updated = false;
        for (edge_id, (&tail, &head)) in tails.iter().zip(heads.iter()).enumerate() {
            let weight = lengths.get(edge_id).copied().unwrap_or(1.0).abs();
            let u = tail as usize;
            let v = head as usize;
            if dist[u] + weight < dist[v] {
                dist[v] = dist[u] + weight;
                prev[v] = Some((u, edge_id));
                updated = true;
            }
            if dist[v] + weight < dist[u] {
                dist[u] = dist[v] + weight;
                prev[u] = Some((v, edge_id));
                updated = true;
            }
        }
        if !updated {
            break;
        }
    }
    prev[end]?;
    let mut edges = Vec::new();
    let mut current = end;
    while current != start {
        let (p, edge_id) = prev[current]?;
        let dir = if tails[edge_id] as usize == p && heads[edge_id] as usize == current {
            1
        } else {
            -1
        };
        edges.push((edge_id, dir));
        current = p;
    }
    edges.reverse();
    Some(edges)
}
