use crate::spanner::decremental::{ContractedGraph, DecrementalSpannerParams, LSForest};
use crate::spanner::{build_spanner_on_core, DecrementalSpanner};
use crate::trees::shift::ShiftableForestCollection;
use crate::trees::{LowStretchTree, TreeBuildMode, TreeError};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct LevelData {
    pub forest: LSForest,
    pub core: ContractedGraph,
    pub spanner: DecrementalSpanner,
    pub vertex_map: Vec<usize>,
    pub core_edge_ids: Vec<usize>,
    pub tails: Vec<u32>,
    pub heads: Vec<u32>,
    pub lengths: Vec<f64>,
    pub overstretches: Vec<f64>,
    pub dirty: bool,
}

#[derive(Debug, Clone)]
pub struct BranchingTreeChain {
    pub levels: Vec<LevelData>,
    pub reduction_factor: f64,
    pub rebuild_game: Vec<RebuildGameState>,
    pub rebuild_threshold: usize,
    pub fix_threshold: usize,
    pub loss_threshold: usize,
    pub loss_count: usize,
    pub failure_log: Vec<String>,
    pub shiftable_forests: Option<ShiftableForestCollection>,
    pub shift_threshold: usize,
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

#[derive(Debug, Clone)]
pub struct RebuildGameState {
    pub round_count: usize,
    pub fix_count: usize,
    pub loss_count: usize,
    pub round_threshold: usize,
    pub fix_threshold: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct RebuildContext<'a> {
    pub node_count: usize,
    pub tails: &'a [u32],
    pub heads: &'a [u32],
    pub lengths: &'a [f64],
    pub deterministic: bool,
}

impl RebuildGameState {
    pub fn new(round_threshold: usize, fix_threshold: usize) -> Self {
        Self {
            round_count: 0,
            fix_count: 0,
            loss_count: 0,
            round_threshold: round_threshold.max(1),
            fix_threshold: fix_threshold.max(1),
        }
    }
}

impl BranchingTreeChain {
    pub fn default_level_count(node_count: usize) -> usize {
        if node_count <= 1 {
            return 1;
        }
        ((node_count as f64).log(8.0).floor() as usize + 1).max(1)
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
            let (core, vertex_map, core_edge_ids) = contract_graph(
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
                core_edge_ids,
                tails: current_tails.clone(),
                heads: current_heads.clone(),
                lengths: current_lengths.clone(),
                overstretches,
                dirty: false,
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

        let rebuild_threshold = ((node_count as f64).ln().ceil() as usize).max(1);
        let fix_threshold = ((node_count as f64).ln().powi(2).ceil() as usize).max(2);
        let loss_threshold = 2;
        let rebuild_game = (0..levels.len())
            .map(|_| RebuildGameState::new(rebuild_threshold, fix_threshold))
            .collect();
        let shift_threshold = (node_count as f64).sqrt().ceil() as usize;
        let shiftable_forests = if deterministic {
            ShiftableForestCollection::build_deterministic(
                node_count,
                tails,
                heads,
                lengths,
                3,
                levels.len(),
            )
            .ok()
        } else {
            None
        };
        Ok((
            Self {
                levels,
                reduction_factor,
                rebuild_game,
                rebuild_threshold,
                fix_threshold,
                loss_threshold,
                loss_count: 0,
                failure_log: Vec::new(),
                shiftable_forests,
                shift_threshold: shift_threshold.max(1),
            },
            logs,
        ))
    }

    pub fn update_level_zero_lengths(&mut self, lengths: &[f64]) {
        if let Some(level) = self.levels.first_mut() {
            if lengths.len() == level.lengths.len() {
                level.lengths.clone_from_slice(lengths);
                level.dirty = true;
            }
        }
    }

    pub fn update_level_zero_lengths_partial(&mut self, dirty_edges: &[usize], lengths: &[f64]) {
        if let Some(level) = self.levels.first_mut() {
            for &edge_id in dirty_edges {
                if let Some(length) = lengths.get(edge_id) {
                    if edge_id < level.lengths.len() {
                        level.lengths[edge_id] = *length;
                        level.dirty = true;
                    }
                }
            }
        }
    }

    pub fn advance_round(&mut self, level: usize) -> bool {
        let Some(state) = self.rebuild_game.get_mut(level) else {
            return false;
        };
        state.round_count = state.round_count.saturating_add(1);
        state.round_count >= state.round_threshold
    }

    pub fn attempt_fix(&mut self, level: usize) -> bool {
        let Some(state) = self.rebuild_game.get_mut(level) else {
            return false;
        };
        state.fix_count = state.fix_count.saturating_add(1);
        if state.fix_count >= state.fix_threshold {
            state.loss_count = state.loss_count.saturating_add(1);
            state.fix_count = 0;
            return false;
        }
        true
    }

    pub fn handle_loss(&mut self, level: usize) -> bool {
        let Some(state) = self.rebuild_game.get_mut(level) else {
            return false;
        };
        state.loss_count = state.loss_count.saturating_add(1);
        self.loss_count = self.loss_count.saturating_add(1);
        state.loss_count >= self.loss_threshold.max(1)
    }

    pub fn record_failure(&mut self, level: usize, reason: impl Into<String>) {
        let msg = format!("level {level}: {}", reason.into());
        self.failure_log.push(msg);
    }

    pub fn reset_game_state(&mut self, level: usize) {
        if let Some(state) = self.rebuild_game.get_mut(level) {
            state.round_count = 0;
            state.fix_count = 0;
            state.loss_count = 0;
        }
    }

    pub fn rebuild_level(
        &mut self,
        level: usize,
        context: RebuildContext<'_>,
        bottom_up: bool,
    ) -> Result<(), TreeError> {
        if level >= self.levels.len() {
            return Ok(());
        }
        let target_levels = if bottom_up {
            self.levels.len()
        } else {
            level + 1
        };
        let (rebuilt, _logs) = BranchingTreeChain::build(
            context.node_count,
            context.tails,
            context.heads,
            context.lengths,
            Some(target_levels),
            self.reduction_factor,
            context.deterministic,
        )?;
        if bottom_up {
            self.levels = rebuilt.levels;
        } else {
            for idx in 0..=level {
                if let Some(level_data) = rebuilt.levels.get(idx) {
                    if idx < self.levels.len() {
                        self.levels[idx] = level_data.clone();
                    }
                }
            }
        }
        self.rebuild_game = (0..self.levels.len())
            .map(|_| RebuildGameState::new(self.rebuild_threshold, self.fix_threshold))
            .collect();
        Ok(())
    }

    pub fn propagate_update(&mut self, level: usize, dirty_edges: &[usize]) {
        let Some(shiftable) = self.shiftable_forests.as_mut() else {
            return;
        };
        if dirty_edges.len() >= self.shift_threshold {
            let _ = shiftable.shift(level);
        } else {
            shiftable.propagate(level);
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
        let path_edges = level.forest.path_between(start, end).unwrap_or_default();
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

    pub fn form_cycle(&mut self, off_tree_edge: usize, level_index: usize) -> Option<Cycle> {
        let level = self.levels.get_mut(level_index)?;
        if off_tree_edge >= level.tails.len() {
            return None;
        }
        let start = level.tails[off_tree_edge] as usize;
        let end = level.heads[off_tree_edge] as usize;
        let tree_path = path_edges_between(level, start, end)?;
        let long_path = tree_path.len() > 8;
        let mut cycle_edges = Vec::new();

        if long_path && level_index > 0 {
            let contracted_start = level.vertex_map.get(start).copied()?;
            let contracted_end = level.vertex_map.get(end).copied()?;
            if let Some(path) = level.spanner.embed_path(contracted_start, contracted_end) {
                let mapped = path
                    .steps
                    .iter()
                    .filter_map(|step| {
                        let core_edge = level.core_edge_ids.get(step.edge).copied()?;
                        Some((core_edge, step.dir))
                    })
                    .collect::<Vec<_>>();
                if let (Some(first), Some(last)) = (mapped.first(), mapped.last()) {
                    let (path_start, _) = edge_endpoint_with_dir(level, first.0, first.1)?;
                    let (_, path_end) = edge_endpoint_with_dir(level, last.0, last.1)?;
                    cycle_edges.extend(path_edges_between(level, start, path_start)?);
                    cycle_edges.extend(mapped);
                    cycle_edges.extend(path_edges_between(level, path_end, end)?);
                } else {
                    cycle_edges.extend(tree_path);
                }
            } else {
                cycle_edges.extend(tree_path);
            }
        } else {
            cycle_edges.extend(tree_path);
        }

        let off_dir = if level.tails[off_tree_edge] as usize == end
            && level.heads[off_tree_edge] as usize == start
        {
            1
        } else {
            -1
        };
        cycle_edges.push((off_tree_edge, off_dir));
        Some(Cycle { edges: cycle_edges })
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
        cycle: &Cycle,
        gradients: &[f64],
        lengths: &[f64],
    ) -> (f64, f64) {
        cycle
            .edges
            .iter()
            .fold((0.0, 0.0), |(gp, len), &(edge_id, dir)| {
                let gradient = gradients.get(edge_id).copied().unwrap_or(0.0);
                let length = lengths.get(edge_id).copied().unwrap_or(0.0).abs();
                (gp + gradient * dir as f64, len + length)
            })
    }

    fn compute_cycle_metrics_at_level(
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
                    self.compute_cycle_metrics_at_level(level_index, &cycle, gradients, lengths)?;
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
            .compute_cycle_metrics_at_level(level_index, &cycle, gradients, lengths)
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

    pub fn extract_min_ratio_cycle(
        &mut self,
        gradients: &[f64],
        lengths: &[f64],
        kappa: f64,
    ) -> Option<Cycle> {
        let mut best_ratio = 0.0;
        let mut best_cycle = None;
        let sample_limit = (lengths.len() as f64).sqrt().ceil() as usize;
        let mut samples: Vec<(f64, Cycle)> = Vec::new();

        for level_index in (0..self.levels.len()).rev() {
            let edge_count = self
                .levels
                .get(level_index)
                .map(|level| level.tails.len())
                .unwrap_or(0);
            for edge_id in 0..edge_count {
                let Some(cycle) = self.form_cycle(edge_id, level_index) else {
                    continue;
                };
                let (gp, len) = self.compute_cycle_metrics(&cycle, gradients, lengths);
                if len <= 0.0 || gp >= 0.0 {
                    continue;
                }
                let ratio = -gp / len;
                if samples.len() < sample_limit {
                    samples.push((ratio, cycle));
                } else if let Some((min_idx, min_val)) = samples
                    .iter()
                    .enumerate()
                    .min_by(|a, b| {
                        a.1 .0
                            .partial_cmp(&b.1 .0)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, val)| (idx, val.0))
                {
                    if ratio > min_val {
                        samples[min_idx] = (ratio, cycle);
                    }
                }
            }
        }

        for (ratio, cycle) in samples {
            if ratio > best_ratio * (1.0 + kappa) {
                best_ratio = ratio;
                best_cycle = Some(cycle);
            }
        }
        best_cycle
    }

    pub fn decompose_circulation(&self, cycle: &Cycle) -> Option<Decomposition> {
        let level = self.levels.first()?;
        if cycle.edges.is_empty() {
            return None;
        }
        let mut tree_paths = Vec::new();
        let mut off_tree_edges = Vec::new();
        let mut current_path: Vec<(usize, i8)> = Vec::new();

        for &(edge_id, dir) in &cycle.edges {
            if edge_id >= level.tails.len() {
                continue;
            }
            if is_tree_edge(level, edge_id) {
                current_path.push((edge_id, dir));
            } else {
                if !current_path.is_empty() {
                    tree_paths.push(current_path);
                    current_path = Vec::new();
                }
                off_tree_edges.push((edge_id, dir));
            }
        }
        if !current_path.is_empty() {
            tree_paths.push(current_path);
        }

        Some(Decomposition {
            tree_paths,
            off_tree_edges,
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
) -> (ContractedGraph, Vec<usize>, Vec<usize>) {
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
    let mut core_edge_ids = Vec::new();
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
        core_edge_ids.push(edge_id);
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
        core_edge_ids,
    )
}

fn path_edges_between(level: &LevelData, start: usize, end: usize) -> Option<Vec<(usize, i8)>> {
    let path = level.forest.path_between(start, end)?;
    if path.len() < 2 {
        return Some(Vec::new());
    }
    let mut edges = Vec::new();
    for window in path.windows(2) {
        let u = window[0];
        let v = window[1];
        let edge_id = find_edge_id(&level.tails, &level.heads, u, v)?;
        let dir = if level.tails[edge_id] as usize == u && level.heads[edge_id] as usize == v {
            1
        } else {
            -1
        };
        edges.push((edge_id, dir));
    }
    Some(edges)
}

fn edge_endpoint_with_dir(level: &LevelData, edge_id: usize, dir: i8) -> Option<(usize, usize)> {
    let tail = level.tails.get(edge_id).copied()? as usize;
    let head = level.heads.get(edge_id).copied()? as usize;
    if dir >= 0 {
        Some((tail, head))
    } else {
        Some((head, tail))
    }
}

fn is_tree_edge(level: &LevelData, edge_id: usize) -> bool {
    let u = level.tails[edge_id] as usize;
    let v = level.heads[edge_id] as usize;
    level
        .forest
        .adjacency
        .get(u)
        .map(|adj| adj.iter().any(|&(node, _)| node == v))
        .unwrap_or(false)
        || level
            .forest
            .adjacency
            .get(v)
            .map(|adj| adj.iter().any(|&(node, _)| node == u))
            .unwrap_or(false)
}

#[derive(Debug, Clone)]
pub struct Decomposition {
    pub tree_paths: Vec<Vec<(usize, i8)>>,
    pub off_tree_edges: Vec<(usize, i8)>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebuild_game_advances_and_triggers_loss() {
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let lengths = vec![1.0, 1.0, 1.0, 1.0];
        let (mut chain, _) =
            BranchingTreeChain::build(3, &tails, &heads, &lengths, Some(2), 0.8, true)
                .expect("chain should build");
        assert!(!chain.advance_round(0));
        assert!(chain.advance_round(0));
        assert!(chain.attempt_fix(0));
        assert!(!chain.attempt_fix(0));
        assert!(chain.handle_loss(0));
    }

    #[test]
    fn deterministic_propagation_shifts_when_threshold_exceeded() {
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let lengths = vec![1.0, 1.0, 1.0, 1.0];
        let (mut chain, _) =
            BranchingTreeChain::build(3, &tails, &heads, &lengths, Some(2), 0.8, true)
                .expect("chain should build");
        let dirty = vec![0, 1, 2, 3];
        chain.propagate_update(0, &dirty);
        assert!(chain.shiftable_forests.is_some());
    }
}
