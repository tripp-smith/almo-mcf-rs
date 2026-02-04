use crate::trees::{LowStretchTree, TreeBuildMode, TreeError, UnionFind};

#[derive(Debug, Clone, Copy)]
pub struct MwuConfig {
    pub eta: f64,
    pub iterations: usize,
    pub seed: u64,
}

impl Default for MwuConfig {
    fn default() -> Self {
        Self {
            eta: 0.5,
            iterations: 4,
            seed: 0,
        }
    }
}

pub fn sample_weighted_trees(
    node_count: usize,
    tails: &[u32],
    heads: &[u32],
    lengths: &[f64],
    circulation: &[f64],
    samples: usize,
    config: MwuConfig,
) -> Result<Vec<(LowStretchTree, f64)>, TreeError> {
    if node_count == 0 {
        return Err(TreeError::EmptyGraph);
    }
    if tails.len() != heads.len()
        || tails.len() != lengths.len()
        || lengths.len() != circulation.len()
    {
        return Err(TreeError::MissingEdgeLengths);
    }

    let edge_count = tails.len();
    let mut weights: Vec<f64> = vec![1.0; edge_count];
    let mut trees = Vec::with_capacity(samples);

    for sample_idx in 0..samples.max(1) {
        for _ in 0..config.iterations.max(1) {
            let mut edge_ids: Vec<usize> = (0..edge_count).collect();
            edge_ids.sort_by(|&a, &b| {
                let wa = weights[a].max(1e-12_f64);
                let wb = weights[b].max(1e-12_f64);
                let score_a = lengths[a].abs() / wa;
                let score_b = lengths[b].abs() / wb;
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut uf = UnionFind::new(node_count);
            let mut tree_edges = vec![false; edge_count];
            for &edge_id in &edge_ids {
                let u = tails[edge_id] as usize;
                let v = heads[edge_id] as usize;
                if uf.union(u, v) {
                    tree_edges[edge_id] = true;
                }
            }
            let tree = LowStretchTree::build_from_tree_edges(
                node_count, tails, heads, lengths, tree_edges,
            )?;

            for edge_id in 0..edge_count {
                let stretch = tree
                    .edge_stretch(edge_id, tails, heads, lengths)
                    .unwrap_or(0.0);
                let influence = circulation[edge_id].abs();
                if influence > 0.0 && stretch.is_finite() {
                    weights[edge_id] *= (config.eta * stretch).exp();
                }
            }

            if sample_idx + 1 == samples.max(1) {
                break;
            }
        }

        let tree = LowStretchTree::build_low_stretch_with_mode(
            node_count,
            tails,
            heads,
            lengths,
            TreeBuildMode::Randomized,
            config.seed ^ (sample_idx as u64),
        )?;
        let mut stretch_sum = 0.0;
        let mut stretch_count = 0.0;
        for (edge_id, circulation_value) in circulation.iter().enumerate().take(edge_count) {
            if circulation_value.abs() > 0.0 {
                if let Some(stretch) = tree.edge_stretch(edge_id, tails, heads, lengths) {
                    stretch_sum += stretch;
                    stretch_count += 1.0;
                }
            }
        }
        let avg_stretch = if stretch_count > 0.0 {
            stretch_sum / stretch_count
        } else {
            0.0
        };
        trees.push((tree, avg_stretch));
    }

    if trees.is_empty() {
        let tree = LowStretchTree::build_low_stretch_with_mode(
            node_count,
            tails,
            heads,
            lengths,
            TreeBuildMode::Deterministic,
            0,
        )?;
        trees.push((tree, 0.0));
    }
    Ok(trees)
}

pub fn deterministic_weighted_tree(
    node_count: usize,
    tails: &[u32],
    heads: &[u32],
    lengths: &[f64],
    weights: &[f64],
) -> Result<LowStretchTree, TreeError> {
    if node_count == 0 {
        return Err(TreeError::EmptyGraph);
    }
    if tails.len() != heads.len() || tails.len() != lengths.len() || lengths.len() != weights.len()
    {
        return Err(TreeError::MissingEdgeLengths);
    }
    let mut edge_ids: Vec<usize> = (0..tails.len()).collect();
    let mut scores = Vec::with_capacity(edge_ids.len());
    for edge_id in 0..edge_ids.len() {
        let w = weights[edge_id].max(1e-12_f64);
        scores.push(lengths[edge_id].abs() / w);
    }
    edge_ids.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let mut uf = UnionFind::new(node_count);
    let mut tree_edges = vec![false; tails.len()];
    for edge_id in edge_ids {
        let u = tails[edge_id] as usize;
        let v = heads[edge_id] as usize;
        if uf.union(u, v) {
            tree_edges[edge_id] = true;
        }
    }
    LowStretchTree::build_from_tree_edges(node_count, tails, heads, lengths, tree_edges)
}
