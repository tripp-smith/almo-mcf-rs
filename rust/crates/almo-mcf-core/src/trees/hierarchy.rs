use crate::trees::forest::DynamicForest;
use crate::trees::{LowStretchTree, TreeError};

#[derive(Debug, Clone)]
pub struct HierarchyLevel {
    pub node_count: usize,
    pub tails: Vec<u32>,
    pub heads: Vec<u32>,
    pub lengths: Vec<f64>,
    pub forest: DynamicForest,
}

#[derive(Debug, Clone)]
pub struct HierarchicalTreeChain {
    pub levels: Vec<HierarchyLevel>,
}

impl HierarchicalTreeChain {
    pub fn build(
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        lengths: &[f64],
        levels: usize,
    ) -> Result<Self, TreeError> {
        if node_count == 0 {
            return Err(TreeError::EmptyGraph);
        }
        if tails.len() != heads.len() || tails.len() != lengths.len() {
            return Err(TreeError::MissingEdgeLengths);
        }
        let mut current_node_count = node_count;
        let mut current_tails = tails.to_vec();
        let mut current_heads = heads.to_vec();
        let mut current_lengths = lengths.to_vec();
        let mut chain = Vec::new();

        for _ in 0..levels.max(1) {
            let tree = LowStretchTree::build_low_stretch(
                current_node_count,
                &current_tails,
                &current_heads,
                &current_lengths,
                0,
            )?;
            let forest = DynamicForest::new_from_tree(
                current_node_count,
                current_tails.clone(),
                current_heads.clone(),
                current_lengths.clone(),
                tree.tree_edges.clone(),
            )?;

            let mut component_map = vec![usize::MAX; current_node_count];
            let mut next_id = 0;
            for node in 0..current_node_count {
                let root = forest.tree.root[node];
                if component_map[root] == usize::MAX {
                    component_map[root] = next_id;
                    next_id += 1;
                }
                component_map[node] = component_map[root];
            }

            let mut next_tails = Vec::new();
            let mut next_heads = Vec::new();
            let mut next_lengths = Vec::new();
            for edge_id in 0..current_tails.len() {
                let u = current_tails[edge_id] as usize;
                let v = current_heads[edge_id] as usize;
                let cu = component_map[u];
                let cv = component_map[v];
                if cu == cv {
                    continue;
                }
                next_tails.push(cu as u32);
                next_heads.push(cv as u32);
                next_lengths.push(current_lengths[edge_id]);
            }

            chain.push(HierarchyLevel {
                node_count: current_node_count,
                tails: current_tails,
                heads: current_heads,
                lengths: current_lengths,
                forest,
            });

            if next_id <= 1 || next_tails.is_empty() {
                break;
            }
            current_node_count = next_id;
            current_tails = next_tails;
            current_heads = next_heads;
            current_lengths = next_lengths;
        }

        Ok(Self { levels: chain })
    }

    pub fn shift_level(&mut self, level: usize) -> Option<()> {
        let level = self.levels.get_mut(level)?;
        level.forest.rebuild_all();
        Some(())
    }
}
