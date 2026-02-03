use crate::graph::{Graph, NodeId};
use crate::trees::forest::DynamicForest;
use crate::trees::lsst::{build_lsst, Tree};
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

#[derive(Debug, Clone)]
pub struct Level {
    pub forest: Vec<Tree>,
    pub core: Graph,
    pub mapping: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct Decomposition {
    pub levels: Vec<Level>,
}

impl Decomposition {
    pub fn decompose(graph: &Graph, depth: usize) -> Result<Self, TreeError> {
        let mut levels = Vec::new();
        let mut current = graph.clone();
        let mut gamma = 1.1;
        for _ in 0..depth.max(1) {
            let forest = build_lsst(&current, gamma)?;
            let (core, mapping) = contract_low_degree(&current, gamma);
            levels.push(Level {
                forest,
                core: core.clone(),
                mapping,
            });
            if core.node_count() <= (current.node_count() / 2).max(1) {
                current = core;
            } else {
                break;
            }
            gamma = (gamma * 1.05).min(2.0);
        }
        Ok(Self { levels })
    }

    pub fn unwind_path(&self, path: &[NodeId]) -> Vec<NodeId> {
        let mut current = path.to_vec();
        for level in self.levels.iter().rev() {
            let mut expanded = Vec::new();
            for node in &current {
                let idx = node.0;
                let original = level.mapping.get(idx).copied().unwrap_or(idx);
                expanded.push(NodeId(original));
            }
            current = expanded;
        }
        current
    }
}

fn contract_low_degree(graph: &Graph, gamma: f64) -> (Graph, Vec<usize>) {
    let n = graph.node_count();
    let threshold = ((n as f64).ln().max(1.0) * gamma).ceil() as usize;
    let mut degrees = vec![0usize; n];
    for (_, edge) in graph.edges() {
        degrees[edge.tail.0] += 1;
        degrees[edge.head.0] += 1;
    }

    let mut mapping = vec![usize::MAX; n];
    let mut next_id = 0;
    for (node, value) in mapping.iter_mut().enumerate().take(n) {
        if *value != usize::MAX {
            continue;
        }
        if degrees[node] < threshold {
            *value = next_id;
            next_id += 1;
        }
    }
    for value in mapping.iter_mut().take(n) {
        if *value == usize::MAX {
            *value = next_id;
            next_id += 1;
        }
    }

    if next_id > (n / 2).max(1) {
        for (node, value) in mapping.iter_mut().enumerate().take(n) {
            *value = node / 2;
        }
        next_id = (n / 2).max(1);
    }

    let mut core = Graph::new(next_id);
    for (_, edge) in graph.edges() {
        let u = mapping[edge.tail.0];
        let v = mapping[edge.head.0];
        if u == v {
            continue;
        }
        let _ = core.add_edge(NodeId(u), NodeId(v), edge.lower, edge.upper, edge.cost);
    }
    (core, mapping)
}
