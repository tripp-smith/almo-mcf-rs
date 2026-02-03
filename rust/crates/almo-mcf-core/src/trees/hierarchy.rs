use crate::graph::{Graph, NodeId};
use crate::spanner::build_spanner_on_core;
use crate::spanner::decremental::{
    ContractedGraph, DecrementalSpanner, DecrementalSpannerParams, LSForest,
};
use crate::trees::forest::DynamicForest;
use crate::trees::lsst::{build_lsst, Tree};
use crate::trees::{LowStretchTree, TreeBuildMode, TreeError};

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
    pub spanner: Option<DecrementalSpanner>,
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
                spanner: None,
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

#[derive(Debug, Clone)]
pub struct DecompConfig {
    pub reduction_factor: f64,
    pub deterministic: bool,
}

impl Default for DecompConfig {
    fn default() -> Self {
        Self {
            reduction_factor: 0.125,
            deterministic: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BranchingTreeChain {
    pub levels: Vec<Level>,
    pub dirty_levels: Vec<bool>,
}

impl BranchingTreeChain {
    pub fn build_chain(residual_graph: &Graph, config: &DecompConfig) -> Result<Self, TreeError> {
        let node_count = residual_graph.node_count();
        if node_count == 0 {
            return Err(TreeError::EmptyGraph);
        }
        let d = ((node_count as f64).log(8.0).floor() as usize) + 1;
        let (mut current_tails, mut current_heads, mut current_lengths) =
            graph_edges_to_arrays(residual_graph);
        let mut current_nodes = node_count;
        let mut levels = Vec::new();
        let mut dirty_levels = Vec::new();
        let build_mode = if config.deterministic {
            TreeBuildMode::Deterministic
        } else {
            TreeBuildMode::Randomized
        };

        for _ in 0..d.max(1) {
            let tree = LowStretchTree::build_low_stretch_with_mode(
                current_nodes,
                &current_tails,
                &current_heads,
                &current_lengths,
                build_mode,
                0,
            )?;
            let forest = build_lsst(
                &build_graph_from_edges(
                    current_nodes,
                    &current_tails,
                    &current_heads,
                    &current_lengths,
                ),
                1.1,
            )?;
            let (core_graph, mapping, contracted) = contract_from_tree(
                current_nodes,
                &current_tails,
                &current_heads,
                &current_lengths,
                &tree,
            );
            let params =
                DecrementalSpannerParams::from_graph(contracted.node_count, contracted.tails.len());
            let ls_forest =
                LSForest::from_tree(&tree, &current_tails, &current_heads, &current_lengths);
            let spanner = Some(build_spanner_on_core(&contracted, &ls_forest, params));

            levels.push(Level {
                forest,
                core: core_graph.clone(),
                mapping,
                spanner,
            });
            dirty_levels.push(false);

            let next_nodes = core_graph.node_count();
            if next_nodes <= 1 {
                break;
            }
            if next_nodes > (current_nodes as f64 * config.reduction_factor).ceil() as usize
                && next_nodes > 1
            {
                break;
            }
            current_nodes = next_nodes;
            current_tails = contracted.tails.clone();
            current_heads = contracted.heads.clone();
            current_lengths = contracted.lengths.clone();
        }

        Ok(Self {
            levels,
            dirty_levels,
        })
    }

    pub fn mark_dirty(&mut self, level: usize) {
        if let Some(flag) = self.dirty_levels.get_mut(level) {
            *flag = true;
        }
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

fn graph_edges_to_arrays(graph: &Graph) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let mut tails = Vec::new();
    let mut heads = Vec::new();
    let mut lengths = Vec::new();
    for (_, edge) in graph.edges() {
        tails.push(edge.tail.0 as u32);
        heads.push(edge.head.0 as u32);
        lengths.push(edge.cost.abs().max(1.0));
    }
    (tails, heads, lengths)
}

fn build_graph_from_edges(
    node_count: usize,
    tails: &[u32],
    heads: &[u32],
    lengths: &[f64],
) -> Graph {
    let mut graph = Graph::new(node_count);
    for edge_id in 0..tails.len() {
        let tail = NodeId(tails[edge_id] as usize);
        let head = NodeId(heads[edge_id] as usize);
        let cost = lengths.get(edge_id).copied().unwrap_or(1.0);
        let _ = graph.add_edge(tail, head, 0.0, f64::INFINITY, cost);
    }
    graph
}

fn contract_from_tree(
    node_count: usize,
    tails: &[u32],
    heads: &[u32],
    lengths: &[f64],
    tree: &LowStretchTree,
) -> (Graph, Vec<usize>, ContractedGraph) {
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

    let mut core = Graph::new(next_id.max(1));
    let mut new_tails = Vec::new();
    let mut new_heads = Vec::new();
    let mut new_lengths = Vec::new();
    for edge_id in 0..tails.len() {
        let u = component_map[tails[edge_id] as usize];
        let v = component_map[heads[edge_id] as usize];
        if u == v {
            continue;
        }
        let _ = core.add_edge(NodeId(u), NodeId(v), 0.0, f64::INFINITY, lengths[edge_id]);
        new_tails.push(u as u32);
        new_heads.push(v as u32);
        new_lengths.push(lengths[edge_id]);
    }
    (
        core,
        component_map,
        ContractedGraph {
            node_count: next_id.max(1),
            tails: new_tails,
            heads: new_heads,
            lengths: new_lengths,
            supernodes,
        },
    )
}
