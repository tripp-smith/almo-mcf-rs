use std::collections::HashMap;

use crate::graph::Graph;
use crate::spanner::construction::Spanner;
use crate::trees::forest::DynamicForest;

pub type Forest = DynamicForest;

#[derive(Debug, Clone)]
pub struct CoreGraph {
    pub node_count: usize,
    pub tails: Vec<u32>,
    pub heads: Vec<u32>,
    pub lengths: Vec<f64>,
    pub supernodes: Vec<Vec<usize>>,
}

pub fn contract_with_spanner(graph: &Graph, forest: &Forest, spanner: &Spanner) -> CoreGraph {
    let node_count = graph.node_count();
    let mut component_map = vec![usize::MAX; node_count];
    let mut supernodes: Vec<Vec<usize>> = Vec::new();
    let mut next_id = 0;
    for node in 0..node_count {
        let root = forest.tree.root[node];
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

    let mut edge_map: HashMap<(usize, usize), f64> = HashMap::new();
    if let Some(level0) = spanner.levels.first() {
        for &edge_id in &level0.sparse_edges {
            if let Some((u, v)) = spanner.edge_info(edge_id) {
                let cu = component_map.get(u).copied().unwrap_or(u);
                let cv = component_map.get(v).copied().unwrap_or(v);
                if cu == cv {
                    continue;
                }
                let key = ordered_pair(cu, cv);
                edge_map.entry(key).or_insert(1.0);
            }
        }
    }

    let mut tails = Vec::new();
    let mut heads = Vec::new();
    let mut lengths = Vec::new();
    for ((u, v), length) in edge_map {
        tails.push(u as u32);
        heads.push(v as u32);
        lengths.push(length);
    }

    let min_size = ((node_count as f64).ln().max(1.0)) as usize;
    let limit = (node_count / min_size.max(1)).max(1);
    if tails.len() > limit {
        tails.truncate(limit);
        heads.truncate(limit);
        lengths.truncate(limit);
    }

    CoreGraph {
        node_count: next_id.max(1),
        tails,
        heads,
        lengths,
        supernodes,
    }
}

fn ordered_pair(u: usize, v: usize) -> (usize, usize) {
    if u <= v {
        (u, v)
    } else {
        (v, u)
    }
}
