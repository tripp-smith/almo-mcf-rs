use rand::prelude::SliceRandom;
use rand::{Rng, SeedableRng};

use crate::graph::{EdgeId, Graph, NodeId};
use crate::trees::{TreeError, UnionFind};

#[derive(Debug, Clone)]
pub struct Tree {
    pub node_count: usize,
    pub tails: Vec<u32>,
    pub heads: Vec<u32>,
    pub lengths: Vec<f64>,
    pub tree_edges: Vec<bool>,
    pub parent: Vec<usize>,
    pub parent_edge: Vec<usize>,
    pub depth: Vec<usize>,
    pub prefix_length: Vec<f64>,
    pub root: Vec<usize>,
    up: Vec<Vec<usize>>,
}

impl Tree {
    pub fn build_from_tree_edges(
        node_count: usize,
        tails: Vec<u32>,
        heads: Vec<u32>,
        lengths: Vec<f64>,
        tree_edges: Vec<bool>,
    ) -> Result<Self, TreeError> {
        if node_count == 0 {
            return Err(TreeError::EmptyGraph);
        }
        if tails.len() != heads.len()
            || tails.len() != lengths.len()
            || tails.len() != tree_edges.len()
        {
            return Err(TreeError::MissingEdgeLengths);
        }
        let mut tree = Self {
            node_count,
            tails,
            heads,
            lengths,
            tree_edges,
            parent: vec![usize::MAX; node_count],
            parent_edge: vec![usize::MAX; node_count],
            depth: vec![0; node_count],
            prefix_length: vec![0.0; node_count],
            root: vec![usize::MAX; node_count],
            up: Vec::new(),
        };
        tree.rebuild_from_tree_edges();
        Ok(tree)
    }

    pub fn promote_root(&mut self, new_root: NodeId) -> bool {
        if new_root.0 >= self.node_count {
            return false;
        }
        if self.node_count == 0 {
            return false;
        }
        self.rebuild_with_order(new_root.0);
        true
    }

    pub fn delete_edge(&mut self, u: NodeId, v: NodeId) -> bool {
        let edge_id = self.find_edge_id(u, v);
        let Some(edge_id) = edge_id else {
            return false;
        };
        if !self.tree_edges.get(edge_id).copied().unwrap_or(false) {
            return false;
        }
        self.tree_edges[edge_id] = false;
        let mut component = vec![usize::MAX; self.node_count];
        let mut stack = vec![u.0];
        component[u.0] = 0;
        while let Some(node) = stack.pop() {
            for (next, _) in self.tree_neighbors(node) {
                if component[next] != usize::MAX {
                    continue;
                }
                component[next] = 0;
                stack.push(next);
            }
        }
        if component[v.0] == 0 {
            self.rebuild_from_tree_edges();
            return true;
        }

        let mut best_edge = None;
        for edge_idx in 0..self.tails.len() {
            if self.tree_edges[edge_idx] {
                continue;
            }
            let a = self.tails[edge_idx] as usize;
            let b = self.heads[edge_idx] as usize;
            let ca = component[a];
            let cb = component[b];
            if ca == usize::MAX || cb == usize::MAX || ca == cb {
                continue;
            }
            let length = self.lengths[edge_idx];
            if best_edge.is_none_or(|(_, best_len)| length < best_len) {
                best_edge = Some((edge_idx, length));
            }
        }

        if let Some((edge_idx, _)) = best_edge {
            self.tree_edges[edge_idx] = true;
        }
        self.rebuild_from_tree_edges();
        true
    }

    pub fn compute_stretch_overestimate(&self, path: &[NodeId]) -> f64 {
        if path.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0;
        for window in path.windows(2) {
            let u = window[0].0;
            let v = window[1].0;
            if let Some(distance) = self.path_length(u, v) {
                total += distance;
            } else {
                return f64::INFINITY;
            }
        }
        total
    }

    pub fn path_length(&self, u: usize, v: usize) -> Option<f64> {
        let lca = self.lca(u, v)?;
        let dist = self.prefix_length[u] + self.prefix_length[v] - 2.0 * self.prefix_length[lca];
        Some(dist)
    }

    pub fn lca(&self, mut u: usize, mut v: usize) -> Option<usize> {
        if self.root.get(u)? != self.root.get(v)? {
            return None;
        }
        if self.depth[u] < self.depth[v] {
            std::mem::swap(&mut u, &mut v);
        }
        let diff = self.depth[u] - self.depth[v];
        for k in 0..self.up.len() {
            if diff & (1 << k) != 0 {
                u = self.up[k][u];
            }
        }
        if u == v {
            return Some(u);
        }
        for k in (0..self.up.len()).rev() {
            if self.up[k][u] != self.up[k][v] {
                u = self.up[k][u];
                v = self.up[k][v];
            }
        }
        Some(self.parent[u])
    }

    fn rebuild_with_order(&mut self, root_start: usize) {
        let mut order: Vec<usize> = (0..self.node_count).collect();
        if let Some(pos) = order.iter().position(|&node| node == root_start) {
            order.swap(0, pos);
        }
        self.rebuild_from_tree_edges_with_order(&order);
    }

    fn rebuild_from_tree_edges(&mut self) {
        let order: Vec<usize> = (0..self.node_count).collect();
        self.rebuild_from_tree_edges_with_order(&order);
    }

    fn rebuild_from_tree_edges_with_order(&mut self, order: &[usize]) {
        let mut adjacency: Vec<Vec<(usize, usize)>> = vec![Vec::new(); self.node_count];
        for (edge_id, (&tail, &head)) in self.tails.iter().zip(self.heads.iter()).enumerate() {
            if !self.tree_edges[edge_id] {
                continue;
            }
            let u = tail as usize;
            let v = head as usize;
            adjacency[u].push((v, edge_id));
            adjacency[v].push((u, edge_id));
        }

        self.parent.fill(usize::MAX);
        self.parent_edge.fill(usize::MAX);
        self.depth.fill(0);
        self.prefix_length.fill(0.0);
        self.root.fill(usize::MAX);

        for &start in order {
            if start >= self.node_count {
                continue;
            }
            if self.parent[start] != usize::MAX {
                continue;
            }
            self.parent[start] = start;
            self.parent_edge[start] = usize::MAX;
            self.root[start] = start;
            self.depth[start] = 0;
            self.prefix_length[start] = 0.0;
            let mut stack = vec![start];
            while let Some(node) = stack.pop() {
                for &(neighbor, edge_id) in &adjacency[node] {
                    if self.parent[neighbor] != usize::MAX {
                        continue;
                    }
                    self.parent[neighbor] = node;
                    self.parent_edge[neighbor] = edge_id;
                    self.root[neighbor] = start;
                    self.depth[neighbor] = self.depth[node] + 1;
                    self.prefix_length[neighbor] = self.prefix_length[node] + self.lengths[edge_id];
                    stack.push(neighbor);
                }
            }
        }

        let mut max_pow = 1;
        while (1usize << max_pow) <= self.node_count.max(1) {
            max_pow += 1;
        }
        self.up = vec![vec![0; self.node_count]; max_pow];
        if self.node_count > 0 {
            self.up[0][..self.node_count].copy_from_slice(&self.parent[..self.node_count]);
        }
        for k in 1..max_pow {
            for node in 0..self.node_count {
                let mid = self.up[k - 1][node];
                self.up[k][node] = self.up[k - 1][mid];
            }
        }
    }

    fn tree_neighbors(&self, node: usize) -> Vec<(usize, usize)> {
        let mut neighbors = Vec::new();
        for (edge_id, (&tail, &head)) in self.tails.iter().zip(self.heads.iter()).enumerate() {
            if !self.tree_edges[edge_id] {
                continue;
            }
            let u = tail as usize;
            let v = head as usize;
            if u == node {
                neighbors.push((v, edge_id));
            } else if v == node {
                neighbors.push((u, edge_id));
            }
        }
        neighbors
    }

    fn find_edge_id(&self, u: NodeId, v: NodeId) -> Option<usize> {
        for (edge_id, (&tail, &head)) in self.tails.iter().zip(self.heads.iter()).enumerate() {
            if (tail as usize == u.0 && head as usize == v.0)
                || (tail as usize == v.0 && head as usize == u.0)
            {
                return Some(edge_id);
            }
        }
        None
    }
}

pub fn build_lsst(graph: &Graph, gamma: f64) -> Result<Vec<Tree>, TreeError> {
    let node_count = graph.node_count();
    if node_count == 0 {
        return Err(TreeError::EmptyGraph);
    }

    let (tails, heads, lengths) = symmetrized_edges(graph);
    let edge_count = tails.len();
    if edge_count == 0 {
        let tree = Tree::build_from_tree_edges(
            node_count,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )?;
        return Ok(vec![tree]);
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(0x5eed_cafe_u64);
    let rounds = ((node_count as f64).ln().ceil() as usize).max(1);
    let mut weights = vec![1.0_f64; edge_count];
    let mut trees = Vec::with_capacity(rounds);

    for _ in 0..rounds {
        let tree_edges = randomized_mst(node_count, &tails, &heads, &lengths, &weights, &mut rng);
        let tree = Tree::build_from_tree_edges(
            node_count,
            tails.clone(),
            heads.clone(),
            lengths.clone(),
            tree_edges.clone(),
        )?;
        update_weights(
            node_count,
            &tails,
            &heads,
            &tree_edges,
            gamma.max(0.0),
            &mut weights,
        );
        trees.push(tree);
    }

    Ok(trees)
}

pub fn estimate_average_stretch(
    graph: &Graph,
    forest: &[Tree],
    samples: usize,
) -> Result<f64, TreeError> {
    let node_count = graph.node_count();
    if node_count == 0 {
        return Err(TreeError::EmptyGraph);
    }
    if forest.is_empty() {
        return Ok(f64::INFINITY);
    }
    let adjacency = build_weighted_adjacency(graph);
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x51ab_1e_u64);
    let mut stretch_sum = 0.0;
    let mut stretch_count = 0.0;

    for _ in 0..samples.max(1) {
        let u = rng.gen_range(0..node_count);
        let v = rng.gen_range(0..node_count);
        if u == v {
            continue;
        }
        let graph_dist = dijkstra_distance(&adjacency, u, v)?;
        if !graph_dist.is_finite() || graph_dist <= 0.0 {
            continue;
        }
        let mut tree_dist = 0.0;
        for tree in forest {
            if let Some(dist) = tree.path_length(u, v) {
                tree_dist += dist;
            } else {
                tree_dist += graph_dist;
            }
        }
        let avg_tree_dist = tree_dist / forest.len() as f64;
        stretch_sum += avg_tree_dist / graph_dist;
        stretch_count += 1.0;
    }

    if stretch_count == 0.0 {
        Ok(f64::INFINITY)
    } else {
        Ok(stretch_sum / stretch_count)
    }
}

fn randomized_mst(
    node_count: usize,
    tails: &[u32],
    heads: &[u32],
    lengths: &[f64],
    weights: &[f64],
    rng: &mut rand::rngs::StdRng,
) -> Vec<bool> {
    let edge_count = tails.len();
    let mut edge_ids: Vec<usize> = (0..edge_count).collect();
    edge_ids.shuffle(rng);
    let mut scores = Vec::with_capacity(edge_count);
    for edge_id in 0..edge_count {
        let w = weights[edge_id].max(1e-12_f64);
        let jitter = 1.0 + rng.gen::<f64>() * 1e-3;
        scores.push(lengths[edge_id].abs() / w * jitter);
    }
    edge_ids.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));

    let mut uf = UnionFind::new(node_count);
    let mut tree_edges = vec![false; edge_count];
    for edge_id in edge_ids {
        let u = tails[edge_id] as usize;
        let v = heads[edge_id] as usize;
        if uf.union(u, v) {
            tree_edges[edge_id] = true;
        }
    }
    tree_edges
}

fn update_weights(
    node_count: usize,
    tails: &[u32],
    heads: &[u32],
    tree_edges: &[bool],
    gamma: f64,
    weights: &mut [f64],
) {
    let mut degree = vec![0usize; node_count];
    for (edge_id, &is_tree) in tree_edges.iter().enumerate() {
        if !is_tree {
            continue;
        }
        let u = tails[edge_id] as usize;
        let v = heads[edge_id] as usize;
        degree[u] += 1;
        degree[v] += 1;
    }
    for (edge_id, &is_tree) in tree_edges.iter().enumerate() {
        if !is_tree {
            continue;
        }
        let u = tails[edge_id] as usize;
        let v = heads[edge_id] as usize;
        if degree[u] > 0 {
            weights[edge_id] += gamma / degree[u] as f64;
        }
        if degree[v] > 0 {
            weights[edge_id] += gamma / degree[v] as f64;
        }
    }
}

fn symmetrized_edges(graph: &Graph) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let mut tails = Vec::new();
    let mut heads = Vec::new();
    let mut lengths = Vec::new();
    for (_, edge) in graph.edges() {
        let length = edge.cost.abs().max(1.0);
        tails.push(edge.tail.0 as u32);
        heads.push(edge.head.0 as u32);
        lengths.push(length);
        tails.push(edge.head.0 as u32);
        heads.push(edge.tail.0 as u32);
        lengths.push(length);
    }
    (tails, heads, lengths)
}

fn build_weighted_adjacency(graph: &Graph) -> Vec<Vec<(usize, f64)>> {
    let mut adjacency = vec![Vec::new(); graph.node_count()];
    for (_, edge) in graph.edges() {
        let u = edge.tail.0;
        let v = edge.head.0;
        let length = edge.cost.abs().max(1.0);
        adjacency[u].push((v, length));
        adjacency[v].push((u, length));
    }
    adjacency
}

fn dijkstra_distance(
    adjacency: &[Vec<(usize, f64)>],
    start: usize,
    target: usize,
) -> Result<f64, TreeError> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(Clone, Copy)]
    struct State {
        cost: f64,
        node: usize,
    }

    impl PartialEq for State {
        fn eq(&self, other: &Self) -> bool {
            self.cost == other.cost
        }
    }

    impl Eq for State {}

    impl PartialOrd for State {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            other.cost.partial_cmp(&self.cost)
        }
    }

    impl Ord for State {
        fn cmp(&self, other: &Self) -> Ordering {
            self.partial_cmp(other).unwrap_or(Ordering::Equal)
        }
    }

    let mut dist = vec![f64::INFINITY; adjacency.len()];
    let mut heap = BinaryHeap::new();
    dist[start] = 0.0;
    heap.push(State {
        cost: 0.0,
        node: start,
    });
    while let Some(State { cost, node }) = heap.pop() {
        if node == target {
            return Ok(cost);
        }
        if cost > dist[node] {
            continue;
        }
        for &(next, weight) in &adjacency[node] {
            let next_cost = cost + weight;
            if next_cost < dist[next] {
                dist[next] = next_cost;
                heap.push(State {
                    cost: next_cost,
                    node: next,
                });
            }
        }
    }
    Ok(dist[target])
}

pub fn edge_endpoints(graph: &Graph, edge: EdgeId) -> Option<(NodeId, NodeId)> {
    graph.edge(edge).map(|edge| (edge.tail, edge.head))
}
