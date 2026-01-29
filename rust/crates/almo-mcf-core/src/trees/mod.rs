#[derive(Debug, Clone)]
pub struct LowStretchTree {
    pub parent: Vec<usize>,
    pub parent_edge: Vec<usize>,
    pub depth: Vec<usize>,
    pub prefix_length: Vec<f64>,
    pub root: Vec<usize>,
    pub tree_edges: Vec<bool>,
    up: Vec<Vec<usize>>,
}

#[derive(Debug)]
pub enum TreeError {
    EmptyGraph,
    MissingEdgeLengths,
}

#[derive(Debug, Clone)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        let seed = if seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            seed
        };
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for idx in (1..values.len()).rev() {
            let j = (self.next_u64() as usize) % (idx + 1);
            values.swap(idx, j);
        }
    }
}

impl LowStretchTree {
    pub fn build(
        node_count: usize,
        tails: &[u32],
        heads: &[u32],
        lengths: &[f64],
        seed: u64,
    ) -> Result<Self, TreeError> {
        if node_count == 0 {
            return Err(TreeError::EmptyGraph);
        }
        if tails.len() != heads.len() || tails.len() != lengths.len() {
            return Err(TreeError::MissingEdgeLengths);
        }

        let mut adjacency: Vec<Vec<(usize, usize)>> = vec![Vec::new(); node_count];
        for (edge_id, (&tail, &head)) in tails.iter().zip(heads.iter()).enumerate() {
            let u = tail as usize;
            let v = head as usize;
            adjacency[u].push((v, edge_id));
            adjacency[v].push((u, edge_id));
        }

        let mut rng = XorShift64::new(seed);
        let mut order: Vec<usize> = (0..node_count).collect();
        rng.shuffle(&mut order);
        for neighbors in adjacency.iter_mut() {
            rng.shuffle(neighbors);
        }

        let mut parent = vec![usize::MAX; node_count];
        let mut parent_edge = vec![usize::MAX; node_count];
        let mut depth = vec![0; node_count];
        let mut prefix_length = vec![0.0; node_count];
        let mut root = vec![usize::MAX; node_count];
        let mut tree_edges = vec![false; tails.len()];

        for &start in &order {
            if parent[start] != usize::MAX {
                continue;
            }
            parent[start] = start;
            parent_edge[start] = usize::MAX;
            root[start] = start;
            depth[start] = 0;
            prefix_length[start] = 0.0;
            let mut stack = vec![start];
            while let Some(node) = stack.pop() {
                for &(neighbor, edge_id) in &adjacency[node] {
                    if parent[neighbor] != usize::MAX {
                        continue;
                    }
                    parent[neighbor] = node;
                    parent_edge[neighbor] = edge_id;
                    root[neighbor] = start;
                    depth[neighbor] = depth[node] + 1;
                    prefix_length[neighbor] = prefix_length[node] + lengths[edge_id];
                    tree_edges[edge_id] = true;
                    stack.push(neighbor);
                }
            }
        }

        let mut max_pow = 1;
        while (1usize << max_pow) <= node_count {
            max_pow += 1;
        }
        let mut up = vec![vec![0; node_count]; max_pow];
        up[0][..node_count].copy_from_slice(&parent[..node_count]);
        for k in 1..max_pow {
            for node in 0..node_count {
                let mid = up[k - 1][node];
                up[k][node] = up[k - 1][mid];
            }
        }

        Ok(Self {
            parent,
            parent_edge,
            depth,
            prefix_length,
            root,
            tree_edges,
            up,
        })
    }

    pub fn lca(&self, mut u: usize, mut v: usize) -> Option<usize> {
        if self.root[u] != self.root[v] {
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

    pub fn path_length(&self, u: usize, v: usize) -> Option<f64> {
        let lca = self.lca(u, v)?;
        let dist = self.prefix_length[u] + self.prefix_length[v] - 2.0 * self.prefix_length[lca];
        Some(dist)
    }

    pub fn path_edges(
        &self,
        u: usize,
        v: usize,
        tails: &[u32],
        heads: &[u32],
    ) -> Option<Vec<(usize, i8)>> {
        let lca = self.lca(u, v)?;
        let mut edges = Vec::new();
        let mut curr = u;
        while curr != lca {
            let edge_id = self.parent_edge[curr];
            let parent = self.parent[curr];
            let dir = edge_direction(curr, parent, edge_id, tails, heads);
            edges.push((edge_id, dir));
            curr = parent;
        }

        let mut tail_edges = Vec::new();
        curr = v;
        while curr != lca {
            let edge_id = self.parent_edge[curr];
            let parent = self.parent[curr];
            let dir = edge_direction(curr, parent, edge_id, tails, heads);
            tail_edges.push((edge_id, dir));
            curr = parent;
        }
        tail_edges.reverse();
        for (edge_id, dir) in tail_edges {
            edges.push((edge_id, -dir));
        }
        Some(edges)
    }
}

fn edge_direction(from: usize, to: usize, edge_id: usize, tails: &[u32], heads: &[u32]) -> i8 {
    let tail = tails[edge_id] as usize;
    let head = heads[edge_id] as usize;
    if tail == from && head == to {
        1
    } else {
        -1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_covers_all_nodes() {
        let tails = vec![0, 1, 2, 3];
        let heads = vec![1, 2, 3, 0];
        let lengths = vec![1.0, 1.0, 1.0, 1.0];
        let tree = LowStretchTree::build(4, &tails, &heads, &lengths, 42).unwrap();
        for node in 0..4 {
            assert_ne!(tree.parent[node], usize::MAX);
            assert_ne!(tree.root[node], usize::MAX);
        }
    }

    #[test]
    fn path_length_is_deterministic() {
        let tails = vec![0, 0, 1, 2];
        let heads = vec![1, 2, 3, 3];
        let lengths = vec![1.0, 2.0, 1.0, 2.0];
        let tree_a = LowStretchTree::build(4, &tails, &heads, &lengths, 7).unwrap();
        let tree_b = LowStretchTree::build(4, &tails, &heads, &lengths, 7).unwrap();
        assert_eq!(tree_a.parent, tree_b.parent);
        let dist = tree_a.path_length(0, 3).unwrap();
        assert!(dist > 0.0);
    }

    #[test]
    fn lca_queries_work() {
        let tails = vec![0, 1, 1, 3];
        let heads = vec![1, 2, 3, 4];
        let lengths = vec![1.0, 1.0, 1.0, 1.0];
        let tree = LowStretchTree::build(5, &tails, &heads, &lengths, 1).unwrap();
        let lca = tree.lca(2, 4).unwrap();
        assert!(lca < 5);
    }
}
