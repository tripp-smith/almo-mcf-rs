pub mod dynamic;

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

    fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) * (1.0 / ((1u64 << 53) as f64))
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

    pub fn build_low_stretch(
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
        let mut rng = XorShift64::new(seed);
        let mut edge_ids: Vec<usize> = (0..tails.len()).collect();
        edge_ids.sort_by(|&a, &b| {
            let jitter_a = lengths[a] * (1.0 + 1e-6 * rng.next_f64());
            let jitter_b = lengths[b] * (1.0 + 1e-6 * rng.next_f64());
            jitter_a
                .partial_cmp(&jitter_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut uf = UnionFind::new(node_count);
        let mut tree_edges = vec![false; tails.len()];
        for edge_id in edge_ids {
            let u = tails[edge_id] as usize;
            let v = heads[edge_id] as usize;
            if uf.union(u, v) {
                tree_edges[edge_id] = true;
            }
        }

        let mut adjacency: Vec<Vec<(usize, usize)>> = vec![Vec::new(); node_count];
        for (edge_id, (&tail, &head)) in tails.iter().zip(heads.iter()).enumerate() {
            if !tree_edges[edge_id] {
                continue;
            }
            let u = tail as usize;
            let v = head as usize;
            adjacency[u].push((v, edge_id));
            adjacency[v].push((u, edge_id));
        }

        let mut parent = vec![usize::MAX; node_count];
        let mut parent_edge = vec![usize::MAX; node_count];
        let mut depth = vec![0; node_count];
        let mut prefix_length = vec![0.0; node_count];
        let mut root = vec![usize::MAX; node_count];

        for start in 0..node_count {
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
        let path_len = self.depth[u]
            .saturating_add(self.depth[v])
            .saturating_sub(2 * self.depth[lca]);
        let mut edges = Vec::with_capacity(path_len);
        let mut curr = u;
        while curr != lca {
            let edge_id = self.parent_edge[curr];
            let parent = self.parent[curr];
            let dir = edge_direction(curr, parent, edge_id, tails, heads);
            edges.push((edge_id, dir));
            curr = parent;
        }

        let mut tail_edges = Vec::with_capacity(self.depth[v].saturating_sub(self.depth[lca]));
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

    pub fn fundamental_cycle(
        &self,
        edge_id: usize,
        tails: &[u32],
        heads: &[u32],
    ) -> Option<Vec<(usize, i8)>> {
        if edge_id >= tails.len() || edge_id >= heads.len() {
            return None;
        }
        if self.tree_edges.get(edge_id).copied().unwrap_or(false) {
            return None;
        }
        let tail = tails[edge_id] as usize;
        let head = heads[edge_id] as usize;
        let mut cycle = self.path_edges(head, tail, tails, heads)?;
        cycle.push((edge_id, 1));
        Some(cycle)
    }

    pub fn edge_stretch(
        &self,
        edge_id: usize,
        tails: &[u32],
        heads: &[u32],
        lengths: &[f64],
    ) -> Option<f64> {
        if edge_id >= tails.len() || edge_id >= heads.len() || edge_id >= lengths.len() {
            return None;
        }
        let length = lengths[edge_id];
        if length <= 0.0 {
            return None;
        }
        let tail = tails[edge_id] as usize;
        let head = heads[edge_id] as usize;
        let tree_distance = self.path_length(tail, head)?;
        Some(tree_distance / length)
    }

    pub fn average_stretch(&self, tails: &[u32], heads: &[u32], lengths: &[f64]) -> Option<f64> {
        if tails.is_empty() {
            return None;
        }
        let mut total = 0.0;
        let mut count = 0.0;
        for edge_id in 0..tails.len() {
            if let Some(stretch) = self.edge_stretch(edge_id, tails, heads, lengths) {
                total += stretch;
                count += 1.0;
            }
        }
        if count == 0.0 {
            None
        } else {
            Some(total / count)
        }
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

#[derive(Debug, Clone)]
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let root = self.find(self.parent[x]);
            self.parent[x] = root;
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) -> bool {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return false;
        }
        if self.rank[ra] < self.rank[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        if self.rank[ra] == self.rank[rb] {
            self.rank[ra] += 1;
        }
        true
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

    #[test]
    fn path_length_matches_path_edges() {
        let tails = vec![0, 1, 1, 3, 2];
        let heads = vec![1, 2, 3, 4, 4];
        let lengths = vec![1.0, 2.0, 1.5, 1.0, 3.0];
        let tree = LowStretchTree::build(5, &tails, &heads, &lengths, 9).unwrap();
        let path_edges = tree.path_edges(2, 4, &tails, &heads).unwrap();
        let mut length_sum = 0.0;
        for (edge_id, _) in path_edges {
            length_sum += lengths[edge_id];
        }
        let dist = tree.path_length(2, 4).unwrap();
        assert!((dist - length_sum).abs() < 1e-9);
    }

    #[test]
    fn fundamental_cycle_includes_off_tree_edge() {
        let tails = vec![0, 1, 2, 0];
        let heads = vec![1, 2, 0, 2];
        let lengths = vec![1.0, 1.0, 1.0, 0.5];
        let tree = LowStretchTree::build_low_stretch(3, &tails, &heads, &lengths, 4).unwrap();
        let off_tree_edge = tree
            .tree_edges
            .iter()
            .position(|&is_tree| !is_tree)
            .expect("off-tree edge exists");
        let cycle = tree
            .fundamental_cycle(off_tree_edge, &tails, &heads)
            .expect("cycle exists");
        assert!(cycle.iter().any(|(edge_id, _)| *edge_id == off_tree_edge));
    }

    #[test]
    fn low_stretch_build_uses_tree_edges() {
        let tails = vec![0, 0, 1, 2];
        let heads = vec![1, 2, 2, 3];
        let lengths = vec![1.0, 3.0, 1.0, 1.0];
        let tree = LowStretchTree::build_low_stretch(4, &tails, &heads, &lengths, 5).unwrap();
        assert_eq!(tree.tree_edges.iter().filter(|&&b| b).count(), 3);
        assert!(tree.path_length(0, 3).is_some());
    }

    #[test]
    fn average_stretch_is_finite() {
        let tails = vec![0, 1, 2, 0, 3];
        let heads = vec![1, 2, 3, 3, 4];
        let lengths = vec![1.0, 1.5, 2.0, 2.5, 1.0];
        let tree = LowStretchTree::build_low_stretch(5, &tails, &heads, &lengths, 11).unwrap();
        let stretch = tree.average_stretch(&tails, &heads, &lengths).unwrap();
        assert!(stretch.is_finite());
        assert!(stretch >= 1.0);
    }

    #[test]
    fn edge_stretch_returns_none_for_invalid_length() {
        let tails = vec![0];
        let heads = vec![1];
        let lengths = vec![0.0];
        let tree = LowStretchTree::build(2, &tails, &heads, &lengths, 3).unwrap();
        assert!(tree.edge_stretch(0, &tails, &heads, &lengths).is_none());
    }
}
