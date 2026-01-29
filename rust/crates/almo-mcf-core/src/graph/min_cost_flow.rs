use crate::McfError;

#[derive(Debug, Clone)]
pub(crate) struct Edge {
    pub(crate) to: usize,
    pub(crate) rev: usize,
    pub(crate) cap: i64,
    pub(crate) cost: i64,
}

#[derive(Debug)]
pub(crate) struct MinCostFlow {
    pub(crate) graph: Vec<Vec<Edge>>,
}

impl MinCostFlow {
    pub(crate) fn new(nodes: usize) -> Self {
        Self {
            graph: vec![Vec::new(); nodes],
        }
    }

    pub(crate) fn add_edge(
        &mut self,
        from: usize,
        to: usize,
        cap: i64,
        cost: i64,
    ) -> (usize, usize) {
        let from_index = self.graph[from].len();
        let to_index = self.graph[to].len();
        self.graph[from].push(Edge {
            to,
            rev: to_index,
            cap,
            cost,
        });
        self.graph[to].push(Edge {
            to: from,
            rev: from_index,
            cap: 0,
            cost: -cost,
        });
        (from_index, to_index)
    }

    pub(crate) fn shortest_path_bellman_ford(
        &self,
        source: usize,
    ) -> (Vec<i64>, Vec<usize>, Vec<usize>) {
        let n = self.graph.len();
        let mut dist = vec![i64::MAX / 4; n];
        let mut prev_node = vec![usize::MAX; n];
        let mut prev_edge = vec![usize::MAX; n];
        dist[source] = 0;

        for _ in 0..n {
            let mut updated = false;
            for u in 0..n {
                let du = dist[u];
                if du >= i64::MAX / 8 {
                    continue;
                }
                for (edge_idx, edge) in self.graph[u].iter().enumerate() {
                    if edge.cap <= 0 {
                        continue;
                    }
                    let nd = du.saturating_add(edge.cost);
                    if nd < dist[edge.to] {
                        dist[edge.to] = nd;
                        prev_node[edge.to] = u;
                        prev_edge[edge.to] = edge_idx;
                        updated = true;
                    }
                }
            }
            if !updated {
                break;
            }
        }
        (dist, prev_node, prev_edge)
    }

    pub(crate) fn min_cost_flow(
        &mut self,
        source: usize,
        sink: usize,
        mut flow: i64,
    ) -> Result<i128, McfError> {
        let mut total_cost: i128 = 0;

        while flow > 0 {
            let (dist, prev_node, prev_edge) = self.shortest_path_bellman_ford(source);

            if dist[sink] >= i64::MAX / 8 {
                return Err(McfError::Infeasible);
            }

            let mut add_flow = flow;
            let mut v = sink;
            while v != source {
                let u = prev_node[v];
                let eidx = prev_edge[v];
                if u == usize::MAX || eidx == usize::MAX {
                    return Err(McfError::Infeasible);
                }
                let cap = self.graph[u][eidx].cap;
                if cap < add_flow {
                    add_flow = cap;
                }
                v = u;
            }

            v = sink;
            while v != source {
                let u = prev_node[v];
                let eidx = prev_edge[v];
                let rev = self.graph[u][eidx].rev;
                let edge_cost = self.graph[u][eidx].cost as i128;
                self.graph[u][eidx].cap -= add_flow;
                self.graph[v][rev].cap += add_flow;
                total_cost += edge_cost * add_flow as i128;
                v = u;
            }

            flow -= add_flow;
        }

        Ok(total_cost)
    }
}
