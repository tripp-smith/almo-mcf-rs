pub mod graph;
pub mod ipm;
pub mod min_ratio;
pub mod numerics;
pub mod rounding;
pub mod spanner;
pub mod trees;

#[derive(Debug, Clone)]
pub struct McfProblem {
    pub tails: Vec<u32>,
    pub heads: Vec<u32>,
    pub lower: Vec<i64>,
    pub upper: Vec<i64>,
    pub cost: Vec<i64>,
    pub demands: Vec<i64>,
    pub edge_count: usize,
    pub node_count: usize,
}

#[derive(Debug, Clone)]
pub struct McfSolution {
    pub flow: Vec<i64>,
    pub cost: i128,
}

#[derive(Debug, Clone)]
pub enum Strategy {
    FullDynamic,
    PeriodicRebuild { rebuild_every: usize },
}

#[derive(Debug, Clone)]
pub struct McfOptions {
    pub seed: u64,
    pub time_limit_ms: Option<u64>,
    pub tolerance: f64,
    pub max_iters: usize,
    pub strategy: Strategy,
    pub threads: usize,
}

impl Default for McfOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            time_limit_ms: None,
            tolerance: 1e-9,
            max_iters: 10_000,
            strategy: Strategy::PeriodicRebuild { rebuild_every: 25 },
            threads: 1,
        }
    }
}

#[derive(Debug)]
pub enum McfError {
    InvalidInput(String),
    Infeasible,
}

impl McfProblem {
    pub fn new(
        tails: Vec<u32>,
        heads: Vec<u32>,
        lower: Vec<i64>,
        upper: Vec<i64>,
        cost: Vec<i64>,
        demands: Vec<i64>,
    ) -> Result<Self, McfError> {
        let edge_count = tails.len();
        if heads.len() != edge_count
            || lower.len() != edge_count
            || upper.len() != edge_count
            || cost.len() != edge_count
        {
            return Err(McfError::InvalidInput(
                "edge arrays must have identical length".to_string(),
            ));
        }
        let node_count = demands.len();
        for (&t, &h) in tails.iter().zip(heads.iter()) {
            if t as usize >= node_count || h as usize >= node_count {
                return Err(McfError::InvalidInput(
                    "edge endpoint outside node range".to_string(),
                ));
            }
        }
        let demand_sum: i64 = demands.iter().sum();
        if demand_sum != 0 {
            return Err(McfError::InvalidInput(
                "demands must sum to zero".to_string(),
            ));
        }
        for (&lo, &up) in lower.iter().zip(upper.iter()) {
            if lo > up {
                return Err(McfError::InvalidInput(
                    "lower bound exceeds upper bound".to_string(),
                ));
            }
        }
        Ok(Self {
            tails,
            heads,
            lower,
            upper,
            cost,
            demands,
            edge_count,
            node_count,
        })
    }

    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    pub fn node_ids(&self) -> graph::IdMapping {
        graph::IdMapping::from_range(self.node_count as u32)
    }

    pub fn edge_endpoints(&self, edge_id: usize) -> Option<(u32, u32)> {
        if edge_id >= self.edge_count {
            return None;
        }
        Some((self.tails[edge_id], self.heads[edge_id]))
    }
}

#[derive(Debug, Clone)]
struct Edge {
    to: usize,
    rev: usize,
    cap: i64,
    cost: i64,
}

#[derive(Debug)]
struct MinCostFlow {
    graph: Vec<Vec<Edge>>,
}

impl MinCostFlow {
    fn new(nodes: usize) -> Self {
        Self {
            graph: vec![Vec::new(); nodes],
        }
    }

    fn add_edge(&mut self, from: usize, to: usize, cap: i64, cost: i64) -> (usize, usize) {
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

    fn shortest_path_bellman_ford(&self, source: usize) -> (Vec<i64>, Vec<usize>, Vec<usize>) {
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

    fn min_cost_flow(
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

pub fn min_cost_flow_exact(
    problem: &McfProblem,
    _opts: &McfOptions,
) -> Result<McfSolution, McfError> {
    let n = problem.node_count;
    let m = problem.edge_count();

    let mut demand = problem.demands.clone();
    let mut residual_upper = vec![0_i64; m];
    for (i, residual) in residual_upper.iter_mut().enumerate() {
        let lo = problem.lower[i];
        let up = problem.upper[i];
        if lo > up {
            return Err(McfError::InvalidInput(
                "lower bound exceeds upper bound".to_string(),
            ));
        }
        *residual = up - lo;
        let tail = problem.tails[i] as usize;
        let head = problem.heads[i] as usize;
        demand[tail] = demand[tail].saturating_add(lo);
        demand[head] = demand[head].saturating_sub(lo);
    }

    let total_nodes = n + 2;
    let source = n;
    let sink = n + 1;
    let mut mcf = MinCostFlow::new(total_nodes);
    let mut edge_refs = Vec::with_capacity(m);

    for (i, &cap) in residual_upper.iter().enumerate() {
        if cap < 0 {
            return Err(McfError::InvalidInput(
                "negative residual capacity".to_string(),
            ));
        }
        let tail = problem.tails[i] as usize;
        let head = problem.heads[i] as usize;
        let (idx, _rev) = mcf.add_edge(tail, head, cap, problem.cost[i]);
        edge_refs.push((tail, idx, cap));
    }

    let mut total_demand = 0_i64;
    for (node, &b) in demand.iter().enumerate() {
        if b > 0 {
            mcf.add_edge(node, sink, b, 0);
        } else if b < 0 {
            mcf.add_edge(source, node, -b, 0);
            total_demand += -b;
        }
    }

    if total_demand > 0 {
        mcf.min_cost_flow(source, sink, total_demand)?;
    }

    let mut flow = vec![0_i64; m];
    for (i, (tail, idx, cap)) in edge_refs.iter().enumerate() {
        let edge = &mcf.graph[*tail][*idx];
        let used = cap - edge.cap;
        flow[i] = problem.lower[i] + used;
    }

    let cost = flow
        .iter()
        .zip(problem.cost.iter())
        .map(|(&f, &c)| f as i128 * c as i128)
        .sum::<i128>();

    Ok(McfSolution { flow, cost })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solve(
        tail: Vec<u32>,
        head: Vec<u32>,
        lower: Vec<i64>,
        upper: Vec<i64>,
        cost: Vec<i64>,
        demand: Vec<i64>,
    ) -> McfSolution {
        let problem = McfProblem::new(tail, head, lower, upper, cost, demand).unwrap();
        min_cost_flow_exact(&problem, &McfOptions::default()).unwrap()
    }

    #[test]
    fn solves_simple_supply_demand() {
        let solution = solve(
            vec![0, 0, 1],
            vec![1, 2, 2],
            vec![0, 0, 0],
            vec![5, 5, 5],
            vec![2, 1, 3],
            vec![-3, 0, 3],
        );
        assert_eq!(solution.flow, vec![0, 3, 0]);
        assert_eq!(solution.cost, 3);
    }

    #[test]
    fn respects_lower_bounds() {
        let solution = solve(
            vec![0, 1],
            vec![1, 2],
            vec![2, 1],
            vec![5, 4],
            vec![1, 2],
            vec![-3, 0, 3],
        );
        assert_eq!(solution.flow, vec![3, 3]);
        assert_eq!(solution.cost, 9);
    }

    #[test]
    fn detects_infeasible() {
        let problem =
            McfProblem::new(vec![0], vec![1], vec![0], vec![1], vec![1], vec![-2, 2]).unwrap();
        let err = min_cost_flow_exact(&problem, &McfOptions::default()).unwrap_err();
        assert!(matches!(err, McfError::Infeasible));
    }
}
