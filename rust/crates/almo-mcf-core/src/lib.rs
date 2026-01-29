pub mod graph;
pub mod ipm;
pub mod min_ratio;
pub mod numerics;
pub mod rounding;
pub mod spanner;
pub mod trees;

use crate::graph::min_cost_flow::MinCostFlow;

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
    fn handles_parallel_edges_with_costs() {
        let solution = solve(
            vec![0, 0],
            vec![1, 1],
            vec![0, 0],
            vec![1, 4],
            vec![1, 3],
            vec![-3, 3],
        );
        assert_eq!(solution.flow, vec![1, 2]);
        assert_eq!(solution.cost, 7);
    }

    #[test]
    fn supports_negative_costs() {
        let solution = solve(
            vec![0, 1],
            vec![1, 2],
            vec![0, 0],
            vec![3, 3],
            vec![-2, 1],
            vec![-2, 0, 2],
        );
        assert_eq!(solution.flow, vec![2, 2]);
        assert_eq!(solution.cost, -2);
    }

    #[test]
    fn honors_lower_bound_circulation() {
        let solution = solve(
            vec![0, 1, 2],
            vec![1, 2, 0],
            vec![1, 1, 1],
            vec![3, 3, 3],
            vec![1, 1, 1],
            vec![0, 0, 0],
        );
        assert_eq!(solution.flow, vec![1, 1, 1]);
        assert_eq!(solution.cost, 3);
    }

    #[test]
    fn detects_infeasible() {
        let problem =
            McfProblem::new(vec![0], vec![1], vec![0], vec![1], vec![1], vec![-2, 2]).unwrap();
        let err = min_cost_flow_exact(&problem, &McfOptions::default()).unwrap_err();
        assert!(matches!(err, McfError::Infeasible));
    }

    #[test]
    fn rejects_unbalanced_demands() {
        let err =
            McfProblem::new(vec![0], vec![1], vec![0], vec![3], vec![1], vec![1, 0]).unwrap_err();
        assert!(matches!(err, McfError::InvalidInput(_)));
    }

    #[test]
    fn rejects_invalid_bounds() {
        let err =
            McfProblem::new(vec![0], vec![1], vec![5], vec![2], vec![1], vec![-1, 1]).unwrap_err();
        assert!(matches!(err, McfError::InvalidInput(_)));
    }

    #[test]
    fn exposes_edge_metadata_helpers() {
        let problem = McfProblem::new(
            vec![0, 1],
            vec![1, 2],
            vec![0, 0],
            vec![3, 4],
            vec![1, 2],
            vec![-1, 0, 1],
        )
        .unwrap();

        assert_eq!(problem.edge_count(), 2);
        assert_eq!(problem.edge_endpoints(1), Some((1, 2)));
        assert_eq!(problem.edge_endpoints(2), None);
    }
}
