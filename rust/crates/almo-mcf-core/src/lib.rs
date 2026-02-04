pub mod graph;
pub mod hsfc;
pub mod ipm;
pub mod min_ratio;
pub mod numerics;
pub mod rounding;
pub mod scaling;
pub mod spanner;
pub mod trees;

use crate::graph::min_cost_flow::MinCostFlow;
use crate::ipm::{IpmStats, IpmTermination};
use crate::rounding::round_fractional_flow;

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
    pub ipm_stats: Option<IpmSummary>,
    pub solver_mode: SolverMode,
}

#[derive(Debug, Clone)]
pub struct IpmSummary {
    pub iterations: usize,
    pub final_gap: f64,
    pub termination: IpmTermination,
    pub oracle_mode: OracleMode,
}

#[derive(Debug, Clone)]
pub enum Strategy {
    FullDynamic,
    PeriodicRebuild { rebuild_every: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OracleMode {
    Dynamic,
    Fallback,
    Hybrid,
}

impl Default for OracleMode {
    fn default() -> Self {
        OracleMode::Hybrid
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverMode {
    Classic,
    Ipm,
    IpmScaled,
    ClassicFallback,
}

#[derive(Debug, Clone)]
pub struct McfOptions {
    pub seed: u64,
    pub time_limit_ms: Option<u64>,
    pub tolerance: f64,
    pub max_iters: usize,
    pub strategy: Strategy,
    pub oracle_mode: OracleMode,
    pub threads: usize,
    pub alpha: Option<f64>,
    pub use_ipm: Option<bool>,
    pub approx_factor: f64,
    pub deterministic: bool,
    pub initial_flow: Option<Vec<i64>>,
    pub initial_perturbation: f64,
    pub use_scaling: Option<bool>,
    pub force_cost_scaling: bool,
    pub disable_capacity_scaling: bool,
}

impl Default for McfOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            time_limit_ms: None,
            tolerance: 1e-9,
            max_iters: 10_000,
            strategy: Strategy::PeriodicRebuild { rebuild_every: 25 },
            oracle_mode: OracleMode::Hybrid,
            threads: 1,
            alpha: None,
            use_ipm: None,
            approx_factor: 0.1,
            deterministic: true,
            initial_flow: None,
            initial_perturbation: 0.0,
            use_scaling: None,
            force_cost_scaling: false,
            disable_capacity_scaling: false,
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
    opts: &McfOptions,
) -> Result<McfSolution, McfError> {
    if should_use_scaling(problem, opts) {
        return scaling::solve_mcf_with_scaling(problem, opts);
    }
    if should_use_classic(problem, opts) {
        return solve_classic_with_mode(problem, SolverMode::Classic);
    }

    let ipm_result = match ipm::run_ipm(problem, opts) {
        Ok(result) => result,
        Err(err) => {
            if matches!(err, McfError::InvalidInput(_)) {
                return Err(err);
            }
            return solve_classic_with_mode(problem, SolverMode::ClassicFallback).or(Err(err));
        }
    };
    let ipm_stats = Some(IpmSummary::from_ipm(
        &ipm_result.stats,
        ipm_result.termination,
    ));

    finalize_ipm_solution(problem, ipm_result, ipm_stats)
}

pub fn min_cost_flow_scaled(
    problem: &McfProblem,
    opts: &McfOptions,
) -> Result<McfSolution, McfError> {
    let mut opts = opts.clone();
    opts.use_scaling = Some(true);
    scaling::solve_mcf_with_scaling(problem, &opts)
}

fn should_use_classic(problem: &McfProblem, opts: &McfOptions) -> bool {
    const SMALL_EDGE_LIMIT: usize = 12;
    const SMALL_NODE_LIMIT: usize = 8;
    if opts.use_ipm == Some(false) {
        return true;
    }
    if opts.max_iters == 0 {
        return true;
    }
    if opts.use_ipm == Some(true) {
        return false;
    }
    problem.edge_count() <= SMALL_EDGE_LIMIT || problem.node_count <= SMALL_NODE_LIMIT
}

fn should_use_scaling(problem: &McfProblem, opts: &McfOptions) -> bool {
    if opts.force_cost_scaling {
        return true;
    }
    if opts.use_scaling == Some(false) {
        return false;
    }
    if should_use_classic(problem, opts) {
        return false;
    }
    if opts.use_ipm == Some(false) || opts.max_iters == 0 {
        return false;
    }
    if opts.use_scaling == Some(true) {
        return true;
    }

    let m = problem.edge_count().max(1) as i64;
    let bound = m.saturating_pow(3).max(1);
    let poly_log_m = 10.0 * (m as f64).log2().max(1.0);
    let max_capacity = problem
        .upper
        .iter()
        .map(|value| value.abs())
        .max()
        .unwrap_or(0);
    let max_cost = problem
        .cost
        .iter()
        .map(|value| value.abs())
        .max()
        .unwrap_or(0);

    let log_u = (max_capacity.max(1) as f64).log2();
    let log_c = (max_cost.max(1) as f64).log2();

    log_u > poly_log_m || log_c > poly_log_m || max_capacity > bound || max_cost > bound
}

fn rounding_gap_threshold(problem: &McfProblem) -> f64 {
    let edge_count = problem.edge_count().max(1) as f64;
    let max_upper = problem
        .upper
        .iter()
        .map(|&value| value.abs() as f64)
        .fold(1.0_f64, |acc, value| acc.max(value).max(1.0));
    let m_u = (edge_count * max_upper).max(1.0);
    m_u.powf(-10.0)
}

fn solve_classic_with_mode(
    problem: &McfProblem,
    solver_mode: SolverMode,
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

    Ok(McfSolution {
        flow,
        cost,
        ipm_stats: None,
        solver_mode,
    })
}

impl IpmSummary {
    pub(crate) fn from_ipm(stats: &IpmStats, termination: IpmTermination) -> Self {
        Self {
            iterations: stats.iterations,
            final_gap: stats.last_gap,
            termination,
            oracle_mode: stats.oracle_mode,
        }
    }
}

pub(crate) fn finalize_ipm_solution(
    problem: &McfProblem,
    ipm_result: ipm::IpmResult,
    ipm_stats: Option<IpmSummary>,
) -> Result<McfSolution, McfError> {
    if ipm_result.termination != IpmTermination::Converged {
        let mut solution = solve_classic_with_mode(problem, SolverMode::ClassicFallback)?;
        solution.ipm_stats = ipm_stats;
        return Ok(solution);
    }

    let gap_threshold = rounding_gap_threshold(problem);
    if ipm_result.stats.last_gap > gap_threshold {
        let mut solution = solve_classic_with_mode(problem, SolverMode::ClassicFallback)?;
        solution.ipm_stats = ipm_stats;
        return Ok(solution);
    }

    let rounded = match round_fractional_flow(problem, &ipm_result.flow) {
        Ok(solution) => solution,
        Err(err) => {
            if matches!(err, McfError::Infeasible) {
                let mut solution = solve_classic_with_mode(problem, SolverMode::ClassicFallback)?;
                solution.ipm_stats = ipm_stats;
                return Ok(solution);
            }
            return Err(err);
        }
    };
    let mut solution = rounded;
    solution.ipm_stats = ipm_stats;
    solution.solver_mode = SolverMode::Ipm;
    Ok(solution)
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
        assert!(solution.ipm_stats.is_none());
        assert_eq!(solution.solver_mode, SolverMode::Classic);
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
        assert!(solution.ipm_stats.is_none());
        assert_eq!(solution.solver_mode, SolverMode::Classic);
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
        assert!(solution.ipm_stats.is_none());
        assert_eq!(solution.solver_mode, SolverMode::Classic);
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
        assert!(solution.ipm_stats.is_none());
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
        assert!(solution.ipm_stats.is_none());
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

    #[test]
    fn uses_ipm_for_larger_instances() {
        let mut tails = Vec::new();
        let mut heads = Vec::new();
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        let mut cost = Vec::new();
        let node_count = 9;
        for i in 0..13 {
            let tail = i % node_count;
            let head = (i + 1) % node_count;
            tails.push(tail as u32);
            heads.push(head as u32);
            lower.push(0);
            upper.push(2);
            cost.push((i as i64 % 3) - 1);
        }
        let demand = vec![0_i64; node_count];
        let problem = McfProblem::new(tails, heads, lower, upper, cost, demand).unwrap();
        let opts = McfOptions {
            strategy: Strategy::FullDynamic,
            max_iters: 50,
            ..McfOptions::default()
        };
        let solution = min_cost_flow_exact(&problem, &opts).unwrap();
        let stats = solution.ipm_stats.expect("expected IPM stats");
        assert!(stats.iterations <= opts.max_iters);
        assert!(stats.final_gap.is_finite());
    }

    #[test]
    fn falls_back_to_classic_on_small_graphs() {
        let problem = McfProblem::new(
            vec![0, 1, 2],
            vec![1, 2, 0],
            vec![0, 0, 0],
            vec![2, 2, 2],
            vec![1, 1, 1],
            vec![0, 0, 0],
        )
        .unwrap();
        let opts = McfOptions {
            strategy: Strategy::FullDynamic,
            ..McfOptions::default()
        };
        let solution = min_cost_flow_exact(&problem, &opts).unwrap();
        assert!(solution.ipm_stats.is_none());
    }

    #[test]
    fn falls_back_to_classic_when_ipm_iteration_limit_hits() {
        let mut tails = Vec::new();
        let mut heads = Vec::new();
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        let mut cost = Vec::new();
        let node_count = 9;
        for i in 0..13 {
            let tail = i % node_count;
            let head = (i + 1) % node_count;
            tails.push(tail as u32);
            heads.push(head as u32);
            lower.push(0);
            upper.push(2);
            cost.push(-1);
        }
        let demand = vec![0_i64; node_count];
        let problem = McfProblem::new(tails, heads, lower, upper, cost, demand).unwrap();
        let opts = McfOptions {
            strategy: Strategy::PeriodicRebuild { rebuild_every: 2 },
            max_iters: 1,
            ..McfOptions::default()
        };
        let solution = min_cost_flow_exact(&problem, &opts).unwrap();
        let stats = solution.ipm_stats.expect("expected IPM stats");
        assert!(
            matches!(
                stats.termination,
                IpmTermination::IterationLimit | IpmTermination::Converged
            ),
            "unexpected termination: {:?}",
            stats.termination
        );
    }
}
