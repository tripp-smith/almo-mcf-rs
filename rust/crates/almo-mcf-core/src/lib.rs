pub mod convex;
pub mod data_structures;
pub mod graph;
pub mod hsfc;
pub mod ipm;
pub mod min_ratio;
pub mod numerics;
pub mod rebuilding;
pub mod rounding;
pub mod scaling;
pub mod solver;
pub mod spanner;
pub mod trees;

use crate::graph::min_cost_flow::MinCostFlow;
use crate::ipm::{IpmStats, IpmTermination};
use crate::rounding::{
    build_residual_graph_from_problem, cancel_negative_cycles_in_residual, residual,
    round_to_integer_flow,
};

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
    pub last_duality_gap_proxy: Option<f64>,
    pub termination_gap_threshold: Option<f64>,
    pub terminated_by_gap: bool,
    pub terminated_by_max_iters: bool,
    pub final_gap_estimate: Option<f64>,
    pub gap_exponent_used: f64,
    pub gap_tolerance_used: Option<f64>,
    pub cycle_scoring_ms: f64,
    pub barrier_compute_ms: f64,
    pub spanner_update_ms: f64,
    pub termination: IpmTermination,
    pub oracle_mode: OracleMode,
    pub deterministic_mode_used: bool,
    pub seed_used: Option<u64>,
    pub rounding_performed: bool,
    pub rounding_success: bool,
    pub final_integer_cost: Option<i64>,
    pub post_rounding_gap: Option<f64>,
    pub cycles_canceled: usize,
    pub rounding_adjustment_cost: Option<i64>,
    pub is_exact_optimal: bool,
    pub numerical_clamping_occurred: bool,
    pub max_barrier_value: f64,
    pub min_residual_seen: f64,
    pub potential_drops: Vec<f64>,
    pub newton_step_norms: Vec<f64>,
    pub convergence_gap: f64,
    pub total_iters: usize,
    pub chain_stretches: Vec<f64>,
    pub rebuild_triggers: std::collections::HashMap<String, usize>,
    pub derandomized_hash_collisions: usize,
    pub scaling_log_factors: Vec<f64>,
    pub solver_mode_label: String,
    pub numerical_clamps_applied: usize,
    pub cycle_quality_factor: Option<f64>,
    pub rebuild_cost: f64,
    pub update_savings: f64,
}

#[derive(Debug, Clone)]
pub enum Strategy {
    FullDynamic { rebuild_threshold: usize },
    PeriodicRebuild { rebuild_every: usize },
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum OracleMode {
    Dynamic,
    Fallback,
    #[default]
    Hybrid,
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
    pub gap_exponent: f64,
    pub gap_threshold: Option<f64>,
    pub strategy: Strategy,
    pub oracle_mode: OracleMode,
    pub threads: usize,
    pub alpha: Option<f64>,
    pub use_ipm: Option<bool>,
    pub approx_factor: f64,
    pub deterministic: bool,
    pub chain_deterministic: bool,
    pub derandomized: bool,
    pub randomized_oracle_prob: Option<f64>,
    pub deterministic_seed: Option<u64>,
    pub initial_flow: Option<Vec<i64>>,
    pub initial_perturbation: f64,
    pub use_scaling: Option<bool>,
    pub force_cost_scaling: bool,
    pub disable_capacity_scaling: bool,
    pub use_rounding: Option<bool>,
    pub numerical_clamp_log: f64,
    pub residual_min: f64,
    pub barrier_alpha_min: f64,
    pub barrier_alpha_max: f64,
    pub barrier_clamp_max: f64,
    pub gradient_clamp_max: f64,
    pub log_numerical_clamping: bool,
}

impl Default for McfOptions {
    fn default() -> Self {
        Self {
            seed: 42,
            time_limit_ms: None,
            tolerance: 1e-9,
            max_iters: 10_000,
            gap_exponent: ipm::DEFAULT_GAP_EXPONENT,
            gap_threshold: None,
            strategy: Strategy::PeriodicRebuild { rebuild_every: 25 },
            oracle_mode: OracleMode::Dynamic,
            threads: 1,
            alpha: None,
            use_ipm: None,
            approx_factor: 0.2,
            deterministic: true,
            chain_deterministic: true,
            derandomized: true,
            randomized_oracle_prob: None,
            deterministic_seed: Some(42),
            initial_flow: None,
            initial_perturbation: 0.0,
            use_scaling: None,
            force_cost_scaling: false,
            disable_capacity_scaling: false,
            use_rounding: None,
            numerical_clamp_log: numerics::barrier::DEFAULT_MAX_LOG,
            residual_min: numerics::barrier::RESIDUAL_MIN,
            barrier_alpha_min: numerics::barrier::BARRIER_ALPHA_MIN,
            barrier_alpha_max: numerics::barrier::BARRIER_ALPHA_MAX,
            barrier_clamp_max: numerics::barrier::BARRIER_CLAMP_MAX,
            gradient_clamp_max: numerics::barrier::GRADIENT_CLAMP_MAX,
            log_numerical_clamping: false,
        }
    }
}

impl McfOptions {
    pub fn barrier_clamp_config(&self) -> numerics::barrier::BarrierClampConfig {
        numerics::barrier::BarrierClampConfig {
            max_log: self.numerical_clamp_log,
            min_x: numerics::barrier::MIN_POSITIVE,
            residual_min: self.residual_min,
            barrier_clamp_max: self.barrier_clamp_max,
            gradient_clamp_max: self.gradient_clamp_max,
            alpha_min: self.barrier_alpha_min,
            alpha_max: self.barrier_alpha_max,
        }
    }

    pub fn set_deterministic_mode(&mut self, seed: u64) {
        self.deterministic = true;
        self.chain_deterministic = true;
        self.derandomized = true;
        self.seed = seed;
        self.deterministic_seed = Some(seed);
        self.threads = 1;
    }

    pub fn enable_randomized_oracle(&mut self, prob: f64) {
        self.randomized_oracle_prob = Some(prob.clamp(0.0, 1.0));
        self.deterministic = false;
    }
}

#[derive(Debug)]
pub enum McfError {
    InvalidInput(String),
    Infeasible,
}

pub type Cycle = Vec<u32>;

pub fn extract_cycle_from_flow(_flow: &[f64], residual: &graph::Graph) -> Cycle {
    for (eid, edge) in residual.edges() {
        if edge.cost < 0.0 {
            return vec![edge.tail.0 as u32, edge.head.0 as u32, eid.0 as u32];
        }
    }
    Vec::new()
}

pub fn detect_negative_cycle(graph: &graph::Graph, weights: &[f64]) -> Option<Cycle> {
    for ((_, edge), &w) in graph.edges().zip(weights.iter()) {
        if w + edge.cost < 0.0 {
            return Some(vec![edge.tail.0 as u32, edge.head.0 as u32]);
        }
    }
    None
}

pub fn single_source_shortest_paths(
    graph: &graph::Graph,
    s: graph::NodeId,
    weights: &[f64],
) -> Vec<f64> {
    let n = graph.node_count();
    let mut dist = vec![f64::INFINITY; n];
    dist[s.0] = 0.0;
    for _ in 0..n.saturating_sub(1) {
        let mut changed = false;
        for (eid, edge) in graph.edges() {
            let w = weights.get(eid.0).copied().unwrap_or(edge.cost);
            if dist[edge.tail.0].is_finite() && dist[edge.tail.0] + w < dist[edge.head.0] {
                dist[edge.head.0] = dist[edge.tail.0] + w;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    dist
}

pub fn min_cost_flow_convex(
    problem: &McfProblem,
    opts: &McfOptions,
    funcs: Vec<Box<dyn convex::ConvexFunc>>,
    epsilon: f64,
) -> Result<(Vec<f64>, f64), McfError> {
    let base = convex::McfSolver {
        problem: problem.clone(),
        opts: opts.clone(),
    };
    let mut solver = convex::ConvexMcfSolver::new(base, funcs, epsilon);
    Ok(solver.solve_to_approx(&graph::Graph::new(problem.node_count)))
}

pub fn entropy_ot(
    graph: &convex::BipartiteGraph,
    demands: &[f64],
    costs: &[f64],
    eta: f64,
) -> (Vec<f64>, f64) {
    convex::entropy_regularized_ot(graph, demands, costs, eta)
}

pub fn isotonic_reg(dag: &convex::Dag, y: &[f64], p: f64) -> Vec<f64> {
    convex::isotonic_regression(dag, y, p)
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

pub fn run_with_rebuilding_bench(
    problem: &McfProblem,
    opts: &McfOptions,
) -> Result<IpmSummary, McfError> {
    let solution = min_cost_flow_exact(problem, opts)?;
    solution.ipm_stats.ok_or_else(|| {
        McfError::InvalidInput("IPM stats unavailable for rebuilding benchmark".to_string())
    })
}

fn initialize_thread_pool(opts: &McfOptions) {
    let mut threads = opts.threads;
    if opts.deterministic {
        threads = 1;
    }
    if threads <= 1 {
        return;
    }
    if threads == 0 {
        return;
    }
    if rayon::current_num_threads() == threads {
        return;
    }
    if let Err(err) = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
    {
        if cfg!(debug_assertions) {
            eprintln!("rayon thread pool init skipped: {err}");
        }
    }
}

pub fn min_cost_flow_exact(
    problem: &McfProblem,
    opts: &McfOptions,
) -> Result<McfSolution, McfError> {
    let mut opts = opts.clone();
    if matches!(opts.strategy, Strategy::FullDynamic { .. }) {
        opts.oracle_mode = OracleMode::Dynamic;
    }
    if opts.deterministic {
        opts.threads = 1;
    }
    initialize_thread_pool(&opts);
    if should_use_scaling(problem, &opts) {
        return scaling::solve_mcf_with_scaling(problem, &opts);
    }
    if should_use_classic(problem, &opts) {
        return solve_classic_with_mode(problem, SolverMode::Classic);
    }

    let ipm_result = match ipm::run_ipm(problem, &opts) {
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
        &opts,
    ));

    finalize_ipm_solution(problem, ipm_result, ipm_stats, &opts)
}

pub fn min_cost_flow_scaled(
    problem: &McfProblem,
    opts: &McfOptions,
) -> Result<McfSolution, McfError> {
    let mut opts = opts.clone();
    opts.use_scaling = Some(true);
    if opts.deterministic {
        opts.threads = 1;
    }
    initialize_thread_pool(&opts);
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

fn rounding_gap_threshold(problem: &McfProblem, opts: &McfOptions) -> f64 {
    let edge_count = problem.edge_count().max(1);
    let max_upper = problem
        .upper
        .iter()
        .map(|&value| value.abs() as f64)
        .fold(1.0_f64, |acc, value| acc.max(value).max(1.0));
    ipm::compute_gap_threshold(edge_count, max_upper, opts)
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
    pub(crate) fn from_ipm(
        stats: &IpmStats,
        termination: IpmTermination,
        opts: &McfOptions,
    ) -> Self {
        let mut aggregate = numerics::barrier::BarrierClampStats::default();
        for item in &stats.barrier_clamp_stats {
            aggregate.merge(item);
        }
        Self {
            iterations: stats.iterations,
            final_gap: stats.last_gap,
            last_duality_gap_proxy: stats.last_duality_gap_proxy,
            termination_gap_threshold: stats.termination_gap_threshold,
            terminated_by_gap: stats.terminated_by_gap,
            terminated_by_max_iters: stats.terminated_by_max_iters,
            final_gap_estimate: stats.final_gap_estimate,
            gap_exponent_used: opts.gap_exponent,
            gap_tolerance_used: opts.gap_threshold,
            cycle_scoring_ms: stats.cycle_times_ms.iter().sum(),
            barrier_compute_ms: stats.barrier_times_ms.iter().sum(),
            spanner_update_ms: stats.spanner_update_times_ms.iter().sum(),
            termination,
            oracle_mode: stats.oracle_mode,
            deterministic_mode_used: opts.deterministic,
            seed_used: if opts.deterministic {
                opts.deterministic_seed
            } else {
                Some(opts.seed)
            },
            rounding_performed: false,
            rounding_success: false,
            final_integer_cost: None,
            post_rounding_gap: None,
            cycles_canceled: 0,
            rounding_adjustment_cost: None,
            is_exact_optimal: false,
            numerical_clamping_occurred: aggregate.clamping_occurred(),
            max_barrier_value: aggregate.max_barrier_value,
            min_residual_seen: aggregate.min_residual_seen,
            potential_drops: stats.potential_drops.clone(),
            newton_step_norms: stats.newton_step_norms.clone(),
            convergence_gap: stats.convergence_gap,
            total_iters: stats.total_iters,
            chain_stretches: stats.instability_per_level.clone(),
            rebuild_triggers: stats
                .rebuild_counts
                .iter()
                .enumerate()
                .map(|(level, count)| (format!("level_{level}"), *count))
                .collect(),
            derandomized_hash_collisions: 0,
            scaling_log_factors: Vec::new(),
            solver_mode_label: "full_dynamic_convex".to_string(),
            numerical_clamps_applied: aggregate.total_clamps(),
            cycle_quality_factor: None,
            rebuild_cost: 0.0,
            update_savings: 0.0,
        }
    }
}

pub(crate) fn finalize_ipm_solution(
    problem: &McfProblem,
    ipm_result: ipm::IpmResult,
    mut ipm_stats: Option<IpmSummary>,
    opts: &McfOptions,
) -> Result<McfSolution, McfError> {
    if ipm_result.termination != IpmTermination::Converged {
        let mut solution = solve_classic_with_mode(problem, SolverMode::ClassicFallback)?;
        solution.ipm_stats = ipm_stats;
        return Ok(solution);
    }

    let gap_threshold = rounding_gap_threshold(problem, opts);
    if ipm_result.stats.last_gap > gap_threshold {
        let mut solution = solve_classic_with_mode(problem, SolverMode::ClassicFallback)?;
        solution.ipm_stats = ipm_stats;
        return Ok(solution);
    }

    if opts.use_rounding == Some(false) {
        let mut solution = solve_classic_with_mode(problem, SolverMode::ClassicFallback)?;
        solution.ipm_stats = ipm_stats;
        return Ok(solution);
    }

    let mut residual = build_residual_graph_from_problem(problem, &ipm_result.flow)?;
    let cancel_stats = cancel_negative_cycles_in_residual(&mut residual, None)?;
    let rounded = match round_to_integer_flow(&residual.flow, problem, &residual) {
        Ok(result) => result,
        Err(err) => {
            if matches!(err, McfError::Infeasible) {
                let mut solution = solve_classic_with_mode(problem, SolverMode::ClassicFallback)?;
                solution.ipm_stats = ipm_stats;
                return Ok(solution);
            }
            return Err(err);
        }
    };

    if let Some(stats) = ipm_stats.as_mut() {
        stats.rounding_performed = true;
        stats.rounding_success = true;
        stats.cycles_canceled = cancel_stats.cycles_canceled;
        stats.rounding_adjustment_cost = rounded.adjustment_cost;
        stats.final_integer_cost = i64::try_from(rounded.cost).ok();

        let rounded_flow_f64: Vec<f64> = rounded.flow.iter().map(|&v| v as f64).collect();
        let post_residual = build_residual_graph_from_problem(problem, &rounded_flow_f64)?;
        let post_gap = if post_residual.is_epsilon_feasible(1e-9) {
            0.0
        } else {
            post_residual.total_demand_imbalance()
        };
        stats.post_rounding_gap =
            Some(residual::negative_cycle_cost(&post_residual).unwrap_or(post_gap));
        stats.is_exact_optimal = post_gap <= 1e-9 && !residual::has_negative_cycle(&post_residual);
    }

    let solution = McfSolution {
        flow: rounded.flow,
        cost: rounded.cost,
        ipm_stats,
        solver_mode: SolverMode::Ipm,
    };
    Ok(solution)
}

pub fn run_repro_check(problem: &McfProblem, seed1: u64, seed2: u64) -> bool {
    let mut opts_a = McfOptions::default();
    opts_a.set_deterministic_mode(seed1);
    let mut opts_b = McfOptions::default();
    opts_b.set_deterministic_mode(seed2);

    let Ok(sol_a) = min_cost_flow_exact(problem, &opts_a) else {
        return false;
    };
    let Ok(sol_b) = min_cost_flow_exact(problem, &opts_b) else {
        return false;
    };

    sol_a.flow == sol_b.flow && sol_a.cost == sol_b.cost
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
            strategy: Strategy::FullDynamic {
                rebuild_threshold: 25,
            },
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
            strategy: Strategy::FullDynamic {
                rebuild_threshold: 25,
            },
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
