use crate::graph::min_cost_flow::MinCostFlow;
use crate::min_ratio::dynamic::FullDynamicOracle;
use crate::min_ratio::oracle::{DynamicUpdateOracle, SparseFlowDelta};
use crate::min_ratio::{MinRatioOracle, OracleQuery};
use crate::numerics::barrier::BarrierClampStats;
use crate::{McfError, McfOptions, McfProblem, OracleMode, Strategy};
use std::time::Instant;

mod potential;
mod rebuilding;
mod search;
mod termination;
use rebuilding::{Outcome, RebuildingGame};

pub use potential::{Potential, Residuals};
pub use termination::{
    compute_gap_threshold, estimate_duality_gap, should_terminate, DEFAULT_GAP_EXPONENT,
    DEFAULT_GAP_THRESHOLD_FACTOR,
};

#[derive(Debug, Default, Clone)]
pub struct IpmState {
    pub flow: Vec<f64>,
    pub potential: f64,
}

#[derive(Debug, Default, Clone)]
pub struct IpmStats {
    pub iterations: usize,
    pub last_step_size: f64,
    pub potentials: Vec<f64>,
    pub last_gap: f64,
    pub last_duality_gap_proxy: Option<f64>,
    pub termination_gap_threshold: Option<f64>,
    pub terminated_by_gap: bool,
    pub terminated_by_max_iters: bool,
    pub final_gap_estimate: Option<f64>,
    pub oracle_mode: OracleMode,
    pub cycle_times_ms: Vec<f64>,
    pub barrier_times_ms: Vec<f64>,
    pub update_times_ms: Vec<f64>,
    pub spanner_update_times_ms: Vec<f64>,
    pub instability_per_level: Vec<f64>,
    pub rebuild_counts: Vec<usize>,
    pub oracle_update_count: usize,
    pub oracle_fallback_count: usize,
    pub barrier_clamp_stats: Vec<BarrierClampStats>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpmTermination {
    Converged,
    IterationLimit,
    TimeLimit,
    NoImprovingCycle,
}

#[derive(Debug, Clone)]
pub struct IpmResult {
    pub flow: Vec<f64>,
    pub stats: IpmStats,
    pub termination: IpmTermination,
}

#[derive(Debug, Clone)]
pub struct FeasibleFlow {
    pub flow: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub enum IpmRunKind {
    Default,
    CostScaling,
    CapacityScaling,
}

#[derive(Debug, Clone, Copy)]
pub struct IpmRunContext {
    pub kind: IpmRunKind,
    pub round: usize,
}

#[derive(Debug, Clone, Default)]
pub struct HiddenStableFlow {
    pub width: Vec<f64>,
    pub monotonicity_counter: usize,
}

impl HiddenStableFlow {
    pub fn record_width(&mut self, w_t: f64) {
        self.width.push(w_t);
        self.monotonicity_counter = self.monotonicity_counter.saturating_add(1);
    }

    pub fn width_sum(&self) -> f64 {
        self.width.iter().sum()
    }
}

pub fn initialize_feasible_flow(problem: &McfProblem) -> Result<FeasibleFlow, McfError> {
    let base_flow = solve_feasible_flow_with_costs(
        problem.node_count,
        &problem.tails,
        &problem.heads,
        &problem.lower,
        &problem.upper,
        &problem.demands,
        None,
    )?;
    let flow = push_inside_strict(problem, &base_flow)?;
    Ok(FeasibleFlow { flow })
}

pub fn run_ipm(problem: &McfProblem, opts: &McfOptions) -> Result<IpmResult, McfError> {
    // Deterministic mode ensures tree, spanner, and oracle updates are reproducible.
    run_ipm_with_context(
        problem,
        opts,
        IpmRunContext {
            kind: IpmRunKind::Default,
            round: 0,
        },
    )
}

pub fn run_ipm_with_context(
    problem: &McfProblem,
    opts: &McfOptions,
    context: IpmRunContext,
) -> Result<IpmResult, McfError> {
    let mut adjusted_opts = opts.clone();
    if matches!(
        context.kind,
        IpmRunKind::CostScaling | IpmRunKind::CapacityScaling
    ) {
        adjusted_opts.tolerance = (adjusted_opts.tolerance * 10.0).max(1e-8);
        let max_u = problem
            .upper
            .iter()
            .map(|&value| (value as f64).abs())
            .fold(1.0_f64, f64::max);
        let max_c = problem
            .cost
            .iter()
            .map(|&value| (value as f64).abs())
            .fold(1.0_f64, f64::max);
        let scale_factor = max_u.max(max_c).log10().ceil().max(1.0);
        adjusted_opts.numerical_clamp_log =
            (adjusted_opts.numerical_clamp_log - scale_factor * 10.0).max(100.0);
    }

    let bounds = CostBounds::new(problem)?;
    let low = bounds.low as f64;
    let high = bounds.high as f64;
    let epsilon = adjusted_opts.tolerance.max(1e-8);
    let (best_cost, best) = binary_search_with_result(problem, &adjusted_opts, low, high, epsilon)?;

    if let Some(best) = best {
        return Ok(best);
    }

    run_ipm_with_lower_bound(problem, &adjusted_opts, best_cost)
}

pub(crate) fn run_ipm_with_lower_bound(
    problem: &McfProblem,
    opts: &McfOptions,
    cost_lower_bound: f64,
) -> Result<IpmResult, McfError> {
    let feasible = initialize_feasible_flow_with_options(problem, opts)?;
    let mut flow = feasible.flow;
    let lower: Vec<f64> = problem.lower.iter().map(|&v| v as f64).collect();
    let upper: Vec<f64> = problem.upper.iter().map(|&v| v as f64).collect();
    let cost: Vec<f64> = problem.cost.iter().map(|&v| v as f64).collect();
    let clamp_config = opts.barrier_clamp_config();
    let potential = Potential::new_with_lower_bound(
        &upper,
        opts.alpha,
        None,
        opts.threads,
        cost_lower_bound,
        clamp_config,
    );
    let termination_target = potential.termination_target();

    let mut fallback_oracle = None;
    let mut dynamic_oracle = None;
    let fallback_rebuild_every = match opts.strategy {
        Strategy::PeriodicRebuild { rebuild_every } => rebuild_every,
        Strategy::FullDynamic => 25,
    };
    let approx_kappa = opts
        .approx_factor
        .max((problem.edge_count().max(1) as f64).powf(-0.01));
    match opts.oracle_mode {
        OracleMode::Fallback => {
            fallback_oracle = Some(MinRatioOracle::new_with_mode(
                opts.seed,
                fallback_rebuild_every,
                opts.deterministic,
                opts.deterministic_seed,
            ));
        }
        OracleMode::Dynamic => {
            let dynamic_seed = if opts.deterministic {
                opts.deterministic_seed.unwrap_or(0)
            } else {
                opts.seed
            };
            dynamic_oracle = Some(FullDynamicOracle::new(
                dynamic_seed,
                3,
                1,
                10,
                approx_kappa,
                opts.deterministic,
            ));
        }
        OracleMode::Hybrid => {
            fallback_oracle = Some(MinRatioOracle::new_with_mode(
                opts.seed,
                fallback_rebuild_every,
                opts.deterministic,
                opts.deterministic_seed,
            ));
            let dynamic_seed = if opts.deterministic {
                opts.deterministic_seed.unwrap_or(0)
            } else {
                opts.seed
            };
            dynamic_oracle = Some(FullDynamicOracle::new(
                dynamic_seed,
                3,
                1,
                10,
                approx_kappa,
                opts.deterministic,
            ));
        }
    }

    let start = Instant::now();
    let mut stats = IpmStats {
        iterations: 0,
        last_step_size: 0.0,
        potentials: Vec::new(),
        last_gap: f64::INFINITY,
        last_duality_gap_proxy: None,
        termination_gap_threshold: None,
        terminated_by_gap: false,
        terminated_by_max_iters: false,
        final_gap_estimate: None,
        oracle_mode: opts.oracle_mode,
        cycle_times_ms: Vec::new(),
        barrier_times_ms: Vec::new(),
        update_times_ms: Vec::new(),
        spanner_update_times_ms: Vec::new(),
        instability_per_level: Vec::new(),
        rebuild_counts: Vec::new(),
        oracle_update_count: 0,
        oracle_fallback_count: 0,
        barrier_clamp_stats: Vec::new(),
    };
    let mut termination = IpmTermination::IterationLimit;
    let mut using_fallback = matches!(opts.oracle_mode, OracleMode::Fallback);
    let mut stall_iters = 0usize;
    let mut last_kappa: Option<f64> = None;
    let mut last_oracle_ratio: Option<f64> = None;
    let dynamic_loss_threshold = (problem.edge_count().max(1) as f64).powf(0.1).ceil() as usize;
    let update_eps = opts.tolerance.max(1e-12);
    let max_u = upper
        .iter()
        .copied()
        .fold(1.0_f64, |acc, value| acc.max(value.abs()).max(1.0));
    let edge_count = problem.edge_count().max(1);
    let mut hidden_stable_flow = HiddenStableFlow::default();
    let mut rebuilding_game = RebuildingGame::new(3);

    for iter in 0..opts.max_iters {
        if let Some(limit) = opts.time_limit_ms {
            if start.elapsed().as_millis() as u64 >= limit {
                termination = IpmTermination::TimeLimit;
                break;
            }
        }

        let barrier_start = Instant::now();
        let (gradient, lengths, residuals, clamp_stats) =
            compute_gradient_and_lengths(&potential, &cost, &flow, &lower, &upper);
        stats
            .barrier_times_ms
            .push(barrier_start.elapsed().as_secs_f64() * 1000.0);
        if opts.log_numerical_clamping && clamp_stats.clamping_occurred() {
            eprintln!(
                "warning: numerical clamping occurred (total_clamps={}) at iter {}",
                clamp_stats.total_clamps(),
                iter
            );
        }
        stats.barrier_clamp_stats.push(clamp_stats.clone());
        let current_potential =
            potential.value_with_residuals_and_stats(&cost, &flow, &residuals, None);
        stats.potentials.push(current_potential);
        stats.last_gap = potential.duality_gap(&cost, &flow, &residuals);
        let mut gap_estimate = termination::estimate_duality_gap_with_config(
            &flow,
            &cost,
            &residuals,
            potential.alpha,
            edge_count,
            max_u,
            &potential.clamp_config,
        );
        if let Some(ratio) = last_oracle_ratio {
            gap_estimate = gap_estimate.max((-ratio).max(0.0));
        }
        stats.last_duality_gap_proxy = Some(gap_estimate);
        stats.termination_gap_threshold =
            Some(termination::compute_gap_threshold(edge_count, max_u, opts));
        if termination::should_terminate(gap_estimate, edge_count, max_u, opts) {
            termination = IpmTermination::Converged;
            stats.iterations = iter;
            stats.terminated_by_gap = true;
            stats.final_gap_estimate = Some(gap_estimate);
            break;
        }
        if matches!(opts.oracle_mode, OracleMode::Hybrid) && !using_fallback {
            if let (Some(prev), Some(kappa)) =
                (stats.potentials.iter().rev().nth(1).copied(), last_kappa)
            {
                let reduction = prev - current_potential;
                if reduction < kappa * kappa {
                    stall_iters = stall_iters.saturating_add(1);
                } else {
                    stall_iters = 0;
                }
                if stall_iters >= 10 {
                    eprintln!(
                        "warning: dynamic oracle stalled (reduction {:.3e}); monitoring stability",
                        reduction
                    );
                    stall_iters = 0;
                }
            }
        }
        if current_potential <= termination_target {
            termination = IpmTermination::Converged;
            stats.iterations = iter;
            if iter + 1 >= opts.max_iters {
                termination = IpmTermination::IterationLimit;
                stats.terminated_by_max_iters = true;
            }
            break;
        }
        if stats.last_gap < opts.tolerance {
            termination = IpmTermination::Converged;
            stats.iterations = iter;
            if iter + 1 >= opts.max_iters {
                termination = IpmTermination::IterationLimit;
                stats.terminated_by_max_iters = true;
            }
            break;
        }

        let w_t = 100.0
            * (1.0
                + gradient
                    .iter()
                    .zip(lengths.iter())
                    .map(|(g, l)| (g * l).abs())
                    .sum::<f64>());
        hidden_stable_flow.record_width(w_t);
        let cycle_start = Instant::now();
        let best = if !using_fallback {
            if let Some(oracle) = dynamic_oracle.as_mut() {
                let width_threshold = (edge_count as f64).powf(0.1) * w_t.max(1.0);
                if hidden_stable_flow.width_sum() > width_threshold {
                    rebuilding_game.record(0, Outcome::Loss);
                    if rebuilding_game.force_rebuild_level().is_some()
                        && matches!(opts.oracle_mode, OracleMode::Hybrid)
                    {
                        using_fallback = true;
                        stats.oracle_fallback_count = stats.oracle_fallback_count.saturating_add(1);
                    }
                } else {
                    rebuilding_game.record(0, Outcome::Win);
                }
                let best = oracle
                    .best_cycle(
                        iter,
                        problem.node_count,
                        &problem.tails,
                        &problem.heads,
                        &gradient,
                        &lengths,
                    )
                    .map_err(|err| McfError::InvalidInput(format!("{err:?}")))?;
                if matches!(opts.oracle_mode, OracleMode::Hybrid)
                    && oracle.loss_count() > dynamic_loss_threshold
                {
                    using_fallback = true;
                    eprintln!(
                        "warning: dynamic oracle exceeded loss threshold ({}); falling back to SSP",
                        oracle.loss_count()
                    );
                }
                best
            } else {
                None
            }
        } else if let Some(oracle) = fallback_oracle.as_mut() {
            oracle
                .best_cycle(OracleQuery {
                    iter,
                    node_count: problem.node_count,
                    tails: &problem.tails,
                    heads: &problem.heads,
                    gradients: &gradient,
                    lengths: &lengths,
                })
                .map_err(|err| McfError::InvalidInput(format!("{err:?}")))?
        } else {
            None
        };
        stats
            .cycle_times_ms
            .push(cycle_start.elapsed().as_secs_f64() * 1000.0);
        let Some(best) = best else {
            if iter + 1 >= opts.max_iters {
                termination = IpmTermination::IterationLimit;
                stats.terminated_by_max_iters = true;
            } else {
                termination = IpmTermination::NoImprovingCycle;
            }
            stats.iterations = iter;
            break;
        };
        last_oracle_ratio = Some(best.ratio);

        if best.ratio >= -opts.tolerance {
            termination = IpmTermination::Converged;
            stats.iterations = iter;
            if iter + 1 >= opts.max_iters {
                termination = IpmTermination::IterationLimit;
                stats.terminated_by_max_iters = true;
            }
            break;
        }

        let mut delta = vec![0.0_f64; flow.len()];
        for (edge_id, dir) in best.cycle_edges {
            delta[edge_id] += dir as f64;
        }

        let delta_norm = delta
            .iter()
            .zip(lengths.iter())
            .map(|(d, l)| l * d * d)
            .sum::<f64>()
            .sqrt();
        if delta_norm > 0.0 {
            for d in &mut delta {
                *d /= delta_norm;
            }
        }

        let kappa = (-best.ratio).max(0.0);
        last_kappa = Some(kappa);
        if kappa < potential.kappa_floor() {
            termination = IpmTermination::Converged;
            stats.iterations = iter;
            if iter + 1 >= opts.max_iters {
                termination = IpmTermination::IterationLimit;
                stats.terminated_by_max_iters = true;
            }
            break;
        }

        // Theorem 4.2: each accepted step should reduce the potential by
        // Omega(kappa^2), with kappa >= m^{-o(1)}.
        let required_reduction = potential
            .reduction_floor(current_potential)
            .max(0.01 * kappa * kappa);
        if let Some((candidate_flow, step)) = search::line_search(search::LineSearchInput {
            flow: &flow,
            delta: &delta,
            cost: &cost,
            lower: &lower,
            upper: &upper,
            potential: &potential,
            current_potential,
            required_reduction,
            gradient: &gradient,
            lengths: &lengths,
        }) {
            flow = candidate_flow;
            stats.last_step_size = step;
            if let Some(oracle) = dynamic_oracle.as_mut() {
                let flow_updates: Vec<(usize, f64)> = delta
                    .iter()
                    .enumerate()
                    .filter_map(|(edge_id, &dir)| {
                        let change = dir * step;
                        if change.abs() > update_eps {
                            Some((edge_id, change))
                        } else {
                            None
                        }
                    })
                    .collect();
                if !flow_updates.is_empty() {
                    let flow_delta = SparseFlowDelta::new(flow_updates);
                    if let Err(err) = oracle.notify_flow_change(&flow_delta) {
                        if matches!(opts.oracle_mode, OracleMode::Hybrid) {
                            using_fallback = true;
                            stats.oracle_fallback_count =
                                stats.oracle_fallback_count.saturating_add(1);
                            eprintln!("warning: dynamic oracle requested rebuild ({err})");
                        }
                    }
                }

                let barrier_start = Instant::now();
                let (new_gradient, new_lengths, _, clamp_stats) =
                    compute_gradient_and_lengths(&potential, &cost, &flow, &lower, &upper);
                stats
                    .barrier_times_ms
                    .push(barrier_start.elapsed().as_secs_f64() * 1000.0);
                stats.barrier_clamp_stats.push(clamp_stats);
                let mut g_updates = Vec::new();
                let mut ell_updates = Vec::new();
                for (edge_id, (&g_new, &l_new)) in
                    new_gradient.iter().zip(new_lengths.iter()).enumerate()
                {
                    if (g_new - gradient[edge_id]).abs() > update_eps {
                        g_updates.push((edge_id, g_new));
                    }
                    if (l_new - lengths[edge_id]).abs() > update_eps {
                        ell_updates.push((edge_id, l_new));
                    }
                }
                if !g_updates.is_empty() || !ell_updates.is_empty() {
                    let update_start = Instant::now();
                    let update_result = oracle.update_many(&g_updates, &ell_updates);
                    let elapsed = update_start.elapsed().as_secs_f64() * 1000.0;
                    stats.update_times_ms.push(elapsed);
                    stats.spanner_update_times_ms.push(elapsed);
                    stats.oracle_update_count = stats.oracle_update_count.saturating_add(1);
                    stats.instability_per_level = oracle.instability_per_level();
                    stats.rebuild_counts = oracle.rebuild_counts();
                    if let Err(err) = update_result {
                        if matches!(opts.oracle_mode, OracleMode::Hybrid) {
                            using_fallback = true;
                            stats.oracle_fallback_count =
                                stats.oracle_fallback_count.saturating_add(1);
                            eprintln!("warning: dynamic oracle requested rebuild ({err})");
                        }
                    }
                }
            }
        } else {
            termination = if iter + 1 >= opts.max_iters {
                IpmTermination::IterationLimit
            } else {
                IpmTermination::NoImprovingCycle
            };
            if iter + 1 >= opts.max_iters {
                stats.terminated_by_max_iters = true;
            }
            stats.iterations = iter;
            break;
        }

        stats.iterations = iter + 1;
        if iter + 1 >= opts.max_iters {
            termination = IpmTermination::IterationLimit;
            stats.terminated_by_max_iters = true;
            stats.final_gap_estimate = Some(gap_estimate);
            break;
        }
    }

    let (_final_gradient, _, final_residuals, clamp_stats) =
        compute_gradient_and_lengths(&potential, &cost, &flow, &lower, &upper);
    stats.barrier_clamp_stats.push(clamp_stats);
    stats.last_gap = potential.duality_gap(&cost, &flow, &final_residuals);
    if stats.final_gap_estimate.is_none() {
        stats.final_gap_estimate = stats.last_duality_gap_proxy;
    }

    Ok(IpmResult {
        flow,
        stats,
        termination,
    })
}

fn compute_gradient_and_lengths(
    potential: &Potential,
    cost: &[f64],
    flow: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> (Vec<f64>, Vec<f64>, Residuals, BarrierClampStats) {
    let mut clamp_stats = BarrierClampStats::default();
    let residuals = Residuals::new_with_stats(
        flow,
        lower,
        upper,
        potential.min_delta,
        Some(&mut clamp_stats),
    );
    let gradient =
        potential.gradient_with_residuals_and_stats(cost, flow, &residuals, Some(&mut clamp_stats));
    let lengths = potential.lengths_from_gradient_with_stats(&gradient, Some(&mut clamp_stats));
    (gradient, lengths, residuals, clamp_stats)
}

fn flow_cost(flow: &[f64], cost: &[i64]) -> f64 {
    flow.iter()
        .zip(cost.iter())
        .map(|(f, c)| *f * (*c as f64))
        .sum()
}

fn cost_lower_bound(problem: &McfProblem) -> i128 {
    problem
        .cost
        .iter()
        .zip(problem.lower.iter().zip(problem.upper.iter()))
        .map(|(&c, (&lo, &up))| {
            if c >= 0 {
                c as i128 * lo as i128
            } else {
                c as i128 * up as i128
            }
        })
        .sum()
}

#[allow(dead_code)]
fn binary_search_optimal_cost(
    problem: &McfProblem,
    opts: &McfOptions,
    low: f64,
    high: f64,
    epsilon: f64,
) -> Result<f64, McfError> {
    let (best_cost, _) = binary_search_with_result(problem, opts, low, high, epsilon)?;
    Ok(best_cost)
}

fn binary_search_with_result(
    problem: &McfProblem,
    opts: &McfOptions,
    mut low: f64,
    mut high: f64,
    epsilon: f64,
) -> Result<(f64, Option<IpmResult>), McfError> {
    let mut best: Option<IpmResult> = None;
    let mut best_cost = high;
    let mut iter = 0;
    while high - low > epsilon && iter < 64 {
        let mid = 0.5 * (low + high);
        let result = run_ipm_with_lower_bound(problem, opts, mid)?;
        let final_cost = flow_cost(&result.flow, &problem.cost);
        let feasible = result.termination == IpmTermination::Converged
            && (final_cost <= mid + opts.tolerance || result.stats.last_gap < opts.tolerance);

        if feasible {
            best = Some(result);
            best_cost = mid;
            high = mid;
        } else {
            low = mid;
        }
        iter += 1;
    }

    Ok((best_cost, best))
}

struct CostBounds {
    low: i128,
    high: i128,
}

impl CostBounds {
    fn new(problem: &McfProblem) -> Result<Self, McfError> {
        let feasible = initialize_feasible_flow(problem)?;
        let upper_bound = flow_cost(&feasible.flow, &problem.cost).ceil() as i128;
        let lower_bound = cost_lower_bound(problem);
        let (low, high) = if lower_bound <= upper_bound {
            (lower_bound, upper_bound)
        } else {
            (upper_bound, lower_bound)
        };
        Ok(Self { low, high })
    }
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        let value = self.next_u64() >> 11;
        (value as f64) / ((1_u64 << 53) as f64)
    }
}

fn initialize_feasible_flow_with_options(
    problem: &McfProblem,
    opts: &McfOptions,
) -> Result<FeasibleFlow, McfError> {
    if let Some(flow) = opts.initial_flow.as_deref() {
        validate_base_flow(problem, flow)?;
        let flow = push_inside_strict(problem, flow)?;
        return Ok(FeasibleFlow { flow });
    }

    if opts.initial_perturbation <= 0.0 {
        return initialize_feasible_flow(problem);
    }

    let mut rng = Lcg::new(opts.seed ^ 0x9E3779B97F4A7C15);
    let scale = (opts.initial_perturbation.abs() * 1000.0).max(1.0) as i64;
    let edge_costs: Vec<i64> = (0..problem.edge_count())
        .map(|_| {
            let value = (rng.next_f64() * 2.0 - 1.0) * scale as f64;
            value.round() as i64
        })
        .collect();

    let base_flow = solve_feasible_flow_with_costs(
        problem.node_count,
        &problem.tails,
        &problem.heads,
        &problem.lower,
        &problem.upper,
        &problem.demands,
        Some(&edge_costs),
    )?;
    let flow = push_inside_strict(problem, &base_flow)?;
    Ok(FeasibleFlow { flow })
}

fn solve_feasible_flow_with_costs(
    node_count: usize,
    tails: &[u32],
    heads: &[u32],
    lower: &[i64],
    upper: &[i64],
    demands: &[i64],
    edge_costs: Option<&[i64]>,
) -> Result<Vec<i64>, McfError> {
    let n = node_count;
    let m = lower.len();
    if tails.len() != m || heads.len() != m || upper.len() != m {
        return Err(McfError::InvalidInput(
            "edge arrays must have identical length".to_string(),
        ));
    }
    if demands.len() != n {
        return Err(McfError::InvalidInput(
            "demand array length mismatch".to_string(),
        ));
    }

    let mut demand = demands.to_vec();
    let mut residual_upper = vec![0_i64; m];
    for (i, residual) in residual_upper.iter_mut().enumerate() {
        let lo = lower[i];
        let up = upper[i];
        if lo > up {
            return Err(McfError::InvalidInput(
                "lower bound exceeds upper bound".to_string(),
            ));
        }
        *residual = up
            .checked_sub(lo)
            .ok_or_else(|| McfError::InvalidInput("capacity overflow".to_string()))?;
        let tail = tails[i] as usize;
        let head = heads[i] as usize;
        demand[tail] = demand[tail]
            .checked_add(lo)
            .ok_or_else(|| McfError::InvalidInput("demand overflow".to_string()))?;
        demand[head] = demand[head]
            .checked_sub(lo)
            .ok_or_else(|| McfError::InvalidInput("demand overflow".to_string()))?;
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
        let tail = tails[i] as usize;
        let head = heads[i] as usize;
        let cost = edge_costs
            .and_then(|values| values.get(i))
            .copied()
            .unwrap_or(0);
        let (idx, _rev) = mcf.add_edge(tail, head, cap, cost);
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
        flow[i] = lower[i]
            .checked_add(used)
            .ok_or_else(|| McfError::InvalidInput("flow overflow".to_string()))?;
    }

    Ok(flow)
}

fn push_inside_strict(problem: &McfProblem, base_flow: &[i64]) -> Result<Vec<f64>, McfError> {
    const SCALE: i64 = 2;
    const EPS_SCALED: i64 = 1;
    let m = problem.edge_count();
    let n = problem.node_count;
    if base_flow.len() != m {
        return Err(McfError::InvalidInput(
            "base flow length mismatch".to_string(),
        ));
    }

    let mut lower_delta = vec![0_i64; m];
    let mut upper_delta = vec![0_i64; m];
    for i in 0..m {
        let lo_scaled = scale_checked(problem.lower[i], SCALE)?
            .checked_add(EPS_SCALED)
            .ok_or_else(|| McfError::InvalidInput("scaled lower overflow".to_string()))?;
        let up_scaled = scale_checked(problem.upper[i], SCALE)?
            .checked_sub(EPS_SCALED)
            .ok_or_else(|| McfError::InvalidInput("scaled upper overflow".to_string()))?;
        if lo_scaled > up_scaled {
            return Err(McfError::Infeasible);
        }
        let base_scaled = scale_checked(base_flow[i], SCALE)?;
        lower_delta[i] = lo_scaled
            .checked_sub(base_scaled)
            .ok_or_else(|| McfError::InvalidInput("delta lower overflow".to_string()))?;
        upper_delta[i] = up_scaled
            .checked_sub(base_scaled)
            .ok_or_else(|| McfError::InvalidInput("delta upper overflow".to_string()))?;
    }

    let zero_demands = vec![0_i64; n];
    let delta = solve_feasible_flow_with_costs(
        n,
        &problem.tails,
        &problem.heads,
        &lower_delta,
        &upper_delta,
        &zero_demands,
        None,
    )?;
    let mut flow = vec![0_f64; m];
    for i in 0..m {
        let total_scaled = scale_checked(base_flow[i], SCALE)?
            .checked_add(delta[i])
            .ok_or_else(|| McfError::InvalidInput("flow overflow".to_string()))?;
        flow[i] = total_scaled as f64 / SCALE as f64;
    }
    Ok(flow)
}

fn scale_checked(value: i64, scale: i64) -> Result<i64, McfError> {
    value
        .checked_mul(scale)
        .ok_or_else(|| McfError::InvalidInput("scaled value overflow".to_string()))
}

fn validate_base_flow(problem: &McfProblem, flow: &[i64]) -> Result<(), McfError> {
    if flow.len() != problem.edge_count() {
        return Err(McfError::InvalidInput(
            "initial flow length mismatch".to_string(),
        ));
    }
    let mut balance = vec![0_i64; problem.node_count];
    for (idx, &value) in flow.iter().enumerate() {
        let lower = problem.lower[idx];
        let upper = problem.upper[idx];
        if value < lower || value > upper {
            return Err(McfError::InvalidInput(
                "initial flow violates bounds".to_string(),
            ));
        }
        let tail = problem.tails[idx] as usize;
        let head = problem.heads[idx] as usize;
        balance[tail] = balance[tail]
            .checked_sub(value)
            .ok_or_else(|| McfError::InvalidInput("balance overflow".to_string()))?;
        balance[head] = balance[head]
            .checked_add(value)
            .ok_or_else(|| McfError::InvalidInput("balance overflow".to_string()))?;
    }
    for (node, &demand) in problem.demands.iter().enumerate() {
        if balance[node] != demand {
            return Err(McfError::InvalidInput(
                "initial flow violates demands".to_string(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_feasible(problem: &McfProblem, flow: &[f64]) {
        assert_eq!(flow.len(), problem.edge_count());
        let mut balance = vec![0_f64; problem.node_count];
        for (idx, (&f, (&tail, &head))) in flow
            .iter()
            .zip(problem.tails.iter().zip(problem.heads.iter()))
            .enumerate()
        {
            let lo = problem.lower[idx] as f64;
            let up = problem.upper[idx] as f64;
            assert!(
                f >= lo - 1e-9 && f <= up + 1e-9,
                "flow out of bounds at edge {idx}: {f} not in [{lo}, {up}]"
            );
            balance[tail as usize] -= f;
            balance[head as usize] += f;
        }
        for (node, (&b, &d)) in balance.iter().zip(problem.demands.iter()).enumerate() {
            assert!(
                (b - d as f64).abs() <= 1e-9,
                "node {node} imbalance mismatch: got {b}, expected {d}"
            );
        }
    }

    #[test]
    fn initializes_feasible_flow_with_lower_bounds() {
        let problem = McfProblem::new(
            vec![0, 1],
            vec![1, 2],
            vec![1, 1],
            vec![3, 3],
            vec![1, 1],
            vec![-2, 0, 2],
        )
        .unwrap();
        let feasible = initialize_feasible_flow(&problem).unwrap();
        assert_feasible(&problem, &feasible.flow);
    }

    #[test]
    fn initializer_rejects_infeasible_instance() {
        let problem =
            McfProblem::new(vec![0], vec![1], vec![0], vec![1], vec![1], vec![-2, 2]).unwrap();
        let err = initialize_feasible_flow(&problem).unwrap_err();
        assert!(matches!(err, McfError::Infeasible));
    }

    #[test]
    fn initializer_rejects_missing_strict_interior() {
        let problem =
            McfProblem::new(vec![0], vec![1], vec![0], vec![0], vec![1], vec![-1, 1]).unwrap();
        let err = initialize_feasible_flow(&problem).unwrap_err();
        assert!(matches!(err, McfError::Infeasible));
    }

    #[test]
    fn initializer_produces_strictly_interior_flows_on_random_graphs() {
        let mut rng = Lcg::new(7);
        for _ in 0..5 {
            let n = 4 + rng.next_range(3) as usize;
            let m = 8 + rng.next_range(5) as usize;
            let mut tails = Vec::with_capacity(m);
            let mut heads = Vec::with_capacity(m);
            let mut lower = Vec::with_capacity(m);
            let mut upper = Vec::with_capacity(m);
            let mut cost = Vec::with_capacity(m);
            let mut base_flow = Vec::with_capacity(m);

            for _ in 0..m {
                let tail = rng.next_range(n as u32);
                let mut head = rng.next_range(n as u32);
                if head == tail {
                    head = (head + 1) % n as u32;
                }
                tails.push(tail);
                heads.push(head);
                let lo = rng.next_range(3) as i64;
                let slack = 2 + rng.next_range(3) as i64;
                let up = lo + slack;
                lower.push(lo);
                upper.push(up);
                cost.push(rng.next_range(5) as i64);
                base_flow.push(lo + 1);
            }

            let mut demands = vec![0_i64; n];
            for (idx, &f) in base_flow.iter().enumerate() {
                demands[tails[idx] as usize] -= f;
                demands[heads[idx] as usize] += f;
            }

            let problem = McfProblem::new(tails, heads, lower, upper, cost, demands).unwrap();
            let feasible = initialize_feasible_flow(&problem).unwrap();
            assert_feasible(&problem, &feasible.flow);
            for (idx, &f) in feasible.flow.iter().enumerate() {
                let lo = problem.lower[idx] as f64;
                let up = problem.upper[idx] as f64;
                assert!(
                    f > lo + 1e-9 && f < up - 1e-9,
                    "edge {idx} is not strictly interior: {f} not in ({lo}, {up})"
                );
            }
        }
    }

    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u32(&mut self) -> u32 {
            self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
            (self.state >> 32) as u32
        }

        fn next_range(&mut self, max: u32) -> u32 {
            if max == 0 {
                0
            } else {
                self.next_u32() % max
            }
        }
    }

    fn cycle_problem(cost: Vec<i64>) -> McfProblem {
        McfProblem::new(
            vec![0, 1, 2],
            vec![1, 2, 0],
            vec![0, 0, 0],
            vec![2, 2, 2],
            cost,
            vec![0, 0, 0],
        )
        .unwrap()
    }

    #[test]
    fn ipm_decreases_potential_monotonically() {
        let problem = cycle_problem(vec![-2, -1, -3]);
        let opts = McfOptions {
            tolerance: 1e-6,
            max_iters: 4,
            ..McfOptions::default()
        };
        let result = run_ipm(&problem, &opts).unwrap();
        let potentials = result.stats.potentials;
        assert!(potentials.len() >= 2);
        for window in potentials.windows(2) {
            assert!(
                window[1] <= window[0] + 1e-9,
                "potential increased from {} to {}",
                window[0],
                window[1]
            );
        }
        assert!(result.stats.last_step_size > 0.0);
    }

    #[test]
    fn ipm_stops_on_iteration_limit() {
        let problem = cycle_problem(vec![-2, -1, -3]);
        let opts = McfOptions {
            tolerance: 1e-12,
            max_iters: 1,
            ..McfOptions::default()
        };
        let result = run_ipm(&problem, &opts).unwrap();
        assert!(
            matches!(
                result.termination,
                IpmTermination::IterationLimit | IpmTermination::Converged
            ),
            "unexpected termination: {:?}",
            result.termination
        );
    }

    #[test]
    fn ipm_respects_time_limit() {
        let problem = cycle_problem(vec![-2, -1, -3]);
        let opts = McfOptions {
            time_limit_ms: Some(0),
            ..McfOptions::default()
        };
        let result = run_ipm(&problem, &opts).unwrap();
        assert_eq!(result.termination, IpmTermination::TimeLimit);
    }

    #[test]
    fn ipm_converges_on_simple_network() {
        let problem = cycle_problem(vec![2, 1, 3]);
        let opts = McfOptions {
            tolerance: 1e-6,
            max_iters: 20,
            ..McfOptions::default()
        };
        let result = run_ipm(&problem, &opts).unwrap();
        assert_eq!(result.termination, IpmTermination::Converged);
    }

    #[test]
    fn line_search_achieves_required_potential_drop() {
        let problem = cycle_problem(vec![-3, -2, -4]);
        let feasible = initialize_feasible_flow(&problem).unwrap();
        let flow = feasible.flow;
        let lower: Vec<f64> = problem.lower.iter().map(|&v| v as f64).collect();
        let upper: Vec<f64> = problem.upper.iter().map(|&v| v as f64).collect();
        let cost: Vec<f64> = problem.cost.iter().map(|&v| v as f64).collect();
        let potential = Potential::new(
            &upper,
            None,
            1,
            McfOptions::default().barrier_clamp_config(),
        );

        let (gradient, lengths, _residuals, _clamp_stats) =
            compute_gradient_and_lengths(&potential, &cost, &flow, &lower, &upper);
        let mut oracle = MinRatioOracle::new(3, 1);
        let best = oracle
            .best_cycle(OracleQuery {
                iter: 0,
                node_count: problem.node_count,
                tails: &problem.tails,
                heads: &problem.heads,
                gradients: &gradient,
                lengths: &lengths,
            })
            .unwrap()
            .unwrap();

        let mut delta = vec![0.0_f64; flow.len()];
        for (edge_id, dir) in best.cycle_edges {
            delta[edge_id] += dir as f64;
        }

        let current_potential = potential.value(&cost, &flow, &lower, &upper);
        let kappa = (-best.ratio).max(0.0);
        let required_reduction = potential
            .reduction_floor(current_potential)
            .max(0.01 * kappa * kappa);
        let (candidate_flow, _step) = search::line_search(search::LineSearchInput {
            flow: &flow,
            delta: &delta,
            cost: &cost,
            lower: &lower,
            upper: &upper,
            potential: &potential,
            current_potential,
            required_reduction,
            gradient: &gradient,
            lengths: &lengths,
        })
        .expect("line search should find improvement");
        let candidate_potential = potential.value(&cost, &candidate_flow, &lower, &upper);
        let min_required = required_reduction.min(potential.reduction_floor(current_potential));
        assert!(candidate_potential <= current_potential - min_required);
        for (idx, &value) in candidate_flow.iter().enumerate() {
            assert!(value > lower[idx] && value < upper[idx]);
        }
    }

    #[test]
    fn binary_search_respects_cost_bounds() {
        let problem = McfProblem::new(
            vec![0, 0, 1],
            vec![1, 2, 2],
            vec![0, 0, 0],
            vec![5, 5, 5],
            vec![2, 1, 3],
            vec![-3, 0, 3],
        )
        .unwrap();
        let bounds = CostBounds::new(&problem).unwrap();
        let result =
            run_ipm_with_lower_bound(&problem, &McfOptions::default(), bounds.low as f64).unwrap();
        let final_cost = flow_cost(&result.flow, &problem.cost);
        assert!(final_cost <= bounds.high as f64);
        assert!(final_cost >= bounds.low as f64 - 1e-6);
    }
}
