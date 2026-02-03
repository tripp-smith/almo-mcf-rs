use crate::ipm::{self, IpmResult, IpmRunContext, IpmRunKind, IpmTermination};
use crate::{
    finalize_ipm_solution, IpmSummary, McfError, McfOptions, McfProblem, McfSolution, SolverMode,
};
use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct CostScalingRound {
    pub round: usize,
    pub midpoint: BigInt,
    pub solved_cost: BigInt,
    pub iterations: usize,
    pub termination: IpmTermination,
}

#[derive(Debug, Clone)]
pub struct CapacityScalingPhase {
    pub phase: usize,
    pub divisor: i64,
    pub iterations: usize,
    pub termination: IpmTermination,
}

#[derive(Debug, Default, Clone)]
pub struct ScalingStats {
    pub cost_rounds: Vec<CostScalingRound>,
    pub capacity_phases: Vec<CapacityScalingPhase>,
}

#[derive(Debug, Clone)]
pub struct CapacityScalingResult {
    pub scaled_problem: McfProblem,
    pub scale_factor: i64,
    pub initial_flow: Option<Vec<i64>>,
}

#[derive(Debug, Clone)]
pub struct CostScalingInfo {
    pub divisor: i64,
    pub bound: i64,
    pub max_scaled_cost: i64,
}

#[derive(Debug, Clone)]
pub struct CapacityScalingInfo {
    pub divisor: i64,
    pub bound: i64,
    pub max_scaled_capacity: i64,
}

pub fn solve_mcf_with_scaling(
    problem: &McfProblem,
    opts: &McfOptions,
) -> Result<McfSolution, McfError> {
    let mut stats = ScalingStats::default();
    let capacity = capacity_scaling(problem, opts, &mut stats)?;
    let mut round_opts = opts.clone();
    round_opts.initial_flow = capacity.initial_flow.clone();
    let ipm_result = cost_scaling(&capacity.scaled_problem, &round_opts, &mut stats)?;
    let ipm_stats = Some(IpmSummary::from_ipm(
        &ipm_result.stats,
        ipm_result.termination,
    ));
    let mut solution = finalize_ipm_solution(&capacity.scaled_problem, ipm_result, ipm_stats)?;
    solution.solver_mode = SolverMode::IpmScaled;

    if capacity.scale_factor != 1 {
        for flow in &mut solution.flow {
            *flow = flow
                .checked_mul(capacity.scale_factor)
                .ok_or_else(|| McfError::InvalidInput("flow scaling overflow".to_string()))?;
        }
        solution.cost = solution
            .cost
            .checked_mul(capacity.scale_factor as i128)
            .ok_or_else(|| McfError::InvalidInput("cost scaling overflow".to_string()))?;
    }

    Ok(solution)
}

pub fn cost_scaling(
    problem: &McfProblem,
    opts: &McfOptions,
    stats: &mut ScalingStats,
) -> Result<IpmResult, McfError> {
    let (scaled_problem, info) = build_cost_scaled_problem(problem)?;
    let (mut low, mut high) = cost_bounds(problem)?;
    let max_rounds = (bigint_bits(&high) + 1).max(1) as usize;

    let mut best: Option<IpmResult> = None;
    let mut last_flow = feasible_initial_flow(problem, opts.initial_flow.clone());

    for round in 0..max_rounds {
        if low > high {
            break;
        }
        let midpoint = (&low + &high) / BigInt::from(2);
        let mut round_opts = opts.clone();
        round_opts.initial_flow = feasible_initial_flow(&scaled_problem, last_flow.clone());

        let start = Instant::now();
        let ipm_result = ipm::run_ipm_with_context(
            &scaled_problem,
            &round_opts,
            IpmRunContext {
                kind: IpmRunKind::CostScaling,
                round,
            },
        )?;
        let elapsed_ms = start.elapsed().as_millis();

        let solved_cost = flow_cost_bigint(&ipm_result.flow, &problem.cost);
        eprintln!(
            "[cost_scaling] round={round} midpoint={midpoint} cost={solved_cost} scaled_max={} elapsed_ms={elapsed_ms}",
            info.max_scaled_cost
        );

        stats.cost_rounds.push(CostScalingRound {
            round,
            midpoint: midpoint.clone(),
            solved_cost: solved_cost.clone(),
            iterations: ipm_result.stats.iterations,
            termination: ipm_result.termination,
        });

        last_flow = feasible_initial_flow(&scaled_problem, Some(flow_to_i64(&ipm_result.flow)));

        if solved_cost <= midpoint {
            high = midpoint - BigInt::one();
            best = Some(ipm_result);
        } else {
            low = midpoint + BigInt::one();
        }
    }

    if let Some(best) = best {
        return Ok(best);
    }

    ipm::run_ipm_with_context(
        &scaled_problem,
        opts,
        IpmRunContext {
            kind: IpmRunKind::CostScaling,
            round: 0,
        },
    )
}

pub fn capacity_scaling(
    problem: &McfProblem,
    opts: &McfOptions,
    stats: &mut ScalingStats,
) -> Result<CapacityScalingResult, McfError> {
    let (scale_factor, bound) = capacity_scale_factor(problem);
    let mut divisor = max_power_of_two_divisor(&problem.lower, &problem.upper, &problem.demands);
    if divisor < scale_factor {
        divisor = scale_factor;
    }

    let mut scaled_problem = scale_problem(problem, divisor)?;
    let mut last_flow = feasible_initial_flow(&scaled_problem, opts.initial_flow.clone());
    let mut phase = 0;

    while divisor >= scale_factor {
        let mut round_opts = opts.clone();
        round_opts.initial_flow = feasible_initial_flow(&scaled_problem, last_flow.clone());
        let ipm_result = ipm::run_ipm_with_context(
            &scaled_problem,
            &round_opts,
            IpmRunContext {
                kind: IpmRunKind::CapacityScaling,
                round: phase,
            },
        )?;

        eprintln!(
            "[capacity_scaling] phase={phase} divisor={divisor} iterations={} bound={bound}",
            ipm_result.stats.iterations
        );

        stats.capacity_phases.push(CapacityScalingPhase {
            phase,
            divisor,
            iterations: ipm_result.stats.iterations,
            termination: ipm_result.termination,
        });

        if divisor == scale_factor {
            last_flow = feasible_initial_flow(&scaled_problem, Some(flow_to_i64(&ipm_result.flow)));
            break;
        }

        let next_divisor = divisor / 2;
        let scaled_flow = scale_flow(&ipm_result.flow, divisor / next_divisor);
        scaled_problem = scale_problem(problem, next_divisor)?;
        last_flow = feasible_initial_flow(&scaled_problem, Some(scaled_flow));
        divisor = next_divisor;
        phase += 1;
    }

    Ok(CapacityScalingResult {
        scaled_problem,
        scale_factor: divisor,
        initial_flow: last_flow,
    })
}

pub fn build_cost_scaled_problem(
    problem: &McfProblem,
) -> Result<(McfProblem, CostScalingInfo), McfError> {
    let m = problem.edge_count().max(1) as i64;
    let bound = m.saturating_pow(3).max(1);
    let max_cost = max_abs(&problem.cost);
    let divisor = cost_divisor(max_cost, bound);
    let scaled_cost = problem
        .cost
        .iter()
        .map(|&value| value / divisor)
        .collect::<Vec<_>>();

    let max_scaled_cost = max_abs(&scaled_cost);
    let scaled_problem = McfProblem::new(
        problem.tails.clone(),
        problem.heads.clone(),
        problem.lower.clone(),
        problem.upper.clone(),
        scaled_cost,
        problem.demands.clone(),
    )?;

    Ok((
        scaled_problem,
        CostScalingInfo {
            divisor,
            bound,
            max_scaled_cost,
        },
    ))
}

pub fn build_capacity_scaled_problem(
    problem: &McfProblem,
) -> Result<(McfProblem, CapacityScalingInfo), McfError> {
    let (scale_factor, bound) = capacity_scale_factor(problem);
    let scaled_problem = scale_problem(problem, scale_factor)?;
    let max_scaled_capacity = max_abs(&scaled_problem.upper);

    Ok((
        scaled_problem,
        CapacityScalingInfo {
            divisor: scale_factor,
            bound,
            max_scaled_capacity,
        },
    ))
}

fn cost_bounds(problem: &McfProblem) -> Result<(BigInt, BigInt), McfError> {
    let lower = cost_lower_bound(problem);
    let upper = cost_upper_bound(problem);
    let (low, high) = if lower <= upper {
        (lower, upper)
    } else {
        (upper, lower)
    };
    Ok((low, high))
}

fn cost_lower_bound(problem: &McfProblem) -> BigInt {
    problem
        .cost
        .iter()
        .zip(problem.lower.iter().zip(problem.upper.iter()))
        .fold(BigInt::zero(), |acc, (&c, (&lo, &up))| {
            let value = if c >= 0 { lo } else { up };
            acc + BigInt::from(c) * BigInt::from(value)
        })
}

fn cost_upper_bound(problem: &McfProblem) -> BigInt {
    let max_capacity = max_abs(&problem.upper).max(1);
    problem.cost.iter().fold(BigInt::zero(), |acc, &c| {
        acc + BigInt::from(c.abs()) * BigInt::from(max_capacity)
    })
}

fn flow_cost_bigint(flow: &[f64], cost: &[i64]) -> BigInt {
    flow.iter()
        .zip(cost.iter())
        .fold(BigInt::zero(), |acc, (f, &c)| {
            let value = f.round() as i128;
            acc + BigInt::from(value) * BigInt::from(c)
        })
}

fn flow_to_i64(flow: &[f64]) -> Vec<i64> {
    flow.iter().map(|value| value.round() as i64).collect()
}

fn scale_flow(flow: &[f64], factor: i64) -> Vec<i64> {
    flow.iter()
        .map(|value| {
            (value.round() as i128)
                .checked_mul(factor as i128)
                .map(|value| value as i64)
                .unwrap_or_else(|| {
                    if value.is_sign_negative() {
                        i64::MIN
                    } else {
                        i64::MAX
                    }
                })
        })
        .collect()
}

fn scale_problem(problem: &McfProblem, divisor: i64) -> Result<McfProblem, McfError> {
    if divisor <= 0 {
        return Err(McfError::InvalidInput(
            "capacity scaling divisor must be positive".to_string(),
        ));
    }

    let lower = scale_values(&problem.lower, divisor)?;
    let upper = scale_values(&problem.upper, divisor)?;
    let demands = scale_values(&problem.demands, divisor)?;

    McfProblem::new(
        problem.tails.clone(),
        problem.heads.clone(),
        lower,
        upper,
        problem.cost.clone(),
        demands,
    )
}

fn scale_values(values: &[i64], divisor: i64) -> Result<Vec<i64>, McfError> {
    values
        .iter()
        .map(|&value| {
            if value % divisor != 0 {
                return Err(McfError::InvalidInput(
                    "capacity scaling requires divisible values".to_string(),
                ));
            }
            Ok(value / divisor)
        })
        .collect()
}

fn cost_divisor(max_cost: i64, bound: i64) -> i64 {
    if max_cost <= bound {
        1
    } else {
        let needed = (max_cost + bound - 1) / bound;
        needed.max(1)
    }
}

fn capacity_scale_factor(problem: &McfProblem) -> (i64, i64) {
    let m = problem.edge_count().max(1) as i64;
    let bound = m.saturating_pow(3).max(1);
    let max_capacity = max_abs(&problem.upper).max(1);
    let needed = (max_capacity + bound - 1) / bound;
    let divisor = next_power_of_two(needed).max(1);
    let common = max_power_of_two_divisor(&problem.lower, &problem.upper, &problem.demands);
    let divisor = divisor.min(common).max(1);
    (divisor, bound)
}

fn max_power_of_two_divisor(lower: &[i64], upper: &[i64], demands: &[i64]) -> i64 {
    let mut min_trailing = u32::MAX;
    for value in lower.iter().chain(upper.iter()).chain(demands.iter()) {
        if *value == 0 {
            continue;
        }
        let trailing = value.abs().trailing_zeros();
        min_trailing = min_trailing.min(trailing);
    }
    if min_trailing == u32::MAX {
        return 1;
    }
    1_i64 << min_trailing.min(62)
}

fn max_abs(values: &[i64]) -> i64 {
    values.iter().map(|value| value.abs()).max().unwrap_or(0)
}

fn next_power_of_two(value: i64) -> i64 {
    if value <= 1 {
        1
    } else {
        let shifted = (value - 1) as u64;
        shifted.next_power_of_two() as i64
    }
}

fn bigint_bits(value: &BigInt) -> u64 {
    if value.is_zero() {
        return 1;
    }
    let mut value = value.abs();
    let mut bits = 0;
    while value > BigInt::zero() {
        value >>= 1;
        bits += 1;
    }
    bits.max(1)
}

fn feasible_initial_flow(problem: &McfProblem, flow: Option<Vec<i64>>) -> Option<Vec<i64>> {
    let flow = flow?;
    if flow.len() != problem.edge_count() {
        return None;
    }
    let mut balance = vec![0_i64; problem.node_count];
    for (idx, &value) in flow.iter().enumerate() {
        let lower = problem.lower[idx];
        let upper = problem.upper[idx];
        if value < lower || value > upper {
            return None;
        }
        let tail = problem.tails[idx] as usize;
        let head = problem.heads[idx] as usize;
        balance[tail] = balance[tail].checked_sub(value)?;
        balance[head] = balance[head].checked_add(value)?;
    }
    for (node, &demand) in problem.demands.iter().enumerate() {
        if balance[node] != demand {
            return None;
        }
    }
    Some(flow)
}
