use crate::ipm::{self, IpmResult, IpmRunContext, IpmRunKind};
use crate::{McfError, McfOptions, McfProblem};
use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};
use std::time::Instant;

use super::{record_cost_round, update_initial_flow, ScalingStats};
use super::utils::{bigint_bits, feasible_initial_flow, max_abs};

#[derive(Debug, Clone)]
pub struct CostScalingInfo {
    pub divisor: i64,
    pub bound: i64,
    pub max_scaled_cost: i64,
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

        record_cost_round(stats, round, midpoint.clone(), solved_cost.clone(), &ipm_result);

        last_flow = update_initial_flow(&scaled_problem, last_flow, &ipm_result);

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
    if max_scaled_cost > bound {
        return Err(McfError::InvalidInput(
            "cost scaling failed to reduce bounds".to_string(),
        ));
    }
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

pub fn needs_cost_scaling(problem: &McfProblem) -> bool {
    let m = problem.edge_count().max(1) as i64;
    let bound = m.saturating_pow(3).max(1);
    max_abs(&problem.cost) > bound
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

fn cost_divisor(max_cost: i64, bound: i64) -> i64 {
    if max_cost <= bound {
        1
    } else {
        let needed = (max_cost + bound - 1) / bound;
        needed.max(1)
    }
}
