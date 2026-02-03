use crate::ipm::{self, IpmRunContext, IpmRunKind};
use crate::{McfError, McfOptions, McfProblem};

use super::utils::{
    feasible_initial_flow, max_abs, max_power_of_two_divisor, next_power_of_two, scale_flow,
    scale_problem,
};
use super::{record_capacity_phase, update_initial_flow, ScalingStats};

#[derive(Debug, Clone)]
pub struct CapacityScalingResult {
    pub scaled_problem: McfProblem,
    pub scale_factor: i64,
    pub initial_flow: Option<Vec<i64>>,
}

#[derive(Debug, Clone)]
pub struct CapacityScalingInfo {
    pub divisor: i64,
    pub bound: i64,
    pub max_scaled_capacity: i64,
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

        record_capacity_phase(stats, phase, divisor, &ipm_result);

        if divisor == scale_factor {
            last_flow = update_initial_flow(&scaled_problem, last_flow, &ipm_result);
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

pub fn build_capacity_scaled_problem(
    problem: &McfProblem,
) -> Result<(McfProblem, CapacityScalingInfo), McfError> {
    let (scale_factor, bound) = capacity_scale_factor(problem);
    let scaled_problem = scale_problem(problem, scale_factor)?;
    let max_scaled_capacity = max_abs(&scaled_problem.upper);
    if max_scaled_capacity > bound {
        return Err(McfError::InvalidInput(
            "capacity scaling failed to reduce bounds".to_string(),
        ));
    }

    Ok((
        scaled_problem,
        CapacityScalingInfo {
            divisor: scale_factor,
            bound,
            max_scaled_capacity,
        },
    ))
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
