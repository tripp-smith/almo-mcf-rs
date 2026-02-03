use crate::ipm::{self, IpmResult, IpmRunContext, IpmRunKind, IpmTermination};
use crate::{
    finalize_ipm_solution, IpmSummary, McfError, McfOptions, McfProblem, McfSolution, SolverMode,
};
use num_bigint::BigInt;

pub mod capacity;
pub mod cost;
mod utils;

pub use capacity::{
    build_capacity_scaled_problem, capacity_scaling, CapacityScalingInfo, CapacityScalingResult,
};
pub use cost::{build_cost_scaled_problem, cost_scaling, CostScalingInfo};

use utils::{feasible_initial_flow, flow_to_i64};

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

pub fn solve_mcf_with_scaling(
    problem: &McfProblem,
    opts: &McfOptions,
) -> Result<McfSolution, McfError> {
    let mut stats = ScalingStats::default();

    let capacity = if opts.disable_capacity_scaling {
        CapacityScalingResult {
            scaled_problem: problem.clone(),
            scale_factor: 1,
            initial_flow: feasible_initial_flow(problem, opts.initial_flow.clone()),
        }
    } else {
        capacity_scaling(problem, opts, &mut stats)?
    };

    let mut round_opts = opts.clone();
    round_opts.initial_flow = capacity.initial_flow.clone();
    let ipm_result = if opts.force_cost_scaling || cost::needs_cost_scaling(&capacity.scaled_problem)
    {
        cost_scaling(&capacity.scaled_problem, &round_opts, &mut stats)?
    } else {
        ipm::run_ipm_with_context(
            &capacity.scaled_problem,
            &round_opts,
            IpmRunContext {
                kind: IpmRunKind::CostScaling,
                round: 0,
            },
        )?
    };

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

pub(crate) fn record_cost_round(
    stats: &mut ScalingStats,
    round: usize,
    midpoint: BigInt,
    solved_cost: BigInt,
    ipm_result: &IpmResult,
) {
    stats.cost_rounds.push(CostScalingRound {
        round,
        midpoint,
        solved_cost,
        iterations: ipm_result.stats.iterations,
        termination: ipm_result.termination,
    });
}

pub(crate) fn record_capacity_phase(
    stats: &mut ScalingStats,
    phase: usize,
    divisor: i64,
    ipm_result: &IpmResult,
) {
    stats.capacity_phases.push(CapacityScalingPhase {
        phase,
        divisor,
        iterations: ipm_result.stats.iterations,
        termination: ipm_result.termination,
    });
}

pub(crate) fn update_initial_flow(
    problem: &McfProblem,
    last_flow: Option<Vec<i64>>,
    ipm_result: &IpmResult,
) -> Option<Vec<i64>> {
    feasible_initial_flow(problem, Some(flow_to_i64(&ipm_result.flow))).or(last_flow)
}
