use crate::ipm::{self, IpmResult, IpmTermination};
use crate::scaling;
use crate::{McfError, McfOptions, McfProblem, McfSolution};

pub fn set_dynamic_max_iters(m: usize, approx_factor: f64) -> usize {
    let mm = m.max(1) as f64;
    let eps = approx_factor.abs().max(1e-12);
    let iters = mm.sqrt() * (mm / eps).ln().max(1.0);
    iters.ceil() as usize
}

pub fn run_ipm_pipeline(problem: &McfProblem, opts: &McfOptions) -> Result<IpmResult, McfError> {
    let mut tuned = opts.clone();
    tuned.max_iters = tuned
        .max_iters
        .min(set_dynamic_max_iters(problem.edge_count(), tuned.approx_factor).max(1));
    ipm::run_ipm(problem, &tuned)
}

pub fn successive_shortest_path_fallback(problem: &McfProblem) -> Result<McfSolution, McfError> {
    crate::min_cost_flow_exact(
        problem,
        &McfOptions {
            use_ipm: Some(false),
            ..McfOptions::default()
        },
    )
}

pub fn run_full_ipm(problem: &McfProblem, opts: &McfOptions) -> Result<McfSolution, McfError> {
    let mut tuned = opts.clone();
    tuned.max_iters = set_dynamic_max_iters(problem.edge_count(), tuned.approx_factor).max(1);

    match crate::min_cost_flow_exact(problem, &tuned) {
        Ok(sol) => Ok(sol),
        Err(_) => successive_shortest_path_fallback(problem),
    }
}

pub fn min_cost_flow_edges_with_scaling(
    problem: &McfProblem,
    opts: &McfOptions,
) -> Result<McfSolution, McfError> {
    scaling::solve_mcf_with_scaling(problem, opts)
}

pub fn should_trigger_fallback(result: &IpmResult, max_iters: usize) -> bool {
    matches!(result.termination, IpmTermination::IterationLimit)
        && result.stats.iterations > 2 * max_iters
}
