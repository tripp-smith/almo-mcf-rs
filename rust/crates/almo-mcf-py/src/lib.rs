#![allow(clippy::useless_conversion)]

use almo_mcf_core::ipm::IpmTermination;
use almo_mcf_core::{ipm, min_cost_flow_exact, McfOptions, McfProblem, SolverMode, Strategy};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn build_problem(
    n: usize,
    tail: PyReadonlyArray1<'_, i64>,
    head: PyReadonlyArray1<'_, i64>,
    lower: PyReadonlyArray1<'_, i64>,
    upper: PyReadonlyArray1<'_, i64>,
    cost: PyReadonlyArray1<'_, i64>,
    demand: PyReadonlyArray1<'_, i64>,
) -> PyResult<McfProblem> {
    let tail_slice = tail.as_slice()?;
    let head_slice = head.as_slice()?;
    if tail_slice.len() != head_slice.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "tail and head arrays must match length",
        ));
    }
    let mut tail_vec = Vec::with_capacity(tail_slice.len());
    let mut head_vec = Vec::with_capacity(head_slice.len());
    for (&t, &h) in tail_slice.iter().zip(head_slice.iter()) {
        let tail_u32 = u32::try_from(t)
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("tail index out of range"))?;
        let head_u32 = u32::try_from(h)
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("head index out of range"))?;
        tail_vec.push(tail_u32);
        head_vec.push(head_u32);
    }
    let lower_slice = lower.as_slice()?;
    let upper_slice = upper.as_slice()?;
    let cost_slice = cost.as_slice()?;
    if lower_slice.len() != tail_vec.len()
        || upper_slice.len() != tail_vec.len()
        || cost_slice.len() != tail_vec.len()
    {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "edge attribute arrays must match tail/head length",
        ));
    }
    let lower_vec = lower_slice.to_vec();
    let upper_vec = upper_slice.to_vec();
    let cost_vec = cost_slice.to_vec();
    let demand_vec = demand.as_slice()?.to_vec();

    if demand_vec.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "demand length does not match n",
        ));
    }

    McfProblem::new(
        tail_vec, head_vec, lower_vec, upper_vec, cost_vec, demand_vec,
    )
    .map_err(|err| pyo3::exceptions::PyValueError::new_err(format!("{err:?}")))
}

#[allow(clippy::too_many_arguments)]
fn build_options(
    strategy: Option<String>,
    oracle_mode: Option<String>,
    rebuild_every: Option<usize>,
    max_iters: Option<usize>,
    tolerance: Option<f64>,
    numerical_clamp_log: Option<f64>,
    residual_min: Option<f64>,
    barrier_alpha_min: Option<f64>,
    barrier_alpha_max: Option<f64>,
    barrier_clamp_max: Option<f64>,
    gradient_clamp_max: Option<f64>,
    log_numerical_clamping: Option<bool>,
    seed: Option<u64>,
    deterministic_seed: Option<u64>,
    threads: Option<usize>,
    alpha: Option<f64>,
    use_ipm: Option<bool>,
    use_scaling: Option<bool>,
    force_cost_scaling: Option<bool>,
    disable_capacity_scaling: Option<bool>,
    approx_factor: Option<f64>,
    deterministic: Option<bool>,
) -> PyResult<McfOptions> {
    let mut opts = McfOptions::default();
    if let Some(value) = max_iters {
        opts.max_iters = value;
    }
    if let Some(value) = tolerance {
        opts.tolerance = value;
    }
    if let Some(value) = numerical_clamp_log {
        if value <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "numerical_clamp_log must be positive",
            ));
        }
        opts.numerical_clamp_log = value;
    }
    if let Some(value) = residual_min {
        if value <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "residual_min must be positive",
            ));
        }
        opts.residual_min = value;
    }
    if let Some(value) = barrier_alpha_min {
        if value <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "barrier_alpha_min must be positive",
            ));
        }
        opts.barrier_alpha_min = value;
    }
    if let Some(value) = barrier_alpha_max {
        if value <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "barrier_alpha_max must be positive",
            ));
        }
        opts.barrier_alpha_max = value;
    }
    if let Some(value) = barrier_clamp_max {
        if value <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "barrier_clamp_max must be positive",
            ));
        }
        opts.barrier_clamp_max = value;
    }
    if let Some(value) = gradient_clamp_max {
        if value <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "gradient_clamp_max must be positive",
            ));
        }
        opts.gradient_clamp_max = value;
    }
    if let Some(value) = log_numerical_clamping {
        opts.log_numerical_clamping = value;
    }
    if let Some(value) = seed {
        opts.seed = value;
    }
    if let Some(value) = deterministic_seed {
        opts.deterministic_seed = Some(value);
    }
    if let Some(value) = threads {
        opts.threads = value;
    }
    if let Some(value) = alpha {
        if value <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "alpha must be positive",
            ));
        }
        opts.alpha = Some(value);
    }
    if let Some(value) = use_ipm {
        opts.use_ipm = Some(value);
    }
    if let Some(value) = use_scaling {
        opts.use_scaling = Some(value);
    }
    if let Some(value) = force_cost_scaling {
        opts.force_cost_scaling = value;
    }
    if let Some(value) = disable_capacity_scaling {
        opts.disable_capacity_scaling = value;
    }
    if let Some(value) = approx_factor {
        if value < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "approx_factor must be non-negative",
            ));
        }
        opts.approx_factor = value;
    }
    if let Some(value) = deterministic {
        opts.deterministic = value;
    }
    if let Some(value) = rebuild_every {
        opts.strategy = Strategy::PeriodicRebuild {
            rebuild_every: value,
        };
    }
    if let Some(mode) = strategy {
        let normalized = mode.to_ascii_lowercase();
        opts.strategy = match normalized.as_str() {
            "full_dynamic" | "full-dynamic" | "fulldynamic" => Strategy::FullDynamic,
            "periodic_rebuild" | "periodic-rebuild" | "periodicrebuild" => {
                let rebuild_every = rebuild_every.unwrap_or(25);
                Strategy::PeriodicRebuild { rebuild_every }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "strategy must be 'full_dynamic' or 'periodic_rebuild'",
                ))
            }
        };
    }
    if let Some(mode) = oracle_mode {
        let normalized = mode.to_ascii_lowercase();
        opts.oracle_mode = match normalized.as_str() {
            "dynamic" => almo_mcf_core::OracleMode::Dynamic,
            "fallback" => almo_mcf_core::OracleMode::Fallback,
            "hybrid" => almo_mcf_core::OracleMode::Hybrid,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "oracle_mode must be 'dynamic', 'fallback', or 'hybrid'",
                ))
            }
        };
    }
    if opts.deterministic {
        opts.threads = 1;
    }
    Ok(opts)
}

fn solver_mode_label(mode: SolverMode) -> &'static str {
    match mode {
        SolverMode::Classic => "classic",
        SolverMode::Ipm => "ipm",
        SolverMode::IpmScaled => "ipm_scaled",
        SolverMode::ClassicFallback => "classic_fallback",
    }
}

fn stats_to_dict(
    py: Python<'_>,
    solver_mode: SolverMode,
    ipm_stats: Option<almo_mcf_core::IpmSummary>,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("solver_mode", solver_mode_label(solver_mode))?;
    let (termination_label, iterations, final_gap, summary) = if let Some(summary) = ipm_stats {
        let termination_label = match summary.termination {
            IpmTermination::Converged => "converged",
            IpmTermination::IterationLimit => "iteration_limit",
            IpmTermination::TimeLimit => "time_limit",
            IpmTermination::NoImprovingCycle => "no_improving_cycle",
        };
        (
            termination_label,
            summary.iterations,
            summary.final_gap,
            Some(summary),
        )
    } else {
        ("classic", 0, 0.0, None)
    };
    dict.set_item("termination", termination_label)?;
    dict.set_item("iterations", iterations)?;
    dict.set_item("final_gap", final_gap)?;
    if let Some(summary) = summary {
        dict.set_item("cycle_scoring_ms", summary.cycle_scoring_ms)?;
        dict.set_item("barrier_compute_ms", summary.barrier_compute_ms)?;
        dict.set_item("spanner_update_ms", summary.spanner_update_ms)?;
        dict.set_item("rounding_performed", summary.rounding_performed)?;
        dict.set_item("rounding_success", summary.rounding_success)?;
        dict.set_item("final_integer_cost", summary.final_integer_cost)?;
        dict.set_item("post_rounding_gap", summary.post_rounding_gap)?;
        dict.set_item("cycles_canceled", summary.cycles_canceled)?;
        dict.set_item("rounding_adjustment_cost", summary.rounding_adjustment_cost)?;
        dict.set_item("is_exact_optimal", summary.is_exact_optimal)?;
        dict.set_item("deterministic_mode_used", summary.deterministic_mode_used)?;
        dict.set_item("seed_used", summary.seed_used)?;
        dict.set_item(
            "numerical_clamping_occurred",
            summary.numerical_clamping_occurred,
        )?;
        dict.set_item("max_barrier_value", summary.max_barrier_value)?;
        dict.set_item("min_residual_seen", summary.min_residual_seen)?;
    }
    Ok(dict.to_object(py))
}

#[allow(clippy::too_many_arguments, clippy::useless_conversion)]
#[pyfunction]
fn min_cost_flow_edges(
    py: Python<'_>,
    n: usize,
    tail: PyReadonlyArray1<'_, i64>,
    head: PyReadonlyArray1<'_, i64>,
    lower: PyReadonlyArray1<'_, i64>,
    upper: PyReadonlyArray1<'_, i64>,
    cost: PyReadonlyArray1<'_, i64>,
    demand: PyReadonlyArray1<'_, i64>,
) -> PyResult<Py<PyArray1<i64>>> {
    let problem = build_problem(n, tail, head, lower, upper, cost, demand)?;
    let solution = min_cost_flow_exact(&problem, &McfOptions::default())
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}")))?;

    Ok(PyArray1::from_vec_bound(py, solution.flow).unbind())
}

#[allow(clippy::too_many_arguments, clippy::useless_conversion)]
#[pyfunction]
#[pyo3(signature = (
    n,
    tail,
    head,
    lower,
    upper,
    cost,
    demand,
    *,
    strategy = None,
    oracle_mode = None,
    rebuild_every = None,
    max_iters = None,
    tolerance = None,
    numerical_clamp_log = None,
    residual_min = None,
    barrier_alpha_min = None,
    barrier_alpha_max = None,
    barrier_clamp_max = None,
    gradient_clamp_max = None,
    log_numerical_clamping = None,
    seed = None,
    deterministic_seed = None,
    threads = None,
    alpha = None,
    use_ipm = None,
    use_scaling = None,
    force_cost_scaling = None,
    disable_capacity_scaling = None,
    approx_factor = None,
    deterministic = None
))]
fn min_cost_flow_edges_with_options(
    py: Python<'_>,
    n: usize,
    tail: PyReadonlyArray1<'_, i64>,
    head: PyReadonlyArray1<'_, i64>,
    lower: PyReadonlyArray1<'_, i64>,
    upper: PyReadonlyArray1<'_, i64>,
    cost: PyReadonlyArray1<'_, i64>,
    demand: PyReadonlyArray1<'_, i64>,
    strategy: Option<String>,
    oracle_mode: Option<String>,
    rebuild_every: Option<usize>,
    max_iters: Option<usize>,
    tolerance: Option<f64>,
    numerical_clamp_log: Option<f64>,
    residual_min: Option<f64>,
    barrier_alpha_min: Option<f64>,
    barrier_alpha_max: Option<f64>,
    barrier_clamp_max: Option<f64>,
    gradient_clamp_max: Option<f64>,
    log_numerical_clamping: Option<bool>,
    seed: Option<u64>,
    deterministic_seed: Option<u64>,
    threads: Option<usize>,
    alpha: Option<f64>,
    use_ipm: Option<bool>,
    use_scaling: Option<bool>,
    force_cost_scaling: Option<bool>,
    disable_capacity_scaling: Option<bool>,
    approx_factor: Option<f64>,
    deterministic: Option<bool>,
) -> PyResult<(Py<PyArray1<i64>>, Option<PyObject>)> {
    let problem = build_problem(n, tail, head, lower, upper, cost, demand)?;
    let opts = build_options(
        strategy,
        oracle_mode,
        rebuild_every,
        max_iters,
        tolerance,
        numerical_clamp_log,
        residual_min,
        barrier_alpha_min,
        barrier_alpha_max,
        barrier_clamp_max,
        gradient_clamp_max,
        log_numerical_clamping,
        seed,
        deterministic_seed,
        threads,
        alpha,
        use_ipm,
        use_scaling,
        force_cost_scaling,
        disable_capacity_scaling,
        approx_factor,
        deterministic,
    )?;

    let solution = min_cost_flow_exact(&problem, &opts)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}")))?;

    let stats = Some(stats_to_dict(py, solution.solver_mode, solution.ipm_stats)?);

    Ok((PyArray1::from_vec_bound(py, solution.flow).unbind(), stats))
}

#[allow(clippy::too_many_arguments, clippy::useless_conversion)]
#[pyfunction]
#[pyo3(signature = (
    n,
    tail,
    head,
    lower,
    upper,
    cost,
    demand,
    *,
    strategy = None,
    oracle_mode = None,
    rebuild_every = None,
    max_iters = None,
    tolerance = None,
    numerical_clamp_log = None,
    residual_min = None,
    barrier_alpha_min = None,
    barrier_alpha_max = None,
    barrier_clamp_max = None,
    gradient_clamp_max = None,
    log_numerical_clamping = None,
    seed = None,
    deterministic_seed = None,
    threads = None,
    alpha = None,
    force_cost_scaling = None,
    disable_capacity_scaling = None,
    approx_factor = None,
    deterministic = None
))]
fn min_cost_flow_edges_with_scaling(
    py: Python<'_>,
    n: usize,
    tail: PyReadonlyArray1<'_, i64>,
    head: PyReadonlyArray1<'_, i64>,
    lower: PyReadonlyArray1<'_, i64>,
    upper: PyReadonlyArray1<'_, i64>,
    cost: PyReadonlyArray1<'_, i64>,
    demand: PyReadonlyArray1<'_, i64>,
    strategy: Option<String>,
    oracle_mode: Option<String>,
    rebuild_every: Option<usize>,
    max_iters: Option<usize>,
    tolerance: Option<f64>,
    numerical_clamp_log: Option<f64>,
    residual_min: Option<f64>,
    barrier_alpha_min: Option<f64>,
    barrier_alpha_max: Option<f64>,
    barrier_clamp_max: Option<f64>,
    gradient_clamp_max: Option<f64>,
    log_numerical_clamping: Option<bool>,
    seed: Option<u64>,
    deterministic_seed: Option<u64>,
    threads: Option<usize>,
    alpha: Option<f64>,
    force_cost_scaling: Option<bool>,
    disable_capacity_scaling: Option<bool>,
    approx_factor: Option<f64>,
    deterministic: Option<bool>,
) -> PyResult<(Py<PyArray1<i64>>, Option<PyObject>)> {
    let problem = build_problem(n, tail, head, lower, upper, cost, demand)?;
    let mut opts = build_options(
        strategy,
        oracle_mode,
        rebuild_every,
        max_iters,
        tolerance,
        numerical_clamp_log,
        residual_min,
        barrier_alpha_min,
        barrier_alpha_max,
        barrier_clamp_max,
        gradient_clamp_max,
        log_numerical_clamping,
        seed,
        deterministic_seed,
        threads,
        alpha,
        None,
        Some(true),
        force_cost_scaling,
        disable_capacity_scaling,
        approx_factor,
        deterministic,
    )?;
    opts.use_scaling = Some(true);

    let solution = min_cost_flow_exact(&problem, &opts)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}")))?;

    let stats = Some(stats_to_dict(py, solution.solver_mode, solution.ipm_stats)?);

    Ok((PyArray1::from_vec_bound(py, solution.flow).unbind(), stats))
}

#[allow(clippy::too_many_arguments, clippy::useless_conversion)]
#[pyfunction]
#[pyo3(signature = (
    n,
    tail,
    head,
    lower,
    upper,
    cost,
    demand,
    *,
    strategy = None,
    oracle_mode = None,
    rebuild_every = None,
    max_iters = None,
    tolerance = None,
    numerical_clamp_log = None,
    residual_min = None,
    barrier_alpha_min = None,
    barrier_alpha_max = None,
    barrier_clamp_max = None,
    gradient_clamp_max = None,
    log_numerical_clamping = None,
    seed = None,
    deterministic_seed = None,
    threads = None,
    alpha = None,
    use_ipm = None,
    use_scaling = None,
    force_cost_scaling = None,
    disable_capacity_scaling = None,
    approx_factor = None,
    deterministic = None
))]
fn run_ipm_edges(
    py: Python<'_>,
    n: usize,
    tail: PyReadonlyArray1<'_, i64>,
    head: PyReadonlyArray1<'_, i64>,
    lower: PyReadonlyArray1<'_, i64>,
    upper: PyReadonlyArray1<'_, i64>,
    cost: PyReadonlyArray1<'_, i64>,
    demand: PyReadonlyArray1<'_, i64>,
    strategy: Option<String>,
    oracle_mode: Option<String>,
    rebuild_every: Option<usize>,
    max_iters: Option<usize>,
    tolerance: Option<f64>,
    numerical_clamp_log: Option<f64>,
    residual_min: Option<f64>,
    barrier_alpha_min: Option<f64>,
    barrier_alpha_max: Option<f64>,
    barrier_clamp_max: Option<f64>,
    gradient_clamp_max: Option<f64>,
    log_numerical_clamping: Option<bool>,
    seed: Option<u64>,
    deterministic_seed: Option<u64>,
    threads: Option<usize>,
    alpha: Option<f64>,
    use_ipm: Option<bool>,
    use_scaling: Option<bool>,
    force_cost_scaling: Option<bool>,
    disable_capacity_scaling: Option<bool>,
    approx_factor: Option<f64>,
    deterministic: Option<bool>,
) -> PyResult<(Py<PyArray1<f64>>, PyObject)> {
    let problem = build_problem(n, tail, head, lower, upper, cost, demand)?;
    let opts = build_options(
        strategy,
        oracle_mode,
        rebuild_every,
        max_iters,
        tolerance,
        numerical_clamp_log,
        residual_min,
        barrier_alpha_min,
        barrier_alpha_max,
        barrier_clamp_max,
        gradient_clamp_max,
        log_numerical_clamping,
        seed,
        deterministic_seed,
        threads,
        alpha,
        use_ipm,
        use_scaling,
        force_cost_scaling,
        disable_capacity_scaling,
        approx_factor,
        deterministic,
    )?;

    let ipm_result = ipm::run_ipm(&problem, &opts)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}")))?;
    let mut aggregate = almo_mcf_core::numerics::barrier::BarrierClampStats::default();
    for item in &ipm_result.stats.barrier_clamp_stats {
        aggregate.merge(item);
    }
    let summary = almo_mcf_core::IpmSummary {
        iterations: ipm_result.stats.iterations,
        final_gap: ipm_result.stats.last_gap,
        cycle_scoring_ms: ipm_result.stats.cycle_times_ms.iter().sum(),
        barrier_compute_ms: ipm_result.stats.barrier_times_ms.iter().sum(),
        spanner_update_ms: ipm_result.stats.spanner_update_times_ms.iter().sum(),
        termination: ipm_result.termination,
        oracle_mode: ipm_result.stats.oracle_mode,
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
    };
    let stats = stats_to_dict(py, SolverMode::Ipm, Some(summary))?;
    Ok((
        PyArray1::from_vec_bound(py, ipm_result.flow).unbind(),
        stats,
    ))
}

#[pymodule]
fn _core(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(min_cost_flow_edges, module)?)?;
    module.add_function(wrap_pyfunction!(min_cost_flow_edges_with_options, module)?)?;
    module.add_function(wrap_pyfunction!(min_cost_flow_edges_with_scaling, module)?)?;
    module.add_function(wrap_pyfunction!(run_ipm_edges, module)?)?;
    module.add("__doc__", "Rust core bindings for almo-mcf")?;
    module.add(
        "__all__",
        vec![
            "min_cost_flow_edges",
            "min_cost_flow_edges_with_options",
            "min_cost_flow_edges_with_scaling",
            "run_ipm_edges",
            "__version__",
        ],
    )?;
    Ok(())
}
