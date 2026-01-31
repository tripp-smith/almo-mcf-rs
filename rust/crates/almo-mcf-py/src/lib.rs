#![allow(clippy::useless_conversion)]

use almo_mcf_core::ipm::IpmTermination;
use almo_mcf_core::{ipm, min_cost_flow_exact, McfOptions, McfProblem, Strategy};
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

fn build_options(
    strategy: Option<String>,
    rebuild_every: Option<usize>,
    max_iters: Option<usize>,
    tolerance: Option<f64>,
    seed: Option<u64>,
    threads: Option<usize>,
) -> PyResult<McfOptions> {
    let mut opts = McfOptions::default();
    if let Some(value) = max_iters {
        opts.max_iters = value;
    }
    if let Some(value) = tolerance {
        opts.tolerance = value;
    }
    if let Some(value) = seed {
        opts.seed = value;
    }
    if let Some(value) = threads {
        opts.threads = value;
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
    Ok(opts)
}

fn stats_to_dict(
    py: Python<'_>,
    termination: IpmTermination,
    iterations: usize,
    final_gap: f64,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("iterations", iterations)?;
    dict.set_item("final_gap", final_gap)?;
    let termination_label = match termination {
        IpmTermination::Converged => "converged",
        IpmTermination::IterationLimit => "iteration_limit",
        IpmTermination::TimeLimit => "time_limit",
        IpmTermination::NoImprovingCycle => "no_improving_cycle",
    };
    dict.set_item("termination", termination_label)?;
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
    rebuild_every = None,
    max_iters = None,
    tolerance = None,
    seed = None,
    threads = None
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
    rebuild_every: Option<usize>,
    max_iters: Option<usize>,
    tolerance: Option<f64>,
    seed: Option<u64>,
    threads: Option<usize>,
) -> PyResult<(Py<PyArray1<i64>>, Option<PyObject>)> {
    let problem = build_problem(n, tail, head, lower, upper, cost, demand)?;
    let opts = build_options(strategy, rebuild_every, max_iters, tolerance, seed, threads)?;

    let solution = min_cost_flow_exact(&problem, &opts)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}")))?;

    let stats = if let Some(ipm_stats) = solution.ipm_stats {
        Some(stats_to_dict(
            py,
            ipm_stats.termination,
            ipm_stats.iterations,
            ipm_stats.final_gap,
        )?)
    } else {
        None
    };

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
    rebuild_every = None,
    max_iters = None,
    tolerance = None,
    seed = None,
    threads = None
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
    rebuild_every: Option<usize>,
    max_iters: Option<usize>,
    tolerance: Option<f64>,
    seed: Option<u64>,
    threads: Option<usize>,
) -> PyResult<(Py<PyArray1<f64>>, PyObject)> {
    let problem = build_problem(n, tail, head, lower, upper, cost, demand)?;
    let opts = build_options(strategy, rebuild_every, max_iters, tolerance, seed, threads)?;

    let ipm_result = ipm::run_ipm(&problem, &opts)
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}")))?;
    let stats = stats_to_dict(
        py,
        ipm_result.termination,
        ipm_result.stats.iterations,
        ipm_result.stats.last_gap,
    )?;
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
    module.add_function(wrap_pyfunction!(run_ipm_edges, module)?)?;
    module.add("__doc__", "Rust core bindings for almo-mcf")?;
    module.add(
        "__all__",
        vec![
            "min_cost_flow_edges",
            "min_cost_flow_edges_with_options",
            "run_ipm_edges",
            "__version__",
        ],
    )?;
    Ok(())
}
