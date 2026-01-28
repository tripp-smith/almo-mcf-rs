use almo_mcf_core::{min_cost_flow_exact, McfOptions, McfProblem};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

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

    let problem = McfProblem::new(
        tail_vec, head_vec, lower_vec, upper_vec, cost_vec, demand_vec,
    )
    .map_err(|err| pyo3::exceptions::PyValueError::new_err(format!("{err:?}")))?;
    let solution = min_cost_flow_exact(&problem, &McfOptions::default())
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(format!("{err:?}")))?;

    Ok(PyArray1::from_vec_bound(py, solution.flow).unbind())
}

#[pymodule]
fn _core(_py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(min_cost_flow_edges, module)?)?;
    module.add("__doc__", "Rust core bindings for almo-mcf")?;
    module.add("__all__", vec!["min_cost_flow_edges", "__version__"])?;
    Ok(())
}
