import numpy as np
import pytest

from almo_mcf import _core


def test_min_cost_flow_edges_with_lower_bounds():
    flow = _core.min_cost_flow_edges(
        2,
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        np.asarray([5], dtype=np.int64),
        np.asarray([2], dtype=np.int64),
        np.asarray([-3, 3], dtype=np.int64),
    )
    assert flow.tolist() == [3]


def test_min_cost_flow_edges_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="tail and head"):
        _core.min_cost_flow_edges(
            2,
            np.asarray([0], dtype=np.int64),
            np.asarray([1, 1], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
            np.asarray([0, 0], dtype=np.int64),
        )


def test_min_cost_flow_edges_with_options_returns_stats():
    flow, stats = _core.min_cost_flow_edges_with_options(
        2,
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        np.asarray([5], dtype=np.int64),
        np.asarray([2], dtype=np.int64),
        np.asarray([-3, 3], dtype=np.int64),
        strategy="periodic_rebuild",
        rebuild_every=5,
        max_iters=10,
        tolerance=1e-6,
        seed=3,
        threads=1,
        alpha=0.001,
    )
    assert flow.tolist() == [3]
    assert stats is not None
    assert {"iterations", "final_gap", "termination", "solver_mode"} <= set(stats.keys())


def test_min_cost_flow_edges_with_scaling():
    flow, stats = _core.min_cost_flow_edges_with_scaling(
        2,
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        np.asarray([1], dtype=np.int64),
        np.asarray([5], dtype=np.int64),
        np.asarray([-1, 1], dtype=np.int64),
        max_iters=25,
    )
    assert flow.tolist() == [1]
    assert stats is not None
    assert {"iterations", "final_gap", "termination", "solver_mode"} <= set(stats.keys())
