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
