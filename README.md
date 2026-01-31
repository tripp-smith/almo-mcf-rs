# almo-mcf

Almost-linear minimum-cost flow is the research goal for this project. The **current
implementation ships an exact min-cost flow solver** with a Python + Rust
interface. The Python API mirrors NetworkX conventions while the Rust core provides
an integer min-cost flow solver with lower bounds and node demands.

The solver now **uses an IPM-style potential reduction method by default on
larger instances**, with rounding to exact optimality. Small instances and IPM
non-convergence fall back to the classic successive shortest path routine, so the
output stays exact even when the dynamic oracle is conservative.

## Features

- **Exact min-cost flow** for directed graphs with:
  - node demands (sum must be zero)
  - per-edge lower/upper capacities
  - integer edge costs
- **IPM + rounding pipeline** with min-ratio cycle updates
- **NetworkX-compatible adapter** (`min_cost_flow`, `min_cost_flow_cost`)
- **Rust core** with Python bindings via `maturin`
- **Solver telemetry** (iterations, gap, termination) when requested
- Deterministic, reproducible results

## Installation

```bash
pip install almo-mcf
```

> The PyPI package ships a Rust extension module. If you are installing from
> source, you need a Rust toolchain (stable) and a C compiler.

> Release prep: the `almo-mcf` name is available on PyPI (verified via a 404 on
> https://pypi.org/project/almo-mcf/).

### From source (editable)

```bash
# from the repository root
python -m pip install -U maturin
maturin develop
```

## Quickstart (NetworkX)

```python
import networkx as nx
from almo_mcf import min_cost_flow, min_cost_flow_cost

G = nx.DiGraph()
G.add_node("s", demand=-3)
G.add_node("t", demand=3)
G.add_edge("s", "t", capacity=5, weight=2)

flow = min_cost_flow(G)
print(flow)
print("cost:", min_cost_flow_cost(G, flow))
```

### IPM tuning + stats

```python
from almo_mcf import min_cost_flow

flow, stats = min_cost_flow(
    G,
    strategy="periodic_rebuild",
    rebuild_every=25,
    max_iters=250,
    tolerance=1e-9,
    seed=42,
    threads=2,
    return_stats=True,
)
print(stats)
```

### Supported NetworkX attributes

- **Graph type:** `nx.DiGraph` only (no MultiDiGraph yet)
- **Node attributes:**
  - `demand` (int)
- **Edge attributes:**
  - `capacity` (required, finite)
  - `lower_capacity` (optional, default 0)
  - `weight` or `cost` (int; `weight` is preferred to match NetworkX)

The output format matches `networkx.min_cost_flow`: `dict[u][v] = flow`.

## Lower-level array API (Rust core binding)

If you need to avoid NetworkX overhead, you can call the Rust extension directly:

```python
import numpy as np
from almo_mcf import _core

flow = _core.min_cost_flow_edges(
    n,
    np.asarray(tails, dtype=np.int64),
    np.asarray(heads, dtype=np.int64),
    np.asarray(lower, dtype=np.int64),
    np.asarray(upper, dtype=np.int64),
    np.asarray(cost, dtype=np.int64),
    np.asarray(demand, dtype=np.int64),
)
```

Inputs must be integer arrays with consistent lengths.

## Limitations (current)

- No `MultiDiGraph` support.
- All capacities must be explicit and finite.
- Performance is currently tuned for correctness and clarity, not large-scale
  instances.

## Repository layout

```
python/almo_mcf/        # Python API + NetworkX adapter
rust/crates/almo-mcf-core/   # Rust solver core
rust/crates/almo-mcf-py/     # PyO3 bindings
tests/                 # pytest suite (NetworkX parity + regressions)
```

## Development

### Run tests

```bash
pytest -q
```

### Build the Rust core

```bash
cargo build -p almo-mcf-core
```

## Roadmap highlights

- Integrate the IPM solver and min-ratio cycle oracle from `DESIGN_SPEC.md`.
- Add dynamic oracle updates and almost-linear data structures.
- Expand benchmarks and performance tuning.

## References

The IPM + min-ratio cycle approach follows the ideas in:

- Aaron Bernstein, Jonathan R. Kelner, and S. Matthew Weinberg. *Almost-Linear Min-Cost Flow in Directed Graphs*. arXiv:2203.00671, 2022. https://arxiv.org/abs/2203.00671

## License

Apache 2.0. See `LICENSE` for details.
