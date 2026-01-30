# almo-mcf

Almost-linear minimum-cost flow is the research goal for this project, but the **current
implementation is a pragmatic, exact min-cost flow solver** with a Python + Rust
interface. The Python API mirrors NetworkX conventions while the Rust core provides
an integer min-cost flow solver with lower bounds and node demands.

This README reflects the **current codebase** and what you can use today. The
algorithmic IPM (interior point method) and dynamic min-ratio oracle described in
`DESIGN_SPEC.md` are scaffolded but not yet wired into the default solver.

## Features (current)

- **Exact min-cost flow** for directed graphs with:
  - node demands (sum must be zero)
  - per-edge lower/upper capacities
  - integer edge costs
- **NetworkX-compatible adapter** (`min_cost_flow`, `min_cost_flow_cost`)
- **Rust core** with Python bindings via `maturin`
- Deterministic, reproducible results

## Status

The IPM-based almost-linear solver is **not yet the default path**. The current
solver uses a successive shortest path routine (Bellman–Ford in the residual
network) after reducing lower bounds and introducing a super-source/super-sink
for feasibility. This makes the library correct for small to medium instances,
with performance that scales similarly to classic polynomial-time algorithms.

## Installation

```bash
pip install almo-mcf
```

> The PyPI package ships a Rust extension module. If you are installing from
> source, you need a Rust toolchain (stable) and a C compiler.

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
- The IPM-based almost-linear algorithm is **not yet enabled**.
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

## License

Apache 2.0. See `LICENSE` for details.
