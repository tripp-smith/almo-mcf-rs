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
- **Capacity and cost scaling** for large \(U/C\) bounds
- **NetworkX-compatible adapter** (`min_cost_flow`, `min_cost_flow_cost`)
- **Rust core** with Python bindings via `maturin`
- **Solver telemetry** (iterations, gap, termination) when requested
- Deterministic, reproducible results with an opt-out randomized mode

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
    use_ipm=True,            # force IPM path (set False for classic SSP)
    use_scaling=None,        # auto-detect large U/C, set True/False to override
    deterministic=True,      # default; set False for randomized cycle selection
    strategy="periodic_rebuild",
    rebuild_every=25,
    max_iters=250,
    tolerance=1e-9,
    seed=42,
    threads=2,
    approx_factor=0.2,
    return_stats=True,
)
print(stats)
```

### Almost-Linear Mode

The solver defaults to the almost-linear IPM path on larger instances. The
`use_ipm` flag forces the solver path, while `deterministic=True` disables
randomized sampling in the dynamic oracle to match the derandomized pipeline
from Paper 2. When `return_stats=True`, the stats dictionary includes a
`solver_mode` key so you can log whether the run used `ipm`, `ipm_scaled`,
`classic`, or `classic_fallback`.

```python
flow, stats = min_cost_flow(
    G,
    use_ipm=True,       # prefer IPM for large graphs
    deterministic=True, # reproducible shifting instead of random sampling
    return_stats=True,
)
print(stats["solver_mode"])
```

### Scaling for large capacity/cost bounds

When inputs contain very large capacities or costs, the solver automatically
enables capacity/cost scaling to keep the IPM rounds within polynomial bounds.
You can override this behavior via `use_scaling=True` or
`use_scaling=False`, or call the explicit helper `min_cost_flow_scaled`.

### Deterministic vs. randomized solver behavior

Deterministic mode is enabled by default, which fixes the min-ratio cycle updates
and dynamic sparsification choices for reproducibility. To experiment with the
original randomized behavior (useful for performance comparisons), set
`deterministic=False` on solver calls such as `min_cost_flow` or the extension
reducers like `min_cost_flow_convex` and `max_flow_via_min_cost_circulation`.

When deterministic mode is enabled, the solver uses stable tie-breaking on edge
and node IDs, deterministic sparsification, and lexicographically smallest path
embeddings. You can optionally provide `deterministic_seed` to influence
tie-breaking hashes for debugging; this does **not** enable random sampling. The
stats dictionary returned with `return_stats=True` includes
`deterministic_mode_used` and `seed_used` so you can log reproducibility
metadata.

### Extensions (convex costs, max-flow, isotonic regression)

```python
import networkx as nx
from almo_mcf import min_cost_flow_convex, max_flow_via_min_cost_circulation

G = nx.MultiDiGraph()
G.add_node("s", demand=-2)
G.add_node("t", demand=2)
G.add_edge("s", "t", capacity=2, convex_cost=[0, 1, 6])
G.add_edge("s", "t", capacity=2, convex_cost=[0, 2, 4])
flow = min_cost_flow_convex(G)

H = nx.DiGraph()
H.add_edge("s", "t", capacity=5)
max_flow_value, max_flow = max_flow_via_min_cost_circulation(H, "s", "t")
```

### Supported NetworkX attributes

- **Graph type:** `nx.DiGraph` and `nx.MultiDiGraph`
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

# Force scaling in the core solver (useful for very large U/C bounds).
flow, stats = _core.min_cost_flow_edges_with_scaling(
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

### Benchmarks

There are benchmark scripts that compare IPM vs the classic successive
shortest-path (SSP) routine across sizes and capacity/cost ranges:

```bash
python tests/bench_ipm_vs_ssp.py --runs 3 --nodes 30 60 --edges 150 300 --capacity 10 50 --cost 5 20
python tests/bench_ipm_vs_networkx.py --nodes 60 --edges 300 --runs 3
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

- Li Chen, Rasmus Kyng, Yang P. Liu, Richard Peng, Maximilian Probst Gutenberg, and Sushant Sachdeva. *Maximum Flow and Minimum-Cost Flow in Almost-Linear Time*. arXiv:2203.00671, 2022. https://arxiv.org/abs/2203.00671
- Jan van den Brand, Li Chen, Rasmus Kyng, Yang P. Liu, Richard Peng, Maximilian Probst Gutenberg, Sushant Sachdeva, and Aaron Sidford. *A Deterministic Almost-Linear Time Algorithm for Minimum-Cost Flow*. arXiv:2309.16629, 2023. https://arxiv.org/abs/2309.16629

## License

Apache 2.0. See `LICENSE` for details.
