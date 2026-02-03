# almo-mcf-rs
Almost-linear minimum-cost flow (Python + Rust)

# Project: Almost-linear minimum-cost flow (Python + Rust)

## Naming

* GitHub repo: `almo-mcf` (short for “almost-linear min-cost flow”)
* Rust crate: `almo_mcf`
* PyPI package: `almo-mcf`

I did not find an existing PyPI project page for `almo-mcf` (nor close spellings like `almomcf`) via direct PyPI lookups/search, but you should still do one last manual check before first publish.

## Scope and non-negotiables

Implement the Chen–Kyng–Liu–Peng–Probst Gutenberg–Sachdeva framework for exact min-cost flow via:

1. a potential-reduction interior point method (IPM) with the paper’s polynomial barrier, and
2. repeated solution of *undirected min-ratio cycle* subproblems to a coarse multiplicative accuracy, as the paper states (a large (m^{o(1)}) factor is permitted), and
3. rounding to an exact optimum when the duality gap proxy is tiny, per the paper’s termination discussion. 

The full asymptotic guarantee relies on a sophisticated dynamic data structure (stable “flow chasing”, tree-chain hierarchy, dynamic spanner with explicit embeddings). You should implement that structure as a first-class module (not handwaved), but keep a “fallback” path that is slower yet simpler (rebuild trees/spanners periodically) so correctness is never gated on the hardest performance components.

## Algorithm overview (what you are building)

### Min-cost flow model

Support directed graphs with:

* node demands (b_v) (positive = net inflow required, negative = net outflow supply), (\sum_v b_v = 0)
* per-edge lower/upper capacities (u^-_e, u^+_e)
* per-edge costs (c_e) (integral), bounded by (C)
* capacities bounded by (U)

Target: exact optimal integral min-cost flow. The theorem statement in the paper targets this regime. 

### Potential-reduction IPM core

Maintain a feasible flow (f) strictly inside box constraints (u^- < f < u^+). Use the paper’s potential:

* a log term tracking the current primal cost gap, plus
* barrier terms using (x^{-\alpha}) (not (-\log x)), with (\alpha = 1/(1000 \log(mU))) in the paper’s exposition. 

Define:

* gradient (g(f) = \nabla \Phi(f))
* lengths (\ell_e(f)) from the barrier contributions (the paper defines (\ell_e(f)) explicitly as a sum of ((u^+_e-f_e)^{-1-\alpha}) and ((f_e-u^-_e)^{-1-\alpha})). 

Each IPM iteration needs a *circulation* (\Delta) (i.e., (B^\top \Delta = 0)) that has sufficiently negative ratio:
[
\frac{g^\top \Delta}{\lVert L \Delta\rVert_1} \le -\kappa
]
where (L) is the diagonal operator induced by the lengths (the paper uses (kL\Delta k_1)). Scale (\Delta) so (kL\Delta k_1 = \kappa/50). Then a Taylor argument yields a guaranteed decrease in (\Phi). 

The key reduction: the optimal (or near-optimal) (\Delta) can be taken to be a simple cycle in an *undirected* representation, giving the “min-ratio cycle” subproblem sequence. 

### Undirected min-ratio cycle subproblem

At iteration (t), solve (approximately):
[
\min_{B^\top \Delta = 0} \frac{g^\top \Delta}{\lVert L \Delta\rVert_1}
]
to a coarse multiplicative factor (paper allows (m^{o(1)})). 

“Warm-up” static approach in the paper:

* sample a probabilistic low-stretch spanning tree (T)
* any circulation decomposes into fundamental cycles formed by an off-tree edge plus the tree path between its endpoints
* evaluate candidate cycles (p(e \oplus T[v,u])) and pick the best ratio among them; with constant probability one is a constant-factor approximation. 

The almost-linear result comes from maintaining these structures dynamically across many slowly changing instances (stable gradients/lengths), using a recursive hierarchy and dynamic spanners. 

### Deterministic derandomization framework (Phase II)

The derandomization framework in arXiv:2309.16629 describes deterministic replacements for the randomized sampling in the original min-ratio cycle framework (arXiv:2203.00671). For our integration we treat determinism as a first-class mode:

* **Vertex sparsification:** deterministic hierarchical decompositions replace random tree sampling so the vertex-reduction hierarchy is reproducible.
* **Edge sparsification:** deterministic dynamic spanners replace randomized embeddings and ensure stable update triggers.
* **Cycle selection:** min-ratio cycle evaluation uses deterministic low-stretch tree construction and stable tie-breaking across edges.

These notes are captured in `docs/derandomization.md`, which tracks the key paper comparisons and highlights the deterministic invariants we maintain in the IPM loop. 

### Solver modes and integration flags

The public solver entry points expose a `use_ipm` flag (to force IPM vs classic
SSP) and a `deterministic` flag (to disable randomized sampling in the
min-ratio cycle oracle). To make integration/debugging easier, solver outputs
carry a `solver_mode` label (`ipm`, `ipm_scaled`, `classic`, or
`classic_fallback`) so downstream tooling can log when the IPM loop or scaling
pipeline was used.

## Architecture

### Repository layout

```
almo-mcf/
  pyproject.toml
  README.md
  LICENSE
  python/
    almo_mcf/
      __init__.py
      nx.py              # NetworkX adapters
      typing.py
      _version.py
  rust/
    Cargo.toml           # workspace
    crates/
      almo-mcf-core/
        Cargo.toml
        src/
          lib.rs
          graph/
          ipm/
          min_ratio/
          trees/
          spanner/
          rounding/
          numerics/
      almo-mcf-py/
        Cargo.toml       # pyo3 bindings
        src/lib.rs
  tests/
    test_correctness_small.py
    test_against_networkx.py
    test_rounding_integrality.py
    test_regression_known_instances.py
  benches/
    bench_ipm.rs         # criterion benches in Rust
    bench_python.py      # optional pytest-benchmark
  .github/
    workflows/
      ci.yml
      wheels.yml
      release.yml
```

### Public API (Python)

Expose a NetworkX-friendly interface:

```python
import networkx as nx
from almo_mcf import min_cost_flow, min_cost_flow_cost

flow_dict = min_cost_flow(G)              # returns dict[u][v] = flow
cost = min_cost_flow_cost(G, flow_dict)
```

Supported NetworkX conventions:

* `G` is `nx.DiGraph`
* node attribute: `demand` (int)
* edge attributes:

  * `capacity` (upper; int, default = +inf not allowed for this solver; require explicit bound)
  * `lower_capacity` (int, default 0)
  * `weight` or `cost` (int; prefer `weight` to match NetworkX cost-flow)
    Return format must match NetworkX’s `min_cost_flow` output shape.

Also expose a lower-level API for performance users that accepts contiguous arrays (nodes remapped to `0..n-1`).

### Rust core API

Core crate exports:

```rust
pub struct McfProblem { ... }     // directed, lower/upper, costs, demands
pub struct McfSolution { pub flow: Vec<i64>, pub cost: i128 }

pub fn min_cost_flow_exact(p: &McfProblem, opts: &McfOptions) -> Result<McfSolution>;
```

`McfOptions`:

* `seed: u64`
* `time_limit_ms: Option<u64>`
* `tolerance: f64` (controls high-accuracy phase before final rounding)
* `max_iters: usize`
* `strategy: Strategy`:

  * `FullDynamic` (target paper DS)
  * `PeriodicRebuild { rebuild_every: usize }` (fallback)
* `threads: usize` (rayon)

### Implementation map (current code)

The following files anchor the design above to the current implementation:

* IPM driver + option handling: `rust/crates/almo-mcf-core/src/ipm/mod.rs`, `rust/crates/almo-mcf-core/src/lib.rs`.
* Potential/gradient/lengths: `rust/crates/almo-mcf-core/src/ipm/potential.rs`,
  `rust/crates/almo-mcf-core/src/numerics/barrier.rs`.
* Min-ratio oracle + low-stretch tree reduction: `rust/crates/almo-mcf-core/src/min_ratio/mod.rs`,
  `rust/crates/almo-mcf-core/src/trees/mod.rs`.
* Dynamic spanner/tree-chain scaffolding: `rust/crates/almo-mcf-core/src/spanner/`,
  `rust/crates/almo-mcf-core/src/trees/dynamic/`.
* Rounding to integral optimum: `rust/crates/almo-mcf-core/src/rounding/mod.rs`.
* Python NetworkX adapter + scaling: `python/almo_mcf/nx.py`.
* Extensions/reductions (convex costs, max-flow, etc.): `python/almo_mcf/extensions.py`.

## Data representation and invariants

### Graph storage

Use an adjacency + edge-list hybrid:

* Edge list arrays (SoA):

  * `tail[e], head[e] : u32`
  * `lower[e], upper[e] : i64`
  * `cost[e] : i64`
  * `flow[e] : f64` (interior iterates use floating)
* Maintain reverse-edge index for residual/updates only if needed for feasibility repairs. The IPM itself works on primal variables, but you will still need feasibility projection utilities.

Keep a mapping layer:

* Python nodes → contiguous ids (`u32`)
* Python edge key `(u,v,k)` (MultiDiGraph not supported initially; reject) → edge id

### Feasibility

You must start from a strictly feasible flow (f) with:

* flow conservation: (Bf = b)
* strict bounds: (u^- + \epsilon < f < u^+ - \epsilon)

Implementation requirement:

* implement a robust “strictly feasible initializer”:

  1. reduce lower bounds by setting (f := u^-), adjust node imbalances accordingly
  2. add a super-source/super-sink circulation construction to find any feasible flow within remaining capacities
  3. “push inside” the box by a small margin: for each edge, clamp away from bounds by (\epsilon) while preserving feasibility using a circulation repair (solve a small auxiliary flow).
     If strict feasibility fails, return a clear error (`Infeasible`).

This initializer is not optional; without strict interior iterates the barrier lengths blow up.

## IPM module spec (Rust)

### Numerics policy

* Use `f64` for iterates and gradients/lengths.
* Use `i128` for final cost accumulation to avoid overflow when summing (c_e f_e).
* Centralize all barrier/gradient computations in `numerics/barrier.rs`.
* Implement guarded pow:

  * compute ((x)^{-1-\alpha}) as `exp(-(1+alpha)*ln(x))`
  * clamp `x` below by `x_min = exp(-k * ln(m*U))` consistent with the paper’s boundedness rationale. 
* Reject instances whose implied `m*U` would make `alpha` ill-conditioned in `f64` (documented limits).

### Per-iteration steps

For iteration `t`:

1. Compute `phi = Φ(f)`, terminate if `phi <= threshold` (paper discusses terminating once the cost gap proxy is tiny and then rounding). 

2. Compute:

   * `g = ∇Φ(f)` (vector in R^E)
   * `ell[e]` (lengths, positive) 

3. Call min-ratio cycle oracle:

   * input: undirected view of the current instance, with:

     * per-undirected-edge length = `ell[e]` (symmetric)
     * per-undirected-edge gradient contribution flips sign with traversal direction, as the paper notes 
   * output: an implicitly represented cycle/circulation `Δ`:

     * as a list of directed edge traversals with signed amounts (unit circulation), or
     * as a small set of off-tree edges + tree paths (preferred)

4. Scale `Δ` so `||LΔ||_1 = κ/50` where `κ` is derived from the achieved ratio estimate. Follow the paper’s scaling convention. 

5. Line-search (must exist even if the paper uses a fixed scaling):

   * choose step size `η` so that `f + ηΔ` stays strictly within `(lower+eps, upper-eps)`
   * accept if `Φ` decreases; otherwise backtrack (geometric factor 0.5)
   * cap backtracking steps

6. Update `f := f + ηΔ`

7. Update dynamic oracle structures:

   * apply gradient/length updates to the dynamic min-ratio DS
   * track per-edge “significant flow accumulation” flags (the paper’s DS supports identifying edges that accumulated significant flow). 

### Termination and rounding to exact integral optimum

When the potential indicates tiny suboptimality (paper indicates (c^\top f - F^* \le (mU)^{-10}) at a threshold, then “standard techniques let us round to an exact optimal flow”). 

Implement rounding:

1. Convert the nearly-optimal fractional flow into a residual instance with polynomially bounded costs/capacities via cost-scaling/capacity-scaling reductions (the paper includes reductions in its appendices; you should implement them directly rather than rely on external solvers). 
2. Run a final exact correction:

   * solve a min-cost circulation on the residual with small capacities to eliminate fractional parts
   * enforce integrality by canceling cycles in the residual graph until all edge flows are integral
3. Validate optimality:

   * check feasibility and integrality
   * compute cost and cross-check with an exact reference on small/medium tests (see test plan)

## Min-ratio cycle oracle spec

You need two implementations:

### A. Periodic rebuild oracle (correctness-first fallback)

This is the “static warm-up” approach lifted from the paper’s Section 2.2 discussion:

1. Build `s` independent probabilistic low-stretch spanning trees with respect to current lengths `ell` (paper cites classic constructions and uses this as the warm-up). 

   * In practice:

     * implement a randomized tree builder using:

       * repeated low-diameter decomposition + Kruskal-like sampling, or
       * integrate an existing low-stretch tree algorithm if you can keep it auditable and deterministic under seed
   * You must store:

     * parent array
     * depth
     * prefix sums of tree lengths for path length queries
     * LCA structure (binary lifting) for fast path extraction

2. For each tree `T`:

   * For each non-tree edge `e=(u,v)`:

     * construct fundamental cycle = edge `e` plus tree path `T[v,u]` (paper uses `e ⊕ T[v,u]`). 
     * compute:

       * numerator `g^T p(cycle)` (signed along traversal)
       * denominator `||L p(cycle)||_1` (sum of lengths along traversed directed edges)
     * take ratio

3. Pick cycle with minimum ratio (most negative). Return it as `Δ` (unit circulation).

4. Rebuild schedule:

   * default: rebuild every `R` IPM iterations (configurable)
   * early rebuild if a monitored stability condition trips:

     * too many edges changed length by > factor `τ`
     * or total variation in gradients exceeds a budget

This oracle will be slower (potentially (O(m \log n)) per rebuild plus scanning non-tree edges), but it is straightforward and directly matches the paper’s warm-up decomposition logic. 

### B. Full dynamic oracle (paper-grade performance path)

Implement the dynamic data structure sketched in the paper overview:

* maintain a collection of (m^{o(1)}) spanning trees
* return an (m^{o(1)})-approximate min-ratio cycle implicitly as “off-tree edges + tree paths”
* support updates to edges and to `g_e`, `ell_e`
* maintain dynamic spanners with explicit embeddings under edge updates and vertex splits
* handle adaptivity via the “stable” adversary model (“Hidden Stable-Flow Chasing”, Theorem 6.2 in the paper overview). 

Concrete implementation requirements:

1. **Tree-chain hierarchy**

   * levels `0..L`
   * each level maintains:

     * a set of trees (or tree distributions)
     * an edge sparsifier/spanner for non-tree edges
     * a mapping from original edges to embedded paths at that level

2. **Dynamic spanner with explicit embeddings**

   * maintain subgraph `H` with ~`O(n polylog n)` edges
   * for each removed/original edge, store its embedding path in `H`
   * operations:

     * insert/delete edge
     * vertex split (needed by the paper’s vertex reduction machinery)
     * query embedding path length and explicit edge list
       The paper explicitly highlights this spanner-with-embeddings as a key contribution and needed for edge reduction. 

3. **Update budgeting and rebuilding**

   * each level tracks an “instability budget”
   * when exceeded, rebuild that level from the next coarser representation (paper has dedicated rebuilding sections)
   * ensure rebuild work amortizes across many cheap updates

4. **Cycle extraction**

   * implement fast path extraction on maintained trees
   * output cycle as:

     * list of (edge_id, signed_amount)
     * with at most `polylog(n)` segments

5. **Determinism and seeds**

   * all randomized components must be seeded from `McfOptions.seed`
   * expose `--seed` in CLI/test harness
   * record seeds that cause failures as regression fixtures

This is large, but it is still “just engineering” if you keep every piece modular and testable in isolation.

## Parallelism and SIMD

### Rayon parallelism (high ROI)

Use `rayon` in these hotspots:

1. **Cycle scoring** (fallback oracle and dynamic oracle leaf computations)

   * scoring `m - (n-1)` off-tree edges is embarrassingly parallel
   * parallel reduce to the best ratio

2. **Gradient/length recomputation**

   * `g_e` and `ell_e` are per-edge; compute in parallel

3. **Path queries batch**

   * when scoring many edges, compute their tree-path aggregates in parallel
   * store LCA tables in cache-friendly arrays

### SIMD (targeted, not everywhere)

Use `std::simd` (or `packed_simd`-style patterns if stable) only where it pays:

* barrier computations over contiguous arrays:

  * compute `(u_plus - f)` and `(f - u_minus)` vectors
  * compute `ln(x)` and `exp(k*ln(x))` are not great SIMD candidates in portable Rust, but you can SIMD the affine preprocessing and clamp steps
* cost accumulation and residual checks: vectorizable integer loops

Do not attempt SIMD inside pointer-heavy graph traversals.

## Python bindings (PyO3 + maturin)

* Bindings crate: `almo-mcf-py` using `pyo3` + `maturin`
* Provide two entry points:

  1. `min_cost_flow_edges(n, tails, heads, lower, upper, cost, demand) -> flow`
  2. `min_cost_flow_nx(G) -> dict` implemented in Python by converting NetworkX to arrays

Memory handling:

* accept NumPy arrays (`PyReadonlyArray1`) for inputs
* return a NumPy array `flow[e]` to Python, then map to dict in adapter

Threading:

* release the GIL during `min_cost_flow_exact`
* keep Python adapter single-threaded

## Test suite specification

### Correctness tests (must pass on CI)

1. **Golden small graphs**

   * hand-constructed instances with known optima:

     * single s-t path
     * parallel edges with different costs
     * negative costs with feasible bounded flows
     * lower bounds that force circulation
   * verify:

     * feasibility
     * integrality
     * exact cost

2. **Property-based randomized tests (small)**

   * generate directed graphs `n=5..30`, `m=O(n^2)` sparse/dense variants
   * random integer costs in `[-10,10]`, capacities in `[0,20]`, random lower bounds
   * generate feasible demands by:

     * pick random supply nodes, random demand nodes, balance totals
     * run a feasibility constructor; skip if infeasible
   * compare against NetworkX `network_simplex` or `min_cost_flow` for exact cost + feasibility on these sizes
   * assert equal optimal cost and that returned flow is feasible/integral

3. **Regression seeds**

   * any failing seed is checked in as a deterministic test vector

### Performance tests (gated / not required for PR merge)

1. **Benchmark categories**

   * transportation grid graphs
   * random Erdos–Renyi
   * min-cost circulation heavy instances
   * bipartite assignment reduction cases

2. Metrics

   * wall time
   * iterations
   * oracle time share
   * rebuild counts
   * max barrier length encountered (numerical health)

Use:

* Rust: `criterion`
* Python: `pytest-benchmark` (optional, nightly only)

### Numerical stability tests

* forced near-saturation cases (flows near `u^-` or `u^+`)
* verify:

  * no NaNs/Infs
  * strict feasibility maintained
  * potential decreases monotonically after accepted steps

## GitHub Actions and packaging

### Build system

* `pyproject.toml`:

  * `build-system`: `maturin>=1.7`
  * `project` metadata, classifiers, Python `>=3.10`
* Use `setuptools-scm` for versioning from tags.

### Workflows

#### 1) `ci.yml` (fast checks on PR)

Jobs:

1. **lint-python**

   * `ruff`, `python -m compileall`
2. **lint-rust**

   * `cargo fmt --check`
   * `cargo clippy --all-targets -- -D warnings`
3. **test-python (source build)**

   * OS: ubuntu-latest
   * python: 3.10, 3.12
   * build extension via `maturin develop`
   * run `pytest -q`
4. **test-rust**

   * `cargo test -q`

#### 2) `wheels.yml` (build artifacts on pushes/tags)

Use `cibuildwheel` or `maturin-action`. Given you’ve been using `cibuildwheel` elsewhere, standardize on it.

Matrix:

* OS: ubuntu-latest, windows-latest, macos-latest
* Python: cp310, cp311, cp312, cp313
* Architectures:

  * macOS: x86_64 + arm64 (separate builds; do not rely on universal2 unless you deliberately choose to)
  * linux: x86_64 (manylinux), optionally aarch64 later
  * windows: AMD64

Key settings:

* Linux: build `manylinux_2_28` (or default manylinux2014) wheels
* `CIBW_TEST_COMMAND`: run a small correctness subset (not heavy randomized)
* `CIBW_BEFORE_TEST`: `pip install -r tests/requirements.txt`

Artifacts:

* upload wheels + sdist

#### 3) `release.yml` (publish to PyPI on tag)

* Trigger: `push` tags `v*`
* Steps:

  1. run `wheels.yml` as reusable workflow or duplicate build
  2. publish with PyPI Trusted Publishing (OIDC)
  3. create GitHub Release with attached artifacts


## Documentation deliverables

* `README.md` must include:

  * problem definition + supported constraints
  * NetworkX example
  * performance notes: `Strategy=PeriodicRebuild` vs `FullDynamic`
  * reproducibility: `seed` control
  * limitations (no MultiDiGraph initially, finite capacities required)

* `docs/` (optional) with:

  * a short “math notes” page explaining:

    * potential function
    * how the cycle oracle maps to `Δ`
    * termination and rounding

## Validation plan (what “done” means)

Your implementation is complete when:

1. On all correctness tests up to `n<=30`, your solver matches NetworkX exact min-cost flow costs and returns integral feasible flows.

2. On medium random instances (`n=1e3..1e4`, `m=1e4..1e5`) the solver runs without numerical failures, and performance is at least competitive with classic baselines in Rust (even if not yet paper-optimal).

3. Wheels build and tests pass on macOS (x86_64 + arm64), Linux manylinux x86_64, Windows AMD64 via GitHub Actions, and `pip install almo-mcf` works end-to-end.


Below is the same specification recast as an **issue-ready, task-by-task engineering checklist**, with **explicit acceptance criteria per module**.
The structure is designed to map cleanly onto GitHub issues, epics, and milestones.

---

# Epic 0 — Project scaffolding and repo hygiene

### Task 0.1 — Create repository and workspace layout

**Description:**
Create the repo `almo-mcf` with the full Python + Rust workspace structure.

**Deliverables:**

* Repo with:

  ```
  almo-mcf/
    pyproject.toml
    README.md
    LICENSE
    python/almo_mcf/
    rust/Cargo.toml
    rust/crates/almo-mcf-core/
    rust/crates/almo-mcf-py/
    tests/
    benches/
    .github/workflows/
  ```

**Acceptance criteria:**

* `cargo build` succeeds in workspace root
* `pip install -e .` succeeds (even before solver is implemented)
* Repo contains stub modules for all planned Rust submodules:

  * `graph`, `ipm`, `min_ratio`, `trees`, `spanner`, `rounding`, `numerics`

---

### Task 0.2 — Build system and versioning

**Description:**
Wire up `maturin`, `setuptools-scm`, and workspace Cargo config.

**Deliverables:**

* `pyproject.toml` with:

  * `maturin>=1.7`
  * Python >= 3.10
  * setuptools-scm versioning
* Root `Cargo.toml` workspace
* Crate manifests for core + Python binding crates

**Acceptance criteria:**

* `maturin develop` builds a stub extension
* `python -c "import almo_mcf"` works
* Version auto-derives from Git tag

---

# Epic 1 — Graph representation and feasibility

### Task 1.1 — Directed graph data structures (Rust)

**Description:**
Implement SoA edge storage and node mappings.

**Deliverables:**

* `McfProblem` struct:

  * tails, heads
  * lower, upper, cost
  * demands
  * edge_count, node_count
* ID mapping utilities

**Acceptance criteria:**

* Unit tests:

  * Construct problem from arrays
  * Validate invariants:

    * ∑ demands = 0
    * lower ≤ upper
    * indices in range

---

### Task 1.2 — Strict feasibility initializer

**Description:**
Implement lower-bound normalization and super-source feasibility flow.

**Deliverables:**

* Lower-bound reduction
* Auxiliary circulation solver for feasibility
* “Push-inside” ε-margin repair

**Acceptance criteria:**

* Given any feasible instance:

  * Returns a strictly interior flow
* Given infeasible instance:

  * Returns `Infeasible` error
* Tests:

  * Feasible random small graphs
  * Forced infeasible cases

---

# Epic 2 — Barrier numerics and potential

### Task 2.1 — Barrier and gradient computation

**Description:**
Implement Φ(f), ∇Φ(f), and length ℓₑ(f).

**Deliverables:**

* `numerics/barrier.rs`

  * `phi(f)`
  * `grad(f)`
  * `lengths(f)`
* α = 1 / (1000 log(mU)) logic
* Guarded pow / log

**Acceptance criteria:**

* For random interior flows:

  * No NaN / Inf
* Unit tests:

  * Finite values for all edges
  * ℓₑ > 0
* Monotonicity sanity:

  * Moving flow toward bounds increases Φ

---

### Task 2.2 — Numerical safety clamps

**Description:**
Clamp barrier arguments and exponentials.

**Deliverables:**

* Minimum x threshold
* Centralized clamp logic

**Acceptance criteria:**

* Stress tests with flows extremely near bounds
* No overflow / underflow exceptions

---

# Epic 3 — Min-ratio cycle oracle (fallback)

### Task 3.1 — Low-stretch tree builder

**Description:**
Implement randomized low-stretch spanning tree generator.

**Deliverables:**

* Tree builder with seed
* Parent, depth, prefix sums
* LCA via binary lifting

**Acceptance criteria:**

* Tree covers all nodes
* Path length queries O(log n)
* Deterministic under seed

---

### Task 3.2 — Fundamental cycle scoring

**Description:**
Evaluate ratios for non-tree edges.

**Deliverables:**

* Cycle extraction (edge + tree path)
* gᵀp(cycle)
* ‖Lp(cycle)‖₁

**Acceptance criteria:**

* Unit tests:

  * Correct numerator and denominator on hand graphs
* For random instances:

  * Always returns a circulation Δ

---

### Task 3.3 — Periodic rebuild oracle

**Description:**
Rebuild trees every R iterations.

**Deliverables:**

* Oracle interface
* Rebuild schedule
* Stability triggers

**Acceptance criteria:**

* Always returns Δ with negative ratio when one exists
* Passes correctness tests for IPM steps

---

# Epic 4 — Interior Point Method core

### Task 4.1 — IPM iteration loop

**Description:**
Implement one IPM iteration.

**Deliverables:**

* Compute Φ, g, ℓ
* Call oracle
* Scale Δ
* Line search
* Flow update

**Acceptance criteria:**

* Φ decreases monotonically on accepted steps
* Flow remains strictly interior
* Tests:

  * Simple networks converge

---

### Task 4.2 — Termination criteria

**Description:**
Detect near-optimal termination.

**Deliverables:**

* Threshold logic
* Iteration cap
* Timeout support

**Acceptance criteria:**

* Stops on small duality proxy
* Stops on max_iters
* Reports reason

---

# Epic 5 — Rounding to exact integral solution

### Task 5.1 — Residual correction instance

**Description:**
Build residual min-cost circulation instance.

**Deliverables:**

* Fractional decomposition
* Residual graph

**Acceptance criteria:**

* Residual instance has polynomial bounds
* Feasible if input near-optimal

---

### Task 5.2 — Integral correction

**Description:**
Cancel fractional cycles.

**Deliverables:**

* Cycle canceler
* Final flow integrality enforcement

**Acceptance criteria:**

* Output flow integral
* Feasible
* Same optimal cost

---

# Epic 6 — Python bindings and NetworkX adapter

### Task 6.1 — PyO3 bindings

**Description:**
Expose core solver.

**Deliverables:**

* `min_cost_flow_edges(...)`
* GIL release
* NumPy I/O

**Acceptance criteria:**

* Called from Python
* Returns flow vector
* Thread-safe

---

### Task 6.2 — NetworkX adapter

**Description:**
Match NetworkX API.

**Deliverables:**

* `min_cost_flow(G)`
* `min_cost_flow_cost(G, flow)`

**Acceptance criteria:**

* Drop-in compatible with `nx.min_cost_flow`
* Supports:

  * demand
  * capacity
  * lower_capacity
  * weight

---

# Epic 7 — Dynamic oracle (paper-grade)

### Task 7.1 — Tree-chain hierarchy

**Description:**
Implement multilevel tree structure.

**Deliverables:**

* Level abstraction
* Rebuild logic
* Instability budgets

**Acceptance criteria:**

* Produces cycles equivalent to fallback oracle
* Deterministic under seed

---

### Task 7.2 — Dynamic spanner with embeddings

**Description:**
Maintain H and embedding paths.

**Deliverables:**

* Insert/delete edges
* Vertex splits
* Path extraction

**Acceptance criteria:**

* Embedding correctness tests
* All paths valid in H

---

### Task 7.3 — Full dynamic oracle

**Description:**
Use hierarchy + spanner to return Δ.

**Deliverables:**

* Oracle interface
* Update hooks
* Approx ratio logic

**Acceptance criteria:**

* Returns valid Δ
* Stable across many IPM iterations
* Matches fallback oracle on small tests

---

# Epic 8 — Parallelism and SIMD

### Task 8.1 — Rayon parallel scoring

**Description:**
Parallelize oracle scoring.

**Deliverables:**

* Rayon map-reduce

**Acceptance criteria:**

* Same result as serial
* Speedup >1.5× on m>50k

---

### Task 8.2 — Barrier vectorization

**Description:**
SIMD preprocess loops.

**Deliverables:**

* std::simd guarded loops

**Acceptance criteria:**

* No correctness regressions
* Measurable speedup

---

# Epic 9 — Test suite

### Task 9.1 — Golden correctness tests

**Description:**
Hand graphs.

**Acceptance criteria:**

* Exact cost match
* Feasible, integral

---

### Task 9.2 — Property-based tests

**Description:**
Random graphs vs NetworkX.

**Acceptance criteria:**

* No mismatches for n≤30
* No solver crashes

---

### Task 9.3 — Regression seeds

**Description:**
Store failing seeds.

**Acceptance criteria:**

* Deterministic replay
* CI stable

---

### Task 9.4 — Numerical stability tests

**Description:**
Near-bound flows.

**Acceptance criteria:**

* No NaN/Inf
* Strict interior preserved

---

# Epic 10 — Performance benchmarks

### Task 10.1 — Rust criterion benches

**Acceptance criteria:**

* Bench suite runs
* Reports iterations, time

---

### Task 10.2 — Python benchmarks

**Acceptance criteria:**

* pytest-benchmark runs
* No regressions

---

# Epic 11 — CI and packaging

### Task 11.1 — CI workflow

**Acceptance criteria:**

* Lint + tests pass on PR

---

### Task 11.2 — Wheel builds

**Acceptance criteria:**

* Wheels for:

  * macOS x86_64 + arm64
  * Linux manylinux x86_64
  * Windows AMD64
* `pip install almo-mcf` works

---

### Task 11.3 — PyPI release

**Acceptance criteria:**

* Tag triggers upload
* Trusted publishing works

---

# Epic 12 — Documentation

### Task 12.1 — README

**Acceptance criteria:**

* NetworkX example
* Constraints documented
* Strategy modes explained

---

### Task 12.2 — Math notes

**Acceptance criteria:**

* Φ, Δ, oracle described
* Rounding explained

---

# Definition of Done (global)

The project is “done” when:

1. All correctness tests up to n ≤ 30 match NetworkX exactly
2. All outputs are integral and feasible
3. IPM converges without numerical failures
4. Wheels build and install cleanly on macOS, Linux, Windows
5. `pip install almo-mcf` + NetworkX adapter works end-to-end
6. Performance is at least competitive with classic solvers on medium graphs
7. Full dynamic oracle passes parity tests with fallback oracle on small graphs

---
