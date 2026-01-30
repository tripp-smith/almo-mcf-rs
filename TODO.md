### Phase 1: Foundational Components

**Complete the graph module (rust/crates/almo-mcf-core/src/graph)**
- [ ] Define basic directed graph struct with nodes (demands) and edges (capacity lower/upper, cost, reverse edge ref)
- [ ] Implement node/edge ID mapping and iterators (incoming/outgoing edges per node)
- [ ] Add residual graph view (forward/backward edges based on current flow)
- [ ] Implement incidence matrix builder (sparse B matrix representation if needed for numerics)
- [ ] Add feasibility check: sum demands = 0, no negative cycles on costs alone
- [ ] Add undirected view helper (ignore directions for cycle finding in min-ratio)

**Complete numerics utilities (rust/crates/almo-mcf-core/src/numerics)**
- [ ] Set up high-precision f64 wrapper or use f64 with careful epsilon handling
- [ ] Implement vector operations: dot product, ℓ1 norm, ℓ2 norm, scaled addition
- [ ] Add safe log/exp functions with overflow protection for barrier terms
- [ ] Implement gradient proxy computation (g_e = c_e + μ (1/(u_e - f_e) - 1/f_e) or similar)
- [ ] Add tolerance constants and duality-gap proxy function
- [ ] Write unit tests for numerical stability on small vectors

### Phase 2: Core Algorithm Components

**Implement potential-reduction IPM core (rust/crates/almo-mcf-core/src/ipm)**
- [ ] Define Potential struct with log barrier and cost-gap terms (α = 1/(1000 log(mU)))
- [ ] Implement initial strictly feasible flow finder (use classic successive shortest paths as bootstrap)
- [ ] Write function to compute current gradient vector g(f) and edge lengths ℓ_e(f)
- [ ] Implement main IPM iteration loop skeleton (while gap > tol { find direction; line search; update })
- [ ] Add line-search function (Armijo or simple fraction scaling to stay feasible)
- [ ] Implement termination check (duality gap proxy < ε)
- [ ] Write basic integration test on tiny MCF instance (e.g., 3-node example)

**Implement undirected min-ratio cycle subproblem (rust/crates/almo-mcf-core/src/min_ratio)**
- [ ] Define MinRatioOracle trait or struct that takes g, ℓ and returns approx min-ratio circulation
- [ ] Implement static version: sample low-stretch tree → enumerate fundamental cycles
- [ ] Add cycle evaluation: compute reduced cost g^T Δ and normalized length ||L Δ||_1
- [ ] Implement approximate solver (target m^{o(1)} accuracy, e.g., via top-k cycles)
- [ ] Add warm-start support (cache previous tree or spanner across IPM steps)
- [ ] Write test: verify ratio on hand-crafted negative-ratio cycle

**Implement rounding to exact optimum (rust/crates/almo-mcf-core/src/rounding)**
- [ ] Implement residual graph extractor from fractional flow
- [ ] Add cycle-canceling step on residual graph to reach integral flow (small number of augmentations)
- [ ] Ensure cost monotonicity during rounding (only augment along negative reduced cost cycles)
- [ ] Add final integrality check and cost verification
- [ ] Write test: round a known fractional optimum to integral on small instance

### Phase 3: Dynamic Data Structures

**Implement low-stretch spanning trees (rust/crates/almo-mcf-core/src/trees)**
- [ ] Implement probabilistic low-stretch tree construction (MST + random sampling or ABP style)
- [ ] Add tree path query (LCA + distance in tree metric)
- [ ] Implement fundamental cycle extractor given tree and off-tree edge
- [ ] Add basic dynamic update (edge insertion/deletion with rebuild fallback)
- [ ] Write benchmark on path query time vs graph size

**Implement dynamic spanners and hierarchy (rust/crates/almo-mcf-core/src/spanner)**
- [ ] Implement recursive tree spanner construction (chain decomposition or hierarchy)
- [ ] Add spanner maintenance across small flow changes (stability-aware rebuild)
- [ ] Implement periodic rebuild trigger (every k IPM steps or stretch threshold)
- [ ] Add flow-chasing oracle using spanner paths for cycle finding
- [ ] Write integration test with dummy lengths/gradients

### Phase 4: Integration and API Completion

**Integrate IPM into Rust core solver (rust/crates/almo-mcf-core/src/lib.rs)**
- [ ] Refactor min_cost_flow_exact to accept McfOptions.strategy
- [ ] Implement IPM branch (call ipm solver when strategy = FullDynamic or PeriodicRebuild)
- [ ] Wire min-ratio oracle into IPM direction finding
- [ ] Add fallback to classic successive shortest paths for debugging/small instances
- [ ] Handle infeasible/unbounded errors consistently
- [ ] Update McfSolution to include IPM stats (iterations, final gap)

**Complete Python public API (python/almo_mcf)**
- [ ] Create typing.py with type aliases (FlowDict, Cost, Demand, etc.)
- [ ] Add _version.py with __version__ string
- [ ] Extend nx.py: implement min_cost_flow() returning flow dict
- [ ] Implement min_cost_flow_cost() returning scalar
- [ ] Add low-level array-based API in _core.py (if planned)
- [ ] Write example usage in __init__.py docstring

### Phase 5: Testing, Performance, and Polish

**Expand tests for IPM components (tests/)**
- [ ] Add unit tests for each ipm/min_ratio/rounding/trees/spanner module
- [ ] Update test_regression_known_instances.py to run new IPM solver
- [ ] Extend test_against_networkx.py with IPM mode and assert cost equality
- [ ] Add property-based tests (hypothesis) for feasibility preservation

**Add benchmarks and tuning (benches/)**
- [ ] Update bench_ipm.rs with realistic graph families
- [ ] Add Python benchmark script comparing IPM vs NetworkX
- [ ] Tune parameters (threads, rebuild frequency, α constant) via command-line flags
- [ ] Run and record baseline timings on medium instances

**Update documentation and release prep**
- [ ] Update README.md: mark IPM as enabled, add usage examples, remove "scaffold" warnings
- [ ] Expand docs/math-notes/ with IPM derivation and references
- [ ] Verify package name availability on PyPI (almo-mcf)
- [ ] Add IPM coverage to .github/workflows CI
- [ ] Tag and release v0.1.0 (or next version) once all checks pass

Work through the list top-to-bottom within each phase, and complete phases in order (1 → 5) to minimize blockers. This gives you ~60 granular, checkable tasks that fully realize the almost-linear MCF implementation described in DESIGN_SPEC.md. Good luck — this is an ambitious but exciting project!
