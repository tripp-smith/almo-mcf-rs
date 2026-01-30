### Phase 1: Foundational Components

**Complete the graph module (rust/crates/almo-mcf-core/src/graph)**  
- [x] Define basic directed graph struct with nodes (demands) and edges (capacity lower/upper, cost, reverse edge ref)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/graph/mod.rs or new .rs file (e.g., graph.rs).  
  Verification: Not implemented. Repo browse shows the graph directory exists but content is insufficient/described as scaffolded; no detailed structs for MCF-specific graph in current classic solver.  

- [x] Implement node/edge ID mapping and iterators (incoming/outgoing edges per node)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/graph/mod.rs or graph.rs.  
  Verification: Not implemented. Basic graph ops may exist partially in lib.rs for residual graphs, but no advanced ID mappings or iterators per browse.  

- [x] Add residual graph view (forward/backward edges based on current flow)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/graph/mod.rs or residual.rs.  
  Verification: Partially implemented in lib.rs for classic solver (Bellman-Ford on residuals), but not as a modular view; needs expansion for IPM.  

- [x] Implement incidence matrix builder (sparse B matrix representation if needed for numerics)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/graph/mod.rs or matrix.rs.  
  Verification: Not implemented. No mention of incidence matrix in browsed content.  

- [x] Add feasibility check: sum demands = 0, no negative cycles on costs alone  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/graph/mod.rs or utils.rs.  
  Verification: Not fully implemented. Classic solver in lib.rs handles feasibility via super-source/sink, but no standalone check for negative cycles without flow.  

- [x] Add undirected view helper (ignore directions for cycle finding in min-ratio)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/graph/mod.rs or undirected.rs.  
  Verification: Not implemented. Required for min-ratio but absent in current scaffold.  

**Complete numerics utilities (rust/crates/almo-mcf-core/src/numerics)**  
- [x] Set up high-precision f64 wrapper or use f64 with careful epsilon handling  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/numerics/mod.rs or precision.rs.  
  Verification: Not implemented. Directory has barrier.rs (possibly numerical barrier) and mod.rs, but no precision wrappers per browse.  

- [x] Implement vector operations: dot product, ℓ1 norm, ℓ2 norm, scaled addition  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/numerics/mod.rs or vec_ops.rs.  
  Verification: Not implemented. No vector ops detailed; current solver uses basic numerics in lib.rs.  

- [x] Add safe log/exp functions with overflow protection for barrier terms  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/numerics/barrier.rs.  
  Verification: Not implemented. barrier.rs exists but content insufficient; likely stub without safe math funcs.  

- [x] Implement gradient proxy computation (g_e = c_e + μ (1/(u_e - f_e) - 1/f_e) or similar)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/numerics/mod.rs or gradient.rs.  
  Verification: Not implemented. No gradient computation in browsed numerics.  

- [x] Add tolerance constants and duality-gap proxy function  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/numerics/mod.rs.  
  Verification: Not implemented. Tolerances may be hardcoded in lib.rs, but no proxies.  

- [x] Write unit tests for numerical stability on small vectors  
  Touched Codebase Parts: rust/crates/almo-mcf-core/tests/numerics.rs or similar (but tests are in root/tests, Python-focused).  
  Verification: Not implemented. Tests exist for stability (test_numerical_stability.py), but Rust-side units absent.  

### Phase 2: Core Algorithm Components

**Implement potential-reduction IPM core (rust/crates/almo-mcf-core/src/ipm)**  
- [ ] Define Potential struct with log barrier and cost-gap terms (α = 1/(1000 log(mU)))  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/ipm/mod.rs or potential.rs.  
  Verification: Not implemented. ipm dir has mod.rs but insufficient content; scaffolded per README.  

- [ ] Implement initial strictly feasible flow finder (use classic successive shortest paths as bootstrap)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/ipm/mod.rs or init.rs.  
  Verification: Not implemented. Classic finder in lib.rs, but not integrated for IPM.  

- [ ] Write function to compute current gradient vector g(f) and edge lengths ℓ_e(f)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/ipm/mod.rs or gradient.rs.  
  Verification: Not implemented. No such functions in scaffold.  

- [ ] Implement main IPM iteration loop skeleton (while gap > tol { find direction; line search; update })  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/ipm/mod.rs.  
  Verification: Not implemented. Core loop absent.  

- [ ] Add line-search function (Armijo or simple fraction scaling to stay feasible)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/ipm/mod.rs or search.rs.  
  Verification: Not implemented.  

- [ ] Implement termination check (duality gap proxy < ε)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/ipm/mod.rs.  
  Verification: Not implemented.  

- [ ] Write basic integration test on tiny MCF instance (e.g., 3-node example)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/tests/ipm.rs or root/tests/test_correctness_small.py (expand).  
  Verification: Not implemented. Small tests exist but not for IPM.  

**Implement undirected min-ratio cycle subproblem (rust/crates/almo-mcf-core/src/min_ratio)**  
- [ ] Define MinRatioOracle trait or struct that takes g, ℓ and returns approx min-ratio circulation  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/min_ratio/mod.rs or oracle.rs.  
  Verification: Not implemented. Dir has dynamic.rs and mod.rs, but partial (dynamic oracle stub); no full trait.  

- [ ] Implement static version: sample low-stretch tree → enumerate fundamental cycles  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/min_ratio/static.rs.  
  Verification: Not implemented. No static version.  

- [ ] Add cycle evaluation: compute reduced cost g^T Δ and normalized length ||L Δ||_1  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/min_ratio/mod.rs.  
  Verification: Not implemented.  

- [ ] Implement approximate solver (target m^{o(1)} accuracy, e.g., via top-k cycles)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/min_ratio/mod.rs.  
  Verification: Not implemented.  

- [ ] Add warm-start support (cache previous tree or spanner across IPM steps)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/min_ratio/dynamic.rs.  
  Verification: Not implemented. dynamic.rs exists but stub.  

- [ ] Write test: verify ratio on hand-crafted negative-ratio cycle  
  Touched Codebase Parts: rust/crates/almo-mcf-core/tests/min_ratio.rs.  
  Verification: Not implemented. No such test.  

**Implement rounding to exact optimum (rust/crates/almo-mcf-core/src/rounding)**  
- [ ] Implement residual graph extractor from fractional flow  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/rounding/mod.rs or residual.rs.  
  Verification: Not implemented. Dir exists but insufficient; partial residual in lib.rs.  

- [ ] Add cycle-canceling step on residual graph to reach integral flow (small number of augmentations)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/rounding/mod.rs.  
  Verification: Not implemented.  

- [ ] Ensure cost monotonicity during rounding (only augment along negative reduced cost cycles)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/rounding/mod.rs.  
  Verification: Not implemented.  

- [ ] Add final integrality check and cost verification  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/rounding/mod.rs.  
  Verification: Not implemented.  

- [ ] Write test: round a known fractional optimum to integral on small instance  
  Touched Codebase Parts: rust/crates/almo-mcf-core/tests/rounding.rs.  
  Verification: Not implemented.  

### Phase 3: Dynamic Data Structures

**Implement low-stretch spanning trees (rust/crates/almo-mcf-core/src/trees)**  
- [ ] Implement probabilistic low-stretch tree construction (MST + random sampling or ABP style)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/trees/mod.rs or lsst.rs.  
  Verification: Not implemented. Dir has mod.rs but insufficient content.  

- [ ] Add tree path query (LCA + distance in tree metric)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/trees/mod.rs.  
  Verification: Not implemented.  

- [ ] Implement fundamental cycle extractor given tree and off-tree edge  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/trees/mod.rs.  
  Verification: Not implemented.  

- [ ] Add basic dynamic update (edge insertion/deletion with rebuild fallback)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/trees/dynamic.rs.  
  Verification: Not implemented.  

- [ ] Write benchmark on path query time vs graph size  
  Touched Codebase Parts: rust/crates/almo-mcf-core/benches/trees.rs (benches dir not found at root, likely add to crate benches).  
  Verification: Not implemented. No benches dir.  

**Implement dynamic spanners and hierarchy (rust/crates/almo-mcf-core/src/spanner)**  
- [ ] Implement recursive tree spanner construction (chain decomposition or hierarchy)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/spanner/mod.rs or construct.rs.  
  Verification: Not implemented. Insufficient content.  

- [ ] Add spanner maintenance across small flow changes (stability-aware rebuild)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/spanner/mod.rs.  
  Verification: Not implemented.  

- [ ] Implement periodic rebuild trigger (every k IPM steps or stretch threshold)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/spanner/mod.rs.  
  Verification: Not implemented.  

- [ ] Add flow-chasing oracle using spanner paths for cycle finding  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/spanner/oracle.rs.  
  Verification: Not implemented.  

- [ ] Write integration test with dummy lengths/gradients  
  Touched Codebase Parts: rust/crates/almo-mcf-core/tests/spanner.rs.  
  Verification: Not implemented.  

### Phase 4: Integration and API Completion

**Integrate IPM into Rust core solver (rust/crates/almo-mcf-core/src/lib.rs)**  
- [ ] Refactor min_cost_flow_exact to accept McfOptions.strategy  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/lib.rs.  
  Verification: Not implemented. lib.rs uses classic algorithm; options defined but not used for IPM.  

- [ ] Implement IPM branch (call ipm solver when strategy = FullDynamic or PeriodicRebuild)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/lib.rs.  
  Verification: Not implemented.  

- [ ] Wire min-ratio oracle into IPM direction finding  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/lib.rs.  
  Verification: Not implemented.  

- [ ] Add fallback to classic successive shortest paths for debugging/small instances  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/lib.rs.  
  Verification: Partially implemented (classic is default), but not as fallback.  

- [ ] Handle infeasible/unbounded errors consistently  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/lib.rs.  
  Verification: Partially implemented in classic; needs IPM extension.  

- [ ] Update McfSolution to include IPM stats (iterations, final gap)  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/lib.rs.  
  Verification: Not implemented.  

**Complete Python public API (python/almo_mcf)**  
- [x] Create typing.py with type aliases (FlowDict, Cost, Demand, etc.)  
  Touched Codebase Parts: python/almo_mcf/typing.py (new file).  
  Verification: Not implemented. File does not exist.  

- [x] Add _version.py with __version__ string  
  Touched Codebase Parts: python/almo_mcf/_version.py (new file).  
  Verification: Not implemented. File does not exist.  

- [ ] Extend nx.py: implement min_cost_flow() returning flow dict  
  Touched Codebase Parts: python/almo_mcf/nx.py.  
  Verification: Partially implemented (exists with validation), but needs full flow dict return if incomplete.  

- [ ] Implement min_cost_flow_cost() returning scalar  
  Touched Codebase Parts: python/almo_mcf/nx.py.  
  Verification: Partially implemented; exists but tie to IPM.  

- [ ] Add low-level array-based API in _core.py (if planned)  
  Touched Codebase Parts: python/almo_mcf/_core.py (new or extend).  
  Verification: Not implemented. No _core.py listed.  

- [x] Write example usage in __init__.py docstring  
  Touched Codebase Parts: python/almo_mcf/__init__.py.  
  Verification: Partially implemented (__init__.py exists for smoke), but expand docstring.  

### Phase 5: Testing, Performance, and Polish

**Expand tests for IPM components (tests/)**  
- [ ] Add unit tests for each ipm/min_ratio/rounding/trees/spanner module  
  Touched Codebase Parts: tests/test_ipm.py, test_min_ratio.py, etc. (or Rust tests in crate).  
  Verification: Not implemented. Existing tests cover classic (e.g., test_against_networkx.py).  

- [ ] Update test_regression_known_instances.py to run new IPM solver  
  Touched Codebase Parts: tests/test_regression_known_instances.py (not listed, but regression_seeds.py exists; assume similar).  
  Verification: Not implemented for IPM.  

- [ ] Extend test_against_networkx.py with IPM mode and assert cost equality  
  Touched Codebase Parts: tests/test_against_networkx.py.  
  Verification: Not implemented for IPM.  

- [ ] Add property-based tests (hypothesis) for feasibility preservation  
  Touched Codebase Parts: tests/test_property_randomized.py (exists; extend).  
  Verification: Partially implemented; needs IPM focus.  

**Add benchmarks and tuning (benches/)**  
- [ ] Update bench_ipm.rs with realistic graph families  
  Touched Codebase Parts: rust/crates/almo-mcf-core/benches/bench_ipm.rs (benches dir not found; create).  
  Verification: Not implemented. No benches dir.  

- [ ] Add Python benchmark script comparing IPM vs NetworkX  
  Touched Codebase Parts: tests/test_benchmarks.py (exists; extend).  
  Verification: Not implemented for IPM.  

- [ ] Tune parameters (threads, rebuild frequency, α constant) via command-line flags  
  Touched Codebase Parts: rust/crates/almo-mcf-core/src/lib.rs (McfOptions).  
  Verification: Not implemented. Options exist but not tuned for IPM.  

- [ ] Run and record baseline timings on medium instances  
  Touched Codebase Parts: Documentation or benches output (e.g., README.md).  
  Verification: Not implemented.  

**Update documentation and release prep**  
- [ ] Update README.md: mark IPM as enabled, add usage examples, remove "scaffold" warnings  
  Touched Codebase Parts: README.md.  
  Verification: Not implemented. Current README notes scaffold status.  

- [ ] Expand docs/math-notes/ with IPM derivation and references  
  Touched Codebase Parts: docs/math-notes/ (dir exists; files like ipm.md).  
  Verification: Partially implemented (expanded randomized coverage), but needs full IPM.  

- [ ] Verify package name availability on PyPI (almo-mcf)  
  Touched Codebase Parts: None (external check), but note in README or pyproject.toml.  
  Verification: Not implemented (manual step).  

- [ ] Add IPM coverage to .github/workflows CI  
  Touched Codebase Parts: .github/workflows/ (exists; extend YAML files).  
  Verification: Not implemented. Current CI lacks IPM.  

- [ ] Tag and release v0.1.0 (or next version) once all checks pass  
  Touched Codebase Parts: Git tags/releases.  
  Verification: Not implemented.
