### 1. Implement the Potential-Reduction IPM Framework (Core Driver from Paper 1, Section 4)
This is the main loop for potential reduction via min-ratio cycle queries and flow updates.

- [x] Define the potential function \(\Phi(f)\) in code, including the log term for cost and power barriers for capacities (use parameters like \(\alpha = 1/(1000 \log(mU))\)).
- [x] Implement gradient \(g(f)\) and length \(\ell(f)\) computations for a given flow \(f\).
- [x] Add initial feasible flow finder (e.g., zero flow or simple augmentation).
- [x] Implement binary search for optimal cost \(F^*\).
- [x] Code the main IPM iteration: Approximate min-ratio cycle query, circulation extraction, step size \(\eta\) computation, and flow update.
- [x] Add convergence check (potential reduction by \(\Omega(\kappa^2)\), with \(\kappa \geq m^{-o(1)}\)).
- [x] Integrate with existing preprocessing (demands, capacities) and test on small graphs for correctness.

### 2. Implement Capacity and Cost Scaling (From Paper 1, Appendix C)
Reduces inputs to poly bounds for IPM applicability.

- [x] Implement cost scaling (Algorithm 9): Binary search cost bounds, solve rounded instances using the IPM.
- [x] Implement capacity scaling (Algorithm 10): Binary search bottlenecks, compute unit-capacity min-cost circulations.
- [x] Ensure each scaling round calls the core IPM solver.
- [x] Handle polynomial bounds on \(U\) and \(C\) (e.g., via big integers in Rust if needed).
- [x] Test scaling independently on graphs with large \(U/C\), verifying reduction to poly(m) bounds.
- [x] Integrate as a wrapper around the main solver in the Python API.

### 3. Implement Low-Stretch Decomposition and Forests (From Papers 1 & 2, Sections 6)
Handles graph decomposition; start with randomized (Paper 1), then derandomize (Paper 2).

- [x] Implement basic low-stretch spanning tree (LSST) construction with stretch \(O(\gamma_{LSST} \log^4 n)\) (randomized sampling).
- [x] Add forest operations: Promote roots, delete edges, compute stretch overestimates.
- [x] Implement multiplicative-weight sampling for fixed circulations (Lemma 6.6, sample \(B = O(\log n)\) trees).
- [x] For derandomization: Precompute multiple forests (\(s\) forests), implement shifting logic to cycle through them.
- [x] Add lazy update propagation to forests during shifts.
- [x] Test forest stretch and updates on random graphs, measuring amortized times.
- [x] Integrate into hierarchical levels (e.g., recurse on contracted cores).

### 4. Implement Dynamic Decremental Spanner (From Paper 1, Theorem 5.1)
Sparsifies the core graph with embeddings.

- [x] Define spanner parameters: \(O(n)\) edges, congestion/path length bounds, \(O(n^{1/L})\) recourse per batch (\(L = (\log m)^{1/4}\)).
- [x] Implement spanner construction: Build levels, expander decomposition for sparsification.
- [x] Add update operations: Batch insert/delete edges, split vertices, query embeddings.
- [x] Handle decremental updates (e.g., project affected edges, embed into levels).
- [x] Test spanner quality (stretch, size) on decremental graph sequences.
- [x] Ensure compatibility with core-graph contraction (e.g., output sparse \(S(G, F)\)).

### 5. Implement Branching-Tree-Chain and Min-Ratio Cycle Extraction (From Paper 1, Sections 7 & Algorithm 5)
The query mechanism for approx min-ratio cycles.

- [x] Build hierarchical structure: \(d = \Omega(\log_{1/8} n)\) levels, each with LSD forest, core contraction, and spanner.
- [x] Implement cycle extraction: For off-tree edges, form fundamental cycles (tree paths + spanner cycles).
- [x] Compute gradient products and length overestimates for each cycle.
- [x] Select max-ratio cycle (approx min-ratio via \(m^{o(1)}\) factor).
- [x] Add circulation decomposition into \(O(\log n)\) tree-paths + \(m^{o(1)}\) off-tree edges.
- [x] Test end-to-end cycle queries on residual graphs, verifying approximation factors.
- [x] Integrate with IPM loop for repeated queries.

### 6. Implement HSFC Updates and Re-Building Game (From Paper 1, Section 6 & Algorithm 6; Enhanced in Paper 2)
Amortizes updates and rebuilds.

- [x] Define HSFC properties: Circulations \(c(t)\), widths \(w(t)\), stability (doubles only on updated edges).
- [x] Implement lazy gradient/length updates using HSFC guarantees.
- [x] Code re-building game: Track round/fix counts per level, preemptively rebuild on thresholds, handle losses.
- [x] Add failure detection and bottom-up rebuilds (amortized \(O(m^{o(1)}(m + T))\)).
- [x] For derandomization: Apply recursive shifting to propagate updates only to active forests.
- [x] Test amortization: Simulate adversarial updates, measure rebuild frequency.
- [x] Integrate updates into the tree-chain query structure.

### 7. Integration, Optimization, and Testing for Full Almost-Linear Solver
Brings everything together, ensures determinism, and verifies performance.

- [x] Replace successive shortest path with IPM as default solver (toggle via flag).
- [x] Ensure full determinism by using Paper 2's shifting (no random sampling).
- [x] Optimize Rust code for large graphs (e.g., efficient data structures, parallelization if applicable).
- [x] Expand benchmarks: Compare runtimes vs. classical solver on large instances (aim for near-linear scaling).
- [x] Add unit/integration tests for new components (e.g., IPM convergence, derandomized forests).
- [x] Update docs (README, DESIGN_SPEC.md) with usage for the almost-linear mode.
- [x] Publish release on PyPI, fix any remaining TODOs (e.g., Clippy warnings). (Release prep done; publish requires credentials.)

### 8. Execution Plan: Enhance IPM for Almost-Linear Iterations (step-by-step)
This plan operationalizes the expanded checklist into concrete implementation + verification steps.

#### Phase A — Core one-step Newton integration (Paper §4.1, Lemma 4.3)
- [ ] A1. Add `IpmEngine` internal struct in `rust/crates/almo-mcf-core/src/ipm/mod.rs` to hold per-iteration state (`flow`, `potential`, `chain`, `rebuilding_game`, `stats`).
- [ ] A2. Implement `one_step_analysis(&mut self, current_flow: &Flow, potential: f64) -> (FlowUpdate, f64)` in `ipm/mod.rs`.
- [ ] A3. Inside `one_step_analysis`, compute gradient via current barrier model and keep the exact expression notes for `∇φ` in comments.
- [ ] A4. Add private `compute_newton_step(&self, gradients: &[f64]) -> Vec<f64>` that maps cycle embedding output into the Newton direction basis.
- [ ] A5. Query min-ratio cycle through dynamic oracle path and apply leaving rule to compute `θ`.
- [ ] A6. Form `Δ = θ · cycle` and evaluate candidate flow update.
- [ ] A7. Add exact drop assertion check path: `φ(x + Δ) <= φ(x) - 0.5 * ||∇φ(x)||^2 / trace(H)` with numerical tolerance.
- [ ] A8. Wire chain and rebuilding hooks per iteration: `chain.update(t)` then `rebuilding_game.play_round(t, cycle_size, quality_ok)`.
- [ ] A9. Add/update test `test_one_step_potential_drop` with seeded random graphs and strict tolerance.

#### Phase B — Stability window + initialization + rounding (Paper §§4.2–4.3)
- [ ] B1. Add `enforce_stability_bounds(&mut self, x: &mut Flow) -> bool` to clamp slacks to `[ε, 1-ε]`.
- [ ] B2. Derive `ε` from graph size/polynomial bound and centralize in helper.
- [ ] B3. Add `find_initial_point(&self, graph: &Graph) -> (Flow, f64)` using midpoint initialization and demand perturbation repair.
- [ ] B4. Implement geometric-mean fallback `x_e^0 = sqrt(l_e u_e)` when midpoint violates feasibility/stability.
- [ ] B5. Add `check_final_rounding(&self, approx_flow: &Flow) -> ExactFlow` using residual shortest-path repair.
- [ ] B6. In one-step path, guard oversized moves with clamp if `||Δ|| > sqrt(m)`.
- [ ] B7. Add test `test_stability_and_init` (multiple random graphs, 10-step run, stability assertions).

#### Phase C — Iteration bound controls for almost-linear regime (Theorem 4.13)
- [ ] C1. Introduce solver orchestration module (`solver.rs` or equivalent integration in `lib.rs`) and expose dynamic iteration budgeting API.
- [ ] C2. Implement `set_dynamic_max_iters(m, approx_factor)` as `O(sqrt(m) log(m/ε))` style cap.
- [ ] C3. Refactor pipeline loop to stop on gap `< ε` or max iteration exceeded.
- [ ] C4. Add early-stop trigger when potential drop stays below target floor (`1/m^{o(1)}` practical proxy).
- [ ] C5. Add periodic rebuild call every 25 iterations to force chain refresh.
- [ ] C6. Set `McfOptions::approx_factor` default to `0.2` and thread it into all scaling/IPM contexts.
- [ ] C7. Add test `test_iteration_optimization` with large random instances and convergence bound checks.

#### Phase D — Explicit Appendix C scaling reductions
- [ ] D1. In `scaling/`, add `reduce_to_polynomial_costs(costs, U, C)` returning reduced costs + log metadata.
- [ ] D2. Implement cost reduction formula `c' = c / 2^k` style with tracked `k` per phase.
- [ ] D3. Add `reduce_to_polynomial_capacities(capacities, demands)` with phased halving and metadata.
- [ ] D4. Refactor scaled solver entrypoint to call reductions before IPM pipeline.
- [ ] D5. Add `unscale_flow(scaled_flow, log_factors)` and verify integral recovery/overflow safety.
- [ ] D6. Add test `test_scaling_reductions` with very large `U`/`C` and exactness cross-check.

#### Phase E — Telemetry and convergence verification
- [ ] E1. Extend `IpmStats` with `potential_drops`, `newton_step_norms`, `convergence_gap`, `total_iters`.
- [ ] E2. Log drop and norm on every accepted step in one-step function.
- [ ] E3. Add cumulative-drop assertion path for debug/profile builds.
- [ ] E4. Implement `verify_almost_linear_iters(m, stats)` with configurable practical bound.
- [ ] E5. Merge IPM stats with dynamic oracle/chain metrics into public `SolverStats`/`IpmSummary`.
- [ ] E6. Add `test_ipm_telemetry` including JSON export snapshot validation.

#### Phase F — End-to-end pipeline and fallback behavior
- [ ] F1. Implement `run_full_ipm` orchestration: init → iterate (oracle/chain/rebuilding) → rounding → unscale.
- [ ] F2. Add fallback path to SSP when non-convergence (`iters > 2 * max_iters` or persistent no-improvement).
- [ ] F3. Update public `min_cost_flow_edges` path to default enhanced IPM for larger graphs (`m > 100`).
- [ ] F4. Add integration test `integration_enhanced_ipm` for mixed-size and large-scale instances.
- [ ] F5. Add benchmark `ipm_almost_linear` and compare against SSP baseline with reproducible seeds.

#### Phase G — Mandatory validation command sequence (triage before merge)
- [ ] G1. Python package install check: `uv pip install -e . --system`.
- [ ] G2. Python tests: `pytest`.
- [ ] G3. Python tests (quiet): `pytest -q tests/`.
- [ ] G4. Formatting: `cargo fmt --check` (run from `rust/`).
- [ ] G5. Full Rust tests with strict warnings policy (or equivalent `-D warnings` enforcement).
- [ ] G6. Clippy strict: `cargo clippy --workspace -- -D warnings`.
- [ ] G7. Required targeted tests:
  - [ ] `cargo test -q -p almo-mcf-core --test min_ratio`
  - [ ] `cargo test -q -p almo-mcf-core --test min_ratio_queries`
  - [ ] `cargo test -q -p almo-mcf-core --test test_rebuilding_game_rounds`
  - [ ] `cargo test -q -p almo-mcf-core --test test_oracle_rebuilding_integration`
  - [ ] `cargo test -q -p almo-mcf-core --test test_dynamic_embeddings`
  - [ ] `cargo test -q -p almo-mcf-core --test test_derandomized_reproducibility`
  - [ ] `cargo test -q -p almo-mcf-core --test test_amortized_guarantees`

#### Completion gate
- [ ] Mark this section complete only when every box above is green and all paper-linked invariants/tests are passing.
