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

- [ ] Implement basic low-stretch spanning tree (LSST) construction with stretch \(O(\gamma_{LSST} \log^4 n)\) (randomized sampling).
- [ ] Add forest operations: Promote roots, delete edges, compute stretch overestimates.
- [ ] Implement multiplicative-weight sampling for fixed circulations (Lemma 6.6, sample \(B = O(\log n)\) trees).
- [ ] For derandomization: Precompute multiple forests (\(s\) forests), implement shifting logic to cycle through them.
- [ ] Add lazy update propagation to forests during shifts.
- [ ] Test forest stretch and updates on random graphs, measuring amortized times.
- [ ] Integrate into hierarchical levels (e.g., recurse on contracted cores).

### 4. Implement Dynamic Decremental Spanner (From Paper 1, Theorem 5.1)
Sparsifies the core graph with embeddings.

- [ ] Define spanner parameters: \(O(n)\) edges, congestion/path length bounds, \(O(n^{1/L})\) recourse per batch (\(L = (\log m)^{1/4}\)).
- [ ] Implement spanner construction: Build levels, expander decomposition for sparsification.
- [ ] Add update operations: Batch insert/delete edges, split vertices, query embeddings.
- [ ] Handle decremental updates (e.g., project affected edges, embed into levels).
- [ ] Test spanner quality (stretch, size) on decremental graph sequences.
- [ ] Ensure compatibility with core-graph contraction (e.g., output sparse \(S(G, F)\)).

### 5. Implement Branching-Tree-Chain and Min-Ratio Cycle Extraction (From Paper 1, Sections 7 & Algorithm 5)
The query mechanism for approx min-ratio cycles.

- [ ] Build hierarchical structure: \(d = \Omega(\log_{1/8} n)\) levels, each with LSD forest, core contraction, and spanner.
- [ ] Implement cycle extraction: For off-tree edges, form fundamental cycles (tree paths + spanner cycles).
- [ ] Compute gradient products and length overestimates for each cycle.
- [ ] Select max-ratio cycle (approx min-ratio via \(m^{o(1)}\) factor).
- [ ] Add circulation decomposition into \(O(\log n)\) tree-paths + \(m^{o(1)}\) off-tree edges.
- [ ] Test end-to-end cycle queries on residual graphs, verifying approximation factors.
- [ ] Integrate with IPM loop for repeated queries.

### 6. Implement HSFC Updates and Re-Building Game (From Paper 1, Section 6 & Algorithm 6; Enhanced in Paper 2)
Amortizes updates and rebuilds.

- [ ] Define HSFC properties: Circulations \(c(t)\), widths \(w(t)\), stability (doubles only on updated edges).
- [ ] Implement lazy gradient/length updates using HSFC guarantees.
- [ ] Code re-building game: Track round/fix counts per level, preemptively rebuild on thresholds, handle losses.
- [ ] Add failure detection and bottom-up rebuilds (amortized \(O(m^{o(1)}(m + T))\)).
- [ ] For derandomization: Apply recursive shifting to propagate updates only to active forests.
- [ ] Test amortization: Simulate adversarial updates, measure rebuild frequency.
- [ ] Integrate updates into the tree-chain query structure.

### 7. Integration, Optimization, and Testing for Full Almost-Linear Solver
Brings everything together, ensures determinism, and verifies performance.

- [ ] Replace successive shortest path with IPM as default solver (toggle via flag).
- [ ] Ensure full determinism by using Paper 2's shifting (no random sampling).
- [ ] Optimize Rust code for large graphs (e.g., efficient data structures, parallelization if applicable).
- [ ] Expand benchmarks: Compare runtimes vs. classical solver on large instances (aim for near-linear scaling).
- [ ] Add unit/integration tests for new components (e.g., IPM convergence, derandomized forests).
- [ ] Update docs (README, DESIGN_SPEC.md) with usage for the almost-linear mode.
- [ ] Publish release on PyPI, fix any remaining TODOs (e.g., Clippy warnings).
