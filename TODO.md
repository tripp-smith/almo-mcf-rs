### 1. Integrate IPM and Min-Ratio Cycle Solver

- [x] Implement the potential function Φ(f) (Eq. 3 in paper) in Rust core, including power barrier x^{-α} with α = 1/(1000 log mU)
- [x] Add gradient g(f) and length ℓ(f) computation to the Rust solver, ensuring bit-complexity bounds (exp(log^{O(1)} m))
- [x] Implement the IPM iteration loop: Reduce Φ by m^{-o(1)} per iteration, terminating when Φ ≤ -200m log mU
- [x] Integrate min-ratio cycle oracle (min_{B^T Δ=0} g^T Δ / ||L Δ||_1) as a subroutine in Rust, with mo(1) approximation support
- [x] Wire IPM as optional/default path in Python API (e.g., flag in min_cost_flow to toggle IPM vs. successive shortest path)
- [x] Add rounding to exact optimal flow when c^T f - F^* ≤ (mU)^{-10}
- [x] Verify IPM stability lemmas (Lemmas 4.9-4.10) in code via unit tests on small graphs

### 2. Implement Dynamic Data Structures

- [x] Implement dynamic spanner with embeddings (Theorem 5.1): Maintain subgraph H with Õ(n) edges, explicit path embeddings of length mo(1), amortized mo(1) changes per update
- [x] Add low-stretch spanning tree (LST) computation with stretch str_{T,ℓ}^e = Õ(1) in expectation
- [x] Build recursive hierarchy: Reduce vertices via partial tree embeddings, edges via spanners
- [x] Implement tree-chain maintenance: Support returning mo(1)-approx min-ratio cycles (union of mo(1) off-tree edges + tree paths)
- [x] Add circulation routing along cycles (pass circulations through tree-chain with length upper bounds)
- [x] Implement approximate min-ratio cycle finder in tree-chain
- [x] Handle non-oblivious adversaries: Integrate rebuilding game (analyze game algorithm, dynamic min-ratio using game)
- [x] Ensure amortized mo(1) time per update/query (insert/delete edge, update g/ℓ, identify high-flow edges)

### 3. Handle Edge Cases and Extensions

- [x] Add capacity/cost scaling reductions to poly(m) bounded U/C
- [x] Implement max-flow reduction to min-cost circulation (add t→s edge)
- [x] Support MultiDiGraph in NetworkX adapter
- [x] Add general convex objectives (edge-separable cost(f) = ∑_e cost_e(f_e), e.g., p-norms, entropy-regularized OT, matrix scaling)
- [x] Implement applications: Bipartite matching, negative cycle detection, vertex connectivity, Gomory-Hu trees, sparsest cuts
- [x] Handle directed acyclic graphs (DAGs) for isotonic regression
- [x] Add error handling for invalid inputs (e.g., sum demands ≠ 0, infinite capacities)

### 4. Optimization, Testing, and Documentation

- [x] Expand pytest suite: Add IPM-specific tests, large-instance regressions, and parity with NetworkX on random graphs
- [x] Implement benchmarks: Compare IPM vs. successive shortest path on varying m/n/U/C
- [x] Tune performance: Optimize for large instances (parallelize tree computations if possible, reduce constants in mo(1))
- [x] Update README: Document IPM usage, examples for extensions, performance claims
- [x] Expand DESIGN_SPEC.md: Detail full integration, with code references
- [x] Update TODO.md: Mark completed items, add any new gaps found during testing
- [x] Add math notes in docs/ for IPM proofs/stability

### 5. Deployment and Maintenance

- [ ] Build and test PyPI package with full IPM
- [ ] Publish first release on GitHub (tag v0.1.0, include changelog)
- [ ] Expand CI workflows: Add tests for IPM, linting (rustfmt, clippy), and cross-platform builds
- [ ] Add CONTRIBUTING.md: Guidelines for issues/PRs
- [ ] Monitor for scalability: Test on m=10^5+ graphs, fix any memory/time issues
- [ ] Solicit feedback: Add issue templates for bugs/feature requests aligned with paper
