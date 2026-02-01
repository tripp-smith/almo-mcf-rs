# Phase I

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

- [x] Build and test PyPI package with full IPM
- [x] Publish first release on GitHub (tag v0.1.0, include changelog)
- [x] Expand CI workflows: Add tests for IPM, linting (rustfmt, clippy), and cross-platform builds
- [x] Add CONTRIBUTING.md: Guidelines for issues/PRs
- [x] Monitor for scalability: Test on m=10^5+ graphs, fix any memory/time issues
- [x] Solicit feedback: Add issue templates for bugs/feature requests aligned with paper

# Phase II

### 1. Review and Integrate Derandomization Framework

- [x] Read and annotate the paper (arXiv:2309.16629) for key sections on derandomization, comparing to the original randomized framework in arXiv:2203.00671
- [x] Update DESIGN_SPEC.md to include deterministic variants, highlighting differences in vertex and edge sparsification
- [x] Modify IPM loop to support deterministic cycle finding, ensuring compatibility with existing randomized paths

### 2. Implement Deterministic Vertex Sparsification

- [x] Replace random tree sampling with deterministic hierarchical graph decomposition for vertex reduction
- [x] Implement recursive decomposition algorithm to preserve connectivity for min-ratio cycle detection
- [x] Add support for subpolynomial amortized updates in the decomposition structure
- [x] Test vertex sparsification on small graphs for determinism and correctness against randomized version

### 3. Implement Deterministic Dynamic Spanner for Edge Sparsification

- [ ] Develop dynamic spanner data structure to maintain sparse supergraph embeddings under edge insertions/deletions
- [ ] Ensure spanner preserves approximate distances with subpolynomial stretch
- [ ] Integrate amortized m^{o(1)} time per update for edge changes
- [ ] Verify spanner on dynamic graphs, comparing memory and time to original randomized spanner

### 4. Enhance Dynamic Data Structures for Determinism

- [ ] Adapt tree-chain maintenance to use deterministic low-stretch spanning trees
- [ ] Implement fully dynamic low-stretch spanning tree with subpolynomial average stretch and update time
- [ ] Update circulation routing and min-ratio cycle finder to work with deterministic components
- [ ] Ensure handling of non-oblivious adversaries via deterministic rebuilding strategies

### 5. Update Core Solver and API

- [ ] Wire deterministic path as optional/default in Rust core and Python API (e.g., flag for deterministic vs. randomized)
- [ ] Add support for polynomially bounded edge lengths in dynamic trees
- [ ] Implement max-flow and min-cost flow using the deterministic algorithm, verifying m^{1+o(1)} time bounds theoretically

### 6. Testing and Validation

- [ ] Expand tests to include deterministic-specific cases, ensuring reproducibility (no randomness variance)
- [ ] Add benchmarks comparing deterministic vs. randomized versions on varying graph sizes
- [ ] Validate against original paper's theorems (e.g., main result for exact flows)
- [ ] Test stability guarantees from IPM with deterministic updates

### 7. Documentation and Deployment

- [ ] Update README to document deterministic enhancements, usage flags, and performance claims
- [ ] Expand docs/ with notes on derandomization techniques and pseudocode adaptations
- [ ] Update TODO.md to track integration progress and mark related items
- [ ] Reference the derandomization paper (arXiv:2309.16629) in README.md, adding it to the References section
- [ ] Build and test updated PyPI package with deterministic features
- [ ] Publish release with deterministic support, including changelog highlighting improvements
