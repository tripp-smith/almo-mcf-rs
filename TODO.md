Task List for Implementing the Full Dynamic Oracle (Achieving Paper Asymptotics)
Implement the full dynamic oracle for min-ratio cycle queries to achieve the almost-linear time guarantees from the Chen et al. (FOCS 2022) paper. This dynamic data structure supports coarse multiplicative approximations (m^{o(1)} factor) for min-ratio cycles in undirected graphs with slowly changing lengths and gradients, under a stable adversary model. It enables amortized m^{o(1)} time per query and update during the interior point method iterations, replacing slower periodic rebuild approaches.
Goal: Develop a dynamic min-ratio cycle finder that maintains a recursive hierarchy of trees and spanners, handles updates to edge lengths and gradients with amortized polylog(n) time, and outputs cycles as implicit representations (off-tree edges plus tree paths) with at most polylog(n) segments, ensuring the overall min-cost flow solver reaches m^{1+o(1)} runtime on graphs with m edges.
Dependencies: Require a working static min-ratio cycle oracle (using probabilistic low-stretch spanning trees and fundamental cycle scoring) for comparison testing, and integration hooks into the interior point method for passing updated lengths and gradients.
Overall Acceptance Criteria:
	•	Produce a valid circulation Δ with m^{o(1)}-approximate min-ratio guarantee.
	•	Sustain performance over at least 1000 interior point method iterations on graphs with 1000 nodes without disproportionate rebuild times.
	•	Match or exceed the quality of static oracle results on small test graphs.
	•	Demonstrate amortized query times in benchmarks, aiming for less than m log n per operation after initial setup.

Task 7.1: Implement Tree-Chain Hierarchy Basics
Implement a multilevel hierarchy with levels from 0 to L (where L is approximately log log n or another small o(1) depth) to enable recursive edge reduction. Structure each level to hold multiple spanning trees (or distributions over trees) and to sparsify non-tree edges using spanners, facilitating efficient min-ratio queries across changing graph conditions.
Status: ✅ Completed
Status: ✅ Completed
Status: ✅ Completed
Status: ✅ Completed
Status: ✅ Completed
Status: ✅ Completed
Status: ✅ Completed
Status: ✅ Completed
Status: ✅ Completed
Status: ✅ Completed
Status: ✅ Completed
Status: ✅ Completed
Status: ✅ Completed
Status: ✅ Completed
Status: ✅ Completed
Status: ✅ Completed
Effort: Medium. Dependencies: Use existing low-stretch spanning tree construction and lowest common ancestor query mechanisms. Deliverables:
	•	Define a HierarchyLevel structure that includes a vector of trees (each with parent arrays, depths, prefix sums for path lengths, and lowest common ancestor tables), a spanner subgraph, and mappings for embedding original edges into paths within the spanner.
	•	Create a hierarchy initialization function that sets up L levels, beginning with the finest level (full graph) and progressing to coarser ones.
	•	Develop recursive building logic that constructs each level i based on embeddings from the next coarser level i+1.
	•	Add per-level tracking for instability budgets, using counters to monitor accumulated changes in lengths or gradients against a threshold τ set to m^{o(1)}.
Acceptance Criteria:
	•	Construct the hierarchy successfully on small graphs with 10 to 100 nodes.
	•	Ensure the number of levels L equals O(log log n + o(1)).
Status: ✅ Completed
Status: ✅ Completed
Status: ✅ Completed
	•	Verify that path lengths in trees maintain a polylog(n) stretch factor.
	•	Limit total edges across all levels to O(m polylog n).
	•	Implement tests: Unit tests for hierarchy construction on toy graphs, verifying tree connectivity and stretch; integration tests simulating level builds from coarser to finer, checking embedding consistency.

Task 7.1.1: Add Recursive Edge Reduction
Add functionality to reduce edges recursively by mapping each edge at a given level i to an approximate path at the coarser level i+1, using spanner embeddings to preserve distances within a polylog(n) factor.
Effort: Low. Deliverables:
	•	Create a function that embeds an edge e at level i into an explicit path (sequence of edges) at level i+1.
	•	Maintain detailed embedding representations as vectors of (edge identifier, direction) for quick aggregation of lengths and ratios.
Acceptance Criteria:
	•	Confirm embeddings preserve lengths with a polylog(n) stretch.
	•	Ensure embedded paths for random edges have lengths at most log n.
	•	Implement tests: Unit tests on simple graphs (e.g., chains or cycles) to validate embedding paths connect endpoints correctly and approximate original distances; property-based tests generating random edges and checking stretch bounds.

Task 7.1.2: Implement Instability Budgeting and Rebuild Triggers
Implement mechanisms to track instability at each hierarchy level (e.g., counting significant changes in lengths or gradients) and to trigger rebuilds when a budget is exceeded, ensuring amortized efficiency.
Effort: Low. Deliverables:
	•	Set budget parameters with τ equal to m^{1/10} or a configurable o(1) exponent.
	•	Develop rebuild triggering logic that, upon exceeding the budget, reconstructs the level using data from the next coarser level.
	•	Incorporate amortization to distribute O(m polylog n) rebuild work over at least τ updates.
Acceptance Criteria:
	•	Simulate sequences of 100 updates and confirm rebuilds occur only after the budget threshold.
	•	Verify rebuilds reset instability without losing data integrity.
	•	Implement tests: Simulation tests applying controlled update sequences and asserting rebuild timing; stress tests with varying τ values to confirm amortization holds without excessive rebuild frequency.

Task 7.2: Implement Dynamic Spanner with Explicit Embeddings
Implement a dynamic spanner that maintains a sparse subgraph H with O(n polylog n) edges, approximating all distances in the original graph, while providing explicit path embeddings for non-spanner edges. Support operations under a stable adversary model to handle gradual changes.
Effort: High. Dependencies: Integrate with the hierarchy from Task 7.1. Deliverables:
Status: ✅ Completed
Status: ✅ Completed
	•	Define a DynamicSpanner structure with adjacency representations for H and a mapping (e.g., hash map) from original edge identifiers to vectors of (embedded edge identifier, sign).
	•	Support operations: Insert or delete edges (updating H and affected embeddings), split vertices (creating duplicates and rerouting), update lengths or gradients on edges (propagating if the change is significant), and query explicit paths with aggregated lengths or ratios.
	•	Adapt standard dynamic spanner algorithms to ensure explicit paths are maintained.
	•	Incorporate stable flow chasing to manage polylog updates with high probability, aligning with the paper’s hidden stable-flow model.
Acceptance Criteria:
	•	Maintain the spanner property where distances in H are at most polylog(n) times those in the original graph.
	•	Achieve amortized polylog(n) time for all operations.
	•	Limit H to O(n log² n) edges.
	•	Implement tests: Stress tests with 1000 operations (inserts, deletes, splits) on 1000-node graphs, verifying embedding validity (paths connect correctly without cycles); property-based tests confirming distance approximations hold post-updates.

Task 7.2.1: Add Vertex Reduction Machinery
Add support for vertex operations required in reductions, such as splitting a vertex into multiples and adjusting incident structures accordingly.
Effort: Medium. Deliverables:
	•	Create vertex split functionality that duplicates a vertex, repartitions edges, and updates embeddings in a single atomic step.
	•	Propagate these changes upward through hierarchy levels.
Acceptance Criteria:
	•	Ensure all paths remain connected after splits.
	•	Implement tests: Specific tests on linear chain graphs, splitting mid-vertices and validating adjusted embeddings; integration tests checking propagation to higher levels without breaking connectivity.

Task 7.2.2: Implement Update Propagation for Lengths and Gradients
Implement propagation of updates to edge lengths and gradients from the interior point method, applying changes only when significant to preserve stability.
Effort: Medium. Deliverables:
	•	Develop a batch update function that processes a vector of (edge, new length, new gradient) changes, updates the spanner if necessary, and increments instability flags.
	•	Include stability checks assuming changes derive from a stable underlying flow, per the paper’s model.
Acceptance Criteria:
	•	Handle minor changes without triggering rebuilds.
	•	Flag large changes to increase the instability budget.
	•	Ensure updated ratios align with manual calculations.
	•	Implement tests: Unit tests on update batches, verifying propagation and ratio parity; simulation tests mimicking interior point method sequences to confirm stability under gradual changes.

Task 7.3: Implement Full Dynamic Min-Ratio Cycle Query
Implement the query mechanism that traverses the hierarchy to compute an m^{o(1)}-approximate min-ratio cycle, outputting it in an implicit form suitable for the interior point method.
Effort: High. Dependencies: Complete Tasks 7.1 through 7.2. Deliverables:
	•	Create a query function that takes current gradients and lengths, then outputs a circulation Δ as a vector of (edge identifier, sign, amount) for a unit circulation.
	•	Traverse from the coarsest level downward, using embeddings to identify candidate cycles recursively.
	•	Extract cycles using tree paths combined with off-tree or embedded edges, computing the numerator (gradient dot product) and denominator (length norm).
	•	Select the best ratio from O(log n) candidates across trees per level.
	•	Integrate as a configurable strategy in the solver, with fallback handling for rare rebuild failures.
Acceptance Criteria:
	•	Return a Δ with negative ratio if one exists, or certify optimality otherwise.
	•	Achieve approximation within m^{0.1} of the true minimum on test instances.
	•	Attain O(polylog n) amortized query time.
	•	Limit output to polylog n segments.
	•	Implement tests: Correctness tests on synthetic graphs with known min-ratios; benchmark tests comparing query times to static approaches; approximation quality tests measuring ratio gaps.

Task 7.3.1: Add Fast Path Extraction and Aggregation
Add efficient extraction and summation of gradients and lengths along paths in the structure.
Effort: Low. Deliverables:
	•	Develop a path aggregator that, given a path vector, computes signed gradient sums and length totals.
	•	Use cached prefix sums on trees for constant-time aggregates on tree paths, aided by lowest common ancestor queries.
Acceptance Criteria:
	•	Produce correct sums on predefined paths.
	•	Support parallel computation over multiple candidates.
	•	Implement tests: Unit tests on hand-crafted paths verifying sums; performance tests ensuring constant-time tree path aggregation.

Task 7.3.2: Implement Adaptivity Handling (Stable Adversary)
Implement protections for adaptive queries under the stable model from the paper (Theorem 6.2), using randomization to ensure high-probability success.
Effort: Medium. Deliverables:
	•	Add randomization in spanner and tree constructions to counter adaptivity.
	•	Track and flag edges with significant accumulated flow for coordination with the interior point method.
Acceptance Criteria:
	•	Succeed with probability at least 1 - 1/poly(m) on adversarial sequences.
	•	Implement tests: Regression tests simulating interior point method-like adaptive updates, asserting low failure rates; probabilistic tests running multiple seeded executions to verify stability.

Task 7.4: Integration and End-to-End Testing
Integrate the dynamic oracle into the overall min-cost flow solver and conduct comprehensive testing.
Effort: Low. Dependencies: All prior tasks in this list. Deliverables:
	•	Configure the solver to use the dynamic oracle as an option.
	•	Add logging for metrics like rebuild counts and instability levels.
	•	Extend benchmarks to compare dynamic versus static performance on graphs with 1000 to 10,000 nodes.
Acceptance Criteria:
	•	Solve complete min-cost flow problems faster than static methods on medium-sized graphs.
	•	Avoid any losses in numerical accuracy or correctness compared to established baselines.
	•	Ensure continuous integration passes with the dynamic mode activated.
	•	Implement tests: End-to-end tests on full flow instances, verifying speedups and cost optimality; regression tests confirming no divergences from static oracle results.
