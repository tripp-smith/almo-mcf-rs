# Engineering Task List (Items 8–48)

This list captures the remaining engineering tasks (8–48) in a concise, issue-ready checklist.

8. [x] Implement lower-bound normalization and super-source feasibility flow for the strict feasibility initializer.
9. [x] Add auxiliary circulation solver for feasibility and ε-margin push-inside repair.
10. [x] Test the initializer on feasible random small graphs to return strictly interior flows and on infeasible instances to return Infeasible errors.
11. [x] Implement a randomized low-stretch spanning tree generator with seed, including parent, depth, prefix sums, and LCA via binary lifting.
12. [x] Verify the tree covers all nodes, path length queries are O(log n), and it is deterministic under seed.
13. [x] Implement fundamental cycle extraction as edge plus tree path and compute numerator g^T p(cycle) and denominator ||L p(cycle)||_1.
14. [x] Add unit tests for correct numerator and denominator on hand graphs and ensure it always returns a circulation Δ on random instances.
15. [x] Implement the periodic rebuild oracle interface, rebuild schedule every R iterations, and stability triggers.
16. [x] Test that the oracle returns Δ with negative ratio when one exists and passes correctness tests for IPM steps.
17. [x] Implement the IPM iteration loop to compute Φ, g, ℓ, call the oracle, scale Δ, perform line search, and update flow.
18. [x] Ensure Φ decreases monotonically on accepted steps and flow remains strictly interior.
19. [x] Test that simple networks converge.
20. [x] Add termination criteria based on threshold, iteration cap, and timeout support, reporting the reason for stopping.
21. [x] Build the residual min-cost circulation instance from fractional decomposition and residual graph.
22. [x] Verify the residual instance has polynomial bounds and is feasible if input is near-optimal.
23. [x] Implement the cycle canceler for integral correction and final flow integrality enforcement.
24. [x] Ensure output flow is integral, feasible, and has the same optimal cost.
25. [x] Verify it can be called from Python, returns flow vector, and is thread-safe.
26. [x] Implement the NetworkX adapter to match the API, including min_cost_flow(G) and min_cost_flow_cost(G, flow).
27. [x] Add graph conversion from nx.DiGraph to Rust format, rejection of MultiDiGraph, and enforcement of finite capacities.
28. [x] Support demand, capacity, lower_capacity, and weight attributes.
29. [x] Implement multilevel tree structure with level abstraction, rebuild logic, and instability budgets.
30. [x] Ensure it produces cycles equivalent to the fallback oracle and is deterministic under seed.
31. [x] Maintain the subgraph H with embeddings, supporting insert/delete edges, vertex splits, and path extraction.
32. [x] Add tests for embedding correctness and validity of all paths in H.
33. [x] Implement the full dynamic oracle using hierarchy and spanner to return Δ, with update hooks and approx ratio logic.
34. [x] Verify it returns valid Δ, is stable across IPM iterations, and matches fallback on small tests.
35. [x] Parallelize oracle scoring using Rayon map-reduce for cycle scoring, gradient computation, and path queries.
36. Ensure same results as serial and achieve speedup greater than 1.5x on m greater than 50k.
37. Add SIMD using std::simd for barrier preprocess loops and cost accumulation.
38. Verify no correctness regressions and measurable speedup.
39. [x] Implement golden correctness tests with hand-constructed instances for single paths, parallel edges, negative costs, and lower bounds.
40. [x] Verify exact cost match, feasibility, and integrality.
41. [x] Add property-based randomized tests for small graphs versus NetworkX, ensuring no mismatches for n≤30 and no crashes.
42. [x] Store failing seeds as deterministic regression tests and ensure CI stability.
43. [x] Implement numerical stability tests for near-bound flows, confirming no NaN/Inf and strict interior preserved.
44. Populate Rust Criterion benchmarks with metrics for iterations and time.
45. Add Python pytest-benchmark and ensure no regressions.
46. Verify pip install almo-mcf works.
47. Add optional docs/ folder with math notes explaining potential function, cycle oracle mapping to Δ, and termination with rounding.
48. Validate the complete implementation by ensuring all correctness tests up to n≤30 match NetworkX exactly, outputs are integral and feasible, IPM converges without numerical failures, wheels build and install cleanly on multiple platforms, pip install almo-mcf plus NetworkX adapter works end-to-end, performance is competitive on medium graphs, and full dynamic oracle passes parity tests with fallback on small graphs.
