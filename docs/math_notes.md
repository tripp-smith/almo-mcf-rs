# Math notes: potential, cycle oracle, and rounding

This document summarizes the high-level math structures used by the solver: the
interior-point potential, how a min-ratio cycle maps to a circulation update,
and why the termination/rounding steps recover an exact optimum.

## Potential function and barrier structure

We maintain a strictly interior flow `f` with `u^- < f < u^+` and define a
potential that combines a cost-gap proxy with barrier penalties:

* The cost term tracks progress toward optimality.
* The barrier term uses a power barrier (not `-log`) to discourage moves toward
  either capacity bound.

Let `alpha = 1 / (1000 * log(mU))` for `m` edges and capacity scale `U`. The
barrier contributes edge lengths:

```
ℓ_e(f) = (u^+_e - f_e)^(-1-alpha) + (f_e - u^-_e)^(-1-alpha)
```

The gradient `g = ∇Φ(f)` and the lengths `ℓ` drive the next circulation update.
The algorithm ensures all operations remain inside strict interior margins to
avoid singularities in the barrier terms.

## Min-ratio cycle oracle → circulation update

Each IPM iteration seeks a circulation `Δ` (so `BᵀΔ = 0`) that makes the ratio

```
(gᵀΔ) / ||LΔ||₁
```

sufficiently negative, where `L` is the diagonal operator induced by the
lengths `ℓ`. The undirected min-ratio cycle reduction observes that a near-best
`Δ` can be taken to be a simple cycle. The fallback oracle constructs a
low-stretch spanning tree and evaluates fundamental cycles formed by each
non-tree edge plus the tree path between its endpoints. For a candidate cycle
`C`, we compute:

* numerator: `gᵀ p(C)` (signed along the traversal),
* denominator: `||L p(C)||₁` (sum of lengths along the cycle).

The best ratio cycle yields a unit circulation `Δ`. We scale `Δ` to satisfy
`||LΔ||₁ = κ/50` where `κ` is the achieved ratio, then line-search to preserve
strict feasibility and ensure the potential decreases.

## Termination and rounding to exact optimum

When the potential indicates the primal-dual gap proxy is tiny, we transition to
rounding. The rounding step converts the fractional flow into a residual
min-cost circulation instance with polynomially bounded capacities/costs and
then cancels cycles to remove fractional parts. This produces:

* a feasible integral flow,
* the same optimal cost as the fractional optimum (within the threshold).

Together, the termination criteria and rounding ensure the output is an exact
min-cost flow, while preserving the feasibility and integrality guarantees of
the original problem.
