# Math notes: IPM potential, cycle oracle, and rounding

This document summarizes the math structure used by the solver: the interior
point method (IPM) potential, how a min-ratio cycle maps to a circulation
update, and why the termination/rounding steps recover an exact optimum.

## Problem setup and notation

We solve a minimum-cost flow instance with:

- directed graph `G = (V, E)` and incidence matrix `B`,
- lower/upper capacities `u^- ≤ f ≤ u^+`,
- node demands `b` with `∑ b_i = 0`,
- linear cost `cᵀ f`.

The feasible region is `{ f ∈ R^m : B f = b, u^- < f < u^+ }`. The IPM iterates
over strictly interior flows to keep barrier terms finite.

## Potential function and barrier structure

We define a potential `Φ(f)` that combines a cost-gap proxy with a smooth
barrier:

```
Φ(f) = α · (cᵀ f - lower_bound) + ∑_e ψ(f_e - u^-_e) + ψ(u^+_e - f_e)
```

where `ψ(x) = x^(-α)` is a power barrier, and
`α = 1 / (1000 * log(mU))` for edge count `m` and capacity scale `U`.

The barrier induces per-edge lengths:

```
ℓ_e(f) = (u^+_e - f_e)^(-1-α) + (f_e - u^-_e)^(-1-α)
```

These lengths penalize moves near either bound. The gradient `g = ∇Φ(f)`
combines the linear cost direction with barrier contributions and is used to
find a descent circulation.

## Min-ratio cycle oracle → circulation update

Each IPM iteration searches for a circulation `Δ` satisfying `Bᵀ Δ = 0` that
makes the ratio

```
(gᵀ Δ) / ||L Δ||₁
```

as negative as possible, where `L` is the diagonal operator of lengths `ℓ`.
The undirected min-ratio reduction guarantees a near-optimal circulation can
be represented by a simple cycle. The solver therefore:

1. Builds a low-stretch spanning tree.
2. Enumerates fundamental cycles induced by off-tree edges.
3. Evaluates each cycle `C` using the path vector `p(C)`:
   - numerator: `gᵀ p(C)` (signed along traversal),
   - denominator: `||L p(C)||₁` (sum of lengths along the cycle).

The best ratio cycle yields a unit circulation `Δ`. We scale `Δ` so
`||L Δ||₁ = κ/50`, where `κ` is the achieved ratio, then line-search to keep the
flow strictly feasible and to ensure the potential decreases.

## Termination and rounding to exact optimum

When the potential indicates the primal-dual gap proxy is small, we switch to
rounding:

1. Construct a residual min-cost circulation instance from the fractional flow.
2. Cancel negative reduced-cost cycles to remove fractional parts.
3. Verify integrality and cost preservation.

This produces a feasible integral flow with the same optimal cost (within the
chosen tolerance) as the fractional IPM solution.

## References

- Aaron Bernstein, Jonathan R. Kelner, and S. Matthew Weinberg. *Almost-Linear
  Min-Cost Flow in Directed Graphs*. arXiv:2203.00671, 2022.
  https://arxiv.org/abs/2203.00671
