# Derandomization notes

This note summarizes the deterministic updates from the derandomization framework in arXiv:2309.16629 and contrasts them with the randomized min-ratio cycle framework in arXiv:2203.00671.

## Key takeaways from arXiv:2309.16629

- Deterministic cycle selection replaces random tree sampling with fixed-order decompositions so that each IPM iteration yields reproducible min-ratio cycles.
- Vertex sparsification is driven by deterministic hierarchical decompositions that avoid probabilistic sampling in the reduction hierarchy.
- Edge sparsification relies on deterministic spanner maintenance that keeps embeddings stable without random jitter.

## Comparison to the randomized framework (arXiv:2203.00671)

- Randomized low-stretch trees sample a tree distribution to guarantee expected stretch; the derandomized approach fixes a consistent ordering and tie-breaking so that tree construction is repeatable.
- Randomized dynamic spanners use probabilistic update schedules; deterministic variants rely on structured rebuild triggers that do not depend on random bits.

These notes are referenced from DESIGN_SPEC.md to document the deterministic pathway within the IPM loop and min-ratio cycle oracles.
