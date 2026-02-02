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

## Pseudocode adaptations (deterministic variants)

The following sketches describe how the randomized components are replaced by deterministic,
reproducible counterparts in this codebase.

### Deterministic vertex sparsification

```
procedure DeterministicVertexDecomposition(G, tree, cluster_target):
    order <- fixed ordering of vertices (e.g., ID order)
    clusters <- empty
    for v in order:
        if v not assigned:
            cluster <- BFS in tree, stopping after cluster_target vertices
            assign cluster with deterministic tie-breaking
            clusters.append(cluster)
    return clusters
```

Key changes:
- Replace random sampling with a stable traversal order.
- Use deterministic tie-breaking for cluster boundaries.

### Deterministic dynamic spanner maintenance

```
procedure DeterministicDynamicSpanner(update, rebuild_every):
    apply update to graph
    if updates_since_rebuild >= rebuild_every:
        rebuild full spanner deterministically
    else:
        repair local embeddings deterministically
```

Key changes:
- Rebuild cadence depends on fixed counters, not randomness.
- Embeddings are repaired with deterministic BFS ordering.

### Deterministic min-ratio cycle selection

```
procedure DeterministicMinRatioCycle(oracle_state):
    candidates <- collect off-tree edges in fixed order
    for edge in candidates:
        compute ratio(edge)
    return argmin ratio with deterministic tie-break
```

Key changes:
- Enumeration uses stable ordering of edges.
- Ties are broken deterministically to ensure reproducibility.
