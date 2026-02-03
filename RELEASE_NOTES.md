# Release Notes

## 0.2.0

### Highlights
- Almost-linear IPM path is the default for larger instances, with automatic
  scaling for large capacity/cost bounds.
- Deterministic mode is the default and uses derandomized shifting to ensure
  reproducible oracle behavior.
- Solver statistics now include `solver_mode` to log which path was used
  (`ipm`, `ipm_scaled`, `classic`, or `classic_fallback`).

### Python usage
```python
flow, stats = min_cost_flow(graph, use_ipm=True, deterministic=True, return_stats=True)
print(stats["solver_mode"])
```
