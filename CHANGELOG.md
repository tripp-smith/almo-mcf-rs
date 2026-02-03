# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2026-04-15
### Added
- Deterministic solver pipeline with reproducible min-ratio cycles and sparsification updates.
- Documentation updates covering deterministic usage flags and derandomization notes.
- Solver stats now include a solver_mode label for IPM/classic path tracking.

## [0.1.0] - 2026-02-01
### Added
- Initial release of the almost-linear minimum-cost flow solver with IPM support.
- Dynamic data structures, extensions (matching, connectivity, cuts), and DAG isotonic regression utilities.
- Python/Rust API integration with NetworkX adapters and convex objectives.
- CI workflows for linting, testing, and cross-platform wheel builds.
