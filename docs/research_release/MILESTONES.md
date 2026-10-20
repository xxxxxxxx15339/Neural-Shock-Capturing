# Execution record

## A — Baseline and architecture

Completed on the requested feature branch. See [audit](AUDIT.md) and
[baseline manifest](baseline/manifest.json). Preserved 498 research files by hash.
Baseline regression: **97 passed, 34 subtests passed**. Integration regression:
**63 passed**. New V2 tests: **14 passed**, including comparison against the
original dataset-backed training evaluator for advection, Burgers and Euler.

Shared reconstruction supports historical policies, deterministic TENO5, V2
soft/hard gates, soft/hard neural-only, and separately identified saved-head
gated variants. No head is synthesized for a backbone checkpoint. Exact Euler
split-flux stencil extraction is shared with the solver. Stage diagnostics can
remain on device for batch evaluation. P0 output directories are protected.

Full-validation and smoke JSON configurations were declared. The successful smoke
is `results/sweep-smoke-r2`; the first smoke's JAX tracing exception is preserved
in `results/sweep-smoke/failure.log`. It led to a check that permits traced dt
only in the explicitly requested device evaluation path. Ordinary diagnostics
retain host input validation. No numerical baseline was changed.

Scientific conclusion: evaluator parity and zero-strength identity are supported;
V2 performance and high-order claims await the studies below.
