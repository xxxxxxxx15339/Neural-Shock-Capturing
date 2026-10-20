# Repository audit before implementation — 2026-10-20

Base: `release/p0-validation`, commit `9919414`. The working tree was clean.
Work proceeds on `feat/nn-teno5-v2-research-release`. No AGENTS.md was found in
the repository or its ancestor directories. No default-branch change or push is authorized.

The README, P0 validation report, phase history, unrolled-loss guide, JAX solver,
reconstruction/projection, checkpoint loader, dataset builder/loader, V2 training
script, model metadata and numerical tests were inspected before implementation.

## Existing evidence and reusable infrastructure

* P0 has strict checkpoint loading (backbone, linear head, rich head), matched
  historical ablations, stage-level conservation/repair diagnostics, spatial and
  temporal convergence tools, and non-overwriting artifact exports.
* P0 historical benchmarks use point samples and ungated hard neural masks. V2
  uses finite-volume cell-average targets and deterministic gating with **soft**
  candidate weights. Their reported scores are different estimands.
* V2 networks both retain 5→64→32→16→8→3 and have independent backbone checkpoints.
  Scalar was trained at strength .75; Euler at .25, each for 15 SSPRK3 steps.
* Stored scalar selection score is 1.0357177404; Euler 0.9561784830.
  Selection used at most two trajectories per family per resolution (36 scalar,
  30 Euler), rather than all 615/307 validation trajectories. Previously examined
  test reports are historical evidence and will not select any new policy.
* Dataset: 6,144 independent analytic PDE problems, with exact-reference cell
  averages. No experimental measurements. Four stored times plus a separate
  15-step target. The full validation split is 922 independent problems.
* Scalar reconstruction first reconstructs **states**, then applies a nonlinear
  LF flux; Euler reconstructs positive/negative characteristic split **fluxes**.
  This distinction must remain explicit, particularly for nonlinear Burgers
  consistency. V2 warm-up currently uses characteristic **states**, not the
  actual Euler split-flux inputs.
* Euler LF alpha uses adjacent states, not a bound over the entire stencil.
  Positivity is not guaranteed. Training repair is periodic neighbor replacement
  plus a fallback; it is not conservative. Unguarded evaluation is essential.
* Existing Shu–Osher references have unresolved spatial/temporal uncertainty and
  point-sample semantics. They cannot certify a new finite-volume benchmark win.
* `data/data` and `data/nnteno_v2/data` appear duplicated; hashes will determine
  exact duplication. Preserve both, recommend `data/nnteno_v2/data` as canonical.
* Available environment: Python 3.11.15, JAX/jaxlib 0.10.2, NumPy 2.4.6,
  Optax 0.2.8, h5py 3.16.0, CPU backend. Existing lock lacks h5py.

## Scope and evidence rules

Preserve all models, datasets, tests, historical results and frozen P0 artifacts.
Add policy names to the shared reconstruction factory and keep P0 defaults.
Measure full-validation strength sensitivity before considering training.
Independent final cases are declared before evaluating them and never used for
selection. Constant-state ratios are excluded below a declared error floor;
absolute errors remain visible. Mixed/failed experiments remain in the report.

Milestone records, commands, manifests, checksums and generated results accompany
this audit. Implementation, scientific performance and release readiness are
separate acceptance decisions.
