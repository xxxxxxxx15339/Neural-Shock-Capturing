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

## B — Scientific diagnostics

Completed full validation: **11,064 method–trajectory rollouts** (922 trajectories,
six strengths, two policies). Smoke succeeded before the full run. Results are in
`results/full-validation/`; all-norm distributions, extrema and per-case ratios
are in `results/derived-validation/`. Failures are retained, never masked into
an accepted policy. Constants/near-zero baseline ratios are excluded as declared.

Minimum zero-failure macro L1 policy is **soft λ=.25** for both checkpoints:
scalar .97697556 (95% bootstrap .97417575–.97881537), Euler .94793833
(.94360842–.95233397). Weighted ratios are .94317 and .93436. These are
validation-selected findings; no test data or final benchmark selected them.
The original scalar .75 correction is worse on the full corpus. Hard policies
behave differently; strong Euler corrections fail. Checkpoint weights are unchanged.

Solver-input extraction and normalized distribution diagnostics are complete on
training snapshots. The first diagnostic hit a NumPy boolean-negation exception;
the fixed diagnostic is `results/input-mismatch-r2`, with the failed execution
retained separately. Input mismatch is observed; causal retraining benefit is not.
The original warm-up and explicit split-flux warm-up ablation are both available.

Additional stage/repair diagnosis shows nonconservative mass/energy changes on
a failed strong-correction case, with the actual RK-weighted changes accounted
for. At selected .25 that case needs no repairs. Gate/correction observations on
time-dependent selected trajectories accompany the prospective benchmark profiles.

## C — Numerical validation (execution in progress)

Full spatial ladder: N=40,80,160,320,640; temporal controls use matched final time
.05 and dt/dt/2 with dt proportional to dx^(5/3). Fine neural Euler rollouts are
expensive on CPU. A completed temporal report is required before marking this
study complete; operational progress logs are not complete scientific artifacts.

A separate completed general nonlinear Euler instantaneous test exposes second-
order consistency in the historical split-flux finite-volume treatment. The
optional state-flux variant measures order approximately five on that diagnostic,
with a dedicated regression; it is not a validated shock solver replacement.
The original baseline is preserved. Current CPU regression: **115 passed,
34 subtests passed**. Legacy Python 3.7 network smoke: **2 passed**. Local protected
file check: **498 unchanged**. `pip check`: no broken requirements.

## D — Targeted improvement

Validation-based strength calibration is the supported improvement; no new model
training is claimed. The Euler input ablation and finite-volume state-flux variant
are explicit experimental options, with their remaining validation listed.
Optimizer/RNG resumption, horizon curricula and multi-seed retraining are future
work because a controlled retraining benefit has not been established here.

## E — Independent benchmarking (execution in progress)

`configs/v2/independent.json` was declared after freezing the validation-selected
policy and before any final case was evaluated. Eleven IVPs at two grids include
four independently seeded random Riemann problems, long-time advection, post-shock
and interacting Burgers, Sod/contact, Shu–Osher and reflecting-wall blast waves.
Numerical references have spatial ladders and dt/2 controls. Reference failures
or unresolved margins remain failures/unresolved. Isolated performance timing
follows completion of other numerical workers.

## F — Public research presentation (assembly in progress)

The concise README links to this evidence, the original README is archived, and
LinkedIn/CV/contribution/storyboard drafts are written. Reproducible figure and
seven-page PDF generation read numerical artifacts; no fake curves substitute
for missing studies. Final portfolio and acceptance checks follow numerical runs.

## Final milestone assessment

**A complete.** `fc00b77` freezes baseline identities and integrates the shared
policies. **B complete.** `61b0f70` records full-validation strength/input studies.
The weights and 498 protected research files remain unchanged.

**C core measurements complete; original fine temporal protocol incomplete.**
Selected-policy spatial tests cover 40/80/160/320/640 cells. Completed dt/dt/2
rollouts cover 40/80/160 at physical time .05. All these raw stages are admissible;
at 160 cells, temporal Cauchy changes are under .5% of the observed state L1
error. The original extended neural Euler temporal worker was stopped for CPU
cost (exit 130); its status/progress is retained, not called a pass. General
nonlinear historical Euler remains approximately second order. Critical-point
orders differ from five. Conservation and nonconservative repair budgets are
explicitly measured.

**D complete as inference calibration; training ablations unperformed.** Soft
λ=.25 is frozen from validation only. No new checkpoint, training-seed robustness
or causal warm-up benefit is claimed. The optional state-flux consistency variant
and exact split-flux warm-up are independent experimental options.

**E completed with explicit stress/reference failures.** Twenty original final
configurations have matched six-policy accuracy/stability profiles. Original
blast fine temporal reference work was stopped for CPU cost (exit 130), before
coarse neural accuracy evaluation. A random-state wall audit exposed nonzero
soft-neural mass/energy flux with mirror-only ghosts. The explicit closed-wall
correction has a regression. Its t=.001 smoke fails for V2; full-time blast
128/256-cell runs therefore assess stability only. TENO5 is admissible at both
grids; selected V2 is not. No full-time blast accuracy reference or win is claimed.

Isolated timing completed on AMD Ryzen 5 4600H CPU, float64: 32 compiled kernels,
one first/warm-up execution and nine synchronized warmed repeats each. At 512
cells the selected hybrid costs 10.88× TENO5 on advection (higher error) and
2.63× on Sod (about 4.63% lower L1). The measured discrete equal-budget ladder
does not establish a cheaper neural route to the same accuracy.

**F complete as reviewable portfolio materials.** README, measured report,
acceptance matrix and reproduction commands accompany 14 PNG/PDF/SVG figures,
a seven-page PDF carousel, LinkedIn drafts, CV bullets, attribution and video
storyboard. Public posting, pushing and merging are not performed.

Final regression: **117 CPU tests + 34 subtests passed**, plus **2 legacy smoke
tests passed**. A unified-CLI zero-strength rollout is bit-identical to TENO5;
reference-cache checks verify conservative restriction/reuse and reject mismatched
physical configurations. Artifact verification recomputes **134 exported profile
metrics**, checks **469 artifact hashes**, and confirms all **498 protected files**.
Remote CI, a fresh V2 install and GPU execution remain untested.

Publication assessment: suitable for a qualified public research-software
portfolio; not a validated universally superior/fifth-order/stable/faster neural
method. The unfinished original protocols and scientific failures remain listed
in the acceptance matrix and measured report.
