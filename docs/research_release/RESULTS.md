# NN-TENO5 V2 research release — measured findings

This report distinguishes existing checkpoint scores, full-validation policy selection, prospective final cases and experimental numerical variants. No network weights were changed. The original test split was not accessed for selection. All values are mathematical numerical experiments, not measured flow data.

## Full-validation correction study

All 615 scalar and 307 Euler validation trajectories were evaluated at six strengths with separate soft/hard candidate policies: **11,064 method–trajectory rollouts**. Every comparison uses the same case-specific physical time, 15 SSPRK3 steps, grid, boundary and reference.

| Physics | Policy | λ | Macro L1 ratio | 95% interval | Weighted ratio | Failures |
|---|---|---:|---:|---|---:|---:|
| scalar | soft | 0 | 1.000000 | 1.000000–1.000000 | 1.000000 | 0 |
| scalar | soft | 0.1 | 0.985322 | 0.983251–0.986136 | 0.960349 | 0 |
| scalar | soft | 0.25 | 0.976976 | 0.974176–0.978815 | 0.943174 | 0 |
| scalar | soft | 0.5 | 0.992541 | 0.988246–0.995933 | 0.981072 | 0 |
| scalar | soft | 0.75 | 1.046434 | 1.039368–1.052159 | 1.106080 | 0 |
| scalar | soft | 1 | 1.127515 | 1.118796–1.135148 | 1.282550 | 0 |
| scalar | hard | 0 | 1.000000 | 1.000000–1.000000 | 1.000000 | 0 |
| scalar | hard | 0.1 | 1.005265 | 0.995932–1.007327 | 0.983495 | 0 |
| scalar | hard | 0.25 | 1.027271 | 1.015589–1.030428 | 1.022319 | 0 |
| scalar | hard | 0.5 | 1.055541 | 1.040720–1.060534 | 1.065820 | 0 |
| scalar | hard | 0.75 | 1.086366 | 1.074514–1.093826 | 1.114013 | 0 |
| scalar | hard | 1 | 1.117810 | 1.104993–1.128379 | 1.154715 | 0 |
| euler | soft | 0 | 1.000000 | 1.000000–1.000000 | 1.000000 | 0 |
| euler | soft | 0.1 | 0.971109 | 0.969411–0.972839 | 0.965705 | 0 |
| euler | soft | 0.25 | 0.947938 | 0.943608–0.952334 | 0.934355 | 0 |
| euler | soft | 0.5 | 0.964660 | 0.957020–0.972924 | 0.966373 | 0 |
| euler | soft | 0.75 | 1.043512 | 1.032464–1.055546 | 1.061648 | 1 |
| euler | soft | 1 | 1.164256 | 1.147473–1.181816 | 1.180608 | 1 |
| euler | hard | 0 | 1.000000 | 1.000000–1.000000 | 1.000000 | 0 |
| euler | hard | 0.1 | 1.001591 | 0.998632–1.004527 | 1.001876 | 0 |
| euler | hard | 0.25 | 1.023218 | 1.016226–1.029859 | 1.031961 | 0 |
| euler | hard | 0.5 | 1.055173 | 1.045051–1.064668 | 1.069643 | 1 |
| euler | hard | 0.75 | 1.092616 | 1.078567–1.106825 | 1.130557 | 1 |
| euler | hard | 1 | 1.145721 | 1.128267–1.166912 | 1.202346 | 2 |

Ratios exclude constant controls and family×resolution baseline errors ≤1e-10; absolute norms remain reported. Macro averages weight family×resolution equally. Weighted ratios divide trajectory mean errors. Bootstrap intervals use 2,000 paired stratified trajectory resamples, seed 20261020. Intervals condition on valid cases and do not correct for policy selection or quantify training-seed/reference uncertainty. Failed policies cannot be selected.

**Validation selection:** soft λ=.25 for both existing backbones. Scalar training used λ=.75; this inference calibration improves its full-validation score without retraining. Euler retains its training strength .25. Hard masks are not equivalent to the trained soft policy.

[Per-case measurements](results/full-validation/per_case.json), [bootstrap/group summary](results/full-validation/summary.json), [all-norm distributions and extrema](results/derived-validation/statistics.json), [per-case ratios](results/derived-validation/per_case_relative.json).

## Gate behavior and long-time prospective evaluation

The manifest declared 11 IVPs, each at two grids (22 configurations). Twenty original non-blast configurations completed. The original finest blast temporal reference was stopped for CPU cost. A separate closed-wall smoke with a 128/256/512 ladder and dt/2 exposed selected-model failure before t=.001. Full-time closed-wall blast results therefore assess stability only, with no accuracy reference or error score. Four Riemann IVPs are independently seeded; other entries are prospective canonical/stress configurations. The two grids of one IVP are not independent trajectories. The final suite did not retune any model or correction strength. The original 2048-cell blast temporal protocol remains incomplete.

| Case | N | TENO5 L1 | Selected soft L1 | Ratio | Valid raw stages | Reference ranking |
|---|---:|---:|---:|---:|---|---|
| independent-advection-square | 192 | 0.0121775 | 0.0137186 | 1.12655 | True | resolved |
| independent-advection-square | 96 | 0.0243811 | 0.0243197 | 0.997485 | True | resolved |
| independent-burgers-interaction | 192 | 0.00226497 | 0.00212156 | 0.936683 | True | unresolved / no reference |
| independent-burgers-interaction | 96 | 0.00358645 | 0.00350747 | 0.97798 | True | unresolved / no reference |
| independent-burgers-shock | 192 | 0.000410798 | 0.000341362 | 0.830973 | True | resolved |
| independent-burgers-shock | 96 | 0.00125584 | 0.00105107 | 0.836951 | True | resolved |
| independent-contact | 192 | 0.00221497 | 0.00236723 | 1.06874 | True | resolved |
| independent-contact | 96 | 0.00445055 | 0.00444731 | 0.999272 | True | resolved |
| independent-riemann-0 | 192 | 0.0070853 | 0.00669155 | 0.944427 | True | resolved |
| independent-riemann-0 | 96 | 0.0121911 | 0.0110692 | 0.907974 | True | resolved |
| independent-riemann-1 | 192 | 0.00862018 | 0.0080324 | 0.931813 | True | resolved |
| independent-riemann-1 | 96 | 0.0150273 | 0.013523 | 0.899899 | True | resolved |
| independent-riemann-2 | 192 | 0.0159101 | 0.0153315 | 0.96363 | True | resolved |
| independent-riemann-2 | 96 | 0.0266181 | 0.0258356 | 0.970603 | True | resolved |
| independent-riemann-3 | 192 | 0.00185241 | 0.00180072 | 0.972098 | True | resolved |
| independent-riemann-3 | 96 | 0.00375258 | 0.00355784 | 0.948104 | True | resolved |
| independent-shu-osher | 128 | 0.1834 | 0.18124 | 0.988223 | True | unresolved / no reference |
| independent-shu-osher | 256 | 0.0782673 | 0.0754645 | 0.964188 | True | unresolved / no reference |
| independent-sod | 192 | 0.00265136 | 0.00253802 | 0.95725 | True | resolved |
| independent-sod | 96 | 0.00550796 | 0.00519117 | 0.942486 | True | resolved |
| wall-closed-woodward-colella | 128 | unavailable | unavailable | unavailable | False | unresolved / no reference |
| wall-closed-woodward-colella | 256 | unavailable | unavailable | unavailable | False | unresolved / no reference |

All six policies, checkpoints, profiles, sampled error-growth curves, gate activation/rejected candidates, soft/hard decisions and correction distributions are saved in [the benchmark artifacts](results/independent/). Historical learned blends use their own P0 checkpoints and policy names. They are comparisons on matched configurations, not the same method generation.

Advection and Euler Riemann references are analytic cell averages. Burgers post-shock/interaction and Shu–Osher use numerical references, a three-grid ladder and a finest-grid dt/2 comparison. Twice the sum of final spatial and temporal Cauchy changes is an uncertainty proxy, not a rigorous bound. A ranking is marked resolved only if the proxy is below 25% of the method margin and both methods have admissible raw stages. Full-time blast accuracy is unassessed after its failed neural smoke; no numerical reference is fabricated. Frozen P0 point-sampled references cannot replace these cell averages.

**Wall-boundary correction:** the original mirror-only blast configuration is retained as an unfinished reference study; a separate random-state wall audit exposed its defective neural wall contract. The soft network need not obey reflection/sign symmetry; random admissible wall stencils measured nonzero mass/energy flux. `reflective-wall` explicitly sets numerical wall mass/energy flux to zero while retaining wall momentum flux. Its dedicated regression verifies closed-wall conservation. The separately named closed-wall run freezes the same λ/model. Reference computations now save/cache each grid/timestep result immediately; after the failed neural smoke, the full-time run is explicitly stability-only. This is a correctness correction, not final-benchmark tuning. Admissible raw stages alone do not validate a physical boundary.

Observed gate/correction distributions are summarized in [gate_diagnostics.json](gate_diagnostics.json). Solution/error plots show both favorable and unfavorable comparisons. A lower final error alone does not establish that oscillations decreased; component extrema, overshoot/undershoot and TV are provided for that separate assessment.

For example, at 192-cell Sod the selected method lowers L1 error from .00265136 to .00253802, while density TV rises from .877466 to .879756 and density undershoot relative to the exact final range reaches .000350685 (TENO5: zero). At 192-cell long-time advection, error increases by 12.65% and TV increases from 2.400000 to 2.400435. These observations support a mixed accuracy/variation tradeoff, not a universal sharpening claim.

Gate activation is not synonymous with a substantial correction: in the time-sampled 192-cell advection trajectory, the gate activates on 56.76% of direction/interface observations, but 45.78% of active corrections are below 1e-12×max(|classical reconstruction|,1). In Sod these fractions are 37.31% and 62.87%. Potential sensitivity to small numerical variations is a hypothesis; these statistics alone do not isolate its cause.

## Smooth spatial and temporal accuracy

Spatial: 40/80/160/320/640 cells; exact cell averages; semidiscrete divergence isolates spatial error. Completed CPU rollout controls: 40/80/160 cells at common physical time .05, dt proportional to dx^(5/3), and dt/2 at every completed grid. Both checkpoints use selected λ=.25. **The original full temporal protocol through 640 was stopped for CPU cost and remains incomplete**; its exit 130 and progress log are preserved. This is a resource limit, not a numerical pass.

| Physics | Method | Profile | Last spatial L1 order | Last rollout L1 order |
|---|---|---|---:|---:|
| scalar | teno5 | regular | 4.9992 | 4.9852 |
| scalar | teno5 | critical-sin4 | 4.0134 | 4.4409 |
| scalar | gated-soft | regular | 4.9992 | 4.9852 |
| scalar | gated-soft | critical-sin4 | 4.0170 | 4.4407 |
| scalar | gated-hard | regular | 4.9992 | 4.9852 |
| scalar | gated-hard | critical-sin4 | 4.0098 | 3.8049 |
| scalar | neural-soft | regular | 3.0103 | not run |
| scalar | neural-soft | critical-sin4 | 2.9913 | not run |
| euler | teno5 | regular | 4.9996 | 4.9877 |
| euler | teno5 | critical-sin4 | 4.0172 | 4.5119 |
| euler | gated-soft | regular | 4.9996 | 4.9877 |
| euler | gated-soft | critical-sin4 | 4.0237 | 4.2225 |
| euler | gated-hard | regular | 4.9996 | 4.9877 |
| euler | gated-hard | critical-sin4 | 4.0127 | 3.9481 |
| euler | neural-soft | regular | 3.0729 | not run |
| euler | neural-soft | critical-sin4 | 3.1150 | not run |

[Completed CPU norm/order data and stage diagnostics](results/temporal-cpu/convergence.json), [selected spatial ladder](results/selected-spatial/convergence.json), [incomplete extended temporal status](results/convergence/status.json). Regular smooth spatial errors approach fifth order. Smooth critical-point profiles do not establish fifth order, and neural-only approaches third order. Interpret rollout rates with absolute errors and temporal Cauchy changes; no uncompleted fine-grid temporal result is claimed.

A broader nonlinear smooth Euler instantaneous test gives a different conclusion:

| Solver | Reconstruction | Last spatial L1 order |
|---|---|---:|
| historical-split-flux | teno5 | 2.0120 |
| historical-split-flux | gated-soft | 2.0083 |
| historical-split-flux | gated-hard | 2.0181 |
| experimental-fv-state-flux | teno5 | 4.9999 |
| experimental-fv-state-flux | gated-soft | 4.9999 |
| experimental-fv-state-flux | gated-hard | 4.9999 |

For nonlinear fluxes, F(cell-average U) differs from cell-average F(U) at O(dx²). The historical Euler split-flux path feeds the former into a flux reconstruction with cell-average coefficients, explaining the measured approximately second-order consistency on this general finite-volume test. The constant-pressure/velocity entropy wave is a special linear-in-density case.

An **optional experimental** characteristic-state reconstruction followed by a Rusanov flux approaches fifth order on this spatial diagnostic. It preserves the baseline and has dedicated consistency/conservation tests; its shock behavior, positivity and training remain unvalidated. No replacement checkpoint or general fifth-order neural claim is made. [Measured data](results/nonlinear-euler/nonlinear_euler.json).

## Conservation, admissibility and historical repair

All 11,064 validation rollouts have per-stage observations and conservative boundary budgets. The largest conservation residual across valid full-validation trajectories is approximately 1.42e-14. Selected λ=.25 Euler has minimum stage density 0.09545, minimum pressure 0.08357 and maximum effective CFL 0.33403. Scalar selected maximum effective CFL is 0.29781. These finite-corpus observations are not a general positivity or CFL stability theorem.

On the failing strong-correction Euler validation case, soft λ=.75 with the historical repair changes integrated mass by about 1.61115e-3 and energy by 8.88843e-4 (one repaired stage row). Hard λ=1 changes mass by 4.31769e-3 and energy by 2.98722e-3 (three rows). With repair increments included, balance residuals return to roundoff. Without them the physical balance error remains. At λ=.25 this case requires no repair. Original training intervention histories were not recorded at every update and cannot be reconstructed from sparse logs. [Repair evidence](results/repair/repair.json).

## Euler warm-up input mismatch

Training-data-only diagnostics compare four trajectories per family×grid. Warm-up characteristic states use initial snapshots; positive/negative solver flux inputs also use later exact snapshots. Mean normalized marginal Wasserstein distances are approximately .04549 (positive flux) and .02768 (negative flux). Teacher BCE rises from .67622 on warm-up states to .69319/.68924 on split fluxes; hard teacher disagreement rises from .43714 to .49922/.49790. These macro summaries weight family×grid equally.

The input mismatch is measurable after normalization. It does not establish that correcting it improves PDE outcomes: the sample times differ, inputs are exact-reference snapshots rather than neural trajectories, and teacher agreement is not numerical accuracy. The original warm-up remains available; `--pretraining-inputs split-flux` supplies an explicit future ablation. No retraining or causal improvement is claimed. [Diagnostic data](results/input-mismatch-r2/input_mismatch.json).

## Computational cost

CPU float64, one fixed IVP per physics, four grids, one warm-up plus nine synchronized repeats. Production kernels exclude diagnostic/export overhead; lowering, compilation and first execution are separate.

| Physics | N | Method | Warm median [s] | Compile [s] | L1 |
|---|---:|---|---:|---:|---:|
| scalar | 64 | teno5 | 0.0022224 | 0.1982 | 0.0208941 |
| scalar | 64 | gated-soft | 0.0169985 | 0.41927 | 0.0196579 |
| scalar | 128 | teno5 | 0.00767938 | 0.20611 | 0.016532 |
| scalar | 128 | gated-soft | 0.0599563 | 0.37617 | 0.0158873 |
| scalar | 256 | teno5 | 0.0289298 | 0.21446 | 0.00813234 |
| scalar | 256 | gated-soft | 0.225987 | 0.37576 | 0.0081928 |
| scalar | 512 | teno5 | 0.111489 | 0.27079 | 0.00462569 |
| scalar | 512 | gated-soft | 1.21335 | 0.38364 | 0.00499121 |
| euler | 64 | teno5 | 0.0198675 | 0.70716 | 0.00897385 |
| euler | 64 | gated-soft | 0.0363202 | 0.81591 | 0.00842934 |
| euler | 128 | teno5 | 0.0548274 | 0.58649 | 0.0041678 |
| euler | 128 | gated-soft | 0.139345 | 0.82584 | 0.00385155 |
| euler | 256 | teno5 | 0.209311 | 0.61368 | 0.0020044 |
| euler | 256 | gated-soft | 0.604839 | 0.83038 | 0.00185818 |
| euler | 512 | teno5 | 0.960704 | 0.74736 | 0.0012043 |
| euler | 512 | gated-soft | 2.52801 | 0.81125 | 0.00114853 |

[Timing repeats, per-step/cell/interface costs, temporary memory estimates and equal-budget/equal-accuracy comparisons](results/performance/performance.json). Hardware is recorded in [timing provenance](results/performance/provenance.json). Process RSS is a lifetime high-water mark; temporary storage is a compiler estimate. These measurements do not establish a general speedup or GPU performance.

## Engineering and scientific release assessment

The final CPU regression suite passed **117 tests and 34 subtests**; the original baseline was 97 tests and 34 subtests. The separate Python 3.7 legacy network smoke passed **2 tests**. All 498 protected files matched the baseline inventory. The recorded environment passes pip check; a fresh V2 environment and GPU path were not independently installed/tested. CI is configured but has not run remotely.

A qualified public portfolio of **research implementation and rigorous investigation** is supported. A paper-level claim of universally superior, fifth-order, conservative positivity-preserving or faster NN-TENO5 is **not supported**. Outstanding work includes general Euler finite-volume shock validation, critical-point accuracy, stable conservative positivity treatment, controlled multi-seed training and warm-up ablations, broader/stronger reference studies and hardware portability.

[Acceptance matrix](ACCEPTANCE.md) · [reproduction commands](REPRODUCE.md) · [milestone record](MILESTONES.md) · [portfolio materials](../portfolio/LINKEDIN_POST.md). No push, merge or public publication was performed.
