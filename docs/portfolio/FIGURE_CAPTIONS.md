# Reproducible figure captions

**pipeline** — Conceptual architecture diagram. Neural candidate weights, deterministic gating and correction strength are separate; all methods update a conservative numerical-flux difference.

**smoothness_gate** — Illustrative decision diagram. The detector may reject candidates at smooth critical points; measured convergence remains necessary.

**correction_strength** — Full validation: 615 scalar and 307 Euler trajectories; equal weight per nonconstant family×resolution. Shading: paired stratified trajectory bootstrap 95% intervals, conditional on valid cases. Failed settings are ineligible for selection. Constants and baseline errors ≤1e-10 are excluded from ratios.

**strength_families_scalar** — Scalar validation strength sensitivity by family and resolution. Solid: soft; dashed: hard. All per-trajectory failures remain in the accompanying JSON report.

**strength_families_euler** — Euler validation strength sensitivity by family and resolution. Solid: soft; dashed: hard. All per-trajectory failures remain in the accompanying JSON report.

**smooth_convergence** — Semidiscrete cell-average divergence errors: no SSPRK3 temporal error. Euler uses an advected entropy wave; the general nonlinear Euler diagnostic is separate. Critical-point behavior is reported without a fifth-order guarantee.

**nonlinear_euler_consistency** — Instantaneous nonlinear smooth Euler diagnostic. Historical split-flux treatment approaches second order on cell averages; optional characteristic-state finite-volume flux approaches fifth order. The experimental variant has no completed shock/positivity validation.

**scalar_discontinuity** — independent-advection-square N=192. Analytic finite-volume cell-average reference. Gate/correction values come from the selected method trajectory. Euler correction is in characteristic split-flux space and is not a conservative-state correction.

**gate_evolution** — Gate and reconstruction-correction evolution along the selected scalar numerical trajectory, sampled at 22 times. This is measured numerical output, not an illustration.

**euler_sod** — independent-sod N=192. Analytic finite-volume cell-average reference. Gate/correction values come from the selected method trajectory. Euler correction is in characteristic split-flux space and is not a conservative-state correction.

**shu_osher** — independent-shu-osher N=256. Numerical reference; ranking subject to measured Cauchy uncertainty. Gate/correction values come from the selected method trajectory. Euler correction is in characteristic split-flux space and is not a conservative-state correction.

**final_benchmark** — Prospective cases declared after validation selection and before evaluation. Gray indicates unresolved reference ranking, invalid stages or the mirror-only wall defect. The wall-closed rerun is a distinct physical-boundary correction with no strength tuning. Numerical-reference Cauchy tests qualify interpretation. These cases do not support universal superiority.

**accuracy_runtime** — Synchronized CPU float64 execution, median of nine repeats after warm-up; compilation reported separately. Labels show grid cells. One fixed IVP per physics; no GPU or general speedup claim.

**accuracy_resolution** — Measured accuracy against grid resolution on the same timed IVPs. Compare with accuracy_runtime; a same-grid improvement alone is not a speedup.
