# P0 reproducibility and scientific validation

Validated on Linux x86_64 / CPU, 2026-10-14, on `release/p0-validation`.
Final artifact assembly and preservation checks: 2026-10-15 UTC.
The source revision is `4e514b19d12247f839dba5af396154cdc37d96ec` plus the recorded
working-tree source hashes. No model was retrained or selected using the new tests.
No checkpoint, historical metric file, plot, dataset, or existing test was replaced.
No commit, push, or release publication was performed.

## Reproduction and identities

Use the unified CLI and isolated environment instructions in [README](../README.md).
The [manifest](../src/differentiable/benchmarks.json) freezes all problem definitions,
model hashes and the constant blend `sigmoid(-3) = 0.04742587317756678`. Canonical
cases remain point-sampled at `i*dx`; interpolation and timestep rules are unchanged.
The new smooth studies separately use true cell averages and controlled time error.

| Selection | Checkpoint | SHA-256 |
|---|---|---|
| scalar | `models/nn_teno5_unrolled_jax_params_run6_learnedlambda.pkl` | `3d942ba76d1eb5ede3119bca360392bdbe4ce6a13ab56ab8f124323d4095674b` |
| euler | `models/nn_teno5euler_unrolled_jax_params_curriculum_phase18_learnedlambda.pkl` | `11db25a8c5417e18e1277a563de07c0977471ef6d52430e7aedaa1e37d044c2d` |

Both selected files are NumPy pickle dictionaries with `backbone` and `lambda_head`,
all finite float64 arrays. Backbone weight/bias shapes are `(5,64)/(64,)`,
`(64,32)/(32,)`, `(32,16)/(16,)`, `(16,8)/(8,)`, `(8,3)/(3,)`; the mixing head is
`(3,1)/(1,)`. The strict loader also accepts earlier backbone lists and Phase 21's
`lambda_head_v2`, shaped `(8,8)/(8,)` then `(8,1)/(1,)`. Unknown schemas, wrong shapes,
non-floating arrays, non-finite parameters, unreadable files and hash mismatches fail
explicitly. Keras HDF5 models remain in the separate legacy environment.

The original non-finite Euler checkpoint is preserved and rejected for evaluation.
`curriculum.pkl` and `phase15_smoothmix.pkl` are byte-identical; scalar `run4` and
`run5` are also byte-identical. Filenames alone cannot recover distinct provenance.

Fresh venvs installed the three complete dependency locks from package distributions:

| Environment | Python | Main numerical packages | Regression tests |
|---|---|---|---|
| Numerical | 3.11.15 | NumPy 1.26.4, Matplotlib 3.8.4; no JAX/Keras/TF | 33 passed |
| JAX | 3.11.15 | JAX/jaxlib 0.10.2, NumPy 2.4.6, Optax 0.2.8; no Keras/TF | 64 passed |
| Legacy | 3.7.12 | TensorFlow 1.15.0, Keras 2.2.4, NumPy 1.18.5, h5py 2.10.0; no JAX | 2 passed |

**99 tests passed**, including all original 49. `pip check` reported no broken
requirements in any environment. Legacy TF emitted deprecation/unavailable-GPU
messages; CPU smoke tests passed. Other operating systems and GPUs were not tested.
Environment details, commands and logs are in [reproduction](../src/differentiable/validation/p0-2026-10-14/reproduction/).
The original 49 tests and 112 protected model/plot/test/data files were verified
against the baseline hashes. The existing working-tree PDF deletions and unrelated
presentation/report work were left intact.

## Matched ablations and prospective held-out cases

Every method uses the same grid, timestep, physical flux/LF splitting, SSPRK3,
initial state, boundary conditions and reference within a case. Neural ablations
share the saved backbone and probability threshold 0.5; classical TENO5 retains
its existing selection cutoff 1e-5. Fixed blend uses the
same post-training backbone as learned blend; it changes only lambda. Neural-only
sets lambda to one, TENO5 to zero. Warm-start backbone controls are separately
exported to distinguish backbone changes from gate changes.

Scalar errors below are RMS (historically called L2). Euler errors are the arithmetic
mean of the three conserved-component L1 errors. Componentwise norms are exported;
this historical aggregate was not silently rescaled or changed.

| Case | TENO5 | Neural only | Fixed blend | Learned blend | Learned/TENO5 |
|---|---:|---:|---:|---:|---:|
| canonical-advection | 0.0989598541 | 475.29392 | 0.0974064292 | 0.0955212916 | 0.9652530 |
| canonical-burgers | 0.000816696989 | 0.00351331532 | 0.000712430673 | 0.00066492691 | 0.8141660 |
| canonical-sod | 0.00825832267 | 0.0102771852 | 0.00828539063 | 0.00828709346 | 1.0034839 |
| canonical-shu-osher | 0.213189213 | 0.256105842 | 0.209236937 | 0.209263935 | 0.9815878 |
| heldout-advection-square-96 | 0.0704607706 | 5.41666834 | 0.0685729091 | 0.0678452442 | 0.9628797 |
| heldout-advection-fourier-192 | 8.76549974e-06 | 0.33092444 | 1.16730951e-06 | 5.50098595e-06 | 0.6275724 |
| heldout-burgers-fourier-96 | 9.10790048e-06 | 2.39344652e-05 | 8.40501585e-06 | 8.13199131e-06 | 0.8928503 |
| heldout-burgers-fourier-192 | 6.79616333e-06 | 1.58006483e-05 | 5.96330262e-06 | 5.59493268e-06 | 0.8232487 |
| heldout-sod-96 | 0.00748351125 | 0.00707695543 | 0.00737255684 | 0.00737369808 | 0.9853260 |
| heldout-sod-192 | 0.00372694178 | 0.00481272242 | 0.00369368901 | 0.00369187944 | 0.9905922 |
| heldout-shu-osher-160 | 0.142058412 | 0.203990115 | 0.137661006 | 0.137627261 | 0.9688075 |
| heldout-shu-osher-240 | 0.155609717 | 0.283100643 | 0.143772862 | 0.144212857 | 0.9267600 |

Full [JSON](../src/differentiable/validation/p0-2026-10-14/reproduction/benchmarks/metrics.json),
[CSV](../src/differentiable/validation/p0-2026-10-14/reproduction/benchmarks/metrics.csv),
NPZ states/references/histories and PNG/PDF plots accompany these values. Each run
has hashes, resolved configurations, software/machine metadata and explicit validity
flags. The numerical-only artifact checker independently recomputed all 48 metrics,
verified every output file hash, model identities, matched timesteps and repair counts.

Canonical scalar cases selected the Phase 19 checkpoint, so they are validation
cases, not held-out evidence. The eight new configurations were written before their
first evaluation: new grids 96/192, step/Fourier profiles, Burgers Fourier data,
Sod locations/times, and Shu–Osher amplitude/frequency/phase variants at 160/240 cells.
No retraining, blend tuning or checkpoint selection followed those results. Complete
historical training-sample logs are unavailable; retrospective overlap cannot be
proved beyond the available source/configuration records.

The learned blend has lower error on the six new cases with analytic references.
The two numerical-reference Shu–Osher cases remain subject to convergence checks.
These finite-suite outcomes do not establish general superiority. Neural-only
advection grows dramatically while remaining finite (canonical RMS error about 475).
Finiteness and conservation alone are not stability proofs. Fixed blend also beats
the learned gate on some configurations, including canonical Euler, so gains cannot
all be attributed to learning the mixing head.


Warm-start attribution controls (the same predeclared constant blend):

| Case | Warm-start fixed | Final fixed | Final learned |
|---|---:|---:|---:|
| canonical-advection | 0.0977368001 | 0.0974064292 | 0.0955212916 |
| canonical-burgers | 0.000696104663 | 0.000712430673 | 0.00066492691 |
| canonical-shu-osher | 0.209076054 | 0.209236937 | 0.209263935 |
| canonical-sod | 0.00831396434 | 0.00828539063 | 0.00828709346 |

The warm-start constant blend already improves the two scalar canonical cases and
the historical Shu–Osher comparison. It is better than the final learned head on
canonical Shu–Osher under that reference. Scalar raw neural-only advection changes
from error 0.14018 before training to about 475 after training through the blend.
The blended method must therefore be evaluated as the actual solver; the raw
backbone is not independently demonstrated to be a stable reconstruction. These
controls do not establish that learning the small mixing head is necessary for
the Euler difference. [Full controls](../src/differentiable/validation/p0-2026-10-14/warmstart-controls/) retain every result.

## Repair: failing regression first, then a narrow correction

The two new positive-state/neighbor-average tests failed on the original code,
then passed after the fix. Logs and original source are retained in
[baseline evidence](../src/differentiable/validation/p0-2026-10-14/baseline/).
The correction applies the existing last-resort floor/cap only to rows still invalid
after neighbor averaging. Admissible states, including density/pressure below 0.05,
are returned unchanged. No floor, cap, neighbor stencil, or redistribution parameter
was added or tuned. Default benchmark evaluation remains unguarded with no automatic
retry. Repaired results must identify modified rows and conservative-state changes.

| Assessment | Unguarded error | Historical guard error | Corrected guard error | Corrected changed stage rows |
|---|---:|---:|---:|---:|
| canonical-sod | 0.00828709346004 | 0.00828709346004 | 0.00828709346004 | 0 |
| canonical-shu-osher | 0.209263935454 | 0.209263935454 | 0.209263935454 | 0 |
| low-positive-entropy-wave | 6.19980533686e-10 | 0.0466452578528 | 6.19980533686e-10 | 0 |

For the low-positive entropy wave, the historical guard adds about 0.0400 integrated
mass and 0.0999 energy. The correction makes no repairs and preserves the unguarded
accuracy and balance to roundoff. Canonical Euler errors differ only at roundoff.
All 48 official unguarded method runs have finite raw substages and zero repairs;
all Euler stage densities and pressures are positive. Scalar finiteness does not
establish positivity or stability. Conservation is measured using the actual numerical boundary fluxes and
SSPRK3 stage weights; repair contributions receive their corresponding RK weights.
Canonical learned-blend Euler balance residuals are at most about 2.1e-12.

**Limitation:** neighbor averaging and the retained invalid-cluster fallback are not
globally conservative. Their changes are explicitly measured; no conservation claim
is made for repaired states. The historical neighbor rule is periodic, even when
used as an optional diagnostic on a transmissive problem. A boundary-aware,
conservative positivity method requires a separate design/review before any broader
algorithmic change. No such change is included here.

The historical fixed Euler timestep uses the initial speed and safety factor.
For canonical Sod, nominal CFL 0.4 yields measured maximum stage CFL about 0.57355.
This is reported, not silently changed to an adaptive benchmark.

## Smooth convergence

Spatial derivative studies use 32/64/128/256 finite-volume cells. Rollout studies
use 32/64/128 cells and dt proportional to dx^(5/3), with exact cell-average scalar
advection / Euler entropy-wave references. The 12-point quadrature and settings
were fixed before evaluation. These are distinct from the historical benchmarks.

| System | Method | Last spatial L1 order | Last rollout L1 order |
|---|---|---:|---:|
| advection | teno5 | 4.9952 | 4.9819 |
| advection | neural-only | 2.9943 | 2.9740 |
| advection | fixed-blend | 2.9551 | 2.8135 |
| advection | learned-blend | 2.9683 | 2.8685 |
| euler | teno5 | 4.9957 | 4.9758 |
| euler | neural-only | 3.0064 | 2.3334 |
| euler | fixed-blend | 2.9730 | 2.8149 |
| euler | learned-blend | 2.9734 | 2.8165 |

[Full convergence data](../src/differentiable/validation/p0-2026-10-14/reproduction/smooth/smooth_summary.json)
and PNG/PDF plots are exported. The neural methods approach about third spatial
order on these grids; fifth-order accuracy must not be claimed for them. A small
constant blend can retain a lower-order neural error term. Single-grid improvements
do not override this measured limitation.

## Shu–Osher reference convergence and sensitivity

The historical 320-cell result is retained verbatim as a separately named comparison.
It gives ratio 0.9815878248. On the canonical 320/640/1280/2560/5120 reference ladder,
ratios are approximately 0.981588 / 0.983678 / 0.984224 / 0.984703 / 0.984990.
TENO5's mean-state error changes from 0.213189 at 320 reference cells to 0.267192 at
5120. The last target-grid Cauchy difference is about 0.002398, compared with a method
error margin about 0.004011. This is too large for a certified win.

The predeclared temporal sequence uses dt, dt/2, dt/4 at 5120 cells. The empirical
ranking criterion requires a consistent recent sign and twice the sum of spatial
and temporal target-grid changes below 25% of the final method margin. This is an
empirical Cauchy criterion, not a rigorous error bound. It is not relaxed to recover
a favorable result. Additional held-out Shu–Osher studies use their own references.

| Study | Status | Finest learned/TENO5 | Spatial target change | Temporal target change |
|---|---|---:|---:|---:|
| canonical | reference-unresolved | 0.98479793 | 0.002398251 | 0.001020155 |
| heldout160 | reference-unresolved | 0.96841422 | 0.005308024 | 0.0007644248 |
| heldout240 | reference-unresolved | 0.92650492 | 0.004456863 | 0.0007097436 |

Independent NumPy WENO7 references were also run at 1280/2560 cells, adaptive CFL
0.35, in the clean numerical environment, with positivity validation at every RK
stage and zero repairs. Canonical learned/TENO5 ratios are 0.9845188 / 0.9847630.
The target-grid WENO7 reference change is 0.0035055, so this supports the observed
sign without independently certifying the narrow margin. Actual reference CFL is
recorded as `reference_metadata.cfl`; the nested case descriptor retains the
canonical case's nominal CFL in this archived run.

A strict cross-snapshot comparison at 1e-12 reference tolerance failed for the two
fine held-out references: maximum sample changes were 0.00171979 and 0.000745516.
All 48 coarse method-state arrays were bit-for-bit identical. Reference changes
occurred across instrumentation snapshots; a compiler/graph effect is plausible
but its precise cause was not isolated. The failed check is retained, without
relaxing its tolerance, in
[cross-snapshot-verification.log](../src/differentiable/validation/p0-2026-10-14/cross-snapshot-verification.log).
Each archived run keeps its own source/environment fingerprint. This is another
reason to withhold an unconditional Shu–Osher win claim.

Completed [reference JSON](../src/differentiable/validation/p0-2026-10-14/reference-aggregate/reference_summary.json),
[condensed results](../src/differentiable/validation/p0-2026-10-14/summary/summary.json),
[comparison CSV](../src/differentiable/validation/p0-2026-10-14/summary/comparisons.csv),
and [canonical blend overlays](../src/differentiable/validation/p0-2026-10-14/summary/canonical_blends.png)
are available for review. Each individual reference study exports convergence plots
and full reference states. The canonical study's original multi-case worker was
interrupted after that study completed because standalone workers completed the
two held-out studies; its exit 130 and preserved partial output are operational,
not an unreported numerical failure.

## Changed implementation and remaining limits

- Strict loading/inference: `model_io.py`, `reconstruction.py` and adapters in the
  historical scalar, Euler and presentation evaluators. Historical scalar imports
  no longer load models or execute simulations. Training scripts were not changed.
- Evaluation/export/verification: `benchmark_cases.py`, `benchmarks.json`,
  `diagnostics.py`, `evaluation.py`, `evaluate.py`, `artifacts.py`,
  `verify_artifacts.py`, `summarize.py`, and the `Evaluate_Benchmarks.py` entry point.
- Scientific studies/reproduction: `reference_study.py`, `numpy_reference.py`,
  `smooth_study.py`, `repair_study.py`, `warmstart_control.py`, `reproduce.py`.
- Numerical correction: only the justified repair mask; default flux calculations
  are preserved, with optional boundary-flux/repair observations for diagnostics.
- Added tests: checkpoint schema/failure contracts, ablation endpoints, import/CLI
  isolation, final-model conservation, stage positivity, repair counts/budgets,
  positive-state regression, smooth convergence, and original-solver parity.
- Documentation/dependencies: README, additive Results.md note, repaired image links,
  this report and three clean-environment dependency locks. The old legacy/numerical
  top-level pins remain intact; JAX now uses a validated complete lock.

No universal accuracy/order/stability claim, conservative repaired scheme, rigorous
reference error bound, GPU/platform portability guarantee, or complete historical
training provenance has been established. The original phase history and rejected
experiments remain available. Public release and committing/pushing require the
owner's review.
