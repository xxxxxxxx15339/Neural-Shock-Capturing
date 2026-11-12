# Verified four-method results

Evidence status for all following scores: **NEW MATCHED EVALUATION**. Full precision
values and failure records are in `results/raw/evaluated_rows.json`; every displayed
number is generated from that file. The hashed protocol preceded the tests.
122 measured rollouts include 108 main-method cases and
same-backbone controls; 21 failures are retained.

## Model identities

TENO5 is the unchanged classical implementation. NENO6 is the verified
paper-linked author checkpoint (6→64→32→16→8→4), ported to the common solver.
NN-TENO5 is the original Keras HDF5 classifier with calibrated hard thresholds.
V2 is **phase21 rich learned head for Euler**, **run6 linear learned head for scalar**.
These are different physics-specific checkpoints, not one universal model.
Euler applies the same learned head separately to each of three characteristic
fields and both flux-splitting directions, giving six lambdas per interface.
The scalar blending formula applies to each such reconstruction before projection
to one shared conservative face flux. Plotted means average these six values;
they are not a single fluid-level learned coefficient.
The later dataset-V2 fixed-lambda checkpoint is excluded from this identity.
See `AUDIT.md` and `CHECKPOINT_INVENTORY.csv` for checksums and training provenance.

## Main comparison

| Benchmark | N | Metric | TENO5 | NENO6 | NN-TENO5 | V2 learned head |
| --- | --- | --- | --- | --- | --- | --- |
| sod | 128 | density_l1 | 0.004452127 | 0.004234734 | 0.004434132 | 0.004455158 |
| sod | 128 | l1 | 0.006344701 | 0.006111865 | 0.006303808 | 0.006351273 |
| lax | 128 | l1 | 0.02632591 | 0.02636503 | 0.02597659 | 0.02578272 |
| shu_osher | 128 | l1 | 0.2304674 | 0.2097592 | 0.2280766 | 0.2270421 |
| shu_osher | 128 | regional_density_l1 | 0.3560382 | 0.3426656 | 0.3569309 | 0.3525779 |
| burgers_pre | 128 | l1 | 3.906289e-05 | 3.570585e-05 | 3.906289e-05 | 3.107954e-05 |
| burgers_post | 128 | l1 | 0.001818696 | 0.001478109 | 0.001682797 | 0.001825657 |
| burgers_hat | 128 | l1 | 0.006151853 | 0.005839439 | 0.006177979 | 0.005864129 |
| advection_square | 128 | l1 | 0.01865898 | 0.01302991 | 0.0188542 | 0.0176154 |
| advection_smooth | 256 | l1 | 7.599293e-10 | 3.915144e-07 | 7.599293e-10 | 1.640334e-07 |

Euler `l1` is the arithmetic mean of density, momentum and energy L1. Scalar
`l1` is mean absolute scalar error. Entries from different norms are not averaged
into a headline percentage. NN-TENO5 and learned-head backbones differ.

## Percentage improvement (positive means smaller error)

| Benchmark | N | Metric | NENO6 vs TENO5 | NN vs TENO5 | V2 vs TENO5 | V2 vs NN |
| --- | --- | --- | --- | --- | --- | --- |
| sod | 128 | density_l1 | +4.883% | +0.404% | -0.068% | -0.474% |
| sod | 128 | l1 | +3.670% | +0.645% | -0.104% | -0.753% |
| lax | 128 | l1 | -0.149% | +1.327% | +2.063% | +0.746% |
| shu_osher | 128 | l1 | +8.985% | +1.037% | +1.486% | +0.454% |
| shu_osher | 128 | regional_density_l1 | +3.756% | -0.251% | +0.972% | +1.220% |
| burgers_pre | 128 | l1 | +8.594% | +0.000% | +20.437% | +20.437% |
| burgers_post | 128 | l1 | +18.727% | +7.472% | -0.383% | -8.489% |
| burgers_hat | 128 | l1 | +5.078% | -0.425% | +4.677% | +5.080% |
| advection_square | 128 | l1 | +30.168% | -1.046% | +5.593% | +6.570% |
| advection_smooth | 256 | l1 | -51419.851% | +0.000% | -21485.356% | -21485.356% |

V2-vs-NN describes the whole saved method difference. It does not isolate the head.
Zeros/ties and regressions are preserved; no threshold or coefficient was fitted
to these tests. Archived results have different settings and remain separate.

## Same-backbone causal control

All three columns below share the selected phase21/run6 backbone. Fixed lambda
is the predeclared sigmoid(-3)=0.047425873177566781.

| N=128 case | Pure same backbone | Fixed same backbone | Learned same backbone | Learned vs pure | Learned vs fixed |
| --- | --- | --- | --- | --- | --- |
| sod | FAILED | 0.006350602 | 0.006351273 | N/A | -0.011% |
| lax | FAILED | 0.02578176 | 0.02578272 | N/A | -0.004% |
| shu_osher | FAILED | 0.2269573 | 0.2270421 | N/A | -0.037% |
| burgers_pre | 0.0001787713 | 3.363061e-05 | 3.107954e-05 | +82.615% | +7.586% |
| burgers_post | 0.01241355 | 0.001817578 | 0.001825657 | +85.293% | -0.445% |
| burgers_hat | 0.2157578 | 0.005946483 | 0.005864129 | +97.282% | +1.385% |
| advection_square | 2.337324e+07 | 0.01781522 | 0.0176154 | +100.000% | +1.122% |

Positive learned-vs-fixed values quantify the incremental value of an adaptive
head over this specific fixed coefficient. Negative values reject that advantage
on that case. This is not an optimized-fixed-head comparison or a general head
capacity theorem. A pure-backbone failure demonstrates the value of reducing the
neural contribution under this protocol, not that learning lambda is necessary.

## Physical admissibility and discrete conservation

| Method | All completed | Stress completed | Max budget residual | Raw min rho incl failures | Raw min p incl failures |
| --- | --- | --- | --- | --- | --- |
| TENO5 | 25/27 | 8/10 | 1.08451e-11 | 8.819782e-06 | -0.03464694 |
| NENO6 | 23/27 | 6/10 | 1.338663e-11 | 8.445584e-06 | -0.003348447 |
| NN-TENO5 | 21/27 | 4/10 | 9.265477e-12 | 8.82176e-06 | -0.03080745 |
| NN-TENO5-V2 (Learned Mixing Head) | 21/27 | 4/10 | 9.691803e-12 | -0.001017636 | -0.00341922 |

Raw minima include the first invalid stage. These minima are not post-repair
values. Completed fractions include every declared case. Conservation residuals
include boundary exchange and are measured for accepted steps up to the stopping
time. Wall mass/energy fluxes are explicitly zero; momentum need not be conserved
inside a reflecting box because wall pressure exchanges momentum.

| Case | N | Method | First failed step | Stage | Last complete time | Raw min rho | Raw min p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| strong_contrast | 128 | TENO5 | 9 | 1 | 0.0032 | 0.01 | -0.03464694 |
| strong_contrast | 64 | TENO5 | 9 | 1 | 0.0064 | 0.01 | -0.03464694 |
| blast_full | 128 | NENO6 | 1 | 3 | 0 | 0.9299874 | -0.003348447 |
| blast_full | 64 | NENO6 | 1 | 3 | 0 | 0.9300246 | -0.003306109 |
| blast_smoke | 128 | NENO6 | 1 | 3 | 0 | 0.9300991 | -0.003221964 |
| blast_smoke | 64 | NENO6 | 1 | 3 | 0 | 0.9315156 | -0.001749142 |
| blast_full | 128 | NN-TENO5 | 5 | 3 | 0.0001605069 | 0.8670659 | -0.002759457 |
| blast_full | 64 | NN-TENO5 | 5 | 3 | 0.0003206751 | 0.8672212 | -0.002642827 |
| blast_smoke | 128 | NN-TENO5 | 5 | 3 | 0.00016 | 0.8675309 | -0.002412722 |
| blast_smoke | 64 | NN-TENO5 | 5 | 3 | 0.0003076923 | 0.8778964 | -0.0004771119 |
| strong_contrast | 128 | NN-TENO5 | 7 | 1 | 0.0024 | 0.008727534 | -0.03080745 |
| strong_contrast | 64 | NN-TENO5 | 7 | 1 | 0.0048 | 0.008727534 | -0.03080745 |
| blast_full | 128 | NN-TENO5-V2 (Learned Mixing Head) | 3 | 1 | 8.025343e-05 | 0.8714894 | -0.00341922 |
| blast_full | 64 | NN-TENO5-V2 (Learned Mixing Head) | 3 | 1 | 0.0001603376 | 0.8716249 | -0.003403366 |
| blast_smoke | 128 | NN-TENO5-V2 (Learned Mixing Head) | 3 | 1 | 8e-05 | 0.8718952 | -0.003371484 |
| blast_smoke | 64 | NN-TENO5-V2 (Learned Mixing Head) | 3 | 1 | 0.0001538462 | 0.8768223 | -0.002721815 |
| strong_contrast | 128 | NN-TENO5-V2 (Learned Mixing Head) | 1 | 1 | 0 | -0.001017636 | 0.01 |
| strong_contrast | 64 | NN-TENO5-V2 (Learned Mixing Head) | 1 | 1 | 0 | -0.001017636 | 0.01 |

All failed candidates and last admissible states are retained in NPZ files.
No finite full-time error is assigned to failure. No direct state clipping,
averaging, positivity repair, retry or low-order fallback is used in these runs.
This is not a positivity theorem. Strong-blast scores are stability only.
For the point grid, mirrored outer faces lie half a mesh beyond the first/last
sample: these are Woodward–Colella-style stress cases, not certified canonical
blast accuracy results on a wall-aligned cell-average grid.

## Smooth full-solution convergence

| Method | N | L1 | Observed p |
| --- | --- | --- | --- |
| TENO5 | 64 | 7.736965e-07 | N/A |
| TENO5 | 128 | 2.429129e-08 | 4.993257 |
| TENO5 | 256 | 7.599293e-10 | 4.99843 |
| NENO6 | 64 | 2.160363e-05 | N/A |
| NENO6 | 128 | 3.412452e-06 | 2.662393 |
| NENO6 | 256 | 3.915144e-07 | 3.123672 |
| NN-TENO5 | 64 | 7.736965e-07 | N/A |
| NN-TENO5 | 128 | 2.429129e-08 | 4.993257 |
| NN-TENO5 | 256 | 7.599293e-10 | 4.99843 |
| NN-TENO5-V2 (Learned Mixing Head) | 64 | 9.675342e-06 | N/A |
| NN-TENO5-V2 (Learned Mixing Head) | 128 | 1.290793e-06 | 2.906054 |
| NN-TENO5-V2 (Learned Mixing Head) | 256 | 1.640334e-07 | 2.976196 |

The transport problem has exact references and dt≤.15*dx^(5/3). These are observed
full-solution orders, not formal guarantees for Euler/Burgers or shock problems.
SSPRK3 remains third order in time. NENO6 means six-point stencil; the author
optimized coefficients and finite-precision implementation must be judged by
their observed convergence, not the digit in the name. No state-flux solver
correction is introduced into this comparison.

## Shu–Osher numerical-reference sensitivity

| Comparison at target grid | N | Mean conserved L1 difference | Density L1 difference |
| --- | --- | --- | --- |
| reference_shu_512_dt1 → reference_shu_1024_dt1 | 128 | 0.03246127 | 0.0117153 |
| reference_shu_1024_dt1 → reference_shu_2048_dt1 | 128 | 0.01143464 | 0.004658897 |
| reference_shu_2048_dt1 → reference_shu_2048_dt2 | 128 | 0.0009213284 | 0.0003113529 |
| reference_shu_512_dt1 → reference_shu_1024_dt1 | 256 | 0.03159318 | 0.01150014 |
| reference_shu_1024_dt1 → reference_shu_2048_dt1 | 256 | 0.01433586 | 0.005424723 |
| reference_shu_2048_dt1 → reference_shu_2048_dt2 | 256 | 0.0009463725 | 0.0003219554 |

The final reference is TENO5 N=2048 at dt/2, interpolated to evaluation points.
These differences estimate sensitivity, not rigorous uncertainty bounds. Small
reported gains of the same size as reference changes are unresolved as accuracy
claims. `reference_sensitivity` in every Shu row records errors against each
ladder member, so ranking sensitivity is inspectable. An exact shock-entropy
solution and an independently different reference solver are unavailable.

## Mixing-head behavior

| Case | N | Mean lambda | Min | Max | Std | Smooth mean | Shock mean | Corr local error reduction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shu_osher | 128 | 0.04736398 | 0.04734963 | 0.04737201 | 4.879264e-06 | 0.04736403 | 0.04736487 | 0.08493795 |
| shu_osher | 256 | 0.04736385 | 0.04734963 | 0.04737201 | 4.572311e-06 | 0.04736389 | 0.04736367 | -0.005859364 |
| sod | 128 | 0.0473646 | 0.04734963 | 0.04737201 | 5.094596e-06 | 0.04736577 | 0.04736378 | 0.06830312 |
| sod | 256 | 0.0473658 | 0.04734963 | 0.04737201 | 5.085935e-06 | 0.04736801 | 0.04736316 | -0.06246844 |

Smooth/shock masks are the lower half/top decile of measured density-gradient
magnitudes. Correlations are descriptive across dependent spatial samples.
There is no calibrated-uncertainty interpretation. The rich head uses only
three probabilities and five normalized split-flux values, not Mach, pressure
gradient or the entire proposed physics vector. The same-backbone ablation,
not lambda variability by itself, determines whether adaptivity helps.

## Computational costs

| N=128 case | Method | Warm median s | IQR s | Cost / TENO5 | Rollout compile s |
| --- | --- | --- | --- | --- | --- |
| sod | TENO5 | 0.05646431 | 0.002053257 | 1 | 0.6936175 |
| sod | NENO6 | 0.1264928 | 0.002277979 | 2.240225 | 0.7246337 |
| sod | NN-TENO5 | 0.09038581 | 0.001025756 | 1.60076 | 0.6763609 |
| sod | NN-TENO5-V2 (Learned Mixing Head) | 0.1542487 | 0.004310284 | 2.731791 | 1.001177 |
| lax | TENO5 | 0.0947168 | 0.003475864 | 1 | 0.7555563 |
| lax | NENO6 | 0.2110969 | 0.00681515 | 2.228716 | 0.6940235 |
| lax | NN-TENO5 | 0.152864 | 0.001466608 | 1.613906 | 0.6656156 |
| lax | NN-TENO5-V2 (Learned Mixing Head) | 0.2508784 | 0.007336461 | 2.648722 | 0.983587 |
| shu_osher | TENO5 | 0.2022946 | 0.006539582 | 1 | 0.7006409 |
| shu_osher | NENO6 | 0.433792 | 0.005529716 | 2.144358 | 0.7022234 |
| shu_osher | NN-TENO5 | 0.3224945 | 0.03090423 | 1.594183 | 0.8331748 |
| shu_osher | NN-TENO5-V2 (Learned Mixing Head) | 0.5210416 | 0.004909959 | 2.575658 | 0.9853838 |
| burgers_post | TENO5 | 0.01863563 | 0.000215432 | 1 | 0.3144892 |
| burgers_post | NENO6 | 0.1135222 | 0.000483107 | 6.091672 | 0.3334667 |
| burgers_post | NN-TENO5 | 0.07127811 | 0.001067732 | 3.824829 | 0.2937025 |
| burgers_post | NN-TENO5-V2 (Learned Mixing Head) | 0.1181338 | 0.0009469375 | 6.339139 | 0.5051977 |
| advection_square | TENO5 | 0.02053167 | 0.000704092 | 1 | 0.3147183 |
| advection_square | NENO6 | 0.1237461 | 0.000621118 | 6.027084 | 0.3322725 |
| advection_square | NN-TENO5 | 0.07613754 | 0.000413543 | 3.708297 | 0.297284 |
| advection_square | NN-TENO5-V2 (Learned Mixing Head) | 0.1310679 | 0.0007056635 | 6.383696 | 0.4973586 |

Timings use one JAX CPU environment, synchronized three-repeat warmed device
rollouts; IQR describes repeat dispersion. Compilation and host diagnostic time
are separate. CPU inference ports are compared; this is not TensorFlow-vs-JAX
framework speed. Nearest measured cost comparisons across128/256 appear in
`results/nearest_measured_budget.csv`; only rows flagged within25% have roughly
comparable costs, and no interpolated equal-cost accuracy is invented.

## Archived results and reproduction discrepancies

The evidence catalog records every archived scalar with source hash/JSON pointer.
Original scalar and phase18 Euler entry points were rerun into this directory.
Across 160 checked values, 19 exceed
1e-10, with maximum absolute discrepancy 0.000845570185.
The phase18 Euler values, classical scalar results and smooth neural scalar
results reproduce. The original-network square-wave result and its threshold
controls differ slightly; current CPU repeats are deterministic. Archive identity
metadata are insufficient to establish the cause. They remain **ARCHIVED /
REPRODUCTION DISCREPANCY**, not silently marked reproduced. The threshold=1
control is not unconditional all-keep because exactly saturated sigmoid outputs
still meet the >= discard rule. Archived Burgers T=.6 is pre-shock (2/pi≈.6366),
despite its historical smooth-to-shock name. A fidelity-factor acceptance is not
necessarily an improvement. See `results/archive_reproduction.csv`.

## Interpretation and limitations

This establishes reproducible, case-specific differences among authentic saved
reconstructions under a common solver. It separates whole-method and same-backbone
head effects and retains failures and computational costs. It does not establish
universal neural superiority, positivity/TVD/entropy guarantees, fifth-order
nonlinear Euler accuracy, multi-seed training generalization or untouched-test
performance. No trained artifact is promoted from this small comparative suite.
The strongest existing baseline should be chosen per application only with
independent validation. Source/checkpoint integrity and tests are recorded in
`results/raw/package_verification.json` and `logs/`.

## Reproduction

Run `bash experiments/four_method_comparison/reproduce.sh`. Existing verified
run caches are retained; numerical fixtures can be independently regenerated
with `REEXPORT=1`. Exact individual commands and environment paths are in that
script and `requirements_notes.md`. No training, checkpoint writing or GitHub
push is part of reproduction.

## Measured conclusions and diagnostic controls

The scalar learned head improves pre-shock Burgers N=128 by
+7.586% relative to the **same-backbone
fixed blend**, in addition to its main-model improvement. This is a genuine
case-specific incremental head result for the older scalar3→1 head. It does not
establish a benefit for the richer Euler head, which slightly worsens all three
N=128 Euler ablations relative to the predeclared fixed blend. The Euler lambda
range is only about2.24e-5, around .04736, so it behaves almost like fixed mixing.

The smooth control is a material regression: scalar V2 L1 at N=256 is
1.640334e-07, versus 7.599293e-10 for TENO5/original NN,
an error ratio of 215.854. The absolute errors remain small, but the
observed order drops to about3 instead of5. The main extremum statistics include
this control; it is not omitted to make neural gains look larger.

| N=256 initial smooth profile | Any-discard fraction | Discarded candidate fraction | Mean lambda |
| --- | --- | --- | --- |
| TENO5 | 0 | 0 | N/A |
| NENO6 | 0.5859375 | 0.1640625 | N/A |
| NN-TENO5 | 0 | 0 | N/A |
| NN-TENO5-V2 (Learned Mixing Head) | 1 | 0.3333333 | 0.06943573 |

These rejection fractions are measured on the analytically smooth initial
stencils. They identify active neural stencil removal where classical selection
keeps all candidates. This is consistent with lower-order neural contributions
controlling the nonzero blend's asymptotic error; it is a diagnostic mechanism,
not a proof of the full nonlinear scheme's order. The author
[NENO6 paper, section4.1](https://arxiv.org/pdf/2207.08500) uses an optimized
**fourth-order** background constraint, not sixth order; its full neural method's
order must still be measured. Our observed NENO6 smooth orders are in the table.

The separately predeclared finest-grid time-step control gives:

| N=256 method | L1 at dt | L1 at dt/2 | Solution change L1 | Change / original error |
| --- | --- | --- | --- | --- |
| TENO5 | 7.59929266649223e-10 | 7.599318549650694e-10 | 2.7523979189564285e-13 | 0.00036219133013427055 |
| NENO6 | 3.91514423769509e-07 | 3.915145696868584e-07 | 5.950324976668198e-11 | 0.00015198226720176348 |
| NN-TENO5 | 7.59929266649223e-10 | 7.599318549650694e-10 | 2.7523979189564285e-13 | 0.00036219133013427055 |
| NN-TENO5-V2 (Learned Mixing Head) | 1.640334399762456e-07 | 1.6403343398852226e-07 | 2.7616027954008304e-13 | 1.6835608616150159e-06 |

The solution changes are small compared with the measured smooth errors. Thus
the observed order loss is not explained by the tested SSPRK3 timestep alone.
Scaling dt as dx^(5/3) makes temporal truncation scale as dx^5; the dt/2 test
quantifies its actual contribution. This is not fifth-order temporal convergence.

Finite completion does not imply scalar stability. The pure scalar head-backbone
advection control grows catastrophically while remaining finite, and is retained
with its full large error and an amplification warning:

| Case | N | Method | Finite final L1 | Max abs state | Amplification |
| --- | --- | --- | --- | --- | --- |
| advection_square | 128 | Same backbone: pure NN | 2.337324e+07 | 6.062478e+07 | 6.062478e+07 |

The tenfold amplification flag is descriptive and does not terminate the run or
replace its error. Its absolute budget residual is retained, and roundoff
normalization accounts for the very large absolute state magnitudes. No claim
of a physically useful completed solution is made for that control.

### Comparable measured computational budgets

| Case | Neural method / N | TENO5 N | Neural seconds | TENO5 seconds | Time ratio | Error ratio |
| --- | --- | --- | --- | --- | --- | --- |
| shu_osher | NN-TENO5 / 256 | 256 | 1.231239 | 1.022694 | 1.203917 | 0.9897521 |
| burgers_post | NN-TENO5 / 128 | 256 | 0.07127811 | 0.07044948 | 1.011762 | 1.577093 |

These are measured pairs within25% in runtime, without cost interpolation. For
post-shock Burgers, original NN at N=128 costs almost the same as TENO5 N=256
and has about1.58 times its error. Thus its same-grid accuracy advantage does
not translate into an advantage at this measured computational budget. The Shu
pair costs more and its small error difference remains reference-sensitive.

### Physical wall-alignment control

The initial historical point grid puts reflected outer faces at half nodes.
A separate protocol was declared before running a wall-aligned control on
x=(i+.5)dx, physical walls0/1, the same pressure regions and all four methods.
This supplements rather than replaces the original stress results. It performs
no clipping, uses the same wall-flux contract and has no accuracy reference.

| Wall-aligned case | N | Method | Completed | Last complete time | Raw min pressure | First failed stage |
| --- | --- | --- | --- | --- | --- | --- |
| blast_wall_aligned_0.001 | 128 | TENO5 | True | 0.001 | 0.009999999999999998 |  |
| blast_wall_aligned_0.001 | 128 | NENO6 | False | 0.0 | -0.003221963909670545 | 3 |
| blast_wall_aligned_0.001 | 128 | NN-TENO5 | False | 0.00016 | -0.0024127223974142503 | 3 |
| blast_wall_aligned_0.001 | 128 | NN-TENO5-V2 (Learned Mixing Head) | False | 8e-05 | -0.0033714838086982626 | 1 |
| blast_wall_aligned_0.038 | 128 | TENO5 | True | 0.038 | 0.009999999999999998 |  |
| blast_wall_aligned_0.038 | 128 | NENO6 | False | 0.0 | -0.0033484473056493882 | 3 |
| blast_wall_aligned_0.038 | 128 | NN-TENO5 | False | 0.00016050686378035902 | -0.0027594568502862134 | 3 |
| blast_wall_aligned_0.038 | 128 | NN-TENO5-V2 (Learned Mixing Head) | False | 8.025343189017951e-05 | -0.003419219686900057 | 1 |

Source: `configs/wall_geometry_control.json` and
`results/wall_geometry_control.csv`. These 8 supplemental runs
include 6 raw stage failures. They confirm
that the neural blast failures persist under physical wall alignment. Strong
contrast is complementary: NENO6 completes the declared cases while TENO5,
original NN and learned head fail; all methods have regime-dependent robustness.

### Validation and preservation

`logs/cpu_regression.log` records the complete CPU regression suite; legacy neural
construction/load tests are in `logs/legacy_smoke.log`. The initial new entropy
test incorrectly demanded exact mass quadrature from arbitrary point samples of
a discontinuity; it was corrected to symmetric moving-frame samples with the
same strict tolerance. The numerical implementation was unchanged. Initial logs
are preserved and documented, including a supplementary filename collision
fixed with unique sanitized cache keys and identity checks before using results.
The Lax contact diagnostic initially selected the stronger nearby shock inside
its broad window. Exact-wave-separation caps and a dedicated exact-Lax regression
test correct this; the primary L1 errors and percentage comparisons are unchanged.

Original metadata thresholds are compared at their exact float64 values against
float32 classifier outputs. All27 original-network cases were rerun after this
precision audit: no final-state value or success status changed. Initial Sod
timings were repeated without overlapping archival numerical jobs, with identical
final states. All first-pass outputs are retained in preflight subdirectories.
Actual solver-input Keras mask/probability parity is checked separately in
`results/raw/actual_input_parity.json` in addition to the initial random fixtures.

The general recommended baseline remains **classical TENO5**. Neither saved
learned-head specialization is supported as a universal upgrade: scalar gains
are case-specific, Euler adaptivity offers no measured improvement over fixed
mixing here, smooth accuracy regresses and stress failures remain. NENO6 is a
verified research comparator, with favorable and unfavorable cases of its own.
No model was retrained or promoted on these exposed tests.
