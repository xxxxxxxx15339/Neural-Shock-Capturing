# WENO / TENO Neural-Network Schemes — Consolidated Results

Single reference document collecting every numerical comparison and
experiment run across this project: classical vs. neural-network-augmented
WENO5 and TENO5 schemes, on scalar and Euler test problems, plus the
Shu–Osher investigation and retraining attempts.

**Models and methods compared:**

| Method | Type | Model file | Notes |
|---|---|---|---|
| WENO5 (Jiang–Shu) | Classical | — | Baseline, no learning |
| WENO5-NN | Full replacement | `SeemsGood3.h5` | Network directly predicts the reconstructed interface value (Stevens & Colonius style); trained on periodic scalar advection data |
| TENO5 | Classical | — | Baseline, no learning |
| NN-TENO5 | Calibrated hybrid | `models/nn_teno5.h5` | Network decides, per characteristic field, whether to keep/discard candidate stencils (calibrated thresholds `[0.66, 0.50, 0.69]`); falls back toward TENO5's own logic rather than replacing it outright |
| WENO5-NN (Euler, retrained) | Full replacement | *(none accepted)* | Three retraining attempts on Euler-specific data, see Part 4 — none passed the acceptance bar |

---

## Part 1 — WENO5 vs. WENO5-NN

Model: `SeemsGood3.h5` (upstream reference; the default `trained_weno5.h5` is
not present in this checkout).

### 1.1 Periodic step advection
*(Sec. 4.1 setup, reduced: nx=100, 5 periods instead of the paper's 50, for tractable runtime)*

| Metric | WENO5 | WENO5-NN | Δ |
|---|---|---|---|
| L1 error | 4.267e-02 | 3.042e-02 | **-28.7%** |
| L2 error | 1.114e-01 | 9.238e-02 | **-17.1%** |
| L∞ error | 4.294e-01 | 4.072e-01 | -5.2% |
| Max total variation (exact = 2.0) | 2.01042 | 2.00496 | overshoot 0.52% → 0.25% |
| Max shock width | 0.56 | 0.44 | **-21.4% (sharper)** |
| Runtime | 1.06 s | 5.85 s | NN ~5.5× slower |

**Clear win for the NN** on every accuracy metric.

### 1.2 Burgers' equation convergence
*(Sec. 4.2, reduced reference grid: 400 cells instead of the paper's 1600/6400)*

| Cells | Δx | E (WENO5-JS) | E (WENO5-NN) | Δ |
|---|---|---|---|---|
| 25 | 8.0e-2 | 3.698e-02 | 3.489e-02 | -5.7% |
| 50 | 4.0e-2 | 2.407e-02 | 2.262e-02 | -6.0% |
| 100 | 2.0e-2 | 1.472e-02 | 1.404e-02 | -4.6% |
| 200 | 1.0e-2 | 5.558e-03 | 5.609e-03 | +0.9% |

NN wins modestly at coarse resolution; the advantage shrinks and slightly
reverses (within noise) as the grid refines.

### 1.3 Euler Shu–Osher
*(Sec. 4.3, paper's own settings: 300 cells, reference 3200 cells, t=2.0, CFL=0.4)*

```
max_abs_error(WENO5    vs. reference) = 12.278
max_abs_error(WENO5-NN vs. reference) = 12.345   (+0.55%, NN slightly worse)
max_abs_difference(WENO5, WENO5-NN)   = 0.531
```

No advantage for the NN — a near-tie, marginally unfavorable. This result
motivated the investigation in Part 3.

### 1.4 Summary

The NN's advantage is **problem- and resolution-dependent**: strong on the
simple single-discontinuity linear case, shrinking on the nonlinear Burgers
shock as resolution increases, absent on the harder multi-wave Euler
problem — at a consistent ~5× runtime cost.

---

## Part 2 — TENO5 vs. NN-TENO5

Model: `models/nn_teno5.h5`, calibrated thresholds `[0.66, 0.50, 0.69]`,
`fidelity_factor = 1.02` (NN must not exceed 1.02× TENO5's own L1 error to
pass). Cells = 200, CFL = 0.4. Source: `Script_Compare_TENO5.py` →
`models/nn_teno5_scalar_comparison.json`.

| Problem | TENO5 L1 | NN-TENO5 L1 | Identical to TENO5? | TENO5 runtime | NN-TENO5 runtime |
|---|---|---|---|---|---|
| smooth_sine_advection | 3.328e-07 | 3.328e-07 | **Yes, bit-for-bit** | 0.87 s | 15.3 s |
| multifrequency_advection | 2.434e-05 | 2.434e-05 | **Yes, bit-for-bit** | 0.86 s | 15.1 s |
| burgers_smooth_to_shock | 1.328e-04 | 1.328e-04 | **Yes, bit-for-bit** | 0.52 s | 9.0 s |
| step_advection | 1.3317e-02 | 1.3402e-02 | No (+0.64% L1, but -0.43% L∞) | 0.86 s | 15.2 s |

All four problems **passed** the teacher-fidelity check (NN error ≤ 1.02×
TENO5's), and all four passed the threshold-wiring sanity check (forcing
thresholds to all-discard vs. all-keep produces meaningfully different
output — L1 differences from 0.012 to 0.121 — confirming the calibrated
gate is actually doing something, not a no-op).

**Reading:** on 3 of 4 scalar benchmarks, NN-TENO5 reproduces TENO5
*exactly*. On the one problem with a genuine discontinuity (step_advection),
it's a wash — L1 very slightly worse, L∞ very slightly better. This
demonstrates **fidelity/equivalence**, not superiority — the calibrated
network currently behaves as "TENO5, plus a validated pass-through gate,"
not as a scheme that measurably beats TENO5 on these benchmarks. Runtime
cost is substantial: NN-TENO5 is ~15-18× slower than TENO5 on these
problems.

---

## Part 3 — Diagnosing WENO5-NN's Shu–Osher underperformance

### 3.1 Method

Rather than comparing two independently evolved trajectories (which
accumulate drift), every `model.predict()` call during a real NN-driven
Shu–Osher run was intercepted and compared against what the **classical
WENO5 formula would produce on the identical normalized stencil** —
apples-to-apples, same input, two functions.

Setup: cells=100, T=1.0, CFL=0.4 → 2,106 network calls, 210,600 individual
stencil reconstructions logged.

### 3.2 Result: pointwise divergence is tiny

- mean |NN − WENO5| = 0.00126, max = 0.0058 (normalized range [0,1]).
- By characteristic field: field 1 (the **u** / contact–entropy-wave
  characteristic) has the highest average divergence (1.88e-3) vs. fields 0
  and 2 (u∓c, ~0.9–1.0e-3) — mild, not dramatic.

The network is not making gross mispredictions on typical interior stencils.

### 3.3 Where the worst divergences cluster: the domain boundary

Of the top-15 most divergent stencils, 7 sat at `x=0.000` — the left edge of
the domain under Shu–Osher's transmissive boundary condition. Their shapes
were "spike" patterns (e.g. `[1, 0, 0.33, 0, 0]`) and **100% non-monotonic**,
vs. 43% in the scalar training set. The network's original training data
(`data/2ndNewAvgs.csv`) came entirely from a **periodic-domain** scalar
problem — it had never seen a ghost-cell/transmissive-boundary stencil.

### 3.4 Confirmed in the data-generation code

`Script_GenerateEulerTrainingData.py` already had a proper Euler-specific
dataset (100k samples, correct Roe-characteristic Lax–Friedrichs split-flux
stencils matching `NNEuler5` at runtime, including Shu–Osher-like
trajectories) — but explicitly **excluded** every interface within 2 cells
of a domain edge. `Script_TrainOnEulerPrblms.py` (the trainer built for this
data) had never been run — no Euler-specific trained model existed anywhere
in the repo prior to this session.

---

## Part 4 — Retraining experiments (Euler-specific data)

All three used `Script_TrainOnEulerPrblms.py`'s built-in acceptance test:
the trained model must beat classical WENO5's Shu–Osher regional L1 error
(`x ∈ [5.5, 7.3]`, the post-shock oscillatory wave train) by **≥2%**, plus
pass physical-validity and Sod-problem checks.
`WENO5 Shu regional L1 baseline = 3.05831e-01` throughout; 3 attempts each
(reduced from the script's default max of 20).

| Attempt | Change made | Shu regional L1 (NN) | vs. required (≤2.99715e-01) |
|---|---|---|---|
| 1. Baseline | Train on existing interior-only Euler data as-is | 3.05850e-01 (+0.006% vs WENO5) | **Failed** |
| 2. + boundary data | Modified the generator to tag and include boundary-clipped stencils as their own labeled category (10% share); regenerated data (10,000 boundary samples, previously zero); retrained | 3.05849e-01 (+0.006%) | **Failed** |
| 3. + WENO7-referenced targets | For the `shock_oscillation` category only, retargeted training from WENO5-Z5's own reconstruction to a WENO7 reconstruction on a wider 7-point stencil at the same interface (network input space unchanged, still 5 points) | 3.05850e-01 (+0.006%) | **Failed** |

All three attempts converged to the **same value to 4 significant figures**
and passed every check except the Shu-Osher regional-improvement bar
(attempts 2 and 3 also intermittently failed `shu_total_variation_closer`).

### 4.1 Root cause (confirmed numerically)

Attempt 2→3 producing no change was traced to: in the `[5.5, 7.3]`
oscillatory region, WENO5 and WENO7 reconstructions agree almost exactly on
**99%+ of stencils** (both are high-order schemes converging to the same
answer on smooth, well-resolved flow):

```
n samples = 36,864
median |WENO5 - WENO7| target difference = 7.9e-09   (essentially zero)
90th percentile                          = 5.4e-05
99th percentile                          = 4.65e-03
fraction with difference > 0.01          = 0.64%
fraction with difference > 0.05          = 0.19%
```

The "room to beat WENO5" that a better teacher could exploit is concentrated
in under 1% of stencils, and gets drowned out in an aggregate MSE loss
computed over the full ~100k-sample dataset.

**Deeper structural issue:** `Script_GenerateEulerTrainingData.py` trains
the network to imitate a classical scheme's **own reconstructed value on
the same stencil** (distillation). A network optimized this way can only
ever match its teacher — genuine improvement can only emerge where the
network's learned manifold happens to diverge from the teacher in a
beneficial direction (as happened in Part 1.1's periodic-step case, but
evidently not here). Neither missing input coverage (boundary stencils) nor
a better teacher (WENO7) fixes this while the loss remains uniform,
aggregate MSE over the whole dataset.

---

## Cross-cutting conclusions

1. **No single scheme uniformly beats its classical counterpart.** WENO5-NN
   wins clearly on simple periodic problems, is a wash on nonlinear/coarse
   Burgers, and loses (marginally) on Euler Shu–Osher. NN-TENO5 reproduces
   TENO5 almost exactly on smooth problems and is a wash on the one
   discontinuous scalar case tested.
2. **Runtime cost is substantial and consistent**: NN variants are
   ~5–18× slower than their classical counterparts across every test run in
   this project, purely from per-cell network inference.
3. **The Shu–Osher underperformance has an identified, evidenced root
   cause**: (a) a real but non-decisive boundary-coverage gap in training
   data (fixed this session, made no measurable difference to the specific
   regional-L1 metric), and (b) a fundamental ceiling from training the
   network to imitate a classical scheme's own output — the two schemes
   being compared already agree on 99%+ of relevant stencils, so there is
   essentially no exploitable signal for "beating" WENO5 in that region
   under the current training objective.
4. **Untried next steps** (not yet attempted, scoped in prior discussion):
   hard-example reweighting/oversampling of the rare divergent stencils, a
   trajectory-level (rather than pointwise) training loss, or reframing the
   goal from "one network beats WENO5 everywhere" to a calibrated
   NN/classical hybrid gated by region — the pattern NN-TENO5 already
   demonstrates working (Part 2).

## Files touched / produced this session

- `Script_GenerateEulerTrainingData.py` — added `boundary` category,
  `weno7_reconstruction`, `use_reference_target` option, generalized stencil
  gathering, fixed a category-quota bug the boundary-category addition
  exposed.
- `data/eulerAvgs.csv`, `data/eulerFlux.csv`, `data/eulerGroups.csv` —
  regenerated twice (boundary coverage, then WENO7-referenced targets);
  git-tracked, original recoverable via git history.
- No Euler-specific model file was kept as canonical output — the trainer
  only saves a model when it passes every acceptance condition, which none
  of the three attempts did.
