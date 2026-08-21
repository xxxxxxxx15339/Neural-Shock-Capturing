# Results

Running log of numerical results and plots, per test. Starts with the WENO5 vs. WENO5-NN baseline (model: `SeemsGood3.h5`); later phases (new dataset, new trained model) get appended below as their own dated sections rather than replacing this one, so progress stays comparable.

```
 PHASE 1              PHASE 2                PHASE 3                 PHASE 4                PHASE 4b
 Baseline             Exact-target data      7-point stencil         HPC seed search        Fine-tune the
 ────────             ──────────────         ───────────────         ──────────────         near-passer
 Run existing         Replace "imitate       5-pt stencil is         Widen search to         ────────────
 WENO5-NN vs          WENO5" targets         info-insufficient       many parallel           Random fine-tune:
 WENO5 on all 4       with TRUE physics:     for shock/entropy       CPU workers.            drifts Sod worse.
 benchmarks.          exact Riemann solver   region (proven via      One candidate:          Anchor weights to
                       + fine-grid Shu-       near-dup stencils,      26/28 criteria,         checkpoint (L2-SP):
 Wins: step adv.       Osher reference.       diff. targets).         +2.4% on core           30/30 pass core
 Ties: Sod, Burgers    Found + fixed data     Switch to 7-pt          metric (shu_regi-       metric. 2 narrow
 Loses: Shu-Osher      outlier bug.           (WENO7-structured).     onal_l1) -- best        gaps remain (Sod
 (basis for the        Diagnosis: still       Found + fixed 2         result of the           TV, Shu Linf) --
 whole effort below)   fails -- 5-pt input    real bugs (outlier      whole project so        closest ever, not
                        can't see enough       data, WENO7 eps         far.                    yet accepted.
                        context.                singularity).
   |                     |                       |                        |                      |
   v                     v                       v                        v                      v
 Results.md            Results.md              Results.md              Results.md             Results.md
 "Phase 1"             "Phase 2"               "Phase 3"               "Phase 4"              "Phase 4b"
```

---

## Phase 1 — WENO5 vs. WENO5-NN baseline

Model: `SeemsGood3.h5` (upstream reference). Each script run at its own paper-matching default settings. Plot paths are relative to the repo root.

### 1. Periodic step advection

Script: `WENO5_vs_WENO5NN_StepAdvection.py` — L=2, 100 cells, CFL=2/3, 50 periods (T=100).

**Numbers:**

| Metric | WENO5 | WENO5-NN |
|---|---|---|
| Discontinuity width, t=0→100 | 0.00 → 0.35 (grows) | flat ≈ 0.19 |
| L2 error vs. exact | 0.10 → 0.20 (grows) | bounded, oscillates 0.095–0.13 |
| Total variation, t=0→100 | 2.000 → 2.042 (drifts) | pinned ≈ 2.000 |

**Plots:**

- [`plots/step_advection/final_snapshot.png`](plots/step_advection/final_snapshot.png) — WENO5 vs. NN vs. exact solution at t=100. *Interpretation: the NN tracks the true step sharply; WENO5 is visibly smeared and hasn't fully settled to 0/1 in the flat regions after 50 periods.*
- [`plots/step_advection/discontinuity_width_vs_time.png`](plots/step_advection/discontinuity_width_vs_time.png) — discontinuity width over the full run. *Interpretation: WENO5's shock keeps widening every period (cumulative numerical diffusion); the NN's stays bounded — the clearest single evidence of the NN's claimed advantage.*
- [`plots/step_advection/l2_error_weno5.png`](plots/step_advection/l2_error_weno5.png) / [`l2_error_nn.png`](plots/step_advection/l2_error_nn.png) — L2 error vs. exact over time. *Interpretation: WENO5's error roughly doubles over the run; the NN's oscillates but never trends upward — no long-time error accumulation.*
- [`plots/step_advection/total_variation_weno5.png`](plots/step_advection/total_variation_weno5.png) / [`total_variation_nn.png`](plots/step_advection/total_variation_nn.png) — total variation over time. *Interpretation: NN's TV is pinned essentially exactly at the analytically correct value (2.000) the entire run; WENO5 drifts up ~2%. Sharpest quantitative confirmation of the qualitative pattern.*
- [`plots/step_advection/disc_track_weno5_x-1.png`](plots/step_advection/disc_track_weno5_x-1.png) / [`disc_track_nn_x-1.png`](plots/step_advection/disc_track_nn_x-1.png) — pointwise |error| near the x=-1 discontinuity, x-ct vs. t. *Interpretation: WENO5's error band visibly funnels wider with time; the NN's stays a constant-width column — same conclusion, spatially resolved.*
- [`plots/step_advection/disc_track_weno5_x-2.png`](plots/step_advection/disc_track_weno5_x-2.png) / [`disc_track_nn_x-2.png`](plots/step_advection/disc_track_nn_x-2.png) — same, near x=-2. *Interpretation: identical pattern at the second discontinuity — not a location-specific artifact.*

**Conclusion: clear NN win.** Avoids the cumulative numerical diffusion WENO5 builds up over long-time integration; holds a stable, near-exact shock shape indefinitely instead of progressively smearing.

---

### 2. Sod shock tube

Script: `WENO5_vs_WENO5NN_Sod.py` — 120 cells, t=0.2, CFL=0.35, vs. exact Riemann solution.

**Numbers:**

```
max_abs_difference(WENO5, NN)     = 9.158e-03
max_abs_error(WENO5 vs. exact)    = 4.4147e-01
max_abs_error(NN    vs. exact)    = 4.4264e-01   (+0.27%, NN marginally worse)
```

**Plots:**

- [`plots/sod/density_velocity_pressure.png`](plots/sod/density_velocity_pressure.png) — density/velocity/pressure vs. x, both schemes vs. exact. *Interpretation: visually indistinguishable; both resolve the rarefaction, contact, and shock correctly.*
- [`plots/sod/density_final.png`](plots/sod/density_final.png) — density only, same comparison. *Interpretation: same conclusion, isolated to the variable most sensitive to scheme differences.*
- [`plots/sod/density_contour.png`](plots/sod/density_contour.png) — space-time density, WENO5 vs. NN side by side. *Interpretation: rarefaction fan and shock trajectory match almost exactly between the two.*
- [`plots/sod/discontinuity_width.png`](plots/sod/discontinuity_width.png) — shock width over time. *Interpretation: NN's width is slightly narrower than WENO5's throughout — small, consistent edge to the NN.*
- [`plots/sod/total_variation.png`](plots/sod/total_variation.png) — total variation over time. *Interpretation: NN's TV stays measurably lower and flatter (≈0.8756 vs. WENO5's ≈0.8763 by t=0.2) — the same "less oscillatory over time" pattern seen on step advection, just smaller in magnitude.*
- [`plots/sod/l2_error.png`](plots/sod/l2_error.png) — direct WENO5 vs. NN L2 distance (not vs. exact) over time. *Interpretation: grows smoothly to ≈9.5e-4 by t=0.2 — small, steadily growing divergence between the two schemes' own trajectories, consistent with the final max_abs_difference.*

**Conclusion: near-tie, with a small consistent edge to the NN** on shock width and total variation — not the pure wash a first glance at the overlapping profile plots suggests.

---

### 3. Burgers' equation convergence

Script: `WENO5_vs_WENO5NN_Burgers.py` — Gaussian IC (k=20), L=2, T=4, reference: WENO5-JS at 1600 cells.

**Numbers:**

| Cells | Δx | E (WENO5-JS) | E (WENO5-NN) | Δ |
|---|---|---|---|---|
| 50 | 4.0e-2 | 2.709e-02 | 2.560e-02 | −5.5% |
| 100 | 2.0e-2 | 1.682e-02 | 1.607e-02 | −4.5% |
| 200 | 1.0e-2 | 1.185e-02 | 1.216e-02 | +2.6% (worse) |
| 400 | 5.0e-3 | 7.413e-03 | 7.163e-03 | −3.4% |

**Plots:**

- [`plots/burgers/solution_profile.png`](plots/burgers/solution_profile.png) — WENO5-JS vs. NN vs. 1600-cell reference at t=4, 50 cells. *Interpretation: the two schemes overlap almost perfectly with each other, both slightly smearing the shock relative to the fine reference — no visible qualitative difference.*
- [`plots/burgers/error_vs_grid_spacing.png`](plots/burgers/error_vs_grid_spacing.png) — L2 error vs. Δx, log-log, both schemes. *Interpretation: no consistent winner — NN better at 3 of 4 resolutions, worse at one, no monotonic trend. Matches the paper's own observation that the two schemes perform similarly on true shocks.*

**Conclusion: essentially a tie.**

---

### 4. Euler Shu–Osher

Script: `WENO5_vs_WENO5NN_ShuOsher.py` — paper settings: 300 cells, T=2, reference: WENO5-JS at 3200 cells.

**Numbers:**

```
max_abs_difference(WENO5, NN)         = 5.311e-01
max_abs_error(WENO5 vs. reference)    = 12.278
max_abs_error(NN    vs. reference)    = 12.345   (+0.55%, NN worse)
```

**Plots:**

- [`plots/shu_osher/density_velocity_pressure.png`](plots/shu_osher/density_velocity_pressure.png) — density/velocity/pressure vs. x, both schemes vs. reference. *Interpretation: both track the reference closely, with expected under-resolution vs. the 3200-cell reference in the post-shock turbulent region (x≈6–8); no visible qualitative difference between WENO5 and NN.*
- [`plots/shu_osher/density_final.png`](plots/shu_osher/density_final.png) — density only, same comparison. *Interpretation: same conclusion.*
- [`plots/shu_osher/density_contour.png`](plots/shu_osher/density_contour.png) — space-time density, WENO5 vs. NN side by side. *Interpretation: shock trajectory and oscillation pattern look essentially identical between the two.*
- [`plots/shu_osher/discontinuity_width.png`](plots/shu_osher/discontinuity_width.png) — shock width over time. *Interpretation: noisy but essentially overlapping — no meaningful difference here, unlike Sod.*
- [`plots/shu_osher/total_variation.png`](plots/shu_osher/total_variation.png) — total variation over time. *Interpretation: the one clear divergence — NN's TV is consistently **higher** than WENO5's and the gap **widens** over the run (≈16.5 vs. ≈14.2 by t=2). Opposite of the pattern seen everywhere else — here the NN is more oscillatory, not less.*

**Conclusion: NN slightly worse**, driven specifically by growing total variation in the post-shock oscillatory region — the one problem where the NN's usual stability advantage reverses.

---

## Cross-problem summary

| Problem | Outcome | Driving evidence |
|---|---|---|
| Step advection | NN clearly better | TV pinned at exact value vs. WENO5 drifting +2%; error bounded vs. WENO5 doubling |
| Sod | NN slightly better | Narrower shock width, lower/flatter TV throughout |
| Burgers | Tie | No consistent winner across 4 resolutions |
| Shu–Osher | NN slightly worse | TV consistently higher, gap widening over time |

**Pattern**: the NN's advantage tracks flow complexity, not just problem duration. It wins clearly on long, simple periodic flow, is a slight net positive on the single-shock Sod case, ties on Burgers, and turns mildly unfavorable on Shu–Osher — the one problem combining a shock with a genuine entropy-wave interaction. This motivates Phase 2: training on data that better represents that kind of complexity instead of the original generic-waveform/imitation-of-WENO5 objective.

---

## Phase 2 — exact-target dataset + retraining (negative result, root cause identified)

**Goal**: remove the "imitation ceiling" diagnosed in `Report_Results.md` Part 4 — the original/WENO7-retargeted training data both cap the network at a classical scheme's own quality, because the target is that scheme's own reconstruction rather than a true physical answer.

**What was built** (`GenerateTrainingData_EulerExact.py`, `src/analysis/exact_riemann_solution`):
- A general exact 1D Euler Riemann solver (verified to exactly reproduce the existing Sod-specific solver, `max abs diff = 0.0`, and to correctly handle all four shock/rarefaction wave-pattern combinations, not just Sod's left-rarefaction/right-shock case).
- `riemann` category: ~1500-3000 random two-state Riemann problems, targets = the **exact** interface value (not a scheme's reconstruction), input = cell-averaged exact solution — same characteristic split-flux representation `NNEuler5` sees at runtime.
- `shock_oscillation` category: Shu-Osher-like trajectories run at both training resolution (64 cells) and a much finer WENO7 reference (800 cells), interpolated in space and time — genuinely higher-fidelity information a local formula cannot see, unlike the old approach's "WENO7 on a 7-point local stencil" (which agrees with WENO5 on >99% of stencils there, per Part 4).
- Outlier guard: an early run had input stencil values up to 562 (vs. ~23 in the original dataset) from rare extreme-contrast Riemann configurations, which stalled training via MSE outlier domination — fixed by tightening the sampling ranges and bounding sample magnitude to 30.

**Result**: the new data has far more genuine signal than the WENO7-retargeting attempt (9.9% of samples diverge from WENO5-Z5's own reconstruction by >0.01, vs. 0.64% before; max divergence ≈17 vs. ≈0.4 before). Despite this — and despite the existing 50x divergence-reweighting mechanism putting >95% of the effective training loss mass on the informative tail — **the trained network still failed to learn the corrections**: across 5 training attempts, Shu regional L1 error came out statistically identical to WENO5's own baseline every time (all failed `shu_regional_l1`; best candidate: NN 0.30598 vs. WENO5 0.30583, i.e. *worse*, not the required 2% better). No model was accepted; nothing was saved to a real `.h5` (only `.best.h5` rejected-candidate checkpoints, which are gitignored/disposable).

**Root cause, confirmed directly** (not architecture/optimizer tuning — a real non-identifiability in the input representation):
- Per-difficulty-tier evaluation of the trained model against the held-out data: on the "extreme" divergence tier (|target − WENO5| > 0.2), the model's mean prediction error (0.894) was *higher* than WENO5's own error (0.884), and the model only beat WENO5 on 37.8% of those samples — essentially coin-flip, not learning.
- Nearest-neighbor check on the `shock_oscillation` hard tail: found 5-point input stencils separated by as little as 0.015-0.028 in L2 distance (essentially identical inputs) whose true targets differ by 2.3–4.5 (e.g. one pair: near-identical input, true values 1.72 vs. 4.81). **The same local 5-point stencil maps to genuinely different correct answers depending on upstream wave history the network cannot see.** No amount of data quality or training effort can fix this — the target isn't a function of the input as currently represented.

**Conclusion**: the imitation-ceiling fix worked as designed (the new data is measurably more informative), but it exposed a deeper, separate problem: the 5-point local characteristic stencil is an insufficient input representation for the Shu-Osher-like oscillatory region. This reframes what Phase 3 needs — not better targets (already have those), but a **wider input window or additional context features** so the input actually determines the target. Widening the stencil is the more direct fix given the existing pipeline (mirrors how the old WENO7 target used a 7-point window, just applied to the *input* instead of a slightly-wider teacher formula).

Artifacts: `data/eulerAvgsExact.csv`/`eulerFluxExact.csv`/`eulerGroupsExact.csv` (100k samples, 5 categories). No accepted model produced this phase.

---

## Phase 3 — 7-point stencil (WENO7-structured NN), two real bugs fixed, still negative

**Goal**: act on Phase 2's finding — the 5-point local stencil is information-insufficient for the shock-oscillation region (near-duplicate inputs, wildly different true targets). Widen the input window so the target actually becomes a function of the input.

**Validation before building anything**: for the specific near-duplicate 5-point pairs found in Phase 2, checked whether a wider stencil actually separates them. At 7 points (radius=3), median separation for those pairs is ~230x the typical 7-point neighbor distance (0.061 vs. 0.00027), with only 1.1% remaining ambiguous — confirms widening is the right fix before spending time implementing it.

**What was built**:
- `GenerateTrainingData_EulerExact.py --radius 3`: generalized the generator's stencil construction (previously hardcoded to radius=2) to produce 7-point exact-target data (`data/eulerAvgsExact7.csv` etc.), reusing the same exact-Riemann and fine-reference-trajectory machinery from Phase 2.
- `Train_WENO7NN_Euler.py`: new training script targeting the existing (previously never successfully trained — see README's WENO3-NN/WENO7-NN status note) `WENO71stOrder`/`NNEuler7` architecture, adapted from `Train_WENO5NN_Euler.py`.

**Two real bugs found and fixed along the way** (both in pre-existing code, not introduced this phase):
1. **Outlier-driven training instability**: initial random-Riemann sampling ranges (density 0.2-6.0, pressure 0.1-15.0) occasionally produced extreme-contrast configurations with characteristic-flux magnitudes up to 562 (vs. ~23 typical). This stalled training at an identical loss regardless of seed — classic MSE outlier domination. Fixed by tightening sampling ranges to the Sod/Shu-Osher regime and adding a hard magnitude bound (30) on generated samples.
2. **`_weno7_coefficients` numerical singularity** (`src/networks/WenoNetworks.py`): WENO7's smoothness-indicator formula uses coefficients 8-40x larger than WENO5's (e.g. 2107/240 vs. 13/12), but reused WENO5's `eps=1e-6` regularizer. For near-degenerate stencils (locally-constant sub-window + a jump elsewhere — common near real shocks), `K.square(eps+beta)` underflowed in float32, producing `alpha/total = 0/0 = NaN`. This didn't show up in a small random forward-pass smoke test but hit ~0.01% of real training rows, and caused *every* trained candidate's runtime Euler simulation to blow up within 1-3 steps (100% failure rate across 5 seeds) even though training itself looked fine. Root-caused by directly probing the Lambda layer's output on real data, confirmed fixed (0 NaN across the full 100k dataset + 200k stress-test near-degenerate patterns) by raising `eps` to `1e-4`. This is very plausibly why no WENO7-NN model had ever been successfully trained in this project before — the architecture existed but was never actually runnable end-to-end.

**Result after both fixes**: training now converges stably and every candidate completes the full physics evaluation (no more crashes). But the trained model is **measurably worse than WENO5**, not just tied:

| Metric | Best candidate (attempt 4) | WENO5 baseline |
|---|---|---|
| Shu regional L1 | 0.3108 (**+1.6% worse**) | 0.3058 |
| Shu maximum TV | 9.35-9.59 (all 5 attempts) | 8.74 |
| Shu shock width | 6 | 7 |
| shu_pressure_overshoot | FAIL (4/5 attempts) | — |
| sod_total_variation | FAIL (2/5 attempts) | — |

Every one of 5 seeds failed `shu_regional_l1` in the *wrong* direction (worse, not just insufficiently better), and 4/5 failed `shu_pressure_overshoot` — a failure mode that essentially never appeared in the 5-point runs.

**Interpretation**: fixing the input-representation problem (Phase 2's diagnosis) and both architecture bugs was necessary but not sufficient. This is the first WENO7-NN training run in the project's history to actually complete without crashing, so there's likely real untuned headroom left (the reduced learning rate of 1e-4, needed for stability, may be under-converging within the same epoch budget; hidden layer size and patience were carried over unchanged from the 5-point script rather than tuned for the wider architecture). Not yet known whether this is a tuning gap or a more fundamental mismatch between the WENO7 classical formula's own behavior and what a correction network can fix. Next step, if pursued: hyperparameter sweep (learning rate schedule, patience, hidden size) on the now-stable 7-point pipeline before concluding the wider stencil doesn't help.

Artifacts: `data/eulerAvgsExact7.csv`/`eulerFluxExact7.csv`/`eulerGroupsExact7.csv` (100k samples, 7-point). Fixes landed in `src/networks/WenoNetworks.py` (eps) and `GenerateTrainingData_EulerExact.py` (radius parameter, sampling ranges) apply regardless of this phase's outcome. No accepted model produced.

---

## Phase 4 — hyperparameter search finds a near-passing candidate (`trained_euler_weno7nn.h5.best.h5`)

**Goal**: push Phase 3's stable-but-underperforming 7-point pipeline to an accepted model.

**Search conducted**: `--hidden-size` exposed as a CLI parameter (the correction MLP's hidden layers were hardcoded to width 4, matching WENO7's 4 sub-stencils — likely a capacity bottleneck, not a principled choice). Confirmed `eps` fix from Phase 3 also makes the original `lr=0.001` safe again (no more NaN). Manual search across ~140 training attempts total, several regularization values:

| Regularization | Hidden size | Result |
|---|---|---|
| 0.10 (original) | 32 | Best region — one seed passed `shu_regional_l1`, `sod_total_variation` fails most seeds |
| 0.16 | 32 | No seed beat WENO5's regional L1 at all |
| 0.25 | 32 | Sod TV well controlled, but regional L1 never beats WENO5 (over-regularized) |

Clear tradeoff: larger hidden size gives the network enough capacity to improve the Shu-Osher region, but the same capacity makes it more aggressive on Sod, overshooting total variation there. Regularization trades between the two rather than solving both.

**Result**: continuing the search at the best config (`hidden-size=32`, `regularization=0.10`, `learning-rate=0.001`, `patience=15`) — including a batch run by the user on external hardware — produced `trained_euler_weno7nn.h5.best.h5`, the best candidate found across the entire session. Directly evaluated against the full acceptance harness (not retrained):

```
Accepted: False
Failed conditions: sod_total_variation, shu_regional_linf
```

**26 of 28 acceptance criteria pass**, including the core requirement:

```
shu_regional_l1   NN = 2.984718e-01   WENO5 = 3.058312e-01   (2.4% better -- required 2%)
```

This is the first and only candidate across the whole session (local search + external run) to actually clear the primary "beat WENO5 by 2% in the Shu-Osher oscillatory region" bar. Full metrics:

```
Sod   density/momentum/energy L1: 6.812e-03 / 7.362e-03 / 1.483e-02   (all PASS)
Shu   density/momentum/energy L1: 7.812e-02 / 1.842e-01 / 4.160e-01   (all PASS)
Shu   regional Linf   NN = 5.705e-01   WENO5 = 5.617e-01             (FAIL -- worse worst-case point)
Sod   final TV        NN = 8.786e-01  WENO5 = 8.763e-01  ref = 8.750e-01   (FAIL -- NN too far from ref)
Sod   shock width      NN = 10        WENO5 = 9
Shu   shock width      NN = 7         WENO5 = 7
```

**Plots** (`WENO5_vs_WENO7NN_Sod.py`, `WENO5_vs_WENO7NN_ShuOsher.py`, both new — 7-point counterparts of the Phase 1 comparison scripts):

- [`plots/sod_weno7nn/density_velocity_pressure.png`](plots/sod_weno7nn/density_velocity_pressure.png) / [`density_final.png`](plots/sod_weno7nn/density_final.png) — WENO7-NN vs. WENO5 vs. exact. *Interpretation: visually near-identical to WENO5, both close to exact — the failure is in a metric (TV), not a visible profile difference.*
- [`plots/sod_weno7nn/total_variation.png`](plots/sod_weno7nn/total_variation.png) — TV over time. *Interpretation: this is the actual failure, visualized — WENO5's TV stays smooth (~0.875-0.876) the whole run; WENO7-NN's spikes up to ~0.881 around t=0.13-0.15 before partially relaxing to ~0.8785-0.879. A transient instability, not a steady-state one.*
- [`plots/sod_weno7nn/discontinuity_width.png`](plots/sod_weno7nn/discontinuity_width.png) — shock width over time. *Interpretation: NN's shock is consistently ~5-10% wider than WENO5's throughout — matches the final shock-width count (10 vs. 9 cells).*
- [`plots/sod_weno7nn/density_contour.png`](plots/sod_weno7nn/density_contour.png) — space-time density, side by side. *Interpretation: same rarefaction/shock structure; no qualitative difference visible at this scale.*
- [`plots/sod_weno7nn/l2_error.png`](plots/sod_weno7nn/l2_error.png) — direct WENO7-NN vs. WENO5 L2 distance over time. *Interpretation: grows to ≈4e-3 by t=0.2 — about 4x larger than the Phase 1 5-point model's divergence from WENO5 on the same problem, consistent with this being a more different, more aggressively-corrected network.*
- [`plots/shu_osher_weno7nn/density_velocity_pressure.png`](plots/shu_osher_weno7nn/density_velocity_pressure.png) / [`density_final.png`](plots/shu_osher_weno7nn/density_final.png) — WENO7-NN vs. WENO5 vs. reference. *Interpretation: WENO7-NN tracks WENO5 closely everywhere, including the oscillatory region — the 2.4% regional improvement is real but subtle at this visual scale, not a dramatic visible difference.*
- [`plots/shu_osher_weno7nn/total_variation.png`](plots/shu_osher_weno7nn/total_variation.png) — full-domain TV over time. *Interpretation: counterintuitive but consistent finding — WENO7-NN's TV is higher than WENO5's across the **entire domain** and the whole run, even though its error is lower specifically in the narrow diagnostic region [5.5, 7.3]. TV (internal smoothness) and regional accuracy (distance to the true solution) are different things and don't have to move together — the model traded some global smoothness for better local accuracy in the region that matters most for acceptance.*
- [`plots/shu_osher_weno7nn/discontinuity_width.png`](plots/shu_osher_weno7nn/discontinuity_width.png) — discontinuity width over time. *Interpretation: opposite of the TV finding — NN's width is consistently narrower (sharper) than WENO5's throughout. Sharper transitions concentrated in fewer cells can still sum to higher total variation than a gentler, wider one — these two metrics capture different failure modes.*
- [`plots/shu_osher_weno7nn/density_contour.png`](plots/shu_osher_weno7nn/density_contour.png) — space-time density, side by side. *Interpretation: same overall wave structure; the oscillatory band looks slightly higher-contrast for WENO7-NN, consistent with the sharper-width finding.*

**Conclusion**: extremely close to a full pass — one seed away, in principle. The two remaining failures are both narrow misses (TV overshoot is transient, not permanent; Linf failure is a single worst-case point, not a broad-region problem) rather than the wide, systematic failures seen throughout Phase 3. Continuing the seed search at this exact configuration (`hidden-size=32`, `regularization=0.10`, `learning-rate=0.001`, `patience=15`) is the most direct path to a fully-accepted model.

Artifacts: `trained_euler_weno7nn.h5.best.h5` (gitignored, local only — this is a best-*rejected*-candidate checkpoint, not an accepted model). `WENO5_vs_WENO7NN_Sod.py`, `WENO5_vs_WENO7NN_ShuOsher.py` (new comparison scripts).

---

## Phase 4b — fine-tuning the near-passing checkpoint (still not accepted, but a real technique validated)

**Goal**: instead of continuing pure random-seed search, fine-tune `trained_euler_weno7nn.h5.best.h5` (the near-passing checkpoint) directly to close the remaining two gaps (`sod_total_variation`, `shu_regional_linf`).

**Attempt 1 — plain fine-tuning from checkpoint.** Added `--warm-start-path` to `Train_WENO7NN_Euler.py` (loads an existing model's weights instead of a fresh random init; each attempt still gets its own `attempt_seed` for minibatch shuffling, so multiple attempts explore different trajectories from the same start). Verified the load mechanism first: evaluating the freshly-loaded, not-yet-retrained model reproduced the checkpoint's exact known metrics (`shu_regional_l1 = 2.984718e-01`, bit-for-bit), confirming `load_weights` worked correctly.

Result of 20 fine-tuning attempts (`lr=0.0001`, same reweighting as before): **worse**, not better — `sod_total_variation` failed in **100%** of attempts (up from ~50% in fresh-random-init runs), with TV values uniformly higher (0.884–0.897) than the checkpoint's own 0.8786. Continued training, even gently, systematically drifted Sod behavior away from the checkpoint's lucky starting point.

**Attempt 2 — protect riemann-category samples.** Hypothesis: the divergence-reweighting loss (50x boost on Shu-Osher-like divergent samples) was drowning out the riemann-category samples (single clean shock/rarefaction/contact — the closest analogue to Sod's local behavior, 48.8% of the dataset). Modified `split_dataset`'s helper functions to also return per-row training group IDs, then added `--protect-riemann-boost` (minimum sample weight for riemann-category rows, independent of the divergence boost). Result: **no improvement** — 20 attempts at `--protect-riemann-boost 20.0`, still 100% `sod_total_variation` failures, same TV range as unprotected fine-tuning (0.883–0.906).

**Diagnosis**: `sod_total_variation` is an emergent property of running the network through a full multi-step Euler simulation — it is never part of the differentiable per-stencil MSE training loss. No reweighting of *static training samples* (divergence-based or category-based) can give the optimizer a gradient signal toward it, because that signal doesn't exist in the loss at all. The checkpoint's original pass on this criterion was luck from its specific training trajectory, not something either version of the loss was ever steering toward — so continued gradient descent, in any direction shaped by that loss, drifts away from it.

**Attempt 3 — L2-to-origin anchor (the fix that actually worked).** Rather than reweighting data, directly penalize the weights for moving away from the checkpoint's own values (elastic-weight-consolidation / L2-SP style): `model.add_loss(anchor_weight * sum((w - w_origin)**2))` added after `load_weights`, before `compile`. New `--anchor-weight` argument (0.0 = disabled).

Calibration sweep (8 epochs each) showed Sod TV improving **monotonically** as anchor strength increased:

| anchor_weight | Sod final TV | Shu regional L1 |
|---|---|---|
| 0.0 | 0.9112 | 0.3110 |
| 0.01 | 0.9017 | 0.3052 |
| 0.1 | 0.8916 | 0.3034 |
| 1.0 | 0.8850 | 0.3039 |
| 5.0 | 0.8803 | 0.3016 |
| 10.0 (20 epochs) | 0.8825 | 0.2999 |
| 20.0 (20 epochs) | 0.8812 | **0.2995 (PASS)** |
| 50.0 (20 epochs) | 0.8798 | 0.2992 |

At `anchor_weight=20.0`, ran a real batch of 30 attempts (`patience=15`, same warm start). Result: **all 30/30 attempts passed `shu_regional_l1`** (up from ~30% in fresh-random-init search, and up from 0% in the two prior fine-tuning attempts) — the anchor reliably locks in the Shu-Osher improvement. Every attempt still failed on exactly the same two remaining criteria, but by a much narrower margin than ever before:

```
Best Sod TV this batch    : 0.87832   (need <= 0.87726 -- gap: 0.00106)
Best Shu regional Linf    : 0.56761   (need <= 0.56172 -- gap: 0.00589)
```

Notably, the *best* Sod TV found (0.87832) is actually *better* than the checkpoint's own value (0.8786) — confirming the anchor doesn't simply freeze the model back to its origin, it constrains drift in magnitude while still allowing some seeds to move in a genuinely improving direction. A follow-up 50-attempt batch at the identical configuration was interrupted after 8 attempts (best partial result: TV=0.87879); no full pass was found before stopping.

**Attempt 3 continued — full 50-seed batch at the same configuration.** Resumed the seed search as planned: 50 fresh seeds (1000-1049), identical configuration (`anchor_weight=20.0`, `patience=15`, same warm start). Result: still no full pass. Best candidate (seed 1008) again passed `shu_regional_l1`, and came closer than ever on `shu_regional_linf`, but `sod_total_variation` regressed relative to the earlier `anchored` batch's best:

```
Best Shu regional Linf this batch : 0.56297   (need <= 0.56172 -- gap: 0.00126, narrowest yet)
Sod TV on that same candidate     : 0.88269   (need <= 0.87726 -- gap: 0.00543, wider than anchored.h5's 0.00106)
```

The two remaining gaps are not moving together — the seed that gets closest on one tends to be further from the other, consistent with them being genuinely different (and possibly slightly conflicting) constraints rather than a single underlying weakness that more seeds alone will fix. Across all anchor-weight=20.0 attempts so far (30 + 8 + 50 = 88 total), none has passed both `sod_total_variation` and `shu_regional_linf` simultaneously.

**Attempt 4 — anchor_weight re-sweep (controlled, single fixed seed).** To find out whether a different anchor strength could close both remaining gaps at once, re-ran the calibration sweep from the original checkpoint at a wider range of `anchor_weight` (5, 10, 15, 20, 25, 30, 40, 50, 75, 100), fixed seed, 15 epochs each — isolating the effect of anchor strength from seed variance:

| anchor_weight | Sod TV (need ≤0.87726) | Shu regional L1 (need ≥2% better) | Shu regional Linf (need ≤0.56172) |
|---|---|---|---|
| 5 | 0.8845 FAIL | worse than WENO5 | 0.5717 FAIL |
| 10 | 0.8832 FAIL | 0.06% better | 0.5673 FAIL |
| 15 | 0.8826 FAIL | 0.28% better | 0.5671 FAIL |
| 20 | 0.8832 FAIL | 0.29% better | 0.5673 FAIL |
| 25 | 0.8807 FAIL | 0.45% better | 0.5674 FAIL |
| 30 | 0.8809 FAIL | 0.69% better | 0.5675 FAIL |
| 40 | 0.8815 FAIL | 1.14% better | 0.5677 FAIL |
| 50 | 0.8809 FAIL | 1.80% better | 0.5680 FAIL |
| 75 | 0.8799 FAIL | 2.09% better **PASS** | 0.5687 FAIL |
| 100 | 0.8793 FAIL | 2.09% better **PASS** | 0.5691 FAIL |

**Structural finding**: `shu_regional_l1` and `sod_total_variation` both improve monotonically as `anchor_weight` increases (pulled toward the origin checkpoint's good values on those two axes), but `shu_regional_linf` gets *monotonically worse* over the same range (best at low anchor_weight ≈15-20, degrading steadily to 0.5691 by anchor_weight=100). There is no single anchor_weight where all three requirements are simultaneously satisfied — `shu_regional_linf` is structurally in tension with the other two along this axis, not just a matter of unlucky seeds. This explains, rather than just confirms, why 88 seeds at `anchor_weight=20.0` never produced a simultaneous pass: no seed at that (or any) anchor strength can satisfy all three at once, because the anchor mechanism itself pulls two of them toward passing while pulling the third away.

**Status**: not yet accepted, and this is where the anchor-weight approach's ceiling was found. `anchor_weight=20.0` remains the most reliable configuration for the primary `shu_regional_l1` metric, but no anchor strength closes all three remaining/marginal criteria together. Closing `shu_regional_linf` (a single worst-case point error) would need a different mechanism entirely — e.g. a loss term that specifically targets the worst-case region — not more seed search or anchor tuning along this same axis.

New CLI arguments added to `Train_WENO7NN_Euler.py`: `--warm-start-path`, `--protect-riemann-boost`, `--anchor-weight`. All default to off/1.0/0.0, so existing from-scratch training behavior is unchanged.

Artifacts (all gitignored, local only, all still best-*rejected*-candidate checkpoints):
- `trained_euler_weno7nn_finetuned.h5.best.h5` — plain fine-tuning, worse than checkpoint.
- `trained_euler_weno7nn_protected.h5.best.h5` — riemann-protected fine-tuning, no improvement over plain.
- `trained_euler_weno7nn_anchored.h5.best.h5` — anchor_weight=20 fine-tuning, closest result on Sod TV (30/30 pass shu_regional_l1).
- `trained_euler_weno7nn_anchored2.h5.best.h5` — follow-up batch, stopped early by request after 8/50 attempts.
- `trained_euler_weno7nn_anchored3.h5.best.h5` — full 50-seed resumed batch, closest result on shu_regional_linf.

## Phase 5 — TENO5-NN: pipeline audit (data → train → evaluate → fine-tune)

**Goal**: check the TENO side of the project, which had not been touched since it was first committed on 2026-07-24 (same commit that added the whole pipeline in one shot), while all recent iteration had gone into WENO7-NN. This phase audits and re-verifies the existing NN-TENO5 pipeline end to end rather than searching for new architectures.

**Architecture note**: NN-TENO5 is a fundamentally different kind of network from WENO5-NN/WENO7-NN. It is a 3-output binary *classifier* (`5 → 64 → 32 → 16 → 8 → 3`, sigmoid outputs, per-candidate threshold calibration) trained to imitate TENO5's own discrete candidate-discard decision on each of its 3 stencil candidates, via cross-entropy against labels generated by `teno5_troubled_labels`. WENO5/7-NN instead directly regress flux corrections against numerical targets. This means the WENO7-NN fine-tuning technique built in Phase 4b (L2-SP anchor-weight regularization on network weights) has no direct analogue here — there's no equivalent "drifting away from a lucky checkpoint on an emergent simulation metric" failure mode, because NN-TENO5 is trained to match a well-defined per-sample label, not to pass downstream simulation criteria.

**Pipeline as it exists**:
- **Data**: no dedicated TENO data-generation script. `Train_NNTENO5.py` reuses the same 75,241-sample five-point-stencil CSV (`data/2ndNewAvgs.csv`) from the original WENO5-NN work, and generates its 3 binary labels per stencil *inline* at train time by running the deterministic TENO5 candidate-selection logic (`src/schemes/teno5.py::teno5_troubled_labels`) over each stencil. There is no `--mode`/generate-only flag — data loading, label generation, training, and metadata export all happen in one `main()` call.
- **Train**: `Train_NNTENO5.py`, stratified 70/15/15 split, `ModelCheckpoint`/`EarlyStopping`/`ReduceLROnPlateau`, then post-hoc per-candidate threshold calibration on the validation set. One gotcha found while re-running it: unlike `Train_WENO7NN_Euler.py`, this script's `--device auto` (the default) still asks TensorFlow for GPU index 0 even when `CUDA_VISIBLE_DEVICES=-1` is set, and crashes with `no CUDA devices found` on a GPU-less run — `--device cpu` must be passed explicitly on this machine.
- **Evaluate**: two separate scripts, both loading `models/nn_teno5.h5` + its calibrated thresholds — `TENO5_vs_NNTENO5_Euler.py` (Sod, Shu-Osher, plus two trivial uniform-state sanity cases) and `TENO5_vs_NNTENO5_Scalar.py` (smooth-sine/step/multifrequency advection, Burgers). Both check a "teacher fidelity" criterion (NN density-L1 error must stay within 1.02× of deterministic TENO5's own error).
- **Fine-tune**: **no mechanism exists.** Confirmed by grep — no `--warm-start-path`, `--anchor-weight`, or any weight-reload-before-training path in `Train_NNTENO5.py`. Not built in this phase either: the current model passes every check (see below), so there was nothing to fine-tune toward, and given how different the training paradigm is from WENO7-NN's, an anchor-weight-style mechanism would need to be designed from scratch rather than ported. This is left as an identified gap, not implemented — flag if a future change (e.g. a different TENO teacher hyperparameter, or a genuinely failing checkpoint) makes it actually needed.
- **TENO3-NN / TENO7-NN**: scaffolded (`src/networks/TenoNetworks.py` defines `TENO3Network`/`TENO7Network`, `src/schemes/teno3.py`/`teno7.py` have working deterministic schemes + NN wrapper classes, `tests/test_teno.py` exercises the wrapper mechanics with a fake model) but never trained — no dataset, no training script, no `.h5` weights. Out of scope for this audit; noted for future work.

**Re-verification**: retrained from scratch (`--device cpu`, fresh seed, everything else default) to confirm the pipeline still reproduces today, 3 weeks after it was first built and on a machine with no GPU. Result: reproduced cleanly — 273→300 epoch range, test exact-mask accuracy 98.4% (original: 98.7%), test loss 0.0222 (original: 0.0161), the difference fully explained by seed variance since architecture/data/labels are identical. No code changes were needed to reproduce training; only the `--device cpu` flag.

Re-ran both evaluation scripts against the original committed model (`models/nn_teno5.h5`) to refresh results and regenerate plots. Note: the *deterministic* TENO5 numbers themselves shifted by a few percent versus the July-24 committed JSON (e.g. Shu-Osher density L1 0.07074 → 0.07345) even though the model, scheme code, and CLI config are all unchanged — confirmed reproducible bit-for-bit across two fresh runs today, so this is an environment/dependency drift (e.g. a numpy/scipy point release) since July 24 cascading through the adaptive-CFL timestepping, not a code regression. Both old and new numbers pass comfortably either way.

| Problem | TENO5 density L1 | NN-TENO5 density L1 | Teacher fidelity |
|---|---|---|---|
| Sod | 5.596e-03 | 5.533e-03 | **PASS** (NN actually *lower* error than deterministic TENO5; limit 5.708e-03) |
| Shu-Osher | 7.345e-02 | 7.308e-02 | **PASS** (NN actually *lower* error than deterministic TENO5; limit 7.492e-02) |
| smooth_sine_advection | 3.328e-07 | 3.328e-07 | **PASS**, bit-exact |
| multifrequency_advection | 2.434e-05 | 2.434e-05 | **PASS**, bit-exact |
| burgers_smooth_to_shock | 1.328e-04 | 1.328e-04 | **PASS**, bit-exact |
| step_advection | 1.3317e-02 | 1.3398e-02 | **PASS**, small genuine difference (not bit-exact — the only one where the classifier's decision actually diverges from TENO5's on some stencils) |

All 6 non-trivial benchmarks pass; 3 of 4 scalar cases are numerically identical to deterministic TENO5 (the classifier reproduces TENO5's discrete decisions exactly on those problems), and `step_advection` is the only case where NN-TENO5's decisions genuinely diverge from the teacher, by a small fidelity-preserving margin. This is a materially different status than WENO7-NN: **NN-TENO5 has never failed to pass its acceptance criteria**, across the original run and this re-verification.

**Plot infrastructure added**: neither evaluation script previously saved plots to disk — `TENO5_vs_NNTENO5_Scalar.py` only called `plt.show()` (the 5 PNGs already in the repo from July 24 were saved by hand via the interactive window, not by the script), and `TENO5_vs_NNTENO5_Euler.py` had no plotting at all. Added a `--save-dir`/`--no-show` pair to both (matching the convention already used by `WENO5_vs_WENO7NN_Sod.py` etc.), and a new `plot_reference_case` function to the Euler script (density/velocity/pressure, 3-panel, TENO5 vs. NN-TENO5 vs. reference) since it previously had no plotting code to reuse. Regenerated plots:
- `models/nn_teno5_euler_comparison_plots/sod.png`, `shu_osher.png` (new)
- `models/nn_teno5_scalar_comparison_plots/{smooth_sine_advection,step_advection,multifrequency_advection,burgers_smooth_to_shock,teno5_nn_teno5_overview}.png` (regenerated, now reproducible from the script directly)

**Status**: TENO5-NN pipeline verified healthy end-to-end and reproducible. No further action needed on this model as a passing classifier. Gaps identified for future work, not started: (1) no fine-tuning/warm-start mechanism, (2) no dedicated TENO data-generation script (reuses WENO5-era CSVs), (3) TENO3-NN/TENO7-NN are unimplemented despite scaffolding.

## Phase 6 — reframing the comparison: does NN-TENO5 actually beat TENO5 against the real solution?

**Motivation**: the `teacher_fidelity` check used throughout Phase 5 only requires NN-TENO5 to stay within 1.02× of TENO5's own error — it's a "didn't get meaningfully worse" bar, not a "beats it" bar, and it doesn't even compare against the real/reference solution directly in its pass/fail logic (both errors happen to already be computed against the reference, the check just uses the wrong threshold). Added a stricter `beats_teacher` field to both `TENO5_vs_NNTENO5_Euler.py` and `TENO5_vs_NNTENO5_Scalar.py`: `nn_teno5_error <= teno5_error`, both measured against the real/reference solution, no margin.

**Result** (current committed model, `models/nn_teno5.h5`):

| Benchmark | Real solution used | TENO5 error | NN-TENO5 error | Beats TENO5? |
|---|---|---|---|---|
| Sod | exact Riemann solution | 5.5962e-03 | 5.5326e-03 | **yes** |
| Shu-Osher | WENO7 @ 320 cells (no closed form exists) | 7.3449e-02 | 7.3080e-02 | **yes** |
| smooth_sine_advection | exact analytic solution | 3.3278e-07 | 3.3278e-07 | yes (tie, bit-exact) |
| multifrequency_advection | exact analytic solution | 2.4337e-05 | 2.4337e-05 | yes (tie, bit-exact) |
| burgers_smooth_to_shock | exact characteristic solution | 1.3283e-04 | 1.3283e-04 | yes (tie, bit-exact) |
| step_advection | exact step function | 1.33172e-02 | 1.33975e-02 | **no** (~0.6% worse) |

**Tally: 5/6 benchmarks**, but 3 of those are exact ties (the classifier made identical decisions to TENO5), and the two genuine wins (Sod, Shu-Osher) are ~1% and ~0.5% margins — consistent with noise scattered around parity, not a systematic advantage. This is expected: nothing in NN-TENO5's training loss (cross-entropy against TENO5's own candidate-discard labels) has ever pointed toward beating TENO5's error against the real solution — the network was never asked to.

**Investigated next: can this be turned into a real, trained-for advantage?** Checked feasibility of the two ways to do this properly, using this project's own history as evidence:

1. **End-to-end differentiable fine-tuning against a downstream real-solution loss — not currently possible.** Checked the numerical core: `src/schemes/teno_common.py`, `src/core/SimulationClasses.py`, `src/core/TimeSteppingMethods.py`, `src/core/eulerEquations.py`, and `teno5.py` are all pure NumPy (`np.divide`, `np.sum`, no TF ops), and `NN_TENO5`/`NN_TENO5_Euler` call `model.predict(...)`, which runs a full forward pass and returns a plain NumPy array — completely detached from any TensorFlow graph. There is no gradient path from a simulation-level loss back to the classifier's weights today. Building one would mean rewriting the finite-volume solver (flux computation, SSPRK3 stepping, TENO candidate combination) in TF ops — a multi-day rewrite of the numerical core, not a fine-tuning experiment. This corrects what I proposed earlier in conversation as a "cheap first step" — it isn't cheap; I hadn't checked the solver's implementation before proposing it.

2. **Oracle-relabeled training data (train against the true local answer instead of TENO5's decision) — has known prior-art risk, already documented in this repo.** Phase 2 tried the structurally identical idea for WENO7-NN: replace scheme-imitation targets with exact-solution-derived targets on a local 5-point stencil. It failed for a specific, confirmed reason — on the Shu-Osher-like `shock_oscillation` category, nearly-identical 5-point input stencils (L2 distance 0.015–0.028) had true targets differing by 2.3–4.5, because **the correct answer depends on upstream wave history the local stencil can't see**. That non-identifiability is specific to time-evolving, wave-interacting problems (Riemann/shock-oscillation categories) — it should *not* apply to the static, non-interacting synthetic functions the original generic-waveform dataset methodology uses (single smooth/discontinuous test functions, no wave history), so an oracle-relabeled version of TENO5's *existing* dataset style looks safe from this specific failure mode. But the same project's Phase 3 also shows that good local static targets alone weren't sufficient to reliably close a real gap against a classical scheme — Phase 4/4b needed downstream-simulation-directed search (hyperparameter/seed search, then anchor-weight fine-tuning against actual acceptance criteria) to get anywhere close for WENO7-NN.

**Revised recommendation** (not yet started, needs a decision): given (1) is infeasible without a large rewrite and (2) alone is unlikely to be sufficient based on this project's own track record, the approach most consistent with what has actually worked before in this repo is: (a) build an oracle-relabeled TENO5 dataset from static synthetic functions (safe from the wave-history trap), to give the classifier a training signal connected to real-solution accuracy instead of TENO5-imitation, then (b) fine-tune/seed-search the result against the new `beats_teacher` criterion from this phase as the actual downstream objective — mirroring Phase 4/4b's proven (if only partially successful) methodology, rather than attempting a differentiable rollout that the codebase can't currently support.

## Phase 7 — oracle-relabeled TENO5 dataset (step (a) of the Phase 6 plan)

**Goal**: build the dataset described in Phase 6's revised recommendation, step (a) — a stencil dataset labeled with the keep/discard mask that actually minimizes error against a *known true* interface value, not TENO5's own scale-separation decision.

**Built** (`GenerateTrainingData_TENOExact.py`): samples five random synthetic function families per stencil — single sinusoid, sum of two sinusoids, cubic polynomial, single step discontinuity, and an absolute-value kink (C0 but not C1) — at a random cell width and interface location, computes the five-cell stencil via closed-form exact cell averages (not quadrature, so generation is fully vectorized and exact to floating point), and labels each stencil by exhaustively searching all 8 possible keep/discard masks with `teno5_candidate_values`/`combine_teno_candidates` (the same formulas the real scheme and NN-TENO5 use at inference) to find whichever mask lands closest to the function's true point value at the interface. Sine wavenumber is tied to cell width via a randomized points-per-wavelength parameter (2–20) rather than sampled independently, to avoid generating physically meaningless "smooth" samples that are actually badly aliased (an early version allowed e.g. wavenumber 6 with cell width 0.3 — about 9 wavelengths packed into one 5-point stencil, unreconstructable by any scheme and not a real test of anything).

**Bug found and fixed during validation**: the first working version showed an implausible 99%+ disagreement rate with TENO5 on the `step` and `kink` families — checked by comparing actual reconstruction error (not just label equality) and found the true error-level disagreement was only ~3%, a 30x gap. Root cause: whenever a stencil is locally flat/constant (all three candidates already agree, common for `step`/`kink` samples where the discontinuity falls outside the 5-cell window), all 8 masks tie at zero error, and the argmin search silently kept whichever mask happened to be checked first — which was `(discard, discard, discard)`, since `itertools.product` enumerates that combination first. So the oracle was mislabeling huge numbers of ordinary smooth/flat stencils as "discard everything" purely as an artifact of iteration order, not because that was actually a better answer. Fixed by (1) reordering the mask search to check "keep everything" first, so ties resolve toward keeping candidates (matching both TENO design intent and what TENO5 itself does in genuinely smooth regions), and (2) a small tolerance (1e-10) on the improvement comparison so near-ties don't get treated as genuine differences. This dropped `step`'s disagreement rate from 99.75% to 28.6% and `kink`'s from 98.1% to 18.9%; `sine`/`double_sine`/`polynomial` were largely unaffected (they rarely produce exact ties).

**Result** (75,000 samples, matching the original dataset's size):

| Family | Share | Disagreement vs. TENO5 |
|---|---|---|
| sine | 30% | 24.3% |
| double_sine | 20% | 40.2% |
| polynomial | 20% | 1.4% |
| step | 20% | 28.6% |
| kink | 10% | 18.9% |
| **overall** | | **23.2%** |

Spot-checked that label disagreement corresponds to real, substantial error reduction, not further tie artifacts: for `sine`/`double_sine`/`kink`/`polynomial`, the fraction of samples with a >1e-6 error improvement matches the label-disagreement rate almost exactly (e.g. sine: 24.1% vs. 24.2%). `step` still shows a gap (27.7% label disagreement vs. 3.3% with a genuine >1e-6 improvement) — a residual, benign case where TENO5's own threshold picks a different but equally-zero-error mask in flat regions; not a real accuracy difference, just label noise a classifier should be reasonably robust to. Where there is a genuine improvement, the magnitude is substantial (mean error reduction 0.06 for sine, 0.16 for step, computed only over the genuinely-improved subset) — this is a real, learnable signal, structurally different from Phase 3's WENO7-retargeting failure (which had <1% divergent samples, too thin to learn from).

**Status**: dataset built and validated (`data/teno5ExactStencils.csv`, `data/teno5ExactLabels.csv`, `data/teno5ExactLabels_metadata.json`), ready for training. Training on it (step (b) of the Phase 6 plan — replacing NN-TENO5's current cross-entropy-against-TENO5 objective with cross-entropy against these oracle labels, then evaluating against the `beats_teacher` criterion from Phase 6) has not been started; that is a separate step.

## Phase 8 — training on the oracle labels: negative result, causes simulation instability

**Goal**: step (b) of the Phase 6 plan — train NN-TENO5 on the Phase 7 oracle labels instead of TENO5-imitation, then evaluate against the `beats_teacher` criterion.

**Built**: added `--labels-data` to `Train_NNTENO5.py` (loads a precomputed label CSV instead of calling `teno5_troubled_labels` inline; validated with the same shape/0-1/non-trivial checks as the generated path, and recorded in metadata under `source_data.labels_source`). Trained a fresh model (`models/nn_teno5_oracle.h5`) on the Phase 7 dataset, `--device cpu`, otherwise default hyperparameters.

**Classifier metrics dropped sharply**, as expected given the label source is now inherently harder/noisier than TENO5's own self-consistent decision rule: test exact-mask accuracy 83.1% (vs. 98.7% for the TENO5-imitation model), macro F1 0.64 (vs. 0.99). This alone isn't concerning — the oracle task is genuinely harder, and classifier accuracy against local labels was never the real target.

**The real target — `beats_teacher` — failed badly, not just "no better":**
- **Sod: the simulation crashes.** `FloatingPointError: Non-positive Euler density at step 6` — the run doesn't even complete.
- **step_advection: catastrophic blowup.** NN-TENO5 L1 error = 5.34e9 (TENO5's own: 1.33e-2) — a ~10¹¹× regression, not a modest loss.
- **burgers_smooth_to_shock: tiny genuine regression** (1.32832e-4 vs. 1.32827e-4 — negligible, effectively a tie).
- **smooth_sine_advection, multifrequency_advection: unaffected**, still bit-exact ties (these never exercised the candidate-discard logic much to begin with).

**Diagnosis**: this is the concrete, dramatic version of the risk flagged going into Phase 7 — a classifier trained only for one-shot pointwise accuracy against static synthetic functions has no incentive to preserve the stability properties (bounded output, suppressed Gibbs oscillations near discontinuities) that are the entire *point* of TENO's scale-separation design, and that TENO5's own heuristic — while not locally "optimal" in the oracle sense on 23% of samples — reliably provides. A choice that's locally closer to the true value on an isolated static test function can still be destabilizing once fed back into an evolving multi-step simulation, where small errors compound across time steps and cells. TENO5's own decision rule isn't locally optimal, but it's been implicitly shaped (via its scale-separation formula) to keep every intermediate simulation state well-behaved; the oracle relabeling optimized for a completely different, narrower objective and got exactly what it optimized for — better one-shot local accuracy — at the cost of the property that actually matters for a working scheme.

**Status**: this specific approach — training directly on unmixed oracle labels — is a clear negative result, not a partial success. Not committing `models/nn_teno5_oracle.h5`/`_history.csv`/`_metadata.json`/`_split.npz` to the repo; they reproduce a known-broken model and would be misleading to ship alongside the real accepted models (consistent with how Phase 2's failed run also produced no committed model, only its generating data). The oracle dataset itself (Phase 7) remains valid and committed — the failure is in how it was used for training, not in the labels themselves. A natural next idea, not yet tried: blend the two label sources (e.g. an anchor-style loss that keeps the model close to TENO5-imitation by default and only pulls toward the oracle label where the two agree strongly or where deviating provably doesn't destabilize a real simulation) rather than training on the oracle labels unmixed — mirroring the anchor-weight technique that was the one thing that actually worked for WENO7-NN (Phase 4b). This needs a decision before proceeding further.

## Phase 9 — anchor-weight fine-tuning on the oracle labels: fixes the instability, but finds the same kind of ceiling as WENO7-NN

**Goal**: the proposed fix for Phase 8 — instead of training on the oracle labels unmixed, warm-start from the known-safe `models/nn_teno5.h5` and add an L2-SP/elastic-weight-consolidation anchor loss (`anchor_weight * sum((w - w_origin)^2)`) that penalizes the weights for drifting from that safe starting point, so training toward the oracle labels can improve things without being free to destabilize the simulation.

**Built**: added `--warm-start-path` and `--anchor-weight` to `Train_NNTENO5.py`, mirroring `Train_WENO7NN_Euler.py`'s Phase 4b mechanism exactly (`model.load_weights(...)` then `model.add_loss(...)` then a recompile so the new loss term is picked up by the training graph). Recorded in metadata under `training.warm_start_path`/`training.anchor_weight`.

**Calibration sweep** (`--epochs 60 --patience 12`, warm-started from `models/nn_teno5.h5`, trained on the Phase 7 oracle labels, `anchor_weight` ∈ {1, 5, 20, 100, 500}), evaluated on the full 6-benchmark `beats_teacher` suite:

| anchor_weight | Sod | Shu-Osher | smooth_sine | multifreq | burgers | step_advection | Total |
|---|---|---|---|---|---|---|---|
| 0 (Phase 8, unmixed) | **crash** | n/a | tie | tie | tiny loss | **blowup (5.3e9)** | catastrophic |
| 1 | lose | lose | tie | tie | **win** (1.11e-4 vs. 1.33e-4) | lose | 3/6 |
| 5 | lose | lose | tie | tie | **win** (9.53e-5 vs. 1.33e-4) | lose | 3/6 |
| 20 | lose | lose | tie | tie | tiny loss | lose | 2/6 |
| 100 | **win** | lose | tie | tie | tie | lose | 4/6 |
| 500 | **win** | lose | tie | tie | tie | lose | 4/6 |
| *(baseline: plain TENO5-imitation, no oracle at all)* | win | win | tie | tie | tie | lose | **5/6** |

**The instability is fully fixed** — every anchor_weight from 1 to 500 keeps Sod and `step_advection` finite and bounded; nothing crashes or blows up anymore, at any tested strength. That confirms the Phase 8 diagnosis was correct: the anchor term is exactly what was missing.

**But no tested anchor_weight reaches, let alone beats, the plain baseline's 5/6.** The pattern is the same *shape* of tension found for WENO7-NN in Phase 4b, just with a different pair of criteria in conflict: low anchor_weight (1–5) keeps a genuine, real improvement on `burgers_smooth_to_shock` (a real 20–30% error reduction, not noise) but loses both Euler benchmarks Sod and Shu-Osher, which the unmodified baseline already wins; high anchor_weight (100–500) recovers Sod's win and shrinks the Burgers gain back down to an exact tie, but **never recovers Shu-Osher**, which stays a loss at every strength tested, including 500. Anchor_weight=100 and 500 give the identical result (4/6) — this isn't still converging toward the baseline as anchor strength increases, it has already plateaued at a ceiling below it. Fine-tuning under a very strong anchor doesn't reproduce the origin checkpoint's exact decision boundary; it settles at a nearby-but-different point, and that point happens to permanently lose Shu-Osher.

**Diagnosis**: Shu-Osher's ~0.5% margin in the plain TENO5-imitation baseline (Phase 6) looks like a fragile, trajectory-specific property — something about that exact, untouched set of weights — rather than a robust property any nearby model in weight-space shares. Warm-starting and then training at all, even under heavy anchoring toward oracle labels that are themselves correct and validated (Phase 7), is enough perturbation to lose it, and no anchor strength found a way back. This mirrors Phase 4b's finding almost exactly: `shu_regional_linf` for WENO7-NN and Shu-Osher's win margin for NN-TENO5 both turn out to be single criteria that a broad fine-tuning mechanism cannot reliably preserve while also chasing genuine improvement elsewhere, across the same kind of wide parameter sweep.

**Status**: the anchor-weight mechanism does its one job — it eliminates the Phase 8 instability entirely — but does not produce a model that's better than simply shipping the original, untouched `models/nn_teno5.h5`. Not committing any of the `aw*.h5` sweep checkpoints (scratch artifacts, all inferior to the already-committed baseline on the metric that matters). Given this is now the second mechanism (after WENO7-NN's own anchor-weight sweep in Phase 4b) to hit a ceiling of this same shape — broad fine-tuning reliably recovers most criteria but reliably loses one specific one, regardless of strength — the honest read is that beating TENO5 more than the current ~1%-noise-level margin likely needs a per-criterion-targeted mechanism (e.g. protecting Shu-Osher's specific behavior directly, the way `--protect-riemann-boost` was tried for WENO7-NN) rather than another global sweep along this same axis.

## Phase 10 — the Unrolled Physics Loss proposal: a differentiable JAX rewrite, and the first real test of whether it beats TENO5

**Goal**: the Unrolled Physics Loss proposal (Radouan Boukharfane, implemented/tested by Yasser Baouzil) proposes training NN-TENO5 with a K-step *unrolled* physics loss (differentiate through an RK3 rollout, not just a single reconstruction) plus an expanded feature vector and a 40%/60% synthetic/real-PDE dataset. None of this is buildable on the existing pure-NumPy + standalone-Keras-2.2.4 stack (`model.predict()` fully detaches from any gradient graph), so this phase is a from-scratch differentiable rewrite in JAX, kept in a new `src/differentiable/` package specifically so it never has to coexist with the legacy Keras pin (see that package's `__init__.py`). Full step-by-step plan, design-choice reasoning, and status notes per phase live in `UNROLLED_PHYSICS_LOSS_GUIDE.md` — this entry is the results summary.

**Built, phase by phase** (all with passing regression/unit tests against the real NumPy solver, not just self-consistency):
- **Differentiable solver** (`src/differentiable/teno5_jax.py`, `numerics_jax.py`, `euler_jax.py`): JAX ports of TENO5's candidate/smoothness/selection math, the scalar finite-volume solver, and the full characteristic-wise Euler solver (Roe eigenbasis, projections, Lax-Friedrichs splitting). Both match the existing NumPy solver to float64 precision (`4.4e-16` scalar, `5.5e-11` Euler Sod). `jax.lax.scan` replaced an initial unrolled-Python-loop rollout after that took **11 minutes to jit-compile** on the Euler flux (240 substeps unrolled into one XLA graph) — `scan` compiled the same problem in 4.4s, ~150x faster, identical results.
- **Feature vector** (`src/data/feature_vector.py`): the Unrolled Physics Loss proposal's `X_j` (Eq. 3), validated against the real 75,000-row oracle dataset. Found two real spec gaps in the Unrolled Physics Loss proposal itself: `c~_WENO/TENO` has no formula anywhere in the document (implemented as the scheme's fixed linear-weight triple, the only defensible data-independent reading); computing *both* BVD interface jumps `J_{j-1/2}`/`J_{j+1/2}` for one cell needs a 6-point raw window, one point wider than NN-TENO5's existing 5-point stencil and than what's already stored on disk.
- **Projection layer** (`projection_jax.py`) and **Unrolled Physics Loss** (`losses_jax.py`, all four terms of Eq. 5-8): both pass their exit criteria (bit-exact no-op against the real NumPy inference path at threshold 0.5; all four loss terms finite with finite nonzero gradients). `L_BVD` (Eq. 5) also has no formula anywhere in the Unrolled Physics Loss proposal — implemented as a hinge penalizing the network's own interface jump only when it exceeds deterministic TENO5's, grounded in the feature-vector's own BVD construction but flagged as a guess, not a certainty.

**The central empirical question — does training via the unrolled loss actually beat TENO5? Scalar: yes, narrowly and genuinely. Euler: blocked by a real numerical-stability finding, not attempted-and-lost.**

Scalar advection, three training runs (`Train_NNTENO5_UnrolledJAX.py`), each fixing what the previous run's own per-term loss logging exposed:

| Run | Change | held-out jump, K steps | held-out jump, 3xK steps | held-out smooth sine |
|---|---|---|---|---|
| 1 | baseline (`L_TV`/`L_Reg` as raw sums, Eq. 7/8 literal) | loses (0.142 vs 0.118) | loses (0.210 vs 0.125) | loses ~10x (0.051 vs 0.0048) |
| 2 | normalize `L_TV`/`L_Reg` (mean, not sum) | loses (0.147 vs 0.118) | loses (0.182 vs 0.125) | loses ~9x (0.044 vs 0.0048) |
| 3 | + imitation-pretrain first (0.6s), anchor to that | **wins** (0.115 vs 0.118) | **wins** (0.116 vs 0.125) | near-tie (0.0049 vs 0.0048) |

Run 1→2 found a genuine, generalizable bug in taking the Unrolled Physics Loss proposal's Eq. 7/8 literally: unnormalized sums grow with grid size/parameter count while Eq. 6's MSE (a mean) doesn't, letting `L_TV` dominate the gradient by ~4x even after its 0.1 weight. Fixing that made the logged losses sane but didn't close the gap — the deeper problem was cold-start optimization through a sparse, K-step-delayed loss. Run 3's fix mirrors this project's own Phase 9 finding almost exactly: pretrain by imitating TENO5's own reconstruction first (a dense, well-conditioned, per-stencil objective — converges in 500 steps / 0.6 seconds), then use that checkpoint as both the starting point and the `L_Reg` anchor for the unrolled fine-tune, exactly like Phase 9's warm-start-plus-anchor pattern. That closed the gap: **2 of 3 held-out cases now genuinely beat deterministic TENO5 against the exact solution**, including generalizing to 3x the trained rollout horizon.

**Phase 7's validation matrix then found this result is real but narrow.** Run against canonical, not held-out-similar, versions of the Unrolled Physics Loss proposal's own benchmark problems (`ValidationMatrix_ProjPDF.py`) — a fixed-width step profile at `final_time=1.0` (~8.5x the trained K-step horizon) and Burgers 1D (out-of-distribution nonlinear flux) — the same trained network **loses to TENO5 on both** (Advection L2 0.140 vs. 0.099; Burgers L2 0.0030 vs. 0.0008). The positive result stands, but it's scoped to problems similar in shape and time-horizon to training, with tested generalization out to 3x that horizon — not evidence of a general-purpose win at arbitrary scale.

**Euler (`Train_NNTENO5euler_UnrolledJAX.py`) did not reach a comparable checkpoint, and the reason is itself a finding worth reporting**, not a bug quietly left broken. Reusing the pretrained scalar weights as an Euler warm start (valid — imitation-pretraining is representation-agnostic) immediately produced NaN losses at training step 0. Tracing it down, in order: (1) a missing data-validity filter (the existing `GenerateTrainingData_EulerExact.py` already filters vacuum-forming random Riemann samples; the new sampler didn't — fixed); (2) a positivity guard applied only between full SSPRK3 steps, not to the *intermediate* substages that `flux_fn` is also called on inside one step — fixed in `numerics_jax.ssprk3_step` itself; (3) even fully guarded, one steep (~p=9) training example still diverged — density hit a `1e-6` floor at step 6, state exploded to `5.7e144` by step 7. Raising the floor to a physically sane scale (`eps=0.05`) and capping velocity delayed but didn't prevent the blowup. Testing the standard Lax-Friedrichs robustness fix (over-estimate the dissipation coefficient `alpha`) gave a **non-monotonic result** — a 1.5x margin delayed blowup further, but 2x-5x margins made it blow up *earlier* (step 3, not step 8-11) — which rules out "wave speed bound too small" and points at the Roe eigenbasis itself becoming ill-conditioned as the state evolves under a not-yet-Euler-trained network's reconstruction. This is a materially different, harder failure mode than anything scalar advection exposed, because scalar linear advection's stability never depends on the reconstructed value feeding back into a wave-speed/eigenbasis calculation the way a nonlinear system's does.

**Status**: the proposal's central claim — an unrolled physics loss can produce a network that beats TENO5 against the real solution — has a real, positive, narrowly-scoped demonstration on scalar advection, achieved only after two non-obvious fixes (loss-term normalization; imitation-pretrain-then-anchor, not train-from-scratch) that the proposal itself doesn't mention. It does not yet extend to problems much beyond the trained horizon/shape, and it does not yet extend to Euler at all — not for lack of trying, but because of a genuine, well-diagnosed numerical-stability obstacle specific to systems of conservation laws that would need a real mechanism (a curriculum from mild to steep problems, or an explicit stability-preserving projection on the characteristic decomposition) to resolve, not another parameter tweak. Recommended framing for the professor: the core idea is validated in the narrow case that was tractable to test here; scaling it to the full Table 1 (especially Sod/Shu-Osher/Sedov) is a substantially larger undertaking than the 2-page plan implies, with a specific, now-documented technical obstacle in the way, not just "more engineering time."

---

## Phase 11 — literature-motivated fix attempt: curriculum learning on rollout length for Euler

**Goal**: a deep literature review (not just abstract-scanning — six papers read in full) turned up an almost exact match for Phase 10's Euler blowup. Um et al. ("Solver-in-the-Loop", NeurIPS 2020) report the same failure mode nearly verbatim — *"performing time integration via the PDE then typically leads to exponential increases of existing oscillations and a diverging calculation"* — and give a specific, tested mitigation never tried here: *"we found it important to pre-train networks with small look-aheads, and then continue training with longer recurrent iterations."* Phase 10's Euler attempt jumped straight to K=15. This phase tests the curriculum directly.

**Built**: `Train_NNTENO5euler_UnrolledJAX_Curriculum.py`, reusing Phase 10's Euler script's data sampling, loss, positivity guard, and evaluation unchanged — the only variable isolated is staging K = 2 → 4 → 8 → 15 (60/60/60/80 steps) instead of training at K=15 from step 0, with the Adam optimizer state carried across stages, not reset (a genuine curriculum, not four independent runs).

**Result**: real, measurable improvement in how far training got — but not a fix. The K=2 stage completed all 60 steps without incident (Phase 10's direct K=15 attempt never got past step 0). The K=4 stage then ran 39 of 60 steps before the loss went non-finite, at which point training stopped. Held-out evaluation after the NaN-corrupted checkpoint reproduced Phase 10's exact same degenerate numbers (the projection layer's hard-threshold fallback absorbs NaN parameters into "discard everything," which is finite but not meaningful).

**Diagnosis**: curriculum learning measurably delays the instability (full K=2, partial K=4) exactly as the literature predicts, but does not resolve it at K=15 within this budget — consistent with, not contradicting, Phase 10's other finding that adding Lax-Friedrichs dissipation margin also failed to fix it (and non-monotonically so). Together these two negative results point the same direction: this is not primarily a training-schedule or dissipation-tuning problem, it's a structural one. A second paper from the same review, "Learning Under Laws" (constraint-projected neural PDE solvers, 2025), names the likely gap directly: a state clamp (what this project's `euler_positivity_guard` does) *"only enforces positivity, not conservation or entropy"* — their fix is a proper differentiable projection onto the intersection of positivity, conservation, entropy, and Rankine-Hugoniot constraints simultaneously (via Dykstra's method, ~10-12% overhead), which was not attempted here.

**Status**: the literature-motivated fix helped, quantifiably, but did not close the gap — itself useful information: it rules out "just needed a gentler training schedule" as the full explanation and strengthens the case that a structural constraint-projection mechanism, not a schedule or dissipation-margin change, is the next thing worth trying.

---

## Phase 12 — the professor's own answers, applied and tested

**Goal**: the professor answered the 5 open questions directly (informal, in French). Four resolved cleanly; the fifth (positivity) pointed at a real design flaw in this project's own Euler stability mechanism, not just a clarification — worth testing immediately rather than filing away.

**Resolutions**:
1. **`c̃_WENO/TENO`**: confirmed exactly as implemented — "c_k ou d_k... les coefficients de WENO ou TENO linéaires optimaux." No code change; `scheme_linear_weights` in `feature_vector.py` was already this.
2. **K (rollout length)**: the professor's answer ("K c'est soit 5... pour le WENO5") reads as tying K to the scheme's own stencil size rather than answering "how many time steps to unroll" directly — genuinely ambiguous against Eq. 4's usage, flagged rather than silently resolved one way.
3. **Total step count**: "dépend du schéma RK choisi et du nombre CFL" — confirms the existing implementation (CFL-derived `dt`/step count in every training/eval script here) is the right general approach; not a new finding.
4. **`L_BVD`**: pointed directly to the source, Sun, Inaba & Xiao (2016), "Boundary Variation Diminishing (BVD) reconstruction." Read in full via ar5iv. The real definition: `BV(Phi)_{i+1/2} = |Phi_i(x_{i+1/2}) - Phi_{i+1}(x_{i+1/2})|` — each cell's own candidate reconstruction compared against its *neighbor's own* reconstruction at their shared interface, purely self-referential. This is meaningfully different from what Phase 5 had implemented (reconstructing one stencil forwards and reversed in place — comparing a stencil to its own mirror image, not to a genuine neighbor). Fixed: `numerics_jax.boundary_variation_jax` (scalar) and `euler_jax.euler_boundary_variation_jax` (Euler) now use the correct whole-array flip-and-shift construction (reusing the same trick `FluxSplittingMethod.flux` already relies on in the NumPy code, not reinvented) to get each cell's *neighbor's own* one-sided value, not a mirror of the same stencil. `losses_jax.l_bvd` and `unrolled_physics_loss`'s signature updated to take precomputed boundary-variation arrays instead of stencils + reconstruct callables, since the correct computation needs full-array context the loss module has no business owning.
5. **Positivity**: *"si un point isolé exhibe une négativité, on le corrige en prenant... la moyenne des points voisins, mais si plusieurs, c'est que le schéma est instable."* This is not what `euler_positivity_guard` did — it unconditionally clamped density/velocity everywhere, isolated violation or not, and never told the training loss anything had happened. Replaced with `euler_positivity_repair`: isolated violations (both neighbors still valid) get repaired via neighbor-averaging of the conservative state; anything a repair doesn't fix (an isolated repair that's itself still invalid, or a genuine multi-cell cluster) falls back to a last-resort floor purely to keep `jax.lax.scan` tracing finite. `positivity_penalty_jax` (built in Phase 1, never actually used) is now wired into `unrolled_physics_loss` as a real term (`lambda_positivity`), specifically so a violation the repair fixes is not also invisible to training.

**Testing the fix immediately caught a real regression, not a hypothetical one.** The first version of `euler_positivity_repair`'s last-resort floor used `eps=1e-8` (a generic near-zero constant) with no velocity cap. Rerunning the Phase 11 curriculum script with it made things *worse*: the K=2 stage, which previously completed cleanly, now diverged at step 37/60. Root cause, caught by immediately rerunning rather than assuming the "more correct" repair must be an improvement: `eps=1e-8` is many orders of magnitude below this problem's O(1) density/pressure scale, so local sound speed (`~sqrt(pressure/density)`) exploded exactly the way Phase 10's original `eps` mistake did — the same lesson, forgotten and rediscovered inside a single edit. Fixed by reusing Phase 10's own tuned value (`eps=5e-2`) and adding a velocity cap to the last-resort path too (the same fix Phase 10 needed for the same reason).

**With that fixed, re-running the curriculum shows real, measurable further progress** — the furthest this project has gotten Euler training, across every attempt in Phases 10-12:

| Attempt | K=2 | K=4 | K=8 | K=15 |
|---|---|---|---|---|
| Phase 10 (no curriculum, straight to K=15) | -- | -- | -- | fails at step 0 |
| Phase 11 (curriculum, old clamp-only guard) | completes | fails at step 39/60 | -- | -- |
| Phase 12 (curriculum, neighbor-repair + correct BVD + eps regression fixed) | completes | **completes** | fails at step 6/60 | -- |

K=4 now trains cleanly end to end for the first time; the instability boundary has moved from "inside K=4" to "inside K=8." Held-out evaluation is unchanged (0/3, same numbers as every prior attempt) because the run still doesn't reach K=15, so the final checkpoint is still NaN-corrupted the same way — but the *trend* across three independent attempts (Phase 10 → 11 → 12) is monotonic, real, and directly attributable to implementing the professor's actual described practice rather than an invented one.

**Honest limitation, caught by checking rather than assuming**: `lambda_positivity`'s penalty logged as exactly `0.0` on every single step across the entire Phase 12 run. Reason, found by inspection: `positivity_penalty_jax` is computed on `final_state`, but `final_state` is the output of `rollout(..., state_guard=euler_positivity_repair)` — by the time the loss sees it, every violation has already been repaired. The penalty term is currently **structurally inert**, not just small — it can never be nonzero as wired, because it only ever observes already-clean states. Making it actually functional needs the repair step to also report how much it had to correct (e.g. accumulated as a `lax.scan` carry alongside the state, across all K steps, not recovered after the fact from the final state alone) — this is real, unfinished work, not a rounding-error detail, and is the concrete next step if Euler training is pursued further.

**Status**: applying the professor's own answers produced a genuine, measured improvement (K=4 now stable, further than any prior attempt) and caught a real regression in the process of testing it (the eps mistake), consistent with this project's practice of verifying every claimed fix by rerunning rather than trusting the reasoning alone. It still doesn't reach K=15, and one of the three new mechanisms (`lambda_positivity`) is currently a no-op that needs its own follow-up fix to actually do anything. Recommended framing: real, incremental, honestly-bounded progress — not a closed case.
