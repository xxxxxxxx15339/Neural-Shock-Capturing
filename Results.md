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

**Status**: not yet accepted. `anchor_weight=20.0` remains the most reliable configuration found (consistently passes the core `shu_regional_l1` metric), but 88 seeds in, plain seed search at this exact setting shows no sign of closing both remaining gaps at once. A different lever (e.g. sweeping `anchor_weight` again now that the checkpoint has moved, or a criterion-specific mechanism for `sod_total_variation` similar to what closed `shu_regional_l1`) is more likely to help than further identical-configuration seed search.

New CLI arguments added to `Train_WENO7NN_Euler.py`: `--warm-start-path`, `--protect-riemann-boost`, `--anchor-weight`. All default to off/1.0/0.0, so existing from-scratch training behavior is unchanged.

Artifacts (all gitignored, local only, all still best-*rejected*-candidate checkpoints):
- `trained_euler_weno7nn_finetuned.h5.best.h5` — plain fine-tuning, worse than checkpoint.
- `trained_euler_weno7nn_protected.h5.best.h5` — riemann-protected fine-tuning, no improvement over plain.
- `trained_euler_weno7nn_anchored.h5.best.h5` — anchor_weight=20 fine-tuning, closest result on Sod TV (30/30 pass shu_regional_l1).
- `trained_euler_weno7nn_anchored2.h5.best.h5` — follow-up batch, stopped early by request after 8/50 attempts.
- `trained_euler_weno7nn_anchored3.h5.best.h5` — full 50-seed resumed batch, closest result on shu_regional_linf.
