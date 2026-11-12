# NN-TENO5 accuracy upgrade: measured research report

## 1. Objective and decision

Improve accuracy, admissibility and generalization while preserving existing numerical methods and artifacts. **The finite-volume consistency correction is supported; a generally superior upgraded neural method is not established.** Keep the original default and checkpoints. Use the new state-flux path and conservative safeguard as explicitly selected research options.

The completed work covers the Euler consistency correction, conservative safeguards, paired input-matching pilots, scalar/Euler controls, a frozen six-trajectory confirmation set and isolated CPU cost measurements. A richer mixing head and full end-to-end retraining are **deferred**, not presented as completed improvements. Mixed paired training effects and substantial stress-case regressions do not justify increasing model complexity.

## 2. Baseline and verified limitations

Starting commit: `f9d4c2ae7e6009430248f71774996d952e08abbd`; branch `research/accuracy-upgrade-2026-11-12`. CPU: AMD Ryzen 5 4600H with Radeon Graphics; Python 3.11.15, JAX/jaxlib 0.10.2, NumPy 2.4.6, Optax .2.8, h5py 3.16.0. The baseline CPU suite passed 117 tests and 37 subtests. Forty-four established smoke results reproduce archived norms/validity to 1e-12. The separate legacy environment passed two tests. `pip check` passed. A fresh install and GPU execution were not performed.

The historical Euler FV inputs are inconsistent for general nonlinear fluxes; V2 state warm-up differs from split-flux inference; historical repairs can alter conservation; fixed/learned corrections have case-dependent error and stability. The PDF feature vector exists separately and is absent from the saved five-input architecture. The saved V2 checkpoints contain backbones with fixed mixing, while historical heads consume probabilities alone or probabilities plus normalized stencils. Exact weights-only historical training provenance cannot be fully reconstructed. See [audit and mathematical notes](AUDIT_AND_METHOD.md) and [baseline manifest](../baseline_manifest.json).

## 3. Implemented changes and mathematical justification

`src/differentiable/accuracy_upgrade.py` adds characteristic **state-average** reconstruction followed by a two-state Rusanov flux, canonical bidirectional state stencils, periodic/transmissive/reflecting-wall support, optional face admissibility scaling, a global LF reference flux, global shared-flux correction, per-FE-like-stage checks and bounded whole-step retries. Historical factories and source arithmetic are unchanged. No conservative average is clipped or repaired.

The consistency correction addresses `F(Ubar) != average(F(U))`. Conservative updates use one shared flux per interface. The fallback's ideal-gas LF convexity argument requires a valid wave-speed bound and CFL restriction; the implementation also verifies candidate states. A global segment coefficient uses convex density/pressure admissibility, not an assertion that arbitrary high-order mixing is positive. Accepted states and recomputed flux updates are checked. Rejected timesteps advance zero time and contribute zero exchange. See the audit notes for assumptions, boundary budgets, stop-gradient decisions and applicability of the cited FV/DG/FD literature. No general TVD, entropy stability, fifth-order-in-time or unconditional positivity theorem is claimed.

## 4. Protocol

Exact averages and exact face-flux differences isolate spatial error on N=40/80/160/320. Entropy-wave rollouts use N=40/80/160, t=.02, SSPRK3, dt≈.2 dx^(5/3), and dt/2 controls. Shock runs use N=64/128 and identical proposed timesteps across seven variants. Safeguarded retries, if needed, are reported separately. Sod/Lax/rarefaction/contrast/near-vacuum references are analytic Riemann averages. Blast is stability-only. Short-time Shu–Osher uses a 256/512/1024 reference ladder and finest dt/2; its empirical uncertainty is explicitly retained.

Paired A/B/C use seeds 3101/3102, architecture 5→64→32→16→8→3 (3155 parameters), matched 150 warm-up steps and 60 physics updates, K=15, Adam 2e-4 and gradient clipping at 1. A deliberately retains mismatched state warm-up; B uses canonical historical split-flux inputs; C uses canonical FV state inputs with face scaling. Training uses N=64 and full N=64/128 validation. This is smaller than saved V2's 300-update, three-grid budget. Pilot training uses explicit rejection instead of the original nonconservative guard; no rejection occurs. Fixed final checkpoints prevent post-hoc checkpoint selection. Training and validation trajectory IDs are disjoint; historical test datasets are never opened as datasets.

The fresh six-Riemann holdout was seeded and hashed before training. All candidate weights were frozen before its evaluation. It is confirmation only and does not select hyperparameters. Paired bootstrap intervals resample independent trajectories separately at each grid (2000 draws, seed 2026111202), condition on the fixed candidates and do not quantify training-seed or reference uncertainty. Six moderate Riemann IVPs are a narrow population.

Euler aggregate errors average unscaled conserved components with different units; component norms are also retained. Scalar families and grid resolutions are reported separately. No scalar/Euler combined headline statistic is used.

## 5. Smooth accuracy

Last spatial orders, N=160→320:

| Profile | Historical TENO5 order | FV TENO5 order | FV saved V2 order |
| --- | --- | --- | --- |
| density | 4.9998 | 4.99908 | 4.99911 |
| momentum | 2.02842 | 4.029 | 4.04407 |
| pressure | 5.00501 | 5.00397 | 5.00397 |
| coupled | 2.02813 | 5.00027 | 5.00027 |
| critical | 3.22851 | 4.02522 | 4.03332 |


The coupled test recovers order ≈5 with both classical and saved V2 reconstruction. **This is a numerical-formulation result.** Density-only and pressure-only tests are special profiles and do not reveal the full historical limitation. Momentum-only and critical-point FV cases remain near fourth order. The always-retain-candidates diagnostic recovers approximately fifth order on those same profiles while the TENO selector rejects 34 and 78 candidate observations respectively at every tested grid. This controlled ablation implicates smooth-region candidate selection; no unvalidated selector modification is promoted.

The completed entropy-wave time-dependent ladder approaches fifth error reduction when dt shrinks as dx^(5/3). SSPRK3 itself remains third order in time. The dt/2 Cauchy controls and all absolute errors are in `results/temporal.json`. This special linear-in-density Euler solution is not a general nonlinear time-dependent convergence proof.

## 6. Euler accuracy, admissibility and conservation

Mean conserved-component L1; a dash means failed trajectory, not zero error:

| Case, N=128 | Historical TENO5 L1 | Historical V2 L1 | FV TENO5 L1 | FV V2 L1 | Safe FV TENO5 L1 | Safe FV V2 L1 |
| --- | --- | --- | --- | --- | --- | --- |
| sod | 0.00421063 | 0.00387207 | 0.00391322 | 0.0036237 | 0.00391322 | 0.0036237 |
| lax | 0.0154585 | 0.0138616 | 0.0152468 | 0.0138532 | 0.0152468 | 0.0138532 |
| two_rarefactions | 0.0123607 | — | — | — | 0.0121666 | 0.0257889 |
| contrast | — | — | — | — | 0.260434 | 0.287339 |
| near_vacuum | 1.23972e-09 | 8.41124e-10 | 1.05576e-09 | 6.73837e-10 | 1.05576e-09 | 6.73837e-10 |


Sod component errors:

| Sod N=128 variant | Density L1 | Momentum L1 | Energy L1 |
| --- | --- | --- | --- |
| historical_teno | 0.00335391 | 0.00268092 | 0.00659708 |
| historical_p0_learned | 0.00330276 | 0.00263584 | 0.00649564 |
| historical_v2_soft25 | 0.00312213 | 0.00246748 | 0.00602659 |
| fv_teno | 0.00316385 | 0.0025511 | 0.00602471 |
| fv_v2_soft25 | 0.0029549 | 0.00234214 | 0.00557406 |
| safe_fv_teno | 0.00316385 | 0.0025511 | 0.00602471 |
| safe_fv_v2_soft25 | 0.0029549 | 0.00234214 | 0.00557406 |


Failures and conservative boundary-budget residuals across 16 configurations (eight cases × two grids). Blast smoke/full runs are correlated stress cases, not independent statistical samples:

| Variant | Failed configurations / 16 | Min accepted density | Min accepted pressure | Max |mass budget| | Max |momentum budget| | Max |energy budget| |
| --- | --- | --- | --- | --- | --- | --- |
| historical_teno | 2 | 8.81978e-06 | 8.48005e-09 | 3.61473e-14 | 2.12275e-13 | 1.24498e-11 |
| historical_p0_learned | 2 | 8.81992e-06 | 8.48011e-09 | 3.66096e-14 | 2.16716e-13 | 1.27405e-11 |
| historical_v2_soft25 | 8 | 8.76415e-06 | 8.39309e-09 | 3.08642e-14 | 2.75335e-14 | 1.1724e-13 |
| fv_teno | 8 | 8.80142e-06 | 8.45039e-09 | 2.86438e-14 | 2.30926e-14 | 9.9476e-14 |
| fv_v2_soft25 | 8 | 8.75116e-06 | 8.37567e-09 | 2.93099e-14 | 2.66454e-14 | 1.13687e-13 |
| safe_fv_teno | 0 | 8.55528e-06 | 8.12604e-09 | 3.66686e-14 | 2.17604e-13 | 1.25335e-11 |
| safe_fv_v2_soft25 | 0 | 8.49737e-06 | 1.04362e-12 | 3.92724e-14 | 2.24709e-13 | 1.24205e-11 |


Both safeguarded FV variants complete all 16 tested configurations with admissible accepted states, including the full t=.038 reflecting-wall blast. This is finite-corpus evidence only. The historical classical path completes the blast at these settings; it fails the stronger contrast case. Unlimited FV and saved V2 paths fail eight configurations each. Their failures remain visible with first failed step, raw stage diagnostics and partial-time metadata.

At 128-cell two-rarefactions the safeguarded neural error is about twice the safeguarded classical error. The global safeguard can suppress reconstruction far from the troubled location; it is deliberately conservative and can be dissipative. It is retained as an option. In smooth entropy-wave runs it is inactive and reproduces the corresponding unsafeguarded errors to floating-point tolerance. Conservation residuals account for actual boundary fluxes; reflecting-wall momentum exchange is a wall-pressure force, while wall mass/energy fluxes are zero. Large absolute blast energy residuals must be interpreted against its large energy scale and many timesteps, not confused with exact arithmetic.

`results/euler_actual_stages/` is authoritative. `results/euler/` preserves an earlier, overly strict raw-wrapper diagnostic; the corrected wrapper is checked against original SSPRK3. Safeguarded kernels intentionally check FE-like updates more strictly than raw SSPRK convex stages. No repair is used to produce any accepted result.

Short-time Shu–Osher (t=.15, length 10):

| N | Variant | Density L1 | Aggregate L1 | Reference uncertainty proxy | Ranking resolved |
| --- | --- | --- | --- | --- | --- |
| 64 | historical_teno | 0.0211313 | 0.0864747 | 0.00659793 | False |
| 64 | historical_v2_soft25 | 0.0180588 | 0.0738259 | 0.00659793 | False |
| 64 | fv_teno | 0.0206969 | 0.0805962 | 0.00659793 | False |
| 64 | fv_v2_soft25 | 0.0182927 | 0.0692664 | 0.00659793 | False |
| 128 | historical_teno | 0.00786126 | 0.03166 | 0.00882778 | False |
| 128 | historical_v2_soft25 | 0.00680915 | 0.0251782 | 0.00882778 | False |
| 128 | fv_teno | 0.00862707 | 0.0331764 | 0.00882778 | False |
| 128 | fv_v2_soft25 | 0.00725029 | 0.0250835 | 0.00882778 | False |


The uncertainty proxy is twice the sum of the final spatial and temporal Cauchy changes. It is not a rigorous error bound. No ranking is claimed when it exceeds 25% of the method margin. This short-time case is not evidence for canonical t=1.8 shock–acoustic accuracy.

## 7. Matched-input training and generalization

| Seed | Variant | Matched-probe warm BCE | Warm classification accuracy | Warm validation L1 | Final validation L1 | Validation failures | Wall seconds, incl. shared compilation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3101 | A | 0.77747 | 0.80665 | 0.00848978 | 0.00845828 | 0 | 113.95 |
| 3101 | B | 0.78511 | 0.806018 | 0.00847691 | 0.00845444 | 0 | 39.3636 |
| 3101 | C | 0.773085 | 0.821229 | 0.0084319 | 0.00839346 | 0 | 83.6009 |
| 3102 | A | 0.772017 | 0.809492 | 0.00848738 | 0.00838908 | 0 | 38.1937 |
| 3102 | B | 0.79033 | 0.813856 | 0.00846105 | 0.00839998 | 0 | 40.0279 |
| 3102 | C | 0.779799 | 0.813041 | 0.00845127 | 0.00830039 | 0 | 40.266 |


Matched B versus mismatched A changes final validation mean L1 by **-0.04543%** and **+0.12993%** for the two paired seeds. This inconsistent, small effect does not establish improved numerical generalization. Classification changes alone are insufficient. C has lower overall validation error, but it also changes the solver; the effect cannot be attributed solely to corrected pretraining. All six pilots have zero recorded nonfinite gradients, zero rejected physics updates and zero validation failures. Detailed losses, gradient norms, IDs, seeds, checkpoint hashes and budgets are saved beside separate warm/final checkpoints. Training times include compilation reuse and are not a fair A/B/C throughput comparison.

Frozen confirmation set, paired against historical classical TENO5:

| N | Variant | Mean paired L1 ratio to historical TENO5 | 95% paired bootstrap interval | Worst trajectory ratio |
| --- | --- | --- | --- | --- |
| 64 | historical_v2_soft25 | 0.927958 | 0.91717–0.93570 | 0.938052 |
| 64 | fv_teno | 0.957379 | 0.93726–0.97447 | 0.983831 |
| 64 | fv_v2_soft25 | 0.898605 | 0.87800–0.91920 | 0.935868 |
| 64 | safe_fv_v2_soft25 | 0.898605 | 0.87800–0.91920 | 0.935868 |
| 128 | historical_v2_soft25 | 0.930132 | 0.92147–0.93803 | 0.944996 |
| 128 | fv_teno | 0.942556 | 0.92930–0.95749 | 0.976711 |
| 128 | fv_v2_soft25 | 0.875625 | 0.85931–0.88771 | 0.891913 |
| 128 | safe_fv_v2_soft25 | 0.875625 | 0.85931–0.88771 | 0.891913 |


These rows isolate the classical numerical correction and the additional saved neural correction. The complete table also includes both seeds of every pilot and comparisons to **FV classical** TENO5. A/B effects remain small or unfavorable on confirmation; no holdout retuning is performed. Seed 3102 matched B is worse than paired A at both grids. The favorable saved V2 + FV Riemann results do not erase the rarefaction/stress regressions.

Longer frozen-checkpoint Sod evaluation at t=.2, with sampled analytic error curves:

| N | Frozen pilot | Sod t=.2 L1 | Valid |
| --- | --- | --- | --- |
| 64 | historical_teno | 0.00833456 | True |
| 64 | historical_v2_soft25 | 0.00766181 | True |
| 64 | fv_teno | 0.00793985 | True |
| 64 | safe_fv_teno | 0.00793985 | True |
| 64 | historical_pilotA_seed3101 | 0.00779124 | True |
| 64 | historical_pilotB_seed3101 | 0.00779163 | True |
| 64 | fv_faceguard_pilotC_seed3101 | 0.00743847 | True |
| 64 | historical_pilotA_seed3102 | 0.00798699 | True |
| 64 | historical_pilotB_seed3102 | 0.00807038 | True |
| 64 | fv_faceguard_pilotC_seed3102 | 0.00740992 | True |
| 128 | historical_teno | 0.00460913 | True |
| 128 | historical_v2_soft25 | 0.00428714 | True |
| 128 | fv_teno | 0.00434191 | True |
| 128 | safe_fv_teno | 0.00434191 | True |
| 128 | historical_pilotA_seed3101 | 0.00426411 | True |
| 128 | historical_pilotB_seed3101 | 0.00426595 | True |
| 128 | fv_faceguard_pilotC_seed3101 | 0.0040537 | True |
| 128 | historical_pilotA_seed3102 | 0.00437462 | True |
| 128 | historical_pilotB_seed3102 | 0.00442452 | True |
| 128 | fv_faceguard_pilotC_seed3102 | 0.00407819 | True |


These runs contain many more than 15 SSPRK3 steps. All shown trajectories complete. They do not test arbitrary long-time turbulent or shock-interaction behavior. See `results/long_rollout/` and figures for the full error curves.

## 8. Unchanged scalar controls

| Case | N | TENO5 L1 | Saved V2 / TENO5 L1 | Historical learned / TENO5 L1 |
| --- | --- | --- | --- | --- |
| sine | 64 | 4.64626e-08 | 1 | 22.4092 |
| sine | 128 | 4.00416e-09 | 1 | 33.0187 |
| sine | 256 | 4.41438e-10 | 1 | 37.5975 |
| fourier | 64 | 8.28637e-06 | 1 | 4.19149 |
| fourier | 128 | 3.65214e-07 | 1 | 14.0185 |
| fourier | 256 | 2.42463e-08 | 1 | 27.4236 |
| gaussian | 64 | 5.19697e-05 | 0.999986 | 1.29031 |
| gaussian | 128 | 1.91881e-06 | 1 | 6.36056 |
| gaussian | 256 | 9.00743e-08 | 1 | 18.4923 |
| square | 64 | 0.0145638 | 0.954715 | 0.962032 |
| square | 128 | 0.0157602 | 0.958949 | 0.966561 |
| square | 256 | 0.00861167 | 0.956551 | 0.966394 |
| burgers_smooth | 64 | 9.46918e-08 | 1 | 10.449 |
| burgers_smooth | 128 | 3.4897e-09 | 1 | 37.32 |
| burgers_smooth | 256 | 2.21556e-10 | 1 | 74.432 |
| burgers_hat | 64 | 0.00734648 | 0.893167 | 0.945769 |
| burgers_hat | 128 | 0.00328505 | 0.912007 | 0.94901 |
| burgers_hat | 256 | 0.00188947 | 0.94489 | 0.964722 |


The scalar solver was not changed. Burgers smooth is pre-shock; Burgers top-hat includes a moving entropy shock before rarefaction–shock interaction. Its shock positions, speeds, widths, TV and oscillation metrics are saved. No generic post-interaction accuracy is claimed. Smooth advection/Fourier/Gaussian and discontinuous square cases include L1/L2/Linf, overshoot, undershoot, TV, runtime and conservation. Errors remain case- and resolution-dependent; saved neural policies are not universally better.

## 9. Computational cost

Isolated CPU Sod t=.15, N=128; nine synchronized warmed production repeats, without host diagnostic/export overhead:

| N | Variant | Warmed simulation s | Compile s | Seconds / step | L1 |
| --- | --- | --- | --- | --- | --- |
| 128 | historical_teno | 0.138578 | 1.40237 | 0.00101152 | 0.00421063 |
| 128 | historical_p0_learned | 0.345048 | 1.88735 | 0.0025186 | 0.00414475 |
| 128 | historical_v2_soft25 | 0.349852 | 2.00839 | 0.00255366 | 0.00387207 |
| 128 | fv_teno | 0.130118 | 1.25057 | 0.000949768 | 0.00391322 |
| 128 | fv_v2_soft25 | 0.333573 | 1.76348 | 0.00243484 | 0.0036237 |
| 128 | safe_fv_teno | 0.144962 | 1.98634 | 0.00105812 | 0.00391322 |
| 128 | safe_fv_v2_soft25 | 0.353631 | 2.61134 | 0.00258125 | 0.0036237 |


`results/performance.json` contains N=64/128/256, per-step costs, compilation/first execution, executable temporary-memory estimates, process lifetime RSS and discrete-grid equal-budget comparisons. It measures production kernels after other numerical jobs have finished. RSS is not a per-model allocation and compiler memory estimates are not GPU peak memory. Neural corrections can be slower enough that classical TENO5 at a finer measured grid wins under a fixed budget. Do not infer a speedup from lower same-grid error. Exploratory per-case runtime fields include host diagnostics and sometimes concurrent jobs; they are not the headline timing evidence.

## 10. Gates, negative findings and recommendation

- **A passed narrowly:** fifth-order coupled smooth spatial behavior is reproduced. Uniform fifth order, including critical points and general full-time Euler flow, is not established.
- **B retained experimentally:** conservative safeguards support the tested strong states and walls, but global corrections and neural mixing can be inaccurate and costly. They stay opt-in.
- **C did not establish a gain:** two paired input-matching pilots have opposite small effects. Preserve both results and checkpoints.
- **D deferred:** no physics-aware head or joint fine-tuning is justified by this pilot. Existing P0 learned-head rows have their own backbones and are not a controlled head-only comparison against V2.
- **E no global promotion:** keep existing defaults; offer FV state reconstruction as the consistency experiment, and the conservative safeguard when admissibility is required. Freeze the classifier architecture and prefer classical FV as the control for any further neural experiment.

The strongest contribution is isolating a solver consistency correction and replacing hidden repairs with an explicit conservative acceptance mechanism. A defensible next step is a local conservative correction that avoids global suppression, followed by trajectory-level validation and a larger multi-seed budget. Smooth critical-point selection deserves a separate ablation. Neither is implemented or claimed here.

## 11. Limits and unfinished work

No complete N=256/512 Euler shock sweep, canonical full-time Shu–Osher reference, generic coupled Euler full-time fifth-order proof, formal high-order positivity/entropy theorem, GPU study, full original-budget retraining, physics-aware head or exhaustive hyperparameter search is completed. Scalar interaction references and blast accuracy remain unassessed. The small holdout has only six moderate Riemann trajectories and should not support broad fluid-generalization claims. The architecture is unchanged; the fresh checkpoints are pilot artifacts, not recommended replacements.

All original checkpoints/data/historical outputs present in the frozen inventory are unchanged: **592 files verified**, and all **491 tracked release-protected artifacts** pass. Seven ignored historical `.best.h5` files from the old 498-file inventory were already absent at baseline. The full archived check still reports exactly those seven missing files; none is invented or silently restored. See `results/preservation_final.json`. The CPU regression run passed 130 tests and 37 subtests; two additional independent NumPy/wrapper tests passed separately, for 132 unique CPU tests. The separate legacy environment passed two tests. Initial failing development assertions and their corrections are preserved in logs rather than erased.

## 12. Reproduction and artifacts

See [README](../README.md) for commands and [audit notes](AUDIT_AND_METHOD.md) for mathematical assumptions and primary references. Every numerical command requires a fresh output path. Use the recorded environment freeze; installing it freshly is unverified. `baseline_manifest.json` records provenance, hashes and blockers. `artifact_hashes.json` verifies the new deliverables. Reports and all plot curves derive only from recorded solver outputs.
