# Full conservative FV NN-TENO5 training and problem tests

Two new models were trained from scratch, with the existing 3155-parameter backbone, through the actual finite-volume-consistent Euler solver and its conservative SSPRK3 safeguard. The primary candidate is **seed 4101, PDE update 200**, chosen solely by validation error. Its checkpoint is:

`experiments/accuracy_upgrade/training_2026_11_12/checkpoints/fv_conservative_seed4101_step0200.pkl`

SHA-256: `186350338c6fd57ce3bb15ca30a1f828b1aeac687ea0028005cf0696dd251073`. Both seeds, all evaluated checkpoints, final optimizer/sampler states and complete histories are retained. No old checkpoint, dataset or result is overwritten. This report presents all requested completed comparisons; lower validation loss alone is not a general accuracy or stability claim.

The validation-selected primary changes validation macro L1 by **-1.313% relative to old V2 in the same conservative FV solver**. On the tested analytic canonical configurations it has lower error than old V2 in 5/20 and higher error in 15/20. These are correlated configuration counts, not independent statistical samples. Its worst ratio to old V2 is **1.1884** on near_vacuum at N=64; its worst ratio to conservative classical FV TENO5 is **2.0106** on two_rarefactions at N=128. These regressions must accompany favorable means. Fresh-holdout uncertainty and cost are reported separately below.

**Recommendation: retain the old saved V2 weights as the preferred existing neural checkpoint; do not promote these new weights as a general replacement.** Fresh-holdout mean L1 changes relative to old V2 in the same solver are: safe_fv_new_seed4101, N=128: +2.92%; safe_fv_new_seed4102, N=128: +1.95%; safe_fv_new_seed4101, N=256: +8.80%; safe_fv_new_seed4102, N=256: +5.69%. The improved short-horizon validation score did not generalize to this predeclared set. No post-holdout retuning is performed. This is a measured negative generalization result, not a failed training job.

## Training protocol and what changed

- Architecture: 5→64→32→16→8→3, unchanged; deterministic TENO gate, soft neural weights and fixed λ=.25. No richer mixing head or architecture search.
- Canonical warm-up: both directions of the actual characteristic state stencils, 150 teacher steps, batch 256, at most 15000 balanced training stencils, sampled from training trajectories at N=64/128/256.
- Physics training: 300 attempted Adam updates per seed, batch 8, learning rate .0002, global-norm clipping at 1; K=15 actual SSPRK3 steps at each trajectory's own dt. All three grids participate. The objective is per-trajectory component-normalized solution MSE; no TV/BVD terms are added. Rejected batches consume an attempt and are not resampled away; applied-update counts are reported separately.
- Every SSPRK3 FE-like stage uses physical face scaling and shared high/low flux correction. Conservative averages are never repaired. Limiter coefficients and switching decisions are stop-gradient. Failed fixed-time training rollouts or nonfinite gradients reject the optimizer update; three consecutive rejections stop training.
- `lax.map` preserves lazy safety branches across independent training trajectories. The gradient smoke and invalid-batch rejection tests pass.
- Validation uses all 307 Euler validation trajectories at N=64/128/256. Select the smallest zero-failure macro family×grid mean paired L1 ratio against conservative classical FV TENO5, including warm-up. No original test data are read. Both seeds have identical declared budgets; exact sampling histories are logged.

This is a larger experiment than the earlier N=64/60-update pilots. It matches the saved V2 architecture, declared teacher/PDE-attempt budgets, batch size and grid set, but changes the solver, safeguard, normalization and sampling. It is not an input-only causal ablation. The old V2 checkpoint is explicitly reevaluated in the **same** conservative solver to separate training effects from numerical-formulation effects.

In the FV path, λ mixes characteristic **state reconstructions before** evaluating the nonlinear Rusanov flux. This is not a convex blend of two already-computed physical Euler fluxes. The independent admissibility coefficient θ limits shared interface fluxes afterwards. Neither a bounded λ nor neural candidate probabilities alone establish positivity, TVD or entropy stability.

## Validation results and checkpoint selection

Old V2 in the same conservative solver has validation macro L1 ratio **0.956206** to classical FV TENO5.

| Seed | Selected attempt | Applied / attempted PDE updates | Validation L1 ratio / conservative classical | Relative to old V2, same solver | Nonfinite gradient updates | Rejected training updates | Wall seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4101 | 200 | 300/300 | 0.943646 | 0.986865 | 0 | 0 | 733.646 |
| 4102 | 200 | 298/300 | 0.949141 | 0.992612 | 0 | 2 | 696.174 |


Validation curves include every 50-update evaluation, including unfavorable later evaluations. Final training checkpoints are not automatically preferred. Training times include compilation, validation, logging and concurrent regression/control work; do not compare seed timings as pure optimizer throughput. Per-grid first compilations, warmed update medians, loss, gradient norm, limiter activation, stage minima and actual batch case IDs are saved in the histories.

Rejected validation checkpoints: seed 4102, update 300: 3 failed fixed-dt trajectories. Their partial-rollout errors are diagnostic only and are not valid accuracy scores. They are excluded from the validation accuracy curve and checkpoint selection. Inference can retry smaller timesteps, but fixed-time training/validation does not silently change its 15-step budget.

Diagnostic replays of rejected validation trajectories, without using them to select or retune a model:

| Rejected checkpoint case | Mapped failure reproduced | Max low-order CFL | Adaptive inference completed | Retry count | Max conservation residual |
| --- | --- | --- | --- | --- | --- |
| euler-validation-000249-9148285969347245035 | True | 14.239 | True | 3 | 1.40166e-15 |
| euler-validation-000037-2046751549353970294 | True | 0.95675 | True | 3 | 7.84095e-16 |
| euler-validation-000100-2974132583243714859 | True | 1.44964 | True | 1 | 5.91541e-16 |


The configured low-order CFL ceiling is .9. The mapped replay retains the original batch shape and parameter structure. Bounded-retry inference keeps the physical target time and records accepted boundary exchanges; it is not a replacement for the failed fixed-step validation score or a reason to promote that rejected checkpoint.

Training-stage diagnostics, across all sampled batches:

| Seed | Pre-flux-correction invalid cell rows | Global flux-stage activations | Global face-stage activations | Min density | Min pressure | Max low-order CFL |
| --- | --- | --- | --- | --- | --- | --- |
| 4101 | 0 | 0 | 0 | 0.0895788 | 0.0690448 | 0.327743 |
| 4102 | 39 | 39 | 58 | 1.35466e-05 | 6.42471e-11 | 15.9121 |


The safeguard is part of the actual rollout even where it is inactive. Inactive training corrections cannot establish that the network learned positivity or learned how to handle blast/near-vacuum conditions. Any stability provided by active test-time limiting belongs to the numerical safeguard, rather than being credited to the neural network.

## Canonical Euler problems

All methods use identical initial cell averages, proposed timesteps, final physical times and references. The FV controls use the same safeguard as the newly trained networks. Historical TENO5, historical V2, the historical P0 learned model and unlimited FV controls are also rerun. Bounded inference retries advance only accepted physical time. A failed trajectory has no accuracy score.

Aggregate L1 averages unscaled conserved components with different units; density/momentum/energy L1/L2/Linf and pressure error are also exported. Dashes indicate failed accuracy trajectories.

| Case | N | Conservative FV TENO5 L1 | Same solver + old V2 L1 | New seed 4101 L1 | New seed 4102 L1 |
| --- | --- | --- | --- | --- | --- |
| sod | 128 | 0.00391322 | 0.0036237 | 0.00370848 | 0.00368875 |
| sod | 256 | 0.00207148 | 0.00187543 | 0.00199992 | 0.00200668 |
| lax | 128 | 0.0152468 | 0.0138532 | 0.0148996 | 0.0155533 |
| lax | 256 | 0.0101389 | 0.0089188 | 0.0101757 | 0.0105088 |
| two_rarefactions | 128 | 0.0121666 | 0.0257889 | 0.0244626 | 0.0194819 |
| two_rarefactions | 256 | 0.00627005 | 0.0131127 | 0.0123804 | 0.00979855 |
| contrast | 128 | 0.260434 | 0.287339 | 0.307626 | 0.301355 |
| contrast | 256 | 0.150569 | 0.161641 | 0.188106 | 0.16863 |
| near_vacuum | 128 | 1.05576e-09 | 6.73837e-10 | 5.29088e-10 | 6.66246e-10 |
| near_vacuum | 256 | 2.31863e-09 | 2.04253e-09 | 1.96983e-09 | 2.04217e-09 |
| shu_osher_full | 128 | 0.182419 | 0.179829 | 0.183983 | 0.185343 |
| shu_osher_full | 256 | 0.0798187 | 0.0757611 | 0.0782923 | 0.0783702 |


Sod is evaluated at t=.15 and again at t=.2. Lax uses t=.1, two-rarefactions t=.08, contrast t=.012 and near-vacuum t=.1. Short-time and full-time Shu–Osher use t=.15 and t=1.8 on length 10. Shock positions/speeds, contact 10–90% widths, density/pressure TV and extrema are exported for applicable analytic cases. Contact width is a diagnostic in a declared fixed neighborhood, not a universal sharpness metric.

## Stability and conservation

| Variant | Completed / attempted | Minimum accepted density | Minimum accepted pressure | Max mass residual | Max momentum residual | Max energy residual |
| --- | --- | --- | --- | --- | --- | --- |
| fv_teno | 18/30 | 8.04159e-06 | 7.45312e-09 | 1.33582e-12 | 5.40012e-13 | 8.2423e-12 |
| fv_v2_soft25 | 18/30 | 7.97344e-06 | 7.3606e-09 | 1.12621e-12 | 1.26477e-12 | 8.49809e-12 |
| historical_p0_learned | 26/30 | 8.07455e-06 | 7.49882e-09 | 1.36424e-12 | 6.6791e-13 | 1.27405e-11 |
| historical_teno | 29/32 | 8.07708e-06 | 7.50285e-09 | 1.20082e-12 | 1.47793e-12 | 2.48618e-11 |
| historical_v2_soft25 | 20/32 | 8.00999e-06 | 7.40751e-09 | 1.1724e-12 | 9.52127e-13 | 8.92442e-12 |
| safe_fv_new_seed4101 | 32/32 | 7.82444e-06 | 1.03919e-12 | 1.1191e-12 | 1.0516e-12 | 2.49625e-11 |
| safe_fv_new_seed4102 | 32/32 | 7.82853e-06 | 1.12893e-12 | 1.11555e-12 | 1.08002e-12 | 2.4921e-11 |
| safe_fv_teno | 32/32 | 7.90377e-06 | 7.27184e-09 | 1.27187e-12 | 1.19371e-12 | 2.47871e-11 |
| safe_fv_v2_soft25 | 32/32 | 7.83462e-06 | 1.04362e-12 | 1.09424e-12 | 1.37845e-12 | 2.4839e-11 |


The suite includes reflecting-wall blast smoke t=.001 and full-time t=.038. These are **stability tests only** because no trustworthy blast accuracy reference is supplied. Wall mass/energy fluxes are zero; momentum budgets include wall pressure force. Reported conservation residuals include accepted boundary exchanges with SSPRK weights. Large blast energy scales and accumulated roundoff must be considered alongside absolute residuals.

Every failed historical/unlimited trajectory remains in the machine-readable records with partial physical time and first invalid step/stage. Accepted-attempt pre-correction invalid counts, limiter activations and total retries are recorded; detailed per-retry invalid-row counts are not aggregated by the reused trajectory harness. No hidden state clipping is used to conceal failures.

## Fresh confirmation and resolution generalization

Eight independent Riemann/contact/rarefaction IVPs were seeded with 2026111203 and hashed before training. Weights were frozen after validation selection, before any new-model holdout evaluation. These differ from the previously examined holdout. No strength, optimizer, checkpoint or architecture is selected using them. Intervals use 2000 paired trajectory resamples separately at each grid, seed 2026111204. They condition on a trained model and do not quantify training-seed uncertainty. Two resolutions of an IVP and multiple times are not independent trajectories.

| N | Model | Baseline | Mean paired ratio | 95% paired interval | Worst trajectory ratio |
| --- | --- | --- | --- | --- | --- |
| 128 | safe_fv_new_seed4101 | safe_fv_teno | 0.968447 | 0.94265–1.00208 | 1.06524 |
| 128 | safe_fv_new_seed4101 | safe_fv_v2_soft25 | 1.02917 | 0.99781–1.06640 | 1.11699 |
| 128 | safe_fv_new_seed4102 | safe_fv_teno | 0.959388 | 0.93796–0.98480 | 1.02746 |
| 128 | safe_fv_new_seed4102 | safe_fv_v2_soft25 | 1.01953 | 0.99431–1.04951 | 1.08773 |
| 256 | safe_fv_new_seed4101 | safe_fv_teno | 1.01528 | 0.93958–1.10203 | 1.19526 |
| 256 | safe_fv_new_seed4101 | safe_fv_v2_soft25 | 1.08799 | 1.02517–1.16003 | 1.26122 |
| 256 | safe_fv_new_seed4102 | safe_fv_teno | 0.985531 | 0.93094–1.04956 | 1.16465 |
| 256 | safe_fv_new_seed4102 | safe_fv_v2_soft25 | 1.05692 | 1.01206–1.10946 | 1.17279 |


No safeguard activation or retry occurred anywhere in this confirmation comparison: **True**. Thus an active limiter is not the explanation for its observed model-to-model accuracy differences. Contact cases have particularly unfavorable new-model ratios; the per-case files retain the full family breakdown. Short-horizon fitting or rollout-distribution shift are possible mechanisms, not isolated causal findings.

N=512 was absent from the entire training/validation grid set. Canonical Sod/Lax conditions themselves were previously examined, so this is a grid-transfer test, not a fresh-physics holdout:

| Case | Variant | N=512 L1 | Ratio to same-solver classical | Ratio to old V2, same solver |
| --- | --- | --- | --- | --- |
| sod | historical_teno | 0.0010917 | 1.06348 | 1.16452 |
| sod | historical_v2_soft25 | 0.0010045 | 0.97854 | 1.07151 |
| sod | safe_fv_teno | 0.00102653 | 1 | 1.09501 |
| sod | safe_fv_v2_soft25 | 0.000937468 | 0.913236 | 1 |
| sod | safe_fv_new_seed4101 | 0.00101031 | 0.984196 | 1.0777 |
| sod | safe_fv_new_seed4102 | 0.00104659 | 1.01953 | 1.1164 |
| lax | historical_teno | 0.00595203 | 1.0012 | 1.07422 |
| lax | historical_v2_soft25 | 0.005558 | 0.934915 | 1.00311 |
| lax | safe_fv_teno | 0.00594492 | 1 | 1.07294 |
| lax | safe_fv_v2_soft25 | 0.00554078 | 0.932019 | 1 |
| lax | safe_fv_new_seed4101 | 0.0062209 | 1.04642 | 1.12275 |
| lax | safe_fv_new_seed4102 | 0.00647432 | 1.08905 | 1.16849 |


## Reference uncertainty and smooth accuracy

Sod/Lax/Riemann references are analytic cell averages. Short-time Shu uses the earlier FV state reference ladder 256/512/1024 plus finest dt/2. Full-time Shu reuses the existing matched canonical cell-average reference ladder 512/1024/2048 plus finest dt/2. Source hashes and the original scheme identity are retained. The archived finest temporal array was not exported in full; its 256-cell projection is used only at N≤256. It is never fabricated or interpolated to a finer target grid.

For numerical references, twice the final spatial plus temporal Cauchy change is an empirical uncertainty proxy, not an error bound. A method ranking is marked unresolved unless the proxy is below one quarter of the error margin. Full-time Shu profiles and numerical-reference errors are reported even when a ranking is unresolved; these values alone do not establish superiority.

Smooth spatial tests use exact coupled/momentum/critical-point cell averages, periodic boundaries, N=40/80/160/320 and exact instantaneous flux divergences. All absolute norms/orders are in `results/spatial.json`. They do not establish fifth-order time integration: SSPRK3 is third order in time. Existing critical-point limitations remain relevant.

## Classifier diagnostics

Training-only matched-state probes, independent of checkpoint selection:

| Seed | N | Initial BCE / accuracy | Warm-up BCE / accuracy | Selected BCE / accuracy |
| --- | --- | --- | --- | --- |
| 4101 | 64 | 0.820162 / 0.819836 | 0.647173 / 0.816444 | 0.732667 / 0.808804 |
| 4101 | 128 | 0.814542 / 0.827538 | 0.639677 / 0.82563 | 0.716367 / 0.822282 |
| 4101 | 256 | 0.810536 / 0.83002 | 0.635462 / 0.830061 | 0.711619 / 0.827825 |
| 4102 | 64 | 0.824065 / 0.823433 | 0.737931 / 0.499041 | 0.801801 / 0.493457 |
| 4102 | 128 | 0.81709 / 0.828582 | 0.737406 / 0.499352 | 0.788178 / 0.495968 |
| 4102 | 256 | 0.813068 / 0.830103 | 0.735637 / 0.499834 | 0.781327 / 0.499006 |


BCE/accuracy measure teacher imitation, not PDE solution quality. Accuracy is mean per-candidate binary keep/discard agreement, not exact whole-mask accuracy. Numerical validation remains the selection criterion.

## Scalar transfer tests

The new networks were trained on Euler data. The following are explicitly **zero-shot Euler-to-scalar transfer** tests through the unchanged scalar solver, with classical analytic references and matched settings from the archived scalar controls. They are not presented as newly scalar-trained models or as evidence of post-interaction Burgers accuracy.

| Problem | N | Seed 4101 / classical L1 | Seed 4102 / classical L1 |
| --- | --- | --- | --- |
| burgers_hat | 64 | 0.920609 | 0.930501 |
| burgers_hat | 128 | 0.952801 | 0.956017 |
| burgers_hat | 256 | 0.937133 | 0.980036 |
| burgers_smooth | 64 | 1 | 1 |
| burgers_smooth | 128 | 1 | 1 |
| burgers_smooth | 256 | 1 | 1 |
| fourier | 64 | 1 | 1 |
| fourier | 128 | 1 | 1 |
| fourier | 256 | 1 | 1 |
| gaussian | 64 | 0.999977 | 0.999985 |
| gaussian | 128 | 1 | 1 |
| gaussian | 256 | 1 | 1 |
| sine | 64 | 1 | 1 |
| sine | 128 | 1 | 1 |
| sine | 256 | 1 | 1 |
| square | 64 | 1.00495 | 0.985547 |
| square | 128 | 1.01051 | 0.994445 |
| square | 256 | 1.05456 | 1.02921 |


Advection includes sine, Fourier, Gaussian and square profiles; Burgers includes a pre-shock smooth wave and a moving top-hat entropy shock before wave interaction. L1/L2/Linf, TV, extrema and conservative residuals are saved. A ratio above one is a regression, including for tiny smooth errors.

## Isolated CPU cost

After all other experiment workers finish: nine synchronized production repeats, separate compilation/first execution, N=64/128/256, Sod t=.15. The same existing timing implementation is reused with the new frozen models.

| Variant, N=128 | Warmed simulation seconds | Compilation seconds | L1 error |
| --- | --- | --- | --- |
| historical_teno | 0.139078 | 1.44815 | 0.00421063 |
| safe_fv_teno | 0.145008 | 2.0272 | 0.00391322 |
| safe_fv_v2_soft25 | 0.356128 | 2.60782 | 0.0036237 |
| safe_fv_new_seed4101 | 0.361856 | 2.67587 | 0.00370848 |
| safe_fv_new_seed4102 | 0.352703 | 2.78611 | 0.00368875 |


`results/performance.json` also contains measured discrete-grid equal-budget comparisons, per-step costs, executable temporary-memory estimates and lifetime process RSS. Lower same-grid error does not imply better accuracy per unit time. Process RSS is not per-model allocation; compiler estimates are not GPU peak measurements. No GPU claim is made.

## Tests, artifacts and limits

The full CPU suite passes **136 tests and 37 subtests**; the separate legacy environment passes **2 tests**. Four new tests cover full-safeguard gradients and conservation, zero-strength identity, rejection of invalid batched updates, and checkpoint compatibility/determinism. The original experiment's artifact verifier passes before and after training. Its seven already-missing ignored historical checkpoints remain an explicitly recorded baseline limitation.

The original method, original checkpoints and earlier results remain intact. Reproduction and inference instructions are in `../README.md`; exact configuration and candidate hashes are under `../configs/`. Failed/unfavorable runs are retained. No Git push is performed.

Remaining limits: two training seeds, 300-update short-horizon supervision, Euler Riemann-style training data, a small eight-IVP fresh holdout, numerical-reference uncertainty for Shu–Osher, no blast accuracy reference, and no scalar-specific retraining. The global safeguard can be dissipative. Any preferred-model decision must consider problem-wise regressions and CPU cost, rather than validation mean alone.
