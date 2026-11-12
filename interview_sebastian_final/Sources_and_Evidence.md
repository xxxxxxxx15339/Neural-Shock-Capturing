# Sources and evidence — KIT research interview

Prepared from repository HEAD `f9d4c2a`, 29 October 2026. The original V4 PDF, source PowerPoint, research code, model parameters, datasets and historical measurements are unchanged. No training was run to make this deck. The broader investigation is [the research audit](../EULER_STABILITY_RESEARCH_AUDIT.md).

The nine main slides are a 13:15 talk; three technical backups are not included in that timing. Four research generations are explicitly distinguished: original WENO coefficient corrections; Keras neural TENO candidate selection; historical JAX rollout training/blends; later V2 gated NN-TENO5. A result from one generation is never attributed to another backbone.

## Evidence levels

- **Archived measurement:** retained CSV/JSON/NPZ, log or figure; establishes the recorded observation, subject to that run's configuration/provenance.
- **Audit reproduction:** a lightweight test conducted in the preceding repository audit; its complete recipe is in audit §10.4. This deck does not claim a new training experiment.
- **Historical account:** committed phase narrative/commit, without complete surviving raw run logs. Reported improvements are qualified accordingly.
- **Implementation fact:** source code establishes the stated behavior, but does not alone establish that it caused a particular failure.
- **Established theory:** primary numerical-analysis literature; its hypotheses must apply before a guarantee is inherited.
- **Proposed extension:** not implemented or not validated for this learned solver.

## Slide 1 — unchanged original landing

Source: [V4.pdf](../V4.pdf), page 1, 960.009×540 PDF points. The final PDF uses the extracted original vector page. The source copy is `assets/V4_landing-1.pdf`. The editable PowerPoint source is [Scientific_Internship_WENO_NN_UM6P_Professional_LaTeX.pptx](../Scientific_Internship_WENO_NN_UM6P_Professional_LaTeX.pptx); its first-slide original shapes, image and equation objects are preserved. An input hash manifest is in `validation/input_hashes.json`.

The schlieren aircraft photograph is V4's original illustration, not a flow measurement or simulation from this project. The author/supervisor, main image, logo, text, positions, colors and composition are not redesigned. PDF render comparison results are recorded in Presentation_Validation.md.

## Slide 2 — background, scope and generation identity

Implementation sources:

- [src/schemes/](../src/schemes/): WENO3/5/7, TENO3/5/7, ENO baseline, scalar/Euler wrappers. `ENO3.py:5–25` selects between two linear candidates, so the slide deliberately says **ENO baseline; WENO/TENO orders 3/5/7**, without claiming uniform verified third-order ENO.
- [WenoNetworks.py](../src/networks/WenoNetworks.py):21–49 gives coefficient corrections and coefficient-sum consistency; [TenoNetworks.py](../src/networks/TenoNetworks.py):22–75 gives candidate classifiers.
- [scripts/legacy_imitation_training/](../scripts/legacy_imitation_training/) includes the original scalar, Sod and Shu–Osher comparisons and trainers.
- [scripts/unrolled_physics_loss/](../scripts/unrolled_physics_loss/) includes historical differentiable training/curricula/blends and V2 training.
- [Documentation/Results.md](../Documentation/Results.md), phases 1–21; [Documentation/Report_Results.md](../Documentation/Report_Results.md), parts 1–4; [Documentation/P0_Validation.md](../Documentation/P0_Validation.md), generation/metric distinctions and later controlled comparisons.

The timeline is a method history, not a single network's chronology of wins. Neither the order labels nor the internship photograph is experimental proof of the final solver's accuracy/reliability.

## Slide 3 — actual Euler pipeline

- [eulerEquations.py](../src/core/eulerEquations.py):16–45 common Roe face eigenbasis, 55–77 physical Euler flux/speeds and ideal-gas pressure, 79–122 conservative characteristic LF split-flux path.
- [euler_jax.py](../src/differentiable/euler_jax.py):36–96 flux/EOS/basis, 166–204 characteristic LF input stencils, 207–254 shared numerical face flux and boundary budget.
- [numerics_jax.py](../src/differentiable/numerics_jax.py):92–115 SSPRK3; guards are optional and applied per substage.
- [gated_teno5_jax.py](../src/differentiable/gated_teno5_jax.py):19–67 implements `R_T+G*strength*(R_NN−R_T)`; strength zero returns the classical flux exactly.
- [reconstruction.py](../src/differentiable/reconstruction.py):22–74 makes historical/V2 and soft/hard policies distinct.

The on-slide finite-volume equation is the **forward-Euler building block**. SSPRK3's full three-stage composition is in Backup A. The default reconstruction operates on positive/negative characteristic split **fluxes**. The alternative physical-state/Rusanov formulation is named experimental and discussed separately on slide 8.

The current projection matrices are inverses by construction and passed moderate-state inverse/kernel regressions in the audit. No failing-trajectory condition-number time series establishes the basis as the first cause of instability. No quantified neural component-wise-versus-characteristic comparison was found.

## Slide 4 — actual density/pressure failures

### Oracle Keras NN-TENO5: Sod step 6

- Checkpoint: [models/nn_teno5_oracle.h5](../models/nn_teno5_oracle.h5).
- Calibrated thresholds and generation identity: [nn_teno5_oracle_metadata.json](../models/nn_teno5_oracle_metadata.json):7–12,21–43,60–114.
- Configuration: N=120, L=1, T=.2, adaptive CFL=.35; `NN_TENO5_Euler` with those metadata thresholds, initial condition `sod()`.
- Exact observed exception: `FloatingPointError: Non-positive Euler density at step 6.` Reproduced in the audit; recipe in [audit §10.4](../EULER_STABILITY_RESEARCH_AUDIT.md).
- Historical experiment: [Documentation/Results.md](../Documentation/Results.md):407–449, oracle instability and later anchoring attempts; commits `ce91a07`, `109326a`, with oracle artifacts eventually committed in `df4986c`.

The oracle optimizes local reconstruction error on synthetic functions. That objective does not enforce physical Euler stages or repeated nonlinear stability. A uniquely identified first erroneous face/mask has not been established. Anchoring reportedly kept tested runs finite but did not improve all benchmark criteria; do not call it a positivity theorem.

### Later V2: first-invalid pressure trace

- Checkpoint: [models/v2/euler_gated_dataset_v2_seed2026.pkl](../models/v2/euler_gated_dataset_v2_seed2026.pkl).
- SHA-256: `dbae82f977bc5b07b0fb7671971696e69f44541dad450474fc575813f22bb7b5`.
- IVP: `euler-validation-000088-1023461505243801251`, random Riemann, N=128, transmissive boundary, dt=.0017428079661349484, 15 steps, final time .026142119492024225.
- Initial/target data: [data/nnteno_v2/data/euler_validation.h5](../data/nnteno_v2/data/euler_validation.h5), group N128, case_id as above, `states[index,0]` and `short_target[index]`.
- Audit's separate raw-stage trace: soft/.75 first invalid at one-based **step 13, stage 1**; minimum pressure −.0038048537160435105, minimum density ≈1.21304. Soft/.25 completes all 15 steps admissibly. Recipe is in audit §10.4.
- Archived validation: [failure_inventory.json](../docs/research_release/results/full-validation/failure_inventory.json), same IVP/mode/strength. Its fused batched trace has p_min≈−.0570911 and 8 invalid stage rows. The direct repair study's unguarded continuation becomes non-finite. These differing continuations are explicitly retained; the deck uses the separate **first-invalid audit trace**, not an invented common trajectory.

Pressure formula is consistent in the inspected physical solver: `p=(γ−1)(E−m²/(2ρ))`, γ=1.4. Positivity, conservation and entropy stability are not equivalent. The known conservative positivity-limiter remedy is **not yet implemented/validated here**; references R1–R3 below supply the numerical foundation under their assumptions.

## Slide 5 — historical alpha multiplier trace

Data: [models/euler_instability_trace.csv](../models/euler_instability_trace.csv), lines 1–41; a byte copy is supplied in assets. The original [plots/04_euler_instability.png](../plots/04_euler_instability.png) is preserved as `assets/04_euler_instability_original.png`. The displayed chart is re-created from the unchanged CSV using `build_presentation.py:charts`; the old causal title is removed.

| Multiplier | Final relevant stored observation, zero-based index |
|---|---|
| 1 | Density ≈1.94e37 at index 10, non-finite at 11. |
| 1.5 | Final stored index 14 has density ≈6.21e18 and remains finite. It is not a stable result. |
| 2,3,5 | Non-finite at index 3. |

Source: [EulerInstability_TraceCapture.py](../scripts/unrolled_physics_loss/EulerInstability_TraceCapture.py):53–114. The old clamp-only guard is frozen in that script; alpha multiplier changes, while dt and dx do not. The trainer's fixed assumed-speed setup is [Train_NNTENO5euler_UnrolledJAX.py](../scripts/unrolled_physics_loss/Train_NNTENO5euler_UnrolledJAX.py):62–69. The exact historical sampler also needs its corresponding revision for strict reproduction; the current imported sampler changed later.

The LF split and `alpha*dt/dx` scale show a confound. The multiplier changes the explicit stability restriction **and** the neural split-flux input. The saved CSV contains no pressure minima, raw stage states, actual CFL, or condition-number trace. Thus a unique Roe-conditioning cause was suspected but not demonstrated. A cross on the plot marks the next non-finite index at the preceding finite height, not a measured value for NaN.

## Slide 6 — repair budget

- [results/repair/repair.json](../docs/research_release/results/repair/repair.json): match the N=128 case above, `mode='soft'`, `strength=.75`, `repair_policy='historical-repair'`.
- `repaired_stage_rows=1`; final integrated RK-weighted repair exchange is **[.0016111493654163188, −.00005706669322759772, .0008888429518660344]**, representing mass, momentum, energy.
- These are final budget contributions, not the raw unweighted per-stage source. The displayed ΔM equation defines the per-intervention diagnostic; its SSPRK3 composition is explained in the notes. Exact final repair weights: (1/6,2/3,1); boundary-flux weights: (1/6,1/6,2/3).
- [diagnostics.py](../src/differentiable/diagnostics.py):83–133 implements those weights and separates physical balance error from repair-inclusive residual.
- [euler_jax.py](../src/differentiable/euler_jax.py):357–466 implements neighbor replacement, residual floors/caps, periodic neighbor choice and diagnostics. It is not conservative or generally boundary-aware.
- [numerics_jax.py](../src/differentiable/numerics_jax.py):179–233 fixes the historical post-repair penalty issue by accumulating pre-repair penalties at all stages.
- [Documentation/Results.md](../Documentation/Results.md):503–541 records the once-inert penalty and subsequent training completion; some training-specific outcomes remain historical accounts with incomplete per-update logs.
- [Documentation/P0_Validation.md](../Documentation/P0_Validation.md):122–157 and [repair_assessment.json](../src/differentiable/validation/p0-2026-10-14/repair-assessment/repair_assessment.json) establish the separate earlier low-positive-state repair defect and its correction.

The right-hand flowchart is an established **unimplemented/unvalidated neural extension**, not a successful project experiment. Shared-face flux limiting or conservative stage recomputation addresses the conservation issue at a different point in the update. R1/R5 below are the foundations, with admissibility/CFL/limiter assumptions.

## Slide 7 — V2 validation and prospective transfer

### Validation

[full-validation/summary.json](../docs/research_release/results/full-validation/summary.json), row `(physics=euler, mode=soft, strength=.25)`:

- 307 validation trajectories, 15 SSPRK3 steps each; zero failed trajectories by the raw-stage validity criterion.
- Macro family×resolution relative L1 = **.9479383302719695**, shown as .94794.
- This validation set selected the correction strength. It is not an independent universal generalization score or a mathematical guarantee. Constants/tiny-error groups are excluded from ratios as declared; absolute errors remain available. The two grids of one prospective IVP are not independent trajectories.

### Prospective N=192 Sod and contact

Data: [independent/benchmark.json](../docs/research_release/results/independent/benchmark.json); select `v2-selected-soft` and `teno5`, matching case and cells.

- Sod: TENO5 L1=.002651360802799551; selected=.0025380153091719496. Relative error reduction ≈4.275%, rounded to **4.3%**.
- Density TV .8774663254486642→.8797561899875133; density undershoot 0→.0003506854567733031, rounded **3.51×10⁻⁴**.
- Contact: TENO5 L1=.0022149671223096416; selected=.0023672257990848457, ratio≈1.06874, about **6.9% worse**.
- The figure's four random-Riemann entries and Shu entry are taken directly from the same JSON. An asterisk identifies the unresolved numerical-reference Shu ranking; it is not colored as a certified improvement.

The original [final_benchmark.png](../docs/portfolio/figures/final_benchmark.png) and [euler_sod.png](../docs/portfolio/figures/euler_sod.png) are preserved in assets. The main slide's readable Euler-only excerpt uses the same JSON values. The Sod density inset uses the exact N=192 `state` arrays in the supplied selected/classical/reference NPZ copies. The original full plots remain available for inspection; no measurement was altered to improve the visual narrative.

### Blast failure

[wall-closure/benchmark.json](../docs/research_release/results/wall-closure/benchmark.json), `case=wall-closed-woodward-colella`, N=128/256, T=.038, `boundary=reflective-wall`, nominal CFL=.15, initial-speed factor 2:

- Selected V2 soft/.25: non-finite; classical TENO5: finite and admissible at both grids.
- No full-time accuracy reference after the failed short smoke; no error ratio assigned.
- [wall-smoke/benchmark.json](../docs/research_release/results/wall-smoke/benchmark.json): selected model already fails at N=128, T=.001.
- Selected/trained V2 Euler rows are identical policies here, not independent replication.

### Candidate transfer mechanisms and measured mismatch

- [data/nnteno_v2/README.md](../data/nnteno_v2/README.md):21–62 lists the exact analytic corpus and known absences: reflective boundaries, Shu–Osher interactions, generic interacting dynamics. Stronger blast contrasts and long evolution are outside this coverage.
- [Train_Gated_NNTENO5_Dataset_V2.py](../scripts/unrolled_physics_loss/Train_Gated_NNTENO5_Dataset_V2.py):183–234 gives guarded training/unrepaired evaluation and 15-step objective; 238–266 shows characteristic-state warm-up versus optional split-flux input variant.
- [input-mismatch-r2/input_mismatch.json](../docs/research_release/results/input-mismatch-r2/input_mismatch.json): measured normalized input mismatch; this is not a controlled proof that changing warm-up improves PDE accuracy.
- [reconstruction.py](../src/differentiable/reconstruction.py):45–51: soft and hard policies are distinct. At strength .25 their Euler validation macro ratios differ (.94794 soft versus ≈1.02322 hard).

These are distribution/implementation facts and candidate explanations, not isolated causes of the blast failure. Increasing data volume alone supplies no structural theorem.

## Slide 8 — nonlinear Euler consistency and other limitations

Data: [nonlinear_euler.json](../docs/research_release/results/nonlinear-euler/nonlinear_euler.json). The original [nonlinear_euler_consistency.png](../docs/portfolio/figures/nonlinear_euler_consistency.png) is preserved in assets. The displayed chart uses the same classical-TENO5 rows at N=40/80/160/320/640, with larger readable labels.

- Historical split-flux path, classical TENO5: last L1 order **2.0119685376282312**.
- Experimental characteristic-state/Rusanov path: last L1 order **4.999925297394422**.
- Definition: semidiscrete divergence compared to exact finite-volume cell-average physical flux divergence; no temporal truncation error.
- [v2_nonlinear.py](../src/differentiable/v2_nonlinear.py):11–38: smooth varying rho, velocity and pressure, 16-point quadrature and physical flux at faces.
- [euler_jax.py](../src/differentiable/euler_jax.py):166–200 supplies `F(Ubar)` to the split-flux reconstruction; 259–280 is the optional periodic-only state-flux formulation.

The leading nonlinear averaging mismatch is O(dx²) for smooth data. The linear-in-density constant-pressure/velocity entropy-wave subfamily is insufficient to test this general limitation. Proper finite-difference flux reconstruction is a different data interpretation requiring its own derivation. Experimental state-flux **smooth consistency** does not demonstrate shock robustness, positivity or successful neural retraining.

Brief additional issues moved to backup/notes:

- WENO7 coefficient defect: original epsilon 1e-6 yields 10/100,000 non-finite coefficient rows, versus zero at 1e-4 in the same audit forward test. Current beta polynomial and fix: [WenoNetworks.py](../src/networks/WenoNetworks.py):106–137, commit `753d526`. Exact reproduction recipe and indices in audit §10.4.
- `epsilon²=1e-12` is not float32 underflow by itself. Negative cancellation-sensitive computed beta was observed in an instrumented row; the exact original full-graph NaN operation is unresolved. Larger epsilon is a sampled mitigation, not an order/stability proof.
- Neural mirror-only wall boundary lacked the physical zero mass/energy flux contract; `reflective-wall` explicitly corrects it ([euler_jax.py](../src/differentiable/euler_jax.py):220–237). Blast failure persists after this correction. Do not conflate boundary correctness with full-flow robustness.

## Slide 9 and Backup C — structural remedies and research connection

The face-flux limiter formula is a **proposed neural integration of established methods**, not an implemented project result. Choosing theta in [0,1] alone does not prove positivity; an admissible low-order update, a sufficient timestep restriction and coupled face/stage/boundary constraints are needed. Polynomial reproduction/moment constraints address a different accuracy property. The independent reconstruction residual must be combined with a valid stability/error analysis; a residual sensor or network confidence does not certify error by itself.

The main-slide residual is the inviscid Euler form. A convection–diffusion residual additionally contains the diffusion operator. The 2026 preprint's actual model equations and assumptions must not be silently replaced by unrestricted compressible Euler shocks.

### Verified primary references

**R1. Xiangyu Y. Hu, Nikolaus A. Adams, Chi-Wang Shu (2013).** *Positivity-preserving method for high-order conservative schemes solving compressible Euler equations.* Journal of Computational Physics **242**, 169–180. DOI [10.1016/j.jcp.2013.01.024](https://doi.org/10.1016/j.jcp.2013.01.024). Metadata/abstract verified against the [authors' TUM institutional record](https://portal.fis.tum.de/en/publications/positivity-preserving-method-for-high-order-conservative-schemes-/) and the [publisher record](https://www.sciencedirect.com/science/article/pii/S0021999113000557). Flux limiting toward first-order LF supplies the relevant structural foundation, with timestep/accuracy qualifications. This was not tried and failed in the present neural method.

**R2. Xiangxiong Zhang, Chi-Wang Shu (2012).** *Positivity-preserving high order finite difference WENO schemes for compressible Euler equations.* Journal of Computational Physics **231**(5), 2245–2258. DOI [10.1016/j.jcp.2011.11.020](https://doi.org/10.1016/j.jcp.2011.11.020). Verified against the [author's publication list, item 10](https://www.math.purdue.edu/~zhan1966/research/index.html) and [publisher metadata](https://www.sciencedirect.com/science/article/pii/S0021999111006759). Its finite-difference positivity construction is not automatically inherited by the repository's arbitrary neural split fluxes.

**R3. Sigal Gottlieb, Chi-Wang Shu, Eitan Tadmor (2001).** *Strong Stability-Preserving High-Order Time Discretization Methods.* SIAM Review **43**(1), 89–112. DOI [10.1137/S003614450036757X](https://epubs.siam.org/doi/10.1137/S003614450036757X). Verified directly from the SIAM publisher record. SSP transfers a suitable forward-Euler property under its step restriction; it does not create an absent spatial invariant-domain property.

**R4. Jan Giesselmann, Kiwoong Kwon, Sebastian Krumscheid (2026).** *A Posteriori Error Analysis of Runge–Kutta Discontinuous Galerkin Schemes with SIAC Post-Processing for Nonlinear Convection–Diffusion Systems.* [arXiv:2604.01200v1](https://arxiv.org/html/2604.01200v1), submitted 1 April 2026. The primary text was inspected, including its reconstruction, entropy-compatible setting, model stability results and nonlinear regularity dependence. It uses temporal reconstruction and SIAC filtering in periodic Cartesian RKDG settings. Worked analyses include linear advection–diffusion, nonlinear scalar problems and a diffusive p-system. It contains **no learned filter** and does not certify this neural Euler shock solver.

**R5. Jean-Luc Guermond, Murtazo Nazarov, Bojan Popov, Ignacio Tomas (2018).** *Second-Order Invariant Domain Preserving Approximation of the Euler Equations Using Convex Limiting.* SIAM Journal on Scientific Computing **40**(5), A3211–A3239. DOI [10.1137/17M1149961](https://doi.org/10.1137/17M1149961). Verified against the [author-hosted manuscript](https://people.tamu.edu/~guermond/PUBLICATIONS/guermond_nazarov_popov_tomas_SISC_2018.pdf) and [primary preprint arXiv:1710.00417](https://arxiv.org/abs/1710.00417). This is a reference for invariant-domain/convex limiting foundations, not a claim that the current high-order network already satisfies the theorem.

Verification took place online on 29 October 2026. Some direct DOI redirects/full-PDF fetches returned access errors/timeouts; author/institution/publisher-indexed primary records provided the bibliographic details above. No unverified title/author/DOI was retained. No long passages or paper figures were copied into the deck. The paper summaries are qualified; the research extensions in the talk are the applicant's proposals.

## Backup A — precise formulations

Source references are slide 3's implementation files. Characteristic eigenvalues are v−c,v,v+c, c=sqrt(γp/ρ). D(U)=(Fhat_right−Fhat_left)/dx. The displayed three SSPRK3 stages have the correct signs and coefficients. State constraints are ρ>0 and E−m²/(2ρ)>0. R3 supplies the conditional SSP interpretation.

## Backup B — WENO7 forward test

The WENO-JS α/ω formula uses the existing positive linear weights and squared regularized betas. The numerical count is a forward coefficient-layer audit on the stored normalized dataset, not a new 100k-training experiment. Sampling/graph/precision differences can affect exact bad-row counts. The larger epsilon fixed the sampled NaNs but did not establish seventh-order neural accuracy or global Euler robustness. The Taylor/conditioning discussion is a numerical hypothesis, not a fabricated experiment.

## Assets, metrics and source reproducibility

The editable `build_presentation.py` generates native DrawingML text/diagrams plus SVG/PNG equations from `assets/math/equations.tex`, and a real Beamer/TikZ source with editable mathematical expressions. `write_notes.py` generates `Speaker_Notes.md`, `notes_content.json` and the notes embedded in PowerPoint. `assets/ooxml_support.py` is a copied local rendering utility; no original repository script was edited.

Original figure copies remain unmodified; newly plotted figures have distinct descriptive names. Charts use only CSV/JSON/NPZ measurements supplied in assets. The first PDF page is reused without rasterizing it in the final PDF. The slide scripts and layout JSON specify all later objects. Text and diagrams are native editable PPTX shapes. Equations are vector objects with TeX sources, **not native editable Office equation objects**. Numerical curves are figure assets, with their data and regeneration code supplied; the PPTX is not a stack of whole-slide images.

The final validation records exact page counts, landing-page image comparison, numerical assertions, source/PPTX object counts, equation sizes, rendering inspections and remaining practical limits.
