# Targeted scientific revisions — Sebastian interview

Revised 30 October 2026 from the existing nine-slide main PDF and its editable source. This is an incremental presentation revision. Original presentations, sources, notes, figures and research artifacts remain unchanged; 154 original/protected files were hashed before work. The first slide, titles, visual identity, nine-slide sequence and three separate technical backups are retained. No CFD solver, model, checkpoint or experiment record was modified, and no training or CFD experiment was run.

## Slide-by-slide changes

| Slide | Targeted modification | Scientific/design reason |
|---|---|---|
| 1 | No change. Reuse the original vector landing page in PDF and original editable landing objects in PPTX. | Preserve V4's aircraft image and composition exactly. |
| 2 | Replace the general scope list with five compact descriptions of framework, convergence/benchmark, characteristic Euler, neural/JAX/blending and diagnostic work. Name the original WENO-NN, NN-TENO/oracle, historical JAX and later V2 generations. | Make the technical contribution visible without attributing every later validation result to a single network or internship phase. Retain the ENO-baseline versus WENO/TENO-order distinction. |
| 3 | Outline the TENO/NN reconstruction box and label its learned decision; retain the six-step pipeline and all formulas. | Identify exactly where learning affects the numerical method. The default path still reconstructs characteristic split fluxes, rather than physical left/right Euler states. |
| 4 | Add the boxed first-order pressure variation and admissible-reference annotation. Keep both pressure/admissibility equations and all three failure/control rows. Add a compact causal progression and retain the empirical-safeguard versus unimplemented-limiter distinction. | Explain how coupled conserved-variable errors affect internal energy. A pressure-sensitivity identity supplies a mechanism, not a unique causal diagnosis of the recorded failure. |
| 5 | Keep the original LF trace plot. State that the multiplier increased at fixed timestep; label the scale CFL_LF. Explicitly state the three simultaneous confounds and name matched CFL, classical TENO5, first-invalid-stage and condition-number diagnostics. | Earlier divergence is a real observation, but it does not establish that dissipation is inherently destabilizing or that the Roe basis uniquely caused the failure. |
| 6 | Label the quoted values “One repaired stage · final RK-weighted source.” Keep the two flow diagrams, explain shared-face balance and explicitly state that positivity limiting was not implemented or validated here. | Resolve the raw-stage versus accumulated-budget ambiguity using the diagnostic implementation, not only the audit's shortened wording. |
| 7 | Enlarge the resolved-ratio figure labels, move blast failures to a large native red callout, include the successful classical blast controls, and replace the full-profile density inset with an explicitly labeled shock-tail zoom from the same arrays. Omit the unresolved Shu–Osher point/ratio entirely. Add the short-horizon/OOD takeaway. | Make the strongest robustness failure readable, retain favorable Sod/Riemann and unfavorable contact results, expose the actual small range excursion, and avoid visually ranking an unresolved reference. |
| 8 | Retain the original convergence figure and both measured orders; qualify the label as the historical FV path. Expand the notes' FV/FD and spatial-versus-evolution distinction. | This observed consistency defect is shared by classical TENO5 and concerns this averaged-data formulation. It is not a theorem that every LF split-flux WENO method is second order. |
| 9 | Preserve the three-column structure and both equations. Distinguish observed limitations, established techniques and open research. Explicitly state that arbitrary theta and a residual alone do not certify positivity/error. Begin the proposed extension with checkable smooth-state/moment/stability assumptions. | Connect to reliable methods without claiming a completed positivity-limited neural Euler method or extending the RKDG–SIAC preprint beyond its assumptions. |

All unchanged headers, footers, logos, title positions, font family, grayscale panel palette and main-slide numbering are retained. The only central diagram styling change is the reconstruction-block outline on slide 3. Slide 4 uses its existing lower content space for the new mathematical explanation; slide 7 changes figure emphasis within its existing two-column structure. Backups remain outside the timed nine-slide talk.

## Pressure linearization: what it establishes

For an admissible reference state with rho>0 and gamma>1,

\[
p=(\gamma-1)\left(E-\frac{m^2}{2\rho}\right),\qquad v=\frac m\rho.
\]

The derivatives with respect to conserved variables are

\[
\frac{\partial p}{\partial E}=\gamma-1,\qquad
\frac{\partial p}{\partial m}=-(\gamma-1)v,\qquad
\frac{\partial p}{\partial\rho}=\frac12(\gamma-1)v^2.
\]

Therefore the first-order variation is

\[
\delta p\approx(\gamma-1)\left(\delta E-v\,\delta m+\frac12v^2\delta\rho\right).
\]

All coefficients are evaluated at the reference state; finite corrections also have higher-order terms. No unique failing neural face decision or component error is inferred from this equation. It explains why a shared-face conservative update can nevertheless leave the admissible set. Nonnegative normalized weights combine characteristic flux-polynomial candidates, not necessarily admissible physical states. SSPRK3 transfers a suitable forward-Euler invariant-set property only under the required spatial and timestep assumptions.

The observations remain the Keras oracle's Sod density exception at step 6 and the separate later V2 first-invalid pressure trace at step 13, stage 1. Soft/.25 passes the same short V2 case. The new mathematical explanation creates no additional experimental result.

## Repair measurement: precise resolution

The definitive row is [repair.json](../docs/research_release/results/repair/repair.json), case `euler-validation-000088-1023461505243801251`, N=128, `mode=soft`, `strength=.75`, `repair_policy=historical-repair`:

| Diagnostic | Exact retained value |
|---|---:|
| `repaired_stage_rows` | 1 |
| `integrated_repair_exchange[0]` — mass | +0.0016111493654163188 |
| `integrated_repair_exchange[1]` — momentum | −0.00005706669322759772 |
| `integrated_repair_exchange[2]` — energy | +0.0008888429518660344 |

[diagnostics.py](../src/differentiable/diagnostics.py):66–81 forms each unweighted repair increment as the sum of the after-minus-before state. Lines 98–101 combine the increments with the appropriate SSPRK3 weights:

\[
B_{\mathrm{repair}}^{\mathrm{step}}
=\Delta x\left(\frac16r_1+\frac23r_2+r_3\right).
\]

Lines 107–114 sum these exchanges over the rollout, and lines 129–133 export `integrated_repair_exchange`, the repair-inclusive bookkeeping residual and the physical balance error separately. Thus the displayed numbers are **RK-weighted contributions to the final fifteen-step rollout budget, arising from one repaired stage**. They are **not** the raw unweighted stage additions, density/energy error norms or a generic conservation-error estimate for every trajectory.

The user's conditional alternative wording for a raw stage-only measurement was therefore not used. The revised on-slide wording retains “RK-weighted” and makes “one repaired stage” and “final source” explicit. The unweighted per-intervention Delta M equation is still valid; the notes explain why its direct value differs from a final RK-weighted contribution.

The physical balance error in this recorded repaired run equals the repair source up to roundoff. Subtracting that source yields a tiny bookkeeping residual. It does not restore conservation of the original source-free Euler equations.

## Figure integrity and uncertain rankings

- The LF plot and nonlinear Euler consistency plot are unchanged. Only their surrounding explanatory text is revised.
- Resolved Sod/contact/random-Riemann ratios still come from [independent/benchmark.json](../docs/research_release/results/independent/benchmark.json), with the same selected V2 soft/.25 and TENO5 rows. No ratio or experimental measurement was changed.
- The blast callout comes from [wall-closure/benchmark.json](../docs/research_release/results/wall-closure/benchmark.json). Both selected-model runs become non-finite; classical TENO5 completes both controls. A failed run is not assigned an accuracy ratio.
- The unresolved Shu–Osher ranking is indicated by neutral text without a plotted point or numerical score. Its archived value is not being promoted as a reliable improvement.
- The inset uses the byte-identical `sod192_selected.npz`, `sod192_teno5.npz` and `sod192_reference.npz`. It zooms to x=.776–.805, rho=.12445–.12615, explicitly just right of the Sod shock. It does not show the whole density range. The selected final density minimum is .1246493145 at x=.7786458333; the original right state is .125. The labeled undershoot is about 3.51e−4 and does not imply negative density.
- Higher TV is not equated to divergence or entropy instability. The full-profile original figure remains available in the original presentation assets.

## Notes, scientific boundaries and timing

[Speaker_Notes_REVISED.md](Speaker_Notes_REVISED.md) updates every main script and adds separate equation guides, established observations, uncertainty/remedy explanations and technical Q&A. Full preparation guidance is also embedded in the revised PowerPoint. Priority questions address conservative negative pressure, conditional SSP, LF confounding, repair sources, shared-face limiting, blast attribution, FV consistency and DG/SIAC/residual analysis.

The main script has 1,473 words, scheduled for **13:15**, with times 0:25, 1:00, 1:35, 2:15, 1:30, 1:25, 2:00, 1:20 and 1:45. Equation guides, technical answers and backups are not part of that timed script. This is a delivery estimate, not an actual human rehearsal measurement.

Positivity-preserving flux limiting, invariant-domain/convex limiting, conditional SSP and moment constraints are established numerical foundations. They are not presented as inventions or successful implementations of this internship. The proposals remain admissibility-constrained learned corrections, suitable DG/SIAC reconstruction and independent a posteriori assessment. A shared-face limiter preserves conservative form; a positivity guarantee additionally needs an admissible low-order update, appropriate simultaneous face/stage factors, boundary treatment and sufficient CFL. A residual/error bound needs the relevant regular reconstruction and stability analysis.

The verified Giesselmann–Kwon–Krumscheid 2026 RKDG–SIAC convection–diffusion preprint is not treated as a theorem certifying the present neural Euler method. The original [Sources_and_Evidence.md](Sources_and_Evidence.md) retains complete bibliographic details and experiment/code/commit provenance. Its numerical sources remain applicable; this document supersedes the older descriptions of the slide-7 plot, inset and script timing. No new academic result or external experimental claim was introduced.

## Editable revised sources

- [Sebastian_Research_Interview_REVISED.tex](Sebastian_Research_Interview_REVISED.tex): editable Beamer/TikZ slide text and equations.
- [build_presentation_REVISED.py](revised_source/build_presentation_REVISED.py): targeted edits to the original native-PPTX/Beamer generator.
- [notes_revisions.py](revised_source/notes_revisions.py) and [write_notes_REVISED.py](revised_source/write_notes_REVISED.py): revised scripts, equation guides and Q&A.
- `revised_source/assets/`: isolated copies of original assets/data plus derived revised figures and equation assets. No original figure or data file was overwritten.
- [finalize_REVISED.py](revised_source/finalize_REVISED.py) and [validate_REVISED.py](revised_source/validate_REVISED.py): cover preservation, render comparison and evidence validation.

PowerPoint text/diagrams remain native editable objects. Equations remain vector SVGs with PNG fallbacks and editable TeX sources, not native Office equation objects.
