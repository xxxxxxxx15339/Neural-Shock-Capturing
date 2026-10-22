# Professional LaTeX edition

[PowerPoint](../../Scientific_Internship_WENO_NN_UM6P_Professional_LaTeX.pptx) · [PDF](../../Scientific_Internship_WENO_NN_UM6P_Professional_LaTeX.pdf) · [Contact sheet](slide-contact.png)

This edition uses Arial for text, a restrained black/white/gray design, and actual LaTeX formulas in Latin Modern. Green marks lower errors and red marks the slightly higher Sod error. The original shock photograph is retained. All four results figures now compare the final saved NN-TENO5 checkpoint directly with TENO5 and the same reference.

The deck contains 14 main slides, one short reference slide, and a final Thank you slide. There is no mathematical backup section. The requested six-block workflow and six-column benchmark table keep their structure. The detailed slides follow the overview's order: physical setup → classical / neural reconstruction → interface flux / SSPRK3 → rollout training. The numerical implementation slide is directly after the final architecture and before the K-step training slide. Its formulas are ordered as Lax–Friedrichs splitting → split-flux recombination → spatial flux balance → the three SSPRK3 stages → K-step rollout. Euler flux matrices, characteristic-speed formulas, gradient-clipping algebra and loss-term derivations remain omitted. The central four-term physics objective uses α₁–α₄, leaving λ exclusively for learned reconstruction mixing. Euler's additional positivity penalty is noted separately.

The story is now explicit: imitation → solver-in-the-loop training → Euler instability → numerical repairs → dedicated learned mixing → benchmark-dependent improvements. The small mixing head takes the classifier's three probabilities, not the raw stencil, and has four parameters. It is shown separately from both reconstruction paths.

| Slide | Preview | Topic |
| --- | --- | --- |
| 1 | [Cover](slide-01.png) | Project title, author and supervisor |
| 2 | [Research question](slide-02.png) | What can learned reconstruction improve over TENO5? |
| 3 | [Numerical workflow](slide-03.png) | Unchanged six-block ordered workflow and conservative flux balance |
| 4 | [Euler system](slide-04.png) | Density, momentum, energy and valid physical states |
| 5 | [Classical reconstruction](slide-05.png) | ENO, WENO and TENO with one weighted-candidate formula |
| 6 | [Final architecture](slide-06.png) | TENO5 and neural paths, dedicated mixing head, blend and solver update |
| 7 | [Fluxes and RK3](slide-07.png) | Ordered splitting, interface flux, flux balance, RK3 stages and rollout |
| 8 | [Physics loss](slide-08.png) | K actual solver steps and the four-term objective, with simple explanations |
| 9 | [Euler diagnosis](slide-09.png) | Historical failure trace and the concrete training repairs |
| 10 | [Mixing lessons](slide-10.png) | Separate decisions, small initial blend, safeguards and limits |
| 11 | [Scalar results](slide-11.png) | Final Advection / Burgers comparisons with separate local-error zooms |
| 12 | [Euler results](slide-12.png) | Final Sod / Shu–Osher comparisons with separate local-error zooms |
| 13 | [Benchmark summary](slide-13.png) | Requested table, with favorable values green and unfavorable values red |
| 14 | [Conclusion](slide-14.png) | Specific contributions, measured evidence and remaining validation |
| 15 | [References](slide-15.png) | Three primary literature references, with clickable DOI / arXiv links |
| 16 | [Thank you](slide-16.png) | Closing slide and questions |

## LaTeX and reproducibility

[equations.tex](equations.tex) contains the actual LaTeX source for the retained formulas. [equations.json](equations.json) maps them to displayed slide numbers and records their effective size. Equations are embedded as SVG vectors, with transparent PNG fallbacks. Text, panels, tables and diagrams remain editable in PowerPoint.

[professional_slides.py](../../scripts/presentation/professional_slides.py) builds this edition using the shared slide generator and [research_refinements.py](../../scripts/presentation/research_refinements.py). Formula sources are compiled before SVG/PNG conversion and assembly. [slide_changes.json](slide_changes.json) records the slide-by-slide edits.

The Euler state and positivity requirements were checked against [euler_jax.py](../../src/differentiable/euler_jax.py). The loss meanings are checked against [losses_jax.py](../../src/differentiable/losses_jax.py), and the training procedure against [the Euler curriculum script](../../scripts/unrolled_physics_loss/Train_NNTENO5euler_UnrolledJAX_Curriculum.py). Experimental interpretation was checked against [Results.md](../Results.md).

## Final comparisons and scope

[build_final_comparisons.py](../../scripts/presentation/build_final_comparisons.py) evaluates the saved scalar Phase 19 and Euler Phase 18 checkpoints without retraining. Run it with the `teno-jax` Python environment. `--plots-only` redraws the already-saved profiles without rerunning the solvers.

[final_comparison_metrics.json](figures/final_comparison_metrics.json) records unrounded errors, configurations, references, checkpoint names and percentage changes. [final_profiles.npz](figures/final_profiles.npz) contains every plotted numerical / reference profile. All four regenerated comparisons match the documented checkpoint results. The table uses unrounded errors to calculate ratios: advection now rounds to R = 0.9653 and −3.47% rather than 0.9652 / −3.48% calculated from previously rounded errors. Its headline gain remains 3.5%.

Scalar comparisons use 64 cells, CFL 0.5 and exact references. Burgers at t = 0.5 is before shock formation; it is not labeled a post-shock result. Euler comparisons use 120 cells and CFL 0.4. Sod uses an exact Riemann reference; Shu–Osher uses a 320-cell TENO5 numerical reference, not an established converged solution. Euler solution overlays show density, while the lower strips show the local mean absolute error across density, momentum and total energy—the same component average used for the reported mean L₁ errors. Scalar strips show signed solution-minus-reference error. Local-error strips do not obscure the full-domain curves.

Both final Euler runs are finite and physically admissible without invoking state repairs. This is distinct from the historical training safeguards, and is not a positivity-preservation theorem. Four cases at fixed grids do not establish convergence, generality, or a controlled runtime tradeoff. Cost profiling and broader validation are stated as future work. Convex mixing / repair are not presented as formal positivity or TVD guarantees.

## Literature

- Stevens & Colonius (2020), *Enhancement of shock-capturing methods via machine learning*. [DOI](https://doi.org/10.1007/s00162-020-00531-1); metadata checked against the paper and [the authors' Caltech publication list](https://colonius.caltech.edu/publications.html?area=cfd).
- Fu, Hu & Adams (2016), *A family of high-order targeted ENO schemes for compressible-fluid simulations*. [DOI](https://doi.org/10.1016/j.jcp.2015.10.037); metadata checked against the accepted manuscript and [HKUST's publication record](https://researchportal.hkust.edu.hk/en/publications/a-family-of-high-order-targeted-eno-schemes-for-compressible-flui/).
- Li, Fu & Adams (2022), *A six-point neuron-based ENO (NENO6) scheme for compressible fluid dynamics*. [Original arXiv record](https://arxiv.org/abs/2207.08500). This related six-point scheme is not identified as the five-point model implemented here.
