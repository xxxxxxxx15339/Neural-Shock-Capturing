# Refined presentation — slide-by-slide review

[PowerPoint](../../Scientific_Internship_WENO_NN_UM6P_Refined.pptx) · [PDF](../../Scientific_Internship_WENO_NN_UM6P_Refined.pdf)

The supplied 12-slide presentation has been refined throughout, with one additional mathematical backup slide. The original PDF is unchanged. The deck uses a consistent navy, teal and UM6P orange palette, flat panels, aligned margins, and a clear hierarchy of titles, explanations and formulas.

Click a preview to review an individual slide. Further revisions can refer to its slide number.

| Slide | Preview | Main refinement |
| --- | --- | --- |
| 1 | [Cover](slide-01.png) | Removed “Scientific Internship Soutenance”; rebuilt the title, author and supervisor block; retained the original photograph and UM6P logo. |
| 2 | [Physical motivation](slide-02.png) | Balanced the photograph and explanation; added the conservation law and advection/Burgers flux formulas. |
| 3 | [Numerical solver](slide-03.png) | Added cell averages, conservative flux divergence, Lax–Friedrichs splitting, all three SSPRK3 stages and the CFL condition. |
| 4 | [Classical reconstruction](slide-04.png) | Aligned the five-point stencil, three substencils and ENO/WENO/TENO panels; added reconstruction and weight formulas. |
| 5 | [Neural network](slide-05.png) | Corrected the wrapped layer counts; added guarded normalization and soft/hard candidate decisions. |
| 6 | [Learning methodology](slide-06.png) | Separated imitation and physics training; added BCE, K-step evolution, reference error and the complete weighted loss. |
| 7 | [Stable Euler training](slide-07.png) | Added convex mixing, pressure/density admissibility and the actual instability trace; clarified the role of repair and training constraints. |
| 8 | [Scalar results](slide-08.png) | Preserved the earlier WENO-NN plots and distinguished them from final NN-TENO5 results; added exact error values and the L₂ metric. |
| 9 | [Euler results](slide-09.png) | Aligned the Sod and Shu–Osher figures; stated the measured gains, references, grid size and metric. |
| 10 | [Conclusion](slide-10.png) | Replaced crowded sections with one takeaway and three balanced columns for contributions, evidence and future work. |
| 11 | [Physics loss formulas](slide-11.png) | Added the MSE, normalized TV, neighboring-cell BVD and normalized anchor regularization definitions. |
| 12 | [Complete workflow](slide-12.png) | Rebuilt the workflow with editable components; added Roe projection and the gradient update; clarified the distinct scalar/Euler flux paths. |
| 13 | [Reconstruction formulas](slide-13.png) | Added the three candidate polynomials, TENO smoothness sensor, implementation constants and all-rejected fallback. |

## Mathematical and numerical sources

Equations were checked against the project implementation:

- [Scalar solver and SSPRK3](../../src/differentiable/numerics_jax.py)
- [TENO5 candidates and smoothness sensor](../../src/differentiable/teno5_jax.py)
- [Neural projection and convex mixing](../../src/differentiable/projection_jax.py)
- [Unrolled physics loss](../../src/differentiable/losses_jax.py)
- [Euler characteristic fluxes and positivity repair](../../src/differentiable/euler_jax.py)

Final Euler percentages are computed from the saved [Phase 18 benchmark metrics](../../models/nn_teno5euler_jax_comparison_phase18.json). Final scalar values come from [Phase 19 in the experiment log](../Results.md). These use different metrics: scalar L₂ and the mean Euler L₁ over density, momentum and energy. The older WENO-NN figures on slide 8 are labeled separately.

The TV loss and hybrid mixing are not mathematical guarantees of TVD or positivity; the slides reflect that distinction. The positivity loss is evaluated before repair.

## Further editing

Text, panels and diagrams are editable PowerPoint components. Equations include SVG vector artwork and high-resolution PNG fallbacks for compatibility; their LaTeX is recorded in [equations.json](equations.json).

[refine_slides.py](../../scripts/presentation/refine_slides.py) reproduces the deck from its matching original PowerPoint. It writes a separate output and preserves the source.
