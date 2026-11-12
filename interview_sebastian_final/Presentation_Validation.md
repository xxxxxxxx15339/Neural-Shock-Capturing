# Final presentation validation

Completed 29 October 2026 against repository HEAD `f9d4c2a`. **All 71 automated checks passed.** The machine-readable results are in [validation/checks.json](validation/checks.json); the check implementation is [validate_presentation.py](validate_presentation.py). Manual visual/scientific review is described separately below. No training or solver modification was performed to create the presentation.

## Deliverables and presentation timing

| Check | Result |
|---|---|
| Complete final PDF | 12 pages: 9 main slides and 3 technical backups. |
| Main-only PDF | 9 pages; ends at the research question. |
| PowerPoint | 12 slides; slides 10–12 are hidden during the normal slideshow. |
| Source | Real Beamer/TikZ `.tex`, native-OOXML generation code, equation TeX/SVG/PNG assets, plotting code and original numerical records. |
| Speaker notes | Every main slide has a full spoken script, terms, preparation/evidence boundaries and 1–2 technical Q&A. Scripts/Q&A are also embedded in PowerPoint notes. |
| Main speaking time | **795 seconds = 13:15**, excluding backups and Q&A. |
| Main spoken script | **1,586 words**, approximately 120 words/minute with the scheduled pauses. |

Timing is a script-based estimate, not a measured human rehearsal. The speaker should rehearse aloud once. Main-slide durations are 0:25, 1:00, 1:45, 2:00, 1:25, 1:25, 2:00, 1:20 and 1:55.

## Exact preservation of V4's landing page

The first page was inspected before content development. The editable source corresponding to V4 was located and copied into `assets/V4_editable_template.pptx`. The original files were not overwritten.

| Preservation test | Result |
|---|---|
| Original `V4.pdf` SHA-256 | Unchanged: `f137d8701ce2c2d2853e40e279cadcb340460c4d334269ca88f06d76fb342aa6`. |
| Original source PPTX SHA-256 | Unchanged; exact hash retained in `validation/input_hashes.json`. |
| Final PDF first page | Original extracted vector PDF page inserted directly; original 960.009 × 540 point dimensions retained. |
| Final PDF versus V4 page 1 | **0 changed pixels** at 1800 × 1013; mean/max channel error 0. |
| PowerPoint versus V4 page 1 | **0 changed pixels** in the LibreOffice PDF render at 1800 × 1013; mean/max channel error 0. |
| PowerPoint first-slide XML | Byte-identical to the original editable slide XML. |
| Original first-slide image/equation assets | All referenced original media byte-identical. |

Comparison images are [V4_Landing.png](validation/V4_Landing.png), [Final_PDF_Landing.png](validation/Final_PDF_Landing.png) and [PowerPoint_Landing.png](validation/PowerPoint_Landing.png). The aircraft image, original title, author, supervisor, equation, logos, composition and positioning were preserved. No generated image was used.

## Visual inspection and editability

All 12 PDF slides and all 12 native PowerPoint slides were rendered and visually reviewed. Screenshots are in `rendered/` and `validation/native_rendered/`; overviews are [PDF contact sheet](rendered/Contact_Sheet.png) and [PowerPoint contact sheet](validation/PowerPoint_Contact_Sheet.png).

Later slides retain V4's white background, Arial/Liberation Sans typography, dark gray headings, thin gray header/footer rules, neutral panels, UM6P footer identity and original widescreen proportions. Red and green are reserved for measured unfavorable/favorable outcomes. The diagrams are scientific flow diagrams rather than decorative artwork.

Visual review corrected a media-name collision that initially affected the copied cover equation, a wrapped backup heading, superscript-minus font handling and indentation in the repair diagnostic. The final outputs passed the exact cover tests above. No clipped or overlapping content remained in the reviewed renders. Plot axes, legends and mathematical equations were inspected at slide resolution; the larger data-driven figures replace difficult original plot labeling without changing measurements.

- Main equation instances have effective sizes **21–32 pt**; backup equations have a minimum effective size of **19.84 pt**. Small source citations use footer typography rather than body-text sizing.
- The slide manifest has no declared object outside the canvas and no missing image asset. The final TeX log has no overfull boxes or missing-character warnings. Eleven underfull warnings concern fixed-width footer text. Remaining font-path/PDF-version/tag warnings do not affect the reviewed rendering; the final PDF cover is inserted directly from V4.
- The PowerPoint contains **160 native text shapes**, plus native panels, diagram boxes and arrows. Its equations are SVG vector pictures with PNG fallbacks and editable TeX sources, **not native Office equation objects**. Numerical charts remain figure assets with regeneration code/data. The PPTX is not a sequence of full-slide raster pictures.
- Every OOXML part parses, package relationships resolve, and all 12 slides contain the intended embedded scripts/Q&A. Whitespace is normalized when comparing empty OOXML paragraphs; script words and order are retained.

Native rendering was checked with LibreOffice, not Microsoft PowerPoint itself. Font substitution or a different Office renderer can change later-slide spacing; the checked final PDF is the fixed-layout presentation copy. The first-page exact comparison applies to the recorded renderer/resolution. Font installation is listed in README for reproducible builds.

## Numerical and mathematical review

Each claim has a precise mapping in [Sources_and_Evidence.md](Sources_and_Evidence.md). All 50 local evidence links resolve in this checkout. Copied JSON/CSV/NPZ inputs and original figures are byte-identical to their repository sources; no numerical value was altered for presentation. The following assertions passed against the archived records or preceding audit logs:

| Claim | Verified value and qualification |
|---|---|
| Oracle NN-TENO5 failure | Sod N=120: non-positive density at step 6 in the audit reproduction log. This is the Keras oracle generation. |
| V2 first-invalid stage | Soft/.75 on the identified N=128 Riemann case: step 13, stage 1; p_min ≈ −0.00380485. Soft/.25 completes the same 15-step short case. Different archived continuation traces are explicitly distinguished in the notes. |
| Fixed-dt LF sweep | Multipliers 1/1.5/2/3/5: first non-finite zero-based indices 11/none/3/3/3. The finite 1.5 trace ends with enormous density, not a successful stabilization result. |
| Repair budget | One changed stage row; final RK-weighted integrated repair sources: mass +0.0016111493654, energy +0.0008888429519. These are not raw unweighted stage sums. |
| Selected V2 validation | 307 Euler trajectories, no invalid-stage failures; macro family×resolution L1 ratio 0.9479383303. This validation selected strength .25. |
| Independent Sod N=192 | L1 ratio 0.9572500682, a 4.275% reduction; density TV 0.8774663254 → 0.8797561900 and undershoot ≈0.0003506855. |
| Independent contact N=192 | L1 ratio 1.0687408293, a regression. |
| Closed-wall blast | Selected V2 fails at N=128 and 256 after the wall correction; classical TENO5 remains finite. No failed-row accuracy score is invented. |
| Nonlinear smooth Euler consistency | Last spatial L1 orders 2.0119685376 versus 4.9999252974 for historical split flux versus experimental state/Rusanov. No time-integration error enters this diagnostic. |
| WENO7 forward test | 10/100,000 non-finite coefficient rows at epsilon 1e−6; zero in the same test at 1e−4. Epsilon squared is far above float32 underflow; the precise NaN operation remains unresolved. |

The Euler physical flux, ideal-gas pressure, shared-face conservative update, V2 correction formula, three SSPRK3 stages, WENO-JS weights and limited-flux/residual expressions were manually checked against source and standard formulations. In particular:

1. The default Euler path reconstructs characteristic split **fluxes**, not physical left/right states. Roe projection is not described as a Roe Riemann flux.
2. A forward-Euler equation is labeled as the building block; the full SSPRK3 composition is in Backup A. Admissibility is required at intermediate stages, not only at final output.
3. Conservation, positive pressure/density, monotonicity, entropy stability, consistency and error control are not treated as interchangeable properties.
4. The fixed-dt alpha sweep is explicitly confounded by its changing stability scale and neural input distribution. Roe-conditioning causality remains unproven.
5. The averaging mismatch affects the classical FV path too. The fifth-order state-flux result is a smooth consistency result, not a validated shock/positivity remedy.
6. Soft/hard policies and the four model generations are distinguished. The error metric is mean conserved-component L1 where appropriate, not density-only error or a universal percentage gain.

## Attempted fixes, established remedies and research proposals

The deck identifies reduced correction strength, classical blending, guards, direct repair, training penalties and related training controls as actual attempted mechanisms. Their limitations are explained: no enforced invariant-domain property, artificial conservation sources, a once-hidden penalty signal, training/deployment mismatch, or incomplete controlled evidence.

Conservative positivity-preserving flux limiting and convex limiting are identified as established literature foundations **not yet validated in this neural implementation**. A sufficient low-order admissibility property, appropriate CFL and coupled face/stage/boundary choices are required. SSPRK3 does not supply a missing spatial property. An arbitrary limiter coefficient in [0,1] does not prove positive pressure.

The Giesselmann–Kwon–Krumscheid 2026 preprint was verified from its primary text. Its periodic Cartesian RKDG–SIAC convection–diffusion analysis is not presented as a learned-filter result or a theorem for this neural Euler shock solver. A continuous reconstruction residual is useful for a posteriori estimation only with the required reconstruction, stability, regularity, initial/boundary and time-error analysis. Near-shock usefulness remains a research question.

Primary bibliography was checked using author/institutional/publisher/arXiv records. Some direct publisher PDF requests were unavailable; the exact primary metadata sources and access qualification are recorded in Sources_and_Evidence.md.

## Practical limits and final disposition

The presentation is complete and reviewable. All required deliverables are present, all 71 automated checks pass, all slides were rendered, and the numerical claims retain their uncertainty boundaries. No claim of universal generalization, guaranteed neural Euler positivity, resolved blast root cause, reliable Shu–Osher ranking or post-shock a posteriori certification is made. Human rehearsal and a final check on the interview computer remain practical preparation steps rather than completed tests.
