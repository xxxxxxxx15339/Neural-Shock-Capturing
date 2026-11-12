# Final validation — targeted scientific revision

Reviewed 30 October 2026, repository HEAD `f9d4c2a`. The nine-slide main presentation is retained, with the requested scientific clarifications and local readability improvements. **83 automated checks pass**; the detailed results are [revised_source/validation/checks.json](revised_source/validation/checks.json). Manual visual/scientific review is reported separately below. No solver changes, training or new CFD experiments were performed.

## Deliverables and timing

| Item | Verified result |
|---|---|
| [Main revised PDF](Sebastian_Research_Interview_Main_Only_REVISED.pdf) | Exactly 9 pages. |
| [Complete revised PDF](Sebastian_Research_Interview_REVISED.pdf) | 12 pages: nine main slides plus the three existing technical backups. |
| [Revised PowerPoint](Sebastian_Research_Interview_REVISED.pptx) | 12 slides; only the nine main slides show in the normal slideshow. Backups 10–12 are hidden. |
| Editable sources | [Beamer/TikZ source](Sebastian_Research_Interview_REVISED.tex), revised native-OOXML generator, equation TeX/assets, notes source and plotting data in `revised_source/`. |
| [Revised notes](Speaker_Notes_REVISED.md) | All nine main scripts, estimated times, equation guides, observations, uncertainties/remedies, terms and technical Q&A; scripts and full guidance embedded in PPTX notes. |
| Script length | **1,473 words**, excluding guides, preparation, questions and backups. |
| Main talk estimate | **795 seconds = 13:15**. Times: 0:25, 1:00, 1:35, 2:15, 1:30, 1:25, 2:00, 1:20, 1:45. |

The approximately 111-word/minute schedule includes pauses for equations and figures. It is an estimated speaking plan, not a measured human rehearsal. Reading the preparation sections or backup scripts during the main talk would exceed that plan.

## Preservation of originals and design

- All **154 original/protected files** match the SHA-256 snapshot taken before revision. This includes the original presentations, sources, notes, figures, prior package, V4, audit and selected research inputs. The snapshot is [original_files_sha256.json](revised_source/validation/original_files_sha256.json). Git reports no tracked-file changes.
- The main title slide is the original V4 vector page in the revised PDFs, at the original **960.009 × 540 point** dimensions. No image was generated or substituted.
- At **1800 × 1013 pixels**, the revised PDF landing page has **0 changed pixels** versus both V4 page 1 and the original main presentation. The revised native PowerPoint render also has **0 changed pixels** versus V4 page 1.
- The PowerPoint landing slide XML and all its referenced original image/equation assets are byte-identical to the original editable template.
- Main-slide title/header/footer positions, text styles, rules and logo geometry match the original layout registry. The font family, grayscale colors and existing red/green outcome coding are preserved. Slide 3's learned block receives a local outline; no unrelated template or imagery was introduced.
- Backup source content and layout match the original registry, remain separate, and are excluded from the main timing.

The preservation checks compare actual files and rendered pixels; they do not rely on a subjective statement that the cover is “similar.” Comparison evidence is in `revised_source/validation/`, including `V4_Landing.png`, `Original_Landing.png`, `Final_PDF_Landing.png` and `PowerPoint_Landing.png`.

## Rendering and readability

All nine main slides and all three backups were rendered from both the final PDF and the native PowerPoint export. Every main slide also has an original/revised side-by-side comparison in [validation/comparisons](revised_source/validation/comparisons/).

Review images:

- [Main-slide overview](rendered_REVISED/Main_Contact_Sheet.png).
- [Native PowerPoint overview](revised_source/validation/PowerPoint_Main_Contact_Sheet.png).
- [Separate backups](rendered_REVISED/Backup_Contact_Sheet.png).

The review checked text, equations, numerical curves, legends, citations, callouts and their spacing at full slide resolution. The final fixes enlarged slide 4's pressure formula, shortened slide 7's takeaway marker so it clears the citation, and placed slide 5's two conclusion lines explicitly to avoid a native-export footer collision. No clipping, unwanted wrapping or overlapping content remains in the reviewed final renders.

Main equation instances have a minimum effective size of **21 pt**; backups have a minimum of **19.84 pt**. The blast failure is **21 pt native bold text**, with a separate classical-control statement. Plot ratio labels were enlarged. The Sod inset is an explicit shock-tail zoom with readable axes and a larger legend, rather than several visually overlapping full profiles. Source citations retain the original small-footer role; full details are in the notes/evidence documents.

The final TeX log has **no overfull boxes or missing-glyph warnings**. Retained warnings concern fixed-width underfull footer boxes, local font paths and inclusion of the original tagged PDF. The final cover is inserted directly from the original vector page. Every OOXML part parses, all relationships resolve and all embedded notes match their intended text.

Native-PPTX inspection used LibreOffice. Microsoft PowerPoint and the interview projector were not directly tested. Later-slide text spacing can depend on the installed fonts and renderer; the final PDF is the checked fixed-layout copy. Equation objects in PPTX are vector SVGs with PNG fallback and editable TeX sources, **not native Office equations**. Text and diagrams are native editable objects; charts have source data/code. The PPTX is not a stack of whole-slide images.

## Scientific verification

The full modification rationale and exact source locations are in [Scientific_Changes.md](Scientific_Changes.md). The original [Sources_and_Evidence.md](Sources_and_Evidence.md) supplies retained checkpoint/configuration/commit and primary-literature provenance. Numerical copies were checked byte for byte against original CSV/JSON/NPZ artifacts; no measurement was adjusted.

| Requested clarification | Verification and interpretation |
|---|---|
| Pressure mechanism | The boxed formula has the correct derivatives: d p/d E = gamma−1, d p/d m = −(gamma−1)v, d p/d rho = (gamma−1)v²/2. It is explicitly a local linearization at an admissible reference state, not a unique diagnosis of the archived failure. |
| Actual negative states | Keras oracle Sod N120 density failure at step 6 is retained. Later V2 soft/.75 first-invalid pressure is step 13/stage 1 in the identified audit trace; soft/.25 completes the same short 15-step case. Different archived continuation traces are not merged. |
| LF sweep | The unchanged CSV plot preserves its earlier/later divergence observations. Fixed timestep, changing LF splitting, changing effective stability scale and changing neural inputs are stated together. Additional dissipation is not described as inherently harmful; Roe conditioning remains a hypothesis. The proposed diagnostics have not been run as new experiments. |
| Repair budget | `repair.json` reports one repaired stage and final `integrated_repair_exchange` mass +0.0016111493654 and energy +0.0008888429519. `diagnostics.py:101,114,131` proves these are accumulated RK-weighted sources. The label does not call them raw stage additions or solution-error norms. |
| Generalization | Validation 307/zero invalid-stage failures and macro L1 0.9479383303 are retained. Independent Sod ratio 0.9572500682 improves the stated mean-state norm, while TV and undershoot worsen. Contact N192 ratio 1.0687408293 regresses. Selected V2 blast fails at both grids; TENO5 completes the controls. |
| Uncertain rankings and inset | Shu has no displayed accuracy point or ratio. The Sod zoom uses unchanged N192 arrays and explicitly narrows both axes; it is not the full shock-density range. An undershoot below .125 is not negative density. |
| FV consistency | The unchanged figure's last classical-TENO5 spatial orders are 2.0119685376 and 4.9999252974. The historical averaged-data formulation is identified; the claim is not generalized to all LF/FD WENO methods. The fifth-order result is a smooth semidiscrete consistency diagnostic, not validated shock evolution. |
| Reliable-method connection | Shared-face conservative limiting, appropriate low-order admissibility/CFL/factors, conditional SSP and structural moments are distinguished from proposed learned extensions. Arbitrary theta and a residual alone are explicitly insufficient. The 2026 RKDG–SIAC paper is not presented as neural Euler certification. |

Conservation, positivity, monotonicity, entropy stability, formal/spatial consistency, range excursions and a posteriori error control remain distinct throughout the notes. Physical Shu–Osher oscillations are not labeled numerical divergence. No single distribution shift or characteristic defect is claimed to be the proven cause of the blast failure.

## Final disposition

All required revised outputs are present. The main talk retains its nine-slide structure and professional style, and every important limitation now has an observation, a mechanism or explicitly unresolved hypothesis, an explanation of insufficient attempted fixes, and a relevant established numerical foundation. Literature remedies remain unimplemented/unvalidated where that is the actual project status. Final delivery still benefits from the speaker's own rehearsal and a font/display check on the interview computer.
