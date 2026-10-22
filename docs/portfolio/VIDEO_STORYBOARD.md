# 80-second demonstration storyboard

Use only the committed project-generated graphics. Keep the reference caveat
visible whenever a numerical-reference result appears. Do not animate fabricated
intermediate states; use the exported time snapshots if making an animation.

| Time | Visual | Suggested narration |
|---|---|---|
| 0–10 s | Scalar discontinuity plot; highlight error profile | “Capturing a sharp wave accurately is difficult: numerical diffusion blurs it, while aggressive reconstruction can create oscillations.” |
| 10–22 s | Reconstruction pipeline | “I extended a WENO/TENO codebase and implemented a differentiable JAX pipeline for scalar equations and one-dimensional Euler flow.” |
| 22–34 s | Gate decision diagram; architecture text | “Separate small neural networks adjust candidate selection. A deterministic TENO smoothness gate controls where their corrections are used.” |
| 34–46 s | Strength study, λ=.25 marker | “On all 922 validation trajectories, modest soft corrections helped. Larger corrections could fail. These results selected the inference policy.” |
| 46–58 s | Independent benchmark summary; show an unfavorable bar | “Long-time behavior is mixed. I kept the failures and reference uncertainty visible rather than selecting only attractive examples.” |
| 58–69 s | General nonlinear Euler consistency plot | “A broader smooth test exposed a second-order finite-volume limitation. A separate formulation recovered fifth-order spatial consistency on that diagnostic, with further shock validation still needed.” |
| 69–80 s | Terminal: tests and artifact check; GitHub URL | “The contribution is a reproducible numerical investigation: code, analytic references, checkpoint hashes, tests, measured results and limitations. Supervised by Radouan Boukharfane, building on published WENO-NN and TENO work.” |

Suggested screen capture: one CPU smoke command followed by the saved report and
plots. Avoid waiting for a full study during the recording. Show the exact commit
from the final release report. The post and carousel are drafts for the owner's
review; no content has been posted or published automatically.
