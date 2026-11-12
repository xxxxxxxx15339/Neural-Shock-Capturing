# Research interview summary

- **Problem:** NN-TENO5 gains were inconsistent and some Euler corrections produced invalid density/pressure. Historical repairs could violate conservation.
- **Previous method:** five-input candidate classifier plus classical/neural mixing; saved V2 warm-up used characteristic states, inference used characteristic split fluxes.
- **Numerical bottleneck:** reconstructing F(Ubar) as a cell-average flux limits general nonlinear Euler FV consistency. Coupled smooth order was about 2.03.
- **New method:** isolated characteristic-state reconstruction plus Rusanov flux; shared-interface LF flux correction, physical face scaling and bounded stage-aware retries. Original defaults and checkpoints are preserved.
- **What worked:** coupled spatial order ≈5.00; conservative safeguards complete all 16 tested stress configurations; favorable saved V2 + FV results on six frozen moderate Riemann trajectories.
- **What failed:** unlimited FV fails strong cases; safeguarded neural rarefaction error is about twice classical; critical-point/momentum-only order remains near four. An always-retain ablation recovers fifth order but is not a shock method.
- **Training result:** paired matched-input pilots change validation L1 by -0.04543% and +0.12993%; no consistent gain. Solver improvements must not be credited to training.
- **Cost:** synchronized CPU timings and equal-budget comparisons are recorded; lower same-grid error does not imply computational advantage.
- **Validation:** 132 unique CPU tests plus 37 subtests and two legacy tests pass; baseline predictions and protected artifacts are checked. No GPU or fresh-install claim.
- **Decision:** retain experimental options and the existing backbone/defaults. Defer richer mixing heads and full retraining. No universally superior neural solver is claimed.
- **Next work:** local conservative limiting, critical-point selection, full-budget multi-seed trajectory training and broader independent references.

Exact tables, scopes, uncertainty and reproduction commands are in `FINAL_RESEARCH_REPORT.md`. No unmeasured benchmark is presented as completed.
