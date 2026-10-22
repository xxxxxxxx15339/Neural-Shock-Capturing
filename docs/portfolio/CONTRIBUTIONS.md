# Technical contributions and attribution

## Yasser BAOUZIL — project implementation and experimental work

- Extended the inherited WENO-NN research code with classical and neural TENO
  reconstruction and scalar/Euler evaluation paths.
- Implemented the differentiable JAX numerical stack, neural reconstruction,
  SSPRK3 rollouts and unrolled-loss training experiments; integrated separate
  5→64→32→16→8→3 scalar and Euler networks with analytic trajectory data.
- Implemented and evaluated deterministic smoothness-gated neural corrections,
  preserving historical checkpoints and inference-policy compatibility.
- Built numerical regression checks, checkpoint validation, matched evaluation,
  full-validation strength/soft-hard studies and trajectory bootstrap analysis.
- Investigated stage positivity, conservation and repair budgets, critical-point
  accuracy, long-time behavior, solver-input mismatch and computational cost.
- Diagnosed finite-volume consistency in the historical Euler flux path and added
  an explicitly separate state-flux spatial-consistency variant.
- Assembled reproducible experiment configurations, manifests, figures and public
  research materials that preserve unfavorable and incomplete findings.

These are implementation and investigation contributions. The new release does
not claim a novel universal solver, a positivity theorem, an experimental flow
campaign, multi-seed training robustness or a completed causal warm-up ablation.
The experiment reports and Git history distinguish original project work from
this validation/release pass.

## Supervisor and published foundations

**Radouan BOUKHARFANE:** internship supervision, unrolled physics-loss research
proposal and numerical-method clarifications. The proposal is preserved in
`references/Unrolled_Physics_Loss_Proposal.pdf` and its implementation guide.

**Ben Stevens and Tim Colonius:** foundational WENO-NN study and accompanying
implementation from which the original research code derives:
[Caltech publication record](https://authors.library.caltech.edu/records/1xdr6-8n320).

**Lin Fu, Xiangyu Hu and Nikolaus Adams:** foundational TENO methods:
[TUM publication record](https://portal.fis.tum.de/en/publications/a-family-of-high-order-targeted-eno-schemes-for-compressible-flui/).
Classical WENO, Riemann solvers and SSPRK time integration are established
numerical-analysis techniques, not inventions claimed by this internship.

## Three CV bullets

- Developed a differentiable JAX solver and neural TENO5 reconstruction pipeline
  for scalar conservation laws and 1D Euler flow, with separate scalar/Euler
  checkpoints and analytic finite-volume reference data.
- Built a matched evaluation system covering 922 validation trajectories, soft/
  hard strength ablations, bootstrap uncertainty, conservation, stage positivity
  and spatial-convergence diagnostics; documented mixed and negative results.
- Diagnosed Euler input-distribution and finite-volume consistency limitations,
  implemented testable numerical variants, and packaged CPU reproduction,
  checkpoint provenance, numerical tests and research figures.

## GitHub description

Differentiable JAX research on neural TENO5 shock capturing: scalar/Euler solvers,
analytic references, reproducible validation, convergence and conservation diagnostics.

Suggested topics: `scientific-machine-learning`, `computational-fluid-dynamics`,
`jax`, `numerical-methods`, `conservation-laws`, `weno`, `teno`, `shock-capturing`,
`differentiable-programming`, `research-software`.
