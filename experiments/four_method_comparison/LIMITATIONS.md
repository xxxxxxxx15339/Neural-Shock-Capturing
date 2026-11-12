# Limits of the evidence

- NENO6 weights and scalar reconstruction are authentic and author-code parity
  is tested. The author repository does not supply an Euler/Burgers solver or
  training code; these results port its reconstruction into the common solver.
  They must not be represented as reproducing every published NENO6 result.
- Phase21 Euler and run6 scalar are different trained heads/backbones. The later
  dataset-V2 fixed blend is not a learned-head checkpoint. The original HDF5
  classifier is another backbone; use the explicit ablation for causal claims.
- The full proposed PDF physical input vector is not consumed by these artifacts.
  Historical head training uses soft selection, evaluated here with its authentic
  hard inference rule; Euler historical training used nonconservative repairs.
- Point-sampled historical Euler split flux and scalar state-flux/Rusanov paths
  are retained. This does not test the newer finite-volume state-flux correction
  or prove high-order nonlinear Euler accuracy.
- Canonical problems were historically exposed. The protocol is frozen for this
  comparison, but is not a new untouched holdout. No retraining/multi-seed
  variability or reliable independent-trajectory bootstrap interval is available.
- Raw failures are retained. No unconditional positivity, TVD or entropy-stability
  property follows from a convex flux blend. Blast is a stability stress case,
  with half-node outer wall faces for the historical point grid; no certified
  canonical cell-average blast accuracy reference is available.
- Shu reference uses one classical solver, a resolution ladder and dt check.
  Sensitivity estimates are not rigorous bounds; small error differences can be
  unresolved. Only final-time Shu error is available, not a converged time-error
  reference history. Exact-reference Sod/Burgers snapshots provide time controls.
- Archived original-network square-wave metrics have a small deterministic CPU
  reproduction discrepancy. Missing immutable original checkpoint/environment
  metadata prevent a definitive root-cause attribution. This evidence is not
  silently classified as successfully reproduced. The historical threshold=1
  all-keep probe discards saturated-one probabilities and is not unconditional.
- Runtime is CPU-specific, three warmed repeats, no core-affinity control. It
  measures common JAX ports, not the original frameworks. Failed endpoint error
  and complete-rollout runtime are absent, not replaced with finite guesses.
  Closest measured computational budgets may still differ materially.
- The earliest Euler checkpoint has nonfinite weights. Resume dictionaries are
  not inference checkpoint schemas. Missing legacy best-HDF5 references already
  absent from the previous baseline were not reconstructed or relabeled.
