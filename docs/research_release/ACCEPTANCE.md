# Scientific and engineering acceptance criteria

Implementation and numerical evidence are assessed separately. See RESULTS.md
and MILESTONES.md for completed-study details and remaining resource limits.

| Criterion | Assessment | Evidence / remaining condition |
|---|---|---|
| Classical regression | PASS | Final CPU suite: 117 tests + 34 subtests passed; legacy smoke 2 passed |
| Zero-strength identity | PASS | Interface and full Euler rollout equality, soft/hard |
| V2 evaluator agreement | PASS | Matched original V2 advection/Burgers/Euler predictions |
| Conservation budgets | PASS for measured unguarded valid trajectories | Per-case stage-weighted boundary budgets; no claim for arbitrary trajectories |
| Stage positivity | MEASURED, method-dependent | λ=.25 full validation has zero failures; stronger settings fail |
| Historical repair conservation | FAIL | Measured mass/energy changes; repair remains explicitly nonconservative |
| Fifth-order smooth behavior | PARTIAL / FAIL as a general claim | Regular smooth spatial cases near order five; critical points differ; general nonlinear historical Euler near order two |
| Alternative finite-volume Euler | EXPERIMENTAL | Nonlinear smooth spatial order near five; no shock/positivity/training validation |
| Long-time improvement | MIXED | Prospective cases; include unfavorable advection and reference caveats |
| Fine temporal protocol / blast reference | INCOMPLETE | Spatial ladder complete through 640; CPU temporal controls through 160; original fine temporal/blast reference workers stopped for CPU cost |
| Blast stability | FAIL for selected V2 | Correct closed-wall smoke fails before t=.001; full-time stability results retain invalid stages; no full-time accuracy reference |
| Reference certainty | CASE-DEPENDENT | Analytic cell averages for advection/Riemann; numerical Cauchy ladder and temporal control elsewhere |
| Multi-seed improvement | NOT ESTABLISHED | Original V2 uses one training seed per physics; no new training |
| Input warm-up causal benefit | NOT ESTABLISHED | Distribution mismatch measured; optional ablation implemented, not trained |
| CPU reproduction | PASS for recorded environment | Tests, smoke and full validation executed; new clean V2 environment not reinstalled |
| GPU reproduction | NOT TESTED | Official optional setup guidance only |
| Checkpoint compatibility | PASS | Historical backbone, linear/rich heads; no synthesized head |
| Frozen artifact preservation | PASS | 498 local files match baseline hashes; 491 tracked files for clone/CI |
| Data leakage | CONTROLLED with stated limits | Original test excluded from selection; full validation selects λ; final manifest frozen before evaluation |
| Computational speedup | NO GENERAL CLAIM | Use measured synchronized timing; equal-budget tests limited to timed IVPs/grids |
| Portfolio accuracy | REVIEWABLE | Measured plots/captions, explicit negative findings, attribution and draft posts |
| Paper-level novel validated method | NOT READY | General smooth Euler limitation, critical-point behavior, positivity and multi-seed/reference questions remain |

Public presentation as a **qualified research software investigation** is appropriate
once its artifacts have been reviewed. Presenting the historical method as a
universally fifth-order, conservative positivity-preserving, faster neural solver
would fail these criteria. No public posting or repository push is performed here.
