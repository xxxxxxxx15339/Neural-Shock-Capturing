# New-model result

Two new 3155-parameter NN-TENO5 models were trained through the conservative FV
Euler solver, using matched state inputs at N=64/128/256. Each received 150
teacher steps and 300 attempted PDE updates with K=15. Seed 4101 applied all
300 updates; seed 4102 rejected two invalid batches and applied 298.

Validation selected update 200 for both seeds. Seed 4101 is the primary candidate
by validation alone. Its L1 score improved 1.31% over old V2 in the same solver;
seed 4102 improved 0.74%. A later seed-4102 checkpoint failed three fixed-dt
validation trajectories and was not selected. Diagnostic bounded retries recovered
them conservatively, without changing checkpoint selection.

Fresh eight-trajectory holdout, L1 change versus **old V2 in the same solver**:

| Seed | N=128 | N=256 |
| --- | --- | --- |
| 4101 | +2.92% | +8.80% |
| 4102 | +1.95% | +5.69% |

Positive percentages mean worse error. Neither model is promoted as a replacement.
All holdout methods had zero limiter activation/retries, so active limiting does
not explain these accuracy differences.

Both selected models completed all 32 canonical/grid-transfer configurations,
including Sod, Lax, rarefaction, contrast, near vacuum, full-time Shu–Osher,
reflecting-wall blast and unseen 512-cell Sod/Lax. Stability is attributable to
the checked conservative solver, not a learned positivity theorem. Some error
comparisons improved, many regressed. Blast accuracy remains unassessed and
Shu–Osher rankings retain numerical-reference uncertainty.

The CPU suite passed 136 tests and 37 subtests; the legacy suite passed two tests.
The new checkpoints, all late failures, per-case metrics, training histories,
figures and synchronized timings are preserved. Original artifacts are unchanged.

**Decision:** keep the old saved V2 weights as the preferred existing neural
checkpoint. The validation improvement did not generalize. Future work should
investigate trajectory horizons, loss scaling and contact-resolution behavior
using newly declared holdouts, rather than retuning on this one.

See [the full report](TRAINING_RESULTS.md) and
[the selected checkpoint record](../results/primary_candidate.json).
