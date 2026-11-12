# NN-TENO5 accuracy-upgrade experiment

Start with [the measured research report](reports/FINAL_RESEARCH_REPORT.md),
[the interview summary](reports/PRESENTATION_SUMMARY.md), and
[the mathematical/audit notes](reports/AUDIT_AND_METHOD.md).

The Euler FV consistency improvement is measured; there is no claim of a
universally superior neural method. Matched-input retraining has mixed effects.
The optional conservative safeguard completes the tested stress cases but has
accuracy and runtime tradeoffs. Historical defaults, data, models and results
are preserved. No push or merge is performed.

## Implementation

`src/differentiable/accuracy_upgrade.py` is the opt-in module. Example:

```python
import jax
jax.config.update('jax_enable_x64', True)
from src.differentiable.teno5_jax import teno5_reconstruct_jax
from src.differentiable.accuracy_upgrade import flux_variant, retry_step

# Historical factory delegated unchanged when variant='historical'.
flux = flux_variant(teno5_reconstruct_jax, 'periodic', variant='fv_state_flux')

# u contains valid Euler cell averages. Check the returned status and time.
new_u, info = retry_step(u, dt, dx, teno5_reconstruct_jax, boundary='transmissive')
if not bool(info['success']):
    raise FloatingPointError('Admissibility retries exhausted')
accepted_time_increment = float(info['accepted_dt'])
```

The raw flux has an admissible-input contract. Use the checked integrator for
shock problems. Supported boundaries: `periodic`, `transmissive`,
`reflective-wall`. Face scaling, full flux correction and retries are distinct
from neural mixing. No density/pressure clipping is used.

## Result identity

- `historical_teno`: original characteristic split-flux TENO5.
- `historical_p0_learned`: original P0 checkpoint and learned hard blend.
- `historical_v2_soft25`: saved V2 backbone, deterministic gate, soft λ=.25.
- `fv_teno`, `fv_v2_soft25`: state-flux path with the corresponding policy.
- `safe_fv_*`: physical face scaling and conservative FE-like-stage correction.
- Pilot A: historical forward-state warm-up, historical flux.
- Pilot B: actual bidirectional split-flux warm-up, historical flux.
- Pilot C: actual bidirectional state warm-up, FV flux with face scaling.

All pilots use the same original 3155-parameter architecture. Their separate
warm/final checkpoint metadata includes seeds, objectives, training IDs,
validation IDs and budgets. They are smaller-budget pilots, not replacements
for the saved V2 models. `final_holdout/frozen_candidates.json` records exact
checkpoint hashes before confirmation. Historical validation and tests have
already been examined; the holdout is a newly predeclared six-Riemann set.

Use `results/euler_actual_stages/summary.json` for final Euler comparisons.
`results/euler/` deliberately preserves an earlier wrapper with stricter raw
FE-sub-update rejection. It is superseded for raw-solver comparisons. Initial
failing development logs are preserved. No failed trajectory has an accuracy
score in the final benchmark table. Scalar solvers are unchanged.

## Reproduction

Run from the repository root. The existing environment is:
`/home/bazouil/miniconda3/envs/teno-jax/bin/python`.
`configs/environment.freeze.txt` records installed packages; a fresh installation
was not tested. Each numerical command needs a **new** output path. Examples use
`/tmp/accuracy-reproduction-*`; change paths if they already exist.

```bash
export JAX_PLATFORMS=cpu
export MPLCONFIGDIR=/tmp/accuracy-upgrade-mpl
PY=/home/bazouil/miniconda3/envs/teno-jax/bin/python

$PY -m pytest tests/test_core.py tests/test_teno.py tests/test_feature_vector.py tests/test_teno5_jax_regression.py tests/test_teno5_jax_euler_regression.py tests/test_projection_jax.py tests/test_losses_jax.py tests/test_accuracy_upgrade.py tests/test_accuracy_upgrade_inputs.py src/differentiable/tests -q

$PY -m src.differentiable.v2_evaluate sweep --config configs/v2/smoke.json --output /tmp/accuracy-reproduction-baseline
$PY -m experiments.accuracy_upgrade.study spatial --output /tmp/accuracy-reproduction-spatial.json
$PY -m experiments.accuracy_upgrade.study temporal --output /tmp/accuracy-reproduction-temporal.json
$PY -m experiments.accuracy_upgrade.study euler --grids 64 128 --output /tmp/accuracy-reproduction-euler
$PY -m experiments.accuracy_upgrade.scalar --output /tmp/accuracy-reproduction-scalar
$PY -m experiments.accuracy_upgrade.linear_retention --output /tmp/accuracy-reproduction-retention.json
$PY -m experiments.accuracy_upgrade.train --output /tmp/accuracy-reproduction-training --seeds 3101 3102

# Run the following only after the other numerical jobs have finished.
$PY -m experiments.accuracy_upgrade.performance --output /tmp/accuracy-reproduction-performance.json
```

The confirmation/reference/long-rollout CLIs default to this experiment's frozen
checkpoints and benchmark artifacts; this prevents accidental comparison to
arbitrary newly selected models:

```bash
$PY -m experiments.accuracy_upgrade.confirmation holdout --output /tmp/accuracy-reproduction-holdout
$PY -m experiments.accuracy_upgrade.confirmation shu --output /tmp/accuracy-reproduction-shu
$PY -m experiments.accuracy_upgrade.long_rollout --output /tmp/accuracy-reproduction-long
```

These do **not** automatically substitute checkpoints from a newly trained run.
To evaluate those as a new study, explicitly register their paths and freeze a
new candidate manifest before running its holdout. The supplied holdout must
never be reused for tuning. The report/plot generators read the recorded
experiment layout; numerical results are not recomputed by plotting.

The separate legacy smoke environment can be checked without rewriting tracked
Python 3.7 bytecode:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/bazouil/miniconda3/envs/weno-nn/bin/python -m pytest tests/test_networks_smoke.py -q
$PY -m src.differentiable.v2_verify --protected-only
```

`baseline_manifest.json` freezes baseline file identities and records the audit.
`results/preservation_final.json` verifies them after the work. Seven ignored
historical `.best.h5` files referenced by the archived 498-file inventory were
already absent in this checkout; its full check reports those missing paths.
All 491 tracked protected artifacts and all 592 inventoried present files pass.
Use `--tracked-only` on the preservation CLI for this checkout. The new deliverables
are covered by `artifact_hashes.json`. No original data/checkpoint/results file
is an output target.

Verify exported Euler metrics, frozen candidate identities, split separation and
new artifact hashes with:

```bash
$PY -m experiments.accuracy_upgrade.verify
```

The verifier's own output and the hash manifest are excluded from that manifest
to avoid self-reference. `plots_manifest.json` maps figures to their numerical
source artifacts. Initial unsuccessful development/verification logs are retained
alongside the final passing results.
