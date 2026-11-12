# New conservative FV NN-TENO5 training

This run trains the original 3155-parameter classifier from scratch through
**the full conservative FV Euler solver**, including the admissibility safeguard
inside every SSPRK3 rollout. It extends the earlier small C pilots; it does not
replace any prior checkpoint or numerical default.

Start with [the measured results](reports/TRAINING_RESULTS.md),
[training summaries and selected paths](results/training_summary.json),
[problem comparison CSV](results/problem_comparison.csv), and
[frozen candidates](configs/frozen_candidates.json).

Two seeds (4101/4102) receive 150 warm-up steps and 300 attempted PDE updates each, at
N=64/128/256, batch 8, K=15. The original backbone and fixed soft λ=.25 remain
unchanged. See [the exact configuration](configs/training.json) for the objective,
selection rule, physical floors, gradient policy and retry limits.

Checkpoint selection uses all Euler validation trajectories, not the canonical
problem tests or fresh holdout. Both seeds and all validation checkpoints are
retained. The primary candidate is the lowest validation macro ratio across the
two seeds; `results/primary_candidate.json` records its path/hash.

## Inference

Run from the repository root, using Python 3.11/JAX float64. Read the selected
checkpoint path from `results/primary_candidate.json` or the per-seed summaries.

```python
import json
from pathlib import Path
import jax
jax.config.update('jax_enable_x64', True)
from src.differentiable.model_io import load_checkpoint
from src.differentiable.reconstruction import make_reconstruction
from src.differentiable.accuracy_upgrade import retry_step

run = Path('experiments/accuracy_upgrade/training_2026_11_12')
choice = json.loads((run / 'results/primary_candidate.json').read_text())
model = load_checkpoint(choice['checkpoint'], choice['sha256'])
reconstruct = make_reconstruction(model, 'gated-soft', fixed_lambda=.25)

# u contains admissible conservative Euler cell averages, shape (N, 3).
next_u, info = retry_step(u, dt, dx, reconstruct, boundary='transmissive')
if not bool(info['success']):
    raise FloatingPointError('Conservative admissibility retries exhausted')
# Advance physical time only by this amount, which may be smaller than dt.
accepted_dt = float(info['accepted_dt'])
```

Use `periodic`, `transmissive` or `reflective-wall` boundaries as appropriate.
Do not pass this state-input model through the historical split-flux path and
interpret it as the same trained method. A zero neural strength reproduces the
corresponding classical FV reconstruction. The original five-input classifier
format is retained; the saved files contain NumPy parameter arrays compatible
with the existing checkpoint loader.

## Reproduce a new run

The completed directory is immutable experiment evidence. Every run needs a new
subdirectory under `experiments/accuracy_upgrade` with its own configuration.
Do not rerun training into this existing directory.

```bash
export JAX_PLATFORMS=cpu
export MPLCONFIGDIR=/tmp/accuracy-training-mpl
PY=/home/bazouil/miniconda3/envs/teno-jax/bin/python
NEW=experiments/accuracy_upgrade/training_repeat
mkdir -p "$NEW"/configs "$NEW"/checkpoints "$NEW"/results "$NEW"/plots "$NEW"/reports
cp experiments/accuracy_upgrade/training_2026_11_12/configs/training.json "$NEW"/configs/training.json

$PY -m experiments.accuracy_upgrade.full_training prepare --run-dir "$NEW"
$PY -m experiments.accuracy_upgrade.full_evaluation declare --run-dir "$NEW"
$PY -m experiments.accuracy_upgrade.full_training smoke --run-dir "$NEW"
$PY -u -m experiments.accuracy_upgrade.full_training train --run-dir "$NEW"

$PY -u -m experiments.accuracy_upgrade.full_evaluation problems --run-dir "$NEW"
$PY -u -m experiments.accuracy_upgrade.full_evaluation holdout --run-dir "$NEW"
$PY -m experiments.accuracy_upgrade.full_evaluation classification --run-dir "$NEW"
$PY -m experiments.accuracy_upgrade.full_evaluation diagnose --run-dir "$NEW"
$PY -m experiments.accuracy_upgrade.full_evaluation spatial --run-dir "$NEW"
$PY -m experiments.accuracy_upgrade.full_evaluation scalar --run-dir "$NEW"

# Only after every other numerical worker has finished:
$PY -m experiments.accuracy_upgrade.full_evaluation performance --run-dir "$NEW"
$PY -m experiments.accuracy_upgrade.full_report figures --run-dir "$NEW"
```

The reports require test logs in the selected run directory, as in this run.
Run the CPU regression command below into `"$NEW"/results/cpu_regression.log`
and the separate legacy command into `"$NEW"/results/legacy_tests.log` before
assembling the report. Then:

```bash
$PY -m experiments.accuracy_upgrade.full_report report --run-dir "$NEW"
$PY -m experiments.accuracy_upgrade.full_report verify --run-dir "$NEW"
```

The fixed seeds reproduce the declared confirmation set for reproducibility.
Once examined, that set must not be reused to tune a subsequent method and then
called untouched. Declare a different fresh holdout for a new research claim.
Training trajectories and validation trajectories are checked for disjoint IDs;
original test data are never opened as datasets.

Known canonical controls can optionally be computed with the `baselines` action
while fixed-policy training continues. The later `problems` action validates and
reuses their cache, including failed controls. New-model/fresh-holdout evaluation
requires frozen validation-selected checkpoints. Cache reuse never changes a
failed trajectory into a successful one.

## Tests and interpretation

```bash
$PY -m pytest tests/test_core.py tests/test_teno.py tests/test_feature_vector.py tests/test_teno5_jax_regression.py tests/test_teno5_jax_euler_regression.py tests/test_projection_jax.py tests/test_losses_jax.py tests/test_accuracy_upgrade.py tests/test_accuracy_upgrade_inputs.py tests/test_accuracy_upgrade_training.py src/differentiable/tests -q
PYTHONDONTWRITEBYTECODE=1 /home/bazouil/miniconda3/envs/weno-nn/bin/python -m pytest tests/test_networks_smoke.py -q
$PY -m experiments.accuracy_upgrade.verify
```

- Euler tests include Sod, Lax, two-rarefactions, strong contrast, near vacuum,
  short/full Shu–Osher, reflecting-wall blast smoke/full, longer Sod and unseen
  512-cell Sod/Lax grids. Raw failed controls remain in the results.
- Riemann references are analytic cell averages. Shu–Osher numerical references
  retain source hashes and empirical Cauchy uncertainty. Blast is stability-only.
- Scalar advection/Burgers runs are **zero-shot transfer** of Euler-trained
  weights, not scalar-specific training. Scalar solvers are unchanged.
- Training gradients freeze discontinuous limiter decisions; this is not full
  differentiability through switching. No high-order positivity/entropy theorem
  is claimed.
- Only the final isolated production benchmark provides comparable inference
  costs. Exploratory timings can include host diagnostics and concurrent work.
- Final optimizer/sampler states are saved; the CLI starts new runs and does not
  yet expose a resume command. Older checkpoints and experiment outputs are
  preserved, including the seven historical ignored models absent at baseline.

No push or merge is performed by these commands.
