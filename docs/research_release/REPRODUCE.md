# CPU reproduction and experiment provenance

Use Python 3.11 and `requirements-v2.txt` in a fresh environment. The recorded
execution environment is JAX/jaxlib 0.10.2, NumPy 2.4.6, SciPy 1.17.1,
Optax 0.2.8, h5py 3.16.0 and Matplotlib 3.11.1. The existing environment was
used for these runs; a new clean V2 installation was not independently exercised.
P0 clean-environment evidence remains separately archived.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-v2.txt
export JAX_PLATFORMS=cpu
export MPLCONFIGDIR=/tmp/nn-teno5-matplotlib
```

Run commands from the repository root. All numerical output directories must be
new. Checkpoints and frozen P0 directories cannot be evaluation output targets.
No command below retrains, pushes, merges, or publishes anything.

The unified entry point is `python -m src.differentiable.research ACTION`.
Actions `sweep`, `benchmark`, `convergence`, `inputs`, `repair`, `performance`
and `verify` dispatch to the modules below. `p0` retains the historical CLI.
Use `ACTION --help` for each action's options.

The benchmark CLI also accepts repeated `--method` names, `--strength` and
`--checkpoint-source v2|p0` for explicit matched ablations. TENO5 is always
included. For example, `benchmark --manifest configs/v2/independent-smoke.json
--method gated-soft --method neural-soft --strength .25 --output NEW` uses V2
parameters. Historical saved heads require `--checkpoint-source p0`; applying
a P0 head behind a new gate is an inference ablation, **not** a separately
V2-trained learned-strength result. Missing heads fail rather than being invented.

## Regression and smoke

```bash
python -m pytest tests/test_core.py tests/test_teno.py tests/test_feature_vector.py tests/test_teno5_jax_regression.py tests/test_teno5_jax_euler_regression.py tests/test_projection_jax.py tests/test_losses_jax.py src/differentiable/tests -q
python -m src.differentiable.v2_evaluate sweep --config configs/v2/smoke.json --output /tmp/v2-smoke-new
python -m src.differentiable.v2_benchmark --manifest configs/v2/independent-smoke.json --output /tmp/v2-benchmark-smoke-new
python -m src.differentiable.v2_verify --protected-only
```

The CI workflow runs the numerical contracts and checks the 491 **tracked**
protected files. The local full preservation check includes seven additional
ignored historical `.best.h5` checkpoints, for 498 protected files. No ignored
checkpoint was added to Git merely to make a CI check pass. Legacy network smoke
tests require their separate Python 3.7 TensorFlow/Keras environment.

## Explicit full scientific workflows

```bash
python -m src.differentiable.v2_evaluate sweep --config configs/v2/full.json --output /tmp/v2-validation-new
python -m src.differentiable.v2_reports --source /tmp/v2-validation-new --output /tmp/v2-statistics-new
python -m src.differentiable.v2_studies convergence --output /tmp/v2-convergence-new
python -m src.differentiable.research spatial --output /tmp/v2-selected-spatial-new
python -m src.differentiable.research convergence --temporal-max-cells 160 --selected-strength --output /tmp/v2-temporal-cpu-new
python -m src.differentiable.v2_studies inputs --output /tmp/v2-inputs-new
python -m src.differentiable.v2_studies repair --output /tmp/v2-repair-new
python -m src.differentiable.v2_benchmark --manifest configs/v2/independent.json --output /tmp/v2-independent-new
python -m src.differentiable.research benchmark --manifest configs/v2/wall-closure-smoke.json --output /tmp/v2-wall-smoke-new
python -m src.differentiable.research benchmark --manifest configs/v2/wall-closure.json --output /tmp/v2-wall-closed-new
```

The convergence study includes semidiscrete errors at 40/80/160/320/640 cells,
smooth critical points, matched final time 0.05, and dt/dt/2 controls with
dt proportional to dx^(5/3). Fine neural Euler rollouts are expensive on CPU.
The completed resource-limited time controls use `--temporal-max-cells 160
--selected-strength`; the original fine temporal protocol through 640 and the
2048-cell blast temporal reference were stopped and remain explicitly incomplete.
The closed-wall blast smoke exposes neural inadmissibility before t=.001, so
the full-time closed-wall manifest is stability-only and produces no accuracy
reference. Its reference-grid list records the abandoned prospective ladder;
`stability_only` and `reference_limit` are the operative fields.
The general nonlinear smooth Euler diagnostic can be reproduced independently:

```bash
python - <<'PY'
import json
from pathlib import Path
from src.differentiable.artifacts import create_output, finish_manifest, write_json, provenance
from src.differentiable.model_io import load_checkpoint
from src.differentiable.v2_nonlinear import run
config = json.loads(Path('configs/v2/full.json').read_text())
output = create_output('/tmp/v2-nonlinear-euler-new')
write_json(output/'provenance.json', provenance('configs/v2/full.json'))
run({p: load_checkpoint(v['path'], v['sha256']) for p, v in config['models'].items()}, output)
finish_manifest(output)
PY
```

For isolated timing, first let other numerical workers finish:

```bash
python -m src.differentiable.v2_performance --output /tmp/v2-performance-new
```

Timing separates lowering, compilation, first execution and nine synchronized
warmed calls. The uninstrumented production rollout is timed. RSS is a process
high-water mark; executable temporary memory is an estimate, not a measured GPU
peak. Equal-budget conclusions apply only to the measured grid/IVP combinations.

## Methods and statistical definitions

`reconstruction.make_reconstruction` is shared by evaluation and studies.
Historical names retain their hard-mask policies. V2 `gated-soft` and
`gated-hard` use deterministic TENO gating and fixed strength. `neural-soft`
and `neural-hard` bypass that gate. `fixed-blend`, `learned-blend` and
`max-blend` retain historical meanings. `gated-learned-soft/hard` require an
explicit saved head; none is synthesized. The optional finite-volume Euler
state-flux solver is a separately identified numerical variant, not a checkpoint.

Full-validation runs use **every** validation trajectory and each case's own dt,
speed and target time, at exactly 15 steps. Euler is transmissive; scalar is
periodic. Each strength sees identical cases and time grids. L1 is mean absolute
error; L2 is RMS; Linf is the maximum absolute error. Euler aggregate errors use
unscaled conserved components with different physical units, as explicitly
documented; component norms are also available.

Macro relative L1 is the mean family×resolution ratio. Weighted relative L1 is
the ratio of trajectory mean errors. Constants and baseline errors ≤1e-10 are
excluded from ratios. Absolute statistics include constants. Bootstrap sampling
is paired and stratified by family×resolution, with seed 20261020 and 2,000
replicates. Intervals condition on valid cases; failures remain counted and
disqualify a setting from selection. Intervals do not correct for hyperparameter
selection or quantify reference/training-seed uncertainty.

`configs/v2/selected.json` freezes the zero-failure minimum-macro-L1 policy before
prospective final evaluation. The final manifest contains 11 IVPs evaluated at
two resolutions each, including four newly seeded random Riemann IVPs. The two
resolutions of an IVP are not independent trajectories. Final results never
retune strength or model weights.

Numerical Burgers/Shu–Osher/blast references use three grids, conservative
cell-average restriction and a finest-grid dt/2 run. Twice the last spatial plus
temporal Cauchy changes is an empirical uncertainty proxy, not an error bound.
A comparison is only marked resolved when the proxy is below 25% of its margin
and both runs have valid raw stages. A failed reference remains failed. The
historical P0 Shu–Osher point samples cannot substitute for these cell averages;
its reference-uncertainty study is retained in Documentation/P0_Validation.md.

## Euler input ablation and training decision

The input study uses training data only: four trajectories per family per grid,
initial states for warm-up, four reference times for solver split fluxes. It
compares normalized shapes, teacher rejection labels, network BCE and hard
decisions. The distributions differ, but snapshot diagnostics do not establish
that retraining improves numerical outcomes. Full-validation strength calibration
is the supported targeted improvement; no new training was justified as a claim
of completed improvement in this release.

The historical warm-up remains the default. To run a **future controlled ablation**:

```bash
python scripts/unrolled_physics_loss/Train_Gated_NNTENO5_Dataset_V2.py --physics euler --pretraining-inputs state --seed 3101 --run-name euler_state_seed3101 --output-dir /tmp/v2-training-new
python scripts/unrolled_physics_loss/Train_Gated_NNTENO5_Dataset_V2.py --physics euler --pretraining-inputs split-flux --seed 3101 --run-name euler_splitflux_seed3101 --output-dir /tmp/v2-training-new
```

These are prospective training commands, **not completed experiments**. Repeat
with independent seeds and matched budgets; full-validation checkpoint selection,
optimizer/RNG resumption and improved monitoring need additional engineering
before claiming a controlled multi-seed training release. Do not reuse the
already examined original test split or the final benchmark for selection.

## Plots and preservation

The portfolio generator reads the committed artifacts in
`docs/research_release/results/`; it executes no solver:

```bash
python scripts/unrolled_physics_loss/Build_V2_Portfolio.py --output /tmp/v2-portfolio-new
python -m src.differentiable.v2_verify --output /tmp/v2-artifact-check-new.json
```

Profiles, plots, captions, source identities and manifests accompany measured
results. Missing numerical results are labeled incomplete; no synthetic curve
substitutes for an unfinished study.

`data/nnteno_v2/data/` is the recommended canonical corpus. Eight files in
`data/data/` are byte-identical, including metadata; the baseline manifest records
their hashes and sizes. Keep the old layout compatible for now. A later migration
can introduce a documented path resolver or redirect and verify all consumers
before removing duplicate storage. No data deletion or Git-history rewrite is
part of this release.
