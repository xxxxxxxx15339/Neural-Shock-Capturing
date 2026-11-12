# Reproduction commands

From the repository root, regenerate tables, checks and figures using the
identity-checked saved trajectories:

```bash
bash experiments/four_method_comparison/reproduce.sh
```

To independently regenerate the TensorFlow/PyTorch parity fixtures too:

```bash
REEXPORT=1 bash experiments/four_method_comparison/reproduce.sh
```

The script declares the exact numerical, legacy Keras and PyTorch Python paths.
Override `TENO_PY`, `LEGACY_PY` and `TORCH_PY` for another machine. Exporting uses
read-only model loads and no training. Saved original models/results are untouched.
The script preserves existing raw benchmark caches after checking manifest and
checkpoint identities; the absence of console run lines means a verified cache
was used, not that a new PDE simulation was executed.

For a genuinely fresh numerical recomputation, without deleting or overwriting
any saved trajectory, use a new empty output directory:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/four-method-mpl \
  /home/bazouil/miniconda3/envs/teno-jax/bin/python \
  experiments/four_method_comparison/scripts/run_matched_benchmarks.py all \
  --fresh-raw-output experiments/four_method_comparison/results/fresh_full_reproduction
```

The fresh-output option refuses a nonempty directory. It uses the frozen protocol
and the same authentic exported weights. It writes a complete new set of raw
arrays and numerical diagnostics; it does not silently pool repeated runs into
the presentation statistics.

The independent Sod subset was actually executed with:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp/four-method-mpl \
  /home/bazouil/miniconda3/envs/teno-jax/bin/python \
  experiments/four_method_comparison/scripts/run_matched_benchmarks.py benchmarks \
  --cases sod \
  --fresh-raw-output experiments/four_method_comparison/results/fresh_reproduction_sod
```

Its numerical states and failure stages are compared with the presentation runs
by `scripts/verify_package.py`; results are in `results/fresh_reproduction_check.csv`.
Use another empty directory name to repeat it.

Executed regression commands (environment exports as in `reproduce.sh`):

```bash
/home/bazouil/miniconda3/envs/teno-jax/bin/python -m pytest \
  tests src/differentiable/tests --ignore=tests/test_networks_smoke.py -q
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
  /home/bazouil/miniconda3/envs/weno-nn/bin/python -m pytest \
  tests/test_networks_smoke.py -q
```

The common JAX benchmark, reference ladder, supplemental wall/dt controls,
metrics, figure generation and artifact verification are executable as separate
scripts. Exact invocations and output paths are collected in `reproduce.sh`.
The package checksums exclude their own file and the actively written verifier /
end-to-end stdout logs to avoid self-referential hashes. Numerical data, source,
reports, figures and completed test logs are covered.
