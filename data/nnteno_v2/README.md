# NN-TENO5 V2: exact reference trajectory dataset

## Status and provenance

**Generated and verified dataset**: 6,144 independent PDE problems; 24,576 four-time reference snapshots; 3,668,480 snapshot cell positions. Data consist exclusively of **analytic mathematical reference solutions**, evaluated as finite-volume cell averages using closed-form expressions or discontinuity-aligned Gauss–Legendre integration. **No physical sensor, camera or experimental measurements are included.** This is intentional: authentic shock-tube traces are not dense snapshots of the 1D ideal-gas Euler conserved-state field.

The source files and all data were generated during this conversation. The independent public sources considered for future external comparison are recorded in `EXTERNAL_SOURCES.md`; none have been redistributed as if they were included in this archive. `build_dataset.py` can regenerate all data using the recorded seed.

## Contents

- `data/scalar_train.h5`, `scalar_validation.h5`, `scalar_test.h5`
- `data/euler_train.h5`, `euler_validation.h5`, `euler_test.h5`
- `data/cases.jsonl`: all problem parameter sets, seeds, families, mesh sizes, wave speeds and per-case QA residuals
- `data/quality_report.json`: counts, per-file SHA256 hashes, conservation residuals, non-overlap checks
- `build_dataset.py`: reproducible exact-reference generator
- `verify_dataset.py`: independently scan all stored arrays, hashes, positivity and flux budgets
- `load_dataset.py`: HDF5 reader and homogeneous-physics minibatch iterator
- `verification_result.json`: saved successful independent audit output
- `EXTERNAL_SOURCES.md`: experimental and published simulation datasets surveyed but not included

## Physics and exactness

**Scalar:** `u_t + a u_x = 0` (constant `a` *per case*, metadata `advection_speed`); exact Fourier, periodic Gaussian and translated rectangular-wave cell averages. Also inviscid `u_t + (u^2/2)_x = 0`: exact pre-shock characteristic sine waves and entropy Riemann solutions for a periodic top-hat (rarefaction plus shock, before interaction). Includes constant-state controls.

**Euler:** `U_t+F(U)_x=0`, `U=[rho,rho*v,E]`, `p=(gamma-1)(E - rho*v^2/2)`, `gamma=1.4`. Random ideal-gas Riemann problems, Sod-like cases, contacts, two-rarefaction candidates and constant states. Wave speeds and star pressure are computed via an exact Riemann solution; cell integrals are split at all shock, contact and rarefaction-wave boundaries before Gauss–Legendre quadrature. All waves stay away from the physical domain boundaries throughout stored times. Transmissive boundaries are appropriate for these cases.

Array reference values represent **cell averages**, not point samples, of the conserved state. Euler quadrature is 16-point Gauss per smooth interval, with sampled 32-point convergence cross-checks. Smooth Burgers uses 32-point Gauss with 16-point sampled cross-checks.

**Known absences**: physical measurements, post-interaction Burgers top hats, generic post-shock smooth Burgers, Shu–Osher reference trajectories, boundary reflections, chemical reactions, viscosity, heat conduction, multidimensional dynamics, noisy sensor measurements. This is a mathematically exact baseline corpus, not a comprehensive CFD benchmark or experimentally calibrated dataset.

## Data shape

In each `physics_split.h5` file, groups `N064`, `N128`, `N256` each contain:

| key | shape | meaning |
|---|---|---|
| `states` | scalar `(B,4,N)`, Euler `(B,4,N,3)` | reference cell-averaged states at 4 times |
| `times` | `(B,4)` | exact output times, beginning at `0` |
| `short_target` | scalar `(B,N)`, Euler `(B,N,3)` | exact state after 15 SSPRK3 substeps-worth of time |
| `short_time`, `short_dt` | `(B,)` | matched horizon and time step, `short_time == 15*short_dt` |
| `advection_speed` | `(B,)` | physical constant `a` for scalar advection; zero otherwise |
| `family`, `case_id` | `(B,)` | case class and immutable reference ID |

The model **must** use the proper PDE flux: `a*u` for scalar advection, `u**2/2` for Burgers, and the Euler ideal-gas flux for Euler. `dt`, `a`, and Euler wave-speed bounds can differ per case. For JAX, use a per-example `vmap` or properly bucket cases by `dt/speed` instead of using one common `dt` for every sample. Hold `steps=15` fixed during a bucketed unrolled training pass.

## Quality audit, 2026-10-15

All six final HDF5 files pass `verify_dataset.py`, validating their exact hash manifest, finite arrays, dimensions, provenance, positivity, per-case CFL bound and conservation. The largest scalar mass residual was approximately `4.44e-16`, and the largest Euler conserved-state flux budget residual approximately `4.00e-14` in the long snapshots. Final shortest-horizon estimated CFL does not exceed `0.35` using the recorded conservative speed bound. Every row has a unique RNG seed and no trajectory appears in another split.

Canonical independent sanity case: the exact Sod problem `(rho,u,p)_L=(1,0,1)`, `(rho,u,p)_R=(0.125,0,0.1)` gives `p_star=0.3031301780506468`, `u_star=0.9274526200489499`, consistent with established exact Sod references.

**This audits internal numerical consistency, not experimental truth.** Random parameters represent mathematical scenarios, not physically measured initial conditions. Inference-to-measurement generalization requires a separate instrument-calibrated validation program.

## Counts

| Physics | train | validation | test | total |
|---|---:|---:|---:|---:|
| Scalar | 2,867 | 615 | 614 | 4,096 |
| Euler | 1,434 | 307 | 307 | 2,048 |
| Total | 4,301 | 922 | 921 | 6,144 |

Three training and evaluation resolutions: N=64,128,256. Family proportions, realized counts and per-file cryptographic SHA256 are in `quality_report.json`.

## Getting started

```bash
python -m pip install numpy scipy h5py
python verify_dataset.py --root data
python load_dataset.py --root data --physics scalar --split train --cells 128
```

To regenerate in a new directory:

```bash
python build_dataset.py --output ./regenerated --seed 20261015 --scalar 4096 --euler 2048
python verify_dataset.py --root ./regenerated
```

To move to your repository, place the archive contents under `data/nnteno_v2/` or point the loader at the extracted `data/` folder. Example:

```python
from load_dataset import load_bucket, iter_batches
bucket = load_bucket("data/nnteno_v2/data", physics="scalar", split="train", cells=128)
for batch in iter_batches(bucket, batch_size=16, family="burgers"):
    initial, target = batch["initial"], batch["target"]
    dt = batch["dt"]   # shape (B,), not a shared scalar
    steps = batch["steps"]  # 15
```

**Integration warning:** The previously delivered `Train_Gated_NNTENO5_V2.py` pilot samples its own training problems. It does **not yet consume this HDF5 dataset**. The new loader is a starting adapter; connect it explicitly to the unrolled training loops and preserve per-case wave speed/`dt` and appropriate boundary conditions.

## Recommended independent experimental evaluation

Use genuinely recorded shock-tube pressure/schlieren datasets *only after* assembling their apparatus geometry, gas composition, sensor position, calibration, sampling rate and wall/boundary conditions. Do not use measured pressure at one sensor as if it were a complete Euler state vector. See `EXTERNAL_SOURCES.md`.
