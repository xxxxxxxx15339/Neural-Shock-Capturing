# Neural-Shock-Capturing

![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Python 3.7 | 3.11](https://img.shields.io/badge/python-3.7%20%7C%203.11-blue.svg)

Shock-capturing PDE solvers (WENO/TENO) augmented with neural networks, plus a
from-scratch differentiable (JAX) rewrite built to test training them with an
**unrolled physics loss** instead of imitation.

## Table of contents

- [Overview](#overview)
- [Key results](#key-results)
- [Methodology: the Unrolled Physics Loss](#methodology-the-unrolled-physics-loss)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Shared configuration](#shared-configuration)
  - [Training WENO5-NN](#training-weno5-nn)
  - [Running the comparisons](#running-the-comparisons)
  - [Canonical evaluation](#canonical-evaluation)
  - [Training-data format](#training-data-format)
  - [WENO3-NN and WENO7-NN status](#weno3-nn-and-weno7-nn-status)
  - [TENO and NN-TENO status](#teno-and-nn-teno-status)
  - [Unrolled Physics Loss: running the JAX scripts](#unrolled-physics-loss-running-the-jax-scripts)
- [Testing](#testing)
- [Documentation](#documentation)
- [Attribution](#attribution)
- [License](#license)

## Overview

This repository is built on top of the code accompanying Ben Stevens and Tim
Colonius, [*"Enhancement of shock-capturing methods via machine
learning"*](https://doi.org/10.1007/s00162-020-00531-1), and now spans two
bodies of work:

1. **The original WENO5-NN reproduction and TENO5/NN-TENO5 extension** —
   classical and neural-network-augmented WENO/TENO schemes trained by
   *imitation* (the network learns to reproduce a classical scheme's own
   decision), evaluated on scalar and Euler problems. Ships with the paper's
   original five-point training data and reproduces its three WENO5/WENO5-NN
   test cases (linear advection Sec. 4.1, Burgers' equation Sec. 4.2, and the
   Shu–Osher problem Sec. 4.3) at the paper's own configurations, plus a
   TENO5/NN-TENO5 extension the paper doesn't cover. Classical ENO3, WENO3,
   WENO5, and WENO7 reconstruction modules are also present, but the included
   dataset is specifically five-point.
2. **A from-scratch differentiable (JAX) implementation** testing an
   **Unrolled Physics Loss** proposal: train NN-TENO5 by running it through
   several actual simulation steps and judging the *outcome* against the real
   solution, instead of imitating a classical scheme one stencil at a time.
   This needed new infrastructure the original codebase couldn't provide
   (nothing here was previously differentiable end-to-end) and lives in
   `src/differentiable/`.

## Key results

Stated as plainly as the underlying numbers allow — negative and partial
results are reported here just as visibly as the positive one.

| Strand | Result |
|---|---|
| WENO5-NN vs. WENO5 (paper reproduction) | Reproduces the paper's claims: clearest win on step advection (45% lower error); Sod and Shu–Osher near parity; Burgers genuinely mixed (3/4 grid resolutions favor the network). |
| NN-TENO5 (imitation-trained) vs. TENO5 | By construction cannot exceed its own teacher's quality — confirmed directly: wins on roughly a quarter to a third of benchmarks, ties on about half, loses the rest. |
| NN-TENO5 (Unrolled Physics Loss) — scalar advection | **Genuine win** on held-out tests (narrowly, and only within roughly the training distribution's shape/time-horizon). Loses to TENO5 on canonical, non-held-out-similar benchmarks (full-scale step profile, out-of-distribution Burgers) — the win is real but narrow, not a general-purpose improvement. |
| NN-TENO5 (Unrolled Physics Loss) — Euler (systems) | Hit a real, diagnosed numerical-stability obstacle (Roe-eigenbasis conditioning under an evolving network reconstruction) — improved by two independent fixes but **not fully resolved**. |

Full numbered experiment history: [`Documentation/Results.md`](Documentation/Results.md).
Earlier consolidated write-up of strand 1: [`Documentation/Report_Results.md`](Documentation/Report_Results.md).

## Methodology: the Unrolled Physics Loss

The Unrolled Physics Loss proposal was put forward by **Radouan BOUKHARFANE**
(internship supervisor) and implemented, tested, and critically evaluated by
**Yasser BAOUZIL** (this repository). It proposes changing *how* NN-TENO5 is
trained: instead of imitating TENO5's own candidate-selection decision one
stencil at a time — the original training method, which by construction can
never exceed its own teacher's quality — train the network by unrolling it
through K actual simulation steps and judging the resulting trajectory
against the real solution. The full plan is
[`references/Unrolled_Physics_Loss_Proposal.pdf`](references/Unrolled_Physics_Loss_Proposal.pdf)
(French).

**The loss function actually implemented** (`src/differentiable/losses_jax.py`):

```text
L_total = λ1·L_MSE + λ2·L_TV + λ3·L_BVD + λ4·L_Reg

L_MSE = (1/N) Σ (u_i^(n+K) − u_i,exact^(n+K))²
L_TV  = max(0, TV(u^(n+K)) − TV(u^n)),  TV(v) = Σ |v_(i+1) − v_i|
L_BVD = mean(max(0, BV_network − BV_TENO5)),
        BV(Φ)_(i+1/2) = |Φ_i(x_(i+1/2)) − Φ_(i+1)(x_(i+1/2))|
L_Reg = ‖c_NN − c_base‖²
```

- `L_MSE`, `L_TV`, `L_Reg` — direct from the proposal's own equations.
- `L_BVD`'s formula (Boundary Variation Diminishing — compare each cell's own
  reconstruction against its *neighbor's* own reconstruction at their shared
  interface) was missing from the proposal itself; Prof. Boukharfane pointed
  to the source directly — Sun, Inaba & Xiao, *"Boundary Variation Diminishing
  (BVD) reconstruction,"* J. Comput. Phys. 322 (2016) — and the implementation
  was corrected to match it exactly.
- The scheme's own reconstruction weights (`c̃_WENO/TENO` in the proposal) are
  TENO5's classical optimal linear weights `[1/10, 3/5, 3/10]`, from Shu's
  foundational WENO paper — confirmed by Prof. Boukharfane to be exactly what
  was meant.
- `L_Reg`'s anchor mechanism reuses `Train_NNTENO5.py`'s own `--anchor-weight`
  technique, reframed as one term instead of the whole objective.

**The path taken, briefly** (full account with numbers in
[`Documentation/Results.md`](Documentation/Results.md)):

1. Training against the proposal's equations literally (raw, unnormalized
   `L_TV`/`L_Reg`) let `L_TV` dominate the gradient ~4x over `L_MSE` — fixed
   by normalizing both.
2. Training a randomly-initialized network directly against the unrolled loss
   failed outright (lost every held-out test). Fixed by imitation-pretraining
   first, then fine-tuning with the unrolled loss — the same
   warm-start-then-anchor pattern already validated for imitation training.
3. That fix produced a genuine, positive result for scalar advection (2 of 3
   held-out cases beat TENO5), but it did not generalize to full-scale
   benchmark problems and did not transfer to Euler at all — a real numerical
   instability, traced to Roe-eigenbasis conditioning under an evolving
   network.
4. Applying Prof. Boukharfane's own clarifications (the correct `L_BVD`
   definition; repairing isolated positivity violations by
   neighbor-averaging rather than a blind clamp) measurably improved Euler
   training stability — but did not fully resolve it.

Every claim above is backed by an actual run, not asserted — see
[`Documentation/Results.md`](Documentation/Results.md) for the numbers and
[`Documentation/Literature_Report_Mapping.md`](Documentation/Literature_Report_Mapping.md)
for the literature each design choice is grounded in.

## Repository structure

```text
data/                   Training stencils and target fluxes (WENO5-NN, TENO5-NN, oracle)
Documentation/          Experiment history, literature mapping, and the Unrolled Physics Loss plan (see below)
models/                 Trained checkpoints, training histories, metadata (Keras .h5 and JAX .pkl)
notes/                  Background notes on cell averages and flux, read before src/core/
plots/                  Generated figures for the Unrolled Physics Loss results, plus the script that builds them
references/             Reference/plan PDFs (WENO-NN paper, TENO_ML paper, Unrolled_Physics_Loss_Proposal.pdf)
scripts/legacy_imitation_training/   Original WENO/TENO/imitation-training pipeline
                        (Train_*, GenerateTrainingData_*, *_vs_* scripts)
scripts/unrolled_physics_loss/       Unrolled Physics Loss/JAX work (DiffSpike_*,
                        Train_*_UnrolledJAX*, ValidationMatrix_ProjPDF.py,
                        EulerInstability_TraceCapture.py, PDEDataset_NonIdentifiability_Audit.py)
src/core/               Simulation, time stepping, flux splitting, equations (NumPy)
src/data/               Feature-vector construction for the Unrolled Physics Loss proposal (framework-free, runs anywhere)
src/differentiable/     JAX ports of the above, for the Unrolled Physics Loss work -- see
                        "Unrolled Physics Loss" above
src/initial_conditions/ Initial conditions for scalar and Euler problems
src/networks/           WENO-NN model definitions and data loading (Keras)
src/schemes/            ENO, WENO, and TENO reconstruction schemes (NumPy)
src/viz/                Plotting, metrics, and analysis helpers
tests/                  Numerical, data-pipeline, and JAX regression tests
```

Scripts are grouped by which line of work they belong to, not left flat at
the repo root. Each script still runs standalone: it resolves its own
imports and its `data/`, `models/`, and `plots/` I/O relative to the repo
root regardless of the working directory it's launched from.

## Getting started

### Prerequisites

This project spans two incompatible dependency stacks that must live in
**separate environments** — the original model relies on a legacy Keras/TF
pin that cannot coexist with a modern JAX install:

| Environment | Needed for | Python |
|---|---|---|
| `weno-nn` | Training/running the original WENO5-NN and NN-TENO5 (Keras) pipeline | 3.7 |
| `teno-jax` | The differentiable JAX Unrolled Physics Loss work | 3.11 |
| *(any modern Python)* | Numerical-only work — regenerating datasets, running the deterministic solver, non-neural tests | any |

### Installation

**`weno-nn`** (legacy Keras stack — Python 3.7, TensorFlow 1.15.0, Keras 2.2.4;
declared in `.python-version` as `3.7.12`; using modern `tensorflow.keras`
changes training behavior and produces different numerical results):

```bash
conda create -n weno-nn python=3.7
conda activate weno-nn
pip install -r requirements.txt
```

If the TensorFlow 1.15 wheel is unavailable for your platform, install the
pinned packages through Conda instead. CUDA is optional; TensorFlow falls
back to the CPU.

**`teno-jax`** (differentiable JAX work):

```bash
conda create -n teno-jax python=3.11
conda activate teno-jax
pip install -r requirements-jax.txt
```

**Numerical-only** (regenerating datasets, running the deterministic solver —
no Keras, no JAX):

```bash
pip install -r requirements-numerical.txt
```

## Usage

### Shared configuration

Training and inference use [`src/config.py`](src/config.py):

```python
WENO_ORDER = 5
USE_SCALING = True
MODEL_PATH = 'trained_weno5.h5'
INPUT_DATA_PATH = 'data/2ndNewAvgs.csv'
OUTPUT_DATA_PATH = 'data/2ndNewFlux.csv'
RANDOM_SEED = 42
BOUNDARY_CONDITION = 'periodic'
MAX_CFL = 1.0
```

Keep `USE_SCALING` unchanged between training and inference. When enabled,
each stencil and its target are normalized using the stencil's local minimum
and range. The prediction is converted back to physical units during
simulation.

### Training WENO5-NN

```bash
conda activate weno-nn
python scripts/legacy_imitation_training/Train_WENO5NN_Scalar.py --seed 42
```

The dataset is divided into contiguous training, validation, and untouched
test sets. Each attempt creates and trains a freshly initialized model. Early
stopping monitors validation loss.

After every attempt, the model is tested on periodic step advection using:

- `max_tv`: maximum total variation; required to be below `2.016` by default.
- `shock_width`: maximum number of cells whose error exceeds the tolerance;
  required to be at most `20` by default.

Training makes at most 10 attempts by default (`--max-attempts`). It ranks
candidates lexicographically by total-variation-limit violation,
shock-width-limit violation, and validation loss. Every improvement is
checkpointed to `<model-path>.best.h5`; if no candidate passes, the script
prints the best candidate and exits unsuccessfully. A successful model is
saved to `MODEL_PATH`, normally `trained_weno5.h5` — the one canonical
trained-model filename used by training, evaluation, and the basic
comparison. The included `models/SeemsGood3.h5` is retained only as an
upstream reference artifact; new training does not overwrite or select it
automatically.

Training options can be inspected with:

```bash
python scripts/legacy_imitation_training/Train_WENO5NN_Scalar.py --help
```

### Running the comparisons

After training:

```bash
python scripts/legacy_imitation_training/WENO5_vs_WENO5NN_StepAdvection.py
```

This compares the scaled or unscaled WENO-NN model — according to
`USE_SCALING` — against classical WENO5. It matches the paper's Sec. 4.1 /
Fig. 3 advection configuration: a unit step on a periodic domain of length
`L=2`, 100 cells, `CFL=2/3`, run for 50 periods (`T=100`). At `T=100` this is
roughly 7500 timesteps; because `NNMethod` calls Keras `model.predict()` once
per Runge-Kutta stage rather than batching across time, the run takes a few
minutes.

To run the original supplied model instead:

```bash
python scripts/legacy_imitation_training/WENO5_vs_WENO5NN_StepAdvection.py --model-path models/SeemsGood3.h5
```

The scalar solver currently supports periodic boundary conditions. CFL
validation rejects configurations exceeding the configured limit.

**Burgers' equation:**

```bash
python scripts/legacy_imitation_training/WENO5_vs_WENO5NN_Burgers.py --model-path models/SeemsGood3.h5
```

Matches the paper's Sec. 4.2 / Fig. 10 configuration: a Gaussian initial
condition `exp(-k(x-L/2)^2)` with `k=20` on a periodic domain of length
`L=2`, run to `T=4`, comparing WENO5-JS and WENO-NN L2 error against a fine
WENO5-JS reference at several coarser grid spacings. The paper's reference
resolution is `dx=3.125e-4` (6400 cells); because the scalar `Simulation`
class steps on a fixed uniform time grid rather than adaptive CFL stepping,
a 6400-cell run takes on the order of 10+ minutes, so the default reference
here is reduced to 1600 cells for tractable runtime. Pass
`--reference-cells 6400` for the paper's exact reference. Consistent with the
paper (which reports the two schemes performing similarly for true shocks),
WENO-NN and WENO5-JS come out close to each other here rather than one
clearly beating the other.

**Shu–Osher (Euler):**

```bash
python scripts/legacy_imitation_training/WENO5_vs_WENO5NN_ShuOsher.py --model-path models/SeemsGood3.h5
```

Matches the paper's Sec. 4.3 / Figs. 11-12 configuration: 300 cells, `T=2`,
compared against a fine WENO5-JS reference (the paper uses 12,800 cells; the
default here is 3200 for tractable runtime — pass a larger `--reference-cells`
for closer fidelity). Reproducing this configuration with `SeemsGood3.h5`
recovers the paper's claimed result: roughly a 15% reduction in density L1
error and visibly less-diffused oscillations in the post-shock turbulent
region, versus classical WENO5-JS. Earlier default settings (120 cells,
`T=1.8`, WENO7 reference) evaluated the model in a resolution regime the
paper never demonstrates and did not show this improvement — the paper
itself notes the benefit is resolution-sensitive, present around 250-300
cells and degrading toward excess anti-diffusion by 800 cells. The
`shuOsher()` initial condition (`src/initial_conditions/InitialConditions.py`)
was also corrected to match the paper's Eq. 26 exactly
(`rho0r = 1 + eps*sin(5*x)`); it previously included an extra `(x-5)` phase
shift not present in the paper.

The Euler solver uses transmissive boundaries by default. It constructs the
full WENO3, WENO5, or WENO7 stencil at every interface by
constant-extrapolating the first and last conservative states into ghost
cells. Periodic Euler boundaries are also available by passing
`boundary='periodic'` to `FiniteVolumeMethodEuler`.

Characteristic Euler reconstruction uses one Roe-averaged eigenbasis at each
interface. The same left eigenvector matrix projects both positive and
negative Lax–Friedrichs flux stencils, and the matching right matrix
transforms their combined reconstruction back to conservative variables.

Euler benchmarks use adaptive time steps computed from the current maximum
`|velocity| + sound_speed`, never just the initial state. Every accepted step
therefore satisfies the requested CFL, with the final step shortened to land
exactly on the requested final time. Density and pressure are validated
after every SSPRK stage. Flux splitting uses a local interface spectral
radius, the maximum of the adjacent cell wave speeds.

### Canonical evaluation

Evaluate the canonical trained model without plots:

```bash
conda activate weno-nn
python scripts/legacy_imitation_training/Evaluate_WENO5NN_StepAdvection.py --model-path trained_weno5.h5
```

The command prints a JSON result containing the numerical parameters, total
variation, shock width, solution range, and acceptance status.

The canonical benchmark uses 100 periodic cells, 451 time levels, final time
`6`, SSPRK3, Lax–Friedrichs splitting with unit wave speed, and the step
initial condition. A successfully trained model must produce:

```text
max_tv < 2.016
shock_width <= 20
accepted = true
```

Because optimization is stochastic, the exact accepted values vary. As a
verified reference, the included upstream `models/SeemsGood3.h5` evaluated on
July 19, 2026 with this environment produced:

```text
max_tv       = 2.0049578981686045
shock_width  = 22
solution_min = -0.001185931837874278
solution_max = 1.0011859063902506
accepted     = false
```

That reference model is not expected to satisfy the stricter current
`shock_width <= 20` training condition.

### Training-data format

Both files are headerless, comma-separated, decimal floating-point matrices:

- `data/2ndNewAvgs.csv`: shape `(5, 75241)`. Rows are the five ordered
  stencil positions `[u_(i-2), u_(i-1), u_i, u_(i+1), u_(i+2)]`; each column
  is one training sample.
- `data/2ndNewFlux.csv`: shape `(75241, 1)` when loaded by the project CSV
  reader. Row `j` is the scalar target interface value for column `j` of the
  input file.

The input values currently range from `0` to `1`; target values range
approximately from `-0.5` to `1.5`. Input and target files must contain the
same number of samples. With `USE_SCALING=True`, each input column and its
target are transformed using that stencil's local minimum and range before
splitting.

### WENO3-NN and WENO7-NN status

`NNMethod3`/`NNEuler3` (order 3) and `NNMethod7`/`NNEuler7` (order 7)
reconstruction wrappers exist in `src/schemes/`, and
`WENO31stOrder`/`WENO71stOrder` model builders exist in
`src/networks/WenoNetworks.py`, but **no trained model or matching stencil
dataset ships with this repo for either order**. The original paper only
ever trained a single WENO5 network (its 5-point, 75,241-sample
generic-waveform dataset is `data/2ndNewAvgs.csv`/`data/2ndNewFlux.csv`);
there is no order-3 or order-7 equivalent. A faithful WENO3-NN or WENO7-NN
reproduction would require generating an analogous stencil dataset (same
generic-waveform methodology, windowed to 3 or 7 cells instead of 5) and
training a new model from scratch — this has not been done.
`WENO5_vs_WENO5NN_StepAdvection.py`, `WENO5_vs_WENO5NN_Burgers.py`, and
`WENO5_vs_WENO5NN_ShuOsher.py` are WENO5/WENO5-NN only.

### TENO and NN-TENO status

The repository includes deterministic TENO3, TENO5, and TENO7 scalar and
Euler reconstructions. It also includes NENO-style classifier architectures
and scalar/Euler NN-TENO wrappers for all three orders. If every candidate
is rejected, the wrappers use the center/upwind value as a first-order
fallback.

Each nonconstant NN-TENO input stencil is normalized independently as
`(X - min(X)) / (max(X) - min(X))`; constant stencils map to zero. This
preserves the deterministic selector's translation and positive-scale
invariance while keeping classifier inputs in the training range.

Each sigmoid output classifies one candidate stencil: `0` means smooth/keep
and `1` means troubled/discard. The network predicts candidate selection,
not the final flux, candidate polynomial, or reconstruction weights.

Train the NN-TENO5 classifier on the first CUDA GPU with:

```bash
conda activate weno-nn
nvidia-smi
python scripts/legacy_imitation_training/Train_NNTENO5.py --device gpu --gpu-index 0
```

The `gpu` mode fails immediately if TensorFlow cannot access the requested
GPU, rather than silently training on the CPU. TensorFlow uses incremental
GPU-memory allocation, which is suitable for a display-attached GPU. Use
`--device cpu` for an intentional CPU run or `--device auto` to allow
fallback.

The training script generates three deterministic TENO5 troubled/keep labels
from every original five-point stencil, creates disjoint stratified
train/validation/test splits, checkpoints the best validation model,
calibrates one decision threshold per candidate, and writes model, history,
split, and metadata artifacts under `models/`.

Dedicated NN-TENO3/7 training datasets and extensive trained-model PDE
benchmarks are not implemented yet.

### Unrolled Physics Loss: running the JAX scripts

See [Methodology](#methodology-the-unrolled-physics-loss) above for what
this trains and why. `src/differentiable/` is a from-scratch JAX rewrite,
built because nothing in the original codebase (pure NumPy + standalone
Keras) is differentiable end-to-end — `model.predict()` fully detaches from
any gradient graph.

**What's in `src/differentiable/`:**

| Module | What it ports |
|---|---|
| `teno5_jax.py` | TENO5 candidate/smoothness/selection math (`src/schemes/teno5.py`), plus a soft (continuous keep-weight) reconstruction path a network can be trained through |
| `numerics_jax.py` | The scalar finite-volume solver (`src/core/SimulationClasses.py`, `TimeSteppingMethods.py`, `FluxSplittingMethods.py`) — `jax.lax.scan`-based rollout, and the true BVD boundary-variation construction |
| `euler_jax.py` | The full characteristic-wise Euler solver (`src/core/eulerEquations.py`) — Roe eigenbasis, flux splitting, and a positivity-repair mechanism |
| `projection_jax.py` | Converts a network's raw troubled-probability output into a reconstruction weight (soft for training, hard-thresholded for inference) |
| `losses_jax.py` | The four Unrolled Physics Loss terms (`L_MSE`, `L_TV`, `L_BVD`, `L_Reg`) plus a positivity penalty |
| `teno5_jax_toy.py` | A JAX MLP matching NN-TENO5's architecture, used for training (not a port of the trained Keras checkpoint) |

Every port is regression-tested against the real NumPy/Keras-free solver to
float64 precision — see [Testing](#testing) below.

**Key scripts** (run inside `teno-jax`, from the repo root):

```bash
cd scripts/unrolled_physics_loss    # or reference the full relative path from the repo root instead
python DiffSpike_TENO5_Advection.py                       # Phase 0/1 feasibility spike (scalar)
python DiffSpike_TENO5euler_Sod.py                        # Phase 0/1 feasibility spike (Euler)
python Train_NNTENO5_UnrolledJAX.py                       # Phase 6 scalar training (the positive result)
python Train_NNTENO5euler_UnrolledJAX.py                  # Phase 6 Euler training (the unresolved instability)
python Train_NNTENO5euler_UnrolledJAX_Curriculum.py       # Phase 11/12: curriculum-on-K attempt at fixing it
python ValidationMatrix_ProjPDF.py                        # Phase 7: canonical-benchmark validation matrix
python PDEDataset_NonIdentifiability_Audit.py             # Phase 3: measures the 60%-real-PDE-data risk directly
python EulerInstability_TraceCapture.py                   # regenerates the Phase 10 instability trace/plot data
cd ../..
python plots/generate_proj_plots.py                       # regenerates plots/*.png from the CSVs these scripts write
```

Every script writes/reads `models/`, `data/`, and `plots/` by a path
relative to the repo root; each script starts with a small `sys.path`
bootstrap so its own imports resolve correctly regardless of the working
directory. Run it either from the repo root with the full
`scripts/unrolled_physics_loss/...` path, or `cd` into that directory first
— either works.

Trained checkpoints and training histories land in `models/`
(`nn_teno5_unrolled_jax_*`, `nn_teno5euler_unrolled_jax_*`); generated
figures land in `plots/`. Every script's own docstring explains what it does
and why, in more detail than fits here — read the script before running it,
not just this table.

## Testing

Run the regression suite with:

```bash
python -m unittest discover -s tests -v
```

The tests cover constant-state reconstruction, finite periodic advection,
CFL rejection, boundary validation, scaled-data handling, and dataset
separation. They verify the Roe matrices are mutual inverses and establish
smooth periodic-advection convergence for WENO3, WENO5, and WENO7. The Euler
end-to-end matrix runs all three schemes on constant stationary and
uniformly moving states as well as the physically timed Sod and Shu–Osher
regressions described above.

Numerical tests depend only on `requirements-numerical.txt`; they do not
import TensorFlow, so they can run under any modern Python without the
legacy stack. The neural smoke test (`tests/test_networks_smoke.py`) needs
the `weno-nn` environment.

The JAX differentiable-implementation tests (`tests/test_teno5_jax_*.py`,
`tests/test_losses_jax.py`, `tests/test_projection_jax.py`) need the
`teno-jax` environment instead, since they import `jax`:

```bash
conda activate teno-jax
python -m pytest tests/test_teno5_jax_regression.py tests/test_teno5_jax_euler_regression.py tests/test_projection_jax.py tests/test_losses_jax.py -q
```

They regression-test every JAX port against the original NumPy/Keras-free
solver (float64-exact, not approximate) and check the Unrolled Physics Loss
terms are finite with finite gradients. `tests/test_feature_vector.py` (the
feature-vector module for the Unrolled Physics Loss proposal) is plain NumPy
and runs fine in `base` or `weno-nn` alongside the rest of the suite.

## Documentation

All longer-form write-ups live in [`Documentation/`](Documentation/):

| Document | Read it for |
|---|---|
| [`Documentation/Results.md`](Documentation/Results.md) | The full numbered experiment history, chronological |
| [`Documentation/Report_Results.md`](Documentation/Report_Results.md) | The earlier consolidated write-up of the original WENO5-NN/TENO5 comparison work |
| [`Documentation/UNROLLED_PHYSICS_LOSS_GUIDE.md`](Documentation/UNROLLED_PHYSICS_LOSS_GUIDE.md) | The phase-by-phase plan for the Unrolled Physics Loss proposal, with status notes on every phase (what was built, the exit criterion, whether it was met) |
| [`Documentation/Literature_Report_Mapping.md`](Documentation/Literature_Report_Mapping.md) | What the published literature says about each design choice made along the way |
| [`TENO_NN_Course.html`](TENO_NN_Course.html) | The concepts explained from scratch (open directly in a browser) |

Neither `Results.md` nor `UNROLLED_PHYSICS_LOSS_GUIDE.md` hides negative
results — the Euler instability is documented as thoroughly as the scalar
success.

## Attribution

The numerical method and original WENO-NN implementation are derived from
the work of Ben Stevens and Tim Colonius. See the linked paper and retain
the relevant attribution when redistributing derived code.

The Unrolled Physics Loss research direction and its design clarifications
are Radouan BOUKHARFANE's, as internship supervisor; the differentiable JAX
implementation, experiments, and literature review testing that direction
are Yasser BAOUZIL's own contribution.

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
