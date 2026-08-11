# Shocks-and-WENO

Structured implementation of WENO and WENO-NN shock-capturing methods, based on the code accompanying Ben Stevens and Tim Colonius, [“Enhancement of shock-capturing methods via machine learning”](https://doi.org/10.1007/s00162-020-00531-1).

The repository currently ships the original five-point training data and focuses on the WENO5-NN advection experiment. Classical ENO3, WENO3, WENO5, and WENO7 reconstruction modules are also present, but the included dataset is specifically a five-point dataset.

`WENO5_vs_WENO5NN_StepAdvection.py`, `WENO5_vs_WENO5NN_Burgers.py`, and `WENO5_vs_WENO5NN_ShuOsher.py` reproduce the paper's three WENO5/WENO5-NN test cases (linear advection Sec. 4.1, Burgers' equation Sec. 4.2, and the Shu–Osher problem Sec. 4.3 respectively) at the paper's own configurations. See "Running the comparison" below for the exact parameters and known deviations (mainly reduced reference-grid resolution, for tractable runtime).

## Environment

The original model and training behavior rely on the legacy standalone Keras stack:

- Python 3.7
- TensorFlow 1.15.0
- Keras 2.2.4

The supported interpreter version is declared in `.python-version` as Python `3.7.12`. Other Python versions are not supported because TensorFlow 1.15 and Keras 2.2.4 are legacy dependencies.

Using modern `tensorflow.keras` changes training behavior and can produce substantially different numerical results.

Create and activate a Python 3.7 environment, then install the dependencies:

```bash
conda create -n weno-nn python=3.7
conda activate weno-nn
pip install -r requirements.txt
```

If the TensorFlow 1.15 wheel is unavailable for your platform, install the pinned packages through Conda instead. CUDA is optional; TensorFlow falls back to the CPU.

## Shared configuration

Training and inference use [src/config.py](src/config.py):

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

Keep `USE_SCALING` unchanged between training and inference. When enabled, each stencil and its target are normalized using the stencil’s local minimum and range. The prediction is converted back to physical units during simulation.

## Training

Run:

```bash
conda activate weno-nn
python Train_WENO5NN_Scalar.py --seed 42
```

The dataset is divided into contiguous training, validation, and untouched test sets. Each attempt creates and trains a freshly initialized model. Early stopping monitors validation loss.

After every attempt, the model is tested on periodic step advection using:

- `max_tv`: maximum total variation; required to be below `2.016` by default.
- `shock_width`: maximum number of cells whose error exceeds the tolerance; required to be at most `20` by default.

Training makes at most 10 attempts by default (`--max-attempts`). It ranks candidates lexicographically by total-variation-limit violation, shock-width-limit violation, and validation loss. Every improvement is checkpointed to `<model-path>.best.h5`; if no candidate passes, the script prints the best candidate and exits unsuccessfully. A successful model is saved to `MODEL_PATH`, normally:

```text
trained_weno5.h5
```

This is the one canonical trained-model filename used by training, evaluation, and the basic comparison. The included `SeemsGood3.h5` is retained only as an upstream reference artifact; new training does not overwrite or select it automatically.

Training options can be inspected with:

```bash
python Train_WENO5NN_Scalar.py --help
```

## Running the comparison

After training:

```bash
python WENO5_vs_WENO5NN_StepAdvection.py
```

This compares the scaled or unscaled WENO-NN model—according to `USE_SCALING`—against classical WENO5. It matches the paper's Sec. 4.1 / Fig. 3 advection configuration: a unit step on a periodic domain of length `L=2`, 100 cells, `CFL=2/3`, run for 50 periods (`T=100`). At `T=100` this is roughly 7500 timesteps; because `NNMethod` calls Keras `model.predict()` once per Runge-Kutta stage rather than batching across time, the run takes a few minutes.

To run the original supplied model instead:

```bash
python WENO5_vs_WENO5NN_StepAdvection.py --model-path SeemsGood3.h5
```

The scalar solver currently supports periodic boundary conditions. CFL validation rejects configurations exceeding the configured limit.

### Burgers' equation

```bash
python WENO5_vs_WENO5NN_Burgers.py --model-path SeemsGood3.h5
```

Matches the paper's Sec. 4.2 / Fig. 10 configuration: a Gaussian initial condition `exp(-k(x-L/2)^2)` with `k=20` on a periodic domain of length `L=2`, run to `T=4`, comparing WENO5-JS and WENO-NN L2 error against a fine WENO5-JS reference at several coarser grid spacings. The paper's reference resolution is `dx=3.125e-4` (6400 cells); because the scalar `Simulation` class steps on a fixed uniform time grid rather than adaptive CFL stepping, a 6400-cell run takes on the order of 10+ minutes, so the default reference here is reduced to 1600 cells for tractable runtime. Pass `--reference-cells 6400` for the paper's exact reference. Consistent with the paper (which reports the two schemes performing similarly for true shocks), WENO-NN and WENO5-JS come out close to each other here rather than one clearly beating the other.

### Shu–Osher (Euler)

```bash
python WENO5_vs_WENO5NN_ShuOsher.py --model-path SeemsGood3.h5
```

Matches the paper's Sec. 4.3 / Figs. 11-12 configuration: 300 cells, `T=2`, compared against a fine WENO5-JS reference (the paper uses 12,800 cells; the default here is 3200 for tractable runtime — pass a larger `--reference-cells` for closer fidelity). Reproducing this configuration with `SeemsGood3.h5` recovers the paper's claimed result: roughly a 15% reduction in density L1 error and visibly less-diffused oscillations in the post-shock turbulent region, versus classical WENO5-JS. Earlier default settings (120 cells, `T=1.8`, WENO7 reference) evaluated the model in a resolution regime the paper never demonstrates and did not show this improvement — the paper itself notes the benefit is resolution-sensitive, present around 250-300 cells and degrading toward excess anti-diffusion by 800 cells. The `shuOsher()` initial condition (`src/initial_conditions/InitialConditions.py`) was also corrected to match the paper's Eq. 26 exactly (`rho0r = 1 + eps*sin(5*x)`); it previously included an extra `(x-5)` phase shift not present in the paper.

The Euler solver uses transmissive boundaries by default. It constructs the full WENO3, WENO5, or WENO7 stencil at every interface by constant-extrapolating the first and last conservative states into ghost cells. This replaces the former WENO5-specific one-sided closure. Periodic Euler boundaries are also available by passing `boundary='periodic'` to `FiniteVolumeMethodEuler`.

Characteristic Euler reconstruction uses one Roe-averaged eigenbasis at each interface. The same left eigenvector matrix projects both positive and negative Lax–Friedrichs flux stencils, and the matching right matrix transforms their combined reconstruction back to conservative variables.

Euler benchmarks use adaptive time steps computed from the current maximum `|velocity| + sound_speed`, never just the initial state. Every accepted step therefore satisfies the requested CFL, with the final step shortened to land exactly on the requested final time. Density and pressure are validated after every SSPRK stage. Flux splitting uses a local interface spectral radius, the maximum of the adjacent cell wave speeds.

## Canonical evaluation

Evaluate the canonical trained model without plots:

```bash
conda activate weno-nn
python Evaluate_WENO5NN_StepAdvection.py --model-path trained_weno5.h5
```

The command prints a JSON result containing the numerical parameters, total variation, shock width, solution range, and acceptance status.

The canonical benchmark uses 100 periodic cells, 451 time levels, final time `6`, SSPRK3, Lax–Friedrichs splitting with unit wave speed, and the step initial condition. A successfully trained model must produce:

```text
max_tv < 2.016
shock_width <= 20
accepted = true
```

Because optimization is stochastic, the exact accepted values vary. As a verified reference, the included upstream `SeemsGood3.h5` evaluated on July 19, 2026 with this environment produced:

```text
max_tv       = 2.0049578981686045
shock_width  = 22
solution_min = -0.001185931837874278
solution_max = 1.0011859063902506
accepted     = false
```

That reference model is not expected to satisfy the stricter current `shock_width <= 20` training condition.

## Training-data format

Both files are headerless, comma-separated, decimal floating-point matrices:

- `data/2ndNewAvgs.csv`: shape `(5, 75241)`. Rows are the five ordered stencil positions `[u_(i-2), u_(i-1), u_i, u_(i+1), u_(i+2)]`; each column is one training sample.
- `data/2ndNewFlux.csv`: shape `(75241, 1)` when loaded by the project CSV reader. Row `j` is the scalar target interface value for column `j` of the input file.

The input values currently range from `0` to `1`; target values range approximately from `-0.5` to `1.5`. Input and target files must contain the same number of samples. With `USE_SCALING=True`, each input column and its target are transformed using that stencil’s local minimum and range before splitting.

## WENO3-NN and WENO7-NN status

`NNMethod3`/`NNEuler3` (order 3) and `NNMethod7`/`NNEuler7` (order 7) reconstruction wrappers exist in `src/schemes/`, and `WENO31stOrder`/`WENO71stOrder` model builders exist in `src/networks/WenoNetworks.py`, but **no trained model or matching stencil dataset ships with this repo for either order**. The original paper only ever trained a single WENO5 network (its 5-point, 75,241-sample generic-waveform dataset is `data/2ndNewAvgs.csv`/`data/2ndNewFlux.csv`); there is no order-3 or order-7 equivalent. A faithful WENO3-NN or WENO7-NN reproduction would require generating an analogous stencil dataset (same generic-waveform methodology, windowed to 3 or 7 cells instead of 5) and training a new model from scratch — this has not been done. `WENO5_vs_WENO5NN_StepAdvection.py`, `WENO5_vs_WENO5NN_Burgers.py`, and `WENO5_vs_WENO5NN_ShuOsher.py` are WENO5/WENO5-NN only.

## TENO and NN-TENO status

The repository includes deterministic TENO3, TENO5, and TENO7 scalar and
Euler reconstructions.  It also includes NENO-style classifier architectures
and scalar/Euler NN-TENO wrappers for all three orders.  If every candidate is
rejected, the wrappers use the center/upwind value as a first-order fallback.

Each nonconstant NN-TENO input stencil is normalized independently as

```text
(X - min(X)) / (max(X) - min(X))
```

Constant stencils map to zero. This preserves the deterministic selector's
translation and positive-scale invariance while keeping classifier inputs in
the training range.

Each sigmoid output classifies one candidate stencil: `0` means smooth/keep
and `1` means troubled/discard.  The network predicts candidate selection,
not the final flux, candidate polynomial, or reconstruction weights.

Train the NN-TENO5 classifier on the first CUDA GPU with:

```bash
conda activate weno-nn
nvidia-smi
python Train_NNTENO5.py --device gpu --gpu-index 0
```

The `gpu` mode fails immediately if TensorFlow cannot access the requested
GPU, rather than silently training on the CPU. TensorFlow uses incremental
GPU-memory allocation, which is suitable for a display-attached GPU. Use
`--device cpu` for an intentional CPU run or `--device auto` to allow
fallback.

The training script generates three deterministic TENO5 troubled/keep labels
from every original five-point stencil, creates disjoint stratified
train/validation/test splits, checkpoints the best validation model, calibrates
one decision threshold per candidate, and writes model, history, split, and
metadata artifacts under `models/`.

Dedicated NN-TENO3/7 training datasets and extensive trained-model PDE
benchmarks are not implemented yet.

## Tests

Run the regression suite with:

```bash
python -m unittest discover -s tests -v
```

The tests cover constant-state reconstruction, finite periodic advection, CFL rejection, boundary validation, scaled-data handling, and dataset separation. They verify the Roe matrices are mutual inverses and establish smooth periodic-advection convergence for WENO3, WENO5, and WENO7. The Euler end-to-end matrix runs all three schemes on constant stationary and uniformly moving states as well as the physically timed Sod and Shu–Osher regressions described above.

Numerical tests depend only on `requirements-numerical.txt`; they do not import TensorFlow. GitHub Actions runs them under modern Python and runs the neural smoke test separately inside a TensorFlow 1.15 container.

## Repository layout

```text
data/                   Five-point training stencils and target fluxes
src/core/               Simulation, time stepping, flux splitting, equations
src/initial_conditions/ Initial conditions for scalar and Euler problems
src/networks/           WENO-NN model definitions and data loading
src/schemes/            ENO and WENO reconstruction schemes
src/viz/                Plotting, metrics, and analysis helpers
tests/                  Numerical and data-pipeline regression tests
```

## Attribution

The numerical method and original WENO-NN implementation are derived from the work of Ben Stevens and Tim Colonius. See the linked paper and retain the relevant attribution when redistributing derived code.
