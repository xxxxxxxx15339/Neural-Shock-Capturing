# Executed environments

The measured environment is recorded in `results/raw/environment.json`.
Numerical evaluation: Python3.11.15, JAX/jaxlib0.10.2 CPU, NumPy2.4.6,
SciPy1.17.1, matplotlib3.11.1, h5py3.16.0. Use the repository's
`requirements/locks/jax-python311.txt`; no dependency upgrade is required.
CPU was the only JAX backend available. Numerical solution arrays are float64;
the original Keras classifier and author NENO6 normalized mask weights retain
their authentic float32 arithmetic. Network architecture/fixture parity tests
explicitly cover these mixed-precision details.

Original HDF5 execution/export uses the existing `weno-nn` environment
(Python3.7, Keras2.2.4, TensorFlow1.15), with `CUDA_VISIBLE_DEVICES=-1` and
`PYTHONDONTWRITEBYTECODE=1`. Keras models are loaded with `compile=False`.
PyTorch2.12.1 in the base environment safely loads the author state dictionary
with `weights_only=True`, and runs the author model and reconstruction classes
extracted by AST. No model training or optimizer update occurs.

The export NPZ fixtures are included, so ordinary numerical reproduction and
tests need only the JAX environment. To independently regenerate the original
framework parity fixtures, set `LEGACY_PY` and `TORCH_PY` as described in
`reproduce.sh` and use `REEXPORT=1`. Raw exports preserve exact trained parameters.
No TensorFlow/PyTorch import occurs in the timed common JAX numerical solver.
Timing therefore compares the reconstruction methods in one environment, not
the speed of the original framework implementations. Runtime is CPU-specific;
GPU speed, controlled core affinity and cross-machine speed are unverified.

Source/checkpoint preservation is verified against the pre-evaluation evidence
catalog. Runtime compilation and warmed execution are recorded separately;
the three-repeat median/IQR is a descriptive timing estimate, not a statistical
claim across independent machines. Reference uncertainty is numerical sensitivity,
not a rigorous bound. There are too few independent trajectories to bootstrap
confidence intervals by PDE family; cells/timesteps are not independent samples.
