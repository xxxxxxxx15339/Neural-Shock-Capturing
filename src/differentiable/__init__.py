# -*- coding: utf-8 -*-
"""JAX ports used for the the Unrolled Physics Loss proposal "unrolled physics loss" spike (Phase 0/1).

This package is kept separate from `src/schemes/` and `src/networks/` on
purpose. `src/networks/__init__.py` eagerly does `from keras import backend`,
and the legacy training stack is pinned to standalone Keras 2.2.4 + NumPy
1.18.5 (see requirements.txt) -- a stack that cannot coexist in one
environment with a modern JAX install. Keeping these ports in their own
package means:

- importing anything here never triggers a Keras import, so this code runs
  in a plain `jax + numpy` environment (see the `teno-jax` conda env) without
  installing or fighting the legacy pin;
- importing the legacy `src/schemes`/`src/networks` packages never requires
  JAX to be installed, so the existing Keras-based training/eval scripts are
  untouched by this work.

Nothing in this package is imported by the legacy stack, and nothing in the
legacy stack is imported here except the pure-NumPy, Keras-free modules
(`src/schemes/teno5.py`, `src/core/SimulationClasses.py`, etc.) used for
regression testing against the ports.
"""
