"""Regression test: src/differentiable/euler_jax.py vs the NumPy Euler solver.

Run inside `teno-jax` (see tests/test_teno5_jax_regression.py's docstring for
why this is a separate environment from the rest of the test suite):

    conda run -n teno-jax python -m pytest tests/test_teno5_jax_euler_regression.py -q
"""

import unittest

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np

from src.core.eulerEquations import getEulerFlux
from src.core.SimulationClasses import eulerSimulation
from src.core.TimeSteppingMethods import SSPRK3
from src.initial_conditions.InitialConditions import sod
from src.schemes import TENO5euler
from src.schemes.teno5 import teno5_reconstruct

from src.differentiable.euler_jax import get_euler_flux_jax
from src.differentiable.numerics_jax import rollout
from src.differentiable.teno5_jax import teno5_reconstruct_jax


class Euler5JaxRegressionTests(unittest.TestCase):

    def test_sod_rollout_matches_eulersimulation(self):
        cells = 100
        length = 1.0
        final_time = 0.2
        cfl = 0.5
        max_wave_speed = 2.0  # safe upper bound on |vel| + sound_speed for Sod

        dx = length / cells
        dt = cfl * dx / max_wave_speed
        step_count = int(np.ceil(final_time / dt))
        nt = step_count + 1

        x = np.arange(cells, dtype=float) * length / cells
        initial = sod()(x)

        numpy_sim = eulerSimulation(
            cells, nt, length, final_time, SSPRK3(),
            getEulerFlux(TENO5euler()), sod(), 3,
            max_wave_speed=max_wave_speed, max_cfl=cfl,
        )
        numpy_final = numpy_sim.runEuler()[:, -1, :]
        actual_dt = numpy_sim.last_times[1] - numpy_sim.last_times[0]

        jax_flux_fn = get_euler_flux_jax(teno5_reconstruct_jax, boundary='transmissive')
        jax_final = np.asarray(
            rollout(jnp.asarray(initial), jax_flux_fn, actual_dt, dx, step_count)
        )

        np.testing.assert_allclose(jax_final, numpy_final, atol=1e-9)

    def test_single_call_reconstruction_matches(self):
        rng = np.random.default_rng(0)
        stencils = rng.normal(size=(50, 3, 5)) + 1.0  # offset off zero for stability
        got = np.asarray(teno5_reconstruct_jax(jnp.asarray(stencils)))
        want = teno5_reconstruct(stencils)
        np.testing.assert_allclose(got, want, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
