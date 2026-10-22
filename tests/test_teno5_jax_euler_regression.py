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
from src.differentiable.numerics_jax import rollout, ssprk3_step
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
        numpy_history = numpy_sim.runEuler()
        numpy_final = numpy_history[:, -1, :]
        actual_dt = numpy_sim.last_times[1] - numpy_sim.last_times[0]

        jax_flux_fn = get_euler_flux_jax(teno5_reconstruct_jax, boundary='transmissive')
        compiled_flux = jax.jit(jax_flux_fn)
        compiled_step = jax.jit(
            lambda state: ssprk3_step(state, jax_flux_fn, actual_dt, dx)
        )
        # Compare both implementations on identical states, including developed
        # shock profiles, before testing accumulated error across the rollout.
        for step in (0, step_count // 2, step_count - 1):
            with self.subTest(step=step):
                state = numpy_history[:, step, :]
                np.testing.assert_allclose(
                    np.asarray(compiled_flux(jnp.asarray(state))),
                    numpy_sim.flux(state), rtol=1e-12, atol=1e-12,
                )
                np.testing.assert_allclose(
                    np.asarray(compiled_step(jnp.asarray(state))),
                    numpy_sim.RK.stepItEuler(
                        state, numpy_sim.flux, actual_dt, dx, 3,
                    ),
                    rtol=1e-12, atol=1e-12,
                )
        jax_final = np.asarray(
            rollout(jnp.asarray(initial), jax_flux_fn, actual_dt, dx, step_count)
        )

        self.assertEqual(jax_final.dtype, np.dtype(np.float64))
        # Hard stencil selection amplifies CPU roundoff over the 80-step
        # rollout: GitHub runners reached 4.9e-8 in near-zero momentum. Keep
        # the strict kernel/one-step checks above, but give the full trajectory
        # an absolute tolerance consistent with its existing relative one.
        np.testing.assert_allclose(jax_final, numpy_final, rtol=1e-7, atol=1e-7)

    def test_single_call_reconstruction_matches(self):
        rng = np.random.default_rng(0)
        stencils = rng.normal(size=(50, 3, 5)) + 1.0  # offset off zero for stability
        got = np.asarray(teno5_reconstruct_jax(jnp.asarray(stencils)))
        want = teno5_reconstruct(stencils)
        np.testing.assert_allclose(got, want, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
