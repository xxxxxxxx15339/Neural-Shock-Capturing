"""Regression tests: src/differentiable/*_jax.py vs the NumPy originals.

Run inside the `teno-jax` conda environment (jax + numpy + matplotlib; no
Keras, so these do not run in `weno-nn`, and `weno-nn` has no jax so the
reverse is also true):

    conda run -n teno-jax python -m pytest tests/test_teno5_jax_regression.py -q

Kept separate from tests/test_teno.py (which runs in `base`/`weno-nn`)
because this file imports `jax`, and importing `jax` is exactly what
src/differentiable/__init__.py's docstring explains this repo's other three
environments cannot assume is available.
"""

import unittest

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np

from src.core.Equations import adv
from src.core.FluxSplittingMethods import LaxFriedrichs
from src.core.SimulationClasses import Simulation
from src.core.TimeSteppingMethods import SSPRK3
from src.schemes import TENO5
from src.schemes.teno5 import (
    teno5_candidate_values as np_candidate_values,
    teno5_reconstruct as np_reconstruct,
    teno5_smoothness_indicators as np_smoothness,
)

from src.differentiable.numerics_jax import lax_friedrichs_flux, rollout
from src.differentiable.teno5_jax import (
    teno5_candidate_values,
    teno5_reconstruct_jax,
    teno5_smoothness_indicators,
)


class Teno5JaxRegressionTests(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.stencils = rng.normal(size=(500, 5))

    def test_candidate_values_match(self):
        got = np.asarray(teno5_candidate_values(jnp.asarray(self.stencils)))
        want = np_candidate_values(self.stencils)
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_smoothness_indicators_match(self):
        got = np.asarray(teno5_smoothness_indicators(jnp.asarray(self.stencils)))
        want = np_smoothness(self.stencils)
        np.testing.assert_allclose(got, want, atol=1e-12)

    def test_deterministic_reconstruction_matches(self):
        got = np.asarray(teno5_reconstruct_jax(jnp.asarray(self.stencils)))
        want = np_reconstruct(self.stencils)
        np.testing.assert_allclose(got, want, atol=1e-10)

    def test_full_rollout_matches_simulation_run(self):
        cells = 64
        final_time = 0.3
        wave_speed = 1.0
        cfl = 0.5
        dx = 1.0 / cells
        step_count = int(np.ceil(final_time * wave_speed / (cfl * dx)))
        dt = final_time / step_count

        def sine_ic(x):
            return np.sin(2.0 * np.pi * x)

        x = np.linspace(0.0, 1.0, cells, endpoint=False)
        u0 = sine_ic(x)

        numpy_sim = Simulation(
            cells, step_count + 1, 1.0, final_time,
            SSPRK3(), LaxFriedrichs(adv(), wave_speed), TENO5(), sine_ic,
            max_cfl=cfl,
        )
        numpy_final = numpy_sim.run()[:, -1]

        def physical_flux(u):
            return u

        def jax_flux_fn(u):
            return lax_friedrichs_flux(u, physical_flux, wave_speed, teno5_reconstruct_jax)

        jax_final = np.asarray(rollout(jnp.asarray(u0), jax_flux_fn, dt, dx, step_count))

        np.testing.assert_allclose(jax_final, numpy_final, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
