"""Phase 5 exit criterion: all four Unrolled Physics Loss terms compute
without NaN on a real Phase 1 rollout, and jax.grad(L_total) returns finite
gradients w.r.t. network weights.

Run inside `teno-jax`:

    conda run -n teno-jax python -m pytest tests/test_losses_jax.py -q
"""

import unittest

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np

from src.differentiable.losses_jax import l_bvd, l_mse, l_reg, l_tv, unrolled_physics_loss
from src.differentiable.numerics_jax import boundary_variation_jax, lax_friedrichs_flux, rollout
from src.differentiable.projection_jax import project_keep_weight
from src.differentiable.teno5_jax import teno5_reconstruct_jax, teno5_soft_reconstruct_jax
from src.differentiable.teno5_jax_toy import init_toy_params, toy_keep_weight


def _physical_flux(u):
    return u


def _network_reconstruct(params, stencils, mode='soft'):
    troubled = 1.0 - toy_keep_weight(params, stencils)  # toy_keep_weight already returns keep
    keep_weight = project_keep_weight(troubled, mode=mode)
    return teno5_soft_reconstruct_jax(stencils, keep_weight)


def _soft_flux_fn(params, u):
    return lax_friedrichs_flux(
        u, _physical_flux, 1.0, lambda stencils: _network_reconstruct(params, stencils)
    )


class UnrolledPhysicsLossTests(unittest.TestCase):

    def setUp(self):
        self.cells = 32
        self.dx = 1.0 / self.cells
        self.dt = 0.5 * self.dx / 1.0
        self.steps = 10
        x = np.linspace(0.0, 1.0, self.cells, endpoint=False)
        self.u0 = jnp.asarray(np.where((x > 0.25) & (x < 0.75), 1.0, -1.0))
        shifted = (x - self.steps * self.dt) % 1.0  # wave_speed = 1.0
        self.exact_final = jnp.asarray(
            np.where((shifted > 0.25) & (shifted < 0.75), 1.0, -1.0)
        )
        self.params = init_toy_params(jax.random.PRNGKey(0))
        self.anchor_params = self.params  # zero L_Reg baseline case

    def _loss_and_breakdown(self, params):
        final_state = rollout(self.u0, lambda u: _soft_flux_fn(params, u), self.dt, self.dx, self.steps)

        def network_reconstruct(s):
            return _network_reconstruct(params, s)

        network_bv = boundary_variation_jax(final_state, network_reconstruct)
        deterministic_bv = boundary_variation_jax(final_state, teno5_reconstruct_jax)

        return unrolled_physics_loss(
            final_state, self.exact_final, self.u0, network_bv, deterministic_bv,
            params, self.anchor_params,
            lambda_mse=1.0, lambda_tv=1.0, lambda_bvd=1.0, lambda_reg=0.1,
        )

    def test_all_four_terms_finite(self):
        total, breakdown = self._loss_and_breakdown(self.params)
        for name, value in breakdown.items():
            self.assertTrue(np.isfinite(float(value)), '{} is not finite: {}'.format(name, value))

    def test_l_reg_zero_at_anchor(self):
        _, breakdown = self._loss_and_breakdown(self.params)
        self.assertAlmostEqual(float(breakdown['l_reg']), 0.0, places=10)

    def test_l_reg_nonzero_away_from_anchor(self):
        perturbed = [(w + 0.1, b + 0.1) for w, b in self.params]
        value = l_reg(perturbed, self.anchor_params)
        self.assertGreater(float(value), 0.0)

    def test_gradient_through_total_loss_is_finite(self):
        def loss_fn(params):
            total, _ = self._loss_and_breakdown(params)
            return total

        grads = jax.grad(loss_fn)(self.params)
        all_finite = True
        any_nonzero = False
        for dw, db in grads:
            if not (np.isfinite(np.asarray(dw)).all() and np.isfinite(np.asarray(db)).all()):
                all_finite = False
            if np.any(np.asarray(dw) != 0.0) or np.any(np.asarray(db) != 0.0):
                any_nonzero = True
        self.assertTrue(all_finite)
        self.assertTrue(any_nonzero)


if __name__ == '__main__':
    unittest.main()
