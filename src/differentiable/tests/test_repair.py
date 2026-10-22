"""Regression for the positive-state mutation found in the P0 audit."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import numpy as np

from src.differentiable.euler_jax import euler_positivity_repair


def test_admissible_low_density_pressure_is_unchanged():
    state = jnp.tile(jnp.array([0.01, 0.0, 0.01 / 0.4]), (5, 1))
    repaired = np.asarray(euler_positivity_repair(state))
    np.testing.assert_array_equal(repaired, np.asarray(state))


def test_admissible_neighbor_average_is_not_floored():
    state = jnp.tile(jnp.array([0.01, 0.0, 0.01 / 0.4]), (5, 1))
    state = state.at[2].set(jnp.array([-1.0, 0.0, -1.0]))
    expected = np.tile([0.01, 0.0, 0.01 / 0.4], (5, 1))
    np.testing.assert_array_equal(np.asarray(euler_positivity_repair(state)), expected)
