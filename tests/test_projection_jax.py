"""Phase 4 exit criterion: the JAX projection layer at threshold 0.5 in
'hard' mode is a bit-exact no-op against the real NumPy NN-TENO5 inference
path (teno5.py's teno5_neural_reconstruct), not just internally consistent.

Run inside `teno-jax` (imports jax; see other test_teno5_jax_* docstrings
for why this is a separate environment):

    conda run -n teno-jax python -m pytest tests/test_projection_jax.py -q
"""

import unittest

import jax

jax.config.update('jax_enable_x64', True)

import jax.numpy as jnp
import numpy as np

from src.schemes.teno5 import teno5_neural_reconstruct

from src.differentiable.projection_jax import project_keep_weight
from src.differentiable.teno5_jax import teno5_soft_reconstruct_jax


class FakeModel:
    """Stands in for a Keras NN-TENO5 model: .predict() returns fixed,
    pre-chosen troubled-probabilities regardless of input, so both the
    NumPy and JAX paths can be driven by the exact same numbers.
    """

    def __init__(self, probabilities):
        self._probabilities = np.asarray(probabilities, dtype=float)

    def predict(self, flat_inputs, verbose=0):
        assert flat_inputs.shape[0] == self._probabilities.shape[0]
        return self._probabilities


class ProjectionLayerNoOpTests(unittest.TestCase):

    def test_hard_mode_matches_real_numpy_inference_path(self):
        rng = np.random.default_rng(3)
        n = 500
        stencils = rng.normal(size=(n, 5))
        # Cover the full range, including values exactly at/near the 0.5
        # threshold, which is exactly where a hard/soft mismatch would show.
        probabilities = rng.uniform(0.0, 1.0, size=(n, 3))
        probabilities[:20] = 0.5  # exact-threshold edge case

        model = FakeModel(probabilities)
        numpy_result = teno5_neural_reconstruct(model, stencils, threshold=0.5)

        keep_weight = project_keep_weight(
            jnp.asarray(probabilities), mode='hard', threshold=0.5
        )
        jax_result = np.asarray(
            teno5_soft_reconstruct_jax(jnp.asarray(stencils), keep_weight)
        )

        np.testing.assert_allclose(jax_result, numpy_result, atol=1e-12)

    def test_soft_mode_is_continuous_and_bounded(self):
        rng = np.random.default_rng(4)
        probabilities = jnp.asarray(rng.uniform(0.0, 1.0, size=(200, 3)))
        keep_weight = project_keep_weight(probabilities, mode='soft')
        self.assertTrue(bool(jnp.all(keep_weight >= 0.0)))
        self.assertTrue(bool(jnp.all(keep_weight <= 1.0)))
        np.testing.assert_allclose(
            np.asarray(keep_weight), 1.0 - np.asarray(probabilities)
        )

    def test_all_troubled_falls_back_to_center_value(self):
        stencils = jnp.asarray([[1.0, 2.0, 3.0, 4.0, 5.0]])
        troubled = jnp.ones((1, 3))  # every candidate maximally troubled
        keep_weight = project_keep_weight(troubled, mode='hard', threshold=0.5)
        result = teno5_soft_reconstruct_jax(stencils, keep_weight)
        # CENTER_INDEX = 2 -> stencils[..., 2] = 3.0
        np.testing.assert_allclose(np.asarray(result), [3.0])

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            project_keep_weight(jnp.zeros((3,)), mode='bogus')


if __name__ == '__main__':
    unittest.main()
