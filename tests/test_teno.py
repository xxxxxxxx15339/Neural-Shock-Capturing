import unittest

import numpy as np

from src.schemes import (
    TENO3, TENO5, TENO7,
    NN_TENO3, NN_TENO5, NN_TENO7,
    NN_TENO3_Euler, NN_TENO5_Euler, NN_TENO7_Euler,
)
from src.schemes.teno_common import (
    combine_teno_candidates,
    normalize_teno_stencils,
)
from src.schemes.teno3 import LINEAR_WEIGHTS as WEIGHTS3
from src.schemes.teno3 import _candidate_values as candidates3
from src.schemes.teno5 import LINEAR_WEIGHTS as WEIGHTS5
from src.schemes.teno5 import _candidate_values as candidates5
from src.schemes.teno7 import LINEAR_WEIGHTS as WEIGHTS7
from src.schemes.teno7 import _candidate_values as candidates7


class FakeModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.inputs = []

    def predict(self, values, verbose=0):
        self.inputs.append(np.array(values, copy=True))
        if self.probabilities.ndim == 1:
            return np.tile(self.probabilities, (len(values), 1))
        return self.probabilities.copy()


CASES = (
    (3, 2, 1, TENO3, NN_TENO3, NN_TENO3_Euler, candidates3, WEIGHTS3),
    (5, 3, 2, TENO5, NN_TENO5, NN_TENO5_Euler, candidates5, WEIGHTS5),
    (7, 4, 3, TENO7, NN_TENO7, NN_TENO7_Euler, candidates7, WEIGHTS7),
)


class TENOTests(unittest.TestCase):
    def test_normalization_is_affine_invariant(self):
        values = np.array([
            [-2.0, -1.0, 0.0, 2.0, 3.0],
            [4.0, 4.0, 4.0, 4.0, 4.0],
        ])
        expected = np.array([
            [0.0, 0.2, 0.4, 0.8, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        np.testing.assert_allclose(
            normalize_teno_stencils(values),
            expected,
        )
        np.testing.assert_allclose(
            normalize_teno_stencils(7.0 + 3.0 * values),
            expected,
        )

    def test_deterministic_constant_reconstruction(self):
        for width, _, _, deterministic, _, _, _, _ in CASES:
            with self.subTest(order=width):
                values = np.full(24, 2.5)
                original = values.copy()
                result = deterministic().evalF(values)
                np.testing.assert_allclose(result, values)
                np.testing.assert_array_equal(values, original)
                self.assertTrue(np.isfinite(result).all())

    def test_common_combination_and_fallback(self):
        candidates = np.array([[1.0, 3.0], [2.0, 8.0]])
        weights = np.array([0.25, 0.75])
        fallback = np.array([-1.0, -2.0])
        retained = combine_teno_candidates(
            candidates, np.ones_like(candidates), weights, fallback,
        )
        np.testing.assert_allclose(retained, [2.5, 6.5])
        rejected = combine_teno_candidates(
            candidates, np.zeros_like(candidates), weights, fallback,
        )
        np.testing.assert_allclose(rejected, fallback)

    def test_neural_masks_threshold_fallback_and_immutability(self):
        for width, count, center, _, nn_method, _, candidate_fn, weights in CASES:
            values = np.arange(5*width, dtype=float).reshape(5, width)/4.0
            original = values.copy()
            for probabilities in (
                np.zeros(count),
                np.r_[0.5, np.zeros(count-1)],
                np.ones(count),
            ):
                with self.subTest(order=width, probabilities=probabilities):
                    model = FakeModel(probabilities)
                    result = nn_method(model).L(values)
                    keep = np.logical_not(probabilities >= 0.5)
                    expected = combine_teno_candidates(
                        candidate_fn(values),
                        np.broadcast_to(keep, (len(values), count)),
                        weights,
                        values[:, center],
                    )
                    np.testing.assert_allclose(result, expected)
                    np.testing.assert_array_equal(values, original)
                    self.assertEqual(len(model.inputs), 1)
                    np.testing.assert_allclose(
                        model.inputs[0], normalize_teno_stencils(values),
                    )

    def test_euler_uses_one_flat_batch_and_independent_masks(self):
        nx, fields = 4, 3
        for width, count, center, _, _, nn_euler, candidate_fn, weights in CASES:
            values = np.arange(nx*fields*width, dtype=float).reshape(
                nx, fields, width,
            )/5.0
            original = values.copy()
            probabilities = np.zeros((nx*fields, count))
            probabilities[0, 0] = 0.8
            probabilities[1, :] = 0.8
            model = FakeModel(probabilities)
            result = nn_euler(model).L(values)
            keep = np.logical_not(
                probabilities.reshape(nx, fields, count) >= 0.5
            )
            expected = combine_teno_candidates(
                candidate_fn(values), keep, weights, values[:, :, center],
            )
            np.testing.assert_allclose(result, expected)
            np.testing.assert_array_equal(values, original)
            self.assertEqual(result.shape, (nx, fields))
            self.assertEqual(len(model.inputs), 1)
            self.assertEqual(model.inputs[0].shape, (nx*fields, width))


if __name__ == '__main__':
    unittest.main()
