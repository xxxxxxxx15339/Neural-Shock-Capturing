"""Tests for src/data/feature_vector.py (the Unrolled Physics Loss proposal Phase 2 feature vector).

Pure NumPy -- no Keras, no JAX -- so this runs in any of the repo's
environments, e.g.:

    conda run -n base python -m pytest tests/test_feature_vector.py -q
"""

import os
import unittest

import numpy as np

from src.schemes.teno5 import LINEAR_WEIGHTS, STENCIL_SIZE
from src.schemes.teno_common import normalize_teno_stencils
from src.data.feature_vector import (
    euler_feature_vector,
    flatten_feature_vector,
    scalar_feature_vector,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STENCILS_PATH = os.path.join(REPO_ROOT, 'data', 'teno5ExactStencils.csv')


class ScalarFeatureVectorRealDatasetTests(unittest.TestCase):
    """Phase 2's own exit criterion: run over the real oracle dataset."""

    @classmethod
    def setUpClass(cls):
        raw = np.loadtxt(STENCILS_PATH, delimiter=',')  # stored as (5, n_samples)
        cls.stencils = raw.T  # (n_samples, 5)

    def test_shape_and_sample_count(self):
        self.assertEqual(self.stencils.shape[1], STENCIL_SIZE)
        self.assertEqual(self.stencils.shape[0], 75000)

    def test_no_nans_across_full_dataset(self):
        features = scalar_feature_vector(self.stencils)
        for name, value in features.items():
            self.assertTrue(
                np.isfinite(value).all(),
                'feature {!r} contains non-finite values'.format(name),
            )

    def test_u_normalized_matches_existing_normalization(self):
        features = scalar_feature_vector(self.stencils)
        expected = normalize_teno_stencils(self.stencils)
        np.testing.assert_array_equal(features['u_normalized'], expected)

    def test_scheme_linear_weights_is_the_constant_broadcast(self):
        features = scalar_feature_vector(self.stencils)
        np.testing.assert_array_equal(
            features['scheme_linear_weights'], np.broadcast_to(
                LINEAR_WEIGHTS, (self.stencils.shape[0], 3)
            )
        )

    def test_five_wide_window_has_no_j_left(self):
        features = scalar_feature_vector(self.stencils)
        self.assertNotIn('j_left', features)
        self.assertIn('j_right', features)

    def test_flatten_width_for_five_wide_window(self):
        features = scalar_feature_vector(self.stencils)
        flat = flatten_feature_vector(features)
        # u_normalized(5) + scheme_linear_weights(3) + eta(3) + j_right(1)
        self.assertEqual(flat.shape, (75000, 12))


class SixWideWindowTests(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.window = rng.normal(size=(200, 6)) + 1.0  # offset off zero

    def test_both_jumps_present_and_finite(self):
        features = scalar_feature_vector(self.window)
        self.assertIn('j_left', features)
        self.assertIn('j_right', features)
        self.assertTrue(np.isfinite(features['j_left']).all())
        self.assertTrue(np.isfinite(features['j_right']).all())

    def test_j_right_matches_five_wide_call_on_trailing_slice(self):
        wide_features = scalar_feature_vector(self.window)
        narrow_features = scalar_feature_vector(self.window[:, -5:])
        np.testing.assert_array_equal(
            wide_features['j_right'], narrow_features['j_right']
        )

    def test_flatten_width_for_six_wide_window(self):
        features = scalar_feature_vector(self.window)
        flat = flatten_feature_vector(features)
        # u_normalized(5) + scheme_linear_weights(3) + eta(3) + j_left(1) + j_right(1)
        self.assertEqual(flat.shape, (200, 13))

    def test_rejects_wrong_width(self):
        with self.assertRaises(ValueError):
            scalar_feature_vector(self.window[:, :4])


class EulerFeatureVectorTests(unittest.TestCase):

    def test_adds_mach_and_pressure_terms(self):
        rng = np.random.default_rng(2)
        window = rng.normal(size=(50, 5)) + 1.0
        mach = rng.uniform(0.0, 2.0, size=50)
        dpop = rng.uniform(0.0, 1.0, size=50)

        features = euler_feature_vector(window, mach, dpop)
        self.assertIn('mach_number', features)
        self.assertIn('pressure_gradient', features)
        np.testing.assert_array_equal(features['mach_number'], mach)

        flat = flatten_feature_vector(features)
        # 5 + 3 + 3 + 1 (j_right only, 5-wide window) + 1 (mach) + 1 (dP/P)
        self.assertEqual(flat.shape, (50, 14))


if __name__ == '__main__':
    unittest.main()
