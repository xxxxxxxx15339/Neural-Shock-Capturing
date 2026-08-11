import unittest

import numpy as np

from src.networks.WenoNetworks import WENO51stOrder
from src.networks.TenoNetworks import TENO3Network, TENO5Network, TENO7Network


class NeuralPipelineSmokeTests(unittest.TestCase):
    def test_weno5_model_builds_and_predicts(self):
        model = WENO51stOrder(0.15)
        prediction = model.predict(np.zeros((2, 5)), verbose=0)
        self.assertEqual(prediction.shape, (2, 1))
        self.assertTrue(np.isfinite(prediction).all())

    def test_teno_classifiers_build_and_predict_probabilities(self):
        for width, candidates, builder in (
            (3, 2, TENO3Network),
            (5, 3, TENO5Network),
            (7, 4, TENO7Network),
        ):
            with self.subTest(order=width):
                model = builder()
                prediction = model.predict(
                    np.zeros((2, width)), verbose=0,
                )
                self.assertEqual(prediction.shape, (2, candidates))
                self.assertTrue(np.isfinite(prediction).all())
                self.assertTrue(np.all(prediction >= 0.0))
                self.assertTrue(np.all(prediction <= 1.0))


if __name__ == '__main__':
    unittest.main()
