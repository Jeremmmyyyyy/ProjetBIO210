from unittest import TestCase

import numpy as np

from Week2 import hopefieldNetwork as hop


class Test(TestCase):

    def test_generate_patterns(self):
        with self.assertRaises(Exception):
            hop.generate_patterns(-3, -1)

    def test_hebbian_matrix_with_given_pattern(self):
        pattern_matrix = np.array([[1, 1, -1, -1],
                                  [1, 1, -1, 1],
                                  [-1, 1, -1, 1]])

        hebbian_result = np.array([[0., 0.33333333, -0.33333333, -0.33333333],
                                   [0.33333333, 0., -1, 0.33333333],
                                   [-0.33333333, -1, 0., -0.33333333],
                                   [-0.33333333, 0.33333333, -0.33333333, 0.]])

        hebbian_calculated = hop.hebbian_weights(pattern_matrix)

        self.assertTrue(np.allclose(hebbian_result, hebbian_calculated))

    def test_storkey_matrix_with_given_pattern(self):
        pattern_matrix = np.array([[1, 1, -1, -1],
                                   [1, 1, -1, 1],
                                   [-1, 1, -1, 1]])

        storkey_result = np.array([[1.125, 0.25, -0.25, -0.5],
                                   [0.25, 0.625, -1, 0.25],
                                   [-0.25, -1, 0.625, -0.25],
                                   [-0.5, 0.25, -0.25, 1.125]])

        storkey_calculated = hop.storkey_weights(pattern_matrix)
        print(storkey_calculated)

        self.assertTrue(np.allclose(storkey_result, storkey_calculated))
