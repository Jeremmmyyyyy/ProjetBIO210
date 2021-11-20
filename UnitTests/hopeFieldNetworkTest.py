import numpy as np
from unittest import TestCase
from Week2 import hopefieldNetwork as func

################################################################
# Here you can change the different parameters for the Network #
################################################################
number_of_patterns = 80
pattern_size = 1000
perturbations = 80
max_iterations_dynamic = 20
max_iterations_async = 20000
async_iterations_without_change = 3000
################################################################
################################################################


def test_dynamic_hop(pattern_matrix, weight_matrix, perturb, max_iter):
    """
    Perturb all the patterns of the pattern matrix and try to retrieve them
    We use the normal algorithm to find out the pattern
    :param pattern_matrix: pattern matrix
    :param weight_matrix: weight matrix
    :param perturb: number of perturbations in a pattern
    :param max_iter: max iterations to retrieve a pattern
    :return: a list containing True if the pattern is retrieved False else
    """

    answers = []
    for i in range(len(pattern_matrix)):
        perturbed_pattern = func.perturb_pattern(pattern_matrix[i], perturb)
        dynamic = func.dynamics(perturbed_pattern, weight_matrix, max_iter)
        last = dynamic[len(dynamic) - 1]
        answer = func.pattern_match(pattern_matrix, last)
        answers.append(False if answer is None else True)
    return answers


def test_dynamic_async_hop(pattern_matrix, weight_matrix, perturb, max_iter, iter_no_change):
    """
    Perturb all the patterns of the pattern matrix and try to retrieve them
    We use the normal algorithm to find out the pattern
    :param pattern_matrix: pattern matrix
    :param weight_matrix: weight matrix
    :param perturb: number of perturbations in a pattern
    :param max_iter: max iterations to retrieve a pattern
    :param iter_no_change: number of non changing patterns to achieve to stop the convergence algorithm
    :return: a list containing True if the pattern is retrieved False else
    """

    answers = []
    for i in range(len(pattern_matrix)):
        perturbed_pattern = func.perturb_pattern(pattern_matrix[i], perturb)
        dynamic = func.dynamics_async(perturbed_pattern, weight_matrix, max_iter, iter_no_change)
        last = dynamic[len(dynamic) - 1]
        answer = func.pattern_match(pattern_matrix, last)
        answers.append(False if answer is None else True)
    return answers


def analyse_result(results):
    """
    Transform the list in some stats that are printed in the console
    :param results: list of True / False values
    """
    wrong = results.count(False)
    correct = results.count(True)
    if correct != 0:
        print(f"{correct} success and {wrong} errors : {100 - (100 * wrong / correct)} % percent")
    else:
        print(f"{correct} success and {wrong} errors")


class Test(TestCase):
    def test_generate_patterns_normal(self):
        pattern_matrix = func.generate_patterns(number_of_patterns, pattern_size)
        x, y = pattern_matrix.shape
        self.assertEqual(x, number_of_patterns)
        self.assertEqual(y, pattern_size)

    def test_generate_patterns_negative(self):
        with self.assertRaises(Exception):
            func.generate_patterns(-3, -1)

    def test_hebbian_matrix_with_given_pattern(self):
        pattern_matrix = np.array([[1, 1, -1, -1],
                                   [1, 1, -1, 1],
                                   [-1, 1, -1, 1]])

        hebbian_result = np.array([[0., 0.33333333, -0.33333333, -0.33333333],
                                   [0.33333333, 0., -1, 0.33333333],
                                   [-0.33333333, -1, 0., -0.33333333],
                                   [-0.33333333, 0.33333333, -0.33333333, 0.]])

        hebbian_calculated = func.hebbian_weights(pattern_matrix)

        self.assertTrue(np.allclose(hebbian_result, hebbian_calculated))

    def test_storkey_matrix_with_given_pattern(self):
        pattern_matrix = np.array([[1, 1, -1, -1],
                                   [1, 1, -1, 1],
                                   [-1, 1, -1, 1]])

        storkey_result = np.array([[1.125, 0.25, -0.25, -0.5],
                                   [0.25, 0.625, -1, 0.25],
                                   [-0.25, -1, 0.625, -0.25],
                                   [-0.5, 0.25, -0.25, 1.125]])

        storkey_calculated = func.storkey_weights(pattern_matrix)
        print(storkey_calculated)

        self.assertTrue(np.allclose(storkey_result, storkey_calculated))

    def test_storkey_matrix_with_given_pattern_efficiency(self):
        pattern_matrix = np.array([[1, 1, -1, -1],
                                   [1, 1, -1, 1],
                                   [-1, 1, -1, 1]])

        storkey_result = np.array([[1.125, 0.25, -0.25, -0.5],
                                   [0.25, 0.625, -1, 0.25],
                                   [-0.25, -1, 0.625, -0.25],
                                   [-0.5, 0.25, -0.25, 1.125]])

        storkey_calculated = func.storkey_weights_efficient(pattern_matrix)
        print(storkey_calculated)

        self.assertTrue(np.allclose(storkey_result, storkey_calculated))

    def test_synaptic_term(self):
        pattern = np.array([0, 1, 2])

        h = np.array([[0, 1, 2],
                      [0, 1, 2],
                      [0, 1, 2]])

        result = np.array([[0, 0, 0],
                           [0, 1, 2],
                           [0, 2, 4]])

        self.assertTrue(np.allclose(result, func.preAndPost_synaptic_term_computation(pattern, h)))

    def test_convergence_hebbian_storkey(self):
        pattern_matrix = func.generate_patterns(number_of_patterns, pattern_size)
        print("Computing started ... \n")
        hebbian_matrix = func.hebbian_weights(pattern_matrix)
        print("Hebbian matrix created \n")
        storkey_matrix = func.storkey_weights_efficient(pattern_matrix)
        print("Storkey matrix created \n")

        results_hebbian_dynamic = test_dynamic_hop(pattern_matrix,
                                                   hebbian_matrix,
                                                   perturbations,
                                                   max_iterations_dynamic)
        analyse_result(results_hebbian_dynamic)
        results_storkey_dynamic = test_dynamic_hop(pattern_matrix,
                                                   storkey_matrix,
                                                   perturbations,
                                                   max_iterations_dynamic)
        analyse_result(results_storkey_dynamic)

        results_hebbian_async = test_dynamic_async_hop(pattern_matrix,
                                                       hebbian_matrix,
                                                       perturbations,
                                                       max_iterations_async,
                                                       async_iterations_without_change)
        analyse_result(results_hebbian_async)
        results_storkey_async = test_dynamic_async_hop(pattern_matrix,
                                                       storkey_matrix,
                                                       perturbations,
                                                       max_iterations_async,
                                                       async_iterations_without_change)
        analyse_result(results_storkey_async)
