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
        last = dynamic[-1]
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
    percent = (correct / len(results)) * 100
    if correct != 0:
        print(f"{correct} success and {wrong} errors : {percent} % percent")
    else:
        print(f"{correct} success and {wrong} errors")

    return percent


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

        self.assertTrue(np.allclose(storkey_result, storkey_calculated))

    def test_synaptic_term(self):
        pattern = np.array([0, 1, 2])

        h = np.array([[0, 1, 2],
                      [0, 1, 2],
                      [0, 1, 2]])

        result = np.array([[0, 0, 0],
                           [0, 1, 2],
                           [0, 2, 4]])
        result_func = func.preAndPost_synaptic_term_computation(pattern, h)

        self.assertTrue(np.allclose(result, result_func))

    def test_convergence_hebbian_normal(self):
        pattern_matrix = func.generate_patterns(number_of_patterns, pattern_size)
        print("Computing started hebbian_normal ... \n")

        hebbian_matrix = func.hebbian_weights(pattern_matrix)
        print("Hebbian matrix created \n")

        results_hebbian_dynamic = test_dynamic_hop(pattern_matrix,
                                                   hebbian_matrix,
                                                   perturbations,
                                                   max_iterations_dynamic)
        percent = analyse_result(results_hebbian_dynamic)

        if number_of_patterns == 80 and perturbations == 80:
            self.assertTrue(50 < percent <= 100)
        elif number_of_patterns == 200 and perturbations == 80:
            self.assertTrue(percent <= 5)
        else:
            self.assertTrue(percent != 0)

    def test_convergence_storkey_normal(self):
        pattern_matrix = func.generate_patterns(number_of_patterns, pattern_size)
        print("Computing started storkey_normal ... \n")

        storkey_matrix = func.storkey_weights(pattern_matrix)
        print("Storkey matrix created \n")

        results_storkey_dynamic = test_dynamic_hop(pattern_matrix,
                                                   storkey_matrix,
                                                   perturbations,
                                                   max_iterations_dynamic)
        percent = analyse_result(results_storkey_dynamic)

        if number_of_patterns == 80 and perturbations == 80:
            self.assertTrue(90 < percent <= 100)
        elif number_of_patterns == 200 and perturbations == 80:
            self.assertTrue(90 < percent <= 100)
        else:
            self.assertTrue(percent != 0)

    def test_convergence_hebbian_dynamic(self):
        pattern_matrix = func.generate_patterns(number_of_patterns, pattern_size)
        print("Computing started hebbian_dynamic ... \n")

        hebbian_matrix = func.hebbian_weights(pattern_matrix)
        print("Hebbian matrix created \n")

        results_hebbian_async = test_dynamic_async_hop(pattern_matrix,
                                                       hebbian_matrix,
                                                       perturbations,
                                                       max_iterations_async,
                                                       async_iterations_without_change)
        percent = analyse_result(results_hebbian_async)

        if number_of_patterns == 80 and perturbations == 80:
            self.assertTrue(60 < percent <= 100)
        elif number_of_patterns == 200 and perturbations == 80:
            self.assertTrue(percent <= 5)
        else:
            self.assertTrue(percent != 0)

    def test_convergence_storkey_dynamic(self):
        pattern_matrix = func.generate_patterns(number_of_patterns, pattern_size)
        print("Computing started storkey_dynamic ... \n")

        storkey_matrix = func.storkey_weights(pattern_matrix)
        print("Storkey matrix created \n")

        results_storkey_async = test_dynamic_async_hop(pattern_matrix,
                                                       storkey_matrix,
                                                       perturbations,
                                                       max_iterations_async,
                                                       async_iterations_without_change)
        percent = analyse_result(results_storkey_async)

        if number_of_patterns == 80 and perturbations == 80:
            self.assertTrue(90 < percent <= 100)
        elif number_of_patterns == 200 and perturbations == 80:
            self.assertTrue(90 < percent <= 100)
        else:
            self.assertTrue(percent != 0)
