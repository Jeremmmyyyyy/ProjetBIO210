import numpy as np
from Week2 import hopefieldNetwork as hop


def test_dynamic(pattern_matrix, weight_matrix, perturbations, max_iter):
    """
    Perturb all the patterns of the pattern matrix and try to retrieve them
    We use the normal algorithm to find out the pattern
    :param pattern_matrix: pattern matrix
    :param weight_matrix: weight matrix
    :param perturbations: number of perturbations in a pattern
    :param max_iter: max iterations to retrieve a pattern
    :return: a list containing True if the pattern is retrieved False else
    """
    answers = []
    x, y = np.shape(pattern_matrix)

    for i in range(x):
        perturbed_pattern = hop.perturb_pattern(pattern_matrix[i], perturbations)
        dynamic = hop.dynamics(perturbed_pattern, weight_matrix, max_iter)
        last_pattern = dynamic[len(dynamic) - 1]
        hop.pattern_match(pattern_matrix, last_pattern)
        answer = False if hop.pattern_match(pattern_matrix, dynamic) is None else True
        answers.append(answer)

    return answers


def test_dynamic_async(pattern_matrix, weight_matrix, perturbations, max_iter, iter_no_change):
    """
    Perturb all the patterns of the pattern matrix and try to retrieve them
    We use the normal algorithm to find out the pattern
    :param pattern_matrix: pattern matrix
    :param weight_matrix: weight matrix
    :param perturbations: number of perturbations in a pattern
    :param max_iter: max iterations to retrieve a pattern
    :param iter_no_change: number of non changing patterns to achieve to stop the convergence algorithm
    :return: a list containing True if the pattern is retrieved False else
    """
    answers = []
    x, y = np.shape(pattern_matrix)

    for i in range(x):
        perturbed_pattern = hop.perturb_pattern(pattern_matrix[0], perturbations)
        dynamic_async = hop.dynamics_async(perturbed_pattern, weight_matrix, max_iter, iter_no_change)
        answer = False if hop.pattern_match(pattern_matrix, dynamic_async) is None else True
        answers.append(answer)

    return answers


def analyse_result(answers):
    """
    Transform the list in some stats that are printed in the console
    :param answers: list of True / False values
    """
    wrong = answers.count(True)
    correct = answers.count(False)
    if correct != 0:
        print(f"{correct} success and {wrong} errors : {100 - (100 * wrong / correct)} % percent")
    else:
        print(f"{correct} success and {wrong} errors")



