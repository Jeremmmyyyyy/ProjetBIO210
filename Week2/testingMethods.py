import numpy as np

from Week2 import hopefieldNetwork as hop

def generate_patterns(num_of_patterns, size_of_patterns):
    return hop.generate_patterns(num_of_patterns, size_of_patterns)

def create_weight_hebbian(num_of_patterns, size_of_patterns):

    weight_matrix = hop.hebbian_weights(pattern_matrix)
    return pattern_matrix, weight_matrix


def create_weight_storkey():

    weight_matrix = hop.storkey_weights(pattern_matrix)
    return pattern_matrix, weight_matrix


def test_dynamic(pattern_matrix, weight_matrix, perturbations, max_iter):
    answers = []
    x, y = np.shape(pattern_matrix)

    for i in range(x):
        perturbed_pattern = hop.perturb_pattern(pattern_matrix[i], perturbations)
        dynamic = hop.dynamics(perturbed_pattern, weight_matrix, max_iter)
        last_pattern = dynamic[len(dynamic) - 1]
        hop.pattern_match(pattern_matrix, last_pattern)
        answers.append(hop.pattern_match(pattern_matrix, last_pattern))

    return answers


def test_dynamic_async(pattern_matrix, weight_matrix, perturbations, max_iter, iter_no_change):
    answers = []
    x, y = np.shape(pattern_matrix)

    for i in range(x):
        perturbed_pattern = hop.perturb_pattern(pattern_matrix[0], perturbations)
        dynamic_async = hop.dynamics_async(perturbed_pattern, weight_matrix, max_iter, iter_no_change)
        # last_pattern_async = dynamic_async(len(dynamic_async) - 1)
        answers.append(hop.pattern_match(pattern_matrix, dynamic_async))

    return answers


def analyse_result(answers):
    correct = answers.count(True)
    wrong = answers.count(False)
    if correct != 0:
        print(f"{correct} success and {wrong} errors : {100 - (100 * wrong / correct)} % percent")
    else:
        print(f"{correct} success and {wrong} errors")



