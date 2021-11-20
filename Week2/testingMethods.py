from Week2 import hopefieldNetwork as hop


def test_dynamic_hop(pattern_matrix, weight_matrix, perturbations, max_iter):
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
    for i in range(len(pattern_matrix)):
        perturbed_pattern = hop.perturb_pattern(pattern_matrix[i], perturbations)
        dynamic = hop.dynamics(perturbed_pattern, weight_matrix, max_iter)
        last = dynamic[len(dynamic) - 1]
        answer = hop.pattern_match(pattern_matrix, last)
        answers.append(False if answer is None else True)
    return answers


def test_dynamic_async_hop(pattern_matrix, weight_matrix, perturbations, max_iter, iter_no_change):
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
    for i in range(len(pattern_matrix)):
        perturbed_pattern = hop.perturb_pattern(pattern_matrix[i], perturbations)
        dynamic = hop.dynamics_async(perturbed_pattern, weight_matrix, max_iter, iter_no_change)
        last = dynamic[len(dynamic) - 1]
        answer = hop.pattern_match(pattern_matrix, last)
        answers.append(False if answer is None else True)
    return answers


def analyse_result(answers):
    """
    Transform the list in some stats that are printed in the console
    :param answers: list of True / False values
    """
    wrong = answers.count(False)
    correct = answers.count(True)
    if correct != 0:
        print(f"{correct} success and {wrong} errors : {100 - (100 * wrong / correct)} % percent")
    else:
        print(f"{correct} success and {wrong} errors")
