import numpy as np


def storkey_weights(patterns):
    """
    Apply the Storkey rule on a given patten list
    :param patterns: list of all the patterns to learn
    :return: the weight matrix
    """

    number_of_patterns, size_of_patterns = patterns.shape
    old_weights = np.zeros((size_of_patterns, size_of_patterns))
    new_weights = np.zeros((size_of_patterns, size_of_patterns))
    for pattern in patterns:
        h = compute_h(old_weights, pattern)

        for i in range(size_of_patterns):
            for j in range(size_of_patterns):
                new_weights[i][j] = old_weights[i][j] + (1. / size_of_patterns) * (pattern[i] * pattern[j] - pattern[i]
                                                                                   * h[j][i] - pattern[j] * h[i][j])

        old_weights = new_weights.copy()

    return new_weights


def compute_h(old_weights, pattern):
    size = len(old_weights)
    h = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            for k in range(size):
                if (k is not i) and (k is not j):
                    h[i][j] += old_weights[i][k] * pattern[k]
    return h