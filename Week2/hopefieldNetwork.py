import numpy as np


def generate_patterns(num_patterns, pattern_size):
    """
    Generate the patterns to memorize
    :param num_patterns: number of patterns you want to create
    :param pattern_size: size of the patterns
    :return: a 2d array that contains one pattern per line
    """
    return np.random.choice([-1, 1], (num_patterns, pattern_size))


def perturb_pattern(pattern, num_perturb):
    """
    Modify a given pattern with a given amount of changes (flips)
    :param pattern: pattern to modify
    :param num_perturb: number of flips/ perturbations to achieve
    :return: the modified pattern
    """
    i = 0
    new_pattern = pattern.copy()
    changedIndexes = []
    while i != num_perturb:
        randomPosition = np.random.randint(0, len(new_pattern))
        if not changedIndexes.count(randomPosition) > 0:
            new_pattern[randomPosition] = -1 * new_pattern[randomPosition]
            changedIndexes.append(randomPosition)
            i += 1
    return new_pattern


def pattern_match(memorized_patterns, pattern):
    """
    Match a pattern with the memorized one
    :param memorized_patterns: matrix of the initially memorized patterns
    :param pattern: to check
    :return: true if pattern is in memorize_patterns else false
    """

    # print(f"memorized {memorized_patterns} and {pattern}")
    for mem_pattern in memorized_patterns:
        if (mem_pattern == pattern).all():
            return True

    return False
    # TODO update
    # return any((memorized_patterns[:] == pattern).all(1))


def hebbian_weights(patterns):
    """
    Apply the Hebbian rule on a given patten list
    :param patterns: list of all the patterns to learn
    :return: the weight matrix
    Specifications : the algorithm calculates only the upper part of the matrix
    given that it is symmetric and has only zeros in the diagonal
    """
    number_of_patterns, n = patterns.shape
    w = np.zeros((n, n))
    rows, columns = w.shape
    for i in range(rows):
        for j in range(columns):
            matrix_element = 0
            for pattern in range(number_of_patterns):
                if i < j:
                    matrix_element += patterns[(pattern, i)] * patterns[(pattern, j)]
                w[(i, j)] = matrix_element / number_of_patterns
    w += w.T

    return w


def storkey_weights(patterns):
    number_of_patterns, size_of_patterns = patterns.shape
    old_weights = np.zeros((size_of_patterns, size_of_patterns))

    for pattern in patterns:
        first_term = np.outer(pattern, pattern)

        h = np.matmul(old_weights, pattern)

        second_term = np.matmul(pattern, h)

        third_term = np.matmul(h, pattern)

        new_weights = old_weights + (1. / size_of_patterns) * (first_term - second_term - third_term)
        old_weights = new_weights

    np.fill_diagonal(old_weights, 0)
    return old_weights

    # mem = patterns.flatten()
    # n = len(mem)
    #
    # old_weights = np.zeros(n)
    #
    # hebbian_term = np.outer(mem, mem) - np.identity(n)
    #
    # net_inputs = old_weights.dot(mem)
    #
    # pre_synaptic = np.outer(mem, net_inputs)
    # post_synaptic = pre_synaptic.T  # equivalent to np.outer(net_inputs,mem)
    #
    # new_weights = old_weights + (1. / n) * (hebbian_term - pre_synaptic - post_synaptic)


def update(state, weights):
    """
    Apply the update rule to a pattern given a weight matrix
    :param state: pattern to update
    :param weights: matrix
    :return: the updated pattern after 1 iteration of the process
    """
    newState = np.matmul(weights, state)

    for i in range(len(newState)):
        if newState[i] < 0:
            newState[i] = -1
        else:
            newState[i] = 1

    return newState


def update_async(state, weights):
    """
    Only one element of the state is updated with help of the corresponding i-th row of the matrix
    :param state: pattern to update
    :param weights: matrix
    :return: the updated pattern with 1 row of the matrix after 1 iteration of the process
    """
    newState = state.copy()
    random_index = np.random.randint(0, len(state))

    newState[random_index] = -1 if np.matmul(weights[random_index], state) < 0 else 1
    return newState


def dynamics(state, weights, max_iter):
    """
    Update the pattern until convergence to a memorized pattern
    :param state: pattern to analyse
    :param weights: matrix
    :param max_iter: maximum number of iteration if there is no convergence
    :return: the whole state/pattern history
    """
    list_of_iterations = []

    for i in range(max_iter):
        nextState = update(state, weights)
        list_of_iterations.append(nextState)

        if (state == nextState).all():
            break
        state = nextState

    return list_of_iterations


def dynamics_async(state, weights, max_iter, convergence_num_iter):
    """
    Update the pattern until convergence (repetition of the pattern x time)
    :param state: pattern to analyse
    :param weights: matrix
    :param max_iter: maximum number of iteration if there is no convergence
    :param convergence_num_iter: convergence is defined when the pattern is repeated convergence_num_iter times
    :return: the whole state/pattern history
    """
    list_of_iterations = np.array(state)
    convergence_counter = 0

    for i in range(max_iter):
        nextState = update_async(state, weights)
        np.append(list_of_iterations, nextState)

        if (state == nextState).all():
            convergence_counter += 1
        else:
            state = nextState

        if convergence_counter == convergence_num_iter:
            break

    return list_of_iterations
