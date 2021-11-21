import numpy as np


def generate_patterns(num_patterns, pattern_size):
    """
    Generate the patterns to memorize
    :param num_patterns: number of patterns you want to create
    :param pattern_size: size of the patterns
    :return: a 2d array that contains one pattern per line
    """
    if num_patterns <= 0 or pattern_size <= 0:
        raise Exception("ERROR: arguments must be positive")
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

    for mem_pattern in memorized_patterns:
        if (mem_pattern == pattern).all():
            return mem_pattern


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
    """
    Apply the Storkey rule on a given patten list
    :param patterns: list of all the patterns to learn
    :return: the weight matrix
    """

    number_of_patterns, size_of_patterns = patterns.shape
    old_weights = np.zeros((size_of_patterns, size_of_patterns))
    new_weights = np.zeros((size_of_patterns, size_of_patterns))
    for pattern in patterns:
        outer_matrix = np.outer(pattern, pattern)
        h_matrix = compute_h(old_weights.copy(), pattern)
        pre_synaptic_matrix = preAndPost_synaptic_term_computation(pattern, h_matrix.T)
        post_synaptic_matrix = pre_synaptic_matrix.T
        new_weights = old_weights + \
                      (1. / size_of_patterns) * (outer_matrix - pre_synaptic_matrix - post_synaptic_matrix)
        old_weights = new_weights.copy()

    return new_weights


def compute_h(old_weights, pattern):
    np.fill_diagonal(old_weights, 0)
    pattern_matrix = np.broadcast_to(pattern[:, None], (len(pattern), len(pattern))).copy()
    np.fill_diagonal(pattern_matrix, 0)
    h = np.matmul(old_weights, pattern_matrix)
    return h


def preAndPost_synaptic_term_computation(pattern, h):
    synaptic_matrix = np.zeros((len(pattern), len(pattern)))
    for i in range(len(pattern)):
        synaptic_matrix[i] = np.multiply(pattern[i], h[i, :])
    return synaptic_matrix


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
    verifyState = state.copy()
    list_of_iterations.append(verifyState)
    for i in range(max_iter):
        nextState = update(verifyState, weights)
        list_of_iterations.append(nextState)

        if (verifyState == nextState).all():
            break
        verifyState = nextState
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

    current_state = state.copy()
    next_state = update_async(current_state, weights)
    max_iter_counter = 1
    convergence_counter = 0
    list_of_iterations = []

    while max_iter_counter <= max_iter and convergence_counter < convergence_num_iter:
        list_of_iterations.append(current_state)
        current_state = next_state
        next_state = update_async(next_state, weights)
        max_iter_counter += 1

        if (current_state == next_state).all():
            convergence_counter += 1
        else:
            convergence_counter = 0

    list_of_iterations.append(current_state)

    return list_of_iterations


def energy(state, weights):
    wijpi = np.matmul(weights, state.T)
    energy_value = -1 / 2 * np.matmul(wijpi, state.T)
    return energy_value


def compute_energy_for_list(list_of_iterations, weights):
    energy_list = []
    for pattern in list_of_iterations:
        energy_list.append(energy(pattern, weights))
    return energy_list
