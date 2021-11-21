import random
import numpy as np


def generate_patterns(num_patterns, pattern_size):
    """
    This function creates a matrix. Its rows number is num_patterns and its columns number is pattern_size.
    Each row represents a pattern and the number of columns is the number of different features that represent each pattern.
    Each row is a random binary pattern (possible values: {1, −1}) of size pattern_size
    Parameters
    ----
    :param num_patterns: integer that represents the number of row in the matrix
    :param pattern_size: integer that represents the number of columns in the matrix
    Returns
    ----
    :return: returns a 2-dimensional numpy array containing random binary patterns
    """
    return np.random.choice(np.array([-1, 1]), (num_patterns, pattern_size))


# TODO we don't want that a certain position of a pattern can change more than once for a total number of perturbation
def perturb_pattern(pattern, num_perturb):
    """
    This function perturbs a given pattern.
    It samples num_perturb elements of the input pattern uniformly at random and changes their sign.
    Parameters
    ----
    :param pattern: a binary vector representing a pattern
    :param num_perturb: integer which is the number of perturbation(s) wanted in the pattern
    Returns
    ----
    :return: returns the perturbed pattern
    """
    for i in range(num_perturb):
        n = random.randrange(0, len(pattern)-1)
        if pattern[n] == -1:
            pattern[n] = 1
        elif pattern[n] == 1:
            pattern[n] = -1
    return pattern


def pattern_match(memorized_patterns, pattern):
    """
    Match a pattern with the corresponding memorized one.
    In others words, this function checks if pattern is in the memorized_patterns matrix.
    Parameters
    ----
    :param memorized_patterns: 2-dimensional numpy array which represents the initially memorized patterns
    :param pattern: a vector (numpy array) which represents the pattern that we want to know if it is in the memorized_patterns
    Returns
    ----
    :return: returns None if no memorized pattern (in memorized_patterns) matches,
             otherwise it returns the index of the row corresponding to the matching pattern (in memorized_patterns) .
    """
    '''for line in memorized_patterns:
        if (line == pattern).all():
            return line'''

    for i in range(0, np.shape(memorized_patterns)[0]):
        if (memorized_patterns[i] == pattern).all():
            return i


def hebbian_weights(patterns):
    """
    This function applies the hebbian learning rule on some given patterns to create the weight matrix (numpy array).
    Parameters
    ----
    :param patterns: 2-dimensional numpy array which represents all the memorized patterns
    Returns
    ----
    :return: returns a 2-dimensional numpy array which is the Hebbian weights matrix
    """
    N_rows = np.shape(patterns)[0]
    N_columns = np.shape(patterns)[1]
    weights_matrix = np.zeros((N_columns, N_columns))
    for i in range(N_columns):
        for j in range(N_columns):
            wij = 0
            for mu in range(N_rows):
                wij += patterns[(mu, i)] * patterns[(mu, j)]
            if i != j:
                weights_matrix[(i, j)] = wij / N_rows
    return weights_matrix


def sigma(vector):
    """
    This functions allows to compute a binary vector from a vector containing floats.
    All the vector's elements bigger or equal to 0 become 1 and the elements strictly smaller than 0 become -1.
    Parameters
    ----
    :param vector: a vector (numpy array)
    Returns
    ----
    :return: returns the binary vector corresponding to the initial vector
    """
    condition_vector = (vector >= 0).astype(int)
    condition_vector[condition_vector == 0] = -1
    return condition_vector


def update(state, weights):
    """
    This function applies the update rule to a state pattern.
    It computes the dot product between state and weights. Then it calls the sigma function
    to make a binary vector from the initial one.
    Parameters
    ----
    :param state: a vector (numpy array) which represents the network state.
    :param weights: a 2-dimensional numpy array corresponding to the weights matrix of the memorized patterns
    Returns
    ----
    :return: return a vector (numpy array) representing the new state of the network after having apply the update rule.
    """
    return sigma(np.matmul(weights, state))


def update_async(state, weights):
    """
    This function applies the asynchronous update rule to a state pattern.
    However, in-stead of computing the full update p(t+1) = Wp(t), it just updates the i-th component
    of the state vector (with i sampled uniformly at random) by computing the new value p[i](t+1) = w[i] · p(t)
    In the previous expression, w[i] denotes the i-th row of the matrix weights.
    So in others words it computes the dot product between a random element of state and weights.Then it calls
    the sigma function to make a binary vector from the initial one.
    Parameters
    ----
    :param state: a vector (numpy array) which represents the network state.
    :param weights: a 2-dimensional numpy array corresponding to the weights matrix of the memorized patterns
    Returns
    ----
    :return: return a vector (numpy array) representing the new state of the network after having apply
             the asynchronous update rule.
    """
    i = np.random.randint(0, len(weights))
    state_copy = state.copy()
    state_copy[i] = sigma(np.array([np.matmul(np.reshape(weights[i], (1, len(state))), state_copy)]))
    return state_copy


def dynamics(state, weights, max_iter):
    """
    This function runs the dynamical system from an initial state until convergence
    or until a maximum number of steps is reached using the update rule.
    Convergence is achieved when two consecutive updates return the same state.
    Parameters
    ----
    :param state: a vector (numpy array) which represents the network state
    :param weights: a 2-dimensional numpy array corresponding to the weights matrix of the memorized patterns
    :param max_iter: an integer that represents the maximum number of iterations that could be done
    Returns
    ----
    :return: returns a list with the whole state history.
    """
    p0 = state
    p1 = update(p0, weights)
    T = 1
    history_state = []
    while T <= max_iter and not (p0 == p1).all():
        history_state.append(p0)
        p0 = p1
        p1 = update(p1, weights)
        T = T + 1
    history_state.append(p0)
    return history_state


def dynamics_async(state, weights, max_iter, convergence_num_iter):
    """
    This function runs the dynamical system from an initial state until a maximum number of steps is reached
    using the asynchronous update rule.
    With the asynchronous update rule, we can set a softer convergence criterion :
    If the solution does not change for convergence_num_iter steps in a row, then we can say
    that the algorithm has reached convergence.
    Parameters
    ----
    :param state: a vector (numpy array) which represents the network state
    :param weights: a 2-dimensional numpy array corresponding to the weights matrix of the memorized patterns
    :param max_iter: an integer that represents the maximum number of iteration that could be done
    :param convergence_num_iter: an integer that represents the minimal number of iterations during which,
           the solution must not change to reached convergence.
    Returns
    ----
    :return: returns a list with the whole state history.
    """
    p0 = state
    p1 = update_async(p0, weights)
    T = 1
    convergence_counter = 0
    history_state = []

    while T <= max_iter or convergence_counter < convergence_num_iter:
        history_state.append(p0)
        p0 = p1
        p1 = update_async(p1, weights)
        T = T + 1

        if (p0 == p1).all():
            convergence_counter += 1
        else:
            convergence_counter = 0
    history_state.append(p0)
    return history_state


def storkey_weights(patterns):
    """
    This function applies the Storkey learning rule on some given patterns to create the weight matrix (numpy array).
    Parameters
    ----
    :param patterns: 2-dimensional numpy array which represents all the memorized patterns
    Returns
    ----
    :return: returns a 2-dimensional numpy array which is the Storkey weights matrix
    """
    number_patterns, size_patterns = patterns.shape
    old_weights = np.zeros((size_patterns, size_patterns))

    for pattern in patterns:
        first_term = np.outer(pattern, pattern)
        h = np.matmul(old_weights, pattern)
        second_term = np.matmul(pattern, h)
        third_term = second_term.T

        new_weights = old_weights + (1. / size_patterns) * (first_term - second_term - third_term)
        old_weights = new_weights

    np.fill_diagonal(old_weights, 0)
    return old_weights


def energy(state, weights):
    """
    This function computes the energy of a state.
    This calculation is called energy because the patterns which are memorized, either with
    the Hebbian or with the Storkey rule, are local minima of this function.
    Furthermore, the energy is a non-increasing quantity of the dynamical system, meaning that the energy
    at time step t is always greater or equal than the energy at any subsequent step t′ > t.
    Parameters
    ----
    :param state: a vector (numpy array) which represents the network state.
    :param weights: a 2-dimensional numpy array corresponding to the weights matrix of the memorized patterns
    Returns
    ----
    :return: returns an float which represents the energy of the state
    """
    e_sum = 0
    for i in range(0, np.shape(state)[0]):
        for j in range(0, np.shape(state)[0]):
            e_sum += weights[i][j] * state[i] * state[j]
    return (-1/2) * e_sum


def energies(states, weights):
    """
    This function computes the energy of each state contained in the states (matrix)
    Parameters
    ----
    :param states: a 2-dimensional numpy array containing states (one state is in one row)
    :param weights: a 2-dimensional numpy array corresponding to the weights matrix of the memorized patterns
    Returns
    ----
    :return: returns a list containing the energies of all the states. The length of the list is the number
             of states contained in states (the number of rows).
    """
    energy_list = []
    for i in range(0, np.shape(states)[0]):
        energy_list.append(energy(states[i], weights))
    return energy_list