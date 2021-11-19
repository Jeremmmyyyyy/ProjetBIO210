import functions
import numpy as np
import matplotlib.pyplot as plt

def test_hebbian_sync(patterns, nbr_pertubation):
    weight_matrix = functions.hebbian_weights(patterns)

    j = np.random.randint(0, len(patterns))
    base_pattern = patterns[j]

    base_pattern_modif = functions.perturb_pattern(base_pattern, nbr_pertubation)

    patterns_new_list = functions.dynamics(base_pattern_modif, weight_matrix, 20, 1)

    if functions.pattern_match(patterns, patterns_new_list[-1]) != None:
        print("Original network retrieved!!!!! [synchronous update]")
    else:
        print("FALSE!!")


def test_hebbian_async(patterns, nbr_pertubation):
    weight_matrix = functions.hebbian_weights(patterns)

    j = np.random.randint(0, len(patterns))
    base_pattern = patterns[j]

    base_pattern_modif = functions.perturb_pattern(base_pattern, nbr_pertubation)

    patterns_new_list_async = functions.dynamics_async(base_pattern_modif, weight_matrix, 20000, 3000,1)

    if functions.pattern_match(patterns, patterns_new_list_async[-1]) != None:
        print("Original network retrieved!!!!! [asynchronous update]")
    else:
        print("FALSE!!")


def test_storkey_sync(patterns):
    weight_matrix = functions.storkey_weights(patterns)

    j = np.random.randint(0, len(patterns))
    base_pattern = patterns[j]

    base_pattern_modif = functions.perturb_pattern(base_pattern, 80)

    patterns_new_list = functions.dynamics(base_pattern_modif, weight_matrix, 20,1)

    if functions.pattern_match(patterns, patterns_new_list[-1]) != None:
        # if np.equal(patterns_new_matrix, patterns).all():
        print("Original network retrieved!!!!! [synchronous update]")
    else:
        print("FALSE!!")


def test_storkey_async(patterns):
    weight_matrix = functions.storkey_weights(patterns)

    j = np.random.randint(0, len(patterns))
    base_pattern = patterns[j]

    base_pattern_modif = functions.perturb_pattern(base_pattern, 80)

    patterns_new_list_async = functions.dynamics_async(base_pattern_modif, weight_matrix, 20000, 3000, 1)

    if functions.pattern_match(patterns, patterns_new_list_async[-1]) != None:
        print("Original network retrieved!!!!! [asynchronous update]")
    else:
        print("FALSE!!")


def test_plot_hebbian_sync(patterns):
    weight_matrix = functions.hebbian_weights(patterns)

    j = np.random.randint(0, len(patterns))
    base_pattern = patterns[j]

    base_pattern_modif = functions.perturb_pattern(base_pattern, 80)

    patterns_new_list = functions.dynamics(base_pattern_modif, weight_matrix, 20, 1)

    x = functions.energies(patterns_new_list, weight_matrix)
    plt.plot(x, 'b')
    plt.title('Time-energy plot Hebbian/sync')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.grid(False)
    plt.show()


def test_plot_hebbian_async(patterns):
    weight_matrix = functions.hebbian_weights(patterns)

    j = np.random.randint(0, len(patterns))
    base_pattern = patterns[j]

    base_pattern_modif = functions.perturb_pattern(base_pattern, 80)

    patterns_new_list_async = functions.dynamics_async(base_pattern_modif, weight_matrix, 20000, 3000, 1)

    x = functions.energies(patterns_new_list_async, weight_matrix)
    plt.plot(x, 'b')
    plt.title('Time-energy plot Hebbian/async')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.grid(False)
    plt.show()


def test_plot_storkey_sync(patterns):
    weight_matrix = functions.storkey_weights(patterns)

    j = np.random.randint(0, len(patterns))
    base_pattern = patterns[j]

    base_pattern_modif = functions.perturb_pattern(base_pattern, 80)

    patterns_new_list = functions.dynamics(base_pattern_modif, weight_matrix, 20, 1)

    x = functions.energies(patterns_new_list, weight_matrix)
    plt.plot(x, 'b')
    plt.title('Time-energy plot Storkey/sync')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.grid(False)
    plt.show()


def test_plot_storkey_async(patterns):
    weight_matrix = functions.storkey_weights(patterns)

    j = np.random.randint(0, len(patterns))
    base_pattern = patterns[j]

    base_pattern_modif = functions.perturb_pattern(base_pattern, 80)

    patterns_new_list_async = functions.dynamics_async(base_pattern_modif, weight_matrix, 20000, 3000, 1)

    x = functions.energies(patterns_new_list_async, weight_matrix)
    plt.plot(x, 'b')
    plt.title('Time-energy plot Storkey/async')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.grid(False)
    plt.show()