import test_functions
import functions
import numpy as np
import matplotlib.pyplot as plt

patterns = functions.generate_patterns(3, 2500)

#TEST 1: hebbian(sync/async)
'''print('TEST 1: hebbian(sync/async)')
test_functions.test_hebbian_sync(patterns, 80)
test_functions.test_hebbian_async(patterns, 80)'''

#TEST 2: storkey (sync/async)
'''print('TEST 2: storkey(sync/async)')
test_functions.test_storkey_sync(patterns)
test_functions.test_storkey_async(patterns)'''

# TEST 3: plotting hebbian sync
'''test_functions.test_plot_hebbian_sync(patterns)'''

# TEST 4: plotting hebbian async
'''test_functions.test_plot_hebbian_async(patterns)'''

# TODO verifier que c'est bon car des fois croissant et pas decroissant...
# TEST 5: plotting storkey sync
'''test_functions.test_plot_storkey_sync(patterns)'''

# TEST 6: plotting storkey async
'''test_functions.test_plot_storkey_async(patterns)'''

# Vizualization of the pattern evolution
pattern_size = 50
sub_matrix_size = 5


def construction_checkerboard(sub_matrix_size):

    sub_matrix_white = np.ones((sub_matrix_size, sub_matrix_size), dtype=int)
    sub_matrix_black = np.full((sub_matrix_size, sub_matrix_size), -1)

    # TODO pas forcément optimal car pas général et pas de submatrix ??
    #construction of the first row
    sub_matrix = np.concatenate((sub_matrix_white, sub_matrix_black), axis=1)
    for i in range(0,sub_matrix_size-1):
        sub_matrix = np.concatenate((sub_matrix,np.concatenate((sub_matrix_white, sub_matrix_black), axis=1)), axis=1)

    #construction of the full checkerboard
    checkerboard = np.concatenate((sub_matrix,np.flip(sub_matrix)), axis=0)
    for i in range(0,sub_matrix_size*2-2):
        checkerboard = np.concatenate((checkerboard,np.flip(sub_matrix)), axis=0)

    return checkerboard


checkerboard = construction_checkerboard(sub_matrix_size)

#flattern the checkerboard into a vector
vector_checkerboard = checkerboard.flatten(order='C')

#replace of one random pattern in patterns
i = np.random.randint(0, np.shape(patterns)[0])
patterns[i] = vector_checkerboard

weights = functions.hebbian_weights(patterns)
vector_checkerboard_modif = functions.perturb_pattern(vector_checkerboard, 1000)

checkerboard_history_sync = functions.dynamics(vector_checkerboard_modif, weights, 20, 1)
checkerboard_history_async = functions.dynamics_async(vector_checkerboard_modif, weights, 20000, 3000, 1000)

for pattern in checkerboard_history_sync:
    pattern = np.reshape(pattern, (50,50))


for pattern in checkerboard_history_async:
    pattern = np.reshape(pattern, (50, 50))