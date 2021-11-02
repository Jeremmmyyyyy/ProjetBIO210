import numpy as np

import hopefieldNetwork as hop

# pattern_matrix = hop.generate_patterns(80, 1000)
# weight_matrix = hop.hebbian_weights(pattern_matrix)
#
# perturbed_pattern = hop.perturb_pattern(pattern_matrix[0], 80)
#
# dynamic = hop.dynamics(perturbed_pattern, weight_matrix, 20)
# last_pattern = dynamic[len(dynamic)]
#
# print(hop.pattern_match(pattern_matrix, last_pattern))
# print(hop.pattern_match(pattern_matrix[0], last_pattern))

# pattern_matrix = hop.generate_patterns(1, 3)
# weight_matrix = hop.hebbian_weights(pattern_matrix)
# print(f"pattern_matrix  {pattern_matrix[0]}")
# perturbed_pattern = hop.perturb_pattern(pattern_matrix[0], 1)
# print(f"perturbed_pattern  {perturbed_pattern}")
# dynamic = hop.dynamics(perturbed_pattern, weight_matrix, 20)
# last_pattern = dynamic[len(dynamic)-1]
# print(f"last_pattern  {last_pattern}")
#
# test = hop.pattern_match(pattern_matrix, last_pattern)
# print(test)

# pattern_matrix = hop.generate_patterns(3, 50)
# weight_matrix = hop.hebbian_weights(pattern_matrix)
# perturbed_pattern = hop.perturb_pattern(pattern_matrix[0], 3)
# dynamic = hop.dynamics(perturbed_pattern, weight_matrix, 20)
# last_pattern = dynamic[len(dynamic) - 1]
#
# print(hop.pattern_match(pattern_matrix, last_pattern))

pattern_matrix = hop.generate_patterns(80, 1000)
weight_matrix = hop.hebbian_weights(pattern_matrix)
perturbed_pattern = hop.perturb_pattern(pattern_matrix[0], 80)
dynamic = hop.dynamics(perturbed_pattern, weight_matrix, 20)
last_pattern = dynamic[len(dynamic) - 1]

print(hop.pattern_match(pattern_matrix, last_pattern))
