from Week2 import testingMethods as testing
from Week2 import hopefieldNetwork as hop


pattern_matrix = hop.generate_patterns(160, 1000)
print("Computing process started...")
hebbian_matrix = hop.hebbian_weights(pattern_matrix)
print("Hebbian matrix was created")
storkey_matrix = hop.storkey_weights(pattern_matrix)
print("Strokey matrix was created")

answers1_hebbian = testing.test_dynamic(pattern_matrix, hebbian_matrix, 80, 20)
testing.analyse_result(answers1_hebbian)
answers1_strokey = testing.test_dynamic(pattern_matrix, storkey_matrix, 80, 20)
testing.analyse_result(answers1_hebbian)

answers2_hebbian = testing.test_dynamic_async(pattern_matrix, hebbian_matrix, 80, 20000, 3000)
testing.analyse_result(answers2_hebbian)
answers2_strokey = testing.test_dynamic_async(pattern_matrix, storkey_matrix, 80, 20000, 3000)
testing.analyse_result(answers2_strokey)





# pattern_matrix, weight_matrix = testing.create_weight(3, 50)
# print("The weight matrix was created")
#
# # answers1 = testing.test_dynamic(pattern_matrix, weight_matrix, 80, 20)
# # testing.analyse_result(answers1)
# answers2 = testing.test_dynamic_async(pattern_matrix, weight_matrix, 1, 100, 10)
# testing.analyse_result(answers2)

