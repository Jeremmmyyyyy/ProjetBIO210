from Week2 import testingMethods as testing

pattern_matrix, weight_matrix = testing.create_weight(80, 1000)
print("The weight matrix was created")

answers1 = testing.test_dynamic(pattern_matrix, weight_matrix, 80, 20)
testing.analyse_result(answers1)
answers2 = testing.test_dynamic_async(pattern_matrix, weight_matrix, 80, 20000, 3000)
testing.analyse_result(answers2)

# pattern_matrix, weight_matrix = testing.create_weight(3, 50)
# print("The weight matrix was created")
#
# # answers1 = testing.test_dynamic(pattern_matrix, weight_matrix, 80, 20)
# # testing.analyse_result(answers1)
# answers2 = testing.test_dynamic_async(pattern_matrix, weight_matrix, 1, 100, 10)
# testing.analyse_result(answers2)
