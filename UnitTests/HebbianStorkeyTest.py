from unittest import TestCase

import numpy as np


class Test(TestCase):
    def testAlgorithms(self):
        from Week2 import testingMethods as testing
        from Week2 import hopefieldNetwork as hop

        pattern_matrix = hop.generate_patterns(80, 1000)
        print("Computing process started...")
        hebbian_matrix = hop.hebbian_weights(pattern_matrix)
        print("Hebbian matrix was created")
        storkey_matrix = hop.storkey_weights(pattern_matrix)
        print("Storkey matrix was created")

        answers1_hebbian = testing.test_dynamic_hop(pattern_matrix, hebbian_matrix, 80, 20)
        testing.analyse_result(answers1_hebbian)
        answers1_strokey = testing.test_dynamic_hop(pattern_matrix, storkey_matrix, 80, 20)
        testing.analyse_result(answers1_strokey)

        answers2_hebbian = testing.test_dynamic_async_hop(pattern_matrix, hebbian_matrix, 80, 20000, 3000)
        testing.analyse_result(answers2_hebbian)
        answers2_strokey = testing.test_dynamic_async_hop(pattern_matrix, storkey_matrix, 80, 20000, 3000)
        testing.analyse_result(answers2_strokey)

    def testAlgorithmsShort(self):
        from Week2 import testingMethods as testing
        from Week2 import hopefieldNetwork as hop

        pattern_matrix = hop.generate_patterns(3, 50)
        print("Computing process started...")
        hebbian_matrix = hop.hebbian_weights(pattern_matrix)
        print("Hebbian matrix was created")
        storkey_matrix = hop.storkey_weights(pattern_matrix)
        print("Storkey matrix was created")

        answers1_hebbian = testing.test_dynamic_hop(pattern_matrix, hebbian_matrix, 3, 20)
        testing.analyse_result(answers1_hebbian)
        answers1_strokey = testing.test_dynamic_hop(pattern_matrix, storkey_matrix, 3, 20)
        testing.analyse_result(answers1_strokey)

        answers2_hebbian = testing.test_dynamic_async_hop(pattern_matrix, hebbian_matrix, 3, 20000, 3000)
        testing.analyse_result(answers2_hebbian)
        answers2_strokey = testing.test_dynamic_async_hop(pattern_matrix, storkey_matrix, 3, 20000, 3000)
        testing.analyse_result(answers2_strokey)

    def testAlgorithmsVeryShort(self):
        from Week2 import testingMethods as testing
        from Week2 import hopefieldNetwork as hop

        pattern_matrix = hop.generate_patterns(3, 10)
        print("Computing process started...")
        hebbian_matrix = hop.hebbian_weights(pattern_matrix)
        print("Hebbian matrix was created")
        storkey_matrix = hop.storkey_weights(pattern_matrix)
        print("Storkey matrix was created")

        answers1_hebbian = testing.test_dynamic_hop(pattern_matrix, hebbian_matrix, 1, 20)
        testing.analyse_result(answers1_hebbian)
        answers1_strokey = testing.test_dynamic_hop(pattern_matrix, storkey_matrix, 1, 20)
        testing.analyse_result(answers1_strokey)

        answers2_hebbian = testing.test_dynamic_async_hop(pattern_matrix, hebbian_matrix, 1, 20000, 3000)
        testing.analyse_result(answers2_hebbian)
        answers2_strokey = testing.test_dynamic_async_hop(pattern_matrix, storkey_matrix, 1, 20000, 3000)
        testing.analyse_result(answers2_strokey)

    def testStorkey(self):
        from Week2 import testingMethods as testing
        from Week2 import hopefieldNetwork as hop
        answers = []
        pattern_matrix = hop.generate_patterns(80, 1000)
        storkey_weights = hop.storkey_weights(pattern_matrix)
        print(storkey_weights)
        for i in range(0, len(pattern_matrix)):
            perturbed_pattern = hop.perturb_pattern(pattern_matrix[i], 80)
            listTest = hop.dynamics(perturbed_pattern, storkey_weights, 20)

            if hop.pattern_match(pattern_matrix, list[len(listTest) - 1]) is None:
                answers.append(False)
            else:
                answers.append(True)
        print(answers)
        testing.analyse_result(answers)

