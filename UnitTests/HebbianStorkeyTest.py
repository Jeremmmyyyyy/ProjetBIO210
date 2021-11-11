from unittest import TestCase


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

        answers1_hebbian = testing.test_dynamic(pattern_matrix, hebbian_matrix, 80, 20)
        testing.analyse_result(answers1_hebbian)
        answers1_strokey = testing.test_dynamic(pattern_matrix, storkey_matrix, 80, 20)
        testing.analyse_result(answers1_strokey)

        answers2_hebbian = testing.test_dynamic_async(pattern_matrix, hebbian_matrix, 80, 20000, 3000)
        testing.analyse_result(answers2_hebbian)
        answers2_strokey = testing.test_dynamic_async(pattern_matrix, storkey_matrix, 80, 20000, 3000)
        testing.analyse_result(answers2_strokey)
