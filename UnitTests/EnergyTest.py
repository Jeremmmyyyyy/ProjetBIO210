from unittest import TestCase
import matplotlib.pyplot as plot


class Test(TestCase):
    def testEnergy(self):
        from Week2 import hopefieldNetwork as hop

        pattern_matrix = hop.generate_patterns(10, 1000)
        print("Computing process started...")
        hebbian_matrix = hop.hebbian_weights(pattern_matrix)
        print("Hebbian matrix was created")
        storkey_matrix = hop.storkey_weights(pattern_matrix)
        print("Storkey matrix was created")

        perturbed_pattern = hop.perturb_pattern(pattern_matrix[0], 80)

        list_hebbian = hop.dynamics(perturbed_pattern, hebbian_matrix, 20)
        energy_hebbian = hop.compute_energy_for_list(list_hebbian, hebbian_matrix)
        print(energy_hebbian)

        list_storkey = hop.dynamics(perturbed_pattern, storkey_matrix, 20)
        energy_storkey = hop.compute_energy_for_list(list_storkey, storkey_matrix)
        print(energy_storkey)

        plot.plot(energy_hebbian)
        plot.plot(energy_storkey)
        plot.ylabel('Test')
        plot.show()



