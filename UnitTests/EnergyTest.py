from unittest import TestCase
import matplotlib.pyplot as plot


class Test(TestCase):
    def testEnergy(self):
        from Week2 import hopefieldNetwork as hop

        pattern_matrix = hop.generate_patterns(1, 1000)
        print("Computing process started...")
        hebbian_matrix = hop.hebbian_weights(pattern_matrix)
        print("Hebbian matrix was created")
        storkey_matrix = hop.storkey_weights(pattern_matrix)
        print("Storkey matrix was created")

        perturbed_pattern = hop.perturb_pattern(pattern_matrix[0], 80)

        list_hebbian_normal = hop.dynamics(perturbed_pattern, hebbian_matrix, 20)
        energy_hebbian_normal = hop.compute_energy_for_list(list_hebbian_normal, hebbian_matrix)

        list_storkey_normal = hop.dynamics(perturbed_pattern, storkey_matrix, 20)
        energy_storkey_normal = hop.compute_energy_for_list(list_storkey_normal, storkey_matrix)

        list_hebbian_async = hop.dynamics_async(perturbed_pattern, hebbian_matrix, 20000, 3000)
        energy_hebbian_async = hop.compute_energy_for_list(list_hebbian_async, hebbian_matrix)

        list_storkey_async = hop.dynamics_async(perturbed_pattern, storkey_matrix, 20000, 3000)
        energy_storkey_async = hop.compute_energy_for_list(list_storkey_async, storkey_matrix)

        fig, axs = plot.subplots(4)
        fig.set_size_inches(13, 13)
        fig.set_dpi(200)
        axs[0].plot(energy_hebbian_normal, 'tab:blue')
        axs[0].set_title("energy_hebbian_normal")

        axs[1].plot(energy_storkey_normal, 'tab:orange')
        axs[1].set_title("energy_storkey_normal")

        axs[2].plot(energy_hebbian_async, 'tab:green')
        axs[2].set_title("energy_hebbian_async")

        axs[3].plot(energy_storkey_async, 'tab:red')
        axs[3].set_title("energy_storkey_async")

        plot.show()
