import numpy as np
from qm1d.hilbert import *
from qm1d.utils import assert_all_close
from typing import List, Tuple


class Density(Function):

    @property
    def particles(self) -> float:
        return np.trapz(self.values, self.x.values)

    @particles.setter
    def particles(self, val: float):
        self.values /= self.particles
        self.values *= val
        assert_all_close(self.particles, val)

    def calculate_potential(self) -> 'Potential':

        import matplotlib.pyplot as plt

        v = Potential(self.x)
        v.values = -self.values.copy()

        plt.ion()

        while True:
            eig = v.calculate_eigenstates()

            eig_d = Density(self.x)

            for i in range(5):
                eig_d.values += eig[i][1].values ** 2
                plt.plot(self.x.values, eig[i][1].values)

            assert_all_close(eig_d.particles, self.particles)

            plt.plot(self.x.values, v.values / (max(v.values) - min(v.values)), color="red", linestyle=":")

            plt.pause(0.01)
            plt.clf()

            plt.plot(self.x.values, eig_d.values, color="black")
            plt.plot(self.x.values, self.values, color="black", linestyle=":")

            delta_density = eig_d.values - self.values

            if np.trapz(abs(delta_density), self.x.values) < 0.01:
                break

            v.values += delta_density

            plt.show()


class Orbital(Function):
    pass


class Potential(Function):

    def calculate_eigenstates(self) -> List[Tuple[float, Orbital]]:
        # Evaluate the hamiltonian
        h = -0.5 * self.x.laplacian + np.diag(self.values)
        eigenvalues, eigenvectors = np.linalg.eigh(h)

        result = []
        for i in range(self.x.points):

            vals_i = eigenvectors[:, i]

            # Ensure consistent sign of orbitals
            for j in range(len(vals_i)):
                if abs(vals_i[j]) > 0.0001:
                    vals_i *= np.sign(vals_i[j])
                    break

            orb_i = Orbital(self.x)
            orb_i.values = vals_i
            orb_i.values /= orb_i.inner_product(orb_i) ** 0.5
            result.append((eigenvalues[i], orb_i))

        return result
