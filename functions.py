import math
import numpy as np
from qm1d.hilbert import *
from qm1d.utils import assert_all_close
from typing import List, Tuple, Callable, Iterable


class Orbital(Function):

    @property
    def kinetic_energy(self) -> float:
        return -0.5 * self.inner_product(self.laplacian)


class OrbitalSpectrum:

    def __init__(self, eigenvalues: Iterable[float], orbitals: Iterable[Orbital]):
        self._eigenvalues = list(eigenvalues)
        self._orbitals = list(orbitals)

    @property
    def eigenvalues(self) -> List[float]:
        return self._eigenvalues

    @property
    def orbitals(self) -> List[Orbital]:
        return self._orbitals

    def density(self, particles: float) -> 'Density':
        """
        Evaluate the density that results from filling these
        orbitals with the given number of particles (according to Aufbau scheme).

        Parameters
        ----------
        particles: float
            number of particles to put into these orbitals

        Returns
        -------
        density: Density
            Resulting particle density
        """
        result = Density(self.orbitals[0].x)
        full = math.floor(particles)
        result.values = sum(self.orbitals[i].values ** 2 for i in range(full))
        result.values += (particles - full) * self.orbitals[full].values ** 2
        return result


class Potential(Function):

    def calculate_eigenstates(self) -> OrbitalSpectrum:
        # Evaluate the hamiltonian
        h = -0.5 * self.x.laplacian + np.diag(self.values)
        eigenvalues, eigenvectors = np.linalg.eigh(h)

        orbitals = []
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
            orbitals.append(orb_i)

        return OrbitalSpectrum(eigenvalues, orbitals)


class Density(Function):

    @property
    def particles(self) -> float:
        return np.trapz(self.values, self.x.values)

    @particles.setter
    def particles(self, val: float):
        self.values /= self.particles
        self.values *= val
        assert_all_close(self.particles, val)

    def calculate_potential(self, n_elec_tol: float = 0.01,
                            callback: Callable[
                                [Potential, OrbitalSpectrum, 'Density', 'Density'], None] = None) -> 'Potential':

        v = Potential(self.x)
        v.values = -self.values.copy()

        while True:

            # Diagonalize the potential
            eig = v.calculate_eigenstates()

            # Evaluate the resulting density
            eig_d = eig.density(self.particles)

            assert_all_close(eig_d.particles, self.particles)
            delta_density = eig_d.values - self.values

            if np.trapz(abs(delta_density), self.x.values) < n_elec_tol:
                return v

            v.values += delta_density
            v.values -= min(v.values)

            if callback is not None:
                callback(v, eig, self, eig_d)

    @staticmethod
    def calculate_potential_animation_callback(v: Potential, eig: OrbitalSpectrum, targ_d: 'Density', eig_d: 'Density'):
        import matplotlib.pyplot as plt
        plt.ion()
        plt.clf()

        plt.subplot(221)
        plt.fill_between(v.x.values, v.values * 0, v.values)  #
        plt.ylabel(r"$v$")
        plt.xlabel(r"$x$")

        plt.subplot(222)
        for i in range(math.ceil(targ_d.particles)):
            plt.plot(v.x.values, eig.orbitals[i].values)
        plt.ylabel(r"$\phi$")
        plt.xlabel(r"$x$")

        plt.subplot(223)
        plt.plot(v.x.values, eig_d.values, color="black", label=r"$\rho_\phi$")
        plt.plot(v.x.values, targ_d.values, color="black", linestyle=":", label=r"$\rho$")
        plt.ylabel(r"$\rho$")
        plt.xlabel(r"$x$")
        plt.legend()

        plt.tight_layout()

        plt.pause(0.001)
