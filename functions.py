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

    def kinetic_energy(self, particles: float) -> float:
        auf = OrbitalSpectrum.aufbau_weights(particles)
        return sum(a * self.orbitals[i].kinetic_energy for i, a in enumerate(auf))

    def kinetic_energy_density(self, particles: float) -> Function:
        auf = OrbitalSpectrum.aufbau_weights(particles)

        ke_dens = sum(-0.5 * a * self.orbitals[i].values * self.orbitals[i].laplacian.values
                      for i, a in enumerate(auf))

        d = self.density(particles)
        ke_dens /= d.values  # + 1e-5
        ke_dens[d.values < 1e-5] = 0.0

        return Function(self.orbitals[0].x, ke_dens)

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
        auf = OrbitalSpectrum.aufbau_weights(particles)
        return Density(self.orbitals[0].x,
                       sum(a * self.orbitals[i].values ** 2 for i, a in enumerate(auf)))

    @staticmethod
    def aufbau_weights(particles: float) -> np.ndarray:
        n_filled = math.floor(particles)
        result = [1.0] * n_filled + [particles - n_filled]
        result = [r for r in result if r > 1e-10]
        return np.array(result)


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

    def on_values_change(self):
        self._particles = None

    @property
    def particles(self) -> float:
        if self._particles is None:
            self._particles = np.trapz(self.values, self.x.values)
        return self._particles

    @particles.setter
    def particles(self, val: float):
        self.values *= val / self.particles
        assert abs(self.particles - val) < 1e-6

    def calculate_potential(self, n_elec_tol: float = 0.01,
                            callback: Callable[[Potential, OrbitalSpectrum, 'Density', 'Density'], None] = None,
                            max_iter=10000
                            ) -> 'Potential':

        v = Potential(self.x)
        v.values = -self.values.copy()

        for iteration in range(max_iter):

            # Diagonalize the potential
            eig = v.calculate_eigenstates()

            # Evaluate the resulting density
            eig_d = eig.density(self.particles)

            assert_all_close(eig_d.particles, self.particles)
            delta_density = eig_d.values - self.values

            if np.trapz(abs(delta_density), self.x.values) < n_elec_tol:
                return v

            v.values += delta_density * 0.5
            v.values -= min(v.values)

            if callback is not None:
                callback(v, eig, self, eig_d)

        return v

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
