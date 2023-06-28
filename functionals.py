import math

import numpy as np
from functions import *
from typing import Iterable


class Functional(ABC):

    @abstractmethod
    def apply(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)


class EnergyDensityFunctional(Functional, ABC):

    @abstractmethod
    def apply(self, density: Density) -> float:
        raise NotImplementedError()


class PotentialDensityFunctional(Functional, ABC):

    @abstractmethod
    def apply(self, density: Density) -> Potential:
        raise NotImplementedError()


class ExternalPotential(EnergyDensityFunctional):

    def __init__(self, v_ext: Potential):
        self._v_ext = v_ext

    @property
    def v_ext(self) -> Potential:
        return self._v_ext

    def apply(self, density: Density) -> float:
        return density.inner_product(self.v_ext)


class ExactKineticEnergyFunctional(EnergyDensityFunctional):

    def __init__(self, n_elec_tol=0.1, max_iter=10):
        self._n_elec_tol = n_elec_tol
        self._max_iter = max_iter

    def apply(self, density: Density) -> float:
        v = density.calculate_potential(n_elec_tol=self._n_elec_tol,
                                        max_iter=self._max_iter)
        e = v.calculate_eigenstates()
        return e.kinetic_energy(density.particles)


class GuassianRepulsion(PotentialDensityFunctional):

    def apply(self, density: Density) -> Potential:
        pot = Potential(density.x)

        for i, x in enumerate(pot.x.values):
            dxs = abs(pot.x.values - x)
            integrand = density.values * np.exp(-dxs ** 2)
            pot.values[i] = np.trapz(integrand, pot.x.values)

        return pot


class SoftenedCoulombRepulsion(PotentialDensityFunctional):
    def apply(self, density: Density) -> Potential:
        pot = Potential(density.x)

        for i, x in enumerate(pot.x.values):
            dxs = abs(pot.x.values - x)
            integrand = density.values / (1 + dxs)
            pot.values[i] = np.trapz(integrand, pot.x.values)

        return pot


class VonWeizakerKE(EnergyDensityFunctional):

    def apply(self, density: Density) -> float:
        sqrt = Density(density.x, density.values ** 0.5)
        return -0.5 * sqrt.inner_product(sqrt.laplacian)


def _minimize_density_functional_density(x: np.ndarray, particles: float, grid: Grid) -> Density:
    d = Density(grid, x ** 2)
    d.particles = particles
    return d


def _minimize_density_functional_cost(
        x: np.ndarray,
        particles: float,
        grid: Grid,
        energy_functionals: Iterable[EnergyDensityFunctional] = None,
        potential_functionals: Iterable[PotentialDensityFunctional] = None) -> float:
    d = _minimize_density_functional_density(x, particles, grid)
    return sum(f(d) for f in energy_functionals) + \
        sum(ExternalPotential(p(d))(d) for p in potential_functionals)


def minimize_density_functional(
        particles: float,
        grid: Grid,
        energy_functionals: Iterable[EnergyDensityFunctional] = None,
        potential_functionals: Iterable[PotentialDensityFunctional] = None,
        plot=False) -> Density:
    from scipy.optimize import minimize

    energy_functionals = energy_functionals or []
    potential_functionals = potential_functionals or []

    def scipy_cost(x: np.ndarray):
        return _minimize_density_functional_cost(x, particles, grid, energy_functionals, potential_functionals)

    def scipy_gadient(x: np.ndarray) -> np.ndarray:
        from multiprocessing import Pool, cpu_count

        dn = np.identity(len(x)) * 1e-6
        f0 = _minimize_density_functional_cost(x, particles, grid, energy_functionals, potential_functionals)

        with Pool(cpu_count()) as p:
            fnew = p.starmap(_minimize_density_functional_cost,
                             [[x + dn[i], particles, grid, energy_functionals, potential_functionals] for i in
                              range(len(x))])

        g = (np.array(fnew) - f0) / dn[0, 0]
        return g

    history = []

    def callback(x):
        d = _minimize_density_functional_density(x, particles, grid)

        if not plot:
            return

        import matplotlib.pyplot as plt
        plt.ion()
        plt.clf()

        plt.subplot(221)
        plt.plot(d.x.values, d.values)

        plt.subplot(222)
        for v in energy_functionals:
            if isinstance(v, ExternalPotential):
                plt.plot(v.v_ext.x.values, v.v_ext.values)

        for v in potential_functionals:
            plt.plot(v(d).x.values, v(d).values)

        plt.subplot(223)
        history.append([v(d) for v in energy_functionals] +
                       [ExternalPotential(p(d))(d) for p in potential_functionals])

        history[-1].append(sum(history[-1]))
        plt.plot(history)
        plt.pause(0.01)

    width = (max(grid.values) - min(grid.values)) / 4.0
    guess = np.exp(-(grid.values / width) ** 2)

    res = minimize(scipy_cost, guess, callback=callback, jac=scipy_gadient)

    return _minimize_density_functional_density(res.x, particles, grid)
