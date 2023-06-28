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

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)


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


def minimize_density_functional(
        particles: float,
        grid: Grid,
        energy_functionals: Iterable[EnergyDensityFunctional] = None,
        potential_functionals: Iterable[PotentialDensityFunctional] = None,
        plot=False) -> Density:
    from scipy.optimize import minimize

    energy_functionals = energy_functionals or []
    potential_functionals = potential_functionals or []

    def energy(d: Density):
        return \
                sum(f(d) for f in energy_functionals) + \
                sum(ExternalPotential(p(d))(d) for p in potential_functionals)

    def x_to_denisty(x: np.ndarray):
        d = Density(grid, x ** 2)
        d.particles = particles
        return d

    def scipy_cost(x: np.ndarray):
        e = energy(x_to_denisty(x))
        return e

    history = []

    def callback(x):
        d = x_to_denisty(x)

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

    res = minimize(scipy_cost, np.ones(len(grid.values)), callback=callback)
    return x_to_denisty(res.x)
