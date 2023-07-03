import math
import numpy as np
from functions import *
from typing import Iterable
from ladder import *


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


class LadderKineticEnergyFunctional(EnergyDensityFunctional):

    def __init__(self, ladder: LadderOperator = None, gs_map: DensityToGroundStateMap = None):
        self._ladder = ladder or DerivativeLadderOperator()
        self._gs_map = gs_map or NormalizeToGroundState()

    @property
    def ladder(self) -> LadderOperator:
        return self._ladder

    @property
    def gs_map(self) -> DensityToGroundStateMap:
        return self._gs_map

    def apply(self, density: Density) -> float:
        generated = [self._gs_map.apply(density)]
        while len(generated) < density.particles:
            generated.append(self._ladder.apply(generated[-1], density=density))

        auf = OrbitalSpectrum.aufbau_weights(density.particles)
        return sum(a * generated[i].kinetic_energy for i, a in enumerate(auf))


class LocalKEDensityEnergyFunctional(EnergyDensityFunctional):

    def __init__(self, potential: Potential):
        self._v = potential
        self._spectrum = potential.calculate_eigenstates()
        self._ke_density = None
        self._ke_density_particles = None

    def apply(self, density: Density) -> float:

        if self._ke_density is None:
            self._ke_density = self._spectrum.kinetic_energy_density(density.particles)
            self._ke_density_particles = density.particles

        if abs(self._ke_density_particles - density.particles) > 1e-6:
            self._ke_density = None
            self._ke_density_particles = None
            return self.apply(density)

        return self._ke_density.inner_product(density)


class KELDA(EnergyDensityFunctional):

    def __init__(self, potential: Potential, force_interp=False):
        self._v = potential
        self._spectrum = potential.calculate_eigenstates()
        self._last_particles = None
        self._interp = None
        self._force_interp = force_interp

    def lda(self, density: float):
        return self._interp(density)

    def apply(self, density: Density, plot=False) -> float:

        # N used to evaluate KE-LDA
        effective_n = density.particles

        if self._last_particles is None or abs(effective_n - self._last_particles) > 1e-6:
            from scipy.interpolate import interp1d
            from scipy.optimize import curve_fit

            self._last_particles = effective_n
            ref_ke = self._spectrum.kinetic_energy_density(self._last_particles).values
            ref_density = self._spectrum.density(self._last_particles).values

            # Throw away low-density data
            i_keep = ref_density > 1e-5
            ref_density = ref_density[i_keep]
            ref_ke = ref_ke[i_keep]

            def to_fit(d, c0, c1, p1, c2, p2):
                return c0 + c1 * (d ** p1) + c2 * (d ** p2)

            try:
                # Try fitting to curve
                if self._force_interp:
                    raise RuntimeError
                par, cov = curve_fit(to_fit, ref_density, ref_ke)
                self._interp = lambda d: to_fit(d, *par)

            except RuntimeError as e:
                # Fallback to interpolation

                # Get single-valued version of data
                sv = []
                for i in range(len(ref_density)):
                    if len(sv) == 0 or ref_density[i] > sv[-1][0]:
                        sv.append([ref_density[i], ref_ke[i]])
                sv = np.array(sv).T

                # Build interpolation
                scipy_interp = interp1d(sv[0], sv[1])

                d_min, d_max = min(sv[0]), max(sv[0])

                eps = 1e-4
                f_dmax = scipy_interp(d_max)
                f_dmin = scipy_interp(d_min)
                dfdx_dmax = (scipy_interp(d_max) - scipy_interp(d_max - eps)) / eps
                dfdx_dmin = (scipy_interp(d_min + eps) - scipy_interp(d_min)) / eps

                def interp_single(d):
                    try:
                        return scipy_interp(d)
                    except ValueError:

                        if d > d_max - eps:
                            # Linear extrapolation
                            return (d - d_max) * dfdx_dmax + f_dmax

                        if d < d_min + eps:
                            # Linear extrapolation
                            return (d - d_min) * dfdx_dmin + f_dmin

                        return 0.0

                def interp(d):
                    return np.array([interp_single(x) for x in d])

                self._interp = interp
                if not self._force_interp:
                    print(f"KELDA fit failed for N = {density.particles} (eff = {effective_n})")

            if plot:
                import matplotlib.pyplot as plt
                x = np.linspace(0, 2 * max(ref_density), 1000)
                plt.plot(ref_density, ref_ke, label="Reference data")
                plt.plot(x, self._interp(x), label="KELDA fit")
                plt.ylim(min(ref_ke) - 2, max(ref_ke) + 2)
                plt.legend()
                plt.show()

        return density.inner_product(Function(density.x, self._interp(density.values)))


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
        plot=False,
        use_multiprocessing=False) -> Density:
    from scipy.optimize import minimize

    energy_functionals = energy_functionals or []
    potential_functionals = potential_functionals or []

    def scipy_cost(x: np.ndarray):
        return _minimize_density_functional_cost(x, particles, grid, energy_functionals, potential_functionals)

    def scipy_gadient(x: np.ndarray) -> np.ndarray:

        dn = np.identity(len(x)) * 1e-6
        f0 = _minimize_density_functional_cost(x, particles, grid, energy_functionals, potential_functionals)

        if use_multiprocessing:
            from multiprocessing import Pool, cpu_count
            with Pool(cpu_count()) as p:
                fnew = p.starmap(_minimize_density_functional_cost,
                                 [[x + dn[i], particles, grid, energy_functionals, potential_functionals] for i in
                                  range(len(x))])
        else:
            fnew = [_minimize_density_functional_cost(
                x + dn[i], particles, grid, energy_functionals, potential_functionals) for i in range(len(x))]

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

    return _minimize_density_functional_density(res.x, particles, grid), res.fun
