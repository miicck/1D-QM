import math
import time
from qm1d.tensor import Tensor
from functions import *
from typing import Iterable
from ladder import *


class DensityFunctional(ABC):

    @abstractmethod
    def apply(self, density: Density) -> float:
        raise NotImplementedError()

    def functional_derivative(self, density: Density) -> Function:
        return self.functional_derivative_finite_difference(density)

    def functional_derivative_finite_difference(self, density: Density, eps=1e-6) -> Function:
        # Will contain the gradient
        result = Tensor.zeros(density.x.values.shape)

        # Get center value
        f0 = self.apply(density)

        for i in range(len(result)):
            # Perturb the density
            x_before = float(density.values[i])
            density.values[i] += eps
            x_after = density.values[i]

            # Calculate change in functional
            result[i] = (self.apply(density) - f0) / (x_after - x_before)

            # Reset density
            density.values[i] = x_before

        return Function(density.x, result / density.x.spacing)

    def __call__(self, density: Density) -> float:
        return self.apply(density)


class VonWeizakerKE(DensityFunctional):

    def apply(self, density: Density) -> float:
        sqrt = Density(density.x, density.values ** 0.5)
        return -0.5 * sqrt.inner_product(sqrt.laplacian)

    def functional_derivative(self, density: Density) -> Function:
        p = Density(density.x, density.values ** 0.5)
        p1 = p.derivative
        p2 = p1.derivative
        return Function(density.x, -0.5 * p2.values / p.values)


class TL1(DensityFunctional):

    def __init__(self, power: float = 1.0, precompute_derivatives=False):
        self._power = power
        self._precompute_derivatives = precompute_derivatives
        self._derivatives = []

    def apply(self, density: Density) -> float:

        if self._precompute_derivatives:
            return self.apply_precomputed(density)

        result = 0.0

        states = [Orbital(density.x, density.values ** self._power)]
        states[0].normalize()

        auf = OrbitalSpectrum.aufbau_weights(density.particles)
        while len(states) < len(auf):
            states.append(states[-1].derivative)
            states[-1].normalize()

        return -0.5 * sum(a * states[i].inner_product(states[i].laplacian) for i, a in enumerate(auf))

    def apply_precomputed(self, density: Density) -> float:

        auf = OrbitalSpectrum.aufbau_weights(density.particles)

        if len(self._derivatives) == 0:
            self._derivatives = [Tensor.identity(len(density.x.values))]

        while len(self._derivatives) < len(auf):
            self._derivatives.append(density.x.gradient @ self._derivatives[-1])

        total = 0.0
        for i, a in enumerate(auf):
            state = Orbital(density.x, self._derivatives[i] @ (density.values ** self._power))
            state.normalize()
            total += state.inner_product(state.laplacian)

        return -0.5 * total


class ExternalPotential(DensityFunctional):

    def __init__(self, v_ext: Potential):
        self._v_ext = v_ext

    @property
    def v_ext(self) -> Potential:
        return self._v_ext

    def apply(self, density: Density) -> float:
        return density.inner_product(self.v_ext)

    def functional_derivative(self, density: Density) -> Function:
        return self._v_ext


class ExactKineticFunctional(DensityFunctional):

    def __init__(self, n_elec_tol=0.1, max_iter=10):
        self._n_elec_tol = n_elec_tol
        self._max_iter = max_iter

    def apply(self, density: Density) -> float:
        v = density.calculate_potential(n_elec_tol=self._n_elec_tol,
                                        max_iter=self._max_iter)
        e = v.diagonalize_hamiltonian()
        return e.kinetic_energy(density.particles)


class LadderKineticFunctional(DensityFunctional):

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


class LocalKEDensityFunctional(DensityFunctional):

    def __init__(self, potential: Potential):
        self._v = potential
        self._spectrum = potential.diagonalize_hamiltonian()
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


class KELDA(DensityFunctional):

    def __init__(self, potential: Potential, particles: float, allow_n_mismatch: bool = False):
        self._potential = potential
        self._particles = particles
        self._t_lda = None
        self._ref_data = None
        self._allow_n_mismatch = allow_n_mismatch

    def derive_lda(self):
        from scipy.interpolate import interp1d
        from scipy.optimize import curve_fit

        spectrum = self._potential.diagonalize_hamiltonian()
        ref_ke = spectrum.kinetic_energy_density(self._particles).values
        ref_density = spectrum.density(self._particles).values

        # Throw away low-density data
        i_keep = ref_density > 1e-5
        ref_density = ref_density[i_keep]
        ref_ke = ref_ke[i_keep]

        # Get single-valued version of data
        sv = [(ref_density[0], ref_ke[0])]
        for i in range(len(ref_density)):
            if ref_density[i] > sv[-1][0]:
                sv.append((ref_density[i], ref_ke[i]))

        sv = Tensor.from_values(sv).T

        # Build interpolation
        t_lda_interp = interp1d(sv[0], sv[1])
        d_min, d_max = min(sv[0]), max(sv[0])

        # Evaluate quantities needed for linear extrapolation
        eps = 1e-4
        f_dmax = t_lda_interp(d_max)
        f_dmin = t_lda_interp(d_min)
        dfdx_dmax = (t_lda_interp(d_max) - t_lda_interp(d_max - eps)) / eps
        dfdx_dmin = (t_lda_interp(d_min + eps) - t_lda_interp(d_min)) / eps

        def t_lda(density: Tensor):

            result = Tensor.zeros(density.shape)

            # Generate in-range result using interpolator
            d_in_range = Tensor.logical_and(density < d_max, density > d_min)
            result[d_in_range] = Tensor.asarray(t_lda_interp(density[d_in_range]))

            # Generate out-of-range results using linear extrapolation
            d_less = density <= d_min
            result[d_less] = (density[d_less] - d_min) * dfdx_dmin + f_dmin
            d_more = density >= d_max
            result[d_more] = (density[d_more] - d_max) * dfdx_dmax + f_dmax

            return result

        self._t_lda = t_lda
        self._ref_data = [ref_density, ref_ke]

    @property
    def reference_densities(self) -> Tensor:
        if self._ref_data is None:
            self.derive_lda()
        return self._ref_data[0].copy()

    @property
    def reference_kinetic_energy_densities(self) -> Tensor:
        if self._ref_data is None:
            self.derive_lda()
        return self._ref_data[1].copy()

    def t_lda(self, density: Tensor) -> Tensor:
        if self._t_lda is None:
            self.derive_lda()
        return self._t_lda(density)

    def t_lda_derivative(self, density, eps=1e-6):
        return (self.t_lda(density + eps) - self.t_lda(density)) / eps

    def v_eff(self, density: Density) -> Potential:

        if not self._allow_n_mismatch:
            if abs(density.particles - self._particles) > 1e-5:
                raise Exception(f"Tried to use KELDA derived for N = {self._particles} "
                                f"on a density with N = {density.particles}")

        return Potential(density.x, self.t_lda(density.values))

    def apply(self, density: Density, plot=False) -> float:
        return density.inner_product(self.v_eff(density))

    def functional_derivative(self, density: Density) -> Function:
        return Function(density.x,
                        self.v_eff(density).values +
                        density.values * self.t_lda_derivative(density.values))


class CombinedDensityFunctional(DensityFunctional):

    def __init__(self, functionals: Iterable[DensityFunctional], weights: Iterable[float] = None):
        self._functionals = list(functionals)
        self._weights = Tensor.ones(len(self._functionals)) if weights is None else list(weights)

    def apply(self, density: Density) -> float:
        return sum(w * f(density)
                   for w, f in zip(self._weights, self._functionals))

    def functional_derivative(self, density: Density) -> Function:
        vals = sum(w * f.functional_derivative(density).values
                   for w, f in zip(self._weights, self._functionals))
        return Function(density.x, vals)


def minimize_density_functional_timed(*args, **kwargs):
    time_start = time.time()
    results = minimize_density_functional(*args, **kwargs)
    return time.time() - time_start, *results


def minimize_density_functional(
        particles: float,
        grid: Grid,
        functional: DensityFunctional) -> Density:
    from scipy.optimize import minimize

    identity = Tensor.identity(len(grid.values))

    def rho_of_x(x: Tensor) -> Density:
        x2 = x ** 2
        d = Density(grid, particles * x2 / (sum(x2) * grid.spacing))
        return d

    def d_rho_d_x(x: Tensor) -> Tensor:
        x2 = x ** 2
        sum_x2 = sum(x2)
        prefactor = 2 * particles / (grid.spacing * sum_x2)
        return prefactor * (Tensor.einsum("i,ik->ik", x, identity) - Tensor.einsum("k,i->ik", x, x2) / sum_x2)

    def scipy_cost(x):
        x = Tensor.asarray(x)
        return functional(rho_of_x(x))

    def scipy_gradient(x):
        x = Tensor.asarray(x)
        return d_rho_d_x(x).T @ functional.functional_derivative(rho_of_x(x)).values

    width = (max(grid.values) - min(grid.values)) / 4.0
    guess = Tensor.exp(-(grid.values / width) ** 2)
    res = minimize(scipy_cost, guess, jac=scipy_gradient)

    return rho_of_x(Tensor.asarray(res.x)), res.fun
