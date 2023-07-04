import numpy as np
from hilbert import *
from functions import *


class LadderOperator(ABC):

    @abstractmethod
    def apply(self, orbital: Orbital, density: Density) -> Orbital:
        raise NotImplementedError()

    def matrix_elements(self, grid: Grid, density: Density) -> np.ndarray:

        matrix = np.zeros((grid.points, grid.points))
        for i in range(grid.points):
            xi = np.zeros(grid.points)
            xi[i] = 1.0
            xi = Orbital(grid, xi)
            xi.normalize()
            for j in range(grid.points):
                xj = np.zeros(grid.points)
                xj[j] = 1.0
                xj = Orbital(grid, xj)
                xj.normalize()
                matrix[i, j] = xi.inner_product(self.apply(xj, density=density))
        return matrix

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)


class DensityToGroundStateMap(ABC):

    @abstractmethod
    def apply(self, denisty: Density) -> Orbital:
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)


class DerivativeLadderOperator(LadderOperator):

    def apply(self, orbital: Orbital, density: Density) -> Orbital:
        result = Orbital(orbital.x, orbital.derivative.values)
        result.normalize()
        return result


class HarmonicLadderOperator(LadderOperator):

    def apply(self, orbital: Orbital, density: Density) -> Orbital:
        result = Orbital(orbital.x, orbital.derivative.values - orbital.x.values * orbital.values)
        result.normalize()
        return result


class PotentialDerivativeLadder(LadderOperator):

    def __init__(self, v: Potential):
        self._v = v

    def apply(self, orbital: Orbital, density: Density) -> Orbital:
        result = Orbital(orbital.x, orbital.derivative.values - 0.5 * self._v.derivative.values * orbital.values)
        result.normalize()
        return result


class DensityDerivativeLadder(LadderOperator):

    def apply(self, orbital: Orbital, density: Density) -> Orbital:
        density_part = np.zeros(orbital.x.values.shape)
        if density is not None:
            density_part = 0.5 * density.normalized.derivative.values * orbital.values

        result = Orbital(orbital.x, orbital.derivative.values + density_part)
        result.normalize()
        return result


class NormalizeToGroundState(DensityToGroundStateMap):

    def __init__(self, power: float = 1.0):
        self._power = power

    def apply(self, denisty: Density) -> Orbital:
        result = Orbital(denisty.x, denisty.values ** self._power)
        result.normalize()
        return result


class PotentialBiasedGroundState(DensityToGroundStateMap):

    def __init__(self, v: Potential, v_power=1.0, d_power=1.0, v_mix=0.5):
        self._v = v
        self._v_power = v_power
        self._d_power = d_power
        self._v_mix = v_mix
        self._v_orbital =v.diagonalize_hamiltonian().orbitals[0]

    def apply(self, denisty: Density) -> Orbital:
        d_orbital = Orbital(denisty.x, denisty.values ** self._d_power).normalized
        result = Orbital(denisty.x, d_orbital.values * (1 - self._v_mix) + self._v_orbital.values * self._v_mix)
        result.normalize()
        return result


class FixedGroundState(DensityToGroundStateMap):

    def __init__(self, ground_state: Orbital):
        self._ground_state = ground_state

    def apply(self, denisty: Density) -> Orbital:
        return self._ground_state


def plot_orbitals(plt, orbs: Iterable[Orbital]):
    kes = [o.kinetic_energy for o in orbs]
    for i, o in enumerate(orbs):
        plt.plot(o.x.values, o.values + i + 1)
        plt.annotate(f"$T_{i} = {kes[i]:<12.6f}$", (min(o.x.values), i + 1.1))
        plt.axhline(i + 1, color="black", alpha=0.2)
    plt.annotate(f"T = {sum(kes):<12.6f}", (min(o.x.values), i + 2 + 0.1))
    plt.ylim(0, i + 3)


def plot_ladder_result(density: Density,
                       ladder: LadderOperator = None,
                       ground_state_map: DensityToGroundStateMap = None):
    import matplotlib.pyplot as plt

    # Default values
    ground_state_map = ground_state_map or NormalizeToGroundState()
    ladder = ladder or DerivativeLadderOperator()

    pot = density.calculate_potential()
    orbs = pot.diagonalize_hamiltonian()

    plt.subplot(221)
    plt.plot(density.x.values, density.values, label="Input density")
    plt.plot(density.x.values, orbs.density(density.particles).values, linestyle=":", label="Density from orbitals")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\rho$")
    plt.legend()

    plt.subplot(222)
    plot_orbitals(plt, orbs.orbitals[:round(density.particles)])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\phi_i$")

    plt.subplot(223)
    plt.plot(density.x.values, pot.values)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$v$")

    # Generate orbitals from ladder operator
    generated = [ground_state_map.apply(density)]
    while len(generated) < round(density.particles):
        generated.append(ladder.apply(generated[-1]))

    plt.subplot(224)
    plot_orbitals(plt, generated)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$g_i$")

    plt.tight_layout()
    plt.show()
