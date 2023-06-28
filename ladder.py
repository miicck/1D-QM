from hilbert import *
from functions import *


class LadderOperator(ABC):

    @abstractmethod
    def apply(self, orbital: Orbital, density: Density = None) -> Orbital:
        raise NotImplementedError()


class DensityToGroundStateMap(ABC):

    @abstractmethod
    def apply(self, denisty: Density) -> Orbital:
        raise NotImplementedError()


class DerivativeLadderOperator(LadderOperator):

    def apply(self, orbital: Orbital, density: Density = None) -> Orbital:
        result = Orbital(orbital.x, orbital.derivative.values)
        result.normalize()
        return result


class NormalizeToGroundState(DensityToGroundStateMap):

    def apply(self, denisty: Density) -> Orbital:
        result = Orbital(denisty.x, denisty.values)
        result.normalize()
        return result


def plot_ladder_result(desnity: Density,
                       ladder: LadderOperator = None,
                       ground_state_map: DensityToGroundStateMap = None):
    import matplotlib.pyplot as plt

    # Default values
    ground_state_map = ground_state_map or NormalizeToGroundState()
    ladder = ladder or DerivativeLadderOperator()

    pot = desnity.calculate_potential()
    orbs = pot.calculate_eigenstates()

    plt.subplot(221)
    plt.plot(desnity.x.values, desnity.values, label="Input density")
    plt.plot(desnity.x.values, orbs.density(desnity.particles), linestyle=":", label="Density from orbitals")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\rho$")
    plt.legend()

    def plot_orbitals(orbs: Iterable[Orbital]):
        kes = [o.kinetic_energy for o in orbs]
        for i, o in enumerate(orbs):
            plt.plot(o.x.values, o.values + i)
            plt.annotate(f"$T_{i} = {kes[i]:<12.6f}$", (min(o.x.values), i + 0.1))
            plt.axhline(i, color="black", alpha=0.2)
        plt.annotate(f"T = {sum(kes):<12.6f}", (min(o.x.values), i + 1 + 0.1))
        plt.ylim(-1, i + 2)

    plt.subplot(222)
    plot_orbitals(orbs.orbitals[:round(desnity.particles)])
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\phi_i$")

    plt.subplot(223)
    plt.plot(desnity.x.values, pot.values)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$v$")

    # Generate orbitals from ladder operator
    generated = [ground_state_map.apply(desnity)]
    while len(generated) < round(desnity.particles):
        generated.append(ladder.apply(generated[-1]))

    plt.subplot(224)
    plot_orbitals(generated)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$g_i$")

    plt.tight_layout()
    plt.show()
