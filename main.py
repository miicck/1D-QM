import time
import numpy as np
import matplotlib.pyplot as plt
from hilbert import *
from functions import *
from ladder import *
from functionals import *


def ladder_plot():
    # Create a two-well density
    rho = Density(Grid(-5, 5, 101))
    x = rho.x.values
    rho.values = np.exp(-(x - 1) ** 2) ** 2 + 0.5 * np.exp(-(x + 1) ** 2) ** 2
    rho.values = np.exp(-x ** 2 / 2)
    rho.particles = 4

    # Plot comparison with ladder operator
    plot_ladder_result(rho)


def vw_vs_exact_vs_ladder_density_plot(exact_ke_func=False, n_elec=2, show_plot=True):
    # Create an harmonic potential
    grid = Grid(-10, 10, 51)
    v = Potential(grid, grid.values ** 2 / 10)

    t_start = time.time()

    if exact_ke_func:
        dex = minimize_density_functional(
            n_elec, grid, [ExternalPotential(v), ExactKineticEnergyFunctional(max_iter=100)])
    else:
        eigs = v.calculate_eigenstates()
        d = eigs.density(n_elec)
        dex = [d, ExternalPotential(v)(d) + eigs.kinetic_energy(n_elec)]
    dvw = minimize_density_functional(n_elec, grid, [ExternalPotential(v), VonWeizakerKE()])
    dbl = minimize_density_functional(n_elec, grid, [ExternalPotential(v), LadderKineticEnergyFunctional(
        ladder=PotentialDerivativeLadder(v),
        gs_map=NormalizeToGroundState(1)
    )])

    t_elapsed = time.time() - t_start

    plt.plot(v.x.values, dvw[0].values, label=f"vW density, E = {dvw[1]:<12.6f}")
    plt.plot(v.x.values, dex[0].values, label=f"exact density, E = {dex[1]:<12.6f}")
    plt.plot(v.x.values, dbl[0].values, label=f"ladder density, E = {dbl[1]:<12.6f}")
    plt.annotate(f"N = {n_elec} (t = {t_elapsed:.3f}s)", (0, 0))
    plt.legend()

    if show_plot:
        plt.show()


if __name__ == "__main__":

    import sys

    plot = "no_plot" not in sys.argv

    n_max = 4
    for n_elec in range(1, n_max + 1):
        plt.subplot(n_max, 1, n_elec)
        vw_vs_exact_vs_ladder_density_plot(show_plot=False, n_elec=n_elec)

    if plot:
        plt.show()
