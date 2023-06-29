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


def potential_to_density(
        v: Potential,
        n_elec: float,
        ke_functional: EnergyDensityFunctional = None) -> Tuple[Density, float]:
    # Returns the density and energy of a given number of electrons
    # in a given potential, optionally using a specific kinetic energy functional.

    if ke_functional is None:
        # Work out exact result, by diagonalizing
        spectrum = v.calculate_eigenstates()
        d_exact = spectrum.density(n_elec)
        e_exact = v.inner_product(d_exact) + spectrum.kinetic_energy(n_elec)
        return d_exact, e_exact

    return minimize_density_functional(n_elec, v.x, energy_functionals=[ExternalPotential(v), ke_functional])


def profile_pot_to_density():
    # Create an harmonic potential
    grid = Grid(-8, 8, 51)
    v = Potential(grid, 0.5 * grid.values ** 2)
    minimize_density_functional(4, grid, energy_functionals=[ExternalPotential(v), LadderKineticEnergyFunctional()])


def functional_density_plots(plot=True):
    from multiprocessing import Pool, cpu_count

    # Create an harmonic potential
    grid = Grid(-8, 8, 51)
    v = Potential(grid, 0.5 * grid.values ** 2)

    # Number of plots per side
    n_side = 3
    n_max = n_side ** 2

    # Ladder operator and ground state map to use
    ladder = PotentialDerivativeLadder(v)
    gs_map = NormalizeToGroundState()

    # Set of kinetic energy functionals to plot, and names of each
    ke_funcs = {
        None: "Exact",
        VonWeizakerKE(): "vW",
        LadderKineticEnergyFunctional(ladder=ladder, gs_map=gs_map): "Ladder"
    }

    # Arguments to parallelize over
    args = []
    for n in range(1, n_max + 1):
        for ke_func in ke_funcs:
            args.append([v, n, ke_func])

    # Get results in parallel
    with Pool(cpu_count()) as p:
        results: List[Tuple[Density, float]] = list(p.starmap(potential_to_density, args))

    if not plot:
        return

    plt.figure()
    spectrum = v.calculate_eigenstates()
    plt.subplot(121)
    plot_orbitals(plt, spectrum.orbitals[:n_max])
    plt.subplot(222)
    plt.plot(v.x.values, gs_map(spectrum.density(n_max)).values, label=f"g(rho_exact(N = {n_max}))")
    plt.plot(v.x.values, gs_map(spectrum.density(1)).values, label=f"g(rho_exact(N = {1}))")
    plt.plot(v.x.values, spectrum.orbitals[0].values, label="phi_0")
    plt.legend()

    plt.figure()
    for i, (v, n, ke_func) in enumerate(args):
        plt.subplot(n_side, n_side, n)
        d, e = results[i]
        plt.plot(d.x.values, d.values, label=f"{ke_funcs[ke_func]} density (E = {e:.5f})")
        plt.legend()
        plt.annotate(f"N = {n}", (0, 0))

    plt.show()


def main(plot=True):
    functional_density_plots(plot=plot)


if __name__ == "__main__":
    main()
