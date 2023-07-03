import time
import numpy as np
import matplotlib.pyplot as plt
from hilbert import *
from functions import *
from ladder import *
from functionals import *


def candidate_potential() -> Potential:
    grid = Grid(-8, 8, 51)
    return Potential(grid, 0.5 * grid.values ** 2)  # Harmonic oscilaltor
    return Potential(grid, 0.5 * grid.values ** 4 - 4 * grid.values ** 2)  # Anharmonic oscillator
    return Potential(grid, -4 / (abs(grid.values - 2) * 8 + 1) - 4 / (abs(grid.values + 2) * 8 + 1))  # Double well
    return Potential(grid, -1 / (abs(grid.values) + 1))  # Single well


def candidate_gs_map_ladder_combo(v: Potential) -> Tuple[DensityToGroundStateMap, LadderOperator]:
    # External potential-blind operators
    # Motivation:
    #     Strictly speaking, the KE functional should not depend
    #     on the external potential. These operators depend strictly
    #     on the density (either explicitly, or through ladder-operator
    #     generated orbitals).
    return NormalizeToGroundState(), DerivativeLadderOperator()

    # Use density for ground state, potential for ladder
    return NormalizeToGroundState(), PotentialDerivativeLadder(v)

    # Use potential for ground state, density for ladder
    return PotentialBiasedGroundState(v, v_mix=1.0), DensityDerivativeLadder()

    # Ground state generated exactly
    # Motivation:
    #     Given an external potential, the ground state for a 1-electron system
    #     can be found exactly, and looks a lot like the lowest orbital, so use that.
    return PotentialBiasedGroundState(v, v_mix=1.0), PotentialDerivativeLadder(v)


def candidate_gs_map(v: Potential) -> DensityToGroundStateMap:
    return candidate_gs_map_ladder_combo(v)[0]


def candidate_ladder_operator(v: Potential) -> LadderOperator:
    return candidate_gs_map_ladder_combo(v)[1]


def candidate_functionals(v: Potential):
    return {
        None: "Exact",
        VonWeizakerKE(): "vW",
        LadderKineticEnergyFunctional(
            ladder=candidate_ladder_operator(v),
            gs_map=candidate_gs_map(v)): "Ladder",
        KELDA(v, force_interp=True): "KELDA"
    }


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


def plot_exact_ladder_operators():
    v = candidate_potential()
    spectrum = v.calculate_eigenstates()
    ladder = candidate_ladder_operator(v)

    ladder_op_from_proj = sum(
        spectrum.orbitals[i].outer_product(spectrum.orbitals[i - 1])
        for i in range(1, len(spectrum.orbitals))
    )

    i_max = 5
    i_cols = 8

    for i in range(1, i_max + 1):
        # Plot orbital before
        plt.subplot(i_max, i_cols, (i - 1) * i_cols + 1)
        plt.plot(v.x.values, spectrum.orbitals[i - 1].values)

        # Plot exact ladder operator
        plt.subplot(i_max, i_cols, (i - 1) * i_cols + 3)
        plt.imshow(ladder_op_from_proj)
        plt.xlabel("Exact ladder operator")

        # Plot orbitals after application of exact ladder operator
        plt.subplot(i_max, i_cols, (i - 1) * i_cols + 4)
        plt.plot(v.x.values, ladder_op_from_proj @ spectrum.orbitals[i - 1].values)

        # Plot approximate ladder operator
        plt.subplot(i_max, i_cols, (i - 1) * i_cols + 5)
        plt.imshow(ladder.matrix_elements(v.x, spectrum.density(i)))
        plt.xlabel("Approximate ladder operator")

        # Plot orbitals after approximate ladder operator
        plt.subplot(i_max, i_cols, (i - 1) * i_cols + 6)
        plt.plot(v.x.values, ladder.apply(spectrum.orbitals[i - 1], None).values)

        # Plot projection operator
        proj = spectrum.orbitals[i].outer_product(spectrum.orbitals[i - 1])
        plt.subplot(i_max, i_cols, (i - 1) * i_cols + 7)
        plt.imshow(proj)
        plt.xlabel(f"|{i + 1}><{i}|")

        # Plot orbitals after application of projection operator
        plt.subplot(i_max, i_cols, (i - 1) * i_cols + 8)
        plt.plot(v.x.values, proj @ spectrum.orbitals[i - 1].values)

    plt.subplots_adjust(hspace=0.0)

    # Plot exact ladder operator from sum of
    # projection operators, and the one we're using
    plt.figure()
    plt.subplot(221)
    plt.imshow(ladder_op_from_proj)
    plt.subplot(222)
    plt.imshow(ladder.matrix_elements(v.x, spectrum.density(1)))
    plt.subplot(223)
    plt.plot(v.x.values, v.values)

    plt.show()


def density_plots(plot=True):
    from multiprocessing import Pool, cpu_count

    # Get potential
    v = candidate_potential()

    # Number of plots per side
    n_side = 3
    n_max = n_side ** 2

    # Set of kinetic energy functionals to plot, and names of each
    ke_funcs = candidate_functionals(v)

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

    # Plot some information about the exact orbitals, and how well the ground state map works
    plt.figure()
    gs_map = candidate_gs_map(v)
    spectrum = v.calculate_eigenstates()
    plt.subplot(121)
    plot_orbitals(plt, spectrum.orbitals[:n_max])
    plt.subplot(222)
    plt.plot(v.x.values, gs_map(spectrum.density(n_max)).values, label=f"g(rho_exact(N = {n_max}))")
    plt.plot(v.x.values, gs_map(spectrum.density(1)).values, label=f"g(rho_exact(N = {1}))")
    plt.plot(v.x.values, spectrum.orbitals[0].values, label="phi_0")
    plt.legend()

    plt.subplot(224)
    plt.plot(v.x.values, v.values)
    plt.ylabel("v")

    # Plot the densities obtained by various functionals
    plt.figure()
    plt.suptitle("GS densities")
    for i, (v, n, ke_func) in enumerate(args):
        plt.subplot(n_side, n_side, n)
        d, e = results[i]
        plt.plot(d.x.values, d.values, label=f"{ke_funcs[ke_func]} density (E = {e:.5f})")
        plt.legend()
        plt.annotate(f"N = {n}", (0, 0))

    # Plot the orbitals obtained from each density
    plt.figure()
    plt.suptitle("Generated orbitals for ladder GS density")
    for i, (v, n, ke_func) in enumerate(args):
        if not isinstance(ke_func, LadderKineticEnergyFunctional):
            continue
        plt.subplot(n_side, n_side, n)
        d, e = results[i]
        orbs = [ke_func.gs_map(d)]
        while len(orbs) < n:
            orbs.append(ke_func.ladder(orbs[-1], d))
        plot_orbitals(plt, orbs)

    plt.show()


def integer_discontinuity_plots(plot=True):
    from multiprocessing import Pool, cpu_count

    def integer_interp(i, energies):

        # i is at an integer value already
        if abs(ns[i] - round(ns[i])) < 1e-5:
            return energies[i]

        # Find integer value before i
        for j in range(i, -1, -1):
            if abs(ns[j] - round(ns[j])) < 1e-5:
                break

        # Find integer value after i
        for k in range(i, len(ns)):
            if abs(ns[k] - round(ns[k])) < 1e-5:
                break

        # Interpolate between them
        frac = (i - j) / (k - j)
        return energies[j] * (1 - frac) + energies[k] * frac

    # Get potential
    v = candidate_potential()

    # Set of kinetic energy functionals to plot, and names of each
    ke_funcs = candidate_functionals(v)

    # Arguments to parallelize over - must contain integer values
    ns = np.linspace(0, 5, 51)
    ns = list(ns)
    ns.pop(0)  # Don't do 0 electrons
    ns = np.array(ns)

    # Plot results for each KE functional
    for ke_func in ke_funcs:
        # Get results in parallel
        with Pool(cpu_count()) as p:
            results = p.starmap(potential_to_density, [[v, n, ke_func] for n in ns])

        if not plot:
            continue

        es = [r[1] for r in results]

        plt.subplot(211)
        plt.plot(ns, es, marker="+", label=ke_funcs[ke_func])
        plt.legend()
        plt.xlabel("N")
        plt.ylabel("E(N)")

        plt.subplot(212)
        plt.plot(ns, [es[i] - integer_interp(i, es) for i in range(len(ns))], label=ke_funcs[ke_func])
        plt.legend()
        plt.xlabel("N")
        plt.ylabel("E(N) - E(piecewise linear)")

    if plot:
        plt.show()


def main(plot=True):
    density_plots(plot=plot)
    integer_discontinuity_plots(plot=plot)
    plot_exact_ladder_operators()


if __name__ == "__main__":
    main()
