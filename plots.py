import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from typing import Dict
from qm1d.functionals import *


def plot_harmonic_oscillator_exact_density_t():
    v = Potential(Grid(-8, 8, 1001))
    v.values = 0.5 * v.x.values ** 2

    N = 5
    s = v.diagonalize_hamiltonian()
    d = s.density(N)
    k = s.kinetic_energy_density(N)

    plt.plot(d.x.values, d.values, label=fr"$\rho$ ($N={N}$)", color="red")
    plt.plot(k.x.values, k.values, label=r"$t$ ($N={N}$)", color="blue")
    plt.plot(v.x.values, v.values, label=r"$v$", color="black")
    plt.xlim([-4, 4])
    plt.ylim([-2, 8])
    plt.xlabel("$x$")
    plt.legend()
    plt.show()


def plot_harmonic_oscillator_exact_t_lda():
    v = Potential(Grid(-8, 8, 1001))
    v.values = 0.5 * v.x.values ** 2

    N = 5
    s = v.diagonalize_hamiltonian()
    d = s.density(N)
    k = s.kinetic_energy_density(N)

    i_keep = d.values > 1e-3

    kelda = KELDA(v, N)

    for inset in [False, True]:

        if inset:
            ax = plt.gcf().add_axes([0.45, 0.2, 0.4, 0.4])
        else:
            ax = plt.subplots()[-1]

        ax.plot(d.values[i_keep], k.values[i_keep], color="blue", label="exact")
        ax.plot(d.values[i_keep], kelda.t_lda(d.values[i_keep]), color="red", linestyle=":",
                label="single-valued approximation")

        if inset:
            plt.xlim([0.75, 1.05])
            plt.ylim([0.9, 1.8])
        else:
            plt.xlabel(r"$\rho(x)$")
            plt.ylabel(r"$t(x)$")
            plt.legend()
    plt.show()


def plot_densities(v: Potential,
                   ke_functionals: Dict[str, Callable[[Dict], DensityFunctional]],
                   use_multiprocessing: bool = True,
                   show_plot: bool = True):
    s = v.diagonalize_hamiltonian()
    n_side = 3
    n_values = range(1, n_side * n_side + 1)

    # Get densities using KELDA
    results = {}
    for name in ke_functionals:

        def energy_functional(n):
            return CombinedDensityFunctional([
                ExternalPotential(v),
                ke_functionals[name]({"N": n})
            ])

        args = [[n, v.x, energy_functional(n)] for n in n_values]

        if use_multiprocessing:
            with Pool(cpu_count()) as p:
                results[name] = p.starmap(minimize_density_functional_timed, args)
        else:
            results[name] = [minimize_density_functional_timed(*a) for a in args]

    # Get exact results
    exact_results = [s.density(n) for n in n_values]
    exact_results = [[0, d, v.inner_product(d) + s.kinetic_energy(n)] for n, d in zip(n_values, exact_results)]
    results["Exact"] = exact_results

    for n in n_values:
        plt.subplot(n_side, n_side, n)

        for name in results:
            # Plot density
            t, d, e = results[name][n - 1]
            plt.plot(d.x.values, d.values, label=f"{name} (e = {e:.5f})")

        plt.annotate(rf"$N = {n}$", (min(d.x.values), max(d.values) / 10.0))
        plt.xlabel(r"$x$")
        plt.ylabel(r"$\rho(x)$")
        plt.legend(loc="upper right")

    plt.figure()
    for name in results:
        plt.plot(n_values, [r[2] for r in results[name]], label=name)
    plt.xlabel(r"$N$")
    plt.ylabel(r"$E$")
    plt.legend()

    plt.figure()
    for name in results:
        plt.plot(n_values, [r[0] for r in results[name]], label=name)
    plt.xlabel(r"$N$")
    plt.ylabel(r"$Time$")
    plt.legend()

    if show_plot:
        plt.show()


def plot_harmonic_oscillator_densities_kelda(profile=False):
    v = Potential(Grid(-8, 8, 101))
    v.values = 0.5 * v.x.values ** 2
    plot_densities(v, {
        "KELDA": lambda info: KELDA(v, info["N"]),
        "KELDA N = 10": lambda info: KELDA(v, 10, allow_n_mismatch=True),
        "vW": lambda info: VonWeizakerKE()
    }, show_plot=not profile, use_multiprocessing=not profile)


def plot_harmonic_oscillator_densities_ladder(profile=False):
    v = Potential(Grid(-8, 8, 31))
    v.values = 0.5 * v.x.values ** 2
    plot_densities(v, {
        "TL1 (p=1)": lambda info: TL1(power=1),
        # "Ladder": lambda info: LadderKineticFunctional(),
        # "vW": lambda info: VonWeizakerKE()
    }, show_plot=not profile, use_multiprocessing=not profile)


def plot_triple_coulomb_well_densities_kelda():
    grid = Grid(-10, 10, 151)

    v_triple = Potential(grid)
    v_triple.values = -10 / (abs(v_triple.x.values - 3) + 1) - 10 / (abs(v_triple.x.values + 3) + 1) - 10 / (
            abs(v_triple.x.values) + 1)

    v_double = Potential(grid)
    v_double.values = -10 / (abs(v_triple.x.values - 2) + 1) - 10 / (abs(v_triple.x.values + 2) + 1)

    plot_densities(v_triple, {
        "KELDA N = 10 v = double well": lambda info: KELDA(v_double, 10, allow_n_mismatch=True),
        "vW": lambda info: VonWeizakerKE()
    })


def plot_kelda_interpolations():
    grid = Grid(-10, 10, 101)

    potentials = {
        "Single well": Potential(
            grid, -10 / (abs(grid.values) + 1)),
        "Double well": Potential(
            grid, -10 / (abs(grid.values - 2) + 1) - 10 / (abs(grid.values + 2) + 1)),
        "Triple well": Potential(
            grid, -10 / (abs(grid.values - 3) + 1) - 10 / (abs(grid.values + 3) + 1) - 10 / (abs(grid.values) + 1))
    }

    n_values = range(1, 9 + 1)

    for i, n in enumerate(n_values):
        for j, v_name in enumerate(potentials):
            plt.subplot(len(n_values) + 2, len(potentials), 1 + j + i * len(potentials))

            f = KELDA(potentials[v_name], n)
            interp_densities = Tensor.linspace(0, max(f.reference_densities) * 1.1, 1000)
            plt.plot(f.reference_densities, f.reference_kinetic_energy_densities, color="blue")
            plt.plot(interp_densities, f.t_lda(interp_densities), linestyle="dashed", color="red")

            plt.ylim([-5, 5])
            plt.xticks([])
            plt.yticks([])

            if i == len(n_values) - 1:
                plt.xlabel(fr"$\rho$")
                plt.xticks([0, 0.5, 1, 1.5])

            if j == 0:
                plt.ylabel(fr"$t(\rho)$" + f"\nN = {n}")
                plt.yticks([-2.0, 2.0])

    for j, v_name in enumerate(potentials):

        plt.subplot(len(n_values) + 2, len(potentials), 1 + j + (len(n_values) + 1) * len(potentials))
        plt.plot(grid.values, potentials[v_name].values, color="black")
        plt.xlabel(r"$x$" + f"\n{v_name}")

        plt.ylim([-20, 0])
        plt.xticks([])
        plt.yticks([])

        if j == 0:
            plt.ylabel(fr"$v(x)$")

    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()


def plot_ladder_operators(v: Potential, ladder: LadderOperator):
    spectrum = v.diagonalize_hamiltonian()

    # Evaluate exact ladder operator from sum of projection operators
    ladder_op_from_proj = sum(
        spectrum.orbitals[i].outer_product(spectrum.orbitals[i - 1])
        for i in range(1, len(spectrum.orbitals))
    )

    max_orb = 5
    n_cols = 8

    for i in range(1, max_orb + 1):
        # Plot orbital before
        plt.subplot(max_orb, n_cols, (i - 1) * n_cols + 1)
        plt.plot(v.x.values, spectrum.orbitals[i - 1].values)
        plt.xlabel("Exact orbitals")

        # Plot exact ladder operator
        plt.subplot(max_orb, n_cols, (i - 1) * n_cols + 3)
        plt.imshow(ladder_op_from_proj)
        plt.xlabel("Exact ladder operator")

        # Plot orbitals after application of exact ladder operator
        plt.subplot(max_orb, n_cols, (i - 1) * n_cols + 4)
        plt.plot(v.x.values, ladder_op_from_proj @ spectrum.orbitals[i - 1].values)

        # Plot approximate ladder operator
        plt.subplot(max_orb, n_cols, (i - 1) * n_cols + 5)
        plt.imshow(ladder.matrix_elements(v.x, spectrum.density(i)))
        plt.xlabel("Approximate ladder operator")

        # Plot orbitals after approximate ladder operator
        plt.subplot(max_orb, n_cols, (i - 1) * n_cols + 6)
        plt.plot(v.x.values, ladder.apply(spectrum.orbitals[i - 1], None).values)

        # Plot projection operator
        proj = spectrum.orbitals[i].outer_product(spectrum.orbitals[i - 1])
        plt.subplot(max_orb, n_cols, (i - 1) * n_cols + 7)
        plt.imshow(proj)
        plt.xlabel(f"|i+1><i|")

        # Plot orbitals after application of projection operator
        plt.subplot(max_orb, n_cols, (i - 1) * n_cols + 8)
        plt.plot(v.x.values, proj @ spectrum.orbitals[i - 1].values)

    plt.subplots_adjust(hspace=0.0)
    plt.show()


def plot_harmonic_ladder_operators():
    v = Potential(Grid(-8, 8, 51))
    v.values = 0.5 * v.x.values ** 2
    plot_ladder_operators(v, DerivativeLadderOperator())


if __name__ == "__main__":
    plot_harmonic_oscillator_densities_ladder()
