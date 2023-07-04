import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from typing import Dict
from functions import *
from functionals import *


def plot_harmonic_oscillator_exact_density_t():
    v = Potential(Grid(-8, 8, 1001))
    v.values = 0.5 * v.x.values ** 2

    N = 5
    s = v.calculate_eigenstates()
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
    s = v.calculate_eigenstates()
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


def plot_densities(v: Potential, ke_functionals: Dict[str, Callable[[Dict], EnergyDensityFunctional]]):
    s = v.calculate_eigenstates()
    n_side = 3
    n_values = range(1, n_side * n_side + 1)

    # Get densities using KELDA
    results = {}
    for name in ke_functionals:
        args = [[n, v.x, [ExternalPotential(v), ke_functionals[name]({"N": n})]] for n in n_values]
        with Pool(cpu_count()) as p:
            results[name] = p.starmap(minimize_density_functional, args)

    # Get exact results
    exact_results = [s.density(n) for n in n_values]
    exact_results = [[d, v.inner_product(d) + s.kinetic_energy(n)] for n, d in zip(n_values, exact_results)]
    results["Exact"] = exact_results

    for n in n_values:
        plt.subplot(n_side, n_side, n)

        for name in results:
            # Plot density
            d, e = results[name][n - 1]
            plt.plot(d.x.values, d.values, label=f"{name} (e = {e:.5f})")

        plt.annotate(rf"$N = {n}$", (min(d.x.values), max(d.values) / 10.0))
        plt.xlabel(r"$x$")
        plt.ylabel(r"$\rho(x)$")
        plt.legend()

    plt.figure()
    for name in results:
        plt.plot(n_values, [r[1] for r in results[name]], label=name)
    plt.xlabel(r"$N$")
    plt.ylabel(r"$E$")
    plt.legend()

    plt.show()


def plot_harmonic_oscillator_densities():
    v = Potential(Grid(-8, 8, 51))
    v.values = 0.5 * v.x.values ** 2
    plot_densities(v, {
        "KELDA": lambda info: KELDA(v, info["N"]),
        "vW": lambda info: VonWeizakerKE(),
        "Ladder": lambda info: LadderKineticEnergyFunctional()
    })


def plot_ladder_operators(v: Potential, ladder: LadderOperator):
    spectrum = v.calculate_eigenstates()

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
    plot_harmonic_oscillator_densities()
