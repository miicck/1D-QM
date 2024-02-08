import matplotlib.pyplot as plt
from multiprocessing import cpu_count, Pool
from typing import Dict
from qm1d.functionals import *
import numpy as np


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
                   show_plot: bool = True,
                   n_side: int = 3,
                   x_range: Tuple[float] = None,
                   rho_range: Tuple[float] = None):
    s = v.diagonalize_hamiltonian()
    n_values = range(1, n_side * n_side + 1)

    # Get densities using each KE functional
    results = {}
    for name in ke_functionals:

        def energy_functional(n):
            return CombinedDensityFunctional([
                ExternalPotential(v),
                ke_functionals[name]({"N": n})
            ])

        args = [[n, v.x, energy_functional(n), s.density(n)] for n in n_values]

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

        text = rf" $N = {n}$"
        for name in results:
            # Plot density
            t, d, e = results[name][n - 1]
            plt.plot(d.x.values, d.values, label=f"{name}")
            text += f"\n E({name}) = {e:.5f}"
            if x_range is not None:
                plt.xlim(x_range)
            if rho_range is not None:
                plt.ylim(rho_range)

        plt.annotate(text, (x_range[0], rho_range[1] - 0.1), verticalalignment="top")

        if rho_range is None:
            plt.ylabel(r"$\rho(x)$")
        else:
            if n % n_side == 1:
                plt.ylabel(r"$\rho(x)$")
            else:
                plt.yticks([])

        if x_range is None:
            plt.xlabel(r"$x$")
        else:
            if n >= n_side ** 2 - n_side:
                plt.xlabel("$x$")
                plt.xticks(range(round(x_range[0]) + 1, round(x_range[1]), 2))
            else:
                plt.xticks([])

        if n == n_side * n_side:
            plt.legend(loc="lower center")

    if rho_range is not None:
        plt.subplots_adjust(wspace=0)
    if x_range is not None:
        plt.subplots_adjust(hspace=0)

    plt.figure()
    for name in results:
        plt.plot(n_values, np.array([r[2] for r in results[name]]) - np.array([r[2] for r in results["Exact"]]),
                 label=name)
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
    grid = Grid(-10, 10, 301)

    v_triple = Potential(grid)
    v_triple.values = -10 / (abs(v_triple.x.values - 3) + 1) - 10 / (abs(v_triple.x.values + 3) + 1) - 10 / (
            abs(v_triple.x.values) + 1)

    v_double = Potential(grid)
    v_double.values = -10 / (abs(v_triple.x.values - 2) + 1) - 10 / (abs(v_triple.x.values + 2) + 1)

    v_single = Potential(grid)
    v_single.values = -10 / (abs(v_triple.x.values) + 1)

    plot_densities(v_triple, {
        "T_vN": lambda info: KELDA_NEW(v_single, 10),
        "TF": lambda info: PowerFunctional((np.pi ** 2) / 6, 3),
        # "vW": lambda info: VonWeizakerKE(),
    }, x_range=[-5, 5], rho_range=[-0.1, 2.5])


def plot_coulomb_well_orbitals_and_densities():
    grid = Grid(-10, 10, 301)
    v = Potential(grid)
    v.values = -10 / (abs(v.x.values) + 1)

    s = v.diagonalize_hamiltonian()
    N = 4
    rho = s.density(N)

    vw_E = CombinedDensityFunctional([ExternalPotential(v), VonWeizakerKE()])
    tf_E = CombinedDensityFunctional([ExternalPotential(v), ThomasFermiKE()])

    rho_vw, _ = minimize_density_functional(N, grid, vw_E)
    rho_tf, _ = minimize_density_functional(N, grid, tf_E, guess=rho_vw)

    import matplotlib.pyplot as plt

    def set_x(labels=False):
        plt.xlim([-5, 5])
        if not labels:
            plt.xticks([])

    plt.subplot(311)
    # plt.plot(v.x.values, v.values, label=r"$v$", color="black")
    plt.plot(v.x.values, s.density(N), label=r"$\rho_{exact}$")
    plt.plot(v.x.values, rho_vw.values, label=r"$\rho_{vW}$")
    plt.plot(v.x.values, rho_tf.values, label=r"$\rho_{TF}$")
    plt.ylabel(r"$\rho$")
    plt.legend()
    set_x()

    plt.subplot(312)
    plt.plot(v.x.values, rho.values ** 0.5, label=r"$\sqrt{\rho}$")
    for n in range(N):
        plt.plot(v.x.values, s.orbitals[n].values, label=fr"$\phi_{n}$")
    plt.legend()
    plt.ylabel(r"$\phi$")
    set_x()

    plt.subplot(313)
    tau = s.kinetic_energy_density(N).values * rho.values
    tau_tf = rho.tf_kinetic_energy_density * rho.values
    tau_vw = rho.vw_kinetic_energy_density * rho.values
    tau_vw_lap = rho.vw_lap_kinetic_energy_density * rho.values

    plt.plot(v.x.values, tau, label=r"$\tau_{exact}$")
    plt.plot(v.x.values, tau_tf, label=r"$\tau_{TF}$")
    plt.plot(v.x.values, tau_vw, label=r"$\tau_{vW}$")
    plt.plot(v.x.values, tau_vw_lap, label=r"$\tau_{LvW}$")
    plt.ylabel(r"$\tau$")
    plt.xlabel("$x$")
    plt.legend()
    set_x(labels=True)

    ax = plt.gca().inset_axes([0.02, 0.4, 0.4, 0.55])
    ax.plot(v.x.values, tau)
    ax.plot(v.x.values, tau_tf)
    ax.plot(v.x.values, tau_vw)
    ax.plot(v.x.values, tau_vw_lap)
    ax.set_yticks([])
    ax.set_xlim([-5, -1.473])
    ax.set_ylim([-0.06, 0.251])

    plt.show()


def plot_tau_vs_discontinuity():
    grid = Grid(-10, 10, 1001)
    v = Potential(grid)

    p_max = 4

    for p in range(p_max):

        color = p / (p_max-1)
        color = (color, 1-color, 0)

        v.values = -10 / (abs(v.x.values) + (2 ** (-p)))

        spectrum = v.diagonalize_hamiltonian()
        rho = spectrum.density(4)
        tau_exact = spectrum.kinetic_energy_density(4).values * rho.values
        tau_tf = rho.tf_kinetic_energy_density * rho.values

        plt.subplot(311)
        plt.plot(rho.values, tau_exact, color=color, label=f"s = {2**(-p)}")
        plt.plot(rho.values, tau_tf, color=color, linestyle=":")
        plt.legend()
        plt.xlabel(r"$\rho$")
        plt.ylabel(r"$\tau$")

        plt.subplot(312)
        plt.plot(rho.values / max(rho.values), tau_exact / max(tau_exact), color=color)
        plt.plot(rho.values / max(rho.values), tau_tf / max(tau_exact), color=color, linestyle=":")
        plt.xlabel(r"$\rho/\max(\rho)$")
        plt.ylabel(r"$\tau/\max(\tau_{exact})$")

        plt.subplot(313)
        plt.plot(rho.values, tau_exact - tau_tf, color=color)
        plt.xlabel(r"$\rho$")
        plt.ylabel(r"$\tau_{exact} - \tau_{TF}$")

    plt.figure()

    rho_tf, E_TF = minimize_density_functional(4, grid, CombinedDensityFunctional([
        ExternalPotential(v), ThomasFermiKE()
    ]))

    rho_kelda, E_KELDA = minimize_density_functional(4, grid, CombinedDensityFunctional([
        ExternalPotential(v), KELDA_NEW(v, 4)
    ]))

    plt.plot(rho.x.values, rho.values, label=f"$E_{{exact}} = {v.inner_product(rho) + spectrum.kinetic_energy(4)}$")
    plt.plot(rho.x.values, rho_tf.values, label=f"$E_{{TF}} = {E_TF}$")
    plt.plot(rho.x.values, rho_kelda.values, label=f"$E_{{KELDA}} = {E_KELDA}$")
    plt.legend()
    plt.show()


def plot_coulomb_well_lda():
    grid = Grid(-10, 10, 501)
    v = Potential(grid)
    v.values = -10 / (abs(v.x.values) + 1)

    s = v.diagonalize_hamiltonian()
    N = 4
    rho = s.density(N)

    plt.plot(v.x.values, v.values)
    plt.plot(v.x.values, rho.values)
    plt.figure()

    tau = s.kinetic_energy_density(N).values * rho.values
    tau_tf = rho.tf_kinetic_energy_density * rho.values
    tau_vw = rho.vw_kinetic_energy_density * rho.values
    tau_vw_lap = rho.vw_lap_kinetic_energy_density * rho.values

    plt.plot(rho.values, tau, label=r"$\tau_{exact}$")
    plt.plot(rho.values, tau_tf, label=r"$\tau_{TF}$")
    plt.plot(rho.values, tau_vw, label=r"$\tau_{vW}$")
    plt.plot(rho.values, tau_vw_lap, label=r"$\tau_{LvW}$")
    plt.xlabel(r"$\rho(x)$")
    plt.ylabel(r"$\tau(x)$")
    plt.legend(loc=2, bbox_to_anchor=[0.55, 0.95])

    ax = plt.gca().inset_axes([0.1, 0.4, 0.4, 0.5])
    ax.plot(rho.values, tau)
    ax.plot(rho.values, tau_tf)
    ax.plot(rho.values, tau_vw)
    ax.plot(rho.values, tau_vw_lap)
    ax.set_xlim([0, 0.5])
    ax.set_ylim([-0.055, 0.19])

    if False:
        plt.figure()
        tf_rho, tf_E = minimize_density_functional(
            N, grid, CombinedDensityFunctional([ExternalPotential(v), ThomasFermiKE()]), guess=rho)
        plt.plot(rho.x.values, rho.values, label=f"exact E = {v.inner_product(rho) + s.kinetic_energy(N)}")
        plt.plot(rho.x.values, tf_rho.values, label=f"TF E = {tf_E}")
        plt.legend()

    plt.show()


def plot_renorm_tf():
    grid = Grid(-10, 10, 501)
    v = Potential(grid)
    v.values = -10 / (abs(v.x.values) + 1)

    s = v.diagonalize_hamiltonian()
    N = 4
    rho = s.density(N)

    pi2_24, pi2_24_E = minimize_density_functional(
        N, grid, CombinedDensityFunctional([ExternalPotential(v), PowerFunctional((np.pi ** 2) / 24, 3)]))

    pi2_6, pi2_6_E = minimize_density_functional(
        N, grid, CombinedDensityFunctional([ExternalPotential(v), PowerFunctional((np.pi ** 2) / 6, 3)]))

    vw_rho, vw_e = minimize_density_functional(
        N, grid, CombinedDensityFunctional([ExternalPotential(v), VonWeizakerKE()])
    )

    kelda_rho, kelda_E = minimize_density_functional(
        N, grid, CombinedDensityFunctional([ExternalPotential(v), KELDA_NEW(v, N)]))

    plt.plot(rho.x.values, rho.values, label=f"exact E = {v.inner_product(rho) + s.kinetic_energy(N)}")
    plt.plot(rho.x.values, pi2_24.values, label=f"pi^2/24 E = {pi2_24_E}")
    plt.plot(rho.x.values, pi2_6.values, label=f"pi^2/6 E = {pi2_6_E}")
    plt.plot(rho.x.values, kelda_rho.values, label=f"KELDA E = {kelda_E}")
    plt.plot(rho.x.values, vw_rho.values, label=f"vW E = {vw_e}")
    plt.legend()

    plt.show()


def plot_coulomb_well_kelda():
    grid = Grid(-10, 10, 301)
    v = Potential(grid)
    v.values = -10 / (abs(v.x.values) + 1)

    s = v.diagonalize_hamiltonian()
    N = 4
    rho = s.density(N)
    E = v.inner_product(rho) + s.kinetic_energy(N)

    kelda_rho, kelda_E = minimize_density_functional(
        N, grid, CombinedDensityFunctional([ExternalPotential(v), KELDA_NEW(v, N)]))
    tf_rho, tf_E = minimize_density_functional(
        N, grid, CombinedDensityFunctional([ExternalPotential(v), ThomasFermiKE()]))
    vw_rho, ve_E = minimize_density_functional(
        N, grid, CombinedDensityFunctional([ExternalPotential(v), VonWeizakerKE()]))

    plt.plot(v.x.values, rho.values, label=rf"$\rho$  $E = {E:.5f}$")
    plt.plot(v.x.values, tf_rho, label=rf"$\rho_{{TF}}$  $E = {tf_E:.5f}$")
    plt.plot(v.x.values, vw_rho, label=rf"$\rho_{{vW}}$  $E = {ve_E:.5f}$")
    plt.plot(v.x.values, kelda_rho, label=rf"$\rho_{{REF}}$  $E = {kelda_E:.5f}$")
    plt.legend()
    plt.xlim([-5, 5])
    plt.xlabel("$x$")
    plt.ylabel(r"$\rho$")

    plt.show()


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

    tau_range = [-1, 7]
    rho_range = [0, 1.6]

    plt.suptitle(fr"$\rho \in [{rho_range[0]}, {rho_range[1]}]\;\; \tau \in [{tau_range[0]}, {tau_range[1]}]$")

    for i, n in enumerate(n_values):
        for j, v_name in enumerate(potentials):
            plt.subplot(len(n_values) + 2, len(potentials), 1 + j + i * len(potentials))

            f = KELDA(potentials[v_name], n)
            interp_densities = Tensor.linspace(0, max(f.reference_densities) * 2, 1000)
            plt.plot(f.reference_densities, f.reference_kinetic_energy_densities * f.reference_densities, color="blue")
            plt.plot(interp_densities, f.t_lda(interp_densities) * interp_densities, linestyle="dashed", color="red")

            plt.xlim(rho_range)
            plt.ylim(tau_range)

            plt.xticks([])
            plt.yticks([])

            if i == len(n_values) - 1:
                plt.xlabel(fr"$\rho$")
                plt.xticks([])

            if j == 0:
                plt.ylabel(fr"$\tau(\rho)$" + f"\nN = {n}")
                plt.yticks([])

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


def plot_ladder_operators(v: Potential, ladder: LadderOperator, show=True):
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
    if show:
        plt.show()


def plot_harmonic_ladder_operators(profile=False):
    v = Potential(Grid(-8, 8, 51))
    v.values = 0.5 * v.x.values ** 2
    plot_ladder_operators(v, DerivativeLadderOperator(), show=not profile)


if __name__ == "__main__":
    plot_harmonic_oscillator_densities_ladder()
