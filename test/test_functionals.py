import matplotlib.pyplot as plt
from functionals import *


def test_guassian_repulsion():
    density = Density(Grid(-10, 10, 101))
    density.values = np.exp(-density.x.values ** 2)
    density.particles = 2
    v = GuassianRepulsion()(density)


def test_minimize(plot=False):
    grid = Grid(-10, 10, 22)

    v = Potential(grid, grid.values ** 2 / 10)

    dens, value = minimize_density_functional(
        4, grid,
        [ExternalPotential(v), VonWeizakerKE()],
        plot=plot
    )

    assert np.allclose(dens.particles, 4)


def test_kelda(plot=False):
    if plot:
        import matplotlib.pyplot as plt
        from functionals import KELDA

        grid = Grid(-8, 8, 101)

        potentials = [
            ("Harmonic well", Potential(grid, 0.5 * grid.values ** 2)),
            ("Anharmonic well", Potential(grid, 0.5 * grid.values ** 4 - 4 * grid.values ** 2)),
            ("Coulomb-like well", Potential(grid, -1 / (abs(grid.values) + 1))),
            ("Coulomb-like double well",
             Potential(grid, -4 / (abs(grid.values - 2) * 8 + 1) - 4 / (abs(grid.values + 2) * 8 + 1))),
        ]

        n_cols = 2

        for i, (name, v) in enumerate(potentials):

            plt.subplot(1 + len(potentials) // n_cols, n_cols, i + 1)

            spectrum = v.calculate_eigenstates()
            n_max = 4

            for n in range(1, n_max + 1):
                f = n / n_max

                kelda = KELDA(v, force_interp=True)
                rho = spectrum.density(n)
                ke_dens = spectrum.kinetic_energy_density(n)
                kelda.apply(rho)

                x = np.linspace(min(rho.values), max(rho.values), 1000)

                plt.plot(x, kelda.t_lda(x), color=(f, 1 - f, 0), linestyle=":")
                plt.plot(rho.values, ke_dens.values, color=(f, 1 - f, 0))

            plt.xlabel("density")
            plt.ylabel("KE/electron")
            plt.annotate(name, (0, 0))

        plt.show()


if __name__ == "__main__":
    test_minimize(plot=True)
