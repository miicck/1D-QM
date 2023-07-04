import matplotlib.pyplot as plt
import numpy as np
from functionals import *


def test_gradient():
    grid = Grid(-8, 8, 101)
    v = Potential(grid, 0.5 * grid.values ** 2)
    functional = ExternalPotential(v)
    d = Density(v.x)
    df_dro = functional.functional_derivative_finite_difference(d)

    assert np.allclose(df_dro.values, functional.functional_derivative(d).values)

    for n in range(100):
        dd = Density(v.x, np.random.random(d.values.shape) * 1e-6)
        assert np.allclose(functional(dd), df_dro.inner_product(dd))


def test_minimize():
    grid = Grid(-8, 8, 22)
    v = Potential(grid, 0.5 * grid.values ** 2)
    functional = CombinedDensityFunctional([ExternalPotential(v), VonWeizakerKE()])
    density, energy = minimize_density_functional(4, grid, functional)
    assert np.allclose(density.particles, 4)


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

            spectrum = v.diagonalize_hamiltonian()
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
