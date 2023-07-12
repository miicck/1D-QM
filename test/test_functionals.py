import matplotlib.pyplot as plt
from functionals import *


def test_gradient_v_ext():
    grid = Grid(-8, 8, 101)
    v = Potential(grid, 0.5 * grid.values ** 2)
    d = Density(v.x)

    functional = ExternalPotential(v)
    df_dro = functional.functional_derivative_finite_difference(d)
    df_drho_fd = functional.functional_derivative_finite_difference(d)

    assert Tensor.allclose(df_dro.values, df_drho_fd.values)

    for n in range(100):
        dd = Density(v.x, Tensor.random.random(d.values.shape) * 1e-6)
        assert Tensor.allclose(functional(dd), df_dro.inner_product(dd))


def test_gradient_vw():
    grid = Grid(-8, 8, 101)
    v = Potential(grid, 0.5 * grid.values ** 2)
    functional = VonWeizakerKE()
    d = Density(v.x)
    d.values = 1 + Tensor.exp(-d.x.values ** 2)

    df_drho = functional.functional_derivative(d)
    df_drho_fd = functional.functional_derivative_finite_difference(d)

    assert_all_close(df_drho.values[1:-2], df_drho_fd.values[1:-2], atol=0.01,
                     message="df/drho fails against finite differences")


def test_gradient_kelda(plot=False):
    grid = Grid(-8, 8, 101)
    v = Potential(grid, 0.5 * grid.values ** 2)
    functional = KELDA(v, 4)
    d = Density(v.x, Tensor.exp(-v.x.values ** 2))
    d.particles = 4
    df_drho = functional.functional_derivative(d)
    df_drho_fd = functional.functional_derivative_finite_difference(d)

    if plot:
        import matplotlib.pyplot as plt

        dref = functional.reference_densities
        kref = functional.reference_kinetic_energy_densities
        drange = Tensor.linspace(0, max(dref) * 2, 1000)

        plt.subplot(221)
        plt.plot(drange, functional.t_lda(drange), label="Interpolation")
        plt.plot(dref, kref, label="Reference data")
        plt.legend()

        plt.subplot(222)
        plt.plot(drange, functional.t_lda_derivative(drange))

        plt.figure()
        plt.plot(df_drho.values)
        plt.plot(df_drho_fd.values)
        plt.show()

    assert_all_close(df_drho.values, df_drho_fd.values, atol=0.1,
                     message="df/drho fails against finite differences")


def test_minimize():
    grid = Grid(-8, 8, 22)
    v = Potential(grid, 0.5 * grid.values ** 2)
    functional = CombinedDensityFunctional([ExternalPotential(v), VonWeizakerKE()])
    density, energy = minimize_density_functional(4, grid, functional)
    assert abs(density.particles - 4) < 1e-8


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

                x = Tensor.linspace(min(rho.values), max(rho.values), 1000)

                plt.plot(x, kelda.t_lda(x), color=(f, 1 - f, 0), linestyle=":")
                plt.plot(rho.values, ke_dens.values, color=(f, 1 - f, 0))

            plt.xlabel("density")
            plt.ylabel("KE/electron")
            plt.annotate(name, (0, 0))

        plt.show()


if __name__ == "__main__":
    test_minimize(plot=True)
