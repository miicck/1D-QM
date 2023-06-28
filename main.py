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
    rho.particles = 5

    # Plot comparison with ladder operator
    plot_ladder_result(rho)


def vw_vs_exact_ke_plot():
    # Create an harmonic potential
    grid = Grid(-10, 10, 51)
    v = Potential(grid, grid.values ** 2 / 10)

    dvw = minimize_density_functional(2, grid, [ExternalPotential(v), VonWeizakerKE()])
    dex = minimize_density_functional(2, grid, [ExternalPotential(v), ExactKineticEnergyFunctional(max_iter=10)])

    plt.plot(v.x.values, dvw.values, label="vW density")
    plt.plot(v.x.values, dex.values, label="exact density")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    vw_vs_exact_ke_plot()
