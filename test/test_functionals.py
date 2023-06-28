import matplotlib.pyplot as plt
from functionals import *


def test_guassian_repulsion():
    density = Density(Grid(-10, 10, 101))
    density.values = np.exp(-density.x.values ** 2)
    density.particles = 2
    v = GuassianRepulsion()(density)


def test_minimize(plot=False):
    grid = Grid(-10, 10, 31)

    v = Potential(grid, (grid.values - 1) ** 2 / 10)

    dens = minimize_density_functional(
        4, grid,
        [ExternalPotential(v), VonWeizakerKE()],
        plot=plot
    )

    assert np.allclose(dens.particles, 4)


if __name__ == "__main__":
    test_minimize(plot=True)
