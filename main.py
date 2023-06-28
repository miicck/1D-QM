from solver import orbitals_from_density, plot_orbital_density_potential_suite
import numpy as np
import matplotlib.pyplot as plt
from hilbert import *
from functions import *

rho = Density(Grid(-5, 5, 101))

x = rho.x.values
rho.values = np.exp(-(x - 1) ** 2) ** 2 + np.exp(-(x + 1) ** 2) ** 2
rho.particles = 5

rho.plot_with_potential_and_orbitals()
