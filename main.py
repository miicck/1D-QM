from solver import orbitals_from_density, plot_orbital_density_potential_suite
import numpy as np
import matplotlib.pyplot as plt

N = 5

x = np.linspace(-5, 5, 101)

rho = np.exp(-(x-1) ** 2) ** 2 + np.exp(-(x+1) ** 2) ** 2
rho = np.exp(-(x-2) ** 2) ** 2 + np.exp(-x ** 2) ** 2 + np.exp(-(x+2) ** 2) ** 2
#rho = np.exp(-x**2)**2
rho *= N / np.trapz(rho, x)

evals, phi, v, history = orbitals_from_density(x, rho, plot=False, n_elec_tol=0.01)

plot_orbital_density_potential_suite(x, rho, phi, N, v, history)
plt.show()
