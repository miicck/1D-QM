from solver import schrodinger_1d, orbitals_from_density, density_from_orbitals
import numpy as np
import matplotlib.pyplot as plt

N = 5

x = np.linspace(-10, 10, 101)

rho = np.exp(-abs(x))
rho *= N / np.trapz(rho, x)

evals, phi, v = orbitals_from_density(x, rho, plot=False, n_elec_tol=0.1)

plt.subplot(221)
plt.plot(x, rho)
plt.plot(x, density_from_orbitals(phi, N))
plt.plot(x, v / max(abs(v)))

plt.subplot(222)
for i in range(N):
    plt.plot(x, phi[:, i] + i)

plt.subplot(223)
phi_vw = (rho/N)**0.5
plt.plot(x, phi[:, 0])
plt.plot(x, phi_vw)

plt.show()
