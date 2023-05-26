from solver import schrodinger_1d, orbitals_from_density, density_from_orbitals
import numpy as np
import matplotlib.pyplot as plt

N = 5

x = np.linspace(-10, 10, 1001)

rho = np.exp(-abs(x))
rho *= N / np.trapz(rho, x)

evals, phi, v = orbitals_from_density(x, rho, plot=True, n_elec_tol=0.001)

plt.subplot(221)
plt.plot(x, rho, label="rho")
plt.plot(x, density_from_orbitals(phi, N), label="rho orbs")
plt.plot(x, v / max(abs(v)), label="v")
plt.legend()

plt.subplot(222)
for i in range(N):
    plt.plot(x, phi[:, i] + i)
    plt.axhline(i, color="black", alpha=0.5)

plt.subplot(223)
phi_vw = (rho / N) ** 0.5
plt.plot(x, phi[:, 0], label="phi_0")
plt.plot(x, phi_vw, label="sqrt(rho)")
plt.legend()

plt.show()
