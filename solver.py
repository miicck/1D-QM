import math
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as sla


def schrodinger_1d(x: np.ndarray, v: np.ndarray):
    """
    Solves the 1-D scrhodinger equation
    Parameters
    ----------
    x: np.ndarray
        Spatial grid
    v: np.ndarray
        External potential evaluated on the grid

    Returns
    -------
    evals, evecs: np.ndarray
        Eigenvalues and eigenvectors
    """

    dx = x[1] - x[0]

    # Kinetic energy part
    T = np.identity(len(x))
    for i in range(len(x) - 1):
        T[i, i + 1] = -1
        T[i + 1, i] = -1
    T = T / (dx ** 2)

    eigenvalues, eigenvectors = np.linalg.eigh(T + np.diag(v))
    eigenvalues = np.real(eigenvalues)

    for i in range(len(eigenvalues)):

        # Normalize eigenvectors
        norm = np.trapz(np.conj(eigenvectors[:, i]) * eigenvectors[:, i], x) ** 0.5
        eigenvectors[:, i] = eigenvectors[:, i] / norm

        # Ensure consistent sign
        for j in range(len(eigenvectors)):
            if abs(eigenvectors[j, i]) > 1e-4:
                eigenvectors[:, i] *= np.sign(eigenvectors[j, i])
                break

    return eigenvalues, eigenvectors


def density_from_orbitals(orbs: np.ndarray, n_elec: float):
    """
    Returns the electron density obtained by populating
    the given orbitals with the given number of electrons
    according to the aufbau sceheme.

    Parameters
    ----------
    orbs: np.ndarray
        obs[:, i] is the i^th orbital
    n_elec: float
        number of electrons to put into the given orbitals

    Returns
    -------
    density: np.ndarray
    """
    i = 0
    result = np.zeros(orbs[:, 0].shape)
    while n_elec > 0:
        if n_elec < 1:
            result += n_elec * orbs[:, i] ** 2
            break

        result += orbs[:, i] ** 2
        n_elec -= 1.0
        i += 1

    return result


def gradient(x: np.ndarray, orbital: np.ndarray) -> np.ndarray:
    """
    Returns the gradient of the given orbital d\phi/dx

    Parameters
    ----------
    x: np.ndarray
        coordinates
    orbital: np.ndarray
        orbital \phi

    Returns
    -------
    grad: np.ndarray
        Gradient of the given orbital
    """
    grad = np.zeros(orbital.shape)
    for i in range(1, len(orbital)):
        grad[i] = (orbital[i] - orbital[i - 1]) / (x[i] - x[i - 1])
    return grad


def laplacian(x: np.ndarray, orbital: np.ndarray) -> np.ndarray:
    """
    Returns the laplacian of the given orbital d^2\phi/dx^2.

    Parameters
    ----------
    x: np.ndarray
        coordinates
    orbital: np.ndarray
        orbital \phi

    Returns
    -------
    lap: np.ndarray
        Laplacian of the given orbital
    """
    lap = np.zeros(orbital.shape)
    for i in range(2, len(orbital)):
        lap[i - 1] = orbital[i - 2] - 2 * orbital[i - 1] + orbital[i]

    lap[0] = lap[1]
    lap[-2] = lap[-1]

    lap /= (x[1] - x[0]) ** 2
    return lap


def plot_orbital_density_potential_suite(
        x: np.ndarray,
        rho: np.ndarray,
        orbs: np.ndarray,
        nelec: float,
        v: np.ndarray,
        history: Dict):
    """
    Generate the default suite of plots for an orbital/density/potential combination.

    Parameters
    ----------
    x: np.ndarray
        Coordinates
    rho: np.ndarray
        Density
    orbs: np.ndarray
        Orbitals
    nelec: float
        Number of electrons
    v: np.ndarray
        Potential
    history: Dict
        Optimization history

    Returns
    -------
    None
    """

    def new_plot():
        plt.subplot(3, 3, new_plot.n)
        new_plot.n += 1

    new_plot.n = 1

    # Evaluate derived quantities
    orb_g = (rho / np.trapz(rho, x))**0.5
    orb_g = orbs[:, 0]
    rho_phi = density_from_orbitals(orbs, nelec)

    # Plot density
    new_plot()
    plt.plot(x, rho, label="rho")
    plt.plot(x, rho_phi, label=f"rho_phi N = {np.trapz(rho_phi, x):.5f}")
    plt.plot(x, v / max(abs(v)), label="v")
    plt.legend()

    # Plot orbitals
    new_plot()
    for i in range(math.ceil(nelec)):
        orb = orbs[:, i]
        orb /= max(orb) - min(orb)
        plt.plot(x, orb + i)
        plt.axhline(i, color="black", alpha=0.5)

    orb = orb_g.copy()
    orb /= max(orb) - min(orb)
    plt.plot(x, orb, color="black", linestyle=":", label=r"$g$")

    plt.legend()
    plt.ylabel(r"$\phi_i$")

    # Plot gradient of orbitals
    new_plot()
    for i in range(math.ceil(nelec)):
        grad = gradient(x, orbs[:, i])
        grad /= max(grad) - min(grad)
        plt.plot(x, grad + i)
        plt.axhline(i, color="black", alpha=0.5)
    plt.ylabel(r"$d\phi_i/dx$")

    # Plot laplacian of orbitals
    new_plot()
    for i in range(math.ceil(nelec)):
        lap = laplacian(x, orbs[:, i])
        lap /= max(lap) - min(lap)
        plt.plot(x, lap + i)
        plt.axhline(i, color="black", alpha=0.5)
    plt.ylabel(r"$\nabla^2 \phi_i$")

    # Plot successive gradients of vw orbital
    new_plot()
    grad = orb_g.copy()
    ke_from_grads = 0.0
    v_from_grads = 0.0
    for i in range(math.ceil(nelec)):
        # Normalize/eval KE
        grad /= np.trapz(grad * grad, x) ** 0.5
        ke_from_grads += -0.5 * np.trapz(laplacian(x, grad) * grad, x)
        v_from_grads += np.trapz(grad * grad * v, x)

        # Plot scaled
        grad /= max(grad) - min(grad)
        plt.plot(x, grad + i)
        plt.axhline(i, color="black", alpha=0.5)
        grad = gradient(x, grad)

    plt.ylabel(r"$d^ng/dx^n$")
    plt.annotate(f"KE = {ke_from_grads}\n"
                 f"V  = {v_from_grads}", (0, 0.5))

    new_plot()
    plt.plot(history["E_T"], label="T")
    plt.plot(history["E_V"], label="V")
    plt.plot(history["E_tot"], label="Total")
    plt.ylabel("Energy")
    plt.yscale("symlog")
    plt.xlabel("Iteration")
    plt.legend()

    new_plot()
    plt.plot(x, v / rho_phi)
    plt.ylabel("v/rho_phi")
    plt.yscale("symlog")

    new_plot()
    plt.plot(x, orb_g / orbs[:, 0])
    plt.ylabel(r"$g/\phi_0$")
    plt.yscale("symlog")


def orbitals_from_density(x: np.ndarray, rho: np.ndarray, plot: bool = False, n_elec_tol=0.1):
    """
    Works out the orbitals which result in the given density.

    Parameters
    ----------
    x: np.ndarray
        Coordinates
    rho: np.ndarray
        Density
    plot: bool
        Set to True to plot as the solver works
    n_elec_tol: float
        Stop once the density from the
        orbitals is equal to the input density.

    Returns
    -------
    evals: np.ndarray
        Eigenvalues of the orbitals
    phi: np.ndarray
        Orbitals that resolve to the density
    v: np.ndarray
        Effective potential that generates the returned orbitals
    """
    if plot:
        plt.ion()

    n_elec = np.trapz(rho, x)
    v = np.zeros(rho.shape)
    history = {
        "E_T": [],
        "E_V": [],
        "E_tot": []
    }

    while True:
        evals, phi = schrodinger_1d(x, v)
        rho_phi = density_from_orbitals(phi, n_elec)
        d_rho = rho_phi - rho
        v += d_rho

        # Work out kinetic energy
        history["E_T"].append(
            sum(-0.5 * np.trapz(laplacian(x, phi[:, i]) * phi[:, i], x)
                for i in range(math.ceil(n_elec)))
        )

        # Work out potential energy
        history["E_V"].append(np.trapz(v * rho_phi, x))

        # Work out total energy
        history["E_tot"].append(history["E_V"][-1] + history["E_T"][-1])

        if np.trapz(abs(d_rho), x) < n_elec_tol:
            if plot:
                plt.ioff()
                plt.clf()
            return evals, phi, v, history

        if plot:
            plt.clf()
            plot_orbital_density_potential_suite(x, rho, phi, n_elec, v, history)
            plt.pause(1e-6)
