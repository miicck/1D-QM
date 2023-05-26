import math

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
    lap /= 2 * ((x[1] - x[0]) ** 2)
    return lap


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
        import matplotlib.pyplot as plt
        plt.ion()

    n_elec = np.trapz(rho, x)
    v = np.zeros(rho.shape)
    kinetic_energies = []

    while True:
        evals, phi = schrodinger_1d(x, v)
        rho_phi = density_from_orbitals(phi, n_elec)
        d_rho = rho_phi - rho
        v += d_rho

        if np.trapz(abs(d_rho), x) < n_elec_tol:
            if plot:
                plt.ioff()
                plt.clf()
            return evals, phi, v

        if plot:
            plt.clf()

            plt.subplot(221)
            plt.plot(x, rho, label="rho")
            plt.plot(x, rho_phi, label="rho_phi")
            plt.plot(x, v / max(abs(v)), label="v")
            plt.legend()

            ke = 0.0

            plt.subplot(222)
            for i in range(math.ceil(n_elec)):
                plt.plot(x, phi[:, i] + i)
                plt.axhline(i, color="black", alpha=0.5)
                ke += -0.5 * np.trapz(laplacian(x, phi[:, i]), x)

            kinetic_energies.append(ke)

            plt.subplot(223)
            plt.plot(kinetic_energies)
            plt.ylabel("Kinetic energy")
            plt.xlabel("Iteration")

            plt.subplot(224)
            plt.plot(v / rho_phi)
            plt.ylabel("v/rho_phi")
            plt.yscale("symlog")

            plt.pause(0.01)
