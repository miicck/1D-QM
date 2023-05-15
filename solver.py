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
    n_orb = math.ceil(n_elec)
    n_homo = n_elec - math.floor(n_elec)

    if abs(n_homo) < 1e-6:
        assert abs(n_orb - n_elec) < 1e-6
        return sum(orbs[:, i] ** 2 for i in range(n_orb))

    return sum(orbs[:, i] ** 2 for i in range(n_orb - 1)) + orbs[:, n_orb - 1] ** 2 * n_homo


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
    v = np.zeros(x.shape)

    while True:
        evals, phi = schrodinger_1d(x, v)
        rho_phi = density_from_orbitals(phi, n_elec)
        d_rho = rho_phi - rho
        v += d_rho

        if np.trapz(abs(d_rho), x) < n_elec_tol:
            if plot:
                plt.ioff()
            return evals, phi, v

        if plot:
            plt.clf()

            plt.subplot(221)
            plt.plot(x, rho)
            plt.plot(x, rho_phi)
            plt.plot(x, v / max(abs(v)))

            plt.subplot(222)
            for i in range(math.ceil(n_elec)):
                plt.plot(x, phi[:, i] + i)

            plt.pause(0.01)
