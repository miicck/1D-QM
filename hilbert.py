import numpy as np
from abc import ABC, abstractmethod


class Grid:

    def __init__(self, minimum: float, maximum: float, points: int):
        self._values = np.linspace(minimum, maximum, points)
        self._laplacian = None
        self._gradient = None

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def points(self) -> int:
        return len(self.values)

    @property
    def laplacian(self) -> np.ndarray:
        if self._laplacian is None:
            self._laplacian = np.zeros((self.points, self.points))

            for i in range(self.points):
                self._laplacian[i, i] = -2.0

            for i in range(self.points - 1):
                self._laplacian[i, i + 1] = 1.0

            for i in range(1, self.points):
                self._laplacian[i, i - 1] = 1.0

            self._laplacian /= (self.values[1] - self.values[0]) ** 2

        return self._laplacian

    @property
    def gradient(self) -> np.ndarray:
        if self._gradient is None:
            self._gradient = np.zeros((self.points, self.points))

            for i in range(1, self.points - 1):
                self._gradient[i, i - 1] = -0.5
                self._gradient[i, i + 1] = 0.5
            self._gradient /= self.values[1] - self.values[0]

        return self._gradient


class Function:

    def __init__(self, grid: Grid, values: np.ndarray = None):

        self._grid = grid

        if values is not None:
            values = np.array(values)
            assert values.shape == grid.values.shape
        else:
            values = np.zeros(grid.values.shape)

        self.values = values

    @property
    def x(self) -> Grid:
        return self._grid

    @property
    def values(self) -> np.ndarray:
        return self._values

    @values.setter
    def values(self, val: np.ndarray):
        self._values = val
        self._norm = None
        self.on_values_change()

    def on_values_change(self):
        pass

    @property
    def derivative(self) -> 'Function':
        return Function(self.x, self.x.gradient @ self.values)

    @property
    def laplacian(self) -> 'Function':
        lapf = Function(self.x)
        lapf.values = self.x.laplacian @ self.values
        return lapf

    @property
    def norm(self) -> float:
        if self._norm is None:
            self._norm = self.inner_product(self) ** 0.5
        return self._norm

    def normalize(self) -> None:
        self.values /= self.norm

    def inner_product(self, other: 'Function') -> float:
        return np.trapz(self.values * other.values, self.x.values)

    def plot(self, blocking=True):
        import matplotlib.pyplot as plt
        if not blocking:
            plt.ion()

        if not blocking:
            plt.clf()

        plt.plot(self.x.values, self.values)
        plt.xlabel("$x$")
        plt.ylabel("$f(x)$")

        if blocking:
            plt.show()
        else:
            plt.pause(0.01)
