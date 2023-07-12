from abc import ABC, abstractmethod
from typing import Union
from qm1d.tensor import Tensor


class Grid:

    def __init__(self, minimum: float, maximum: float, points: int):
        self._values = Tensor.linspace(minimum, maximum, points)
        self._laplacian = None
        self._gradient = None

    @property
    def values(self) -> Tensor:
        return self._values

    @property
    def spacing(self) -> float:
        return self._values[1] - self._values[0]

    @property
    def points(self) -> int:
        return len(self.values)

    @property
    def laplacian(self) -> Tensor:
        if self._laplacian is None:
            self._laplacian = Tensor.zeros((self.points, self.points))

            for i in range(self.points):
                self._laplacian[i, i] = -2.0

            for i in range(self.points - 1):
                self._laplacian[i, i + 1] = 1.0

            for i in range(1, self.points):
                self._laplacian[i, i - 1] = 1.0

            self._laplacian /= (self.values[1] - self.values[0]) ** 2

        return self._laplacian

    @property
    def gradient(self) -> Tensor:
        if self._gradient is None:
            self._gradient = Tensor.zeros((self.points, self.points))

            for i in range(1, self.points - 1):
                self._gradient[i, i - 1] = -0.5
                self._gradient[i, i + 1] = 0.5
            self._gradient /= self.values[1] - self.values[0]

        return self._gradient


class Function:

    def __init__(self, grid: Grid, values: Tensor = None):

        self._grid = grid

        if values is not None:
            values = values.copy()
            assert values.shape == grid.values.shape, \
                f"Values shape = {values.shape} != grid shape = {grid.values.shape}"
        else:
            values = Tensor.zeros(grid.values.shape)

        self.values = values

    @property
    def x(self) -> Grid:
        return self._grid

    @property
    def values(self) -> Tensor:
        return self._values

    @values.setter
    def values(self, val: Tensor):
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
        return Function(self.x, self.x.laplacian @ self.values)

    @property
    def norm(self) -> float:
        if self._norm is None:
            self._norm = self.inner_product(self) ** 0.5
        return self._norm

    def normalize(self) -> None:
        self.values /= self.norm

    @property
    def normalized(self) -> 'Function':
        f = Function(self.x, self.values)
        f.normalize()
        return f

    def inner_product(self, other: 'Function') -> float:
        return sum(self.values * other.values) * self.x.spacing

    def outer_product(self, other: 'Function') -> Tensor:
        return Tensor.outer(self.values, other.values)

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
