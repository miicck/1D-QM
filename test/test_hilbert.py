import numpy as np
from qm1d.hilbert import *


def test_grid_creation():
    grid = Grid(-10, 10, 100)
    assert np.allclose(np.linspace(-10, 10, 100), grid.values)


def test_grid_laplacian():
    grid = Grid(-10, 10, 5)
    assert grid.laplacian.shape == (5, 5)


def test_function_creation():
    function = Function(Grid(-10, 10, 100))
    function.values = np.sin(function.x.values)
    assert np.allclose(function.values, np.sin(np.linspace(-10, 10, 100)))


def test_function_norm():
    function = Function(Grid(-10, 10, 100))
    function.values = np.sin(function.x.values)
    function.normalize()
    assert np.allclose(function.norm, 1.0)


def test_function_laplacian():
    function = Function(Grid(-10, 10, 1000))
    function.values = np.exp(-function.x.values ** 2)
    assert np.allclose(function.derivative.derivative.values, function.laplacian.values, atol=0.01)
