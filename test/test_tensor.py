from qm1d.tensor import Tensor


def test_linspace():
    t = Tensor.linspace(0, 1, 100)
    assert t.__class__ == Tensor
    assert abs(t[-1] - 1.0) < 1e-8


def test_random():
    t = Tensor.random.random((3, 3))
    assert t.__class__ == Tensor


def test_matmul():
    t = Tensor.random.random((3, 3))
    t2 = Tensor.random.random((3))
    t3 = t @ t2
    assert t3.__class__ == Tensor


def test_copy():
    t = Tensor.random.random((3, 3))
    t2 = t.copy()
    assert t2.__class__ == Tensor
    assert Tensor.allclose(t, t2)
    t2[0, 0] += 1
    assert not Tensor.allclose(t, t2)


def test_torch_convert():
    import numpy as np
    t = Tensor.linspace(0, 1, 101)
    print(t.device)
    t2 = Tensor.asarray(np.linspace(0, 1, 101))
    print(t2.device)
