from typing import Iterable, Union
from qm1d.tensor import Tensor

Comparable = Union[Tensor, Iterable, int, float]


def assert_all_close(m1: Comparable, m2: Comparable,
                     atol=1e-8, message: str = None,
                     rtol=0.0,
                     print_components: int = 100):
    m1, m2 = Tensor.asarray(m1), Tensor.asarray(m2)
    assert m1.shape == m2.shape, f"{message} (shape mismatch: {m1.shape} != {m2.shape})"

    assert isinstance(atol, float), "atol is not a float"
    assert isinstance(rtol, float), "rtol is not a float"

    def format_indices(i):
        j_width = len(str(max(m1.shape))) if len(m1.shape) > 0 else 1
        return ', '.join(f"{j:>{j_width}}" for j in i)

    def printout(m1: Comparable, m2: Comparable):
        ret = "First few components:\n"
        ret += f"{format_indices(m1.shape)}   {'m1':>18} {'m2':>18} {'m1/m2':>18}\n"
        for n, i in enumerate(Tensor.indicies(m1.shape)):
            ret += f"{format_indices(i)} : {m1[i]:>18.10f} {m2[i]:>18.10f} {m1[i] / m2[i]:>18.10f}\n"
            if n >= print_components:
                break
        return ret

    i_max = Tensor.argmax(abs(m1 - m2))

    assert Tensor.allclose(m1, m2, atol=atol, rtol=rtol), \
        f"{message or 'Matrices inequivalent'}\n" \
        f"Max differance = {diff[i_max]}\n" \
        f"          at i = {format_indices(i_max)}\n" \
        f"         m1[i] = {m1[i_max]:>20.10f}\n" \
        f"         m2[i] = {m2[i_max]:>20.10f}\n" \
        f"{printout(m1, m2)}"
