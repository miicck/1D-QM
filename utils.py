from typing import Iterable, Union
from qm1d.tensor import Tensor

Comparable = Union[Tensor, Iterable, int, float]


def assert_all_close(t1: Comparable, t2: Comparable,
                     atol=1e-8, message: str = None,
                     rtol=0.0,
                     print_components: int = 100):
    t1, t2 = Tensor.asarray(t1), Tensor.asarray(t2)
    assert t1.shape == t2.shape, f"{message} (shape mismatch: {t1.shape} != {t2.shape})"
    assert isinstance(atol, float), "atol is not a float"
    assert isinstance(rtol, float), "rtol is not a float"

    def printout(m1: Comparable, m2: Comparable):
        result = f"{'element':>10} {'T1':>15} {'T2':>15} {'T1/T2':>15}"
        result = result + "\n" + "-" * len(result) + "\n"
        for i, (a, b) in enumerate(zip(m1, m2)):
            result += f"{i:>10} {a:>15.10f} {b:>15.10f} {a / b:>15.10f}\n"
            if i >= 100:
                break
        return result

    message = message or "Tensors inequivalent"
    assert Tensor.allclose(t1, t2, atol=atol, rtol=rtol), message + "\n" + printout(t1, t2)
