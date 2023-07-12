from abc import ABC, abstractmethod
from typing import Iterable, Any, Tuple
import numpy as _np
import numpy as np
import torch

TORCH_DEVICE = "cpu"

torch.set_default_dtype(torch.float64)
torch.set_default_device(TORCH_DEVICE)


class _AbstractTensor:

    @classmethod
    @abstractmethod
    def from_numpy(cls, numpy_array) -> '_AbstractTensor':
        raise NotImplementedError

    @classmethod
    def from_values(cls, values):
        return cls.from_numpy(np.array(values))


def _getattr_replace_super_wtih_sub_(obj, item, super, sub, super_to_sub):
    to_wrap = getattr(obj, item)

    if not callable(to_wrap):
        class wrapper:
            def __getattr__(self, item):
                return _getattr_replace_super_wtih_sub_(to_wrap, item, super, sub, super_to_sub)

        return wrapper()

    def wrapper(*args, **kwargs):
        result = to_wrap(*args, **kwargs)
        if isinstance(result, super) and not isinstance(result, sub):
            result = super_to_sub(result)
        return result

    return wrapper


class _NumpyTensorMeta(type(_np.ndarray)):

    def __getattr__(self, item):
        return _getattr_replace_super_wtih_sub_(
            _np, item, _np.ndarray, _NumpyTensor, lambda x: x.view(_NumpyTensor))


class _NumpyTensor(_np.ndarray, _AbstractTensor, metaclass=_NumpyTensorMeta):

    def from_numpy(cls, numpy_array) -> '_AbstractTensor':
        return numpy_array.view(_NumpyTensor)


class _TorchTensorMeta(type(torch.Tensor)):

    def __getattr__(self, item):

        # Match numpy API
        if item == "random":
            torch.random.random = torch.rand

        if item == "identity":
            torch.identity = torch.eye

        def super_to_sub(sup):
            sup.__class__ = _TorchTensor
            return sup

        return _getattr_replace_super_wtih_sub_(
            torch, item, torch.Tensor, _TorchTensor, super_to_sub)


class _TorchTensor(torch.Tensor, _AbstractTensor, metaclass=_TorchTensorMeta):

    def copy(self) -> '_TorchTensor':
        return self.clone().detach()

    def __str__(self):
        if len(self.shape) == 0:
            return str(self.item())
        return super(_TorchTensor, self).__str__()

    def __format__(self, format_spec):
        if len(self.shape) == 0:
            return self.item().__format__(format_spec)
        return super(_TorchTensor, self).__format__(format_spec)

    def __array__(self, dtype=None):
        return self.cpu().numpy()

    @classmethod
    def from_numpy(cls, numpy_array) -> '_AbstractTensor':
        result = torch.from_numpy(numpy_array).to(torch.device(TORCH_DEVICE))
        result.__class__ = _TorchTensor
        return result

    @classmethod
    def asarray(cls, values):
        return cls.from_values(values)


Tensor = _NumpyTensor
