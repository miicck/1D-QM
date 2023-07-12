from abc import ABC, abstractmethod
from typing import Iterable, Any, Tuple
import numpy as _np
import torch


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


class _NumpyTensor(_np.ndarray, metaclass=_NumpyTensorMeta):
    pass


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


class _TorchTensor(torch.Tensor, metaclass=_TorchTensorMeta):

    def copy(self) -> '_TorchTensor':
        return self.clone()


Tensor = _TorchTensor
