from numbers import Number
from typing import Callable, List, Union

from numpy import ndarray
from scipy.sparse import spmatrix
from tensorflow import Tensor
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Layer
from typing_extensions import Literal

from odin.backend.interpolation import Interpolation

__all__ = [
    'Coefficient',
    'NoneType',
    'TensorTypes',
    'LayerTypes',
    'BATCH',
    'EVENT',
    'MCMC',
    'CorrelationMethod',
    'Axes',
    'Axis',
]

Coefficient = Union[Number, Interpolation]

CorrelationMethod = Literal['spearman', 'lasso', 'pearson', 'mutualinfo',
                            'importance']

NoneType = type(None)
TensorTypes = Union[spmatrix, ndarray, Tensor]
LayerTypes = Union[Layer, Model, Sequential, Callable[..., Layer],
                   Callable[[Tensor], Tensor]]

BATCH = Union[int, NoneType]
EVENT = int
MCMC = Union[int, NoneType]

Axes = Union[int, List[int]]
Axis = int
