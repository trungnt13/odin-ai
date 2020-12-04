from numpy import ndarray
from scipy.sparse import spmatrix
from tensorflow import Tensor
from typing_extensions import Literal
from typing import Union, List, Tuple, Callable
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import Model, Sequential

__all__ = [
    'NoneType',
    'TensorTypes',
    'LayerTypes',
    'BATCH',
    'EVENT',
    'MCMC',
    'CorrelationMethod',
]

CorrelationMethod = Literal['spearman', 'lasso', 'pearson', 'mutualinfo',
                            'importance']

NoneType = type(None)
TensorTypes = Union[spmatrix, ndarray, Tensor]
LayerTypes = Union[Layer, Model, Sequential, Callable[..., Layer],
                   Callable[[Tensor], Tensor]]

BATCH = Union[int, NoneType]
EVENT = int
MCMC = Union[int, NoneType]
