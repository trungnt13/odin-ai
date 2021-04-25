import tensorflow as tf
from numbers import Number
from typing import Callable, List, Union, Sequence

from numpy import ndarray
from scipy.sparse import spmatrix
from tensorflow import Tensor
from tensorflow.python.keras import Model, Sequential
from tensorflow.python.keras.layers import Layer
from typing_extensions import Literal

from odin.backend.interpolation import Interpolation
from tensorflow.python.keras.optimizer_v2.optimizer_v2 import OptimizerV2

__all__ = [
  'Coefficient',
  'NoneType',
  'TensorType',
  'LayerType',
  'BATCH',
  'EVENT',
  'MCMC',
  'CorrelationMethod',
  'Axes',
  'Axis',
  'DataType',
  'LabelType',
  'Scalar',
  'Optimizer',
  'Activation'
]

Coefficient = Union[Number, Interpolation]

CorrelationMethod = Literal[
  'spearman', 'lasso', 'pearson', 'mutualinfo', 'importance']

NoneType = type(None)
TensorType = Union[spmatrix, ndarray, Tensor]
Scalar = Union[Tensor, ndarray, Number]
LayerType = Union[Layer, Model, Sequential, Callable[..., Layer],
                  Callable[[Tensor], Tensor]]

BATCH = Union[int, NoneType]
EVENT = int
MCMC = Union[int, NoneType]

Axes = Union[int, Sequence[int]]
Axis = int

DataType = Literal['image', 'audio', 'text', 'gene']
LabelType = Literal['binary', 'categorical', 'factor']

Optimizer = OptimizerV2
Activation = Union[Callable[[TensorType], TensorType], str]
