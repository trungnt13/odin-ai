from numpy import ndarray
from scipy.sparse import spmatrix
from tensorflow import Tensor
from typing_extensions import Literal
from typing import Union, List, Tuple

__all__ = [
    'TensorTypes',
]
TensorTypes = Union[spmatrix, ndarray, Tensor]
