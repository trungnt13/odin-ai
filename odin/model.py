from __future__ import print_function, division, absolute_import

import os
import inspect
from itertools import chain

import numpy as np

from sklearn.base import (BaseEstimator, ClassifierMixin,
                          TransformerMixin, RegressorMixin)

from odin import backend as K
from odin.roles import add_role, has_roles, PARAMETER
from odin.nnet import NNOps, Sequence
from odin.utils.decorators import functionable


class SequentialModel(BaseEstimator, TransformerMixin,
                      ClassifierMixin, RegressorMixin):
    """docstring for Model"""

    def __init__(self, *ops):
        super(SequentialModel, self).__init__()
        self._seq_ops = Sequence(ops, strict_transpose=False)
        self._initialized = False
        # list: (name, dtype, shape)
        self._input_info = []

    def set_inputs(self, *inputs):
        for i in inputs:
            if not K.is_placeholder(i):
                raise ValueError('Only accept input which is placeholder.')

    def fit(*arg, **kwargs):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass

    def transform(self, X):
        pass

    # ==================== pickling methods ==================== #
    def __setstate__(self, states):
        pass

    def __getstate__(self):
        pass
