from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'float32,gpu,tensorflow'

import numpy as np

from odin import nnet as N, backend as K, fuel as F


# ===========================================================================
# Data and const
# ===========================================================================
ds = F.load_mnist()
print(ds)

# ===========================================================================
# Model
# ===========================================================================
input_desc = [
    N.VariableDesc(shape=(None, 28, 28), dtype='float32', name='X'),
    N.VariableDesc(shape=(None,), dtype='float32', name='y')
]
model = N.get_model_descriptor('ladder1')
K.set_training(True); y_train, cost = model(input_desc)
K.set_training(False); y_score = model()
