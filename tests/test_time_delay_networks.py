from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense

from odin import networks_torch as nt
from odin.networks import (TimeDelay, TimeDelayConv, TimeDelayConvTied,
                           TimeDelayDense)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)

x = np.random.rand(12, 80, 23).astype('float32')
tdd = TimeDelay(
    fn_layer_creator=lambda: Dense(units=128),
    delay_context=(-2, -1, 0, 1, 2),  #
)
y = tdd(x)
print(y.shape)

tdd = TimeDelayDense(units=128)
y = tdd(x)
print(y.shape)

tdc = TimeDelayConv(units=128)
y = tdc(x)
print(y.shape)

tdct = TimeDelayConvTied(units=128)
y = tdct(x)
print(y.shape)
