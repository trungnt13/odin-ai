from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf
import torch

from odin import backend as K
from odin import networks as net  # tensorflow networks
from odin import networks_torch as nt  # pytorch networks

tf.random.set_seed(8)
torch.manual_seed(8)
np.random.seed(8)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

x = torch.Tensor(np.random.rand(12, 8))
x1 = torch.Tensor(np.random.rand(12, 25, 8))
# ===========================================================================
# RNN
# ===========================================================================
f = nt.LSTM(units=32,
            go_backwards=True,
            unit_forget_bias=True,
            return_sequences=True,
            return_state=True,
            bidirectional=True)
y = f(x1)
print(x1.shape, [i.shape for i in y])

f = nt.SimpleRNN(units=32, go_backwards=True)
y = f(x1)
print(x1.shape, y.shape)

f = nt.GRU(units=32, go_backwards=False, return_state=True)
y = f(x1)
print(x1.shape, [i.shape for i in y])

# ====== tensorflow ====== #
print()
f = net.LSTM(units=32,
             go_backwards=True,
             unit_forget_bias=True,
             return_sequences=True,
             return_state=True,
             bidirectional=True)
y = f(x1.numpy())
print(x1.shape, [i.shape for i in y])

f = net.SimpleRNN(units=32, go_backwards=True)
y = f(x1.numpy())
print(x1.shape, y.shape)

f = net.GRU(units=32, go_backwards=False, return_state=True)
y = f(x1.numpy())
print(x1.shape, [i.shape for i in y])

print()
# ===========================================================================
# Basics
# ===========================================================================
f = nt.Dense(units=512)
y = f(x)
print(x.shape, y.shape)

# ===========================================================================
# CNN
# ===========================================================================
x = torch.Tensor(np.random.rand(12, 25, 8))
f = nt.Conv1D(filters=128, kernel_size=3)
y = f(x)
print(x.shape, y.shape)

x = torch.Tensor(np.random.rand(12, 25, 8))
f = nt.ConvCausal(filters=128, kernel_size=3)
y = f(x)
print(x.shape, y.shape)

x = torch.Tensor(np.random.rand(12, 25, 8))
f = nt.Conv1D(filters=128, kernel_size=3, data_format='channels_first')
y = f(x)
print(x.shape, y.shape)

x = torch.Tensor(np.random.rand(12, 32, 32, 3))
f = nt.Conv2D(filters=128, kernel_size=3, padding='same')
y = f(x)
print(x.shape, y.shape)

x = torch.Tensor(np.random.rand(12, 32, 32, 32, 3))
f = nt.Conv3D(filters=128, kernel_size=3)
y = f(x)
print(x.shape, y.shape)
