from __future__ import absolute_import, division, print_function

import os

import numpy as np
import torch

from odin import networks_torch as nt
from odin.backend import parse_optimizer

torch.manual_seed(8)
np.random.seed(8)

x = torch.Tensor(np.random.rand(12, 8))

# f = nt.Dense(units=512)
# y = f(x)

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
