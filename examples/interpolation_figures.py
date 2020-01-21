from __future__ import absolute_import, division, print_function

import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from odin import visual as vs
from odin.backend import interpolation

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sns.set()

all_interpolation = interpolation.get()
n = len(all_interpolation)
n_col = 5
n_row = int(np.ceil(n / 5))

x = np.linspace(0., 1., num=250).astype('float32')
plt.figure(figsize=(int(n_col * 3), int(n_row * 2.5)))
for idx, cls in enumerate(all_interpolation):
  plt.subplot(n_row, n_col, idx + 1)
  name = str(cls.__name__).split('.')[-1]
  y = cls()(x)
  plt.plot(x, y)
  plt.title(name)
plt.tight_layout()

x = np.arange(0, 250).astype('float32')
plt.figure(figsize=(int(n_col * 3), int(n_row * 2.5)))
for idx, cls in enumerate(all_interpolation):
  plt.subplot(n_row, n_col, idx + 1)
  name = str(cls.__name__).split('.')[-1]
  y = cls(cyclical=True, norm=50, delayIn=20, delayOut=10, vmin=1., vmax=2.)(x)
  plt.plot(x, y)
  plt.title(name)
plt.tight_layout()

vs.plot_save(log=True)
