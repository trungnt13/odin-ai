from __future__ import absolute_import, division, print_function

import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from odin import visual as vs
from odin.backend.interpolation import Interpolation

sns.set()

n = len(Interpolation)
n_col = 5
n_row = int(np.ceil(n / 5))

x = np.linspace(0., 1., num=250)
plt.figure(figsize=(int(n_col * 2.5), int(n_row * 2.5)))
for idx, fi in enumerate(Interpolation):
  plt.subplot(n_row, n_col, idx + 1)
  name = str(fi).split('.')[-1]
  y = fi(x)
  plt.plot(x, y)
  plt.title(name)
plt.tight_layout()
vs.plot_save()
