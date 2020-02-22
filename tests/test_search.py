from __future__ import absolute_import, division, print_function

import itertools
import os

import numpy as np
import tensorflow as tf

from odin.search import (diagonal_beam_search, diagonal_bruteforce_search,
                         diagonal_greedy_search, diagonal_hillclimb_search)
from odin.utils import UnitTimer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)

shape = (8, 8)
mat = np.random.randint(0, 88, size=shape)
print(mat)

with UnitTimer():
  ids = diagonal_beam_search(mat)
print(ids)
print(mat[:, ids])
print(np.sum(np.diag(mat[:, ids])))

with UnitTimer():
  ids = diagonal_hillclimb_search(mat)
print(ids)
print(mat[:, ids])
print(np.sum(np.diag(mat[:, ids])))

with UnitTimer():
  ids = diagonal_greedy_search(mat)
print(ids)
print(mat[:, ids])
print(np.sum(np.diag(mat[:, ids])))

with UnitTimer():
  ids = diagonal_bruteforce_search(mat)
print(ids)
print(mat[:, ids])
print(np.sum(np.diag(mat[:, ids])))
