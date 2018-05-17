from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'gpu,float32'

import numpy as np

from odin import fuel as F, nnet as N, backend as K

ds = F.FMNIST.load()
print(ds)
