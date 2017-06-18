from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'float32,gpu,tensorflow'

from odin import nnet as N, backend as K
import tensorflow as tf

