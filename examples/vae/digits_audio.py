from __future__ import print_function, division, absolute_import

import os
os.environ['ODIN'] = 'float32,gpu'

from odin import backend as K, nnet as N, fuel as F
from odin.utils import ctext, args_parse

# ===========================================================================
# Loading the dataset
# ===========================================================================
