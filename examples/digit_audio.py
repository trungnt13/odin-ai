from __future__ import print_function, absolute_import, division

from odin.utils import ArgController

args = ArgController().add(
    'backend', 'gpu or cpu', 'gpu'
).parse()

import os
os.environ['ODIN'] = 'float32,gpu'

import numpy as np
