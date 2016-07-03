from __future__ import print_function, division, absolute_import

import os
import inspect
from itertools import chain

import numpy as np

from odin.roles import add_role, has_roles, PARAMETER
from odin.nnet import NNOps
from odin.utils.decorators import functionable
