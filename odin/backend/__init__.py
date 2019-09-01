from __future__ import absolute_import, division, print_function

import inspect
import os
from collections import Mapping
from contextlib import contextmanager

from six.moves import builtins, cPickle

from odin.backend import (keras_callbacks, keras_helpers, losses, metrics,
                          tf_utils)
from odin.backend.tensor import *
from odin.utils import as_tuple, is_path, is_string
