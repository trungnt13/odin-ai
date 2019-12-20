from __future__ import absolute_import, division, print_function

import functools
import inspect
import os
from collections import Mapping
from contextlib import contextmanager

from six import add_metaclass
from six.moves import builtins, cPickle

from odin.backend import keras_callbacks, keras_helpers, losses, metrics
from odin.backend.alias import *
from odin.backend.interpolation import Interpolation
from odin.backend.keras_helpers import Trainer
from odin.backend.maths import *
from odin.backend.tensor import *
from odin.utils import as_tuple, is_path, is_string


# ===========================================================================
# Make the layers accessible through backend
# ===========================================================================
class _nn_meta(type):

  def __getattr__(cls, key):
    fw = get_framework()
    import torch
    import tensorflow as tf

    all_objects = {}
    if fw == torch:
      from odin import networks_torch
      all_objects.update(torch.nn.__dict__)
      all_objects.update(networks_torch.__dict__)
    elif fw == tf:
      from odin import networks
      from tensorflow.python.keras.engine import sequential
      from tensorflow.python.keras.engine import training
      all_objects.update(tf.keras.layers.__dict__)
      all_objects.update(networks.__dict__)
      all_objects.update(sequential.__dict__)
      all_objects.update(training.__dict__)
    else:
      raise NotImplementedError("No neural networks support for framework: " +
                                str(fw))
    return all_objects[key]


@add_metaclass(_nn_meta)
class nn:
  pass
