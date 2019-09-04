from odin.networks.advance_model import AdvanceModel
from odin.networks.distribution_util_layers import *
from odin.networks.math import *
from odin.networks.mixture_density_network import *
from odin.networks.stat_layers import *
from odin.networks.util_layers import *


def register_new_keras_layers(extras=None):
  from tensorflow.python.keras.layers import Layer
  from tensorflow.python.keras.utils.generic_utils import _GLOBAL_CUSTOM_OBJECTS
  globs = dict(globals())
  if extras is not None:
    globs.update(extras)
  for key, val in globs.items():
    if isinstance(val, type) and issubclass(val, Layer):
      _GLOBAL_CUSTOM_OBJECTS[key] = val


register_new_keras_layers()
