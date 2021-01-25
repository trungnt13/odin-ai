from odin.bay.layers.continuous import *
from odin.bay.layers.count_layers import *
from odin.bay.layers.dense_distribution import *
from odin.bay.layers.deterministic_layers import *
from odin.bay.layers.discrete import *
from odin.bay.layers.distribution_util_layers import *
from odin.bay.layers.latents import *
from odin.bay.layers.mixture_layers import *
from odin.bay.layers.autoregressive_layers import *


def _register_distribution_layers():
  # For deserialization.
  import tensorflow as tf
  import inspect
  custom_objects = tf.keras.utils.get_custom_objects()

  for key, value in globals().items():
    if key not in custom_objects and \
      inspect.isclass(value) and \
      issubclass(value, DistributionLambda):
      custom_objects[key] = value


_register_distribution_layers()
