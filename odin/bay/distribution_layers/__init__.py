from odin.bay.distribution_layers.continuous import *
from odin.bay.distribution_layers.count_layers import *
from odin.bay.distribution_layers.discrete import *
from odin.bay.distribution_layers.mixture_layers import *


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
