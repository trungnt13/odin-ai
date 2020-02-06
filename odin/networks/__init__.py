from odin.networks import attention_mechanism
from odin.networks.attention import *
from odin.networks.cudnn_rnn import *
from odin.networks.math_layers import *
from odin.networks.positional_encoder import *
from odin.networks.sequential_networks import *
from odin.networks.time_delay import *
from odin.networks.util_layers import *
from odin.networks.dropout import *


def register_new_keras_layers(extras=None):
  from tensorflow.python.keras.layers import Layer
  import tensorflow as tf
  custom_objects = tf.keras.utils.get_custom_objects()

  globs = dict(globals())
  if extras is not None:
    globs.update(extras)
  for key, val in globs.items():
    if isinstance(val, type) and issubclass(val, Layer):
      custom_objects[key] = val


register_new_keras_layers()
