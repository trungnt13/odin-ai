# TODO: Fix overlap import with odin.bay here
# from odin.networks import attention_mechanism
# from odin.networks.attention import *
from odin.networks.base_networks import *
from odin.networks.conditional_embedding import *
from odin.networks.cudnn_rnn import *
from odin.networks.dropout import *
from odin.networks.positional_encoder import *
from odin.networks.skip_connection import SkipConnection, skip_connect
from odin.networks.time_delay import *
from odin.networks.util_layers import *
from odin.networks.image_networks import *
from odin.networks.resnets import *

def register_new_keras_layers(extras=None):
  import tensorflow as tf
  from tensorflow.python.keras.layers import Layer
  custom_objects = tf.keras.utils.get_custom_objects()

  globs = dict(globals())
  if extras is not None:
    globs.update(extras)
  for key, val in globs.items():
    if isinstance(val, type) and issubclass(val, Layer):
      custom_objects[key] = val


register_new_keras_layers()
