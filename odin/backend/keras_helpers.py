from tensorflow.python.util import nest
from tensorflow.python.keras.layers import Layer

__all__ = [
    'to_keras_objective',
    'copy_keras_metadata',
    'has_keras_meta',
    'add_trainable_weights'
]

def tied_session():
  """ Tied the tensorflow Session and keras Session together
  """
  from odin.autoconfig import get_session
  from tensorflow.python.keras.backend import set_session
  set_session(get_session())

def to_keras_objective(tensor, name=None):
  """ Convert any tensor to an objective function
  (i.e. loss or metric function) for keras fit Model
  """
  if name is None:
    name = tensor.name
  name = name.strip()
  if ':' in name:
    name = name.split(':')[0]
  fn = lambda *args: tensor
  fn.__name__ = str(name)
  return fn

def has_keras_meta(tensor):
  return hasattr(tensor, '_keras_history') and hasattr(tensor, '_keras_mask')

def copy_keras_metadata(keras_tensor, *new_tensors):
  if not hasattr(keras_tensor, '_keras_history') or \
  not hasattr(keras_tensor, '_keras_mask'):
    pass
  else:
    new_tensors = nest.flatten(new_tensors)
    history = keras_tensor._keras_history
    mask = keras_tensor._keras_mask
    for t in new_tensors:
      setattr(t, '_keras_history', history)
      setattr(t, '_keras_mask', mask)
  return new_tensors[0] if len(new_tensors) == 1 else new_tensors

def add_trainable_weights(layer, *variables):
  from odin.backend import is_variable
  variables = nest.flatten(variables)
  assert all(is_variable(v) for v in variables), \
  "All objects from variables must be instance of tensorflow.Variable"

  assert isinstance(layer, Layer), \
  "layer must be instance of tensorflow.python.keras.layers.Layer"

  variables = [v for v in variables
               if v not in layer._trainable_weights]
  layer._trainable_weights = layer._trainable_weights + variables
  return layer
