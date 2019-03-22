from tensorflow.python.util import nest

__all__ = [
    'to_keras_objective',
    'copy_keras_metadata'
]

def tied_session():
  """ Tied the tensorflow Session and keras Session together
  """
  from odin.config import get_session
  from tensorflow.python.keras.backend import set_session
  set_session(get_session())

def to_keras_objective(tensor, name):
  """ Convert any tensor to an objective function
  (i.e. loss or metric function) for keras fit Model
  """
  name = name.strip()
  if ':' in name:
    name = name.split(':')[0]
  fn = lambda *args: tensor
  fn.__name__ = str(name)
  return fn

def copy_keras_metadata(keras_tensor, *new_tensors):
  if not hasattr(keras_tensor, '_keras_history') or \
  not hasattr(keras_tensor, '_keras_mask'):
    raise ValueError(
        "keras_tensor must has Keras metadata _keras_mask and _keras_history")

  new_tensors = nest.flatten(new_tensors)
  history = keras_tensor._keras_history
  mask = keras_tensor._keras_mask
  for t in new_tensors:
    setattr(t, '_keras_history', history)
    setattr(t, '_keras_mask', mask)
  return new_tensors[0] if len(new_tensors) == 1 else new_tensors
