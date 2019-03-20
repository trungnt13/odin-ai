def to_keras_objective(tensor, name):
  """ Convert any tensor to an objective function
  (i.e. loss or metric function) for keras fit Model
  """
  fn = lambda *args: tensor
  fn.__name__ = str(name)
  return fn
