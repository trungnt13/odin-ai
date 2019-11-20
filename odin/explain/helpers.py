import inspect

import numpy as np
import tensorflow as tf
from six import string_types
from tensorflow import keras


def get_pretrained_model(model, model_kwargs={}):
  if isinstance(model, keras.Model):
    return model
  if isinstance(model, string_types):
    pretrained_model = {
        name.lower(): obj
        for name, obj in inspect.getmembers(keras.applications)
        if inspect.isfunction(obj)
    }
    model = pretrained_model[model.strip().lower()]
    model = model(**model_kwargs)
    return model
  raise NotImplementedError("No support for model with type: %s" % type(model))


def _may_add_batch_dim(X, input_shape):
  # add batch dimension if necessary
  if X.ndim == len(input_shape) - 1:
    X = np.expand_dims(X, axis=0) if isinstance(X, np.ndarray) else \
      tf.expand_dims(X, axis=0)
  assert len(input_shape) == X.ndim and all(
      i == j if i is not None else True
      for i, j in zip(input_shape, X.shape)), \
        "Require input_shape=%s but X.shape=%s" % (input_shape, X.shape)
  return X
