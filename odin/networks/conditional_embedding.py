from typing import List

import numpy as np
import tensorflow as tf
from odin import backend as bk
from odin.networks.base_networks import SequentialNetwork
from odin.utils import as_tuple
from tensorflow.python.keras.layers import Dense, Embedding, Lambda, Layer
from typing_extensions import Literal

__all__ = [
    'get_conditional_embedding',
    'RepetitionEmbedding',
    'DictionaryEmbedding',
    'ProjectionEmbedding',
    'SequentialEmbedding',
    'IdentityEmbedding',
    'all_embedder',
]


# ===========================================================================
# Helpers
# ===========================================================================
class Embedder:

  @property
  def event_shape(self) -> List[int]:
    return list(self._event_shape)


def _to_categorical(inputs):
  if inputs.shape.ndims == 1:
    inputs = tf.expand_dims(tf.convert_to_tensor(inputs, dtype_hint=tf.int32),
                            axis=-1)
  elif inputs.shape[-1] > 1:
    inputs = tf.expand_dims(tf.argmax(inputs, axis=-1, output_type=tf.int32),
                            axis=-1)
  else:
    inputs = tf.convert_to_tensor(inputs, dtype_hint=tf.int32)
  return inputs


# ===========================================================================
# Main classes
# ===========================================================================
class IdentityEmbedding(Layer, Embedder):

  def __init__(self,
               n_classes: int,
               event_shape: List[int],
               name: str = 'IdentityEmbedding'):
    super().__init__(name=name)
    self.n_classes = int(n_classes)
    self._event_shape = as_tuple(event_shape, t=int)

  def call(self, inputs, **kwargs):
    return inputs


class RepetitionEmbedding(Layer, Embedder):
  """Expand and repeat the inputs so that it is concatenate-able to the
  shape"""

  def __init__(self,
               n_classes: int,
               event_shape: List[int],
               name: str = 'RepetitionEmbedding'):
    super().__init__(name=name)
    self.n_classes = int(n_classes)
    self._event_shape = as_tuple(event_shape, t=int)

  def call(self, inputs, **kwargs):
    event_dim = len(self.event_shape) + 1
    shape = inputs.shape
    ndim = len(shape)
    if ndim > event_dim:
      raise RuntimeError(f"Cannot broadcast inputs shape={shape[1:]} "
                         f"to event shape={self.event_shape}")
    elif ndim < event_dim:
      n = abs(event_dim - ndim)
      # first expand
      for _ in range(n):
        inputs = tf.expand_dims(inputs, axis=1)
      # now repeat
      for i, s in enumerate(inputs.shape[1:]):
        if s == 1 and self.event_shape[i] != 1:
          inputs = tf.repeat(inputs, self.event_shape[i], axis=i + 1)
    else:
      ...  # do nothing
    return inputs


class DictionaryEmbedding(Embedding, Embedder):
  """Turns positive integers (indexes) (or one-hot encoded vector)
  into dense vectors of fixed size, then reshape the vector to desire
  output shape.
  """

  def __init__(self,
               n_classes,
               event_shape,
               embeddings_initializer='uniform',
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None,
               **kwargs):
    event_shape = as_tuple(event_shape, t=int)
    super().__init__(input_dim=int(n_classes),
                     output_dim=int(np.prod(event_shape)),
                     embeddings_initializer=embeddings_initializer,
                     embeddings_regularizer=embeddings_regularizer,
                     activity_regularizer=activity_regularizer,
                     embeddings_constraint=embeddings_constraint,
                     mask_zero=False,
                     input_length=1,
                     **kwargs)
    self._event_shape = event_shape

  def call(self, inputs, **kwargs):
    inputs = _to_categorical(inputs)
    outputs = super().call(inputs)
    outputs = tf.squeeze(outputs, axis=1)  # remove the time dimension
    outputs = tf.reshape(outputs, [-1] + self.event_shape)
    return outputs


class ProjectionEmbedding(Dense, Embedder):
  """Using Dense network to project inputs to given `event_shape`"""

  def __init__(self,
               n_classes,
               event_shape,
               activation='linear',
               use_bias=True,
               name='ProjectionEmbedding',
               **kwargs):
    event_shape = as_tuple(event_shape, t=int)
    super().__init__(units=int(np.prod(event_shape)),
                     activation=activation,
                     use_bias=use_bias,
                     name=name,
                     **kwargs)
    self.n_classes = int(n_classes)
    self._event_shape = event_shape

  def call(self, inputs, **kwargs):
    outputs = super().call(inputs)
    outputs = tf.reshape(outputs, [-1] + self.event_shape)
    return outputs


class SequentialEmbedding(SequentialNetwork, Embedder):
  """A combination of both dictionary and projection embedding to transform
  the labels into the image space for concatenation.

  This approach is used in ConditionalGAN
  """

  def __init__(self,
               n_classes,
               event_shape,
               embedding_dim=50,
               activation='linear',
               use_bias=True,
               embeddings_initializer='uniform',
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               name='SequentialEmbedding'):
    event_shape = as_tuple(event_shape, t=int)
    layers = [
        Embedding(input_dim=int(n_classes),
                  output_dim=int(embedding_dim),
                  embeddings_initializer=embeddings_initializer,
                  embeddings_regularizer=embeddings_regularizer,
                  activity_regularizer=activity_regularizer,
                  embeddings_constraint=embeddings_constraint,
                  mask_zero=False,
                  input_length=1),
        # remove the time dimension
        Lambda(lambda x: tf.squeeze(x, axis=1)),
        Dense(int(np.prod(event_shape)),
              activation=activation,
              use_bias=use_bias,
              kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer,
              kernel_regularizer=kernel_regularizer,
              bias_regularizer=bias_regularizer,
              activity_regularizer=activity_regularizer,
              kernel_constraint=kernel_constraint,
              bias_constraint=bias_constraint)
    ]
    super().__init__(layers=layers, name=name)
    self._event_shape = event_shape

  def call(self, inputs, **kwargs):
    inputs = _to_categorical(inputs)
    outputs = super().call(inputs, **kwargs)
    outputs = tf.reshape(outputs, [-1] + self.event_shape)
    return outputs


# ===========================================================================
# others
# ===========================================================================
all_embedder = dict(repetition=RepetitionEmbedding,
                    projection=ProjectionEmbedding,
                    dictionary=DictionaryEmbedding,
                    sequential=SequentialEmbedding,
                    identity=IdentityEmbedding)


def get_conditional_embedding(
    method: Literal['repetition', 'projection', 'dictionary', 'identity']
) -> Embedder:
  r""" Three support method for conditional embedding:

      - 'repetition': repeat the labels to match the input image
      - 'projection': embed then project (using Dense layer)
      - 'dictionary': only embed to the given output shape
      - 'identity': keep the original labels
  """
  method = str(method).strip().lower()
  for name, cls in all_embedder.items():
    if method == name or method in name:
      return cls
  raise KeyError(f'Cannot find conditional embedding method for key: {method}, '
                 f'all support methods are: {all_embedder}')
