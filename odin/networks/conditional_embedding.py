from typing import List

import numpy as np
import tensorflow as tf
from odin import backend as bk
from odin.networks.base_networks import SequentialNetwork
from tensorflow.python.keras.layers import Dense, Embedding, Lambda, Layer
from typing_extensions import Literal

__all__ = [
    'get_conditional_embedding', 'RepeaterEmbedding', 'ConditionalEmbedding',
    'ConditionalProjection'
]


class Embedder:

  @property
  def embedding_shape(self) -> List[int]:
    raise NotImplementedError


def get_conditional_embedding(
    method: Literal['repeat', 'project', 'embed']) -> Embedder:
  r""" Three support method for conditional embedding:

      - 'repeat': repeat the labels to match the input image
      - 'project': embed then project (using Dense layer)
      - 'embed': only embed to the given output shape
  """
  method = str(method).strip().lower()
  classes = dict(repeat=RepeaterEmbedding,
                 project=ConditionalProjection,
                 embed=ConditionalEmbedding)
  for name, cls in classes.items():
    if method == name or method in name:
      return cls
  raise KeyError(
      "Cannot find conditional embedding method for key: %s, all support methods are: %s"
      % method, str(classes))


class RepeaterEmbedding(Layer, Embedder):
  r""" Expand and repeat the inputs so that it is concatenate-able to the
  output_shape """

  def __init__(self,
               num_classes: int,
               output_shape: List[int],
               name: str = 'RepeaterEmbedding'):
    super().__init__(name=name)
    self._shape = [int(i) for i in tf.nest.flatten(output_shape)]
    self._ndim = len(self._shape) + 1  # add batch_dim
    self.num_classes = int(num_classes)

  @property
  def embedding_shape(self):
    return tuple([None] + self._shape[:-1] + [self.num_classes])

  def call(self, inputs, **kwargs):
    shape = inputs.shape
    ndim = len(shape)
    if ndim > self._ndim:
      raise RuntimeError("Cannot broadcast inputs shape=%s to output shape=%s" %
                         (shape[1:], self._shape))
    elif ndim < self._ndim:
      n = abs(self._ndim - ndim)
      # first expand
      for _ in range(n):
        inputs = tf.expand_dims(inputs, axis=1)
      # now repeat
      for i, s in enumerate(inputs.shape[1:]):
        if s == 1 and self._shape[i] != 1:
          inputs = tf.repeat(inputs, self._shape[i], axis=i + 1)
    return inputs


class ConditionalEmbedding(Embedding, Embedder):
  r""" Turns positive integers (indexes) (or one-hot encoded vector)
  into dense vectors of fixed size, then reshape the vector to desire
  output shape.
  """

  def __init__(self,
               num_classes,
               output_shape,
               embeddings_initializer='uniform',
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None,
               **kwargs):
    output_shape = [int(i) for i in tf.nest.flatten(output_shape)]
    super().__init__(input_dim=int(num_classes),
                     output_dim=int(np.prod(output_shape)),
                     embeddings_initializer=embeddings_initializer,
                     embeddings_regularizer=embeddings_regularizer,
                     activity_regularizer=activity_regularizer,
                     embeddings_constraint=embeddings_constraint,
                     mask_zero=False,
                     input_length=1,
                     **kwargs)
    self._output_shape = output_shape

  @property
  def embedding_shape(self):
    return tuple([None] + self._output_shape)

  def call(self, inputs, **kwargs):
    if inputs.shape[-1] > 1:
      inputs = tf.expand_dims(tf.argmax(inputs, axis=-1, output_type=tf.int32),
                              axis=-1)
    outputs = super().call(inputs)
    outputs = tf.squeeze(outputs, axis=1)
    outputs = tf.reshape(outputs, [-1] + self._output_shape)
    return outputs


class ConditionalProjection(SequentialNetwork, Embedder):
  r""" A combination of both embedding and projection to transform the labels
  into the image space for concatenation.

  This approach is used in ConditionalGAN
  """

  def __init__(self,
               num_classes,
               output_shape,
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
               name='ConditionalProjection'):
    output_shape = [int(i) for i in tf.nest.flatten(output_shape)]
    self._output_shape = output_shape
    layers = [
        Embedding(input_dim=int(num_classes),
                  output_dim=int(embedding_dim),
                  embeddings_initializer=embeddings_initializer,
                  embeddings_regularizer=embeddings_regularizer,
                  activity_regularizer=activity_regularizer,
                  embeddings_constraint=embeddings_constraint,
                  mask_zero=False,
                  input_length=1),
        Lambda(lambda x: tf.squeeze(x, axis=1)),
        Dense(int(np.prod(output_shape)),
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

  @property
  def embedding_shape(self):
    return tuple([None] + self._output_shape)

  def call(self, inputs, **kwargs):
    if inputs.shape[-1] > 1:
      inputs = tf.expand_dims(tf.argmax(inputs, axis=-1, output_type=tf.int32),
                              axis=-1)
    outputs = super().call(inputs, **kwargs)
    outputs = tf.reshape(outputs, [-1] + self._output_shape)
    return outputs
