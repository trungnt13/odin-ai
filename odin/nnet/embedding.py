from __future__ import print_function, division, absolute_import

import tensorflow as tf
from tensorflow.python.ops import init_ops

from odin import backend as K
from odin.autoconfig import randint
from odin.nnet.base import NNOp


class Embedding(NNOp):
  """ Embedding
  Parameters
  ----------
  input_size: int
      size of dictionary

  output_size : int
      number of dimension of the vector to represent each entity
      in the dictionary.

  W_inti : trainable variable, expression, numpy array or call-able
      Initial value (or specification) for initialize weights matrix
      which has size (input_szie, output_size)
  """

  def __init__(self, input_size, output_size,
               W_init=init_ops.random_uniform_initializer(seed=randint()), **kwargs):
    super(Embedding, self).__init__(**kwargs)
    self.input_size = input_size
    self.output_size = output_size
    self.W_init = W_init

  @property
  def embedding_shape(self):
    """ Return the estimated embedding matrix shape """
    return (self.input_size, self.output_size)

  def _initialize(self):
    self.get_variable_nnop(initializer=self.W_init,
                           shape=(self.input_size, self.output_size),
                           name='W', roles=K.role.EmbeddingWeight)

  def _apply(self, X):
    return tf.gather(self.W, tf.cast(X, tf.int32))
