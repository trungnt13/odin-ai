from __future__ import print_function, division, absolute_import

import numpy as np

from odin import backend as K
from odin.basic import EmbeddingWeights
from .base import NNOps, NNConfig, nnops_initscope


class Embedding(NNOps):
    """ Embedding
    Parameters
    ----------
    input_size: int
        size of dictionary

    output_size : int
        number of dimension of the vector to represent each entity
        in the dictionary.

    W_inti : trainable variable, expression, numpy array or callable
        Initial value (or specification) for initialize weights matrix
        which has size (input_szie, output_size)
    """

    @nnops_initscope
    def __init__(self, input_size, output_size,
                 W_init=K.init.uniform, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.W_init = W_init

    @property
    def embedding_shape(self):
        """ Return the estimated embedding matrix shape """
        return (self.input_size, self.output_size)

    def _initialize(self):
        self.config.create_params(self.W_init,
                             shape=(self.input_size, self.output_size),
                             name='W', roles=EmbeddingWeights, nb_params=1)

    def _apply(self, x):
        input_shape = K.get_shape(x)
        output_shape = input_shape + (self.output_size,)
        x = K.gather(self.W, K.cast(x, 'int32'))
        K.add_shape(x, output_shape)
        return x
