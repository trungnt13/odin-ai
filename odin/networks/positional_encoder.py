from __future__ import absolute_import, division, print_function

import numpy as np
from tensorflow.python import keras

from odin import backend as bk


class PositionalEncoder(keras.layers.Layer):
  r""" Positional encoding follow the approach in (Vaswani et al. 2017)
  For even dimension in the embedding:
    `PE(pos,2i) = sin(pos/10000^(2i/dmodel))`
  and for odd position:
    `PE(pos,2i+1) = cos(pos/10000^(2i/dmodel))`

  Reference:
    Vaswani, A., et al., 2017. Attention Is All You Need. arXiv:1706.03762 [cs].

  """

  def __init__(self,
               output_dim,
               max_len=10000,
               trainable=False,
               mask_zero=False):
    super().__init__()
    self.output_dim = output_dim
    self.mask_zero = bool(mask_zero)
    self.trainable = bool(trainable)
    self.supports_masking = mask_zero
    self.max_len = max_len

    # Applying the cosine to even columns and sin to odds.
    # if zero-masked, dont use the 0 position
    # (i - i % 2) create a sequence of (0,0,1,1,2,2,...) which is needed
    # for two running sequence of sin and cos in odd and even position
    position_encoding = np.array([[
        pos / np.power(10000, (i - i % 2) / output_dim)
        for i in range(output_dim)
    ] if pos != 0 or not mask_zero else [0.] * output_dim
                                  for pos in range(max_len)])
    # [max_len, output_dim]
    position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])  # dim 2i
    position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])  # dim 2i+1
    if not trainable:
      self.position_encoding = bk.array(position_encoding,
                                        dtype='float32',
                                        framework=self)
    else:
      self.position_encoding = bk.variable(initial_value=position_encoding,
                                           dtype='float32',
                                           trainable=True,
                                           framework=self)

  def compute_mask(self, inputs, mask=None):
    if not self.mask_zero:
      return None
    return bk.not_equal(inputs, 0)

  def call(self, sequence, training=None):
    with bk.framework_(self):
      # [batch_size, time_dim]
      positions = bk.tile(bk.expand_dims(bk.arange(sequence.shape[1]), 0),
                          [sequence.shape[0], 1])
      dtype = bk.dtype_universal(positions.dtype)
      if dtype not in ('int32', 'int64'):
        positions = bk.cast(positions, dtype='int32')
      pe = bk.embedding(indices=positions, weight=self.position_encoding)
      return pe

  def get_config(self):
    config = super().get_config()
    config.update({
        'output_dim': self.output_dim,
        'trainable': self.trainable,
        'mask_zero': self.mask_zero,
        'max_len': self.max_len
    })
    return config
