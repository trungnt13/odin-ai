from __future__ import absolute_import, division, print_function

import numpy as np

from odin import backend as bk
from odin.networks.attention import BaseAttention


class HardAttention(BaseAttention):

  def __init__(self,
               input_dim=None,
               causal=False,
               residual=True,
               return_attention=False,
               dropout=0,
               temporal_dropout=False,
               num_heads=0,
               heads_depth=1,
               heads_bias=True,
               heads_norm=0.,
               heads_activation='linear',
               heads_output_mode='cat',
               scale_initializer='vaswani',
               scale_tied=True,
               scale_trainable=False,
               attention_type='mul',
               name=None):
    super(HardAttention, self).__init__(name=name)

  def calculate_scores(self, query, key=None):
    raise NotImplementedError

  def call(self, query, value=None, key=None, mask=None, training=None):
    raise NotImplementedError
