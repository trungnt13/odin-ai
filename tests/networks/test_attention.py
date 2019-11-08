from __future__ import absolute_import, division, print_function

import os
import unittest

import numpy as np

from odin import backend as bk
from odin import networks as net

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)


def _check_attention_outputs(self, y, scores):
  self.assertTrue(
      np.all(bk.isfinite(y).numpy()) and np.all(bk.isnotnan(scores).numpy()))


class AttentionModelTest(unittest.TestCase):

  def test_attention(self):
    n = 18
    Tq = 12
    Tv = 8
    dim = 25
    query = np.random.rand(n, Tq, dim).astype('float32')
    key = np.random.rand(n, Tv, dim).astype('float32')
    value = np.random.rand(n, Tv, dim).astype('float32')
    q_mask = np.random.randint(0, 2, size=(n, Tq)).astype('int32')
    v_mask = np.random.randint(0, 2, size=(n, Tv)).astype('int32')

    for atype in ('loc', 'mul', 'add'):
      for num_heads in (4, 1, 0):
        for hard_attention in (True, False):
          print("Type:", atype, "#heads:", num_heads)
          att = net.Attention(input_dim=dim,
                              return_attention=True,
                              attention_type=atype,
                              num_heads=num_heads,
                              heads_regularization=1e-4,
                              causal=True,
                              residual=True,
                              dropout=0.3,
                              hard_attention=hard_attention)
          # masks
          y_train, a_train = att(query,
                                 key,
                                 value, (q_mask, v_mask),
                                 training=True)
          y_infer, a_infer = att(query,
                                 key,
                                 value, (q_mask, v_mask),
                                 training=False)
          # no masks
          y_train, a_train = att(query, key, value, training=True)
          y_infer, a_infer = att(query, key, value, training=False)
          #
          _check_attention_outputs(self, y_train, a_train)
          _check_attention_outputs(self, y_infer, a_infer)


if __name__ == '__main__':
  unittest.main()
