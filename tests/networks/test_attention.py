from __future__ import absolute_import, division, print_function

import os
import unittest

import numpy as np
from tqdm import tqdm

from odin import backend as bk
from odin import networks as net
from odin.networks.attention_mechanism import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

np.random.seed(8)

n = 18
Tq = 8
Tv = 12
dim = 25


def _check_attention_outputs(self, y, scores):
  self.assertTrue(
      np.all(bk.isfinite(y).numpy()) and np.all(bk.isnotnan(scores).numpy()))


class AttentionModelTest(unittest.TestCase):

  def test_attention(self):
    with bk.framework_('tf'):
      query = bk.variable(np.random.rand(n, Tq, dim).astype('float32'),
                          trainable=True)
      key = bk.variable(np.random.rand(n, Tv, dim).astype('float32'),
                        trainable=True)
      value = bk.variable(np.random.rand(n, Tv, dim).astype('float32'),
                          trainable=True)
      q_mask = np.random.randint(0, 2, size=(n, Tq)).astype('int32')
      v_mask = np.random.randint(0, 2, size=(n, Tv)).astype('int32')

      proj_1 = bk.nn.Dense(1)
      proj_D = bk.nn.Dense(dim)
      proj_V = bk.nn.Dense(1)
      scale = [1. / np.sqrt(dim)] * dim

      num_heads = 2
      q_heads = create_attention_heads(input_dim=query.shape[-1],
                                       num_heads=num_heads,
                                       depth=2)
      k_heads = create_attention_heads(input_dim=key.shape[-1],
                                       num_heads=num_heads,
                                       depth=2)
      v_heads = create_attention_heads(input_dim=value.shape[-1],
                                       num_heads=num_heads,
                                       depth=2)

      for heads in [[q_heads, k_heads, v_heads], [None, None, None]]:
        for input_method in (Inter, Intra):
          print()
          for position in (PosLocalM, PosLocalP, PosGlobal):
            for align_method in (AlignRelax, AlignHard, AlignSoft):
              for score_method in (ScoreLocation, ScoreAdditive, ScoreDotProd,
                                   ScoreCosine, ScoreGeneral):
                am = align_method | score_method | input_method | position
                am.validate()
                print(am)
                try:
                  q, k, v, qm, vm = am.prepare(query, key, value,
                                               (q_mask, v_mask))
                  q, k, v = [
                      i if i is None or j is None else j(i)
                      for i, j in zip([q, k, v], heads)
                  ]
                  with bk.GradientTape() as tape:
                    scores = am.score(
                        q,
                        k,
                        scale=scale,
                        window_width=None,
                        q_proj=proj_1 if ScoreLocation in am else proj_D,
                        target_proj=proj_V)
                    P = am.normalize(scores)
                    out, dist = am.align(scores,
                                         q if v is None else v,
                                         query=q,
                                         v_mask=vm,
                                         q_mask=qm,
                                         causal=True,
                                         residual=True,
                                         dropout=0.3,
                                         training=True,
                                         sample_shape=2)
                    grads = bk.grad(out, [query, key, value], tape=tape)
                  # for name, x, g in zip(["Query", "Key", "Value"], [q, k, v],
                  #                       grads):
                  #   print(" %s" % name)
                  #   print("  -", None if x is None else x.shape)
                  #   print("  -", None if g is None else
                  #         (g.shape, bk.norm(g).numpy()))
                  # print(" Output:", out.shape)
                  # print(" Attention Scores:", scores.shape)
                  # print(" Attention Dist  :",
                  #       dist if isinstance(dist, bay.Distribution) else dist.shape)
                except NotImplementedError as e:
                  print("no support!", e)

  def test_attention_models(self):
    with bk.framework_('tf'):
      query = bk.variable(np.random.rand(n, Tq, dim).astype('float32'),
                          trainable=True)
      key = bk.variable(np.random.rand(n, Tv, dim).astype('float32'),
                        trainable=True)
      value = bk.variable(np.random.rand(n, Tv, dim).astype('float32'),
                          trainable=True)
      q_mask = np.random.randint(0, 2, size=(n, Tq)).astype('int32')
      v_mask = np.random.randint(0, 2, size=(n, Tv)).astype('int32')

    all_kw = []
    for causal in (True, False):
      for residual in (True, False):
        for dropout in (0.0, 0.3):
          for temporal_dropout in (True, False):
            for heads in [
                dict(num_heads=0,
                     heads_depth=1,
                     heads_bias=True,
                     heads_regularization=0.5,
                     heads_activation='linear'),
                dict(num_heads=5,
                     heads_depth=2,
                     heads_bias=True,
                     heads_regularization=0.5,
                     heads_activation='linear')
            ]:
              for scales in [
                  dict(scale_initializer='vaswani',
                       scale_tied=True,
                       scale_trainable=False),
                  dict(
                      scale_initializer='ones',
                      scale_tied=False,
                      scale_trainable=True,
                  )
              ]:
                for hards in [
                    dict(
                        sample_shape=1,
                        temperature=0.5,
                        temperature_trainable=False,
                    ),
                    dict(
                        sample_shape=5,
                        temperature=1.0,
                        temperature_trainable=True,
                    )
                ]:
                  kw = dict(causal=causal,
                            residual=residual,
                            dropout=dropout,
                            temporal_dropout=temporal_dropout)
                  kw.update(heads)
                  kw.update(scales)
                  kw.update(hards)
                  all_kw.append(kw)

    for kw in tqdm(all_kw):
      att = net.SelfAttention(dim, **kw)
      y, a = att(query, mask=(q_mask, v_mask), return_attention=True)
      att.set_methods(alignment=net.attention_mechanism.AlignHard)
      y, a = att(query, mask=(q_mask, v_mask), return_attention=True)
      att.set_methods(alignment=net.attention_mechanism.AlignRelax)
      y, a = att(query, mask=(q_mask, v_mask), return_attention=True)

      att = net.LocalPredictiveAttention(dim, **kw)
      y, a = att([query, value], mask=(q_mask, v_mask), return_attention=True)
      att.set_methods(alignment=net.attention_mechanism.AlignHard)
      y, a = att([query, value], mask=(q_mask, v_mask), return_attention=True)
      att.set_methods(alignment=net.attention_mechanism.AlignRelax)
      y, a = att([query, value], mask=(q_mask, v_mask), return_attention=True)

      att = net.GlobalAttention(dim, **kw)
      y, a = att([query, value], mask=(q_mask, v_mask), return_attention=True)
      att.set_methods(alignment=net.attention_mechanism.AlignHard)
      y, a = att([query, value], mask=(q_mask, v_mask), return_attention=True)
      att.set_methods(alignment=net.attention_mechanism.AlignRelax)
      y, a = att([query, value], mask=(q_mask, v_mask), return_attention=True)


if __name__ == '__main__':
  unittest.main()
