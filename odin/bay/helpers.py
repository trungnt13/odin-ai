from __future__ import print_function, division, absolute_import

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from odin import backend as K
from odin.utils import string_normalize, ctext

# ===========================================================================
# Objectives
# ===========================================================================
def kl_divergence(q, p,
                  use_analytic_kl=False,
                  q_sample=lambda q: q.sample(),
                  reduce_axis=(),
                  name=None):
  """ Calculating KL(q(x)||p(x))

  Parameters
  ----------
  q : the first distribution
  p : the second distribution

  use_analytic_kl : boolean
    if True, use the close-form solution  for

  q_sample : {callable, Tensor}
    callable for extracting sample from `q(x)` (takes q distribution
    as input argument)

  reudce_axis : {None, int, tuple}
    reduce axis when use MCMC to estimate KL divergence, default
    `()` mean keep all original dimensions

  """
  q_name = [i for i in q.name.split('/') if len(i) > 0][-1]
  p_name = [i for i in p.name.split('/') if len(i) > 0][-1]
  with tf.compat.v1.name_scope(name, "KL_q%s_p%s" % (q_name, p_name)):
    if bool(use_analytic_kl):
      return tfd.kl_divergence(q, p)
    else:
      if callable(q_sample):
        z = q_sample(q)
      else:
        z = q_sample
      # calculate the output, then perform reduction
      kl = q.log_prob(z) - p.log_prob(z)
      kl = tf.reduce_mean(input_tensor=kl, axis=reduce_axis)
      return kl
