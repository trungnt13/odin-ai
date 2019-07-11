from __future__ import print_function, division, absolute_import

from numbers import Number
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

# ===========================================================================
# Objectives
# ===========================================================================
def kl_divergence(q, p,
                  use_analytic_kl=False,
                  q_sample=lambda q: q.sample(),
                  reduce_axis=(),
                  auto_remove_independent=True,
                  name=None):
  """ Calculating KL(q(x)||p(x))

  Parameters
  ----------
  q : the first distribution
  p : the second distribution

  use_analytic_kl : bool (default: False)
    if True, use the close-form solution  for

  q_sample : {callable, Tensor, Number}
    callable for extracting sample from `q(x)` (takes q distribution
    as input argument)

  reudce_axis : {None, int, tuple}
    reduce axis when use MCMC to estimate KL divergence, default
    `()` mean keep all original dimensions

  auto_remove_independent : bool (default: True)
    if `q` or `p` is `tfd.Independent` wrapper, get the original
    distribution for calculating the analytic KL

  name : {None, str}

  Returns
  -------
  """
  if auto_remove_independent:
    if isinstance(q, tfd.Independent):
      q = q.distribution
    if isinstance(p, tfd.Independent):
      p = p.distribution

  q_name = [i for i in q.name.split('/') if len(i) > 0][-1]
  p_name = [i for i in p.name.split('/') if len(i) > 0][-1]
  with tf.compat.v1.name_scope(name, "KL_q%s_p%s" % (q_name, p_name)):
    if bool(use_analytic_kl):
      return tfd.kl_divergence(q, p)
    else:
      if callable(q_sample):
        z = q_sample(q)
      elif isinstance(q_sample, Number):
        z = q.sample(int(q_sample))
      else:
        z = q_sample
      # calculate the output, then perform reduction
      kl = q.log_prob(z) - p.log_prob(z)
      kl = tf.reduce_mean(input_tensor=kl, axis=reduce_axis)
      return kl
