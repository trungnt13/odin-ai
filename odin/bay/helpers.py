from __future__ import print_function, division, absolute_import

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from odin import backend as K
from odin.utils import string_normalize, ctext

from odin.bay.distribution_description import get_distribution_description

# ===========================================================================
# Main
# ===========================================================================
def parse_distribution(dist_name,
                       X=None, out_dim=None,
                       support=None, n_eventdim=0,
                       name=None, print_log=True, **kwargs):
  """
  X : Tensor
    output from decoder (not included the output layer)
  dist_name : str
    name of the distribution
  out_dim : int
    number of output dimension

  Return
  ------
  """
  dist_desc = get_distribution_description(dist_name)
  with tf.variable_scope(name, default_name="TrainableDistribution"):
    y = dist_desc.set_print_log(print_log)(X, out_dim, n_eventdim,
                                           **kwargs)
  return y

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
    reduce axis when use MCMC to estimate KL divergence

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
      return tf.reduce_mean(
          input_tensor=q.log_prob(z) - p.log_prob(z),
          axis=reduce_axis)
