from __future__ import division, absolute_import

from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

from odin.autoconfig import get_rng, CONFIG
from odin.backend import tensor as K

FLOATX = CONFIG.floatX
EPSILON = CONFIG.epsilon
PI = np.pi
C = -0.5 * np.log(2 * PI)
_RNG = RandomStreams(seed=get_rng().randint(10e8))

# ===========================================================================
# Variational OPERATIONS
# ===========================================================================
def log_prob_bernoulli(p_true, p_approx, mask=None):
  """ Compute log probability of some binary variables with probabilities
  given by p_true, for probability estimates given by p_approx. We'll
  compute joint log probabilities over row-wise groups.
  Note
  ----
  origin implementation from:
  https://github.com/Philip-Bachman/ICML-2015/blob/master/LogPDFs.py
  Copyright (c) Philip Bachman
  """
  if mask is None:
    mask = T.ones((1, p_approx.shape[1]))
  log_prob_1 = p_true * T.log(p_approx)
  log_prob_0 = (1.0 - p_true) * T.log(1.0 - p_approx)
  log_prob_01 = log_prob_1 + log_prob_0
  row_log_probs = T.sum((log_prob_01 * mask), axis=1, keepdims=True)
  return row_log_probs

#logpxz = -0.5*np.log(2 * np.pi) - log_sigma_decoder - (0.5 * ((x - mu_decoder) / T.exp(log_sigma_decoder))**2)


def log_prob_gaussian(mu_true, mu_approx, les_sigmas=1.0, mask=None):
  """
  Compute log probability of some continuous variables with values given
  by mu_true, w.r.t. gaussian distributions with means given by mu_approx
  and standard deviations given by les_sigmas.
  Note
  ----
  origin implementation from:
  https://github.com/Philip-Bachman/ICML-2015/blob/master/LogPDFs.py
  Copyright (c) Philip Bachman
  """
  if mask is None:
    mask = T.ones((1, mu_approx.shape[1]))
  ind_log_probs = C - T.log(T.abs_(les_sigmas)) - \
  ((mu_true - mu_approx)**2.0 / (2.0 * les_sigmas**2.0))
  row_log_probs = T.sum((ind_log_probs * mask), axis=1, keepdims=True)
  return row_log_probs


def log_prob_gaussian2(mu_true, mu_approx, log_vars=1.0, mask=None):
  """
  Compute log probability of some continuous variables with values given
  by mu_true, w.r.t. gaussian distributions with means given by mu_approx
  and log variances given by les_logvars.
  Note
  ----
  origin implementation from:
  https://github.com/Philip-Bachman/ICML-2015/blob/master/LogPDFs.py
  Copyright (c) Philip Bachman
  """
  if mask is None:
    mask = T.ones((1, mu_approx.shape[1]))
  ind_log_probs = C - (0.5 * log_vars) - \
  ((mu_true - mu_approx)**2.0 / (2.0 * T.exp(log_vars)))
  row_log_probs = T.sum((ind_log_probs * mask), axis=1, keepdims=True)
  return row_log_probs
