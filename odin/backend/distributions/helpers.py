from __future__ import print_function, division, absolute_import

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from odin import backend as K
from odin.utils import string_normalize

def parse_distribution(X, dist_name, out_dim,
                       support=None, name=None):
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
  dist_name = string_normalize(dist_name, lower=True,
                               remove_non_alphanumeric=True,
                               remove_whitespace='_')
  out_dim = int(out_dim)
  with tf.variable_scope(name,
                         default_name='ParseDistribution_%s' % dist_name):
    pass
  if not K.is_tensor(X):
    X = tf.convert_to_tensor(X)
  exit()
  out_dim = X.shape.as_list()[-1]
  assert dist_name in _dist_name, \
  "Support distribution: %s; given: '%s'" % (', '.join(_dist_name), dist_name)
  # ====== some log ====== #
  print(ctext("Parsing variable distribution:", 'lightyellow'))
  print("  Variable    :", ctext(X, 'cyan'))
  print("  Decoder     :", ctext(D, 'cyan'))
  print("  name        :", ctext('%s/%s' % (dist_name, name), 'cyan'))
  print("  independent :", ctext(independent, 'cyan'))
  # ******************** create distribution ******************** #
  with tf.variable_scope(name):
    # ====== Bernoulli ====== #
    if dist_name == 'zibernoulli':
      f = N.Dense(num_units=out_dim, activation=K.linear, name="Logit")
      bern = tfd.Bernoulli(logits=f(D))

      f_pi = N.Dense(num_units=out_dim, activation=tf.nn.sigmoid, name="Pi")
      pi = clip_support(f_pi(D), x_min=0, x_max=1)

      out = ZeroInflated(dist=bern, pi=pi)

    # ====== Bernoulli ====== #
    elif dist_name == 'bernoulli':
      f = N.Dense(num_units=out_dim, activation=K.linear, name="Logit")
      out = tfd.Bernoulli(logits=f(D))

    # ====== Normal ====== #
    elif dist_name == 'normal':
      f_loc = N.Dense(num_units=out_dim, activation=K.linear, name="Location")
      loc = f_loc(D)

      f_log_sigma = N.Dense(num_units=out_dim, activation=K.linear, name="LogSigma")
      log_sigma = clip_support(f_log_sigma(D), x_min=-3, x_max=3)

      out = tfd.Normal(loc=loc, scale=tf.exp(log_sigma))

    # ====== Poisson ====== #
    elif dist_name in ('poisson',):
      f_log_rate = N.Dense(num_units=out_dim, activation=K.linear, name="LogRate")
      logit = clip_support(f_log_rate(D), x_min=-10, x_max=10)
      out = tfd.Poisson(rate=tf.exp(logit), name=name)

    # ====== Zero-inflated Poisson ====== #
    elif dist_name in ('zipoisson',):
      f_log_rate = N.Dense(num_units=out_dim, activation=K.linear, name="LogRate")
      logit = clip_support(f_log_rate(D), x_min=-10, x_max=10)
      pois = tfd.Poisson(rate=tf.exp(logit))

      f_pi = N.Dense(num_units=out_dim, activation=tf.nn.sigmoid, name="Pi")
      pi = clip_support(f_pi(D), x_min=0, x_max=1)

      out = ZeroInflated(dist=pois, pi=pi)

    # ====== Negative Binomial ====== #
    elif dist_name in ('nb',):
      f_log_count = N.Dense(num_units=out_dim, activation=K.linear, name="TotalCount")
      log_count = clip_support(f_log_count(D), x_min=-10, x_max=10)

      f_probs = N.Dense(num_units=out_dim, activation=tf.nn.sigmoid, name="Probs")
      probs = clip_support(f_probs(D), x_min=0, x_max=1)

      out = tfd.NegativeBinomial(
          total_count=tf.exp(log_count),
          probs=probs)

    # ====== Zero-inflated Negative Binomial ====== #
    elif dist_name in ('zinb',):
      f_log_count = N.Dense(num_units=out_dim, activation=K.linear, name="TotalCount")
      log_count = clip_support(f_log_count(D), x_min=-10, x_max=10)

      f_probs = N.Dense(num_units=out_dim, activation=tf.nn.sigmoid, name="Probs")
      probs = clip_support(f_probs(D), x_min=0, x_max=1)

      nb = tfd.NegativeBinomial(
          total_count=tf.exp(log_count),
          probs=probs)

      f_pi = N.Dense(num_units=out_dim, activation=tf.nn.sigmoid, name="Pi")
      pi = clip_support(f_pi(D), x_min=0, x_max=1)

      out = ZeroInflated(dist=nb, pi=pi)

    # ====== beta distribution ====== #
    elif dist_name in ('beta',):
      f_log_alpha = N.Dense(num_units=out_dim, activation=tf.identity,
                            name="LogAlpha")
      log_alpha = clip_support(f_log_alpha(D), x_min=-3, x_max=3)

      f_log_beta = N.Dense(num_units=out_dim, activation=tf.identity,
                           name="LogBeta")
      log_beta = clip_support(f_log_beta(D), x_min=-3, x_max=3)

      out = tfd.Beta(concentration1=tf.exp(log_alpha),
                     concentration0=tf.exp(log_beta))

    # ====== exception ====== #
    else:
      raise RuntimeError("Cannot find distribution with name: '%s', all possible "
        "distributions are: %s" % (dist_name, str(_dist_name)))
    # ====== independent ====== #
    if independent:
      out = tfd.Independent(out, reinterpreted_batch_ndims=1)
  # ====== get the Negative log-likelihood ====== #
  X_tile = tf.tile(tf.expand_dims(X, axis=0),
                   multiples=[D.shape.as_list()[0], 1, 1])
  # negative log likelihood. (n_samples, n_batch)
  NLLK = -out.log_prob(X_tile)
  if NLLK.shape.ndims == 3:
    NLLK = tf.reduce_sum(NLLK, axis=-1)
  # ******************** print the dist ******************** #
  print("  dist        :", ctext(out, 'cyan'))
  for name, p in sorted(out.parameters.items()):
    if name in ('allow_nan_stats', 'validate_args'):
      continue
    print("      %-8s:" % name, ctext(p, 'magenta'))
  print("  NLLK        :", ctext(NLLK, 'cyan'))
  return out, NLLK
