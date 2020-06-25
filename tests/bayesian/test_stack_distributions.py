from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf
from tensorflow_probability.python.distributions import (Bernoulli, Independent,
                                                         NegativeBinomial,
                                                         Normal,
                                                         VectorDeterministic)

from odin.bay.distributions import (NegativeBinomialDisp, ZeroInflated,
                                    concat_distributions)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.random.set_seed(8)
np.random.seed(8)

def test_mvn():
  mu = [1., 2, 3]
  cov = [[0.36, 0.12, 0.06], [0.12, 0.29, -0.13], [0.06, -0.13, 0.26]]
  diag = tf.linalg.diag_part(cov)
  tril = tf.linalg.cholesky(cov)

  d1 = bay.distributions.MultivariateNormalDiag(loc=mu, scale_diag=diag)
  d2 = bay.distributions.MultivariateNormalDiag(loc=mu, scale_diag=diag)

  d1 = bay.distributions.MultivariateNormalTriL(loc=mu, scale_tril=tril)
  d2 = bay.distributions.MultivariateNormalTriL(loc=mu, scale_tril=tril)

  d1 = bay.distributions.MultivariateNormalFullCovariance(loc=mu,
                                                          covariance_matrix=cov)
  d2 = bay.distributions.MultivariateNormalFullCovariance(loc=mu,
                                                          covariance_matrix=cov)
  d = bay.concat_distributions([d1, d2])

def assert_consistent_statistics(d1, d2):
  d = concat_distributions((d1, d2))

  m1 = d1.mean()
  m2 = d2.mean()
  m = d.mean()
  assert np.all(np.isclose(m.numpy(), tf.concat((m1, m2), axis=0).numpy()))

  v1 = d1.variance()
  v2 = d2.variance()
  v = d.variance()
  assert np.all(np.isclose(v.numpy(), tf.concat((v1, v2), axis=0).numpy()))

  # This is because
  # tf.random.set_seed(8)
  # print(tf.random.uniform((3,), seed=1))
  # print(tf.random.uniform((2,), seed=1))
  # # is different from
  # tf.random.set_seed(8)
  # print(tf.random.uniform((5,), seed=1))
  tf.random.set_seed(8)
  s1 = d1.sample()
  s2 = d2.sample()
  tf.random.set_seed(8)
  s = d.sample()
  assert s.shape == tf.concat((s1, s2), axis=0).shape
  assert np.all(np.isclose(s[:s1.shape[0]].numpy(), s1.numpy()))

  try:
    for name in d1.__class__._params_event_ndims().keys():
      p1 = getattr(d1, name)
      p2 = getattr(d2, name)
      p = getattr(d, name)
      assert np.all(np.isclose(p.numpy(), tf.concat((p1, p2), axis=0).numpy()))
  except NotImplementedError:
    pass


shape = (8, 2)
count = np.random.randint(0, 20, size=shape).astype('float32')
probs = np.random.rand(*shape).astype('float32')
logits = np.random.rand(*shape).astype('float32')

assert_consistent_statistics(Bernoulli(probs=probs), Bernoulli(logits=logits))
assert_consistent_statistics(Bernoulli(logits=logits), Bernoulli(logits=logits))
assert_consistent_statistics(
    Independent(Bernoulli(probs=probs), reinterpreted_batch_ndims=1),
    Independent(Bernoulli(logits=logits), reinterpreted_batch_ndims=1))

assert_consistent_statistics(NegativeBinomial(total_count=count, logits=logits),
                             NegativeBinomial(total_count=count, probs=probs))
assert_consistent_statistics(
    Independent(NegativeBinomial(total_count=count, logits=logits),
                reinterpreted_batch_ndims=1),
    Independent(NegativeBinomial(total_count=count, probs=probs),
                reinterpreted_batch_ndims=1))
assert_consistent_statistics(
    ZeroInflated(NegativeBinomial(total_count=count, logits=logits),
                 logits=logits),
    ZeroInflated(NegativeBinomial(total_count=count, probs=probs), probs=probs))
assert_consistent_statistics(
    Independent(ZeroInflated(NegativeBinomial(total_count=count, logits=logits),
                             logits=logits),
                reinterpreted_batch_ndims=1),
    Independent(ZeroInflated(NegativeBinomial(total_count=count, probs=probs),
                             probs=probs),
                reinterpreted_batch_ndims=1))
assert_consistent_statistics(
    ZeroInflated(Independent(NegativeBinomial(total_count=count, logits=logits),
                             reinterpreted_batch_ndims=1),
                 logits=logits),
    ZeroInflated(Independent(NegativeBinomial(total_count=count, probs=probs),
                             reinterpreted_batch_ndims=1),
                 probs=probs))

assert_consistent_statistics(NegativeBinomialDisp(loc=count, disp=count),
                             NegativeBinomialDisp(loc=count, disp=count))
assert_consistent_statistics(
    ZeroInflated(NegativeBinomialDisp(loc=count, disp=count), probs=probs),
    ZeroInflated(NegativeBinomialDisp(loc=count, disp=count), probs=probs))

inflated_dist1 = Bernoulli(logits=logits)
inflated_dist2 = Bernoulli(probs=probs)
assert_consistent_statistics(
    ZeroInflated(NegativeBinomialDisp(loc=count, disp=count),
                 inflated_distribution=inflated_dist1),
    ZeroInflated(NegativeBinomialDisp(loc=count, disp=count),
                 inflated_distribution=inflated_dist2))
