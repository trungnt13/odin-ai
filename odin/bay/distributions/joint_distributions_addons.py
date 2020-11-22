import tensorflow as tf
from tensorflow_probability.python.distributions import (
    JointDistributionCoroutine, JointDistributionNamed,
    JointDistributionSequential)
from tensorflow_probability.python.experimental.marginalize import (
    MarginalizableJointDistributionCoroutine, logeinsumexp)
from tensorflow_probability.python.experimental.marginalize.marginalizable import \
    Marginalizable as _Marginalizable

__all__ = [
    'MarginalizableJointDistributionCoroutine',
    'MarginalizableJointDistributionNamed',
    'MarginalizableJointDistributionSequential',
    'logeinsumexp',
    'JointDistributionConcatenation',
]


class MarginalizableJointDistributionNamed(JointDistributionNamed,
                                           _Marginalizable):

  ...


class MarginalizableJointDistributionSequential(JointDistributionSequential,
                                                _Marginalizable):

  ...


class JointDistributionConcatenation(JointDistributionSequential):
  """The `JointDistributionSequential` is parameterized by a `list` comprised of
  either:

  1. `tfp.distributions.Distribution`-like instances or,
  2. `callable`s which return a `tfp.distributions.Distribution`-like instance.

  Each `list` element implements the `i`-th *full conditional distribution*,
  `p(x[i] | x[:i])`. The "conditioned on" elements are represented by the
  `callable`'s required arguments. Directly providing a `Distribution`-like
  instance is a convenience and is semantically identical a zero argument
  `callable`.

  Examples
  ---------

  ```python
  tfd = tfp.distributions

  # Consider the following generative model:
  #     e ~ Exponential(rate=[100,120])
  #     g ~ Gamma(concentration=e[0], rate=e[1])
  #     n ~ Normal(loc=0, scale=2.)
  #     m ~ Normal(loc=n, scale=g)
  #     for i = 1, ..., 12:
  #       x[i] ~ Bernoulli(logits=m)

  # In TFP, we can write this as:
  joint = tfd.JointDistributionSequential([
                   tfd.Independent(tfd.Exponential(rate=[100, 120]), 1),  # e
      lambda    e: tfd.Gamma(concentration=e[..., 0], rate=e[..., 1]),    # g
                   tfd.Normal(loc=0, scale=2.),                           # n
      lambda n, g: tfd.Normal(loc=n, scale=g),                            # m
      lambda    m: tfd.Sample(tfd.Bernoulli(logits=m), 12)                # x
  ])
  # (Notice the 1:1 correspondence between "math" and "code".)

  x = joint.sample()
  # ==> A length-5 list of Tensors representing a draw/realization from each
  #     distribution.
  joint.log_prob(x)
  # ==> A scalar `Tensor` representing the total log prob under all five
  #     distributions.

  joint.resolve_graph()
  # ==> (('e', ()),
  #      ('g', ('e',)),
  #      ('n', ()),
  #      ('m', ('n', 'g')),
  #      ('x', ('m',)))
  ```
  """

  def sample(self, sample_shape=(), seed=None, name='sample', **kwargs):
    x = super().sample(sample_shape=sample_shape,
                       seed=seed,
                       name=name,
                       **kwargs)
    return tf.stack([tf.cast(i, tf.float32) for i in x], axis=-1)

  def log_prob(self, x):
    if not isinstance(x, (tuple, list)):
      x = tf.unstack(x, axis=-1)
    return super().log_prob(x)

  def log_prob_parts(self, x):
    if not isinstance(x, (tuple, list)):
      x = tf.unstack(x, axis=-1)
    return super().log_prob_parts(tf.unstack(x, axis=-1))
