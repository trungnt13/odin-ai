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
]


class MarginalizableJointDistributionNamed(JointDistributionNamed,
                                           _Marginalizable):

  ...


class MarginalizableJointDistributionSequential(JointDistributionSequential,
                                                _Marginalizable):

  ...
