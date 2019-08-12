from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow_probability.python.bijectors import exp as exp_bijector
from tensorflow_probability.python.distributions import (
    NegativeBinomial, Normal, QuantizedDistribution, TransformedDistribution,
    Uniform)
from tensorflow_probability.python.internal import dtype_util

__all__ = ["qUniform", "qNormal"]


class qNormal(QuantizedDistribution):

  def __init__(self,
               loc=0.,
               scale=1.,
               min_value=None,
               max_value=None,
               validate_args=False,
               allow_nan_stats=True,
               name="qNormal"):
    super(qNormal,
          self).__init__(distribution=Normal(loc=loc,
                                             scale=scale,
                                             validate_args=validate_args,
                                             allow_nan_stats=allow_nan_stats),
                         low=min_value,
                         high=max_value,
                         name=name)


class qUniform(QuantizedDistribution):

  def __init__(self,
               low=0.,
               high=1.,
               min_value=None,
               max_value=None,
               validate_args=False,
               allow_nan_stats=True,
               name="qUniform"):
    super(qUniform,
          self).__init__(distribution=Uniform(low=low,
                                              high=high,
                                              validate_args=validate_args,
                                              allow_nan_stats=allow_nan_stats),
                         low=min_value,
                         high=max_value,
                         name=name)
