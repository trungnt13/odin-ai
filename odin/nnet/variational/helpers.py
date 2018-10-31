from __future__ import print_function, division, absolute_import

import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from odin import backend as K
from odin.utils import string_normalize, ctext

from .distribution_description import get_distribution_description

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
