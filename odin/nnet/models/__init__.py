# -*- coding: utf-8 -*-
"""
This module contain all pretrained Model, which include:

- BNF_*: pretrained deep bottleneck features based on
    acoustic features
- VGG_*: collection of different VGG architecture for image
    recogition

Example
-------
  >>> import numpy as np
  >>> import tensorflow as tf
  ...
  >>> from odin import nnet as N
  ...
  >>> dbf = N.models.BNF_1024_MFCC39()
  >>> print(dbf.get_input_info())
  ... # {'X': ((None, 819), 'float32')}
  ... # name -> (shape, dtype):
  >>> X = np.random.rand(128, 819).astype('float32')
  ...
  >>> with tf.device('/cpu:0'):
  ...     y_cpu = dbf(X)
  ...
  >>> with tf.device('/gpu:0'):
  ...     y_gpu = dbf(X)
  ...
  >>> assert np.all(np.isclose(y_cpu, y_gpu))

"""

from odin.nnet.models.base import Model
from odin.nnet.models.bnf import *
from odin.nnet.models.imagenet import *
