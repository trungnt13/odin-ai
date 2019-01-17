from odin.nnet.models.base import Model

import numpy as np
import tensorflow as tf


class InceptionV3(Model):
  """ One way to perform transfer learning is to remove the
  final classification layer of the network and extract the
  next-to-last layer of the CNN, in this case a 2048
  dimensional vector. """

  def __init__(self, offset_top=0, **kwargs):
    super(InceptionV3, self).__init__(**kwargs)

  def get_input_info(self):
    pass
