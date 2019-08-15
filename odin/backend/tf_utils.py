from __future__ import absolute_import, division, print_function

from contextlib import contextmanager

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging


@contextmanager
def suppress_logging(level=logging.ERROR):
  curr_log = logging.get_verbosity()
  logging.set_verbosity(level)
  yield logging
  logging.set_verbosity(curr_log)
