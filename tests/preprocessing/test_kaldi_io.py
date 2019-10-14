from __future__ import absolute_import, division, print_function

import unittest


def _check_pykaldi():
  try:
    import kaldi
    return True
  except ImportError:
    return False


class KaldiIOTest(unittest.TestCase):

  def test_feature_loader(self):
    if not _check_pykaldi():
      return
