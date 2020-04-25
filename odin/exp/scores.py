from __future__ import absolute_import, division, print_function

import os
import sqlite3


class Scores():

  def __init__(self, path):
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(path):
      raise ValueError("path to %s must be path to a file." % path)

  def commit(self):
    pass
