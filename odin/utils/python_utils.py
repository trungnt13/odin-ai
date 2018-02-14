from __future__ import print_function, division, absolute_import

import os
import inspect

from datetime import datetime
from collections import defaultdict
from six import string_types, add_metaclass

import numpy as np


def get_formatted_datetime(only_number=True):
  if only_number:
    return "{:%H%M%S%d%m%y}".format(datetime.now())
  return "{:%H:%M:%S-%b%d%y}".format(datetime.now())

def get_all_properties(obj):
  """ Return all attributes which are properties of given Object
  """
  properties = []
  clazz = obj if isinstance(obj, type) else obj.__class__
  for key in dir(clazz):
    if '__' in key:
      continue
    val = getattr(clazz, key)
    if isinstance(val, property):
      properties.append(key)
  return properties if isinstance(obj, type) else \
  {p: getattr(obj, p) for p in properties}

# ===========================================================================
# Async file IO
# ===========================================================================
class defaultdictkey(defaultdict):
  def __missing__(self, key):
    if self.default_factory is None:
      raise KeyError(key)
    else:
      ret = self[key] = self.default_factory(key)
      return ret


class abstractclassmethod(classmethod):

  __isabstractmethod__ = True

  def __init__(self, method):
    method.__isabstractmethod__ = True
    super(abstractclassmethod, self).__init__(method)

# ===========================================================================
# Path utils
# ===========================================================================
def select_path(*paths):
  all_paths = []
  for p in paths:
    if isinstance(p, (tuple, list)):
      all_paths += p
    elif isinstance(p, string_types):
      all_paths.append(p)
    else:
      raise ValueError("Given `path` has type: '%s', which must be string or "
                       "list of string")
  # ====== return the first found exists path ====== #
  for p in all_paths:
    if os.path.exists(p):
      return p
  raise RuntimeError("Cannot find any exists path from list: %s" %
    '; '.join(all_paths))
