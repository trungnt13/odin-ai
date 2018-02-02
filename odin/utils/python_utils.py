from __future__ import print_function, division, absolute_import

import inspect
from six import string_types, add_metaclass
from collections import defaultdict

import numpy as np


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
