from __future__ import print_function, division, absolute_import

import inspect
from six import string_types, add_metaclass
from collections import defaultdict

import numpy as np


# ===========================================================================
# Async file IO
# ===========================================================================
class keydefaultdict(defaultdict):
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
