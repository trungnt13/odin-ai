from __future__ import print_function, division, absolute_import

import numpy as np

from six import add_metaclass
from abc import ABCMeta, abstractmethod, abstractproperty

from .base import NNOps


class Distribution(NNOps):
    """ Class for distribution within NN architectures """

    def __init__(self, **kwargs):
        super(Distribution, self).__init__(**kwargs)


class Normal(object):
    """ Normal """

    def __init__(self, mean, inv_std, **kwargs):
        super(Normal, self).__init__(**kwargs)
