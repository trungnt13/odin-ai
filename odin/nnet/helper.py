from __future__ import print_function, division, absolute_import

import inspect
from itertools import chain

import numpy as np

from odin import backend as K
from odin.roles import has_roles, PARAMETER
from odin.utils.decorators import autoinit


from .base import NNOps, NNConfig


class Fork(NNOps):
    """docstring for Fork"""

    def __init__(self, ops, **kwargs):
        super(Fork, self).__init__(**kwargs)
        self.ops = ops


class Switcher(NNOps):
    """ Simple Ops, perform specific Ops while training and other one for
    deploying
    """

    def __init__(self, training, deploying, **kwargs):
        super(Switcher, self).__init__(**kwargs)
        self.training = training
        self.deploying = deploying

    def _initialize(self, *args, **kwargs):
        return NNConfig()

    def _apply(self, *args, **kwargs):
        is_training = False
        for i in chain(args, kwargs.values()):
            if K.is_variable(i) and K.is_training(i):
                is_training = True
        if is_training:
            return self.training(*args, **kwargs)
        else:
            return self.deploying(*args, **kwargs)

    def _transpose(self):
        if hasattr(self.training, 'T') and hasattr(self.deploying, 'T'):
            return Switcher(self.training.T, self.deploying.T,
                            name=self.name + '_transpose')
        raise Exception('One of training or deploying ops do not support transpose.')


class Sequence(NNOps):

    """ Sequence of Operators

    Parameters
    ----------
    strict_transpose : bool
        if True, only operators with transposed implemented are added
        to tranpose operator

    Example
    -------

    """

    @autoinit
    def __init__(self, ops, strict_transpose=False, **kwargs):
        super(Sequence, self).__init__(**kwargs)
        self.ops = []
        if hasattr(strict_transpose, '__call__'):
            raise Exception('You made a funny mistake, ops must be list.')
        if not isinstance(ops, (tuple, list)):
            ops = [ops]
        for i in ops:
            if hasattr(i, '__call__'):
                self.ops.append(i)

    @property
    def parameters(self):
        all_parameters = list(chain(
            *[i.parameters for i in self.ops if hasattr(i, 'parameters')]))
        return [i for i in all_parameters if has_roles(i, PARAMETER)]

    def _initialize(self, *args, **kwargs):
        return NNConfig()

    def _apply(self, x, **kwargs):
        for op in self.ops:
            spec = inspect.getargspec(op._apply)
            keywords = {i: j for i, j in kwargs.iteritems()
                        if spec.keywords is not None or i in spec.args}
            x = op(x, **keywords)
        return x

    def _transpose(self):
        transpose_ops = []
        for i in self.ops:
            if hasattr(i, 'T'):
                transpose_ops.append(i.T)
            elif not self.strict_transpose:
                transpose_ops.append(i)
        # reversed the order of ops for transpose
        transpose_ops = list(reversed(transpose_ops))
        seq = Sequence(transpose_ops)
        return seq

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.ops.__getitem__(key)
        elif isinstance(key, slice):
            return Sequence(self.ops.__getitem__(key))
        elif isinstance(key, str):
            for i in self.ops:
                if hasattr(i, '_name') and i._name == key:
                    return i
        raise ValueError('key can only be int, slice or str.')

    def __setitem__(self, key, value):
        return self.ops.__setitem__(key, value)

    # ==================== Arithemic operator ==================== #
    def __add__(self, other):
        return Sequence(self.ops + other.ops)

    def __sub__(self, other):
        return Sequence([i for i in self.ops if i not in other.ops])

    def __iadd__(self, other):
        self.ops += other.ops

    def __isub__(self, other):
        self.ops = [i for i in self.ops if i not in other.ops]

    def __and__(self, other):
        return Sequence([i for i in self.ops if i in other.ops])

    def __iand__(self, other):
        self.ops = [i for i in self.ops if i in other.ops]

    def __or__(self, other):
        return self.__add__(other)

    def __ior__(self, other):
        return self.__iadd__(other)
