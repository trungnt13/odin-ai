from __future__ import print_function, division, absolute_import

import types
import inspect
from itertools import chain

import numpy as np

from odin import backend as K
from odin.basic import has_roles, PARAMETER
from odin.utils import as_tuple
from odin.utils.decorators import functionable


from .base import NNOps, NNConfig


def _shrink_kwargs(op, kwargs):
    """ Return a subset of kwargs that given op can accept """
    if hasattr(op, '_apply'): #NNOps
        op = op._apply
    elif isinstance(op, functionable): # functionable
        op = op.function
    elif not isinstance(op, types.FunctionType): # callable object
        op = op.__call__
    spec = inspect.getargspec(op)
    keywords = {i: j for i, j in kwargs.iteritems()
                if spec.keywords is not None or i in spec.args}
    return keywords


class HelperOps(NNOps):
    """ HelperOps
    Parameters
    ----------
    ops: NNOps or callable
        list or single NNOps, or callable

    """

    def __init__(self, ops, **kwargs):
        super(HelperOps, self).__init__(**kwargs)
        self.ops = [functionable(i)
                    if isinstance(i, types.FunctionType) else i
                    for i in as_tuple(ops) if callable(i)]

    @property
    def parameters(self):
        all_parameters = list(
            chain(*[i.parameters for i in as_tuple(self.ops)
                    if hasattr(i, 'parameters')])
        )
        return [i for i in all_parameters if has_roles(i, PARAMETER)]

    def _initialize(self, *args, **kwargs):
        return NNConfig()


class Merge(HelperOps):
    """
    Parameters
    ----------
    ops: list of NNOps
        list of inputs operator, we expect one input for each NNOps,
        however, if only one 1 input is given, we apply all NNOps on the
        same input.
    merge_function: callable
        function that convert a list of variables into 1 variable
    """

    def __init__(self, ops, merge_function=None, **kwargs):
        super(Merge, self).__init__(ops, **kwargs)
        self.merge_function = merge_function

    def _apply(self, X, **kwargs):
        X = as_tuple(X, N=len(self.ops))
        results = [op(x, **_shrink_kwargs(op, kwargs))
                   for x, op in zip(X, self.ops)]
        if callable(self.merge_function):
            output = self.merge_function(results)
            for i in as_tuple(output):
                if not isinstance(K.get_shape(i), tuple):
                    raise Exception('returned output from merge_function lost shape '
                                    'information.')
            return output
        else:
            return results


class Switcher(NNOps):
    """ Simple Ops, perform specific Ops while training and other one for
    deploying
    """

    def __init__(self, training, deploying, **kwargs):
        super(Switcher, self).__init__([], **kwargs)
        if not callable(training) or not callable(deploying):
            raise ValueError('training and deploying must be callable')
        self.training = training
        self.deploying = deploying

    def _apply(self, x, **kwargs):
        is_training = False
        for i in as_tuple(x):
            if K.is_variable(i) and K.is_training(i):
                is_training = True
        if is_training:
            self.ops = [self.training]
            return self.training(x, **_shrink_kwargs(self.training, kwargs))
        else:
            self.ops = [self.deploying]
            return self.deploying(x, **_shrink_kwargs(self.deploying, kwargs))

    def _transpose(self):
        if hasattr(self.training, 'T') and hasattr(self.deploying, 'T'):
            return Switcher(self.training.T, self.deploying.T,
                            name=self.name + '_transpose')
        raise Exception('One of training or deploying ops do not support transpose.')


class Sequence(HelperOps):

    """ Sequence of Operators

    Parameters
    ----------
    strict_transpose : bool
        if True, only operators with transposed implemented are added
        to tranpose operator

    Example
    -------

    """

    def __init__(self, ops, strict_transpose=False, **kwargs):
        super(Sequence, self).__init__(ops, **kwargs)
        self.strict_transpose = bool(strict_transpose)

    def _apply(self, x, **kwargs):
        for op in self.ops:
            x = op(x, **_shrink_kwargs(op, kwargs))
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
