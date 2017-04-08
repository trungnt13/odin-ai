from __future__ import print_function, division, absolute_import

import types
import inspect

import numpy as np

from odin import backend as K
from odin.utils import as_tuple, is_number
from odin.utils.decorators import functionable


from .base import NNOps


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
    In general, helper is the operator that take in a list of NNOps
    and make an unique output from them.

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
    def variables(self):
        all_variables = []
        for i in as_tuple(self.ops):
            if hasattr(i, 'variables'):
                all_variables += i.variables
        return list(set(all_variables))


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
        # ====== iteratively appply all ops ====== #
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
            if K.is_variable(i) and K.is_training():
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
    strict_transpose: bool
        if True, only operators with transposed implemented are added
        to tranpose operator
    debug: bool
        if True, print Ops name and its output shape after applying
        each operator in the sequence.
    all_layers: bool
        if True, return the output from all layers instead of only the last
        layer.

    Note
    ----
    You can specify kwargs for a specific NNOps using `params` keywords,
    for example: `params={1: {'noise': 1}}` will applying noise=1 to the
    1st NNOps, or `params={(1, 2, 3): {'noise': 1}}` to specify keywords
    for the 1st, 2nd, 3rd Ops.

    Example
    -------
    """

    def __init__(self, ops, all_layers=False,
                 strict_transpose=False, debug=False, **kwargs):
        super(Sequence, self).__init__(ops, **kwargs)
        self.all_layers = all_layers
        self.strict_transpose = bool(strict_transpose)
        self.debug = debug

    def _apply(self, x, **kwargs):
        # ====== get specific Ops kwargs ====== #
        params = {}
        for k, v in kwargs.get('params', {}).iteritems():
            # check valid keywords
            if isinstance(v, (tuple, list)):
                try: v = dict(v)
                except Exception: pass
            if not isinstance(v, dict): continue
            # check valid keywords
            for i in as_tuple(k):
                params[self.ops[i] if is_number(i) else i] = v
        # ====== print debug ====== #
        if self.debug:
            print('**************** Sequences: %s ****************' % self.name)
            print('Is training:', bool(K.is_training()))
            print('First input:', K.get_shape(x))
        # ====== applying ====== #
        all_outputs = []
        for op in self.ops:
            keywords = _shrink_kwargs(op, kwargs)
            if op in params:
                keywords.update(params[op])
            x = op(x, **keywords)
            all_outputs.append(x)
            # print after finnish the op
            if self.debug:
                print(' ', op.name if isinstance(op, functionable) else str(op), '->',
                    [K.get_shape(i) for i in x]
                    if isinstance(x, (tuple, list)) else K.get_shape(x))
        # end debug
        if self.debug:
            print()
        return all_outputs if self.all_layers else x

    def _transpose(self):
        transpose_ops = []
        for i in self.ops:
            if hasattr(i, 'T'):
                transpose_ops.append(i.T)
            elif not self.strict_transpose:
                transpose_ops.append(i)
        # reversed the order of ops for transpose
        transpose_ops = list(reversed(transpose_ops))
        seq = Sequence(transpose_ops, debug=self.debug,
                       name=self.name + '_transpose')
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
