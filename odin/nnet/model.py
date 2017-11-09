from __future__ import print_function, division, absolute_import

import os
import sys
import inspect
import functools
from types import FunctionType
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from odin import backend as K
from odin.utils.decorators import functionable
from odin.utils import (is_lambda, get_module_from_path,
                        is_primitives, ctext)

from .base import (nnop_scope, get_all_nnops, VariableDesc, NNOp)


# ===========================================================================
# Model descriptor
# ===========================================================================
class Model(object):
    """ Model
    This class allow you to define extremely complex computational graph
    by lumping many nnet operators into once function, but still keeping
    it simple (i.e. just like calling a function).

    The Model will automatically save all states of the function,
    keeps track its relevant inputs, and performing inference is
    straightforward also.

    In short, this descriptor not only store the model itself, but also
    store the way how the model is created.

    Usage
    -----
    >>> @Model
    >>> def model_creator_function(X1, X2, ..., y1, y2, ..., saved_states, **kwargs):
    ...     if save_states is None:
    ...         # create your network here
    ...     else:
    ...         # load saved_states
    ...     return [output1, output2, ...], saved_states
    """

    def __init__(self, func):
        super(Model, self).__init__()
        if not isinstance(func, FunctionType) or is_lambda(func):
            raise ValueError("This decorator can be only used with function, not "
                             "method or lambda function.")
        self._func = func
        self._input_desc = {}
        # mapping from input kwargs -> outputs
        self._last_outputs = {}

    @property
    def input_shape(self):
        return {i: j.shape for i, j in self._input_desc.items()
                if isinstance(j, VariableDesc)}

    # ==================== pickle ==================== #
    def __getstate__(self):
        return [functionable(self._func), self._input_desc, self.nnops]

    def __setstate__(self, states):
        self._func, self._input_desc, nnops = states
        self._func = self._func.function
        self._last_outputs = {}

    # ==================== properties ==================== #
    @property
    def function(self):
        return self._func

    @property
    def name(self):
        return self._func.__name__

    @property
    def variables(self):
        allvars = K.get_all_variables(scope=self.name)
        for o in self.nnops:
            allvars += o.variables
        return list(set(allvars))

    @property
    def parameters(self):
        return [v for v in self.variables
                if K.role.has_roles(v, K.role.Parameter)]

    @property
    def nb_parameters(self):
        n = 0
        for p in self.parameters:
            n += np.prod(p.get_shape().as_list()).astype('int32')
        return n

    @property
    def placeholders(self):
        """Return ordered Placeholders"""
        args = inspect.getargspec(self._func).args
        plh = OrderedDict()
        for i in args:
            j = self._input_desc[i]
            if isinstance(j, VariableDesc):
                plh[i] = j.placeholder
        for i, j in self._input_desc.items():
            if i not in plh and isinstance(j, VariableDesc):
                plh[i] = j.placeholder
        return plh

    @property
    def nnops(self):
        return get_all_nnops(model_scope=self.name)

    # ==================== decorator ==================== #
    def __call__(self, *args, **kwargs):
        spec = inspect.getargspec(self._func)
        # preprocess all arguments
        args = [_check_accepted_inputs(j, self.name + '/' + i)
                for i, j in zip(spec.args, args)]
        kwargs = {i: _check_accepted_inputs(j, self.name + '/' + i)
                  for i, j in kwargs.items()}
        # ====== check inputs ====== #
        if len(self._input_desc) == 0:
            # load to the save_kwargs
            if spec.defaults is not None:
                for key, val in zip(spec.args[::-1], spec.defaults[::-1]):
                    self._input_desc[key] = val
            input_desc = self._input_desc
        else:
            # copy to keep original input_desc
            input_desc = dict(self._input_desc)
        # update recent arguments
        for key, val in zip(spec.args, args):
            input_desc[key] = val
        input_desc.update(kwargs)
        # ====== get inputs variable====== #
        model_inputs = OrderedDict() # map: argument name -> placeholder
        model_data = OrderedDict() # map: placeholder -> numpy array
        placeholders = list(self.placeholders.items())
        for idx, (k, v) in enumerate(input_desc.items()):
            if isinstance(v, VariableDesc):
                v = v.placeholder
                v_dat = None
            elif K.is_placeholder(v):
                v_dat = None
            elif isinstance(v, np.ndarray):
                v_dat = v
                v = placeholders[idx][1]
            model_inputs[k] = v
            # data for placeholder
            if v_dat is not None:
                model_data[v] = v_dat
        if len(model_data) > 0 and len(model_data) != len(model_inputs):
            raise RuntimeError("This model requires %d inputs with shapes: %s, "
                               "but only given %d numpy array with shapes: %s" %
                               (len(model_inputs),
                                '; '.join([v.get_shape() for v in model_inputs.values()]),
                                len(model_data),
                                '; '.join([v.shape for v in model_data])))
        # ====== call the function ====== #
        # finally call the function to get outputs
        with nnop_scope(self.name, id_start=0):
            outputs = self._func(**model_inputs)
        # ====== check outputs values ====== #
        self._last_outputs = outputs
        self._f_outputs = None # reset last function
        # ====== feed data if available ====== #
        if len(model_data) > 0:
            return K.eval(outputs, feed_dict=model_data)
        return outputs

    def __getattr__(self, name):
        # merge the attributes of function to the descriptor
        try:
            return super(Model, self).__getattr__(name)
        except AttributeError:
            return getattr(self._func, name)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        s = "<%s, name: %s, init: %s>\n" % (
            ctext('Model', 'cyan'),
            ctext(self.name, 'MAGENTA'),
            len(self._input_desc) > 0)
        s += '\t%s: %s\n' % (ctext('Core function', 'yellow'),
                           str(self._func))
        s += '\t%s: %s\n' % (ctext('#Parameters', 'yellow'),
                           self.nb_parameters)
        s += '\t%s: %s\n' % (ctext('Size(MB)', 'yellow'),
                           self.nb_parameters * 4 / 1024 / 1024)
        # ====== print input desc info ====== #
        s += '\t%s:\n' % ctext('Input description', 'yellow')
        for i, j in self._input_desc.items():
            s += '\t\t%s: %s\n' % (ctext(str(i), 'red'), str(j))
        # ====== print nnop info ====== #
        s += '\t%s:\n' % ctext('NNOp info', 'yellow')
        for o in self.nnops:
            s += '\n'.join(['\t\t' + i for i in str(o).split('\n')]) + '\n'
        return s

    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)


def get_model_descriptor(name, path = None, prefix = 'model'):
    """ A function is called a model creator when it satisfying 2 conditions:
    * It takes arguments include: input_shape, output_shape
    * It return:
      - an input variables
      - an output variable for training
      - an output variable for inference
      - and an pickle-able object for saving the model.

    Note
    ----
    This method acts just like a shortcut to reduce the redundant works.
    """
    # ====== check path ====== #
    possible_path = ['.', './models', './model', './.models', './.model']
    script_path = os.path.dirname(sys.argv[0])
    if path is None:
        path = [os.path.join(script_path, p) for p in possible_path]
        path = [p for p in path if os.path.exists(p) and os.path.isdir(p)]
    elif not isinstance(path, (tuple, list)):
        path = [path]
    if len(path) == 0:
        raise ValueError("Cannot find any available directory that contain the "
                         "model script.")
    # ====== search for model ====== #
    for p in path:
        model_func = get_module_from_path(name, path = p, prefix = prefix)
        model_func = [f for f in model_func if isinstance(f, Model)]
    if len(model_func) == 0:
        raise ValueError("Cannot find any model creator function with name=%s "
                         "at paths=%s." % (name, ', '.join(path)))
    return model_func[0]
