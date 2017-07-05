from __future__ import print_function, division, absolute_import

import os
import sys
import inspect
import functools
from types import FunctionType

import numpy as np
import tensorflow as tf

from odin import backend as K
from odin.utils.decorators import functionable
from odin.utils import (is_lambda, is_number, get_module_from_path, as_tuple,
                        is_primitives)

from .base import (name_scope, get_all_nnops, VariableDescriptor)


# ===========================================================================
# Helpers
# ===========================================================================
def _check_accepted_inputs(x, backup_name):
    if K.is_tensor(x):
        return VariableDescriptor(shape=x, name=x.name.split(':')[0])
    elif isinstance(x, VariableDescriptor):
        if x._name is None:
            x._name = backup_name
        return x
    elif is_primitives(x):
        return x
    else:
        raise ValueError("The input argument for ModelDescriptor can be: "
            "`Tensor`, `odin.nnet.VariableDescriptor`, and primitive types"
            " (string, number, boolean, None, numpy.ndarray, numpy.generic)")


# ===========================================================================
# Model descriptor
# ===========================================================================
class ModelDescriptor(object):
    """ ModelDescriptor
    This class allow you to define extremely complex computational graph
    by lumping many nnet operators into once function, but still keeping
    it simple (i.e. just like calling a function).

    The ModelDescriptor will automatically save all states of the function,
    keeps track its relevant inputs, and performing inference is
    straightforward also.

    In short, this descriptor not only store the model itself, but also
    store the way how the model is created.

    Usage
    -----
    >>> @ModelDescriptor
    >>> def model_creator_function(X1, X2, ..., y1, y2, ..., saved_states, **kwargs):
    ...     if save_states is None:
    ...         # create your network here
    ...     else:
    ...         # load saved_states
    ...     return [output1, output2, ...], saved_states

    Example
    -------
    >>> import numpy as np
    >>> from odin import nnet
    ...
    >>> @nnet.ModelDescriptor
    >>> def feedforward_vae(X, X1, f, **kwargs):
    ...     check = kwargs['check']
    ...     if f is None:
    ...         f = N.Sequence([
    ...             N.Dense(num_units=10, activation=K.softmax),
    ...             N.Dropout(level=0.5)
    ...         ])
    ...     # f is return for automatically saved
    ...     return f(X), f
    ... # First time initialize the input description
    ...
    >>> K.set_training(True)
    >>> y_train = feedforward_vae(inputs=[N.VariableDescriptor(shape=(8, 8)),
    ...                                   N.VariableDescriptor(shape=(12, 12))],
    ...                           check=True)
    ...
    >>> K.set_training(False); y_score = feedforward_vae()
    ... # Overide default Placeholder
    >>> X = K.placeholder(shape=(12, 12), name='X')
    >>> K.set_training(True); y_train = feedforward_vae([None, X])
    ... # performing inference
    >>> feedforward_vae.f_pred(np.random.rand(8, 8), np.random.rand(12, 12))
    """

    def __init__(self, func):
        super(ModelDescriptor, self).__init__()
        if not isinstance(func, FunctionType) or is_lambda(func):
            raise ValueError("This decorator can be only used with function, not "
                             "method or lambda function.")
        self._func = func
        self._input_desc = {}
        self._opID = [0] # store as reference value
        # ====== cached tensor variables ====== #
        self._last_outputs = None
        self._f_outputs = None

    @property
    def input_shape(self):
        return {i: j.shape for i, j in self._input_desc.iteritems()
                if isinstance(j, VariableDescriptor)}

    @property
    def input_shape_ref(self):
        return {i: j.shape_ref for i, j in self._input_desc.iteritems()
                if isinstance(j, VariableDescriptor)}

    # ==================== pickle ==================== #
    def __getstate__(self):
        return [functionable(self._func), self._input_desc,
                self._opID, self.nnops]

    def __setstate__(self, states):
        (self._func, self._input_desc, self._opID, nnops) = states
        self._func = self._func.function

        self._last_outputs = None
        self._f_outputs = None

    # ==================== properties ==================== #
    @property
    def function(self):
        return self._func

    @property
    def name(self):
        return self._func.__name__

    @property
    def opID(self):
        """ Return the number of Op have been created in this model
        (start from 0)
        """
        return self._opID[0]

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
        return {i: j.placeholder for i, j in self._input_desc.iteritems()
                if isinstance(j, VariableDescriptor)}

    @property
    def last_outputs(self):
        return self._last_outputs

    @property
    def f_outputs(self):
        if self._f_outputs is None:
            if self._last_outputs is None and len(self._input_desc) == 0:
                raise ValueError("No cache value of outputs found, you must "
                    "call this ModelDescriptor with inputs descriptor first.")
            outputs = self._last_outputs
            # get number of actual inputs need for prediction
            self._f_outputs = K.function(
                K.ComputationGraph(outputs).placeholders, outputs)
        return self._f_outputs

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
                  for i, j in kwargs.iteritems()}
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
        model_inputs = {i: j.placeholder if isinstance(j, VariableDescriptor)
                        else j
                        for i, j in input_desc.iteritems()}
        # ====== call the function ====== #
        # finally call the function to get outputs
        _ = [0]
        with name_scope(self.name, id_start=_):
            outputs = self._func(**model_inputs)
        if _[0] > self._opID[0]:
            self._opID[0] = _[0]
        # ====== check outputs values ====== #
        self._last_outputs = outputs
        self._f_outputs = None # reset last function
        return outputs

    def __getattr__(self, name):
        # merge the attributes of function to the descriptor
        try:
            return super(ModelDescriptor, self).__getattr__(name)
        except AttributeError:
            return getattr(self._func, name)

    def __repr__(self):
        return self._func.__repr__()

    def __str__(self):
        return self._func.__str__()

    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)


def get_model_descriptor(name, path=None, prefix='model'):
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
        model_func = get_module_from_path(name, path=p, prefix=prefix)
        model_func = [f for f in model_func if isinstance(f, ModelDescriptor)]
    if len(model_func) == 0:
        raise ValueError("Cannot find any model creator function with name=%s "
                         "at paths=%s." % (name, ', '.join(path)))
    return model_func[0]
