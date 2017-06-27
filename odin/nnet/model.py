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
from odin.utils import (is_lambda, is_number, get_module_from_path, as_tuple)

from .base import (name_scope, _check_dtype, get_all_nnops,
                   VariableDescriptor, InputDescriptor,)


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
        self._input_desc = InputDescriptor()
        self._save_kwargs = {}
        self._opID = [0] # store as reference value
        # ====== cached tensor variables ====== #
        self._last_outputs = None
        self._f_outputs = None

    @property
    def input_shape(self):
        return self._input_desc.shape

    @property
    def input_shape_ref(self):
        return self._input_desc.shape_ref

    # ==================== pickle ==================== #
    def __getstate__(self):
        return [functionable(self._func), self._input_desc,
                self._save_kwargs, self._opID, self.nnops]

    def __setstate__(self, states):
        (self._func, self._input_desc, self._save_kwargs,
            self._opID, nnops) = states
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
        return self._input_desc.placeholders

    @property
    def last_outputs(self):
        return self._last_outputs

    @property
    def f_outputs(self):
        if self._f_outputs is None:
            if self._last_outputs is None and len(self._input_desc) == 0:
                raise ValueError("No cache value of outputs with training mode ENABLED "
                                 "found, you must call this Descriptor with "
                                 "InputDescriptor first.")
            outputs = self._last_outputs
            # get number of actual inputs need for prediction
            self._f_outputs = K.function(
                K.ComputationGraph(outputs).placeholders, outputs)
        return self._f_outputs

    @property
    def nnops(self):
        return get_all_nnops(model_scope=self.name)

    # ==================== decorator ==================== #
    def __call__(self, inputs=None, **kwargs):
        # ====== check inputs ====== #
        if inputs is not None:
            # number
            if is_number(inputs):
                inputs = (inputs,)
            # shape tuple
            if not isinstance(inputs, (tuple, list)) or \
            any(is_number(i) for i in inputs):
                inputs = [inputs]
            # get the input shape
            input_desc = []
            for i in inputs:
                # TensorVariable, Shape-tuple, VariableDescriptor
                if K.is_tensor(i) or isinstance(i, (tuple, list)) or \
                isinstance(i, VariableDescriptor):
                    input_desc.append(VariableDescriptor(i))
                elif i is None: # just a empty place
                    input_desc.append(None)
                else:
                    raise ValueError("input can be TensorVariable, shape tuple, or "
                                     "odin.nnet.VariableDescriptor, but the given "
                                     "argument has type: " + str(type(i)))
            # check if match previous inputs
            if len(self._input_desc) > 0:
                for v1, v2 in zip(self._input_desc, input_desc):
                    if v2 is not None and v1 != v2:
                        raise ValueError('This ModelDescriptor requires input: %s '
                                         ', but the given description is: %s' %
                                         (str(v1), str(v2)))
            # First time specify the input description, None is not eaccepted
            elif any(i is None for i in input_desc):
                raise ValueError("For the first time setting the input description, "
                                 "None value is not accepted.")
            # finally assign the first input description
            else:
                self._input_desc.set_variables(input_desc)
                for i, j in enumerate(self._input_desc._desc):
                    j._name = '%s_inp%.2d' % (self.name, i)
        else:
            inputs = as_tuple(self.placeholders)
        # ====== get inputs variable====== #
        model_inputs = list(as_tuple(self.placeholders))
        # override default inputs with new variable
        for i, j in enumerate(inputs):
            if K.is_tensor(j):
                model_inputs[i] = j
        print(model_inputs)
        # ====== call the function ====== #
        argspecs = inspect.getargspec(self._func)
        nb_inputs = len(self._input_desc)
        if len(argspecs.args) != nb_inputs:
            raise ValueError("This Descriptor requires a function with %d input "
                             "arguments, but the given function has %d arguments, "
                             "which are: %s" % (nb_inputs + 1, nb_inputs,
                             len(argspecs.args), str(argspecs.args)))
        if argspecs.keywords is None:
            kwargs = {}
        elif len(kwargs) > 0: # override saved kwargs
            self._save_kwargs.update(kwargs)
        else: # get the saved kwargs
            kwargs = self._save_kwargs
        # finally call the function to get outputs
        _ = [0]
        with name_scope(self.name, id_start=_):
            outputs = self._func(*model_inputs, **kwargs)
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
