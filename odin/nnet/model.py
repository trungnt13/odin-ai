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

from .base import name_scope, _check_dtype, VariableDescriptor, InputDescriptor


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
        self.input_desc = InputDescriptor()
        self._save_states = None
        self._save_kwargs = {}
        self._opID = [0] # store as reference value
        # ====== cached tensor variables ====== #
        self._last_outputs = None
        self._f_outputs = None

    @property
    def input_shape(self):
        return self.input_desc.shape

    @property
    def input_shape_ref(self):
        return self.input_desc.shape_ref

    # ==================== pickle ==================== #
    def __getstate__(self):
        return [functionable(self._func), self.input_desc,
                self._save_states, self._save_kwargs, self._opID]

    def __setstate__(self, states):
        (self._func, self.input_desc,
            self._save_states, self._save_kwargs, self._opID) = states
        self._func = self._func.function

        self._last_outputs = None
        self._f_outputs = None

    def _check_init_shape(self):
        if len(self.input_desc) == 0:
            raise ValueError("You must set 'inputs' when calling the ModelDescriptor "
                             ", the inputs can be TensorVariables, shape tuple, "
                             "or InputDescriptor.")

    # ==================== properties ==================== #
    @property
    def function(self):
        return self._func

    @property
    def kwargs(self):
        """ Return the most recent kwargs used for the function """
        return self._save_states

    @property
    def name(self):
        return self._func.__name__

    @property
    def opID(self):
        return self._opID[0]

    @property
    def variables(self):
        v = []
        if self._save_states is not None:
            for s in as_tuple(self._save_states):
                if hasattr(s, 'variables'):
                    v += s.variables
        return v

    @property
    def parameters(self):
        params = []
        if self._save_states is not None:
            states = self._save_states
            if not isinstance(states, (tuple, list)):
                states = (states,)
            for s in states:
                if hasattr(s, 'parameters'):
                    params += s.parameters
        return params

    @property
    def nb_parameters(self):
        n = 0
        for p in self.parameters:
            if K.is_variable(p):
                n += np.prod(p.get_shape().as_list()).astype('int32')
        return n

    @property
    def input(self):
        self._check_init_shape()
        return self.input_desc.placeholder

    @property
    def y_train(self):
        # auto-create outputs
        if self._last_outputs['train'] is None and len(self.input_desc) > 0:
            K.set_training(True); self()
        return self._last_outputs['train']

    @property
    def y_score(self):
        # auto-create outputs
        if self._last_outputs['score'] is None and len(self.input_desc) > 0:
            K.set_training(False); self()
        return self._last_outputs['score']

    @property
    def f_train(self):
        """ Note: This only return the train output, no updates is performed """
        if self._f_train is None:
            if self._last_outputs['train'] is None and len(self.input_desc) == 0:
                raise ValueError("No cache value of outputs with training mode ENABLED "
                                 "found, you must call this Descriptor with "
                                 "InputDescriptor first.")
            outputs = self.y_train
            # get number of actual inputs need for prediction
            self._f_train = K.function(K.ComputationGraph(outputs).inputs, outputs)
        return self._f_train

    @property
    def f_pred(self):
        if self._f_pred is None:
            if self._last_outputs['score'] is None and len(self.input_desc) == 0:
                raise ValueError("No cache value of outputs with training mode DISABLE "
                                 "found, you must call this Descriptor with "
                                 "InputDescriptor first.")
            outputs = self.y_score
            # get number of actual inputs need for prediction
            self._f_pred = K.function(K.ComputationGraph(outputs).inputs, outputs)
        return self._f_pred

    @property
    def save_states(self):
        return self._save_states

    # ==================== decorator ==================== #
    def __call__(self, inputs=None, **kwargs):
        # ====== check inputs ====== #
        if inputs is not None:
            if is_number(inputs):
                inputs = (inputs,)
            if not isinstance(inputs, (tuple, list)) or \
            any(is_number(i) for i in inputs):
                inputs = [inputs]
            # get the input shape
            input_desc = []
            for i in inputs:
                if K.is_tensor(i): # TensorVariable
                    shape = i.get_shape().as_list()
                    dtype = _check_dtype(i.dtype.base_dtype)
                    input_desc.append(
                        VariableDescriptor(shape=shape, dtype=dtype, name=i.name))
                elif isinstance(i, (tuple, list)): # Shape tuple
                    shape = tuple(i)
                    input_desc.append(
                        VariableDescriptor(shape=shape, dtype='float32', name=None))
                elif isinstance(i, VariableDescriptor): # VariableDescriptor
                    input_desc.append(i)
                elif i is None: # just a empty place
                    input_desc.append(None)
                else:
                    raise ValueError("input can be TensorVariable, shape tuple, or "
                                     "odin.nnet.VariableDescriptor, but the given "
                                     "argument has type: " + str(type(i)))
            # check if match previous inputs
            if len(self.input_desc) > 0:
                other = InputDescriptor(input_desc)
                if self.input_desc != other:
                    raise ValueError('This ModelDescriptor requires input: %s '
                                     ', but the given description is: %s' %
                                     (str(self.input_desc), str(other)))
            # First time specify the input description, None is not eaccepted
            elif any(i is None for i in input_desc):
                raise ValueError("For the first time setting the input description, "
                                 "None value is not accepted.")
            # finally assign the first input description
            else:
                for i, j in enumerate(input_desc):
                    name = j.name if j.name is not None else ''
                    name = ''.join(name.split(':')[:-1])
                    j._name = '%s_%s%.2d' % (self.name, name, i)
                    self.input_desc.add_variables(j)
        # ====== get inputs variable====== #
        model_inputs = list(as_tuple(self.input))
        # override default inputs with new variable
        if inputs is not None:
            for i, j in enumerate(inputs):
                if K.is_tensor(j):
                    model_inputs[i] = j
        # ====== call the function ====== #
        argspecs = inspect.getargspec(self._func)
        nb_inputs = len(self.input_desc)
        if len(argspecs.args) != nb_inputs + 1:
            raise ValueError("This Descriptor requires a function with %d input "
                             "arguments (%d for inputs variables, and 1 for "
                             "saved_states), but the given function has %d arguments, "
                             "which are: %s" % (nb_inputs + 1, nb_inputs,
                             len(argspecs.args), str(argspecs.args)))
        if argspecs.keywords is None:
            kwargs = {}
        elif len(kwargs) > 0: # override saved kwargs
            self._save_kwargs = kwargs
        else: # get the saved kwargs
            kwargs = self._save_kwargs
        model_inputs.append(self._save_states)
        # finally call the function to get outputs
        with name_scope(self.name, id_start=self._opID):
            outputs = self._func(*model_inputs, **kwargs)
        # ====== check outputs values ====== #
        if outputs is None or len(outputs) != 2:
            raise ValueError("[ModelDescriptor] function must return only 2 objects: "
                             "an output, and a pickle-able object to save the model.")
        if outputs[1] is not None:
            self._save_states = outputs[1]
        # cached last outputs
        outputs = outputs[0]
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
