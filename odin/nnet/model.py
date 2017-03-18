from __future__ import print_function, division, absolute_import

import os
import sys
import inspect
import functools
from types import FunctionType

import numpy as np

from odin import backend as K
from odin.config import get_floatX
from odin.utils import (is_lambda, is_number, get_module_from_path, is_string,
                        as_tuple)


# ===========================================================================
# Helper
# ===========================================================================
def _check_shape(s):
    if is_number(s) or s is None:
        s = (s,)
    if isinstance(s, np.ndarray):
        s = s.tolist()
    if isinstance(s, (tuple, list)):
        if all(is_number(i) or i is None for i in s):
            return True
    return False


def _shape_compare(shape1, shape2):
    """Return True if shape1 == shape2"""
    if len(shape1) != len(shape2):
        return False
    for s1, s2 in zip(shape1, shape2):
        if s1 != s2:
            return False
    return True


# ===========================================================================
# Input descriptor
# ===========================================================================
class InputDescriptor(object):
    """ InputDescriptor
    Store all the necessary information to create placeholder as input
    to any ComputationalGraph.

    Parameters
    ----------
    shape: tuple, list, TensorVariable
        if TensorVariable is given, shape and dtype will be taken from
        given variable
    dtype: dtype
        dtype of input variable
    name: str, None
        specific name for the variable

    Note
    ----
    This object is pickle-able and comparable
    """

    def __init__(self, shape, dtype=None, name=None):
        super(InputDescriptor, self).__init__()
        if K.is_variable(shape):
            if dtype is None: dtype = K.get_dtype(shape, string=True)
            shape = K.get_shape(shape)
        # input the InputDescriptor directly
        elif isinstance(shape, InputDescriptor):
            dtype = shape.dtype if dtype is None else dtype
            name = shape.name if name is None else name
            shape = shape.shape
        # ====== check shape ====== #
        _check_shape(shape)
        if isinstance(shape, np.ndarray):
            shape = shape.tolist()
        self._shape = tuple(shape)
        # ====== check dtype ====== #
        if dtype is None:
            dtype = get_floatX()
        if isinstance(dtype, np.dtype):
            dtype = str(dtype)
        elif is_string(dtype):
            pass
        else:
            dtype = K.get_dtype(dtype, string=True)
        self._dtype = str(dtype)
        # ====== check name ====== #
        self._name = name if name is None else str(name)
        # ====== placeholder ====== #
        self.__placeholder = None

    # ==================== pickle ==================== #
    def __getstate__(self):
        return [self._shape, self._dtype, self._name]

    def __setstate__(self, states):
        self._shape, self._dtype, self._name = states
        self.__placeholder = None

    # ==================== properties ==================== #
    @property
    def placeholder(self):
        if self.__placeholder is None:
            self.__placeholder = K.placeholder(
                shape=self._shape, dtype=self._dtype, name=self._name)
        return self.__placeholder

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    # ==================== override ==================== #
    def __str__(self):
        return "<InputDescriptor - name:%s shape:%s dtype:%s init:%s>" % \
        (str(self._name), str(self._shape), str(self._dtype),
         False if self.__placeholder is None else True)

    def __repr__(self):
        return self.__str__()

    def __cmp__(self, other):
        # ====== compare to a TensorVariable ====== #
        if K.is_variable(other):
            other = InputDescriptor(
                shape=K.get_shape(other), dtype=K.get_dtype(other, string=True))
        # ====== compare to a InputDesriptor ====== #
        if isinstance(other, InputDescriptor):
            if _shape_compare(self._shape, other._shape) \
            and self._dtype == other._dtype:
                return 0
        # ====== compare to a shape tuple (ignore the dtype) ====== #
        elif isinstance(other, (tuple, list)):
            return 0 if _shape_compare(self.shape, other) else 1
        return 1


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

    Example
    -------
    >>> import numpy as np
    >>> from odin import nnet
    >>> @nnet.ModelDescriptor
    >>> def feedforward_vae(X, X1, f):
    ...     if f is None:
    ...         f = N.Sequence([
    ...             N.Dense(num_units=10, activation=K.softmax),
    ...             N.Dropout(level=0.5)
    ...         ])
    ...     # f is return for automatically saved
    ...     return f(X), f
    >>> # First time initialize the input description
    >>> K.set_training(True)
    >>> y_train = feedforward_vae([N.InputDescriptor(shape=(8, 8)),
    ...                            N.InputDescriptor(shape=(12, 12))])
    >>> K.set_training(False); y_score = feedforward_vae()
    >>> # Overide default Placeholder
    >>> X = K.placeholder(shape=(12, 12), name='X')
    >>> K.set_training(True); y_train = feedforward_vae([None, X])
    >>> # performing inference
    >>> feedforward_vae.f_pred(np.random.rand(8, 8), np.random.rand(12, 12))
    """

    def __init__(self, func):
        super(ModelDescriptor, self).__init__()
        if not isinstance(func, FunctionType) or is_lambda(func):
            raise ValueError("This decorator can be only used with function, not "
                             "method or lambda function.")
        self._func = func
        self.input_desc = None
        self._save_states = None
        self._save_kwargs = {}
        # ====== cached tensor variables ====== #
        self._last_outputs = {'train': None, 'score': None}
        self._f_train = None
        self._f_pred = None

    # ==================== pickle ==================== #
    def __getstate__(self):
        from odin.utils.decorators import functionable
        return [functionable(self._func), self.input_desc,
                self._save_states, self._save_kwargs]

    def __setstate__(self, states):
        (self._func, self.input_desc,
            self._save_states, self._save_kwargs) = states
        self._func = self._func.function
        self._last_outputs = {'train': None, 'score': None}
        self._f_train = None
        self._f_pred = None

    def _check_init_shape(self):
        if self.input_desc is None:
            raise ValueError("You must set 'inputs' when calling the ModelDescriptor "
                             ", the inputs can be TensorVariables, shape tuple, "
                             "or InputDescriptor.")

    def check_data(self, X, learn_factor=12.):
        """
        Parameters
        ----------
        learn_factor: float
            your assumption about how many data points a parameter can learn.
        """
        if isinstance(X, (tuple, list)):
            X = X[0]
        if not hasattr(X, 'shape'):
            raise ValueError("The input data must have attribute shape, so we can "
                             "calculate the number of features and samples, but "
                             "given data has type: %s" % str(type(X)))
        shape = X.shape
        if not is_number(shape[0]):
            nb_points = sum(np.prod(s) for s in shape)
            shape = shape[0]
        else:
            nb_points = np.prod(shape)
        nb_samples = shape[0]
        nb_params = self.nb_parameters
        # ====== hard constraint ====== #
        if nb_points / learn_factor < nb_params:
            raise RuntimeError("The number of parameters is: %d, which is "
                               "significant greater than the number of data points "
                               "(only %d data points). It is not recommended to "
                               "train a deep network for this datasets." %
                               (nb_params, nb_points // learn_factor))
        # ====== soft constraint ====== #

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
    def variables(self):
        v = []
        if self._save_states is not None:
            states = self._save_states
            if not isinstance(states, (tuple, list)):
                states = (states,)
            for s in states:
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
            if K.is_trainable_variable(p):
                n += np.prod(K.get_shape(p)).astype('int32')
        return n

    @property
    def placeholder(self):
        self._check_init_shape()
        X = [i.placeholder for i in self.input_desc]
        return X[0] if len(X) == 1 else X

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
                if K.is_variable(i): # TensorVariable
                    shape = K.get_shape(i)
                    input_desc.append(
                        InputDescriptor(shape=shape, dtype=i.dtype, name=i.name))
                elif isinstance(i, (tuple, list)): # Shape tuple
                    shape = tuple(i)
                    input_desc.append(
                        InputDescriptor(shape=shape, dtype='float32', name=None))
                elif isinstance(i, InputDescriptor): # Input Descriptor
                    input_desc.append(i)
                elif i is None: # just a empty place
                    input_desc.append(None)
                else:
                    raise ValueError("input can be TensorVariable, shape tuple, or "
                                     "odin.nnet.InputDescriptor, but the given "
                                     "argument has type: " + str(type(i)))
            # check if match previous inputs
            if self.input_desc is not None:
                for desc1, desc2 in zip(input_desc, self.input_desc):
                    if desc1 is None: continue
                    if desc1 != desc2:
                        raise ValueError('This ModelDescriptor requires input: %s '
                                         ', but the given description is: %s' %
                                         (desc2, desc1))
            # First time specify the input description, None is not eaccepted
            elif any(i is None for i in input_desc):
                raise ValueError("For the first time setting the input description, "
                                 "None value is not accepted.")
            # finally assign the input description
            else:
                self.input_desc = []
                for i, j in enumerate(input_desc):
                    if j.name is None:
                        j._name = '%s%.2d' % (self.name, i)
                    self.input_desc.append(j)
        # ====== get inputs variable====== #
        model_inputs = list(as_tuple(self.placeholder))
        # override default inputs with new variable
        if inputs is not None:
            for i, j in enumerate(inputs):
                if K.is_variable(j):
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
        outputs = self._func(*model_inputs, **kwargs)
        # ====== check outputs values ====== #
        if outputs is None or len(outputs) != 2:
            raise ValueError("[ModelDescriptor] function must return:  "
                             "output and a pickle-able object to save the model.")
        if outputs[1] is not None:
            self._save_states = outputs[1]
        # cached last outputs
        outputs = outputs[0]
        if K.is_training():
            self._last_outputs['train'] = outputs
            self._f_train = None # reset train function
        else:
            self._last_outputs['score'] = outputs
            self._f_pred = None # reset prediciton function
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
