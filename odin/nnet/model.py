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
                        as_tuple, ShapeRef, DtypeRef)
from odin.utils.decorators import functionable


# ===========================================================================
# Helper
# ===========================================================================
def _check_shape(s):
    if callable(s): return functionable(s)
    if is_number(s) or s is None:
        s = (s,)
    elif isinstance(s, np.ndarray):
        s = s.tolist()
    return tuple([int(i) if is_number(i) else None for i in s])


def _check_dtype(dtype):
    if callable(dtype): return functionable(dtype)
    # ====== check dtype ====== #
    if dtype is None:
        dtype = get_floatX()
    elif isinstance(dtype, np.dtype) or is_string(dtype):
        dtype = str(dtype)
    elif isinstance(dtype, VariableDescriptor):
        dtype = DtypeRef(dtype)
    else:
        dtype = K.get_dtype(dtype, string=True)
    return dtype


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
class VariableDescriptor(object):
    """ VariableDescriptor
    Store all the necessary information to create placeholder as input
    to any ComputationalGraph.

    Parameters
    ----------
    shape: tuple, list, TensorVariable, callable
        if TensorVariable is given, shape and dtype will be taken from
        given variable. if a callable object is given, the object must
        return shape information when called without any argument.
    dtype: str, numpy.dtype, callable, InputDescriptor
        dtype of input variable
    name: str, None, callable, InputDescriptor
        specific name for the variable

    Note
    ----
    This object is pickle-able and comparable
    """

    def __init__(self, shape, dtype=None, name=None):
        super(VariableDescriptor, self).__init__()
        # ====== placeholder ====== #
        self.__placeholder = None
        self._name = name if name is None else str(name)
        # Given a TensorVariabe, we don't want to pickle TensorVariable,
        # so copy all necessary information
        if K.is_variable(shape):
            if dtype is None:
                self._dtype = K.get_dtype(shape, string=True)
            self._shape = K.get_shape(shape)
        # input the InputDescriptor directly
        elif isinstance(shape, VariableDescriptor):
            self._shape = ShapeRef(shape)
            self._dtype = DtypeRef(shape) if dtype is None else _check_dtype(dtype)
        # input regular information flow
        else:
            self._shape = _check_shape(shape)
            self._dtype = _check_dtype(dtype)
        # ====== create reference ====== #
        # trick to store self in x, hence, no closure
        self._shape_ref = ShapeRef(self)
        self._dtype_ref = DtypeRef(self)

    # ==================== pickle ==================== #
    def __getstate__(self):
        return (self._shape, self._shape_ref,
                self._dtype, self._dtype_ref, self._name)

    def __setstate__(self, states):
        (self._shape, self._shape_ref,
         self._dtype, self._dtype_ref, self._name) = states
        self.__placeholder = None

    # ==================== properties ==================== #
    def set_placeholder(self, plh):
        if not K.is_placeholder(plh):
            raise ValueError("a placholder must be specified.")
        if K.get_shape(plh) == self.shape and \
        K.get_dtype(plh, string=True) == self.dtype:
            self.__placeholder = plh
        else:
            raise ValueError("This VariableDescriptor require input with shape=%s,"
                             "and dtype=%s, but given a placholder with shape=%s, "
                             "dtype=%s." % (str(self.shape), self.dtype,
                                str(K.get_shape(plh)), K.get_dtype(plh, string=True)))
        return self

    @property
    def placeholder(self):
        if self.__placeholder is None:
            self.__placeholder = K.placeholder(
                shape=self.shape, dtype=self.dtype, name=self.name)
        return self.__placeholder

    @property
    def name(self):
        return self._name

    @property
    def shape(self):
        return self._shape() if callable(self._shape) else self._shape

    @property
    def shape_ref(self):
        """ ref is callable reference to the shape information of
        this descriptor, it will return the actual shape if you
        call it. """
        return self._shape_ref

    @property
    def dtype(self):
        return self._dtype() if callable(self._dtype) else self._dtype

    @property
    def dtype_ref(self):
        """ ref is callable reference to the dtype information of
        this descriptor, it will return the actual dtype if you
        call it. """
        return self._dtype_ref

    # ==================== override ==================== #
    def __str__(self):
        return "<VarDesc - name:%s shape:%s dtype:%s init:%s>" % \
        (str(self.name), str(self.shape), str(self.dtype),
         False if self.__placeholder is None else True)

    def __repr__(self):
        return self.__str__()

    def __cmp__(self, other):
        # ====== compare to a TensorVariable ====== #
        if K.is_variable(other):
            other = VariableDescriptor(
                shape=K.get_shape(other), dtype=K.get_dtype(other, string=True))
        # ====== compare to a InputDescriptor ====== #
        if isinstance(other, VariableDescriptor):
            if _shape_compare(self.shape, other.shape) \
            and self.dtype == other.dtype:
                return 0
        # ====== compare to a shape tuple (ignore the dtype) ====== #
        elif isinstance(other, (tuple, list)):
            return 0 if _shape_compare(self.shape, other) else 1
        return 1


class InputDescriptor(object):

    def __init__(self, desc=None):
        super(InputDescriptor, self).__init__()
        self._desc = []
        self.set_variables(desc)
        # ====== create reference ====== #
        # trick to store self in x, hence, no closure
        self._shape_ref = ShapeRef(self)
        self._dtype_ref = DtypeRef(self)

    def _create_var_desc(self, info):
        if isinstance(info, VariableDescriptor):
            return info
        if isinstance(info, dict):
            return VariableDescriptor(**info)
        info = as_tuple(info)
        # shape tuple is given
        if any(is_number(i) or i is None for i in info):
            return VariableDescriptor(info)
        return VariableDescriptor(*info)

    def set_variables(self, desc):
        if isinstance(desc, InputDescriptor):
            self._desc = desc._desc
        elif desc is not None:
            desc = as_tuple(desc)
            # convert shape tuple to list of shape tuple
            if any(is_number(i) or i is None for i in desc):
                desc = (desc,)
            self._desc = [self._create_var_desc(d) for d in desc]
        return self

    def add_variables(self, desc):
        if desc is not None:
            desc = as_tuple(desc)
            # convert shape tuple to list of shape tuple
            if any(is_number(i) or i is None for i in desc):
                desc = (desc,)
            self._desc += [self._create_var_desc(d) for d in desc]
        return self

    # ==================== properties ==================== #
    def set_placeholder(self, plh):
        plh = [i for i in as_tuple(plh) if i is None or K.is_placeholder(i)]
        if len(plh) < len(self._desc):
            plh += [None] * len(self._desc) - len(plh)
        elif len(plh) > len(self._desc):
            plh = plh[:len(self._desc)]
        for v, p in zip(self._desc, plh):
            if p is not None:
                v.set_placeholder(p)
        return self

    @property
    def placeholder(self):
        plh = [i.placeholder for i in self._desc]
        return plh[0] if len(plh) == 1 else plh

    @property
    def name(self):
        return ','.join([i.name for i in self._desc])

    @property
    def shape(self):
        s = [i.shape for i in self._desc]
        return s[0] if len(s) == 1 else s

    @property
    def shape_ref(self):
        """ ref is callable reference to the shape information of
        this descriptor, it will return the actual shape if you
        call it. """
        return self._shape_ref

    @property
    def dtype(self):
        d = [i.dtype for i in self._desc]
        return d[0] if len(d) == 1 else d

    @property
    def dtype_ref(self):
        """ ref is callable reference to the dtype information of
        this descriptor, it will return the actual dtype if you
        call it. """
        return self._dtype_ref

    # ==================== override ==================== #
    def __iter__(self):
        return self._desc.__iter__()

    def __len__(self):
        return len(self._desc)

    def __getitem__(self, key):
        return self._desc.__getitem__(key)

    def __setitem__(self, key, value):
        if not isinstance(value, VariableDescriptor):
            raise ValueError("InputDescriptor setitem only accept VariableDescriptor.")
        return self._desc.__setitem__(key, value)

    def __str__(self):
        return "<InputDescriptor: %s" % '; '.join([str(i) for i in self._desc])

    def __repr__(self):
        return self.__str__()

    def __cmp__(self, other):
        if not isinstance(other, InputDescriptor):
            raise ValueError("Can only compare a InputDescriptor to another "
                             "InputDescriptor.")
        n = 0
        for d1 in self._desc:
            for d2 in other._desc:
                if d1 == d2: n += 1
        if n == len(self._desc):
            return 0
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
        # ====== cached tensor variables ====== #
        self._last_outputs = {'train': None, 'score': None}
        self._f_train = None
        self._f_pred = None

    @property
    def input_shape(self):
        return self.input_desc.shape

    @property
    def input_shape_ref(self):
        return self.input_desc.shape_ref

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
        if len(self.input_desc) == 0:
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
                if K.is_variable(i): # TensorVariable
                    shape = K.get_shape(i)
                    dtype = K.get_dtype(i, string=True)
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
