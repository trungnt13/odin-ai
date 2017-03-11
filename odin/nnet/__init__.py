from __future__ import print_function, division, absolute_import
import os
import sys
import functools
from types import FunctionType

import numpy as np

from .base import *
from .conv import *
from .noise import *
from .shape import *
from .sampling import *
from .normalization import *
from .embedding import *
from .helper import *
from .rnn import *
from . import shortcuts

from odin import backend as K
from odin.utils import is_lambda, is_number, get_module_from_path


# ===========================================================================
# Helper
# ===========================================================================
def __decorator_apply(dec, func):
    """Decorate a function by preserving the signature even if dec
    is not a signature-preserving decorator.
    This recipe is derived from
    http://micheles.googlecode.com/hg/decorator/documentation.html#id14
    """
    from decorator import FunctionMaker
    return FunctionMaker.create(
        func, 'return decorated(%(signature)s)',
        dict(decorated=dec(func)), __wrapped__=func)


def __check_shape(s):
    if is_number(s) or s is None:
        s = (s,)
    if isinstance(s, np.ndarray):
        s = s.tolist()
    if isinstance(s, (tuple, list)):
        if all(is_number(i) or i is None for i in s):
            return True
    elif K.is_tensor(s):
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
# Model descriptor
# ===========================================================================
class InputDescriptor(object):

    def __init__(self, shape, dtype='float32', name=None):
        super(InputDescriptor, self).__init__()
        self.shape = tuple(shape)
        self.dtype = dtype
        self.name = name

    def __str__(self):
        return "<name:%s shape:%s dtype:%s>" % \
        (str(self.name), str(self.shape), str(self.dtype))

    def __repr__(self):
        return self.__str__()

    def __cmp__(self, other):
        if isinstance(other, InputDescriptor):
            if _shape_compare(self.shape, other.shape) and\
            str(self.dtype) == str(other.dtype):
                return 0
        elif isinstance(other, (tuple, list)):
            return 0 if _shape_compare(self.shape, other) else 1
        return 1


class ModelDescriptor(object):
    """ ModelDescriptor

    Example
    -------
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
    >>> y_train = f([N.InputDescriptor(shape=(8, 8)),
    ...              N.InputDescriptor(shape=(12, 12))])
    >>> K.set_training(False); y_score = f()
    >>> # Overide default Placeholder
    >>> X = K.placeholder(shape=(12, 12), name='X')
    >>> K.set_training(True); y_train = f([None, X])
    """

    def __init__(self, func):
        super(ModelDescriptor, self).__init__()
        if not isinstance(func, FunctionType) or is_lambda(func):
            raise ValueError("This decorator can be only used with function, not "
                             "method or lambda function.")
        self._func = func
        self.input_desc = None
        self._save_states = None
        # ====== cached tensor variables ====== #
        self._inputs = []
        self._last_inputs = {'train': [], 'score': []}
        self._last_outputs = {'train': None, 'score': None}
        self._f_pred = None

    # ==================== pickle ==================== #
    def __getstate__(self):
        from odin.utils.decorators import functionable
        return [functionable(self._func), self.input_desc, self._save_states]

    def __setstate__(self, states):
        self._func, self.input_desc, self._save_states = states
        self._func = self._func.function
        self._inputs = []
        self._last_inputs = {'train': [], 'score': []}
        self._last_outputs = {'train': None, 'score': None}
        self._f_pred = None

    # ==================== properties ==================== #
    def _check_init_shape(self):
        if self.input_desc is None:
            raise ValueError("You must set 'inputs' when calling the ModelDescriptor "
                             ", the inputs can be TensorVariables, shape tuple, "
                             "or InputDescriptor.")

    @property
    def function(self):
        return self._func

    @property
    def inputs(self):
        self._check_init_shape()
        # ====== automatic create inputs if necessary ====== #
        if len(self._inputs) == 0:
            name = self.__name__
            self._inputs = [K.placeholder(shape=desc.shape, dtype=desc.dtype,
                                          name='%s%s' % (name, str(i) if desc.name is None
                                                         else str(desc.name)))
                            for i, desc in enumerate(self.input_desc)]
        return self._inputs

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
    def f_pred(self):
        if self._f_pred is None:
            if self._last_outputs['score'] is None and len(self.input_desc) == 0:
                raise ValueError("No cache value of outputs with training disabled "
                                 "found, you must call this Descriptor with "
                                 "InputDescriptor first.")
            outputs = self.y_score
            # get number of actual inputs need for prediction
            nb_inputs = len(K.ComputationGraph(outputs).inputs)
            self._f_pred = K.function(self._last_inputs['score'][:nb_inputs],
                                      outputs)
        return self._f_pred

    @property
    def save_states(self):
        return self._save_states

    # ==================== decorator ==================== #
    def __call__(self, inputs=None, **kwargs):
        # ====== check inputs ====== #
        if inputs is not None:
            if not isinstance(inputs, (tuple, list)) or is_number(inputs[0]):
                inputs = [inputs]
            # get the input shape
            input_desc = []
            for i in inputs:
                if K.is_tensor(i): # TensorVariable
                    shape = K.get_shape(i)
                    dtype = K.get_dtype(i, string=True)
                    input_desc.append(
                        InputDescriptor(shape=shape, dtype=dtype, name=i.name))
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
            else:
                self.input_desc = input_desc
        # ====== get inputs variable====== #
        model_inputs = list(self.inputs)
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
            self._last_inputs['train'] = model_inputs[:-1]
        else:
            self._last_outputs['score'] = outputs
            self._last_inputs['score'] = model_inputs[:-1]
            self._f_pred = None # reset prediciton function
        return outputs

    def __getattr__(self, name):
        # merge the attributes of function to the descriptor
        if not hasattr(self, name) and hasattr(self._func, name):
            return getattr(self._func, name)
        return super(ModelDescriptor, self).__getattr__(name)

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
                         "at paths=%s." % (name, str(path)))

    return model_func[0]
