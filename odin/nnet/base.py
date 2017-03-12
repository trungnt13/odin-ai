from __future__ import division, absolute_import, print_function

import inspect
import numbers
import warnings
from itertools import chain
from collections import OrderedDict
from abc import ABCMeta, abstractmethod
from six import add_metaclass, types, string_types
from six.moves import zip, range, cPickle

import numpy as np

from odin import backend as K
from odin.basic import (add_role, has_roles, PARAMETER, VariableRole,
                        WEIGHT, BIAS, VARIATIONAL_MEAN, VARIATIONAL_LOGSIGMA)
from odin.utils import as_tuple, uuid, cache_memory, is_number

from .model import InputDescriptor


# ===========================================================================
# Helper
# ===========================================================================
def _initialize_param(name, spec, shape):
    """ return a ndarray or trainable_variable """
    #####################################
    # 0. initializing function.
    if callable(spec):
        spec = spec(shape)
    #####################################
    # 1. Shared variable, just check the shape.
    if K.is_trainable_variable(spec):
        spec_shape = K.get_shape(spec)
        if not isinstance(spec_shape, tuple):
            spec_shape = K.eval(spec_shape)
        if shape is None:
            shape = spec_shape
        elif tuple(shape) != tuple(spec_shape):
            raise Exception('Given variable has different shape from requirement'
                            ', %s != %s' % (str(spec_shape), str(shape)))
    #####################################
    # 2. expression, we can only check number of dimension.
    elif K.is_variable(spec):
        # We cannot check the shape here, Theano expressions (even shared
        # variables) do not have a fixed compile-time shape. We can check the
        # dimensionality though.
        # Note that we cannot assign a name here. We could assign to the
        # `name` attribute of the variable, but the user may have already
        # named the variable and we don't want to override this.
        if shape is not None and K.ndim(spec) != len(shape):
            raise Exception("parameter with name=%s has %d dimensions, should be "
                            "%d" % (name, spec.ndim, len(shape)))
    #####################################
    # 3. numpy ndarray, create shared variable wraper for it.
    elif isinstance(spec, np.ndarray):
        if shape is not None and spec.shape != shape:
            raise RuntimeError("parameter with name=%s has shape %s, should be "
                               "%s" % (name, spec.shape, shape))
    #####################################
    # 5. Exception.
    else:
        raise RuntimeError("cannot initialize parameters: 'spec' is not "
                           "a numpy array, a Theano expression, or a "
                           "callable")
    return spec, shape


def _recurrsive_extract_shape(x):
    shape_list = []
    if not isinstance(x, (tuple, list)):
        x = [x]
    for i in x:
        if K.is_variable(i):
            shape = K.get_shape(i)
            if isinstance(shape, (tuple, list)):
                shape_list.append(shape)
        elif isinstance(i, (tuple, list)):
            shape_list += _recurrsive_extract_shape(i)
    return shape_list


class NNConfig(object):

    def __init__(self, nnops):
        super(NNConfig, self).__init__()
        # name -> variables
        if not isinstance(nnops, NNOps):
            raise ValueError("nnops must be instance of odin.nnet.NNOps")
        self._nnops = nnops
        self._input_desc = []
        self._variables = OrderedDict()

    @property
    def variables(self):
        """ Return the list of all TensorVariables attached to this Config"""
        return self._variables.values()

    @property
    def input(self):
        """ Return the list of all TensorVariables attached to this Config"""
        inputs = [i.placeholder for i in self._input_desc]
        return inputs[0] if len(inputs) == 1 else inputs

    @property
    def input_shape(self):
        shape = [i.shape for i in self._input_desc]
        return shape[0] if len(shape) == 1 else shape

    def check_input_desc(self, input_desc):
        input_desc = as_tuple(input_desc)
        # make shape tuple, become list of shape tuple
        if any(is_number(i) or i is None for i in input_desc):
            input_desc = (input_desc,)
        # have not initialized the input description
        if len(self._input_desc) == 0:
            self._input_desc = [i if isinstance(i, InputDescriptor)
                                else InputDescriptor(i) for i in input_desc]
        # mismatch input desctiption
        elif len(input_desc) != len(self._input_desc):
            raise Exception("This Config required %d inputs, but was given "
                            "%d inputs." % (len(self._input_desc), len(input_desc)))
        for i, j in zip(input_desc, self._input_desc):
            if i != j:
                raise Exception("The config require input with %s, but was given "
                                "other input with %s." % (str(j), str(i)))
        # automatic fetch placeholder to replace raw description
        return [i if K.is_tensor(i) else j.placeholder
                for i, j in zip(input_desc, self._input_desc)]

    def __getattr__(self, name):
        if name in self._variables:
            return self._variables[name]
        elif name not in self.__dict__:
            raise AttributeError('Cannot find attribute with name="%s", for NNOps '
                                 'with name="%s"' % (name, self._nnops.name))
        return super(NNConfig, self).__getattr__(name)

    def create_params(self, spec, shape, name, roles=[], nb_params=1):
        """
        Parameters
        ----------
        spec: variable, numpy.ndarray, function
            specification for initializing the weights
        shape: tuple, list
            expected shape for given variable
        name: str
            name for the variable
        nnops: NNOps
            parent operator of this parameters
        roles: odin.basic.VariableRole
            categories of this variable
        nb_params: int
            number of parameters that horizontally stacked into
            given `shape (e.g. nb_params=2, create 2 parameters with
            given `shape and horizontally stack them into 1 parameters)
            * do NOT support when `spec` is variable.
        """
        if not isinstance(roles, (tuple, list)):
            roles = [roles]
        nnops = self._nnops
        shape = tuple(shape)  # convert to tuple if needed
        if any(d <= 0 for d in shape):
            raise ValueError((
                "Cannot create param with a non-positive shape dimension. "
                "Tried to create param with shape=%r, name=%r") %
                (shape, name))

        # ====== create parameters ====== #
        spec = as_tuple(spec, nb_params)
        spec = [_initialize_param(name, s, shape) for s in spec]
        # check shape returned
        shape = list(set([i[-1] for i in spec]))
        if len(shape) > 1:
            raise Exception('shape are inconsitent among all given "spec", the '
                            'created shape is: %s' % str(shape))
        shape = shape[0]
        # check spec returned
        spec = [i[0] for i in spec]
        if isinstance(spec[0], np.ndarray):
            with K.variable_scope(nnops.name):
                spec = np.concatenate(spec, axis=-1)
                shape = spec.shape
                spec = K.variable(spec, name=name)
        elif K.is_trainable_variable(spec[0]):
            if nb_params > 1:
                with K.variable_scope(nnops.name):
                    spec = np.concatenate([K.get_value(i) for i in spec], axis=-1)
                    shape = spec.shape
                    spec = K.variable(spec, name=name)
            else:
                spec = spec[0]
        elif K.is_variable(spec[0]):
            shape = (shape[0] * nb_params,) if len(shape) == 1 \
                else shape[:-1] + (shape[-1] * nb_params,)
            spec = K.concatenate(spec, axis=-1)
        # ====== assign annotations ====== #
        # only add role for trainable variables
        for i in roles:
            if isinstance(i, VariableRole) and K.is_trainable_variable(spec):
                add_role(spec, i)
        # return actual variable or expression
        # override other parameters with same name
        self._variables[name] = spec
        return spec

    def __str__(self):
        s = ""
        for i in self._input_desc:
            s += str(i) + "\n"
        s += ' - Parameters: ' + ', '.join([str(i) for i in self._variables.values()])
        return s

    # ==================== pickling method ==================== #
    def __getstate__(self):
        return self._nnops, self._input_desc, \
        [(name, K.pickling_variable(var)) for name, var in self._variables.iteritems()]

    def __setstate__(self, states):
        self._nnops = states[0]
        self._input_desc = states[1]
        self._variables = OrderedDict([(name, K.pickling_variable(var))
                           for name, var in states[2]])


# ===========================================================================
# Main Ops
# ===========================================================================
@add_metaclass(ABCMeta)
class NNOps(object):
    """ Basics of all Neural Network operators

    Properties
    ----------
    name: str
        identity of the operator, this name is the scope for its operator
        and should be unique.
    T: NNOps
        transpose operator of this one (NOTE: some ops does not support
        transpose and raise NotImplementedError)
    parameters: list of variables
        list of all parameters associated with this operator scope

    Abstract
    --------
    _apply(self, x, **kwargs): resulted variables
        apply take a list of variables and custom parameters to compute
        output variables
    _initialize(self, x, **kwargs): NNConfig
        create and return NNConfig object, which is identity from
        other configuration

    Override
    --------
    _transpose(self): NNOps
        return another NNOps which is transposed version of this ops

    Note
    ----
    All NNOps are pickle-able!
    if NNOps is applied to a list of inputs, it will process each input seperated
    """

    def __init__(self, name=None, **kwargs):
        super(NNOps, self).__init__()
        self._save_states = {}

        self.name = name
        if name is None:
            self.name = "%s_%s" % (self.__class__.__name__, uuid())

        self._configuration = NNConfig(self)
        self._transpose_ops = None
        self._is_initialized = False

    # ==================== properties ==================== #
    @property
    def T(self):
        """ Return new ops which is transpose of this ops """
        if self._transpose_ops is None:
            self._transpose_ops = self._transpose()
            if not isinstance(self._transpose_ops, NNOps):
                raise ValueError("The _transposed method must return NNOps."
                                 "but the returned object has type=%s" %
                                 str(type(self._transpose_ops)))
        return self._transpose_ops

    @property
    def variables(self):
        if not self._is_initialized:
            raise Exception("This operators haven't initialized.")
        return self._configuration.variables

    @property
    def parameters(self):
        """ return all TensorVariables which have the PARAMETER role"""
        return [i for i in self.variables if has_roles(i, PARAMETER)]

    @property
    def trainable_variables(self):
        """ return all TensorVariables which are trainable """
        return [i for i in self.variables
                if K.is_trainable_variable(i)]

    @property
    def config(self):
        return self._configuration

    @property
    def is_initialized(self):
        return self._is_initialized

    @property
    def input(self):
        """ Create list of placeholder to represent inputs of this NNOps
        """
        return self._configuration.input

    @property
    def nb_input(self):
        return len(self._configuration._input_desc)

    @property
    def input_shape(self):
        return self._configuration.input_shape

    def __setattr__(self, name, value):
        # this record all assigned attribute to pickle them later
        # check hasattr to prevent recursive loop at the beginning before
        # __init__ is called
        if hasattr(self, '_save_states') and name != '_save_states':
            # otherwise, only save primitive types
            if isinstance(value, _PRIMITIVE_TYPES):
                self._save_states[name] = value
        return super(NNOps, self).__setattr__(name, value)

    def __getattr__(self, name):
        # merge the attributes of ops wit its configuration
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self._configuration, name)

    # ==================== abstract method ==================== #
    def _initialize(self, **kwargs):
        """ This function is only called once, for the first time you
        apply this Ops
        """
        return None

    @abstractmethod
    def _apply(self, X, **kwargs):
        raise NotImplementedError

    def _transpose(self):
        raise NotImplementedError

    # ==================== interaction method ==================== #
    def apply(self, X, **kwargs):
        # ====== initialize first ====== #
        # only select necessary arguments
        argspec = inspect.getargspec(self._initialize)
        keywords = {}
        # kwargs must be specified in args, or the _initialize
        # must accept **kwaobject, class_or_type_or_tuplergs
        for i, j in kwargs.iteritems():
            if argspec.keywords is not None or i in argspec.args:
                keywords[i] = j
        # initialize the operator (call the initilazation process)
        X = self._configuration.check_input_desc(X)
        if not self._is_initialized:
            self._initialize(**keywords)
            self._is_initialized = True
        # ====== calculate and return outputs ====== #
        return self._apply(X[0] if len(X) == 1 else X, **kwargs)

    def __call__(self, X, **kwargs):
        return self.apply(X, **kwargs)

    def __str__(self):
        ops_format = '<ops: %s, name: %s, init: %s>'
        return ops_format % (self.__class__.__name__, self.name,
                             self._is_initialized)

    # ==================== Slicing ==================== #
    def __getitem__(self, key):
        return NNSliceOps(self, key)

    # ==================== pickling method ==================== #
    def __getstate__(self):
        return self._save_states

    def __setstate__(self, states):
        self._save_states = states
        for i, j in self._save_states.iteritems():
            setattr(self, i, j)


_PRIMITIVE_TYPES = (tuple, list, dict, string_types, type(True),
                    types.FunctionType, numbers.Number, type(None),
                    K.init.constant, NNConfig, NNOps)


# ===========================================================================
# Helper
# ===========================================================================
class NNSliceOps(NNOps):

    def __init__(self, ops, slice):
        if not isinstance(ops, NNOps):
            raise ValueError('ops must be instance of NNOps, but was given argument '
                             'has %s' % str(type(ops)))
        super(NNSliceOps, self).__init__()
        self._ops = ops
        if not isinstance(slice, (tuple, list)):
            slice = [slice]
        self.slice = slice

    @property
    def variables(self):
        return self._ops.variables

    def _apply(self, X, **kwargs):
        y = self._ops.apply(X, **kwargs)
        return_list = True
        if not isinstance(y, (tuple, list)):
            return_list = False
            y = [y]
        # apply slice and calculate the shape
        output = []
        for i in y:
            shape = K.get_shape(i)
            i = i[self.slice]
            # good to calculate new output shape
            if isinstance(shape, (tuple, list)):
                new_shape = []
                for dim, idx in zip(shape, self.slice):
                    if isinstance(idx, numbers.Number):
                        dim = -1
                    elif dim is not None and isinstance(idx, slice):
                        dim = idx.indices(dim)
                        dim = dim[1] - dim[0]
                    # -1 mean delete that dimension because of int index
                    if dim > 0 or dim is None:
                        new_shape.append(dim)
                # slice is not specified for all dimension
                if len(new_shape) < K.ndim(i):
                    new_shape += shape[len(self.slice):]
                # add the new shape
                K.add_shape(i, new_shape)
            output.append(i)
        # return output
        if return_list:
            return output
        return output[0]

    def __str__(self):
        ops_format = '<ops: %s, name: %s, init: %s, slice: %s>'
        return ops_format % (self._ops.__class__.__name__, self._ops.name,
                             self._ops.is_initialized, str(self.slice))


class NNTransposeOps(NNOps):
    """ TransposeOps
    Create a transposed view of the origin NNOps
    """

    def __init__(self, ops):
        super(NNTransposeOps, self).__init__(name=ops.name + '_transpose')
        if not isinstance(ops, NNOps):
            raise ValueError("NNTransposeOps can only be applied for instance of "
                             "odin.nnet.NNOps, but was given type=%s" % str(type(ops)))
        self._transpose_ops = ops

    def _transpose(self):
        # return original Ops to prevent infinite useless loop of transpose
        return self._transpose_ops

    def _initialize(self, **kwargs):
        if not self._transpose_ops.is_initialized:
            raise RuntimeError("The original NNOps with name:%s have not been "
                               "initialized, you must call the original NNOps "
                               "first." % self._ops)

    def __str__(self):
        ops_format = '<original_ops: %s, name: %s, init: %s>'
        return ops_format % (self._transpose_ops.__class__.__name__,
                             self.name, self._transpose_ops.is_initialized and
                             self.is_initialized)


# ===========================================================================
# Simple ops
# ===========================================================================
class Dense(NNOps):

    def __init__(self, num_units,
                 W_init=K.init.glorot_uniform,
                 b_init=K.init.constant(0),
                 activation=K.linear,
                 **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.activation = (K.linear if activation is None else activation)
        self.W_init = W_init
        self.b_init = b_init
        self.num_units = num_units

    # ==================== abstract methods ==================== #
    def _transpose(self):
        # create the new dense
        return TransposeDense(self)

    def _initialize(self):
        input_shape = self.input_shape
        shape = (input_shape[-1], self.num_units)
        self.config.create_params(self.W_init, shape, 'W', roles=WEIGHT)
        if self.b_init is not None:
            self.config.create_params(self.b_init,
                shape=(self.num_units,), name='b', roles=BIAS)

    def _apply(self, X):
        input_shape = K.get_shape(X)
        # calculate projection
        activation = K.dot(X, self.W)
        # add the bias
        if self.b_init is not None:
            activation = activation + self.b
        # set shape for output
        K.add_shape(activation, input_shape[:-1] + (self.num_units,))
        # Nonlinearity might change the shape of activation
        activation = self.activation(activation)
        return activation


class TransposeDense(NNTransposeOps):

    def _initialize(self):
        super(TransposeDense, self)._initialize()
        self.num_units = self.T.input_shape[-1]
        if self.T.b_init is not None:
            self.config.create_params(self.T.b_init,
                shape=(self.num_units,), name='b', roles=BIAS)

    def _apply(self, X):
        input_shape = K.get_shape(X)
        # calculate projection
        activation = K.dot(X, K.transpose(self.T.W))
        if self.T.b_init is not None:
            activation = activation + self.b
        # set shape for output
        K.add_shape(activation, input_shape[:-1] + (self.num_units,))
        # Nonlinearity might change the shape of activation
        activation = self.T.activation(activation)
        return activation


class VariationalDense(NNOps):

    def __init__(self, num_units,
                 W_init=K.init.symmetric_uniform,
                 b_init=K.init.constant(0),
                 activation=K.linear,
                 seed=None, **kwargs):
        super(VariationalDense, self).__init__(**kwargs)
        self.num_units = num_units
        self.W_init = W_init
        self.b_init = b_init
        # hack to prevent infinite useless loop of transpose
        self.activation = K.linear if activation is None else activation
        self.seed = seed

    # ==================== helper ==================== #
    @cache_memory # same x will return the same mean and logsigma
    def get_mean_logsigma(self, x):
        b_mean = 0. if not hasattr(self, 'b_mean') else self.b_mean
        b_logsigma = 0. if not hasattr(self, 'b_logsigma') else self.b_logsigma
        mean = self.activation(K.dot(x, self.W_mean) + b_mean)
        logsigma = self.activation(K.dot(x, self.W_logsigma) + b_logsigma)
        mean.name = 'variational_mean'
        logsigma.name = 'variational_logsigma'
        add_role(mean, VARIATIONAL_MEAN)
        add_role(logsigma, VARIATIONAL_LOGSIGMA)
        return mean, logsigma

    def sampling(self, x):
        mean, logsigma = self.get_mean_logsigma(x)
        epsilon = K.random_normal(shape=K.get_shape(mean), mean=0.0, std=1.0,
                                  dtype=mean.dtype)
        z = mean + K.exp(logsigma) * epsilon
        return z

    # ==================== abstract methods ==================== #
    def _transpose(self):
        raise NotImplementedError

    def _initialize(self):
        shape = (self.input_shape[-1], self.num_units)
        self.config.create_params(self.W_init, shape, 'W_mean', roles=WEIGHT)
        self.config.create_params(self.W_init, shape, 'W_logsigma', roles=WEIGHT)
        if self.b_init is not None:
            self.config.create_params(
                self.b_init, (self.num_units,), 'b_mean', roles=BIAS)
            self.config.create_params(
                self.b_init, (self.num_units,), 'b_logsigma', roles=BIAS)

    def _apply(self, x):
        input_shape = K.get_shape(x)
        # calculate statistics
        mean, logsigma = self.get_mean_logsigma(x)
        # variational output
        output = mean
        if K.is_training():
            output = self.sampling(x)
        # set shape for output
        K.add_shape(output, input_shape[:-1] + (self.num_units,))
        return output


class ParametricRectifier(NNOps):
    """ This class is adpated from Lasagne:
    Original work Copyright (c) 2014-2015 lasagne contributors
    All rights reserved.
    LICENSE: https://github.com/Lasagne/Lasagne/blob/master/LICENSE
    A layer that applies parametric rectify activation to its input
    following [1]_ (http://arxiv.org/abs/1502.01852)
    Equation for the parametric rectifier linear unit:
    :math:`\\varphi(x) = \\max(x,0) + \\alpha \\min(x,0)`
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    alpha : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the alpha values. The
        shape must match the incoming shape, skipping those axes the alpha
        values are shared over (see the example below).
        See :func:`lasagne.utils.create_params` for more information.
    shared_axes : 'auto', 'all', int or tuple of int
        The axes along which the parameters of the rectifier units are
        going to be shared. If ``'auto'`` (the default), share over all axes
        except for the second - this will share the parameter over the
        minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers. If ``'all'``, share over
        all axes, which corresponds to a single scalar parameter.
    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.
     References
    ----------
    .. [1] K He, X Zhang et al. (2015):
       Delving Deep into Rectifiers: Surpassing Human-Level Performance on
       ImageNet Classification,
       http://link.springer.com/chapter/10.1007/3-540-49430-8_2
    Notes
    -----
    The alpha parameter dimensionality is the input dimensionality minus the
    number of axes it is shared over, which matches the same convention as
    the :class:`BiasLayer`.
    >>> layer = ParametricRectifierLayer((20, 3, 28, 28), shared_axes=(0, 3))
    >>> layer.alpha.get_value().shape
    (3, 28)
    """

    def __init__(self, alpha_init=K.init.constant(0.25),
                 shared_axes='auto', **kwargs):
        super(ParametricRectifier, self).__init__(**kwargs)
        self.alpha_init = alpha_init
        self.shared_axes = shared_axes

    # ==================== abstract methods ==================== #
    def _initialize(self):
        if self.shared_axes == 'auto':
            self.shared_axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif self.shared_axes == 'all':
            self.shared_axes = tuple(range(len(self.input_shape)))
        elif isinstance(self.shared_axes, int):
            self.shared_axes = (self.shared_axes,)

        shape = [size for axis, size in enumerate(self.input_shape)
                 if axis not in self.shared_axes]
        if any(size is None for size in shape):
            raise ValueError("ParametricRectifierLayer needs input sizes for "
                             "all axes that alpha's are not shared over.")
        self.alpha = self.config.create_params(
            self.alpha_init, shape, name="alpha", roles=PARAMETER)

    def _apply(self, x):
        axes = iter(range(K.ndim(self.alpha)))
        pattern = ['x' if input_axis in self.shared_axes
                   else next(axes)
                   for input_axis in range(K.ndim(x))]
        alpha = K.dimshuffle(self.alpha, pattern)
        return K.relu(x, alpha)
