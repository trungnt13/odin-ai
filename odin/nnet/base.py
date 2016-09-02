from __future__ import division, absolute_import, print_function

import inspect
import numbers
import types
import cPickle
import warnings
from itertools import chain
from abc import ABCMeta, abstractmethod
from six import add_metaclass
from six.moves import zip, range

import numpy as np

from odin import backend as K
from odin.roles import (add_role, has_roles, PARAMETER, VariableRole,
                        WEIGHT, BIAS,
                        VARIATIONAL_MEAN, VARIATIONAL_LOGSIGMA,
                        BATCH_NORM_SHIFT_PARAMETER,
                        BATCH_NORM_POPULATION_MEAN,
                        BATCH_NORM_SCALE_PARAMETER,
                        BATCH_NORM_POPULATION_STDEV)
from odin.utils import as_tuple
from odin.utils.decorators import autoinit, cache

# ===========================================================================
# Helper
# ===========================================================================
_primitive_types = (tuple, list, dict, types.StringType, types.BooleanType,
                    types.FunctionType, numbers.Number, types.NoneType,
                    K.init.constant)


def _initialize_param(spec, shape):
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
            raise Exception("parameter variable has %d dimensions, should be "
                            "%d" % (spec.ndim, len(shape)))
    #####################################
    # 3. numpy ndarray, create shared variable wraper for it.
    elif isinstance(spec, np.ndarray):
        if shape is not None and spec.shape != shape:
            raise RuntimeError("parameter array has shape %s, should be "
                               "%s" % (spec.shape, shape))
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

    @autoinit
    def __init__(self, **kwargs):
        super(NNConfig, self).__init__()
        self._paramters = []

    @property
    def parameters(self):
        return self._paramters

    def __getattr__(self, name):
        if name in self._arguments:
            return self._arguments[name]
        for i in self._paramters:
            if name == i.name:
                return i
        raise AttributeError('Cannot find attribute={} in arguments and parameters'
                             '.'.format(name))

    def create_params(self, spec, shape, name, nnops, roles=[],
                      nb_params=1):
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
        roles: odin.roles.VariableRole
            categories of this variable
        nb_params: int
            number of parameters that horizontally stacked into
            given `shape (e.g. nb_params=2, create 2 parameters with
            given `shape and horizontally stack them into 1 parameters)
            * do NOT support when `spec` is variable.
        """
        if not isinstance(roles, (tuple, list)):
            roles = [roles]
        if not isinstance(nnops, NNOps):
            raise Exception('nnops must be instance of odin.nnet.base.NNOps')

        shape = tuple(shape)  # convert to tuple if needed
        if any(d <= 0 for d in shape):
            raise ValueError((
                "Cannot create param with a non-positive shape dimension. "
                "Tried to create param with shape=%r, name=%r") %
                (shape, name))

        # ====== create parameters ====== #
        spec = as_tuple(spec, nb_params)
        spec = [_initialize_param(s, shape) for s in spec]
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
        elif K.is_variable(spec[0]) and nb_params > 1:
            shape = (shape[0] * nb_params,) if len(shape) == 1 \
                else shape[:-1] + (shape[-1] * nb_params,)
            spec = K.concatenate(spec, axis=-1)
            spec.name = nnops.name + '/' + name
        # ====== assign annotations ====== #
        for i in roles:
            if isinstance(i, VariableRole):
                add_role(spec, i)
        # return actual variable or expression
        # override other parameters with same name
        for i, j in enumerate(self._paramters):
            if j.name == name:
                self._paramters[i] = spec
        if spec not in self._paramters:
            self._paramters.append(spec)
        return spec

    def inflate(self, obj):
        """ Infate configuration into given object  """
        for i, j in self._arguments.iteritems():
            setattr(obj, i, j)
        for i in self._paramters:
            name = i.name.split('/')[-1].split(':')[0]
            setattr(obj, name, i)

    def reset(self, obj):
        """  """
        for i in self._arguments.keys():
            setattr(obj, i, None)
        for i in self._paramters:
            setattr(obj, i.name, None)

    def __eq__(self, other):
        if hasattr(other, '_arguments'):
            other = other._arguments
        return self._arguments.__eq__(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def copy(self):
        config = NNConfig(**self._arguments)
        config._paramters = self._paramters
        return config

    def __str__(self):
        s = 'Arguments:\n'
        for i, j in self._arguments.iteritems():
            s += ' - ' + str(i) + ':' + str(j) + '\n'
        s += ' - Parameters: ' + ', '.join([str(i) for i in self._paramters])
        return s

    # ==================== pickling method ==================== #
    def __getstate__(self):
        return self._arguments, [K.pickling_variable(i) for i in self.parameters]

    def __setstate__(self, states):
        self._arguments = states[0]
        for i, j in self._arguments.iteritems():
            setattr(self, i, j)
        self._paramters = [K.pickling_variable(i) for i in states[1]]


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

    """

    ID = 0

    def __init__(self, name=None):
        super(NNOps, self).__init__()
        self.name = name
        if name is None:
            self.name = "%s_%d" % (self.__class__.__name__, NNOps.ID)
        NNOps.ID += 1

        self._configuration = None
        self._transpose_ops = None
        self._arguments = {}

    # ==================== properties ==================== #
    @property
    def T(self):
        """ Return new ops which is transpose of this ops """
        if self._transpose_ops is None:
            self._transpose_ops = self._transpose()
        return self._transpose_ops

    @property
    def parameters(self):
        if self._configuration is None:
            raise Exception("This operators haven't initialized.")
        return [i for i in self._configuration.parameters
                if has_roles(i, PARAMETER)]

    @property
    def configuration(self):
        return self._configuration

    def __setattr__(self, name, value):
        # this record all assigned attribute to pickle them later
        if hasattr(self, '_arguments') and name != '_arguments':
            if name in self._arguments:
                self._arguments[name] = value
            # otherwise, only save primitive types
            elif isinstance(value, _primitive_types):
                self._arguments[name] = value
        super(NNOps, self).__setattr__(name, value)

    # ==================== abstract method ==================== #
    @abstractmethod
    def _initialize(self, x, **kwargs):
        """ This function return NNConfig for given configuration from arg
        and kwargs
        """
        raise NotImplementedError

    @abstractmethod
    def _apply(self, x, **kwargs):
        raise NotImplementedError

    def _transpose(self):
        raise NotImplementedError

    # ==================== interaction method ==================== #
    def apply(self, x, **kwargs):
        return_list = True
        if not isinstance(x, (tuple, list)):
            x = [x]
            return_list = False
        # ====== initialize first ====== #
        # only select necessary arguments
        argspec = inspect.getargspec(self._initialize)
        keywords = {}
        # kwargs must be specified in args, or the _initialize
        # must accept **kwargs
        for i, j in kwargs.iteritems():
            if argspec.keywords is not None or i in argspec.args:
                keywords[i] = j
        # initialize the operator (call the initilazation process)
        if self._configuration is None:
            config = self._initialize(x if len(x) > 1 else x[0],
                                      **keywords)
            if not isinstance(config, NNConfig):
                raise Exception('Returned value from _initialize function must '
                                'be instance of NNConfig.')
            config.inflate(self)
            self._configuration = config
        # check footprint for  warning
        footprint = _recurrsive_extract_shape(x)
        if not hasattr(self, '_footprint'):
            self._footprint = footprint
        elif any(i != j for i, j in zip(self._footprint, footprint)):
            warnings.warn('The initialized footprint is "{}" which '
                          'is different from the given footprint: "{}"'
                          '.'.format(self._footprint, footprint))
        # ====== calculate and return outputs ====== #
        out = [self._apply(i, **kwargs) for i in x]
        if return_list:
            return out
        return out[0]

    def __call__(self, x, **kwargs):
        return self.apply(x, **kwargs)

    def __str__(self):
        ops_format = '<ops: %s, name: %s, init: %s>'
        return ops_format % (self.__class__.__name__, self.name,
                             self._configuration is not None)

    # ==================== Slicing ==================== #
    def __getitem__(self, key):
        return NNSliceOps(self, key)

    # ==================== pickling method ==================== #
    def __getstate__(self):
        return (self.name, self._configuration, self._arguments)

    def __setstate__(self, states):
        name, config, attrs = states
        # increase the ID or will be overlap name
        NNOps.ID += 1
        self.name = name
        self._transpose_ops = None # reset the transpose ops
        for i, j in attrs.iteritems():
            setattr(self, i, j)
        self._arguments = attrs
        self._configuration = config
        if config is not None:
            config.inflate(self)


# ===========================================================================
# Helper
# ===========================================================================
class NNSliceOps(NNOps):

    def __init__(self, ops, slice):
        if not isinstance(ops, NNOps):
            raise ValueError('ops must be instance of NNOps, but given argument '
                             'has %s' % str(type(ops)))
        super(NNSliceOps, self).__init__()
        self.ops = ops
        if not isinstance(slice, (tuple, list)):
            slice = [slice]
        self.slice = slice

    def _initialize(self, x):
        return NNConfig()

    def _apply(self, x, **kwargs):
        y = self.ops.apply(x, **kwargs)
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
        return ops_format % (self.ops.__class__.__name__, self.ops.name,
                             self.ops._configuration is not None,
                             str(self.slice))


# ===========================================================================
# Simple ops
# ===========================================================================
class Dense(NNOps):

    @autoinit
    def __init__(self, num_units,
                 W_init=K.init.glorot_uniform,
                 b_init=K.init.constant(0),
                 activation=K.linear,
                 **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.activation = (K.linear if activation is None else activation)
        # hack to prevent infinite useless loop of transpose
        self._original_dense = None

    # ==================== abstract methods ==================== #
    def _transpose(self):
        if self._original_dense is not None:
            return self._original_dense

        # flip the input and hidden
        num_inputs = self.num_units
        num_units = self.num_inputs
        # create the new dense
        transpose = Dense(num_units=num_units,
                          W_init=self.W_init, b_init=self.b_init,
                          activation=self.activation,
                          name=self.name + '_transpose')
        transpose._original_dense = self
        #create the config
        config = NNConfig(num_inputs=num_inputs)
        config.create_params(self.W.T, shape=(num_inputs, num_units), name='W',
                             nnops=transpose)
        if self.b_init is not None:
            config.create_params(self.b_init, shape=(num_units,), name='b',
                                 nnops=transpose, roles=BIAS)
        # modify the config
        transpose._configuration = config
        config.inflate(transpose)
        return transpose

    def _initialize(self, x):
        input_shape = K.get_shape(x)

        config = NNConfig(num_inputs=input_shape[-1])
        shape = (input_shape[-1], self.num_units)
        config.create_params(self.W_init, shape, 'W', nnops=self, roles=WEIGHT)
        if self.b_init is not None:
            config.create_params(self.b_init, (self.num_units,), 'b',
                                 nnops=self, roles=BIAS)
        return config

    def _apply(self, x):
        input_shape = K.get_shape(x)
        # calculate projection
        activation = K.dot(x, self.W)
        if hasattr(self, 'b') and self.b is not None:
            activation = activation + self.b
        # set shape for output
        K.add_shape(activation, input_shape[:-1] + (self.num_units,))
        # Nonlinearity might change the shape of activation
        activation = self.activation(activation)
        return activation


class VariationalDense(NNOps):

    @autoinit
    def __init__(self, num_units,
                 W_init=K.init.symmetric_uniform,
                 b_init=K.init.constant(0),
                 activation=K.linear,
                 seed=None, **kwargs):
        super(VariationalDense, self).__init__(**kwargs)
        # hack to prevent infinite useless loop of transpose
        self._rng = K.rng(seed=seed)
        self.activation = K.linear if activation is None else activation

    # ==================== helper ==================== #
    @cache # same x will return the same mean and logsigma
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
        epsilon = self._rng.normal(shape=K.get_shape(mean), mean=0.0, std=1.0,
                                   dtype=mean.dtype)
        z = mean + K.exp(logsigma) * epsilon
        return z

    # ==================== abstract methods ==================== #
    def _transpose(self):
        raise NotImplementedError

    def _initialize(self, x):
        input_shape = K.get_shape(x)
        config = NNConfig(num_inputs=input_shape[-1])
        shape = (input_shape[-1], self.num_units)

        config.create_params(self.W_init, shape, 'W_mean',
                             nnops=self, roles=WEIGHT)
        config.create_params(self.W_init, shape, 'W_logsigma',
                             nnops=self, roles=WEIGHT)
        if self.b_init is not None:
            config.create_params(self.b_init, (self.num_units,), 'b_mean',
                                 nnops=self, roles=BIAS)
            config.create_params(self.b_init, (self.num_units,), 'b_logsigma',
                                 nnops=self, roles=BIAS)
        return config

    def _apply(self, x):
        input_shape = K.get_shape(x)
        # calculate statistics
        mean, logsigma = self.get_mean_logsigma(x)
        # variational output
        output = mean
        if K.is_training(x):
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
        See :func:`lasagne.utils.create_param` for more information.
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

    @autoinit
    def __init__(self, alpha_init=K.init.constant(0.25),
                 shared_axes='auto', **kwargs):
        super(ParametricRectifier, self).__init__(**kwargs)

    # ==================== abstract methods ==================== #
    def _initialize(self, x):
        input_shape = K.get_shape(x)
        config = NNConfig(input_shape=x)

        if self.shared_axes == 'auto':
            self.shared_axes = (0,) + tuple(range(2, len(input_shape)))
        elif self.shared_axes == 'all':
            self.shared_axes = tuple(range(len(input_shape)))
        elif isinstance(self.shared_axes, int):
            self.shared_axes = (self.shared_axes,)

        shape = [size for axis, size in enumerate(input_shape)
                 if axis not in self.shared_axes]
        if any(size is None for size in shape):
            raise ValueError("ParametricRectifierLayer needs input sizes for "
                             "all axes that alpha's are not shared over.")
        self.alpha = config.create_params(self.alpha_init, shape, name="alpha",
                                         nnops=self, roles=PARAMETER)
        return config

    def _apply(self, x):
        axes = iter(range(K.ndim(self.alpha)))
        pattern = ['x' if input_axis in self.shared_axes
                   else next(axes)
                   for input_axis in range(K.ndim(x))]
        alpha = K.dimshuffle(self.alpha, pattern)
        return K.relu(x, alpha)
