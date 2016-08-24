from __future__ import division, absolute_import, print_function

import inspect
import numbers
import types
import cPickle
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
from odin.utils.decorators import autoinit, cache

# ===========================================================================
# Helper
# ===========================================================================
_primitive_types = (tuple, list, dict, types.StringType, types.BooleanType,
                    types.FunctionType, numbers.Number, types.NoneType,
                    K.init.constant)


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

    def create_params(self, spec, shape, name, nnops, roles=[]):
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

        #####################################
        # 1. Shared variable, just check the shape.
        if K.is_trainable_variable(spec):
            spec_shape = K.eval(K.get_shape(spec))
            if shape is None:
                shape = spec_shape
            elif tuple(shape) != tuple(spec_shape):
                self.raise_arguments('Given variable has different shape '
                                     'from requirement, %s != %s' %
                                     (str(spec_shape), str(shape)))
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
                self.raise_arguments("parameter variable has %d dimensions, "
                                   "should be %d" % (spec.ndim, len(shape)))
        #####################################
        # 3. numpy ndarray, create shared variable wraper for it.
        elif isinstance(spec, np.ndarray):
            if shape is not None and spec.shape != shape:
                raise RuntimeError("parameter array has shape %s, should be "
                                   "%s" % (spec.shape, shape))
            with K.variable_scope(nnops.name):
                spec = K.variable(spec, name=name)
        #####################################
        # 4. initializing function.
        elif hasattr(spec, '__call__'):
            arr = spec(shape)
            if K.is_trainable_variable(arr):
                spec = arr
            elif K.is_variable(arr) and K.ndim(arr) == len(shape):
                spec = arr
            elif isinstance(arr, np.ndarray):
                with K.variable_scope(nnops.name):
                    spec = K.variable(arr, name=name)
        #####################################
        # 5. Exception.
        else:
            raise RuntimeError("cannot initialize parameters: 'spec' is not "
                               "a numpy array, a Theano expression, or a "
                               "callable")
        # ====== create and return params ====== #
        for i in roles:
            if isinstance(i, VariableRole):
                add_role(spec, i)
        if not K.is_trainable_variable(spec):
            spec.name = name
        # return actual variable or expression
        for i, j in enumerate(self._paramters): # override other parameters with same name
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
    _apply(self, *args, **kwargs): resulted variables
        apply take a list of variables and custom parameters to compute
        output variables
    _initialize(self, *args, **kwargs): NNConfig
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
    def _initialize(self, *args, **kwargs):
        """ This function return NNConfig for given configuration from arg
        and kwargs
        """
        raise NotImplementedError

    @abstractmethod
    def _apply(self, *args, **kwargs):
        raise NotImplementedError

    def _transpose(self):
        raise NotImplementedError

    # ==================== interaction method ==================== #
    def apply(self, *args, **kwargs):
        # ====== initialize first ====== #
        if self._configuration is None:
            # only select necessary arguments
            argspec = inspect.getargspec(self._initialize)
            keywords = {}
            # positional arguments
            for i, j in zip(argspec.args[1:], args):
                keywords[i] = j
            # kwargs must be specified in args, or the _initialize
            # must accept **kwargs
            for i, j in kwargs.iteritems():
                if argspec.keywords is not None or i in argspec.args:
                    keywords[i] = j
            # call the initilazation process
            config = self._initialize(**keywords)
            if not isinstance(config, NNConfig):
                raise Exception('Returned value from _initialize function must '
                                'be instance of NNConfig.')
            config.inflate(self)
            self._configuration = config
        # ====== calculate and return outputs ====== #
        out = self._apply(*args, **kwargs)
        return out

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def __str__(self):
        return self.name

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


class Switcher(NNOps):
    """ Simple Ops, perform specific Ops while training and other one for
    deploying
    """

    def __init__(self, training, deploying, **kwargs):
        super(Switcher, self).__init__(**kwargs)
        self.training = training
        self.deploying = deploying

    def _initialize(self, *args, **kwargs):
        return NNConfig()

    def _apply(self, *args, **kwargs):
        is_training = False
        for i in chain(args, kwargs.values()):
            if K.is_variable(i) and K.is_training(i):
                is_training = True
        if is_training:
            return self.training(*args, **kwargs)
        else:
            return self.deploying(*args, **kwargs)

    def _transpose(self):
        if hasattr(self.training, 'T') and hasattr(self.deploying, 'T'):
            return Switcher(self.training.T, self.deploying.T,
                            name=self.name + '_transpose')
        raise Exception('One of training or deploying ops do not support transpose.')


class Sequence(NNOps):

    """ Sequence of Operators
    Parameters
    ----------
    strict_transpose : bool
        if True, only operators with transposed implemented are added
        to tranpose operator

    Example
    -------
    """

    @autoinit
    def __init__(self, ops, strict_transpose=False, **kwargs):
        super(Sequence, self).__init__(**kwargs)
        self.ops = []
        if hasattr(strict_transpose, '__call__'):
            raise Exception('You made a funny mistake, ops must be list.')
        if not isinstance(ops, (tuple, list)):
            ops = [ops]
        for i in ops:
            if hasattr(i, '__call__'):
                self.ops.append(i)

    @property
    def parameters(self):
        all_parameters = list(chain(
            *[i.parameters for i in self.ops if hasattr(i, 'parameters')]))
        return [i for i in all_parameters if has_roles(i, PARAMETER)]

    def _initialize(self, *args, **kwargs):
        return NNConfig()

    def _apply(self, x):
        for op in self.ops:
            x = op(x)
        return x

    def _transpose(self):
        transpose_ops = []
        for i in self.ops:
            if hasattr(i, 'T'):
                transpose_ops.append(i.T)
            elif not self.strict_transpose:
                transpose_ops.append(i)
        # reversed the order of ops for transpose
        transpose_ops = list(reversed(transpose_ops))
        seq = Sequence(transpose_ops)
        return seq

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.ops.__getitem__(key)
        elif isinstance(key, slice):
            return Sequence(self.ops.__getitem__(key))
        elif isinstance(key, str):
            for i in self.ops:
                if hasattr(i, '_name') and i._name == key:
                    return i
        raise ValueError('key can only be int, slice or str.')

    def __setitem__(self, key, value):
        return self.ops.__setitem__(key, value)

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
