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
from odin.annotations import add_annotation, Annotation
from odin.roles import (add_role, has_roles, PARAMETER, VariableRole,
                        WEIGHT, BIAS,
                        VARIATIONAL_MEAN, VARIATIONAL_LOGSIGMA,
                        BATCH_NORM_SHIFT_PARAMETER,
                        BATCH_NORM_POPULATION_MEAN,
                        BATCH_NORM_SCALE_PARAMETER,
                        BATCH_NORM_POPULATION_STDEV)
from odin.utils.decorators import autoinit, functionable, cache
# ===========================================================================
# Helper
# ===========================================================================
_primitive_types = (tuple, list, dict, str, numbers.Number, types.NoneType)


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

    def create_params(self, spec, shape, name, roles=[], annotations=[]):
        if not isinstance(roles, (tuple, list)):
            roles = [roles]
        if not isinstance(annotations, (tuple, list)):
            annotations = [annotations]

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
        for i in annotations:
            if isinstance(i, Annotation):
                add_annotation(spec, i)
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
            setattr(obj, i.name, i)

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
class NNOps(Annotation):

    ID = 0

    def __init__(self, name=None):
        super(NNOps, self).__init__()
        self._id = NNOps.ID
        self._name = str(name)
        self._configuration = None
        self._transpose_ops = None
        self._arguments = {}
        NNOps.ID += 1

    # ==================== properties ==================== #
    @property
    def name(self):
        return '[' + str(self._id) + ']' + self.__class__.__name__ + '/' + self._name

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

    def __setattr__(self, name, value):
        if hasattr(self, '_arguments') and name != '_arguments':
            if name in self._arguments:
                self._arguments[name] = value
            # otherwise, only save primitive types
            elif isinstance(value, _primitive_types):
                self._arguments[name] = value
        super(NNOps, self).__setattr__(name, value)

    def config(self, *args, **kwargs):
        """
        Note
        ----
        New configuration will be created based on kwargs
        args only for setting the NNConfig directly
        """
        for i in args:
            if isinstance(i, NNConfig):
                self._configuration = i

        # initialized but mismatch configuration
        if self._configuration is not None:
            if len(kwargs) != 0 and self._configuration != kwargs:
                raise ValueError('Initialized configuration: {} is mismatch '
                                 'with new configuration, no support for'
                                 ' kwargs={}'.format(
                                     self._configuration._arguments, kwargs))
        # # not initialized but no information
        # elif len(kwargs) == 0:
        #     raise ValueError('Configuration have not initialized.')

        # still None, initialize configuration
        if self._configuration is None:
            self._configuration = self._initialize(**kwargs)
        self._configuration.inflate(self)
        return self._configuration

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
        out = self._apply(*args, **kwargs)
        # ====== add roles ====== #
        tmp = out
        if not isinstance(tmp, (tuple, list)):
            tmp = [out]
        for o in tmp:
            add_annotation(o, self)
        # return outputs
        return out

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    def __str__(self):
        return self.name

    # ==================== pickling method ==================== #
    def __getstate__(self):
        return (self._id, self._name, self._configuration, self._arguments)

    def __setstate__(self, states):
        id, name, config, attrs = states
        self._id = id
        self._name = name
        self._transpose_ops = None # reset the transpose ops
        for i, j in attrs.iteritems():
            setattr(self, i, j)
        self._arguments = attrs
        self._configuration = config
        # restore the annotations
        if config is not None:
            config.inflate(self)
            for i in config.parameters:
                add_annotation(i, self)


# ===========================================================================
# Simple ops
# ===========================================================================
class Dense(NNOps):

    @autoinit
    def __init__(self, num_units,
                 W_init=K.init.glorot_uniform,
                 b_init=K.init.constant,
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
                          name=self._name + '_transpose')
        transpose._original_dense = self
        #create the config
        config = NNConfig(num_inputs=num_inputs)
        config.create_params(self.W.T, shape=(num_inputs, num_units), name='W')
        if self.b_init is not None:
            config.create_params(self.b_init, shape=(num_units,), name='b',
                                 roles=BIAS, annotations=transpose)
        # modify the config
        transpose.config(config)
        return transpose

    def _initialize(self, num_inputs):
        config = NNConfig(num_inputs=num_inputs)
        shape = (num_inputs, self.num_units)
        config.create_params(self.W_init, shape, 'W', WEIGHT, self)
        if self.b_init is not None:
            config.create_params(self.b_init, (self.num_units,), 'b', BIAS, self)
        return config

    def _apply(self, x):
        input_shape = K.get_shape(x)
        self.config(num_inputs=input_shape[-1])
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
                 b_init=K.init.constant,
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
        add_annotation(mean, self)
        add_role(logsigma, VARIATIONAL_LOGSIGMA)
        add_annotation(logsigma, self)
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

    def _initialize(self, num_inputs):
        config = NNConfig(num_inputs=num_inputs)
        shape = (num_inputs, self.num_units)

        config.create_params(self.W_init, shape, 'W_mean', WEIGHT, self)
        config.create_params(self.W_init, shape, 'W_logsigma', WEIGHT, self)
        if self.b_init is not None:
            config.create_params(self.b_init, (self.num_units,), 'b_mean', BIAS, self)
            config.create_params(self.b_init, (self.num_units,), 'b_logsigma', BIAS, self)
        return config

    def _apply(self, x):
        input_shape = K.get_shape(x)
        self.config(num_inputs=input_shape[-1])
        # calculate statistics
        mean, logsigma = self.get_mean_logsigma(x)
        # variational output
        output = mean
        if K.is_training(x):
            output = self.sampling(x)
        # set shape for output
        K.add_shape(output, input_shape[:-1] + (self.num_units,))
        return output


class BatchNorm(NNOps):
    """ This class is adpated from Lasagne:
    Original work Copyright (c) 2014-2015 lasagne contributors
    All rights reserved.
    LICENSE: https://github.com/Lasagne/Lasagne/blob/master/LICENSE

    Batch Normalization

    This layer implements batch normalization of its inputs, following [1]_:

    .. math::
        y = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\gamma + \\beta

    That is, the input is normalized to zero mean and unit variance, and then
    linearly transformed. The crucial part is that the mean and variance are
    computed across the batch dimension, i.e., over examples, not per example.

    During training, :math:`\\mu` and :math:`\\sigma^2` are defined to be the
    mean and variance of the current input mini-batch :math:`x`, and during
    testing, they are replaced with average statistics over the training
    data. Consequently, this layer has four stored parameters: :math:`\\beta`,
    :math:`\\gamma`, and the averages :math:`\\mu` and :math:`\\sigma^2`
    (nota bene: instead of :math:`\\sigma^2`, the layer actually stores
    :math:`1 / \\sqrt{\\sigma^2 + \\epsilon}`, for compatibility to cuDNN).
    By default, this layer learns the average statistics as exponential moving
    averages computed during training, so it can be plugged into an existing
    network without any changes of the training procedure (see Notes).

    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    axes : 'auto', int or tuple of int
        The axis or axes to normalize over. If ``'auto'`` (the default),
        normalize over all axes except for the second: this will normalize over
        the minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers.
    epsilon : scalar
        Small constant :math:`\\epsilon` added to the variance before taking
        the square root and dividing by it, to avoid numerical problems
    alpha : scalar
        Coefficient for the exponential moving average of batch-wise means and
        standard deviations computed during training; the closer to one, the
        more it will depend on the last batches seen
    beta : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\beta`. Must match
        the incoming shape, skipping all axes in `axes`. Set to ``None`` to fix
        it to 0.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    gamma : Theano shared variable, expression, numpy array, callable or None
        Initial value, expression or initializer for :math:`\\gamma`. Must
        match the incoming shape, skipping all axes in `axes`. Set to ``None``
        to fix it to 1.0 instead of learning it.
        See :func:`lasagne.utils.create_param` for more information.
    mean : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`\\mu`. Must match
        the incoming shape, skipping all axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    inv_std : Theano shared variable, expression, numpy array, or callable
        Initial value, expression or initializer for :math:`1 / \\sqrt{
        \\sigma^2 + \\epsilon}`. Must match the incoming shape, skipping all
        axes in `axes`.
        See :func:`lasagne.utils.create_param` for more information.
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    This layer should be inserted between a linear transformation (such as a
    :class:`DenseLayer`, or :class:`Conv2DLayer`) and its activation. The
    convenience function :func:`batch_norm` modifies an existing layer to
    insert batch normalization in front of its activation.

    The behavior can be controlled by passing keyword arguments to
    :func:`lasagne.layers.get_output()` when building the output expression
    of any network containing this layer.

    During training, [1]_ normalize each input mini-batch by its statistics
    and update an exponential moving average of the statistics to be used for
    validation. This can be achieved by passing ``deterministic=False``.
    For validation, [1]_ normalize each input mini-batch by the stored
    statistics. This can be achieved by passing ``deterministic=True``.

    For more fine-grained control, ``batch_norm_update_averages`` can be passed
    to update the exponential moving averages (``True``) or not (``False``),
    and ``batch_norm_use_averages`` can be passed to use the exponential moving
    averages for normalization (``True``) or normalize each mini-batch by its
    own statistics (``False``). These settings override ``deterministic``.

    Note that for testing a model after training, [1]_ replace the stored
    exponential moving average statistics by fixing all network weights and
    re-computing average statistics over the training data in a layerwise
    fashion. This is not part of the layer implementation.

    In case you set `axes` to not include the batch dimension (the first axis,
    usually), normalization is done per example, not across examples. This does
    not require any averages, so you can pass ``batch_norm_update_averages``
    and ``batch_norm_use_averages`` as ``False`` in this case.

    See also
    --------
    batch_norm : Convenience function to apply batch normalization to a layer

    References
    ----------
    .. [1] Ioffe, Sergey and Szegedy, Christian (2015):
           Batch Normalization: Accelerating Deep Network Training by Reducing
           Internal Covariate Shift. http://arxiv.org/abs/1502.03167.
    """

    @autoinit
    def __init__(self, axes='auto', epsilon=1e-4, alpha=0.1,
                 beta_init=K.init.constant,
                 gamma_init=lambda x: K.init.constant(x, 1.),
                 mean_init=K.init.constant,
                 inv_std_init=lambda x: K.init.constant(x, 1.),
                 activation=K.linear, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        self.activation = K.linear if activation is None else activation

    # ==================== abstract method ==================== #
    # def _transpose(self):
        # return None

    def _initialize(self, input_shape):
        """ This function return NNConfig for given configuration from arg
        and kwargs
        """
        config = NNConfig(input_shape=input_shape)
        if self.axes == 'auto':
            # default: normalize over all but the second axis
            self.axes = (0,) + tuple(range(2, len(input_shape)))
        elif isinstance(self.axes, int):
            self.axes = (self.axes,)
        # create parameters, ignoring all dimensions in axes
        shape = [size for axis, size in enumerate(input_shape)
                 if axis not in self.axes]
        if any(size is None for size in shape):
            raise ValueError("BatchNorm needs specified input sizes for "
                             "all axes not normalized over.")
        # init parameters
        if self.beta_init is not None:
            config.create_params(self.beta_init, shape=shape, name='beta',
                                 roles=BATCH_NORM_SHIFT_PARAMETER,
                                 annotations=self)
        if self.gamma_init is not None:
            config.create_params(self.gamma_init, shape=shape, name='gamma',
                                 roles=BATCH_NORM_SCALE_PARAMETER,
                                 annotations=self)
        config.create_params(self.mean_init, shape=shape, name='mean',
                             roles=BATCH_NORM_POPULATION_MEAN,
                             annotations=self)
        config.create_params(self.inv_std_init, shape=shape, name='inv_std',
                             roles=BATCH_NORM_POPULATION_STDEV,
                             annotations=self)
        return config

    def _apply(self, x):
        import theano

        input_shape = K.get_shape(x)
        is_training = K.is_training(x)
        ndim = K.ndim(x)
        self.config(input_shape=input_shape)
        # ====== training mode ====== #
        input_mean = K.mean(x, self.axes)
        input_inv_std = K.inv(K.sqrt(K.var(x, self.axes) + self.epsilon))

        # Decide whether to use the stored averages or mini-batch statistics
        if not is_training:
            mean = self.mean
            inv_std = self.inv_std
        else: # update the stored averages
            mean = input_mean
            inv_std = input_inv_std
            # Trick: To update the stored statistics, we create memory-aliased
            # clones of the stored statistics:
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_inv_std = theano.clone(self.inv_std, share_inputs=False)
            # set a default update for them:
            running_mean.default_update = ((1 - self.alpha) * running_mean +
                                           self.alpha * input_mean)
            running_inv_std.default_update = ((1 - self.alpha) *
                                              running_inv_std +
                                              self.alpha * input_inv_std)
            # and make sure they end up in the graph without participating in
            # the computation (this way their default_update will be collected
            # and applied, but the computation will be optimized away):
            mean += 0 * running_mean
            inv_std += 0 * running_inv_std
        # prepare dimshuffle pattern inserting broadcastable axes as needed
        param_axes = iter(range(ndim - len(self.axes)))
        pattern = ['x' if input_axis in self.axes
                   else next(param_axes)
                   for input_axis in range(ndim)]

        # apply dimshuffle pattern to all parameters
        beta = 0 if self.beta is None else K.dimshuffle(self.beta, pattern)
        gamma = 1 if self.gamma is None else K.dimshuffle(self.gamma, pattern)
        mean = K.dimshuffle(mean, pattern)
        inv_std = K.dimshuffle(inv_std, pattern)

        # normalize
        normalized = (x - mean) * (gamma * inv_std) + beta
        # set shape for output
        K.add_shape(normalized, input_shape)
        return self.activation(normalized)


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
    def __init__(self, alpha_init=lambda x: K.init.constant(x, 0.25),
                 shared_axes='auto', **kwargs):
        super(ParametricRectifier, self).__init__(**kwargs)

    # ==================== abstract methods ==================== #
    def _initialize(self, input_shape):
        config = NNConfig(input_shape=input_shape)

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
                                          roles=PARAMETER, annotations=self)
        return config

    def _apply(self, x):
        input_shape = K.get_shape(x)
        self.config(input_shape=input_shape)
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
        pass

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
                            name=self._name + '_transpose')
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
        if not isinstance(ops, (tuple, list)):
            ops = [ops]
        for i in ops:
            if inspect.isfunction(i) or inspect.ismethod(i):
                self.ops.append(functionable(i))
            elif hasattr(i, '__call__'):
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
