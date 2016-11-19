from __future__ import print_function, division, absolute_import

import inspect
import warnings
import __builtin__
from functools import wraps
from numbers import Number

import numpy as np

from odin.config import CONFIG, RNG_GENERATOR
from odin.utils import as_tuple
from odin.basic import add_updates, get_shape, add_shape, add_role, ACTIVATION_PARAMETER

from .basic_ops import (is_variable, ndim, expand_dims, repeat, dimshuffle,
                        concatenate, clip, log, one_hot, reshape, constant,
                        eq, ge, lt, cast, mean, argmax, argtop_k, Scan, relu,
                        exp, sqrt, pow, sum, square, switch, flatten, eval,
                        is_trainable_variable, is_training, addbroadcast,
                        random_uniform, random_normal, random_binomial,
                        backend_ops_categorical_crossentropy,
                        backend_ops_binary_crossentropy)
FLOATX = CONFIG.floatX
EPSILON = CONFIG.epsilon


def rnn_decorator(*args, **kwargs):
    """Wraps any method (or function) to allow its iterative application.

    The decorator allows you to implement step_function and assign sequences
    arguments to the function in very flexible way.

    The idea behind this function is characterizing recursive function by
    3 primitive information:
     * `sequences`: the sequences to iterative over
     (i.e nb_samples, nb_time, nb_features)
     * `states`: describe output information (i.e. the initial value of
     output after each timestep)

    In the decorator, you are allowed to provide the `name` (in string) of
    above variables, the process of looking for these name are following:
     * If your `callable` is a method (i.e bound to an object), then the
     variables will be searched in the attributes of the object.
     * If your `callable` is a function (i.e the first argument is not an
     object but variable), then you have to specified all the information
     when you call the function.

    Parameters
    ----------
    sequences : list of strs
        Specifies which of the arguments are elements of input sequences.
        (batch_size, nb_time_step, trailing_dims)
    states : list of strs
        Specifies which of the arguments are the states.

    Sub-Parameters
    --------------
    iterate : bool
        If ``True`` iteration through whole sequence is made.
        By default ``True`` (i.e. False <=> stateful recurrent network)
    backwards : bool
        If ``True``, the sequences are processed in backward
        direction. ``False`` by default.
    n_steps: int
        number of timestep, required if not known in advance (i.e. the
        second dimension of sequences)
    batch_size: int
        batch size of input batch (i.e. the first dimension of sequences)
    repeat_states: bool
        repeat the states first dimension to match the batch_size
    name: str
        name for the scan operator

    Returns
    -------
    recurrent_apply : The new method that applies the RNN to sequences.

    Note
    --------
    sub-parameters is the addition parameters that the step funciton will
    accept
    The arguments inputed directly to the function will override the arguments
    in container object

    Example
    -------
    """
    #####################################
    # 0. Helper functions.
    def to_list(x):
        return [] if x is None else ([x] if not isinstance(x, (tuple, list))
                                     else list(x))

    def find_arg(name, type, container, kwargs):
        # if given name not found, return None
        if not isinstance(name, str):
            raise ValueError('Given sequences, states, contexts must be '
                             'string represent the name of variable in the '
                             'input arguments of step function or attributes '
                             'of container class, name="%s"' % str(name))
        # given name as string
        value = None
        if name in kwargs:
            value = kwargs[name]
        # if the variable is None, find it in the container
        if value is None:
            value = getattr(container, name, None)
        return value

    #####################################
    # 1. Getting all arguments.
    # Decorator can be used with or without arguments
    if len(args) > 1:
        raise Exception('You can use this "recurrent" function in 2 ways: \n'
                        ' - input the step_function directly to *arg, and '
                        'specify other parameters in **kwargs.\n'
                        ' - use this as a decorator, and only need to specify '
                        'the parameters in **kwargs.\n')
    sequences = to_list(kwargs.pop('sequences', []))
    states = to_list(kwargs.pop('states', []))
    if __builtin__.any(not isinstance(i, str) for i in sequences + states):
        raise Exception('"sequences", "contexts", and "states" must be '
                        'string, which specify the name of variable in '
                        'the container or in arguments of step_function.')

    #####################################
    # 2. Create wrapper.
    def recurrent_wrapper(step_function):
        arg_spec = inspect.getargspec(step_function)
        arg_names = arg_spec.args
        # all defaults arguments
        if arg_spec.defaults is not None:
            defaults_args = dict(zip(
                reversed(arg_spec.args),
                reversed(arg_spec.defaults)
            ))
        else:
            defaults_args = dict()
        nb_required_args = len(arg_names) - len(defaults_args)

        @wraps(step_function)
        def recurrent_apply(*args, **kwargs):
            """ Iterates a transition function. """
            # Extract arguments related to iteration and immediately relay the
            # call to the wrapped function if `iterate=False`
            iterate = kwargs.pop('iterate', True)
            # ====== not iterate mode, just return step_function ====== #
            if not iterate:
                return step_function(*args, **kwargs)
            # otherwise, continue, container is the object store all
            # necessary variables
            if is_variable(args[0]) or len(args) == 0:
                container = None
            else:
                container = args[0]
            # ====== additional parameters ====== #
            backwards = kwargs.pop('backwards', False)
            n_steps = kwargs.pop('n_steps', None)
            batch_size = kwargs.pop('batch_size', None)
            repeat_states = kwargs.pop('repeat_states', False)
            name = kwargs.pop('name', None)
            # ====== Update the positional arguments ====== #
            step_args = dict(defaults_args)
            step_args.update(kwargs)
            # key -> positional_args
            for key, value in zip(arg_spec.args, args):
                step_args[key] = value
            # ====== looking for all variables ====== #
            sequences_given = [find_arg(i, 'sequences', container, step_args)
                               for i in sequences]
            states_given = [find_arg(i, 'states', container, step_args)
                            for i in states]
            # check all is variables
            if __builtin__.any(not is_variable(i) and i is not None
                   for i in sequences_given + states_given):
                raise ValueError('All variables provided to sequences, '
                                 'contexts, or states must be Variables.'
                                 'sequences:%s states:%s' %
                                 (str(sequences_given), str(states_given)))
            # ====== configuraiton for iterations ====== #
            # Assumes time dimension is the second dimension
            shape = get_shape(sequences_given[0], not_none=True)
            if n_steps is None:
                n_steps = shape[1]
            if batch_size is None:
                batch_size = shape[0]
            # ====== Ensure all initial states are with the right shape.
            _ = []
            for key, init_val in zip(states, states_given):
                shape = None if init_val is None else get_shape(init_val)
                # only one vector given for 1 batch matrix, should be repeated
                if init_val is not None and (ndim(init_val) == 1 or shape[0] == 1):
                    if repeat_states:
                        init_val = (expand_dims(init_val, 0)
                                    if ndim(init_val) == 1 else init_val)
                        init_val = repeat(init_val, batch_size, axes=0)
                    else:
                        warnings.warn('The "states" should be initialized for all '
                                      'samples in 1 batch (i.e. the first dimension, '
                                      'should be equal to the batch_size, you can '
                                      'repeat the first dimension of "%s"' % key)
                _.append(init_val)
            # Theano issue 1772
            if CONFIG.backend == 'theano':
                from theano import tensor as T
                states_given = [None if state is None else
                                T.unbroadcast(state, *range(state.ndim))
                                for state in _]
            # ====== shuffle sequences variable to get time dimension first
            sequences_given = [dimshuffle(i, (1, 0) + tuple(range(2, ndim(i))))
                               if i is not None else i
                               for i in sequences_given]

            # ====== create steps functions ====== #
            arg_order = ([i for i, j in zip(sequences, sequences_given)
                          if j is not None] +
                         [i for i, j in zip(states, states_given)
                          if j is not None])

            def scan_function(*args):
                # step args contains all kwargs for step function
                step_args.update(zip(arg_order, args))
                # kwargs = dict(step_args)
                kwargs = {i: j for i, j in step_args.iteritems()
                          if i in arg_names}
                # check get all necessary parametesr for step fucntion
                if len(kwargs) < nb_required_args:
                    raise Exception('Step function require %d arguments, but '
                                    'only %d arguments given by Scan operator'
                                    '.' % (len(arg_names), len(kwargs)))
                # Call step_function
                outputs = step_function(**kwargs)
                # check valid number of return
                if not isinstance(outputs, (tuple, list)):
                    outputs = (outputs,)
                if len(outputs) != len(states):
                    raise Exception('Given %d initial states but the step '
                                    'function only return %d outputs'
                                    '.' % (len(states), len(outputs)))
                return outputs
            # ====== run the scan function ====== #
            # print('Sequences:', sequences_given)
            # print('States:', states_given)
            # print('Gobackward:', backwards)
            # print('NSteps:', n_steps)
            # print('BatchSize:', batch_size)
            # print('Repeat:', repeat_states)
            # print('Name:', name)
            results, updates = Scan(
                scan_function,
                sequences=[i for i in sequences_given if i is not None],
                outputs_info=states_given,
                n_steps=n_steps,
                backwards=backwards,
                name=name)
            # all the result in form (nb_time, nb_samples, trailing_dims)
            # we reshape them back to same as input
            results = [dimshuffle(i, [1, 0] + range(2, ndim(i)))
                       for i in to_list(results)]
            # Lasagne+blocks: if scan is backward reverse the output
            # but keras don't do this step (need to validate the performance)
            if backwards:
                results = [r[:, ::-1] for r in results]
            # ====== adding updates for all results if available ====== #
            if updates:
                for key, value in updates.iteritems():
                    for r in results:
                        add_updates(r, key, value)
            return results
        return recurrent_apply
    # NO arguments are passed, just decorator
    if args:
        step_function, = args
        return recurrent_wrapper(step_function)
    # other arguments are passes
    else:
        return recurrent_wrapper


# ===========================================================================
# Advanced activation
# ===========================================================================
def antirectify(x):
    """
    This is the combination of a sample-wise L2 normalization with the
    concatenation of:
        - the positive part of the input
        - the negative part of the input
    The result is a tensor of samples that are twice as large as
    the input samples.
    It can be used in place of a ReLU.
        - Input shape: 2D tensor of shape (samples, n)
        - Output shape: 2D tensor of shape (samples, 2*n)

    Notes
    -----
    When applying ReLU, assuming that the distribution of the previous
    output is approximately centered around 0., you are discarding half of
    your input. This is inefficient.
    Antirectifier allows to return all-positive outputs like ReLU, without
    discarding any data.
    Tests on MNIST show that Antirectifier allows to train networks with
    twice less parameters yet with comparable classification accuracy
    as an equivalent ReLU-based network.

    """
    if ndim(x) != 2:
        raise Exception('This Ops only support 2D input.')
    input_shape = get_shape(x)
    x -= mean(x, axis=1, keepdims=True)
    # l2 normalization
    x /= sqrt(sum(square(x), axis=1, keepdims=True))
    x = concatenate([relu(x, 0), relu(-x, 0)], axis=1)
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, (input_shape[0], input_shape[1] * 2))
    return x


def randrectify(x, lower=0.3, upper=0.8, shared_axes='auto'):
    """ This function is adpated from Lasagne
    Original work Copyright (c) 2014-2015 lasagne contributors
    All rights reserved.
    LICENSE: https://github.com/Lasagne/Lasagne/blob/master/LICENSE

    Applies a randomized leaky rectify activation to x.

    The randomized leaky rectifier was first proposed and used in the Kaggle
    NDSB Competition, and later evaluated in [1]_. Compared to the standard
    leaky rectifier :func:`leaky_rectify`, it has a randomly sampled slope
    for negative input during training, and a fixed slope during evaluation.

    Equation for the randomized rectifier linear unit during training:
    :math:`\\varphi(x) = \\max((\\sim U(lower, upper)) \\cdot x, x)`

    During evaluation, the factor is fixed to the arithmetic mean of `lower`
    and `upper`.

    Parameters
    ----------
    lower : Theano shared variable, expression, or constant
        The lower bound for the randomly chosen slopes.

    upper : Theano shared variable, expression, or constant
        The upper bound for the randomly chosen slopes.

    shared_axes : 'auto', 'all', int or tuple of int
        The axes along which the random slopes of the rectifier units are
        going to be shared. If ``'auto'`` (the default), share over all axes
        except for the second - this will share the random slope over the
        minibatch dimension for dense layers, and additionally over all
        spatial dimensions for convolutional layers. If ``'all'``, share over
        all axes, thus using a single random slope.

     References
    ----------
    .. [1] Bing Xu, Naiyan Wang et al. (2015):
       Empirical Evaluation of Rectified Activations in Convolutional Network,
       http://arxiv.org/abs/1505.00853
    """
    input_shape = get_shape(x)
    # ====== check lower and upper ====== #
    if is_trainable_variable(lower):
        add_role(lower, ACTIVATION_PARAMETER)
        lower.name = 'lower'
    if is_trainable_variable(upper):
        add_role(upper, ACTIVATION_PARAMETER)
        upper.name = 'upper'
    if not is_variable(lower > upper) and lower > upper:
        raise ValueError("Upper bound for Randomized Rectifier needs "
                         "to be higher than lower bound.")
    # ====== check shared_axes ====== #
    if shared_axes == 'auto':
        shared_axes = (0,) + tuple(range(2, len(input_shape)))
    elif shared_axes == 'all':
        shared_axes = tuple(range(len(input_shape)))
    elif isinstance(shared_axes, int):
        shared_axes = (shared_axes,)
    else:
        shared_axes = shared_axes
    # ====== main logic ====== #
    if not is_training(x) or upper == lower:
        x = relu(x, (upper + lower) / 2.0)
    else: # Training mode
        shape = list(input_shape)
        if any(s is None for s in shape):
            shape = list(x.shape)
        for ax in shared_axes:
            shape[ax] = 1

        rnd = random_uniform(tuple(shape),
                      low=lower,
                      high=upper,
                      dtype=FLOATX)
        rnd = addbroadcast(rnd, *shared_axes)
        x = relu(x, rnd)
    add_shape(x, input_shape)
    return x


# ===========================================================================
# Advanced cost function
# ===========================================================================
def bayes_crossentropy(y_pred, y_true, nb_classes=None):
    shape = get_shape(y_pred)
    if ndim(y_pred) == 1:
        y_pred = expand_dims(y_pred, -1)
        # add shape for y_pred0 so we can auto_infer the shape after concat
        y_pred0 = 1. - y_pred; add_shape(y_pred0, shape)
        y_pred = concatenate([y_pred0, y_pred], axis=-1)
    elif isinstance(shape[-1], Number) and shape[-1] == 1:
        # add shape for y_pred0 so we can auto_infer the shape after concat
        y_pred0 = 1. - y_pred; add_shape(y_pred0, shape)
        y_pred = concatenate([y_pred0, y_pred], axis=-1)
    if ndim(y_true) == 1:
        if nb_classes is None:
            raise Exception('y_pred and y_true must be one_hot encoded, '
                            'otherwise you have to provide nb_classes.')
        y_true = one_hot(y_true, nb_classes)
    # avoid numerical instability with _EPSILON clipping
    y_pred = clip(y_pred, EPSILON, 1.0 - EPSILON)
    if nb_classes is None:
        nb_classes = get_shape(y_true)[1]
    # ====== check distribution ====== #
    distribution = sum(y_true, axis=0)
    # ====== init confusion info loss ====== #
    # weighted by y_true
    loss = y_true * log(y_pred)
    # probability distribution of each class
    prob_distribution = dimshuffle(distribution / sum(distribution),
                                   ('x', 0))
    # we need to clip the prior probability distribution also
    prob_distribution = clip(prob_distribution, EPSILON, 1.0 - EPSILON)
    return - 1 / nb_classes * sum(loss / prob_distribution, axis=1)


def categorical_crossentropy(output, target):
    """ NOTE: the crossentropy is different between tensorflow and
    theano """
    input_shape = get_shape(output)
    # scale preds so that the class probas of each sample sum to 1
    output /= sum(output, axis=-1, keepdims=True)
    output = clip(output, EPSILON, 1.0 - EPSILON)
    if ndim(target) == 1:
        target = one_hot(target, get_shape(output)[-1])
    x = backend_ops_categorical_crossentropy(output, target)
    add_shape(x, input_shape[0])
    return x


def squared_error(output, target):
    return square(output - target)


def binary_crossentropy(output, target):
    input_shape = get_shape(output)
    if ndim(output) > 1: output = flatten(output, outdim=1)
    if ndim(target) > 1: target = flatten(target, outdim=1)
    target = cast(target, output.dtype)
    # avoid numerical instability with _EPSILON clipping
    output = clip(output, EPSILON, 1.0 - EPSILON)
    x = backend_ops_binary_crossentropy(output, target)
    add_shape(x, input_shape[0])
    return x


# ===========================================================================
# Metrics
# ===========================================================================
def LevenshteinDistance(s1, s2):
    ''' Implementation of the wikipedia algorithm, optimized for memory
    Reference: http://rosettacode.org/wiki/Levenshtein_distance#Python
    '''
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        newDistances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1 + 1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]


def LER(y_true, y_pred, return_mean=True):
    ''' This function calculates the Labelling Error Rate (PER) of the decoded
    networks output sequence (out) and a target sequence (tar) with Levenshtein
    distance and dynamic programming. This is the same algorithm as commonly used
    for calculating the word error rate (WER), or phonemes error rate (PER).

    Parameters
    ----------
    y_true : ndarray (nb_samples, seq_labels)
        true values of sequences
    y_pred : ndarray (nb_samples, seq_labels)
        prediction values of sequences

    Returns
    -------
    return : float
        Labelling error rate
    '''
    if not hasattr(y_true[0], '__len__') or isinstance(y_true[0], str):
        y_true = [y_true]
    if not hasattr(y_pred[0], '__len__') or isinstance(y_pred[0], str):
        y_pred = [y_pred]

    results = []
    for ytrue, ypred in zip(y_true, y_pred):
        results.append(LevenshteinDistance(ytrue, ypred) / len(ytrue))
    if return_mean:
        return np.mean(results)
    return results


def binary_accuracy(y_pred, y_true, threshold=0.5):
    """ Non-differentiable """
    if ndim(y_pred) > 1: y_pred = reshape(y_pred, (-1,))
    if ndim(y_true) > 1: y_true = reshape(y_true, (-1,))
    y_pred = ge(y_pred, threshold)
    return eq(cast(y_pred, 'int32'),
              cast(y_true, 'int32'))


def categorical_accuracy(y_pred, y_true, top_k=1):
    """ Non-differentiable """
    if ndim(y_true) == ndim(y_pred):
        y_true = argmax(y_true, axis=-1)
    elif ndim(y_true) != ndim(y_pred) - 1:
        raise TypeError('rank mismatch between y_true and y_pred')

    if top_k == 1:
        # standard categorical accuracy
        top = argmax(y_pred, axis=-1)
        return eq(top, y_true)
    else:
        # top-k accuracy
        top = argtop_k(y_pred, top_k)
        y_true = expand_dims(y_true, dim=-1)
        return any(eq(top, y_true), axis=-1)


def Cavg_gpu(y_llr, y_true, Ptar=0.5, Cfa=1., Cmiss=1., softmax_input=False):
    ''' Fast calculation of Cavg (for only 1 clusters)

    Parameters
    ----------
    y_llr: (nb_samples, nb_classes)
        log likelihood ratio: llr = log (P(data|target) / P(data|non-target))
    y_true: numpy array of shape (nb_samples,)
        Class labels.
    softmax_input: boolean
        if True, `y_llr` is the output probability from softmax and perform
        llr transform for `y_llr`

    '''
    if softmax_input:
        y_llr = log(y_llr / (1 - y_llr))

    thresh = np.log(Cfa / Cmiss) - np.log(Ptar / (1 - Ptar))
    n = y_llr.shape[1]

    if isinstance(y_true, (list, tuple)):
        y_true = np.asarray(y_true)
    if ndim(y_true) == 1:
        y_true = one_hot(y_true, n)

    y_false = switch(y_true, 0, 1) # invert of y_true, False Negative mask
    y_positive = switch(ge(y_llr, thresh), 1, 0)
    y_negative = switch(lt(y_llr, thresh), 1, 0) # inver of y_positive
    distribution = clip(sum(y_true, axis=0), 10e-8, 10e8) # no zero values
    # ====== Pmiss ====== #
    miss = sum(y_true * y_negative, axis=0)
    Pmiss = 100 * (Cmiss * Ptar * miss) / distribution
    # ====== Pfa ====== # This calculation give different results
    fa = sum(y_false * y_positive, axis=0)
    Pfa = 100 * (Cfa * (1 - Ptar) * fa) / distribution
    Cavg = mean(Pmiss) + mean(Pfa) / (n - 1)
    return Cavg


def Cavg_cpu(log_llh, y_true, cluster_idx=None,
             Ptar=0.5, Cfa=1, Cmiss=1, softmax_input=False):
    """Compute cluster-wise and total LRE'15 percentage costs.

    Parameters
    ----------
    log_llh: numpy array of shape (n_samples, n_classes)
        There are N log-likelihoods for each of T trials:
        loglh(t,i) = log P(trail_t | class_i) - offset_t,
        where:
            log denotes natural logarithm
            offset_t is an unspecified real constant that may vary by trial
    y: numpy array of shape (n_samples,)
        Class labels.
    cluster_idx: list,
        Each element is a list that represents a particular language
        cluster and contains all class labels that belong to the cluster.
    Ptar: float, optional
        Probability of a target trial.
    Cfa: float, optional
        Cost for False Acceptance error.
    Cmiss: float, optional
        Cost for False Rejection error.
    verbose: int, optional
        0 - print nothing
        1 - print only total cost
        2 - print total cost and cluster average costs
    softmax_input: boolean
        if True, `y_llr` is the output probability from softmax and perform
        llr transform for `y_llr`

    Returns
    -------
    cluster_cost: numpy array of shape (n_clusters,)
        It contains average percentage costs for each cluster as defined by
        NIST LRE-15 language detection task. See
        http://www.nist.gov/itl/iad/mig/upload/LRE15_EvalPlan_v22-3.pdf
    total_cost: float
        An average percentage cost over all clusters.

    """
    if softmax_input:
        log_llh = np.clip(log_llh, 10e-8, 1 - 10e-8)
        log_llh = np.log(log_llh / (1 - log_llh))

    if cluster_idx is None:
        cluster_idx = [list(range(0, log_llh.shape[-1]))]
    # ensure everything is numpy ndarray
    y_true = np.asarray(y_true)
    log_llh = np.asarray(log_llh)

    thresh = np.log(Cfa / Cmiss) - np.log(Ptar / (1 - Ptar))
    cluster_cost = np.zeros(len(cluster_idx))

    for k, cluster in enumerate(cluster_idx):
        L = len(cluster) # number of languages in a cluster
        fa = 0
        fr = 0
        for lang_i in cluster:
            N = np.sum(y_true == lang_i) + .0 # number of samples for lang_i
            for lang_j in cluster:
                if lang_i == lang_j:
                    err = np.sum(log_llh[y_true == lang_i, lang_i] < thresh) / N
                    fr += err
                else:
                    err = np.sum(log_llh[y_true == lang_i, lang_j] >= thresh) / N
                    fa += err

        # Calculate procentage
        cluster_cost[k] = 100 * (Cmiss * Ptar * fr + Cfa * (1 - Ptar) * fa / (L - 1)) / L

    total_cost = np.mean(cluster_cost)

    return cluster_cost, total_cost

# ===========================================================================
# Addition pooling
# ===========================================================================
def poolWTA(x, pool_size=(2, 2), axis=1):
    """ This function is adpated from Lasagne
    Original work Copyright (c) 2014-2015 lasagne contributors
    All rights reserved.
    LICENSE: https://github.com/Lasagne/Lasagne/blob/master/LICENSE

    'Winner Take All' layer

    This layer performs 'Winner Take All' (WTA) across feature maps: zero out
    all but the maximal activation value within a region.

    Parameters
    ----------
    pool_size : integer
        the number of feature maps per region.

    axis : integer
        the axis along which the regions are formed.

    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.

    Notes
    -----
    This layer requires that the size of the axis along which it groups units
    is a multiple of the pool size.
    """
    input_shape = get_shape(x)
    num_feature_maps = input_shape[axis]
    num_pools = num_feature_maps // pool_size

    if input_shape[axis] % pool_size != 0:
        raise ValueError("Number of input feature maps (%d) is not a "
                         "multiple of the region size (pool_size=%d)" %
                         (num_feature_maps, pool_size))

    pool_shape = ()
    arange_shuffle_pattern = ()
    for k in range(axis):
        pool_shape += (input_shape[k],)
        arange_shuffle_pattern += ('x',)

    pool_shape += (num_pools, pool_size)
    arange_shuffle_pattern += ('x', 0)

    for k in range(axis + 1, x.ndim):
        pool_shape += (input_shape[k],)
        arange_shuffle_pattern += ('x',)

    input_reshaped = reshape(x, pool_shape)
    max_indices = argmax(input_reshaped, axis=axis + 1, keepdims=True)

    arange = T.arange(pool_size).dimshuffle(*arange_shuffle_pattern)
    mask = reshape(T.eq(max_indices, arange), input_shape)
    output = x * mask
    add_shape(output, input_shape)
    return output


def poolGlobal(x, pool_function=mean):
    """ Global pooling

    This layer pools globally across all trailing dimensions beyond the 2nd.

    Parameters
    ----------
    pool_function : callable
        the pooling function to use. This defaults to `theano.tensor.mean`
        (i.e. mean-pooling) and can be replaced by any other aggregation
        function.

    Note
    ----
    output_shape = input_shape[:2]
    """
    input_shape = get_shape(x)
    x = pool_function(T.flatten(x, 3), axis=2)
    add_shape(x, input_shape[:2])
    return x


# ===========================================================================
# Noise
# ===========================================================================
def _process_noise_dim(input_shape, dims, ndim):
    """
    By default, each element is kept or dropped independently.  If `noise_shape`
    is specified, it must be
    [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
    will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
    and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
    kept independently and each row and column will be kept or not kept together.

    Examples
    --------
    (None, 10, 10) with noise_dims=2
    => (None, 10, 1)
    """
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    dims = [i % ndim for i in dims]
    # ====== get noise shape ====== #
    if dims is None:
        noise_shape = input_shape
    else:
        return tuple([1 if i in dims else input_shape[i]
                      for i in range(ndim)])
    return noise_shape


def apply_dropout(x, level=0.5, noise_dims=None, noise_type='uniform',
                  rescale=True):
    """Computes dropout.

    With probability `keep_prob`, outputs the input element scaled up by
    `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
    sum is unchanged.


    Parameters
    ----------
    x: A tensor.
        input tensor
    level: float(0.-1.)
        probability dropout values in given tensor
    noise_dims: int or list(int)
        these dimensions will be setted to 1 in noise_shape, and
        used to broadcast the dropout mask.
    noise_type: 'gaussian' (or 'normal'), 'uniform'
        distribution used for generating noise
    rescale: bool
        whether rescale the outputs by dividing the retain probablity
    seed: random seed or `tensor.rng`
        random generator from tensor class

    References
    ----------
    [Dropout: A Simple Way to Prevent Neural Networks from Overfitting Srivastava, Hinton, et al. 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)

    Note
    ----
    This function only apply noise on Variable with TRAINING role
    """
    input_shape = get_shape(x)
    retain_prob = 1. - level
    # ====== not a training variable NO dropout ====== #
    if not is_training(x):
        return x
    if 'normal' in noise_type or 'gaussian' in noise_type:
        randfunc = lambda shape: random_normal(shape=shape, mean=1.0,
                                               std=np.sqrt((1.0 - retain_prob) / retain_prob),
                                               dtype=x.dtype)
    elif 'uniform' in noise_type:
        randfunc = lambda shape: random_binomial(shape=shape, p=retain_prob, dtype=x.dtype)
    else:
        raise ValueError('No support for noise_type=' + noise_type)
    # ====== Dropout ====== #
    shape = get_shape(x, native=True)
    if noise_dims is None:
        x = x * randfunc(shape=shape)
    else:
        noise_shape = _process_noise_dim(shape, noise_dims, ndim(x))
        # auto select broadcast shape
        broadcast = [i for i, j in enumerate(noise_shape) if j == 1]
        if len(broadcast) > 0:
            pattern = randfunc(shape=noise_shape)
            x = x * addbroadcast(pattern, *broadcast)
        else:
            x = x * randfunc(shape=noise_shape)
    if rescale:
        x /= retain_prob
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape)
    return x


def apply_noise(x, level=0.075, noise_dims=None, noise_type='gaussian'):
    """
    Parameters
    ----------
    x: A tensor.
    level : float or tensor scalar
        Standard deviation of added Gaussian noise
    noise_dims: int or list(int)
        these dimensions will be setted to 1 in noise_shape, and
        used to broadcast the dropout mask.
    noise_type: 'gaussian' (or 'normal'), 'uniform'
        distribution used for generating noise
    seed: random seed or `tensor.rng`
        random generator from tensor class

    Note
    ----
    This function only apply noise on Variable with TRAINING role
    """
    input_shape = get_shape(x)
    noise_type = noise_type.lower()
    # ====== not a training variable NO dropout ====== #
    if not is_training(x):
        return x
    # ====== applying noise ====== #
    shape = get_shape(x, native=True)
    noise_shape = (shape if noise_dims is None
                   else _process_noise_dim(shape, noise_dims, ndim(x)))
    if 'normal' in noise_type or 'gaussian' in noise_type:
        noise = random_normal(shape=noise_shape, mean=0.0, std=level, dtype=x.dtype)
    elif 'uniform' in noise_type:
        noise = random_uniform(shape=noise_shape, low=-level, high=level, dtype=x.dtype)
        # no idea why uniform does not give any broadcastable dimensions
        if noise_dims is not None:
            broadcastable = [i for i, j in enumerate(noise_shape) if j == 1]
            if len(broadcastable) > 0:
                noise = addbroadcast(noise, *broadcastable)
    else:
        raise ValueError('No support for noise_type=' + noise_type)
    x = x + noise
    if isinstance(input_shape, (tuple, list)):
        add_shape(x, input_shape)
    return x


# ===========================================================================
# helper function
# ===========================================================================
def L2(variables):
    l2 = constant(0., name='L2const')
    for v in as_tuple(variables):
        l2 = l2 + sum(square(v))
    return l2


def L1(variables):
    l1 = constant(0., name='L1const')
    for v in as_tuple(variables):
        l1 = l1 + sum(abs(v))
    return l1


def L2_normalize(variable, axis):
    norm = sqrt(sum(square(variable), axis=axis, keepdims=True))
    return variable / norm


def jacobian_regularize(hidden, params):
    """ Computes the jacobian of the hidden layer with respect to
    the input, reshapes are necessary for broadcasting the
    element-wise product on the right axis
    """
    hidden = hidden * (1 - hidden)
    L = expand_dims(hidden, 1) * expand_dims(params, 0)
    # Compute the jacobian and average over the number of samples/minibatch
    L = sum(pow(L, 2)) / hidden.shape[0]
    return mean(L)


def correntropy_regularize(x, sigma=1.):
    """
    Note
    ----
    origin implementation from seya:
    https://github.com/EderSantana/seya/blob/master/seya/regularizers.py
    Copyright (c) EderSantana
    """
    return -sum(mean(exp(x**2 / sigma), axis=0)) / sqrt(2 * np.pi * sigma)


def kl_gaussian(mu, logsigma,
                prior_mu=0., prior_logsigma=0.):
    """ KL-divergence between two gaussians.
    Useful for Variational AutoEncoders. Use this as an activation regularizer

    For taking kl_gaussian as variational regularization, you can take mean of
    the return matrix

    Parameters:
    -----------
    mean, logsigma: parameters of the input distributions
    prior_mean, prior_logsigma: paramaters of the desired distribution (note the
        log on logsigma)


    Return
    ------
    matrix: (n_samples, n_features)

    Note
    ----
    origin implementation from:
    https://github.com/Philip-Bachman/ICML-2015/blob/master/LogPDFs.py
    Copyright (c) Philip Bachman
    """
    gauss_klds = 0.5 * (2 * (prior_logsigma - logsigma) +
            (exp(2 * logsigma) / exp(2 * prior_logsigma)) +
            (pow((mu - prior_mu), 2.0) / exp(2 * prior_logsigma)) - 1.0)
    return gauss_klds
