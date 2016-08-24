from __future__ import print_function, division, absolute_import

import cPickle
import types
import inspect
import warnings
from functools import wraps
from numbers import Number
from abc import ABCMeta, abstractmethod
from six import add_metaclass

import numpy as np

from odin.config import auto_config, RNG_GENERATOR
from odin.roles import add_role, add_updates

config = auto_config()
FLOATX = config.floatX
EPSILON = config.epsilon

# store default operators
_any = any

if config['backend'] == 'theano':
    from .theano import *
elif config['backend'] == 'tensorflow':
    from .tensorflow import *

from . import init
from . import optimizers


def pickling_variable(v, target=None):
    """ This function only apply for trainable parameters
    """
    if isinstance(v, str):
        value, dtype, name, roles = cPickle.loads(v)
        v = variable(value, dtype=dtype, name=name, target=target)
        for i in roles:
            add_role(v, i)
        return v
    elif is_trainable_variable(v):
        obj = [get_value(v, borrow=False), v.dtype, v.name,
               getattr(v.tag, 'roles', [])]
        return cPickle.dumps(obj, protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        raise Exception('Variable must be in string form or trainable variable'
                        ' (i.e. SharedVariable in theano)')


# ===========================================================================
# Graph creator helper
# ===========================================================================
def function(inputs, outputs, updates=[]):
    return Function(inputs, outputs, updates=updates)


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
    go_backwards : bool
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
    recurrent_apply : :class:`~blocks.bricks.base.Application`
        The new application method that applies the RNN to sequences.

    Note
    --------
    sub-parameters is the addition parameters that the step funciton will
    accept
    The arguments inputed directly to the function will override the funciton
    in container object (i.e. the firs argument of class methdo)

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
        if name in kwargs:
            return kwargs[name]
        return getattr(container, name, None)

    def find_attr(name, type, container, kwargs, default):
        # find attribute with given name in kwargs and container,
        # given the type must match given type.
        name = str(name)
        val = default
        if name in kwargs:
            val = kwargs[name]
        elif hasattr(container, name):
            val = getattr(container, name)
        try:
            return type(val)
        except:
            return default
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
    if _any(not isinstance(i, str) for i in sequences + states):
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
            go_backwards = find_attr('go_backwards', types.BooleanType,
                container, kwargs, False)
            n_steps = find_attr('n_steps', types.IntType,
                container, kwargs, None)
            batch_size = find_attr('batch_size', types.IntType,
                container, kwargs, None)
            repeat_states = find_attr('repeat_states', types.BooleanType,
                container, kwargs, False)
            name = find_attr('name', types.StringType,
                container, kwargs, None)
            # ====== Update the positional arguments ====== #
            step_args = dict(defaults_args)
            step_args.update(kwargs)
            for key, value in zip(arg_spec.args, args): # key -> positional_args
                step_args[key] = value
            # ====== looking for all variables ====== #
            sequences_given = [find_arg(i, 'sequences', container, step_args)
                               for i in sequences]
            states_given = [find_arg(i, 'states', container, step_args)
                            for i in states]
            # check all is variables
            if _any(not is_variable(i) and i is not None
                   for i in sequences_given + states_given):
                raise ValueError('All variables provided to sequences, '
                                 'contexts, or states must be Variables.')
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
            print('Sequences:', sequences_given)
            print('States:', states_given)
            print('Gobackward:', go_backwards)
            print('NSteps:', n_steps)
            print('BatchSize:', batch_size)
            print('Repeat:', repeat_states)
            print('Name:', name)
            results, updates = Scan(
                scan_function,
                sequences=[i for i in sequences_given if i is not None],
                outputs_info=states_given,
                n_steps=n_steps,
                go_backwards=go_backwards,
                name=name)
            results = to_list(results)
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


def Cavg_gpu(y_llr, y_true, Ptar=0.5, Cfa=1., Cmiss=1.):
    ''' Fast calculation of Cavg (for only 1 clusters) '''
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
         Ptar=0.5, Cfa=1, Cmiss=1):
    """Compute cluster-wise and total LRE'15 percentage costs.

   Args:
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
   Returns:
       cluster_cost: numpy array of shape (n_clusters,)
           It contains average percentage costs for each cluster as defined by
           NIST LRE-15 language detection task. See
           http://www.nist.gov/itl/iad/mig/upload/LRE15_EvalPlan_v22-3.pdf
       total_cost: float
           An average percentage cost over all clusters.
   """
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
# helper function
# ===========================================================================
def L2(*variables):
    l2 = constant(0., name='L2const')
    for v in variables:
        l2 = l2 + sum(square(v))
    return l2


def L1(*variables):
    l1 = constant(0., name='L1const')
    for v in variables:
        l1 = l1 + sum(abs(v))
    return l1


def L2_normalize(variable, axis):
    norm = sqrt(sum(square(variable), axis=axis, keepdims=True))
    return x / norm


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
