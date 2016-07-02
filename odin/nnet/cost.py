from __future__ import division, absolute_import

import numpy as np

from blocks import backend as K
from blocks import autoconfig


# ===========================================================================
# Differentiable
# ===========================================================================
def categorical_crossentropy(output, target, mean=True):
    # avoid numerical instability with EPSILON clipping
    output = K.clip(output, autoconfig.epsilon, 1.0 - autoconfig.epsilon)
    if mean:
        return K.mean(K.categorical_crossentropy(output, target))
    return K.categorical_crossentropy(output, target)


def binary_crossentropy(output, target, mean=True):
    # avoid numerical instability with EPSILON clipping
    output = K.clip(output, autoconfig.epsilon, 1.0 - autoconfig.epsilon)
    if mean:
        return K.mean(K.binary_crossentropy(output, target))
    return K.binary_crossentropy(output, target)


# ===========================================================================
# Not differentiable
# ===========================================================================
def binary_accuracy(y_pred, y_true, threshold=0.5):
    """ Non-differentiable """
    y_pred = K.ge(y_pred, threshold)
    return K.eq(K.cast(y_pred, 'int32'),
                K.cast(y_true, 'int32'))


def categorical_accuracy(y_pred, y_true, top_k=1, mean=True):
    """ Non-differentiable """
    if K.ndim(y_true) == K.ndim(y_pred):
        y_true = K.argmax(y_true, axis=-1)
    elif K.ndim(y_true) != K.ndim(y_pred) - 1:
        raise TypeError('rank mismatch between y_true and y_pred')

    if top_k == 1:
        # standard categorical accuracy
        top = K.argmax(y_pred, axis=-1)
        accuracy = K.eq(top, y_true)
    else:
        # top-k accuracy
        top = K.argtop_k(y_pred, top_k)
        y_true = K.expand_dims(y_true, dim=-1)
        accuracy = K.any(K.eq(top, y_true), axis=-1)

    if mean:
        return K.mean(accuracy)
    return accuracy


# ==================== Regularizations ==================== #
def l2_normalize(x, axis):
    norm = K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True))
    return x / norm


def l2_regularize(x):
    return K.sum(K.square(x))


def l1_regularize(x):
    return K.sum(K.abs(x))


def jacobian_regularize(hidden, params):
    """ Computes the jacobian of the hidden layer with respect to
    the input, reshapes are necessary for broadcasting the
    element-wise product on the right axis
    """
    hidden = hidden * (1 - hidden)
    L = K.expand_dims(hidden, 1) * K.expand_dims(params, 0)
    # Compute the jacobian and average over the number of samples/minibatch
    L = K.sum(K.pow(L, 2)) / hidden.shape[0]
    return K.mean(L)


def correntropy_regularize(x, sigma=1.):
    """
    Note
    ----
    origin implementation from seya:
    https://github.com/EderSantana/seya/blob/master/seya/regularizers.py
    Copyright (c) EderSantana
    """
    return -K.sum(K.mean(K.exp(x**2 / sigma), axis=0)) / K.sqrt(2 * np.pi * sigma)


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
            (K.exp(2 * logsigma) / K.exp(2 * prior_logsigma)) +
            (K.pow((mu - prior_mu), 2.0) / K.exp(2 * prior_logsigma)) - 1.0)
    return gauss_klds
