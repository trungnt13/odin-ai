# ===========================================================================
# This module contains non-differentiable cost for monitor
# ===========================================================================
from __future__ import print_function, division, absolute_import

import numpy as np

from odin.config import autoconfig

if autoconfig['backend'] == 'theano':
    from .theano import (ge, eq, lt, cast, ndim, argmax, any,
                         argtop_k, expand_dims, mean, one_hot,
                         clip, switch, sum)
elif autoconfig['backend'] == 'tensorflow':
    from .tensorflow import (ge, eq, cast)

__all__ = [
    'LevenshteinDistance',
    'LER',
    'Cavg_fast',
    'Cavg',
    'categorical_accuracy',
    'mean_categorical_accuracy',
    'binary_accuracy'
]


# ===========================================================================
# Main metrics
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


# ===========================================================================
# Not differentiable
# ===========================================================================
def binary_accuracy(y_pred, y_true, threshold=0.5):
    """ Non-differentiable """
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


def mean_categorical_accuracy(y_pred, y_true, top_k=1):
    return mean(categorical_accuracy(y_pred, y_true, top_k))


def Cavg_fast(y_llr, y_true, Ptar=0.5, Cfa=1., Cmiss=1.):
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


# ===========================================================================
# Cavg
# ===========================================================================
def Cavg(log_llh, y, cluster_idx=None,
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
    y = np.asarray(y)
    log_llh = np.asarray(log_llh)

    thresh = np.log(Cfa / Cmiss) - np.log(Ptar / (1 - Ptar))
    cluster_cost = np.zeros(len(cluster_idx))

    for k, cluster in enumerate(cluster_idx):
        L = len(cluster) # number of languages in a cluster
        fa = 0
        fr = 0
        for lang_i in cluster:
            N = np.sum(y == lang_i) + .0 # number of samples for lang_i
            for lang_j in cluster:
                if lang_i == lang_j:
                    err = np.sum(log_llh[y == lang_i, lang_i] < thresh) / N
                    fr += err
                else:
                    err = np.sum(log_llh[y == lang_i, lang_j] >= thresh) / N
                    fa += err

        # Calculate procentage
        cluster_cost[k] = 100 * (Cmiss * Ptar * fr + Cfa * (1 - Ptar) * fa / (L - 1)) / L

    total_cost = np.mean(cluster_cost)

    return cluster_cost, total_cost
