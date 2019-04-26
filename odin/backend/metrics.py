from __future__ import print_function, division, absolute_import

import numpy as np
import tensorflow as tf

from odin.utils import is_number, as_tuple
from odin.autoconfig import EPS
from odin.backend.role import (AccuracyValue, return_roles, DifferentialLoss,
                               ConfusionMatrix, add_roles)
from odin.backend.tensor import argsort, dimshuffle, to_nonzeros, to_llr
from odin.backend.helpers import is_tensor

# ===========================================================================
# Losses
# ===========================================================================
@return_roles(AccuracyValue)
def binary_accuracy(y_true, y_pred, threshold=0.5, reduction=tf.reduce_mean,
                    name=None):
  """ Non-differentiable """
  with tf.name_scope(name, "binary_accuracy", [y_pred, y_true, threshold]):
    if y_pred.shape.ndims > 1:
      y_pred = tf.reshape(y_pred, (-1,))
    if y_true.shape.ndims > 1:
      y_true = tf.reshape(y_true, (-1,))
    y_pred = tf.greater_equal(y_pred, threshold)
    match_values = tf.cast(tf.equal(tf.cast(y_pred, 'int32'),
                                    tf.cast(y_true, 'int32')),
                           dtype='int32')
    return reduction(match_values)


@return_roles(AccuracyValue)
def categorical_accuracy(y_true, y_pred, top_k=1, reduction=tf.reduce_mean,
                         name=None):
  """ Non-differentiable """
  with tf.name_scope(name, "categorical_accuracy", [y_true, y_pred]):
    if y_true.shape.ndims == y_pred.shape.ndims:
      y_true = tf.argmax(y_true, axis=-1)
    elif y_true.shape.ndims != y_pred.shape.ndims - 1:
      raise TypeError('rank mismatch between y_true and y_pred')
    if top_k == 1:
      # standard categorical accuracy
      top = tf.argmax(y_pred, axis=-1)
      y_true = tf.cast(y_true, top.dtype.base_dtype)
      match_values = tf.equal(top, y_true)
    else:
      match_values = tf.nn.in_top_k(y_pred, tf.cast(y_true, 'int32'),
                                    k=top_k)
    match_values = tf.cast(match_values, dtype='float32')
    return reduction(match_values)


def confusion_matrix(y_true, y_pred, labels=None, normalize=False,
                     name=None):
  """
  Computes the confusion matrix of given vectors containing
  actual observations and predicted observations.

  Parameters
  ----------
  y_true : 1-d or 2-d tensor variable
      true values
  y_pred : 1-d or 2-d tensor variable
      prediction values
  normalize : bool
      if True, normalize each row to [0., 1.]
  labels : array, shape = [nb_classes], int (nb_classes)
      List of labels to index the matrix. This may be used to reorder
      or select a subset of labels.
      If none is given, those that appear at least once
      in ``y_true`` or ``y_pred`` are used in sorted order.

  Note
  ----
  if you want to calculate: Precision, Recall, F1 scores from the
  confusion matrix, set `normalize=False`

  """
  # ====== numpy ndarray ====== #
  if isinstance(y_true, np.ndarray) or isinstance(y_pred, np.ndarray):
    from sklearn.metrics import confusion_matrix as sk_cm
    nb_classes = None
    if y_true.ndim > 1:
      nb_classes = y_true.shape[1]
      y_true = np.argmax(y_true, axis=-1)
    if y_pred.ndim > 1:
      nb_classes = y_pred.shape[1]
      y_pred = np.argmax(y_pred, axis=-1)
    # get number of classes
    if labels is None:
      if nb_classes is None:
        raise RuntimeError("Cannot infer the number of classes for confusion matrix")
      labels = int(nb_classes)
    elif is_number(labels):
      labels = list(range(labels))
    cm = sk_cm(y_true=y_true, y_pred=y_pred, labels=labels)
    if normalize:
      cm = cm.astype('float32') / np.sum(cm, axis=1, keepdims=True)
    return cm
  # ====== tensorflow tensor ====== #
  with tf.name_scope(name, 'confusion_matrix', [y_true, y_pred]):
    from tensorflow.contrib.metrics import confusion_matrix as tf_cm
    nb_classes = None
    if y_true.shape.ndims == 2:
      nb_classes = y_true.shape.as_list()[-1]
      y_true = tf.argmax(y_true, -1)
    elif y_true.shape.ndims != 1:
      raise ValueError('actual must be 1-d or 2-d tensor variable')
    if y_pred.shape.ndims == 2:
      nb_classes = y_pred.shape.as_list()[-1]
      y_pred = tf.argmax(y_pred, -1)
    elif y_pred.shape.ndims != 1:
      raise ValueError('pred must be 1-d or 2-d tensor variable')
    # check valid labels
    if labels is None:
      if nb_classes is None:
        raise RuntimeError("Cannot infer the number of classes for confusion matrix")
      labels = int(nb_classes)
    elif is_number(labels):
      labels = int(labels)
    elif hasattr(labels, '__len__'):
      labels = len(labels)
    # transpose to match the format of sklearn
    cm = tf_cm(labels=y_true, predictions=y_pred,
               num_classes=labels)
    if normalize:
      cm = tf.cast(cm, dtype='float32')
      cm = cm / tf.reduce_sum(cm, axis=1, keep_dims=True)
    return add_roles(cm, ConfusionMatrix)


def detection_matrix(y_true, y_pred):
  pass

# ===========================================================================
# Speech task metrics
# ===========================================================================
def compute_Cavg(y_llr, y_true, cluster_idx=None,
                 Ptrue=0.5, Cfa=1., Cmiss=1.,
                 probability_input=False):
  ''' Fast calculation of Cavg (for only 1 clusters)

  Parameters
  ----------
  y_llr: (nb_samples, nb_classes)
      log likelihood ratio: llr = log (P(data|target) / P(data|non-target))
  y_true: numpy array of shape (nb_samples,)
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
  probability_input: boolean
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

  '''
  # ====== For tensorflow ====== #
  if is_tensor(y_llr) and is_tensor(y_true):
    if probability_input:
      y_llr = to_llr(y_llr)
    thresh = np.log(Cfa / Cmiss) - np.log(Ptrue / (1 - Ptrue))
    nb_classes = y_llr.shape[1].value
    if isinstance(y_true, (list, tuple)):
      y_true = np.asarray(y_true)
    if y_true.shape.ndims == 1:
      y_true = tf.one_hot(y_true, depth=nb_classes, axis=-1)
    y_true = tf.cast(y_true, y_llr.dtype.base_dtype)
    # ====== statistics ====== #
    # invert of y_true, False Negative mask
    y_false = 1. - y_true
    y_positive = tf.cast(tf.greater_equal(y_llr, thresh),
                         y_llr.dtype.base_dtype)
    # invert of y_positive
    y_negative = tf.cast(tf.less(y_llr, thresh), y_llr.dtype.base_dtype)
    distribution = tf.clip_by_value(
        tf.reduce_sum(y_true, axis=0), 10e-8, 10e8) # no zero values
    # ====== Pmiss ====== #
    miss = tf.reduce_sum(y_true * y_negative, axis=0)
    Pmiss = 100 * (Cmiss * Ptrue * miss) / distribution
    # ====== Pfa ====== # This calculation give different results
    fa = tf.reduce_sum(y_false * y_positive, axis=0)
    Pfa = 100 * (Cfa * (1 - Ptrue) * fa) / distribution
    Cavg = tf.reduce_mean(Pmiss) + tf.reduce_mean(Pfa) / (nb_classes - 1)
    return Cavg
  # ====== for numpy ====== #
  if probability_input:
    y_llr = to_llr(y_llr)
  if cluster_idx is None:
    cluster_idx = [list(range(0, y_llr.shape[-1]))]
  # ensure everything is numpy ndarray
  y_true = np.asarray(y_true)
  y_llr = np.asarray(y_llr)
  # threshold
  thresh = np.log(Cfa / Cmiss) - np.log(Ptrue / (1 - Ptrue))
  cluster_cost = np.zeros(len(cluster_idx))
  for k, cluster in enumerate(cluster_idx):
    L = len(cluster) # number of languages in a cluster
    fa = 0
    fr = 0
    for lang_i in cluster:
      N = np.sum(y_true == lang_i, dtype='float32') # number of samples for lang_i
      N = max(N, 1.) # prevent divide by 0, which give NaN return
      for lang_j in cluster:
        if lang_i == lang_j:
          err = np.sum(y_llr[y_true == lang_i, lang_i] < thresh) / N
          fr += err
        else:
          err = np.sum(y_llr[y_true == lang_i, lang_j] >= thresh) / N
          fa += err
    # Calculate procentage
    cluster_cost[k] = 100 * (Cmiss * Ptrue * fr + Cfa * (1 - Ptrue) * fa / (L - 1)) / L
  total_cost = np.mean(cluster_cost)
  return cluster_cost, total_cost


def compute_Cnorm(y_true, y_score,
                  Ptrue=[0.1, 0.5], Cfa=1., Cmiss=1.,
                  probability_input=False):
  """ Computes normalized detection cost function (DCF) given
    the costs for false accepts and false rejects as well as a priori
    probability for target speakers.

    * This is the actual cost, different from the min cost (minDCF)

    (By convention, the more positive the score,
    the more likely is the target hypothesis.)

  Parameter
  ---------
  y_true: {array [n_samples], or list of array}
      each array is labels of binary or multi-classes
      detection tasks, each array can be an array of
      classes indices, or one-hot-encoded matrix.
      If multiple array are given, calculating `equalized cost`
      of all partitions, an example of 2 partitions are:
      VAST and MLSR14 files
  y_score: {array [n_samples, n_classes], or list of array}
      the outputs scores, can be probabilities values or log-likelihood
      values by default, the
  Ptrue: float [0.,1.], or list of float
      hypothesized prior probabilities of positive class,
      you can given multiple values by providing an array
  Cfa: float
      weight for False Alarm - False Positive error
  Cmiss: float
      weight for Miss - False Negative error

  Return
  ------
  C_norm: array [len(Ptrue)]
      minimum detection cost accordingly for each given value of `Ptrue`.
  C_norm_array: array [len(Ptrue), n_classes]
      minimum detection cost for each class, accordingly to each
      given value of `Ptrue`

  """
  y_true = as_tuple(y_true, t=np.ndarray)
  y_score = as_tuple(y_score, t=np.ndarray)
  if len(y_true) != len(y_score):
    raise ValueError("There are %d partitions for `y_true`, but %d "
                     "partitions for `y_score`." % (len(y_true), len(y_score)))
  if len(set(i.shape[1] for i in y_score)) != 1:
    raise ValueError("The number of classes among scores array is inconsistent.")
  nb_partitions = len(y_true)
  # ====== preprocessing ====== #
  y_true = [np.argmax(i, axis=-1) if i.ndim >= 2 else i
            for i in y_true]
  nb_classes = y_score[0].shape[1]
  # threshold
  Ptrue = np.asarray(as_tuple(Ptrue), dtype=float)
  nb_threshold = len(Ptrue)
  # log(beta) is threshold, i.e.
  # if Ptrue=0.5 => beta=1. => threshold=0.
  beta = (Cfa / Cmiss) * ((1 - Ptrue) / Ptrue)
  beta = np.clip(beta, a_min=np.finfo(float).eps, a_max=np.inf)
  # ====== Cavg ====== #
  global_cm_array = np.zeros(shape=(nb_threshold, nb_classes, nb_classes))
  # Apply threshold on the scores and compute the confusion matrix
  for scores, labels in zip(y_score, y_true):
    actual_TP_per_class = np.lib.arraysetops.unique(
        ar=labels, return_counts=True)[1]
    if probability_input: # special case input is probability values
      scores = to_llr(scores)
    for theta_ix, theta in enumerate(np.log(beta)):
      thresholded_scores = (scores > theta).astype(int)
      # compute confusion matrix, this is different from
      # general implementation of confusion matrix above
      cm = np.zeros(shape=(nb_classes, nb_classes), dtype=np.int64)
      for i, (trial, target) in enumerate(zip(thresholded_scores, labels)):
        cm[target, :] += trial
      # miss and fa
      predic_TP_per_class = cm.diagonal()
      # Compute the number of miss per class
      nb_miss_per_class = actual_TP_per_class - predic_TP_per_class
      cm_miss_fa = cm
      cm_miss_fa[np.diag_indices_from(cm)] = nb_miss_per_class
      cm_probabilities = cm_miss_fa / actual_TP_per_class[:, None]
      # update global
      global_cm_array[theta_ix] += cm_probabilities
  # normalize by partitions
  global_cm_array /= nb_partitions
  # Extract probabilities of false negatives from confusion matrix
  p_miss_arr = global_cm_array.diagonal(0, 1, 2)
  p_miss = p_miss_arr.mean(1)
  # Extract probabilities of false positives from confusion matrix
  p_false_alarm_arr = (global_cm_array.sum(1) - p_miss_arr) / (nb_classes - 1)
  p_false_alarm = p_false_alarm_arr.mean(1)
  # Compute costs per languages
  C_Norm_arr = p_miss_arr + beta[:, None] * p_false_alarm_arr
  # Compute overall cost
  C_Norm = p_miss + beta * p_false_alarm
  return C_Norm, C_Norm_arr


def compute_minDCF(Pfa, Pmiss, Cmiss=1, Cfa=1, Ptrue=0.5):
  """ Estimating the min value of the  detection
  cost function (DCF)

  Parameters
  ----------
  Pfa: array, [n_samples]
      false alarm rate or false positive rate
  Pmiss: array, [n_samples]
      miss rate or false negative rate
  Cmiss: scalar
      weight for false positive mistakes
  Cfa: scalar
      weight for false negative mistakes
  Ptrue: scalar [0., 1.]
      prior probability of positive cases.

  Return
  ------
  min_DCF: scalar
      minimum value of the detection cost function for
      a given detection error trade-off curve
  Pfa_optimum: scalar
      and false alarm trade-off probabilities.
  Pmiss_optimum: scalar
      the correcponding miss

  """
  assert Pmiss.shape == Pfa.shape
  Pfalse = 1 - Ptrue
  # detection cost function vector
  DCF_vector = (Cmiss * Pmiss * Ptrue) + \
      (Cfa * Pfa * Pfalse)
  # get the optimal value and corresponding index
  min_idx = np.argmin(DCF_vector)
  min_val = DCF_vector[min_idx]
  return min_val, Pfa[min_idx], Pmiss[min_idx]


def compute_EER(Pfa, Pmiss):
  """ computes the equal error rate (EER) given
  Pmiss or False Negative Rate
  and
  Pfa or False Positive Rate
  calculated for a range of operating points on the DET curve

  @Author: "Timothee Kheyrkhah, Omid Sadjadi"
  """
  fpr, fnr = Pfa, Pmiss
  diff_pm_fa = fnr - fpr
  x1 = np.flatnonzero(diff_pm_fa >= 0)[0]
  x2 = np.flatnonzero(diff_pm_fa < 0)[-1]
  a = (fnr[x1] - fpr[x1]) / (fpr[x2] - fpr[x1] - (fnr[x2] - fnr[x1]))
  return fnr[x1] + a * (fnr[x2] - fnr[x1])


def compute_AUC(x, y, reorder=False):
  """Compute Area Under the Curve (AUC) using the trapezoidal rule

  This is a general function, given points on a curve.  For computing the
  area under the ROC-curve, see :func:`roc_auc_score`.  For an alternative
  way to summarize a precision-recall curve, see
  :func:`average_precision_score`.

  Parameters
  ----------
  x : array, shape = [n]
      x coordinates.
  y : array, shape = [n]
      y coordinates.
  reorder : boolean, optional (default=False)
      If True, assume that the curve is ascending in the case of ties, as for
      an ROC curve. If the curve is non-ascending, the result will be wrong.

  Returns
  -------
  auc : float

  Examples
  --------
  >>> import numpy as np
  >>> from sklearn import metrics
  >>> y = np.array([1, 1, 2, 2])
  >>> pred = np.array([0.1, 0.4, 0.35, 0.8])
  >>> fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
  >>> metrics.auc(fpr, tpr)
  0.75

  """
  from sklearn.metrics import auc
  return auc(x, y, reorder)


def roc_curve(y_true, y_score, pos_label=None, sample_weight=None,
              drop_intermediate=True):
  """Compute Receiver operating characteristic (ROC)

  @copy from sklearn for convenience

  Note: this implementation is restricted to the binary classification task.

  Parameters
  ----------

  y_true : array, shape = [n_samples]
      True binary labels in range {0, 1} or {-1, 1}.  If labels are not
      binary, pos_label should be explicitly given.
  y_score : array, shape = [n_samples]
      Target scores, can either be probability estimates of the positive
      class, confidence values, or non-thresholded measure of decisions
      (as returned by "decision_function" on some classifiers).
  pos_label : int or str, default=None
      Label considered as positive and others are considered negative.
  sample_weight : array-like of shape = [n_samples], optional
      Sample weights.
  drop_intermediate : boolean, optional (default=True)
      Whether to drop some suboptimal thresholds which would not appear
      on a plotted ROC curve. This is useful in order to create lighter
      ROC curves.

  Returns
  -------
  fpr : array, shape = [>2]
      Increasing false positive rates such that element i is the false
      positive rate of predictions with score >= thresholds[i].
  tpr : array, shape = [>2]
      Increasing true positive rates such that element i is the true
      positive rate of predictions with score >= thresholds[i].
  thresholds : array, shape = [n_thresholds]
      Decreasing thresholds on the decision function used to compute
      fpr and tpr. `thresholds[0]` represents no instances being predicted
      and is arbitrarily set to `max(y_score) + 1`.

  Notes
  -----
  Since the thresholds are sorted from low to high values, they
  are reversed upon returning them to ensure they correspond to both ``fpr``
  and ``tpr``, which are sorted in reversed order during their calculation.

  References
  ----------
  .. [1] `Wikipedia entry for the Receiver operating characteristic
          <https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_

  Examples
  --------
  >>> import numpy as np
  >>> from sklearn import metrics
  >>> y = np.array([1, 1, 2, 2])
  >>> scores = np.array([0.1, 0.4, 0.35, 0.8])
  >>> fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
  >>> fpr
  array([ 0. ,  0.5,  0.5,  1. ])
  >>> tpr
  array([ 0.5,  0.5,  1. ,  1. ])
  >>> thresholds
  array([ 0.8 ,  0.4 ,  0.35,  0.1 ])

  """
  from sklearn.metrics import roc_curve
  return roc_curve(y_true, y_score, pos_label,
                   sample_weight, drop_intermediate)


def prc_curve(y_true, y_probas, pos_label=None,
              sample_weight=None):
  """Compute precision-recall pairs for different probability thresholds

  Note: this implementation is restricted to the binary classification task.

  The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
  true positives and ``fp`` the number of false positives. The precision is
  intuitively the ability of the classifier not to label as positive a sample
  that is negative.

  The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
  true positives and ``fn`` the number of false negatives. The recall is
  intuitively the ability of the classifier to find all the positive samples.

  The last precision and recall values are 1. and 0. respectively and do not
  have a corresponding threshold.  This ensures that the graph starts on the
  x axis.

  Read more in the :ref:`User Guide <precision_recall_f_measure_metrics>`.

  Parameters
  ----------
  y_true : array, shape = [n_samples]
      True targets of binary classification in range {-1, 1} or {0, 1}.
  y_probas : array, shape = [n_samples]
      Estimated probabilities or decision function.
  pos_label : int or str, default=None
      The label of the positive class
  sample_weight : array-like of shape = [n_samples], optional
      Sample weights.

  Returns
  -------
  precision : array, shape = [n_thresholds + 1]
      Precision values such that element i is the precision of
      predictions with score >= thresholds[i] and the last element is 1.
  recall : array, shape = [n_thresholds + 1]
      Decreasing recall values such that element i is the recall of
      predictions with score >= thresholds[i] and the last element is 0.
  thresholds : array, shape = [n_thresholds <= len(np.unique(probas_pred))]
      Increasing thresholds on the decision function used to compute
      precision and recall.

  Examples
  --------
  >>> import numpy as np
  >>> from sklearn.metrics import precision_recall_curve
  >>> y_true = np.array([0, 0, 1, 1])
  >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
  >>> precision, recall, thresholds = precision_recall_curve(
  ...     y_true, y_scores)
  >>> precision  # doctest: +ELLIPSIS
  array([ 0.66...,  0.5       ,  1.        ,  1.        ])
  >>> recall
  array([ 1. ,  0.5,  0.5,  0. ])
  >>> thresholds
  array([ 0.35,  0.4 ,  0.8 ])

  """
  from sklearn.metrics import precision_recall_curve
  return precision_recall_curve(y_true, y_probas,
                                pos_label, sample_weight)


def det_curve(y_true, y_score, pos_label=None, sample_weight=None):
  """Detection Error Tradeoff
  Compute error rates for different probability thresholds

  @Original implementaion from NIST
  The function is adapted to take input format same as
  NIST original code and `sklearn.metrics`

  Note: this implementation is restricted to the binary classification task.
  (By convention, the more positive the score,
  the more likely is the target hypothesis.)

  Parameters
  ----------
  y_true : array, shape = [n_samples]
      True targets of binary classification in range {-1, 1} or {0, 1}.
  y_score : array, shape = [n_samples]
      Estimated probabilities or decision function.
  pos_label : int, optional (default=None)
      The label of the positive class
  sample_weight : array-like of shape = [n_samples], optional
      Sample weights.

  Returns
  -------
  with `n_samples = n_true_samples + n_false_samples`
  P_fa: array, shape = [n_samples]
      fpr - False Positive rate, or false alarm probabilities
  P_miss : array, shape = [n_samples]
      fnr - False Negative rate, or miss probabilities

  References
  ----------
  .. [1] `Wikipedia entry for Detection error tradeoff
          <https://en.wikipedia.org/wiki/Detection_error_tradeoff>`_
  .. [2] `The DET Curve in Assessment of Detection Task Performance
          <http://www.itl.nist.gov/iad/mig/publications/storage_paper/det.pdf>`_
  .. [3] `2008 NIST Speaker Recognition Evaluation Results
          <http://www.itl.nist.gov/iad/mig/tests/sre/2008/official_results/>`_
  .. [4] `DET-Curve Plotting software for use with MATLAB
          <http://www.itl.nist.gov/iad/mig/tools/DETware_v2.1.targz.htm>`_

  Examples
  --------
  >>> import numpy as np
  >>> from odin import backend as K
  >>> y_true = np.array([0, 0, 1, 1])
  >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
  >>> fnr, fpr = K.metrics.det_curve(y_true, y_scores)
  >>> print(fpr)
  array([ 0.5,  0.5,  0. ])
  >>> print(fnr)
  array([ 0. ,  0.5,  0.5])
  >>> print(thresholds)
  array([ 0.35,  0.4 ,  0.8 ])
  """
  # ====== ravel everything in cased of multi-classes ====== #
  y_score = y_score.ravel()
  y_true = np.array(y_true)
  if y_true.ndim >= 2:
    y_true = np.argmax(y_true, axis=-1)
  nb_classes = len(np.lib.arraysetops.unique(y_true))
  # multi-classes
  if nb_classes > 2:
    total_samples = nb_classes * len(y_true)
    indices = np.arange(0, total_samples, nb_classes) + y_true
    y_true = np.zeros(total_samples, dtype=np.int)
    y_true[indices] = 1
  # ====== check weights ====== #
  if sample_weight is not None:
    if len(sample_weight) != len(y_score):
      raise ValueError("Provided `sample_weight` for %d samples, but got "
                       "scores for %d samples." %
                       (len(sample_weight), len(y_score)))
  else:
    sample_weight = np.ones(shape=(len(y_score),), dtype=y_score.dtype)
  # ====== processing ====== #
  if pos_label is not None:
    y_true = (y_true == pos_label).astype(np.int)
  # ====== start ====== #
  sorted_ndx = np.argsort(y_score)
  y_true = y_true[sorted_ndx]
  # sort the weights also, dont forget this
  sample_weight = sample_weight[sorted_ndx]
  tgt_weights = sample_weight * y_true
  imp_weights = sample_weight * (1 - y_true)
  # FNR
  Pmiss = np.cumsum(tgt_weights) / np.sum(tgt_weights)
  # FPR
  Pfa = 1 - np.cumsum(imp_weights) / np.sum(imp_weights)
  return Pfa, Pmiss
