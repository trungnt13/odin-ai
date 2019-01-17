from __future__ import print_function, division, absolute_import

from abc import ABCMeta, abstractmethod, abstractproperty
from six import add_metaclass

import numpy as np

from odin.fuel import Data
from odin.utils import ctext, is_number, one_hot
from odin.visual import (print_confusion, plot_detection_curve,
                         plot_confusion_matrix, plot_save, figure,
                         plot_Cnorm)

from sklearn.base import (BaseEstimator, TransformerMixin, DensityMixin,
                          ClassifierMixin, RegressorMixin)
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

def evaluate(y_true, y_pred_proba=None, y_pred_log_proba=None,
             labels=None, title='', path=None,
             xlims=None, ylims=None, print_log=True):
  from odin.backend import to_llr
  from odin.backend.metrics import (det_curve, compute_EER, roc_curve,
                                    compute_Cavg, compute_Cnorm,
                                    compute_minDCF)

  def format_score(s):
    return ctext('%.4f' % s if is_number(s) else s, 'yellow')
  nb_classes = None
  # ====== check y_pred ====== #
  if y_pred_proba is None and y_pred_log_proba is None:
    raise ValueError("At least one of `y_pred_proba` or `y_pred_log_proba` "
                     "must not be None")
  y_pred_llr = to_llr(y_pred_proba) if y_pred_log_proba is None \
      else to_llr(y_pred_log_proba)
  nb_classes = y_pred_llr.shape[1]
  y_pred = np.argmax(y_pred_llr, axis=-1)
  # ====== check y_true ====== #
  if isinstance(y_true, Data):
    y_true = y_true.array
  if isinstance(y_true, (tuple, list)):
    y_true = np.array(y_true)
  if y_true.ndim == 2: # convert one-hot to labels
    y_true = np.argmax(y_true, axis=-1)
  # ====== check labels ====== #
  if labels is None:
    labels = [str(i) for i in range(nb_classes)]
  # ====== scoring ====== #
  if y_pred_proba is None:
    ll = 'unknown'
  else:
    ll = log_loss(y_true=y_true, y_pred=y_pred_proba)
  acc = accuracy_score(y_true=y_true, y_pred=y_pred)
  cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
  # C_norm
  cnorm, cnorm_arr = compute_Cnorm(y_true=y_true,
                                   y_score=y_pred_llr,
                                   Ptrue=[0.1, 0.5],
                                   probability_input=False)
  if y_pred_log_proba is not None:
    cnorm_, cnorm_arr_ = compute_Cnorm(y_true=y_true,
                                       y_score=y_pred_log_proba,
                                       Ptrue=[0.1, 0.5],
                                       probability_input=False)
    if np.mean(cnorm) > np.mean(cnorm_): # smaller is better
      cnorm, cnorm_arr = cnorm_, cnorm_arr_
  # DET
  Pfa, Pmiss = det_curve(y_true=y_true, y_score=y_pred_llr)
  eer = compute_EER(Pfa=Pfa, Pmiss=Pmiss)
  minDCF = compute_minDCF(Pfa, Pmiss)[0]
  # PRINT LOG
  if print_log:
    print(ctext("--------", 'red'), ctext(title, 'cyan'))
    print("Log loss :", format_score(ll))
    print("Accuracy :", format_score(acc))
    print("C_norm   :", format_score(np.mean(cnorm)))
    print("EER      :", format_score(eer))
    print("minDCF   :", format_score(minDCF))
    print(print_confusion(arr=cm, labels=labels))
  # ====== save report to PDF files if necessary ====== #
  if path is not None:
    if y_pred_proba is None:
      y_pred_proba = y_pred_llr
    from matplotlib import pyplot as plt
    plt.figure(figsize=(nb_classes, nb_classes + 1))
    plot_confusion_matrix(cm, labels)
    # Cavg
    plt.figure(figsize=(nb_classes + 1, 3))
    plot_Cnorm(cnorm=cnorm_arr, labels=labels, Ptrue=[0.1, 0.5],
               fontsize=14)
    # binary classification
    if nb_classes == 2 and \
    (y_pred_proba.ndim == 1 or (y_pred_proba.ndim == 2 and
                                y_pred_proba.shape[1] == 1)):
      fpr, tpr = roc_curve(y_true=y_true, y_score=y_pred_proba.ravel())
      # det curve
      plt.figure()
      plot_detection_curve(Pfa, Pmiss, curve='det',
                           xlims=xlims, ylims=ylims, linewidth=1.2)
      # roc curve
      plt.figure()
      plot_detection_curve(fpr, tpr, curve='roc')
    # multiclasses
    else:
      y_true = one_hot(y_true, nb_classes=nb_classes)
      fpr_micro, tpr_micro, _ = roc_curve(y_true=y_true.ravel(),
                                          y_score=y_pred_proba.ravel())
      Pfa_micro, Pmiss_micro = Pfa, Pmiss
      fpr, tpr = [], []
      Pfa, Pmiss = [], []
      for i, yi in enumerate(y_true.T):
        curve = roc_curve(y_true=yi, y_score=y_pred_proba[:, i])
        fpr.append(curve[0])
        tpr.append(curve[1])
        curve = det_curve(y_true=yi, y_score=y_pred_llr[:, i])
        Pfa.append(curve[0])
        Pmiss.append(curve[1])
      plt.figure()
      plot_detection_curve(fpr_micro, tpr_micro, curve='roc',
                           linewidth=1.2, title="ROC Micro")
      plt.figure()
      plot_detection_curve(fpr, tpr, curve='roc',
                           labels=labels, linewidth=1.0,
                           title="ROC for each classes")
      plt.figure()
      plot_detection_curve(Pfa_micro, Pmiss_micro, curve='det',
                           xlims=xlims, ylims=ylims, linewidth=1.2,
                           title="DET Micro")
      plt.figure()
      plot_detection_curve(Pfa, Pmiss, curve='det',
                           xlims=xlims, ylims=ylims,
                           labels=labels, linewidth=1.0,
                           title="DET for each classes")
    plot_save(path)

@add_metaclass(ABCMeta)
class Evaluable(object):
  """ Evaluable """

  @abstractproperty
  def labels(self):
    raise NotImplementedError

  def evaluate(self, X, y, labels=None, title='', path=None,
               xlims=None, ylims=None, print_log=True):
    from odin.backend import to_llr
    # ====== check inputs ====== #
    if labels is None:
      labels = self.labels
    if isinstance(y, Data):
      y = y.array
    if isinstance(y, (tuple, list)):
      y = np.array(y)
    if y.ndim == 2: # convert one-hot to labels
      y = np.argmax(y, axis=-1)
    # ====== proba ====== #
    if hasattr(self, 'predict_proba'):
      y_pred_prob = self.predict_proba(X)
    else:
      y_pred_prob = None
    # ====== log proba ====== #
    if hasattr(self, 'predict_log_proba'):
      y_pred_log_prob = self.predict_log_proba(X)
    elif y_pred_prob is not None:
      y_pred_log_prob = to_llr(y_pred_prob)
    else:
      raise ValueError('Class "%s" must has: `predict_proba` or `predict_log_proba`'
                       ' method.' % self.__class__.__name__)
    evaluate(y_true=y, y_pred_proba=y_pred_prob, y_pred_log_proba=y_pred_log_prob,
             labels=labels, title=title, path=path, xlims=xlims, ylims=ylims,
             print_log=print_log)
    return self
