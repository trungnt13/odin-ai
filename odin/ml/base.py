from __future__ import print_function, division, absolute_import

from abc import ABCMeta, abstractmethod, abstractproperty
from six import add_metaclass

import numpy as np

from odin.utils import ctext, is_number
from odin.visual import print_confusion

from sklearn.base import (BaseEstimator, TransformerMixin, DensityMixin,
                          ClassifierMixin, RegressorMixin)
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

@add_metaclass(ABCMeta)
class Evaluable(object):
  """ Evaluable """

  @abstractproperty
  def labels(self):
    raise NotImplementedError

  def evaluate(self, X, y, labels=None, title='', path=None):
    from odin.backend import to_llr
    from odin.backend.metrics import (det_curve, compute_EER, roc_curve,
                                      compute_Cavg, compute_Cnorm,
                                      compute_minDCF)

    def format_score(s):
      return ctext('%.4f' % s if is_number(s) else s, 'yellow')
    # ====== check inputs ====== #
    if labels is None:
      labels = self.labels
    if isinstance(y, (tuple, list)):
      y = np.array(y)
    # ====== prediction ====== #
    if hasattr(self, 'predict_proba'):
      y_pred_prob = self.predict_proba(X)
      y_pred_log_prob = to_llr(y_pred_prob)
      y_pred = np.argmax(y_pred_prob, axis=-1)
    elif hasattr(self, 'predict_log_proba'):
      y_pred_prob = None
      y_pred_log_prob = self.predict_log_proba(X)
      y_pred = np.argmax(y_pred_log_prob, axis=-1)
    else:
      raise ValueError('Class "%s" must has: `predict_proba` or `predict_log_proba`'
                       ' method.' % self.__class__.__name__)
    # ====== scoring ====== #
    if y_pred_prob is None:
      ll = 'unknown'
    else:
      ll = log_loss(y_true=y, y_pred=y_pred_prob)
    acc = accuracy_score(y_true=y, y_pred=y_pred)
    cm = confusion_matrix(y_true=y, y_pred=y_pred)
    Pfa, Pmiss = det_curve(y_true=y, y_score=y_pred_log_prob)
    eer = compute_EER(Pfa, Pmiss)
    minDCF = compute_minDCF(Pfa, Pmiss)[0]
    cnorm, cnorm_arr = compute_Cnorm(y_true=y,
                                     y_score=y_pred_log_prob,
                                     Ptrue=[1, 0.5],
                                     probability_input=False)
    if path is None:
      print(ctext("--------", 'red'), ctext(title, 'cyan'))
      print("Log loss :", format_score(ll))
      print("Accuracy :", format_score(acc))
      print("C_avg   :", format_score(np.mean(cnorm)))
      print("EER      :", format_score(eer))
      print("minDCF   :", format_score(minDCF))
      print(print_confusion(arr=cm, labels=labels))
    else:
      fpr, tpr = roc_curve(y_true=y, y_score=y_pred_prob)
