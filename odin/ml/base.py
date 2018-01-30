from __future__ import print_function, division, absolute_import

from abc import ABCMeta, abstractmethod
from six import add_metaclass

import numpy as np

from odin.utils import ctext
from odin.visual import print_confusion

from sklearn.base import BaseEstimator, TransformerMixin, DensityMixin
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix

@add_metaclass(ABCMeta)
class Evaluable(object):
  """ Evaluable """

  def evaluate(self, X, y, labels=None, title='', path=None):
    from odin.backend import to_llr
    from odin.backend.metrics import (det_curve, compute_EER, roc_curve,
                                      compute_Cavg, compute_Cnorm,
                                      compute_minDCF)

    def format_score(s):
      return ctext('%.4f' % s, 'yellow')
    # ====== check inputs ====== #
    if labels is None:
      labels = self.labels
    if isinstance(y, (tuple, list)):
      y = np.array(y)
    # ====== evaluating ====== #
    y_pred_prob = self.predict_proba(X)
    y_pred = np.argmax(y_pred_prob, axis=-1)
    ll = log_loss(y_true=y, y_pred=y_pred_prob)
    acc = accuracy_score(y_true=y, y_pred=y_pred)
    cm = confusion_matrix(y_true=y, y_pred=y_pred)
    Pfa, Pmiss = det_curve(y_true=y, y_score=to_llr(y_pred_prob))
    eer = compute_EER(Pfa, Pmiss)
    minDCF = compute_minDCF(Pfa, Pmiss)[0]
    cnorm, cnorm_arr = compute_Cnorm(y_true=y,
                                     y_score=y_pred_prob,
                                     Ptrue=[1, 0.5],
                                     probability_input=True)
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
