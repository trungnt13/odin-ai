from __future__ import print_function, division, absolute_import

import os

from six import add_metaclass
from abc import ABCMeta, abstractmethod
from collections import Mapping

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from odin.utils import get_all_files, is_string, as_tuple, is_pickleable
from .signal import delta


@add_metaclass(ABCMeta)
class Extractor(BaseEstimator, TransformerMixin):
    """ Extractor

    The developer must override the `_transform` method.
     - Any returned features in form of `Mapping` (e.g. dictionary) will
       be stored with the new extracted features.
     - If the returned new features is not `Mapping`, all previous extracted
       features will be ignored, and only return the new features.

    """

    def __init__(self):
        super(Extractor, self).__init__()

    def fit(self, X, y=None):
        # Do nothing here
        return self

    @abstractmethod
    def _transform(self, X):
        raise NotImplementedError

    def transform(self, X):
        # NOTE: do not override this method
        y = self._transform(X)
        # ====== check returned types ====== #
        if not isinstance(y, (Mapping, None)):
            raise RuntimeError("Extractor can only return Mapping or None, but "
                               "the returned type is: %s" % str(type(y)))
        # ====== Merge previous results ====== #
        if isinstance(y, Mapping):
            # remove None values
            tmp = {}
            for name, feat in y.iteritems():
                if any(c.isupper() for c in name):
                    raise RuntimeError("name for features cannot contain "
                                       "upper case.")
                if feat is None:
                    continue
                tmp[name] = feat
            y = tmp
            # add old features extracted in X,
            # but do NOT override new features in y
            if isinstance(X, Mapping):
                for name, feat in X.iteritems():
                    if any(c.isupper() for c in name):
                        raise RuntimeError("name for features cannot contain "
                                           "upper case.")
                    if name not in y:
                        y[name] = feat
        return y


# ===========================================================================
# General extractor
# ===========================================================================
def _check_feat_name(feat_type, name):
    if feat_type is None:
        return True
    elif callable(feat_type):
        return bool(feat_type(name))
    return name in feat_type


class DeltaExtractor(Extractor):

    def __init__(self, width=9, order=1, axis=0, feat_type=None):
        super(DeltaExtractor, self).__init__()
        # ====== check width ====== #
        width = int(width)
        if width % 2 == 0 or width < 3:
            raise ValueError("`width` must be odd integer >= 3, give value: %d" % width)
        self.width = width
        # ====== check order ====== #
        self.order = as_tuple(order, t=int)
        # ====== axis ====== #
        self.axis = axis
        self.feat_type = feat_type

    def _transform(self, X):
        if isinstance(X, Mapping):
            max_order = max(self.order)
            for name, feat in X.items():
                if _check_feat_name(self.feat_type, name):
                    all_deltas = delta(data=feat, width=self.width,
                                       order=max_order, axis=self.axis)
                    if not isinstance(all_deltas, (tuple, list)):
                        all_deltas = (all_deltas,)
                    all_deltas = tuple([d for i, d in enumerate(all_deltas)
                                        if (i + 1) in self.order])
                    feat = np.concatenate((feat,) + all_deltas, axis=-1)
                X[name] = feat
        return X


class EqualizeShape0(Extractor):
    """ EqualizeShape0 """

    def __init__(self, feat_type):
        super(EqualizeShape0, self).__init__()
        if feat_type is None:
            pass
        elif callable(feat_type):
            if not is_pickleable(feat_type):
                raise ValueError("`feat_type` must be a pickle-able callable.")
        else:
            feat_type = tuple([f.lower() for f in as_tuple(feat_type, t=str)])
        self.feat_type = feat_type

    def _transform(self, X):
        if isinstance(X, Mapping):
            equalized = {}
            n = min(feat.shape[0]
                    for name, feat in X.iteritems()
                    if _check_feat_name(self.feat_type, name))
            # ====== equalize ====== #
            for name, feat in X.items():
                # cut the features in left and right
                # if the shape[0] is longer
                if _check_feat_name(self.feat_type, name) and feat.shape[0] != n:
                    diff = feat.shape[0] - n
                    diff_left = diff // 2
                    diff_right = diff - diff_left
                    feat = feat[diff_left:-diff_right]
                equalized[name] = feat
            X = equalized
        return X


class RunningStatistics(Extractor):
    """ Running statistics

    Parameters
    ----------
    feat_type: None or list of string
        list of features name will be used for calculating the
        running statistics.
        If None, calculate the statistics for all `numpy.ndarray`
    """

    def __init__(self, feat_type=None, axis=0, name=''):
        super(RunningStatistics, self).__init__()
        self.feat_type = None if feat_type is None else \
            as_tuple(feat_type, t=str)
        self.axis = axis
        self.name = str(name)

    def get_sum1_name(self, feat_name):
        return '%s_%ssum1' % (feat_name, self.name)

    def get_sum2_name(self, feat_name):
        return '%s_%ssum2' % (feat_name, self.name)

    def _transform(self, X):
        if isinstance(X, Mapping):
            # ====== preprocessing feat_type ====== #
            feat_type = self.feat_type
            if feat_type is None:
                feat_type = [name for name, feat in X.iteritems()
                             if isinstance(feat, np.ndarray) and feat.ndim >= 1]
            # ====== calculate the statistics ====== #
            for feat_name in feat_type:
                feat = X[feat_name]
                # ====== SUM of x^1 ====== #
                sum1 = np.sum(feat, axis=self.axis,
                              dtype='float64')
                s1_name = self.get_sum1_name(feat_name)
                if s1_name not in X:
                    X[s1_name] = sum1
                else:
                    X[s1_name] += sum1
                # ====== SUM of x^2 ====== #
                sum2 = np.sum(np.power(feat, 2), axis=self.axis,
                              dtype='float64')
                s2_name = self.get_sum2_name(feat_name)
                if s2_name not in X:
                    X[s2_name] = sum2
                else:
                    X[s2_name] += sum2
        return X
