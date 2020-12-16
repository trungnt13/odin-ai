import inspect
from typing import Optional, Union, Any
from warnings import warn

import numpy as np
from typing_extensions import Literal
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

__all__ = [
    'fast_gbtree_classifier',
    'fast_rf_classifier',
]

Objectives = Literal['reg:squarederror', 'reg:squaredlogerror', 'reg:logistic',
                     'reg:pseudohubererror', 'binary:logistic',
                     'binary:logitraw', 'binary:hinge', 'count:poisson',
                     'survival:cox', 'survival:aft', 'aft_loss_distribution',
                     'multi:softmax', 'multi:softprob', 'rank:pairwise',
                     'rank:ndcg', 'rank:map', 'reg:gamma', 'reg:tweedie']


def fast_gbtree_classifier(
    X,
    y,
    *,
    learning_rate: float = 1.0,
    n_estimators: int = 100,
    subsample: float = 0.8,
    max_depth: Optional[int] = None,
    reg_alpha: Optional[float] = None,  # L1
    reg_lambda: Optional[float] = 1e-05,  # L2
    gamma: Optional[float] = None,
    missing: Optional[Any] = np.nan,
    objective: Objectives = 'binary:logistic',
    grow_policy: Literal['depthwise', 'lossguide'] = 'depthwise',
    tree_method: Literal['auto', 'exact', 'approx', 'hist',
                         'gpu_hist'] = 'auto',
    importance_type: Literal['gain', 'weight', 'cover', 'total_gain',
                             'total_cover'] = 'gain',
    random_state: int = 1,
    n_jobs: Optional[int] = None,
    framework: Literal['auto', 'xgboost', 'sklearn'] = 'auto',
    **kwargs,
) -> GradientBoostingClassifier:
  """Shared interface for XGBoost and sklearn Gradient Boosting Tree Classifier"""
  kw = dict(locals())
  kwargs = kw.pop('kwargs')
  X = kw.pop('X')
  y = kw.pop('y')
  kw.update(kwargs)
  framework = kw.pop('framework')
  ### XGBOOST
  is_xgboost = False
  if framework == 'sklearn':
    XGB = GradientBoostingClassifier
  else:
    try:
      from xgboost import XGBRFClassifier as XGB
      is_xgboost = True
    except ImportError as e:
      warn('Run `pip install xgboost` to get significant '
           'faster GradientBoostingTree')
      XGB = GradientBoostingClassifier
  ### fine-tune the keywords for sklearn
  if not is_xgboost:
    org = dict(kw)
    spec = inspect.getfullargspec(XGB.__init__)
    kw = dict()
    for k in spec.args + spec.kwonlyargs:
      if k in org:
        kw[k] = org[k]
  ### training
  tree = XGB(**kw)
  tree.fit(X, y)
  return tree


def fast_rf_classifier(
    X,
    y,
    *,
    num_classes=2,
    split_algo=1,
    split_criterion=0,
    min_rows_per_node=2,
    min_impurity_decrease=0.0,
    bootstrap_features=False,
    rows_sample=1.0,
    max_leaves=-1,
    n_estimators=100,
    max_depth=16,
    max_features='auto',
    bootstrap=True,
    n_bins=8,
    n_cols=None,
    dtype=None,
    accuracy_metric=None,
    quantile_per_tree=False,
    n_streams=8,
    random_state: int = 1,
    n_jobs: Optional[int] = None,
    framework: Literal['auto', 'cuml', 'sklearn'] = 'auto',
    **kwargs,
):
  kw = dict(locals())
  kwargs = kw.pop('kwargs')
  X = kw.pop('X')
  y = kw.pop('y')
  kw.update(kwargs)
  framework = kw.pop('framework')
  ### import
  is_cuml = False
  if framework == 'sklearn':
    RFC = RandomForestClassifier
  else:
    try:
      from cuml.ensemble import RandomForestClassifier as RFC
      is_cuml = True
    except ImportError as e:
      RFC = RandomForestClassifier
  ### fine-tune keywords
  if is_cuml:
    kw['output_type'] = 'numpy'
    kw['seed'] = kw.pop('random_state')
  else:
    kw = dict()
  ### training
  tree = RFC()
  for k, v in tree.__dict__.items():
    print(k, v)
  exit()
  tree.fit(X, y)
  return tree
