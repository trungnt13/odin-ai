from typing import Optional, Dict
from typing_extensions import Literal
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

__all__ = ['fast_logistic_regression', 'fast_svc']

CUML_SOLVER = set(['qn', 'lbfgs', 'owl'])


def _prepare_kw(local_dict, *args, **kwargs):
  kw = dict(local_dict)
  for key in args:
    kw.pop(key)
  kw.update(kwargs)
  return kw


def fast_svc(
    X,
    y,
    *,
    framework: Literal['auto', 'cuml', 'sklearn'] = 'sklearn',
    **kwargs,
) -> SVC:
  pass


def fast_logistic_regression(
    X,
    y,
    *,
    penalty: Literal['l1', 'l2', 'elasticnet', 'none'] = 'l2',
    C: float = 1.0,
    solver: Literal['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'] = 'lbfgs',
    fit_intercept: bool = True,
    l1_ratio: Optional[float] = None,
    tol: float = 1e-4,
    max_iter: int = 1000,
    class_weight: Optional[Dict[str, float]] = None,
    n_jobs: Optional[int] = None,
    random_state: int = 1,
    framework: Literal['auto', 'cuml', 'sklearn'] = 'sklearn',
    **kwargs,
) -> LogisticRegression:
  """The cuML LogisticRegression is only faster when n_samples > 100000 given 64
  feature dimensions"""
  kw = _prepare_kw(locals(), 'X', 'y', 'kwargs', 'framework', **kwargs)
  ### import
  is_cuml = False
  if framework == 'sklearn':
    LoRe = LogisticRegression
  else:
    try:
      from cuml.linear_model import LogisticRegression as LoRe
      is_cuml = True
      kw.pop('n_jobs')
      kw.pop('random_state')
      # if solver not in CUML_SOLVER:
      kw['solver'] = 'qn'
    except ImportError as e:
      LoRe = LogisticRegression
  ### train
  model = LoRe(**kw)
  model.fit(X, y)
  return model
