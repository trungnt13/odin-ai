from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

def gradient_boosting_tree():
  #TODO:
  # https://xgboost.readthedocs.io/en/latest/python/python_api.html
  try:
    from cuml.ensemble import RandomForestClassifier
  except ImportError as e:
    pass
