import numpy as np

def compute_class_avg(data, labels, nclasses):
  ndim = data.shape[0]
  mu_c = np.zeros((nclasses, ndim))
  for c in labels:  # numeric labels are assumed
    idx = np.flatnonzero(labels == c)
    mu_c[c] = data[:, idx].mean(1)
  return mu_c
