from __future__ import absolute_import, division, print_function

import os
import unittest
from tempfile import mkstemp

import numpy as np

from odin.ml import clustering, fast_dbscan, fast_kmeans, fast_knn

np.random.seed(8)

try:
  import cuml
  _CUML = True
except ImportError:
  _CUML = False


def _prepare():
  from sklearn.datasets import load_iris
  x, y = load_iris(return_X_y=True)
  return x, y, len(np.unique(y))


class ClusteringTest(unittest.TestCase):

  def test_kmeans(self):
    x, y, n = _prepare()
    from sklearn.cluster import MiniBatchKMeans, KMeans

    model = [
        fast_kmeans(x, n_clusters=n, force_sklearn=True),
        fast_kmeans(x, n_clusters=n, batch_size=64)
    ]
    mtype = [KMeans, MiniBatchKMeans]
    if _CUML:
      from cuml.cluster import KMeans
      model.append(fast_kmeans(x, n_clusters=n))
      mtype.append(KMeans)

    for m, t in zip(model, mtype):
      self.assertTrue(isinstance(m, t))
      self.assertTrue(len(np.unique(m.predict(x))) == n)

  def test_knn(self):
    x, y, n = _prepare()
    from sklearn.neighbors import NearestNeighbors
    model = [fast_knn(x, n_neighbors=n, force_sklearn=True)]
    mtype = [NearestNeighbors]
    print(model[0].transform(x))
    exit()
    if _CUML:
      from cuml.neighbors import NearestNeighbors
      model.append(fast_knn(x, n_neighbors=n))
      mtype.append(NearestNeighbors)

    for m, t in zip(model, mtype):
      self.assertTrue(isinstance(m, t))
      self.assertTrue(len(np.unique(m.predict(x))) == n)

  def test_dbscan(self):
    x, y, n = _prepare()
    # model = fast_kmeans(x)


if __name__ == '__main__':
  unittest.main()
