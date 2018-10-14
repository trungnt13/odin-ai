from __future__ import print_function, division, absolute_import

import numpy as np
from odin.utils.mpi import MPI, cpu_count
from odin.utils import wprint
from odin.utils.crypto import md5_checksum

_cached_values = {}

# ===========================================================================
# auto-select best TSNE
# ===========================================================================
def _create_key(kwargs, md5):
  key = dict(kwargs)
  del key['verbose']
  key['md5'] = md5
  return str(list(sorted(key.items(), key=lambda x: x[0])))

def fast_tsne(*X, n_components=2, perplexity=30.0,
              early_exaggeration=8.0, learning_rate=200.0, n_iter=1000,
              n_iter_without_progress=300, min_grad_norm=1e-7,
              metric="euclidean", init="random", verbose=0,
              random_state=5218, method='barnes_hut', angle=0.5,
              n_jobs=4):
  assert len(X) > 0, "No input is given!"
  if isinstance(X[0], (tuple, list)):
    X = X[0]
  if not all(isinstance(x, np.ndarray) for x in X):
    raise ValueError("`X` can only be list of numpy.ndarray or numpy.ndarray")
  # ====== kwarg for creating T-SNE class ====== #
  kwargs = dict(locals())
  del kwargs['X']
  # ====== import proper T-SNE ====== #
  tsne_version = None
  try:
    from tsnecuda import TSNE
    from tsnecuda.NaiveTSNE import NaiveTSNE as _exact_TSNE
    tsne_version = 'cuda'
  except ImportError:
    wprint("Install CUDA-TSNE from `https://github.com/CannyLab/tsne-cuda` "
           "for significant speed up.")
    try:
      from MulticoreTSNE import MulticoreTSNE as TSNE
      tsne_version = 'multicore'
    except ImportError:
      wprint("Install MulticoreTSNE from `pip install git+https://github.com/DmitryUlyanov/Multicore-TSNE.git`"
             ' to accelerate the T-SNE on multiple CPU cores.')
      try:
        from sklearn.manifold import TSNE
        tsne_version = 'sklearn'
      except Exception as e:
        raise e
  # ====== modify kwargs ====== #
  if tsne_version == 'cuda':
    kwargs['random_seed'] = kwargs['random_state']
    kwargs['theta'] = angle
    if method == 'exact':
      TSNE = _exact_TSNE
      del kwargs['theta']
    del kwargs['random_state']
    del kwargs['n_jobs']
    del kwargs['angle']
    del kwargs['method']
  elif tsne_version == 'multicore':
    pass
  else:
    del kwargs['n_jobs']
  # ====== getting cached values ====== #
  results = []
  X_new = []
  for i, x in enumerate(X):
    md5 = md5_checksum(x)
    key = _create_key(kwargs, md5)
    if key in _cached_values:
      results.append((i, _cached_values[key]))
    else:
      X_new.append((i, md5, x))

  # ====== perform T-SNE ====== #
  def apply_tsne(j):
    idx, md5, x = j
    tsne = TSNE(**kwargs)
    return (idx, md5, tsne.fit_transform(x))
  # only 1 X, no need for MPI
  if len(X_new) == 1:
    idx, md5, x = apply_tsne(X_new[0])
    results.append((idx, x))
    _cached_values[_create_key(kwargs, md5)] = x
  else:
    mpi = MPI(jobs=X_new, func=apply_tsne, batch=1,
              ncpu=min(len(X_new), cpu_count() - 1))
    for idx, md5, x in mpi:
      results.append((idx, x))
      _cached_values[_create_key(kwargs, md5)] = x
  # ====== return and clean ====== #
  results = sorted(results, key=lambda a: a[0])
  results = [r[1] for r in results]
  return results[0] if len(results) == 1 else results
