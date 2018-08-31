from __future__ import print_function, division, absolute_import
import os
import pickle

import numpy as np

from odin.utils import mpi, batching, Progbar, crypto
from odin.fuel import MmapData

from .base import BaseEstimator, TransformerMixin, DensityMixin
from .gmm_tmat import GMM, Tmatrix, _split_jobs

# ===========================================================================
# Helper
# ===========================================================================
def _input_data_2_md5(X, sad=None, indices=None):
  X_md5 = crypto.md5_checksum(X)
  sad_md5 = crypto.md5_checksum(sad) if sad is not None else ''
  ids_md5 = crypto.md5_checksum(indices) if indices is not None else ''
  exit()
  return X_md5 + sad_md5 + ids_md5

# ===========================================================================
# Fast combined GMM-Tmatrix training for I-vector extraction
# ===========================================================================
class Ivector(DensityMixin, BaseEstimator, TransformerMixin):
  """ Ivector extraction using GMM and T-matrix """

  def __init__(self, path, nmix=None, tv_dim=None,
               nmix_start=1, niter_gmm=16, niter_tmat=16,
               allow_rollback=True, exit_on_error=False,
               downsample=1, stochastic_downsample=True,
               device='gpu', ncpu=1, gpu_factor_gmm=80, gpu_factor_tmat=3,
               dtype='float32', seed=5218, name=None):
    super(Ivector, self).__init__()
    # ====== auto store arguments ====== #
    for key, val in locals().items():
      if key in ('self', 'path', 'seed'):
        continue
      setattr(self, key, val)
    # ====== create random generator ====== #
    self._rand = np.random.RandomState(seed=seed)
    # ====== check path ====== #
    path = str(path)
    if not os.path.exists(path):
      os.mkdir(path)
    elif not os.path.isdir(path):
      raise ValueError("Path to '%s' is not a directory" % str(path))
    self._path = path
    self._gmm = None
    self._tmat = None

  # ==================== properties ==================== #
  @property
  def gmm(self):
    if self._gmm is None:
      if os.path.exists(self.gmm_path):
        with open(self.gmm_path, 'rb') as f:
          self._gmm = pickle.load(f)
      else:
        self._gmm = GMM(nmix=self.nmix, niter=self.niter_gmm, dtype=self.dtype,
                        downsample=self.downsample,
                        stochastic_downsample=self.stochastic_downsample,
                        device=self.device,
                        ncpu=self.ncpu, gpu_factor=self.gpu_factor_gmm,
                        seed=5218,
                        path=self.gmm_path,
                        name="IvecGMM_%s" % (self.name if self.name is not None else
                                             str(self._rand.randint(10e8))))
    return self._gmm

  @property
  def tmat(self):
    if self._tmat is None:
      if os.path.exists(self.tmat_path):
        with open(self.tmat_path, 'rb') as f:
          self._tmat = pickle.load(f)
      else:
        self._tmat = Tmatrix(tv_dim=self.tv_dim, gmm=self.gmm,
                             niter=self.niter_tmat, dtype=self.dtype,
                             device=self.device,
                             ncpu=self.ncpu, gpu_factor=self.gpu_factor_tmat,
                             cache_path='/tmp', seed=5218,
                             path=self.tmat_path,
                             name='IvecTmat_%s' % (self.name if self.name is not None else
                                                   str(self._rand.randint(10e8))))
    return self._tmat

  @property
  def path(self):
    return self._path

  @property
  def gmm_path(self):
    return os.path.join(self.path, 'gmm.pkl')

  @property
  def tmat_path(self):
    return os.path.join(self.path, 'tmat.pkl')

  @property
  def z_path(self):
    """ Path to zero-th order statistics of the training data"""
    return os.path.join(self.path, 'zstat_train')

  @property
  def f_path(self):
    """ Path to first order statistics of the training data """
    return os.path.join(self.path, 'fstat_train')

  @property
  def ivec_path(self):
    """ Path to first order statistics of the training data """
    return os.path.join(self.path, 'ivec_train')

  @property
  def name_list_path(self):
    """ In case indices is given during training, the order of
    processed files is store at this path """
    return os.path.join(self.path, 'name_list')

  @property
  def feat_dim(self):
    return self.gmm.feat_dim

  # ==================== state query ==================== #
  @property
  def is_gmm_fitted(self):
    return self.gmm.is_fitted

  @property
  def is_tmat_fitted(self):
    return self.is_gmm_fitted and self.tmat.is_fitted

  @property
  def is_fitted(self):
    return self.is_gmm_fitted and self.is_tmat_fitted

  # ==================== sklearn methods ==================== #
  def fit(self, X, indices=None, sad=None,
          refit_gmm=False, refit_tmat=False,
          extract_ivec=True, keep_stats=False):
    """
    Parameters
    ----------
    X : ndarray
      Training data [n_samples, n_features]
    indices : {Mapping, tuple, list}
      in case the data is given by a list of files, `indices`
      act as file indicator mapping from
      'file_name' -> (start_index_in_X, end_index_in_X)
      This mapping can be provided by a dictionary, or list of
      tuple.
    sad : ndarray
      inspired by the "Speech Activity Detection" (SAD) indexing,
      this array is indicator of which samples will be taken into
      training; the shape should be [n_samples,] or [n_samples, 1]
    refit_gmm : bool
      if True, re-fit the GMM even though it is fitted,
      consequently, the T-matrix will be re-fitted
    refit_tmat : bool
      if True, re-fit the T-matrix even though it is fitted
    extract_ivec : bool
      if True, extract the i-vector for training data
    keep_stats : bool
      if True, keep the zero and first order statistics.
      The first order statistics could consume huge amount
      of disk space. Otherwise, they are deleted after training
    """
    n_samples = X.shape[0]
    new_gmm = (not self.gmm.is_fitted or refit_gmm)
    # ====== clean error files ====== #
    if os.path.exists(self.z_path):
      Z = MmapData(self.z_path, read_only=True)
      if Z.shape[0] == 0: # empty file
        os.remove(self.z_path)
      Z.close()
    if os.path.exists(self.f_path):
      F = MmapData(self.f_path, read_only=True)
      if F.shape[0] == 0: # empty file
        os.remove(self.f_path)
      F.close()
    if os.path.exists(self.ivec_path):
      ivec = MmapData(self.ivec_path, read_only=True)
      if ivec.shape[0] == 0: # empty file
        os.remove(self.ivec_path)
      ivec.close()
    # ====== Training the GMM first ====== #
    if new_gmm:
      input_data = [X]
      if sad is not None:
        input_data.append(sad)
      if indices is not None:
        input_data.append(indices)
      self.gmm.fit(input_data)
    # ====== some fun, and confusing logics ====== #
    # GMM need to be fitted before creating T-matrix model
    new_tmat = (not self.tmat.is_fitted or new_gmm or refit_tmat)
    # New I-vector is need when:
    # - only when `extract_ivec=True`
    # - and new T-matrix is trained but no I-vector is extracted
    new_ivec = extract_ivec and \
    (new_tmat or not os.path.exists(self.ivec_path))
    # new stats is only needed when
    # - GMM is updated
    # - training new Tmatrix and the Z and F not exist
    # - extracting new I-vector and the Z and F not exist
    if not new_gmm and \
    (os.path.exists(self.z_path) and os.path.exists(self.f_path)):
      new_stats = False
    else:
      new_stats = new_gmm or new_tmat or new_ivec
    # ====== extract the statistics ====== #
    if new_stats:
      # indices is None, every row is single sample (utterance or image ...)
      if indices is None:
        if os.path.exists(self.z_path):
          os.remove(self.z_path)
        if os.path.exists(self.f_path):
          os.remove(self.f_path)
        Z = MmapData(path=self.z_path, dtype='float32',
                     shape=(n_samples, self.nmix), read_only=False)
        F = MmapData(path=self.f_path, dtype='float32',
                     shape=(n_samples, self.feat_dim * self.nmix),
                     read_only=False)
        jobs, _ = _split_jobs(n_samples, ncpu=mpi.cpu_count(),
                           device='cpu', gpu_factor=1)

        def map_transform(start_end):
          start, end = start_end
          for i in range(start, end):
            # removed by SAD
            if sad is not None and not bool(sad[i]):
              yield None, None, None
            else:
              z, f = self.gmm.transform(X[i][np.newaxis, :],
                                        zero=True, first=True, device='cpu')
              yield i, z, f
        prog = Progbar(target=n_samples,
                       print_report=True, print_summary=False,
                       name="Extracting zero and first order statistics")
        for i, z, f in mpi.MPI(jobs, map_transform,
                               ncpu=None, batch=1):
          if i is not None: # i None means removed by SAD
            Z[i] = z
            F[i] = f
          prog.add(1)
        Z.flush(); Z.close()
        F.flush(); F.close()
      # use directly the transform_to_disk function
      else:
        self.gmm.transform_to_disk(X, indices=indices, sad=sad,
                                   pathZ=self.z_path,
                                   pathF=self.f_path,
                                   name_path=self.name_list_path,
                                   dtype='float32', device=None, ncpu=None,
                                   override=True)
    # ====== Training the T-matrix and extract i-vector ====== #
    if new_tmat or new_ivec:
      Z = MmapData(path=self.z_path, read_only=True)
      F = MmapData(path=self.f_path, read_only=True)
      if new_tmat:
        self.tmat.fit((Z, F))
      if new_ivec:
        self.tmat.transform_to_disk(path=self.ivec_path, Z=Z, F=F,
                                    dtype='float32', device='gpu',
                                    override=True)
      Z.close()
      F.close()
    # ====== clean ====== #
    if not keep_stats:
      if os.path.exists(self.z_path):
        os.remove(self.z_path)
      if os.path.exists(self.f_path):
        os.remove(self.f_path)
    return self

  def transform(self, X, indices=None, sad=None,
                Z=None, F=None):
    pass
