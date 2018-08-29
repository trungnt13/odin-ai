from __future__ import print_function, division, absolute_import

from .base import BaseEstimator, TransformerMixin, DensityMixin
from .gmm_tmat import GMM, Tmatrix
# ===========================================================================
# Fast combined GMM-Tmat training for I-vector extraction
# ===========================================================================
class Ivector(DensityMixin, BaseEstimator, TransformerMixin):
  """ Ivector extraction using GMM and T-matrix """

  def __init__(self, path, nmix=None, tv_dim=None,
               nmix_start=1, niter_gmm=16, niter_tmat=16,
               allow_rollback=True, exit_on_error=False,
               batch_size_cpu='auto', batch_size_gpu='auto',
               downsample=1, stochastic_downsample=True,
               device='cpu', ncpu=1, gpu_factor_gmm=80, gpu_factor_tmat=3,
               dtype='float32', seed=5218, name=None):
    super(Ivector, self).__init__()
