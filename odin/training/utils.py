from __future__ import absolute_import, division, print_function

from typing import Callable, List, Union

import numpy as np
import tensorflow as tf
from scipy import signal
from typeguard import typechecked
from typing_extensions import Literal

__all__ = [
    'EarlyStopping',
]


def exponential_moving_average(x, w):
  """`s[0] = x[0]` and `s[t] = w * x[t] + (1-w) * s[t-1]`"""
  b = [w]
  a = [1, w - 1]
  zi = signal.lfilter_zi(b, a)
  return signal.lfilter(b, a, x, zi=zi * x[0])[0]


class EarlyStopping:
  r"""Generalized interface for early stopping, the algorithm is
  based on generalization loss and the three rules:

  - Stop when generalization error exceeds threshold in a number of
      successive steps.
  - Stop as soon as the generalization loss exceeds a certain threshold.
  - Surpress stopping if the training is still progress rapidly.

  Generalization loss: `GL(t) = losses[-1] / min(losses[:-1]) - 1`

  - if `GL(t)` <= 0 : save the best model
  - if `GL(t)` <= threshold : continue training
  - if `GL(t)` > threshold : stop training

  Progression: `PG(t) = 10 * sum(L) / (k * min(L))` where `L = losses[-k:]`

  The condition for early stopping: `GL(t) / PG(t) >= threshold`

  Parameters
  ----------
  min_improvement : float, optional
      Determine by `generalization_error / progression`, by default 0.
  warmup_niter : int, optional
      Minimum number of iteration until early stop kicks in., by default -1
  patience : int, optional
      after which training will be stopped, by default 0
  reward : float, optional
      added amount to patience when improvement achieved, by default 0.5
  progression_length : int, optional
      Number of steps to look into the past for estimating the training
      progression. If smaller than 2, turn-off progression for early
      stopping, by default 0.
  mode : {'min', 'max'}
      training will stop when the quantity monitored has stopped decreasing;
      in `max` mode it will stop when the quantity monitored has stopped
      increasing, by default 'min'
  losses : Union[List[float], np.ndarray, tf.Tensor], optional
      List of loss values, by default []
  batch_size : int, optional
      batching the losses into mini-batch then applying the
      `reduce_method`, by default 1
  smooth : float, optional
      moving exponential average smooth value, 0.0 means no smoothing and
      the value must < 1.0, by default 0.2
  reduce_method : Callable[[np.ndarray, int], np.ndarray], optional
      for mini-batched losses, by default np.mean
  verbose : bool, optional
      print-out log after each update, by default False

  Return
  -------
  `True` if early-stopping required, otherwise, `False`

  Example
  --------
  ```python
  def callback():
    if es.update(vae.trainer.last_valid_loss):
      vae.trainer.terminate()
    elif es.is_best:
      vae.save_weights()
  ```

  References
  -----------
  Chuang, C.-Y., Torralba, A., Jegelka, S., 2020. Estimating Generalization
      under Distribution Shifts via Domain-Invariant Representations.
      arXiv:2007.03511 [cs, stat].
  Heckel, R., Yilmaz, F.F., 2020. Early Stopping in Deep Networks: Double
      Descent and How to Eliminate it. arXiv:2007.10099 [cs, stat].
  Pang, T., Yang, X., Dong, Y., Su, H., Zhu, J., 2020. Bag of Tricks
      for Adversarial Training. arXiv:2010.00467 [cs, stat].
  Prechelt, L., n.d. Early Stopping | but when? 15.
  Wang, W., Hu, T., Lin, C., Cheng, G., 2020. Regularization Matters:
      A Nonparametric Perspective on Overparametrized Neural Network.
      arXiv:2007.02486 [cs, stat].
  """

  @typechecked
  def __init__(self,
               min_improvement: float = 0.,
               warmup_niter: int = -1,
               patience: int = 2,
               reward: float = 0.5,
               progression_length: int = 0,
               mode: Literal['min', 'max'] = 'min',
               losses: Union[List[float], np.ndarray, tf.Tensor] = [],
               batch_size: int = 1,
               smooth: float = 0.1,
               reduce_method: Callable[[np.ndarray, int], np.ndarray] = np.mean,
               verbose: bool = False):
    self.min_improvement = min_improvement
    self.warmup_niter = max(2, warmup_niter)
    self.patience = patience
    self.reward = reward
    self.progression_length = progression_length
    self.mode = str(mode.lower())
    assert self.mode in ('min', 'max'), \
      f'only support min or max mode, given {self.mode}'
    self._losses = list(losses)
    self.batch_size = int(batch_size)
    self.smooth = float(smooth)
    assert self.smooth < 1.0, \
      f'smoothing must be smaller than 1.0 but given {self.smooth}'
    self.verbose = verbose
    self.reduce_method = reduce_method
    self._is_best = False
    # history
    n = max(warmup_niter, self.n_iter) + 1
    self._patience_history = [self.patience] * n
    self._generalization_history = [0.] * n
    self._progress_history = [1.] * n

  @property
  def n_iter(self) -> int:
    return len(self._losses)

  @property
  def is_best(self) -> bool:
    """Return `True` if the last iteration achieved the best score."""
    return self._is_best

  @property
  def patience_history(self) -> List[float]:
    return list(self._patience_history)

  @property
  def generalization_history(self) -> List[float]:
    return list(self._generalization_history)

  @property
  def progress_history(self) -> List[float]:
    return list(self._progress_history)

  @property
  def losses(self) -> np.ndarray:
    """1D ndarray of the losses, smaller is better"""
    if self.mode == 'min':
      L = self._losses
    else:
      L = [-i for i in self._losses]
    # add abs(min(L)) in case of negative losses
    L = np.asarray(L) + np.abs(np.min(L))
    if self.batch_size > 1:
      mod = L.shape[0] % self.batch_size
      if mod != 0:
        L = np.pad(L, mod, mode='edge')
      L = self.reduce_method(np.reshape(L, (-1, self.batch_size)), axis=-1)
    self._org_L = L
    L = exponential_moving_average(L, w=1. - self.smooth)
    self._ema_L = L
    return L

  def update(self, loss: float) -> bool:
    self._losses.append(loss)
    return self()

  def __call__(self, verbose=None) -> bool:
    losses = self.losses
    if self.n_iter < self.warmup_niter:
      if self.verbose:
        print(f"[EarlyStop] niter:{self.n_iter} Not enough iteration ")
      return False
    # generalization error (smaller is better)
    current = losses[-1]
    last_best = np.min(losses[:-1])
    # >0 <=> improvement (bigger is better)
    generalization = 1 - current / last_best
    # progression (bigger is better)
    if self.progression_length > 1:
      progress = losses[-self.progression_length:]
      progress = 10 * \
        (np.sum(progress) / (self.progression_length * np.min(progress)) - 1)
    else:
      progress = 1.
    # thresholding
    improvement = generalization / progress
    self._is_best = False
    if improvement < self.min_improvement:
      self.patience -= 1
    elif improvement > self.min_improvement:
      self._is_best = True
      self.patience += self.reward
    decision = True if self.patience < 0 else False
    # store history
    self._patience_history.append(self.patience)
    self._generalization_history.append(generalization)
    self._progress_history.append(progress)
    if (bool(verbose) if verbose is not None else self.verbose):
      print(f"[EarlyStop] niter:{self.n_iter} improvement:{improvement:.4f} "
            f"progress:{progress:.4f} patience:{self.patience} "
            f"best:{self.is_best} stop:{decision} "
            f"last10:[{','.join(['%.2f' % i for i in losses[-10:]])}]")
    return decision

  def plot_losses(self, save_path=None, ax=None):
    losses = self.losses
    if len(losses) < 2:
      return None
    from matplotlib import pyplot as plt
    try:
      import seaborn as sns
      sns.set()
    except ImportError:
      pass
    if ax is None:
      ax = plt.gca()
    legends = []
    ## plotting
    min_idx = np.argmin(self._ema_L)
    min_val = self._ema_L[min_idx]
    legends += ax.plot(self._org_L, label='losses', color='red')
    legends += ax.plot(self._ema_L,
                       label=f'smoothed-{self.smooth}',
                       linestyle='--',
                       color='salmon')
    legends += ax.plot(min_idx,
                       min_val,
                       marker='.',
                       markersize=10,
                       alpha=0.5,
                       linewidth=0.0,
                       label='min')
    ## plot the history
    ax = ax.twinx()
    styles = dict(linestyle='-.', linewidth=1.5, alpha=0.4)
    legends += ax.plot(self._patience_history, label='patience', **styles)
    legends += ax.plot(self._generalization_history,
                       label='improvement',
                       **styles)
    legends += ax.plot(self._progress_history, label='progress', **styles)
    ax.grid(False)
    ax.legend(legends, [i.get_label() for i in legends], fontsize=6)
    if save_path is not None:
      ax.get_figure().savefig(save_path, dpi=200)
    return ax
