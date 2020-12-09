from __future__ import absolute_import, division, print_function

from typing import Callable, List, Union
import inspect

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
  warmup_epochs : int, optional
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
    signal = es.update(vae.trainer.last_valid_loss)
    if signal < 0:
      vae.trainer.terminate()
    elif signal > 0:
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
               losses: Union[List[float], np.ndarray, tf.Tensor] = [],
               min_improvement: float = 0.,
               warmup_epochs: int = -1,
               patience: int = 2,
               reward: float = 0.5,
               progression_length: int = 0,
               mode: Literal['min', 'max'] = 'min',
               batch_size: int = 1,
               smooth: float = 0.0,
               reduce_method: Callable[[np.ndarray, int],
                                       np.ndarray] = np.mean):
    self._losses = list(losses)
    self.min_improvement = min_improvement
    self.warmup_epochs = max(2, warmup_epochs)
    self.patience = patience
    self.reward = reward
    self.progression_length = progression_length
    self.mode = str(mode.lower())
    assert self.mode in ('min', 'max'), \
      f'only support min or max mode, given {self.mode}'
    self.batch_size = int(batch_size)
    self.smooth = float(smooth)
    assert self.smooth < 1.0, \
      f'smoothing must be smaller than 1.0 but given {self.smooth}'
    self.reduce_method = reduce_method
    self._is_disabled = False
    # history
    self._patience_history = None
    self._generalization_history = None
    self._progress_history = None

  def enable(self) -> 'EarlyStopping':
    self._is_disabled = False
    return self

  def disable(self) -> 'EarlyStopping':
    """Disable the early stopping, only allow reporting the best models."""
    self._is_disabled = True
    return self

  def __str__(self):
    s = 'EarlyStopping\n'
    for k, v in sorted(self.__dict__.items()):
      if not inspect.ismethod(v) and k[0] != '_':
        s += f' {k}:{v}\n'
    return s[:-1]

  @property
  def n_epochs(self) -> int:
    return len(self._losses)

  @property
  def patience_history(self) -> List[float]:
    if self._patience_history is None:
      n = max(self.warmup_epochs, self.n_epochs) + 1
      self._patience_history = [self.patience] * n
    return self._patience_history

  @property
  def generalization_history(self) -> List[float]:
    if self._generalization_history is None:
      n = max(self.warmup_epochs, self.n_epochs) + 1
      self._generalization_history = [0.] * n
    return self._generalization_history

  @property
  def progress_history(self) -> List[float]:
    if self._progress_history is None:
      n = max(self.warmup_epochs, self.n_epochs) + 1
      self._progress_history = [0.] * n
    return self._progress_history

  @property
  def losses(self) -> np.ndarray:
    """1D ndarray of the losses, smaller is better"""
    if len(self._losses) == 0:
      return []
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

  def update(self, loss: float) -> 'EearlyStopping':
    self._losses.append(loss)
    return self

  def __call__(self, verbose: bool = False) -> int:
    """Applying the early stopping algorithm

    Parameters
    ----------
    verbose : bool, optional
        log the debugging message, by default None

    Returns
    -------
    int : early stopping signal

        - `-1` for stop,
        - `0` for unchange,
        - `1` for best
    """
    losses = self.losses
    if self.n_epochs < self.warmup_epochs:
      if verbose:
        print(f"[EarlyStop] current epochs:{self.n_epochs} "
              f"warmup epochs:{self.warmup_epochs}")
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
    decision = 0
    if improvement < self.min_improvement:  # degrade
      self.patience -= 1
      if self.patience < 0:
        decision = -1
    elif improvement > self.min_improvement:  # improve
      self.patience += self.reward
      decision = 1
    else:  # unchanged
      ...
    # store history
    self.patience_history.append(self.patience)
    self.generalization_history.append(generalization)
    self.progress_history.append(progress)
    if verbose:
      print(
          f"[EarlyStop] disable:{self._is_disabled} "
          f"epochs:{self.n_epochs} improvement:{improvement:.4f} "
          f"progress:{progress:.4f} patience:{self.patience} "
          f"decision:{decision} last10:[{','.join(['%.2f' % i for i in self._losses[-10:]])}]"
      )
    if self._is_disabled:
      return max(0, decision)
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
      fig = plt.figure()
      ax = fig.gca()
    legends = []
    ## plotting
    min_idx = np.argmin(self._ema_L)
    min_val = self._ema_L[min_idx]
    # legends += ax.plot(self._losses, label='losses', color='red')
    legends += ax.plot(self._org_L, label='losses', color='red')
    legends += ax.plot(self._ema_L,
                       label=f'smoothed-{self.smooth}',
                       linestyle='--',
                       color='salmon')
    legends += ax.plot(min_idx,
                       min_val,
                       marker='.',
                       markersize=15,
                       alpha=0.5,
                       linewidth=0.0,
                       label='min')
    ## plot the history
    ax = ax.twinx()
    styles = dict(linestyle='-.', linewidth=1., alpha=0.6)
    legends += ax.plot(self.patience_history,
                       label='patience',
                       color='blue',
                       **styles)
    # legends += ax.plot(self._generalization_history,
    #                    label='improvement',
    #                    **styles)
    # legends += ax.plot(self._progress_history, label='progress', **styles)
    ax.tick_params(axis='y', colors='blue')
    ax.grid(False)
    ax.legend(legends, [i.get_label() for i in legends], fontsize=6)
    if save_path is not None:
      fig.savefig(save_path, dpi=200)
    return ax
