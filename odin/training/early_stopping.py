from __future__ import absolute_import, division, print_function

import inspect
from collections import defaultdict
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
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
    signal = es.update(model.trainer.last_valid_loss)
    if signal < 0:
      model.trainer.terminate()
    elif signal > 0:
      model.save_weights()
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
               smooth: float = 0.4,
               batch_size: int = 1,
               reduce_method: Callable[[np.ndarray, int],
                                       np.ndarray] = np.mean):
    self._losses = list(losses)
    self.min_improvement = min_improvement
    self.warmup_epochs = max(2, warmup_epochs)
    self.patience = patience
    self.reward = reward
    self.progression_length = progression_length
    self._mode = str(mode.lower())
    self.batch_size = int(batch_size)
    self.smooth = float(smooth)
    assert self.smooth < 1.0, \
      f'smoothing must be smaller than 1.0 but given {self.smooth}'
    self.reduce_method = reduce_method
    self._is_disabled = False
    # history: Dict[str, Dict[int, float]]
    self._history = defaultdict(dict)

  @property
  def mode(self) -> str:
    return self._mode

  @mode.setter
  def mode(self, mode):
    assert mode in ('min', 'max'), \
      f'only support min or max mode, given {mode}'
    self._mode = mode

  def enable(self) -> 'EarlyStopping':
    self._is_disabled = False
    return self

  def disable(self) -> 'EarlyStopping':
    """Disable the early stopping, only allow reporting the best models."""
    self._is_disabled = True
    return self

  @property
  def n_epochs(self) -> int:
    return len(self._losses)

  @property
  def patience_history(self) -> Dict[int, float]:
    return self._history['patience']

  @property
  def generalization_history(self) -> Dict[int, float]:
    return self._history['generalization']

  @property
  def progress_history(self) -> Dict[int, float]:
    return self._history['progress']

  @property
  def decision_history(self) -> Dict[int, float]:
    return self._history['decision']

  @property
  def losses(self) -> np.ndarray:
    """1D ndarray of the losses, smaller is better"""
    if len(self._losses) <= self.batch_size:
      return self._losses
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

  def update(self, loss: Number) -> 'EearlyStopping':
    if hasattr(loss, 'numpy'):
      loss = loss.numpy()
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
      return 0
    # generalization error (smaller is better)
    current = losses[-1]
    last_best = np.min(losses[:-1]) + 1e-8
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
    curr_iter = len(self._losses) - 1
    self.patience_history[curr_iter] = self.patience
    self.generalization_history[curr_iter] = generalization
    self.progress_history[curr_iter] = progress
    self.decision_history[curr_iter] = decision
    self._history['losses'][curr_iter] = (self._org_L[-1], self._ema_L[-1])
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

  def plot_losses(self,
                  path: Optional[str] = None,
                  fig: Optional[plt.Figure] = None):
    losses = self.losses
    if len(losses) < 2:
      return None
    ## use seaborn style
    try:
      import seaborn as sns
      sns.set()
    except ImportError:
      pass
    ## prepare the figure
    if fig is None:
      fig = plt.figure(figsize=(12, 4))
    legends = []
    ## plotting
    min_idx = np.argmin(self._ema_L)
    min_val = self._ema_L[min_idx]
    org_losses = self._losses
    iter_ticks = np.linspace(1, len(org_losses), num=min(5, len(org_losses)))
    iter_ticklabels = [int(i) for i in iter_ticks]
    marker_styles = dict(markersize=15, linewidth=0.0, alpha=0.5)
    ax_finalize = lambda axis, title: (
        ax.set_xticks(iter_ticks),
        ax.set_xticklabels(iter_ticklabels),
        ax.tick_params(axis='both', labelsize=8),
        ax.legend(fontsize=8),
        ax.set_title(title),
    )
    ## plot the original losses
    ax = fig.add_subplot(1, 3, 1)
    ax.plot(np.arange(len(org_losses)) + 1, org_losses, color='red')
    styles = dict(marker_styles)
    for it, decision in self.decision_history.items():
      if decision > 0:
        ax.plot(it + 1,
                org_losses[it],
                marker='.',
                color='blue',
                **marker_styles)
      elif decision < 0:
        ax.plot(it + 1,
                org_losses[it],
                marker='x',
                color='red',
                **marker_styles)
    ax_finalize(ax, 'Original Losses (O: save, X: stop)')
    ## plot the aggregated losses
    ax = fig.add_subplot(1, 3, 2)
    x, y1, y2 = [], [], []
    for it, (l, l_smooth) in self._history['losses'].items():
      x.append(it)
      y1.append(l)
      y2.append(l_smooth)
    ax.plot(x, y1, label=f'aggregated-{self.batch_size}')
    ax.plot(x, y2, label=f'smoothed-{self.smooth}', linestyle='--')
    ax.plot(min_idx, min_val, marker='.', label='min', **marker_styles)
    ax_finalize(ax, 'Smoothed Losses')
    ## plot the patience
    ax = fig.add_subplot(1, 3, 3)
    l1 = ax.plot(list(self.patience_history.keys()),
                 list(self.patience_history.values()),
                 color='red',
                 label='patience',
                 linewidth=1.0,
                 alpha=0.6)
    ax_finalize(ax, 'Events')
    org_ax = ax
    ax = ax.twinx()
    l2 = ax.plot(list(self.generalization_history.keys()),
                 list(self.generalization_history.values()),
                 color='blue',
                 label='improvement',
                 linewidth=1.0,
                 alpha=0.6)
    l3 = ax.plot(list(self.progress_history.keys()),
                 list(self.progress_history.values()),
                 color='blue',
                 linestyle='--',
                 label='progress',
                 linewidth=1.0,
                 alpha=0.6)
    ax.grid(False)
    legends = l1 + l2 + l3
    labels = [l.get_label() for l in legends]
    org_ax.legend(legends, labels, fontsize=8)
    ax.tick_params(axis='y', labelcolor='blue', labelsize=8)
    ## save the axis
    plt.tight_layout()
    if path is not None:
      fig.savefig(path, dpi=200)
    return ax

  def __str__(self):
    s = 'EarlyStopping:\n'
    s += f' min_improvement: {self.min_improvement}\n'
    s += f' warmup_epochs: {self.warmup_epochs}\n'
    s += f' patience: {self.patience}\n'
    s += f' reward: {self.reward}\n'
    s += f' progression_length: {self.progression_length}\n'
    s += f' mode: {self.mode}\n'
    s += f' smooth: {self.smooth}\n'
    s += f' batch_size: {self.batch_size}\n'
    s += f' reduce_method: {self.reduce_method}\n'
    s += f' losses: {self._losses}\n'
    return s[:-1]
