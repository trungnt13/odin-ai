from typing import Dict, List, Union

import numpy as np
import tensorflow as tf
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, MultinomialNB
from tensorflow.python.data.ops.dataset_ops import DatasetV2
from tqdm import tqdm


def fast_naive_bayes(
    X,
    distribution='bernoulli',
    alpha=1.0,
    fit_prior=True,
    class_prior=None,
    randome_state=1) -> Union[BernoulliNB, CategoricalNB, MultinomialNB]:
  if distribution == 'bernoulli':
    NB = BernoulliNB
  elif distribution == 'categorical':
    NB = BernoulliNB
  elif distribution == 'multinomial':
    NB = BernoulliNB
  else:
    raise NotImplementedError(f"No support for distribution: {distribution}")


def get_topics_string(lda: LatentDirichletAllocation,
                      vocabulary: Dict[int, str],
                      n_topics: int = 10,
                      n_words: int = 10,
                      show_word_prob: bool = False) -> List[str]:
  topics = lda.components_
  alpha = np.sum(topics, axis=1)
  alpha = alpha / np.sum(alpha)
  topics = topics / np.sum(topics, axis=1, keepdims=True)
  # Use a stable sorting algorithm so that when alpha is fixed
  # we always get the same topics.
  text = []
  for idx, topic_idx in enumerate(
      np.argsort(-alpha, kind="mergesort")[:n_topics]):
    words = topics[topic_idx]
    desc = " ".join(f"{vocabulary[i]}_{words[i]:.2f}"
                    if show_word_prob else f"{vocabulary[i]}"
                    for i in np.argsort(-words)[:n_words])
    text.append(
        f"[#{idx}]index:{topic_idx:3d} alpha={alpha[topic_idx]:.2f} {desc}")
  return np.array(text)


def fast_lda_topics(X,
                    n_components: int = 10,
                    batch_size=128,
                    max_iter=100,
                    doc_topic_prior=None,
                    topic_word_prior=None,
                    learning_decay=.7,
                    learning_offset=10.,
                    total_samples=1e6,
                    max_doc_update_iter=100,
                    n_jobs=2,
                    random_state=1) -> LatentDirichletAllocation:
  r""" Latent dirichlet allocation using online variational Bayes method.
  In each EM update, use mini-batch of training data to update the
  ``components_`` variable incrementally.
  The learning rate is controlled by the ``learning_decay`` and
  the ``learning_offset`` parameters.

  Arguments:
    n_components : int, optional (default=10)
        Number of topics.
    doc_topic_prior : float, optional (default=None)
        Prior of document topic distribution `theta`. If the value is None,
        defaults to `1 / n_components`.
    topic_word_prior : float, optional (default=None)
        Prior of topic word distribution `beta`. If the value is None, defaults
        to `1 / n_components`.
    learning_decay : float, optional (default=0.7)
        It is a parameter that control learning rate in the online learning
        method. The value should be set between (0.5, 1.0] to guarantee
        asymptotic convergence. When the value is 0.0 and batch_size is
        ``n_samples``, the update method is same as batch learning.
        literature, this is called kappa.
    learning_offset : float, optional (default=10.)
        A (positive) parameter that downweights early iterations in online
        learning.  It should be greater than 1.0.
    max_iter : integer, optional (default=10)
        The maximum number of iterations.
    batch_size : int, optional (default=128)
        Number of documents to use in each EM iteration. Only used in online
        learning.
    total_samples : int, optional (default=1e6)
        Total number of documents. Only used in the :meth:`partial_fit` method.
    mean_change_tol : float, optional (default=1e-3)
        Stopping tolerance for updating document topic distribution in E-step.
    max_doc_update_iter : int (default=100)
        Max number of iterations for updating document topic distribution in
        the E-step.
    n_jobs : int or None, optional (default=None)
        The number of jobs to use in the E-step.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    random_state : int, RandomState instance, default=None
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.
  """
  lda = LatentDirichletAllocation(
      n_components=n_components,
      batch_size=batch_size,
      max_iter=max_iter,
      doc_topic_prior=doc_topic_prior,
      topic_word_prior=topic_word_prior,
      learning_method='online',
      learning_decay=learning_decay,
      learning_offset=learning_offset,
      total_samples=total_samples,
      max_doc_update_iter=max_doc_update_iter,
      n_jobs=n_jobs,
      verbose=False,
      random_state=random_state,
  )
  prog = tqdm(desc="Perp(None)", total=max_iter)
  if isinstance(X, (tf.Tensor, tf.SparseTensor)):
    X = X.numpy()
  if isinstance(X, (np.ndarray, sparse.spmatrix)):
    for it in range(max_iter):
      lda.partial_fit(X)
      prog.update(1)
  elif isinstance(X, DatasetV2):
    for it, x in enumerate(
        X.repeat(-1).shuffle(100) if hasattr(X, 'repeat') else X):
      if it >= max_iter:
        break
      if isinstance(x, (tuple, list)):
        x = x[0]
      lda.partial_fit(x.numpy())
      if it % 10 == 0:
        perp = lda.perplexity(x)
        prog.desc = f"Perp({perp:.2f})"
      prog.update(1)
  prog.close()
  return lda
