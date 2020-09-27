import os
import pickle
import re
from abc import ABCMeta, abstractproperty
from itertools import chain
from numbers import Number
from types import MethodType
from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union
from urllib.request import urlretrieve

import numpy as np
import tensorflow as tf
from numpy import ndarray
from odin.fuel.dataset_base import IterableDataset, get_partition
from odin.utils import one_hot
from scipy import sparse
from scipy.sparse import csr_matrix, spmatrix
from six import add_metaclass, string_types
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

try:
  from tokenizers import Encoding
  from tokenizers.implementations import BaseTokenizer
except ImportError:
  Encoding = "Encoding"
  BaseTokenizer = object


# ===========================================================================
# Helpers
# ===========================================================================
_token_pattern = re.compile(r"(?u)\b[a-fA-F]\w+\b")


def _simple_tokenizer(doc: str) -> List[str]:
  return _token_pattern.findall(doc)


def _simple_preprocess(doc: str) -> str:
  doc = doc.lower().strip()
  doc = re.sub(r"'", "", doc)
  doc = re.sub(r"\W", " ", doc)
  doc = re.sub(r"\s+", " ", doc)
  return doc


# ===========================================================================
# Base dataset
# ===========================================================================
@add_metaclass(ABCMeta)
class NLPDataset(IterableDataset):
  r"""
  Arguments:
    algorithm: {'tf', 'tfidf', 'bert'}
      Which algorithm used for tokenizing
        'tf' - term frequency or bag-of-words
        'tfidf' - term count and inverse document frequency
        'count' - count vectorizer
        'bert' - BERT tokenizer
    vocab_size: int
      The size of the final vocabulary, including all tokens and alphabet.
    min_frequency: int
      When building the vocabulary ignore terms that have a document
      frequency strictly lower than the given threshold. This value is also
      called cut-off in the literature.
      If float in range of [0.0, 1.0], the parameter represents a proportion
      of documents, integer absolute counts.
      This parameter is ignored if vocabulary is not None.
    max_frequency : float or int, default=1.0
      When building the vocabulary ignore terms that have a document
      frequency strictly higher than the given threshold (corpus-specific
      stop words).
      If float in range [0.0, 1.0], the parameter represents a proportion of
      documents, integer absolute counts.
      This parameter is ignored if vocabulary is not None.
    limit_alphabet: int
      The maximum different characters to keep in the alphabet.
    max_length : int
      longest document length
    ngram_range : tuple (min_n, max_n), default=(1, 1)
      The lower and upper boundary of the range of n-values for different
      n-grams to be extracted. All values of n such that min_n <= n <= max_n
      will be used. For example an ``ngram_range`` of ``(1, 1)`` means only
      only bigrams.
      Only applies if ``analyzer is not callable``.
  """

  def __init__(self,
               algorithm: str = 'tf',
               vocab_size: int = 1000,
               min_frequency: int = 2,
               max_frequency: float = 0.98,
               limit_alphabet: int = 1000,
               max_length: Optional[int] = 1000,
               ngram_range: Tuple[int, int] = (1, 1),
               vocabulary: Dict[str, int] = None,
               retrain_tokenizer: bool = False,
               cache_path: str = "~/nlp_data"):
    self._cache_path = os.path.abspath(os.path.expanduser(cache_path))
    self._labels = []
    #
    self._vocabulary = None
    if vocabulary is not None:
      vocab_size = len(vocabulary)
      with open(os.path.join(self.cache_path, "bert_vocab.txt"), 'w') as f:
        for special_token in ("[SEP]", "[UNK]", "[CLS]", "[PAD]", "[MASK]"):
          f.write(f"{special_token}\n")
        for term, idx in sorted(vocabulary.items(), key=lambda x: x[-1]):
          f.write(term + '\n')
    self._init_vocabulary = vocabulary
    self.max_length = max_length
    self.vocab_size = int(vocab_size)
    self.min_frequency = int(min_frequency)
    self.max_frequency = float(max_frequency)
    self.limit_alphabet = int(limit_alphabet)
    self.ngram_range = tuple(ngram_range)
    self.retrain_tokenizer = bool(retrain_tokenizer)
    # load exists tokenizer
    algorithm = str(algorithm).lower().strip()
    assert algorithm in ('tf', 'tfidf', 'bert', 'count'), \
      f"Support algorithm: tf, tfidf, count and bert; but given:{algorithm}"
    self.algorithm = algorithm
    self._tokenizer = None

  @property
  def shape(self) -> List[int]:
    return self.transform('train').shape[1:]

  @property
  def labels(self) -> List[str]:
    return np.array(self._labels)

  @abstractproperty
  def train_text(self) -> Iterable[str]:
    raise NotImplementedError

  @abstractproperty
  def valid_text(self) -> Iterable[str]:
    raise NotImplementedError

  @abstractproperty
  def test_text(self) -> Iterable[str]:
    raise NotImplementedError

  @property
  def train_labels(self) -> Union[ndarray, spmatrix]:
    return np.asarray([])

  @property
  def valid_labels(self) -> Union[ndarray, spmatrix]:
    return np.asarray([])

  @property
  def test_labels(self) -> Union[ndarray, spmatrix]:
    return np.asarray([])

  def filter_by_length(
      self,
      inputs: Union[int, List[str], List[Encoding]],
      iqr_multiplier: float = 1.5,
      length_range: Optional[Tuple[int, int]] = None
  ) -> Tuple[List[bool], int, int]:
    r""" Using inter-quartile to filter out outlier documents by their
    tokenized lengths. """
    lengths = np.asarray(
        [
            len(i.split(" ")) if isinstance(i, string_types) else
            (int(i) if isinstance(i, Number) else len(i)) for i in inputs
        ],
        dtype=np.int32,
    )
    if length_range is None:
      q1 = np.quantile(lengths, 0.25)
      q3 = np.quantile(lengths, 0.75)
      iqr = q3 - q1
      lmin = q1 - iqr_multiplier * iqr
      lmax = q3 + iqr_multiplier * iqr
    else:
      lmin, lmax = length_range
    mask = np.logical_and(lengths > lmin, lengths < lmax)
    return mask, lmin, lmax

  def transform(self,
                documents: Optional[Union[str, List[str]]] = None) -> spmatrix:
    r""" Vectorize the input documents """
    # cached transformed dataset
    if isinstance(documents, string_types) and \
      documents in ('train', 'valid', 'test'):
      attr_name = f'_x_{documents}'
      if hasattr(self, attr_name):
        return getattr(self, attr_name)
      x = self.transform(
          get_partition(documents,
                        train=self.train_text,
                        valid=self.valid_text,
                        test=self.test_text))
      setattr(self, attr_name, x)
      return x
    # other data
    if self.algorithm in ('tf', 'tfidf', 'count'):
      x = self.tokenizer.transform(documents)
      # sorted ensure right ordering for Tensorflow SparseTensor
    else:
      if isinstance(documents, Generator):
        documents = [i for i in documents]
      x = sparse.csr_matrix(
          [i.ids for i in self.encode(documents, post_process=True)])
    return x

  @property
  def cache_path(self) -> str:
    if not os.path.exists(self._cache_path):
      os.makedirs(self._cache_path)
    return self._cache_path

  @property
  def tokenizer(self) -> Union[BaseTokenizer, CountVectorizer, TfidfVectorizer]:
    pkl_path = os.path.join(self.tokenizer_path, "model.pkl")
    if self._tokenizer is not None:
      return self._tokenizer
    ### get pickled tokenizer
    if os.path.exists(pkl_path) and not self.retrain_tokenizer:
      with open(pkl_path, 'rb') as f:
        tokenizer = pickle.load(f)
    ### train new tokenizer
    else:
      self.retrain_tokenizer = False
      if self.algorithm == 'bert':
        from tokenizers import BertWordPieceTokenizer
        tokenizer = BertWordPieceTokenizer(
            vocab_file=None if self._init_vocabulary is None else os.path.
            join(self.cache_path, "bert_vocab.txt"))
        tokenizer.enable_truncation(max_length=self.max_length)
        tokenizer.enable_padding(length=self.max_length)
        # train the tokenizer
        if self._init_vocabulary is None:
          path = os.path.join(self.cache_path, 'train.txt')
          with open(path, 'w') as f:
            for i in chain(self.train_text, self.valid_text, self.test_text):
              if len(i) == 0:
                continue
              f.write(i + "\n" if i[-1] != "\n" else i)
          tokenizer.train(files=path,
                          vocab_size=self.vocab_size,
                          min_frequency=self.min_frequency,
                          limit_alphabet=self.limit_alphabet,
                          show_progress=True)
        tokenizer.save_model(self.tokenizer_path)
      elif self.algorithm in ('count', 'tf', 'tfidf'):
        if self.algorithm == 'count':
          tokenizer = CountVectorizer(input='content',
                                      ngram_range=self.ngram_range,
                                      min_df=self.min_frequency,
                                      max_df=self.max_frequency,
                                      max_features=self.vocab_size,
                                      vocabulary=self._init_vocabulary,
                                      tokenizer=_simple_tokenizer,
                                      stop_words='english')
        elif self.algorithm in ('tf', 'tfidf'):
          tokenizer = TfidfVectorizer(
              input='content',
              ngram_range=self.ngram_range,
              min_df=self.min_frequency,
              max_df=self.max_frequency,
              max_features=self.vocab_size,
              stop_words='english',
              vocabulary=self._init_vocabulary,
              tokenizer=_simple_tokenizer,
              use_idf=False if self.algorithm == 'tf' else True)
        tokenizer.fit(
            (_simple_preprocess(i)
             for i in chain(self.train_text, self.valid_text, self.test_text)))
      else:
        raise NotImplementedError
      # save the pickled model
      with open(pkl_path, "wb") as f:
        pickle.dump(tokenizer, f)
    ### assign and return
    self._tokenizer = tokenizer
    return self._tokenizer

  @property
  def tokenizer_path(self) -> str:
    p = os.path.join(
        self.cache_path, f"tokenizer_{self.algorithm}_{self.vocab_size}_"
        f"{self.min_frequency}_{self.max_frequency}_"
        f"{self.limit_alphabet}")
    if not os.path.exists(p):
      os.makedirs(p)
    return p

  @property
  def vocabulary(self) -> Dict[int, str]:
    if self._vocabulary is None:
      if self.algorithm in ('tf', 'tfidf', 'count'):
        vocab = self.tokenizer.vocabulary_
      else:
        vocab = self.tokenizer.get_vocab()
      self._vocabulary = {
          v: k for k, v in sorted(vocab.items(), key=lambda x: x[-1])
      }
    return self._vocabulary

  @property
  def vocabulary_size(self) -> int:
    return len(self.vocabulary)

  def encode(self,
             inputs: Union[str, List[str]],
             add_special_tokens: bool = True,
             post_process: bool = False) -> List[Encoding]:
    r""" Encode sequence of text string """
    is_batch = True
    if isinstance(inputs, string_types):
      inputs = [inputs]
      is_batch = False
    outputs = self.tokenizer.encode_batch(inputs, add_special_tokens=True)
    if post_process:
      outputs = [
          self.tokenizer.post_process(i, add_special_tokens=add_special_tokens)
          for i in outputs
      ]
    return outputs if is_batch else outputs[0]

  def post_process(self,
                   encoding,
                   add_special_tokens: bool = True) -> List[Encoding]:
    r""" Apply all the post-processing steps to the given encodings.

    The various steps are:
        1. Truncate according to global params (provided to `enable_truncation`)
        2. Apply the PostProcessor
        3. Pad according to global params. (provided to `enable_padding`)
    """
    is_batch = True
    if isinstance(encoding, Encoding):
      encoding = [encoding]
      is_batch = False
    outputs = [
        self.tokenizer.post_process(i, add_special_tokens=add_special_tokens)
        for i in encoding
    ]
    return outputs if is_batch else outputs[0]

  def decode(self,
             ids: List[int],
             skip_special_tokens: Optional[bool] = True) -> List[str]:
    r""" Decode sequence of integer indices and return original sequence """
    is_batch = True
    if not isinstance(ids[0], (tuple, list, ndarray)):
      ids = [ids]
      is_batch = False
    outputs = self.tokenizer.decode_batch(
        ids, skip_special_tokens=skip_special_tokens)
    return outputs if is_batch else outputs[0]

  def create_dataset(self,
                     batch_size=64,
                     drop_remainder=False,
                     shuffle=1000,
                     prefetch=tf.data.experimental.AUTOTUNE,
                     cache='',
                     parallel=None,
                     partition='train',
                     inc_labels=False,
                     seed=1) -> tf.data.Dataset:
    r"""
    Arguments:
      partition : {'train', 'valid', 'test'}
      inc_labels : a Boolean or Scalar. If True, return both image and label,
        otherwise, only image is returned.
        If a scalar is provided, it indicate the percent of labelled data
        in the mask.

    Return :
      tensorflow.data.Dataset :
        image - `(tf.float32, (None, 64, 64, 1))`
        label - `(tf.float32, (None, 5))`
        mask  - `(tf.bool, (None, 1))` if 0. < inc_labels < 1.
      where, `mask=1` mean labelled data, and `mask=0` for unlabelled data
    """
    inc_labels = float(inc_labels)
    gen = tf.random.experimental.Generator.from_seed(seed=seed)
    x = self.transform(partition)
    y = get_partition(partition,
                      train=self.train_labels,
                      valid=self.valid_labels,
                      test=self.test_labels)
    # remove empty docs
    indices = np.array(np.sum(x, axis=-1) > 0).ravel()
    x = x[indices]
    if len(y) > 0:
      y = y[indices]
    # convert to one-hot
    if inc_labels > 0 and len(y) > 0 and y.ndim == 1:
      y = one_hot(y, self.n_labels)

    def _process(*data):
      data = tuple([
          tf.cast(
              tf.sparse.to_dense(i) if isinstance(i, tf.SparseTensor) else i,
              tf.float32) for i in data
      ])
      if inc_labels:
        if 0. < inc_labels < 1.:  # semi-supervised mask
          mask = gen.uniform(shape=(1,)) < inc_labels
          return dict(inputs=tuple(data), mask=mask)
        return data
      return data[0]

    # prepare the sparse matrices
    if isinstance(x, spmatrix):
      x = tf.SparseTensor(indices=sorted(zip(*x.nonzero())),
                          values=x.data,
                          dense_shape=x.shape)
    ds = tf.data.Dataset.from_tensor_slices(x)
    if inc_labels > 0:
      if isinstance(y, spmatrix):
        y = tf.SparseTensor(indices=sorted(zip(*y.nonzero())),
                            values=y.data,
                            dense_shape=y.shape)
      y = tf.data.Dataset.from_tensor_slices(y)
      ds = tf.data.Dataset.zip((ds, y))
    # configurate dataset
    ds = ds.map(_process, parallel)
    if cache is not None:
      ds = ds.cache(str(cache))
    # shuffle must be called after cache
    if shuffle is not None and shuffle > 0:
      ds = ds.shuffle(int(shuffle))
    ds = ds.batch(batch_size, drop_remainder)
    if prefetch is not None:
      ds = ds.prefetch(prefetch)
    return ds


# ===========================================================================
# Others
# ===========================================================================
class ImdbReview(NLPDataset):

  def __init__(self):
    import tensorflow_datasets as tfds
    train = tfds.load('imdb_reviews', split='train')
    test = tfds.load('imdb_reviews', split='test')
    print(train)


class TinyShakespear(NLPDataset):

  def __init__(self):
    import tensorflow_datasets as tfds
    'test'
    'train'
    'validation'
    d = tfds.load(name='tiny_shakespeare')['train']
    d = d.map(lambda x: tf.strings.unicode_split(x['text'], 'UTF-8'))
    # train split includes vocabulary for other splits
    vocabulary = sorted(set(next(iter(d)).numpy()))
    d = d.map(lambda x: {'cur_char': x[:-1], 'next_char': x[1:]})
    d = d.unbatch()
    seq_len = 100
    batch_size = 2
    d = d.batch(seq_len)
    d = d.batch(batch_size)


class MathArithmetic(NLPDataset):

  def __init__(self):
    import tensorflow_datasets as tfds
    train_examples, val_examples = tfds.load('math_dataset/arithmetic__mul',
                                             split=['train', 'test'],
                                             as_supervised=True)
