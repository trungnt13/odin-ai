import os
import pickle
from numbers import Number
from types import MethodType
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix, spmatrix
from six import string_types
from sklearn.datasets import fetch_20newsgroups
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from odin.fuel.dataset_base import IterableDataset, get_partition
from odin.utils import one_hot

try:
  from tokenizers import Encoding
  from tokenizers.implementations import BaseTokenizer
except ImportError:
  Encoding = "Encoding"
  BaseTokenizer = object


# ===========================================================================
# Helpers
# ===========================================================================
def _prepare_doc(document: str):
  if os.path.exists(document) and os.path.isfile(document):
    with open(document, "r") as f:
      return f.read()
  return document


class NLPDataset(IterableDataset):

  def __init__(self,
               tokenizer: Optional[BaseTokenizer] = None,
               max_length: Optional[int] = None,
               cache_path: str = "~/nlp_data"):
    self._cache_path = os.path.abspath(os.path.expanduser(cache_path))
    if tokenizer is None and \
      os.path.exists(self.tokenizer_path) and \
        os.path.isdir(self.tokenizer_path):
      pkl_path = os.path.join(self.tokenizer_path, "tokenizer.pkl")
      if os.path.isfile(pkl_path):
        with open(pkl_path, 'rb') as f:
          tokenizer = pickle.load(f)
    self._tokenizer = tokenizer
    # defaults
    self._labels = []
    self._max_length = max_length
    # vectorizers
    self._count_vectorizers: Dict[str, CountVectorizer] = dict()
    self._tfidf_vectorizers: Dict[str, TfidfVectorizer] = dict()

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

  def __call__(self,
               X: Union[str, List[str]],
               y: Optional[List[str]] = None,
               padding: bool = True,
               return_type="ids",
               **kwargs) -> Union[List[Encoding], List[str], List[int]]:
    r""" Transform inputs text and labels into vectors

    Arguments:
      X : input documents
      y : labels (optional)
      padding : perform post-process padding
      return_type : str {'ids', 'tokens', 'encoding', 'tf', 'tfidf'}
      kwargs : dict, extra keywords for `NLPDataset.vectorize` method
    """
    X = np.asarray(self.encode(X))
    # store the documents
    if y is not None:
      y = np.asarray(y)
      ids = self.labels_indices
      y = one_hot(np.asarray([self.labels_indices[i] for i in y]),
                  self.n_labels)
    # prepare the padding
    if self.tokenizer.padding and padding:
      X = self.post_process(X)
    if return_type == "ids":
      X = np.asarray([i.ids for i in X])
    elif return_type == "tokens":
      X = np.asarray([i.tokens for i in X])
    elif return_type == "encoding":
      pass
    elif return_type == "tf":
      X = self.vectorize((i.tokens for i in X), algorithm='tf', **kwargs)
    elif return_type == "tfidf":
      X = self.vectorize((i.tokens for i in X), algorithm='tfidf', **kwargs)
    else:
      raise NotImplementedError(f"No support for return_type='{return_type}'")
    return X if y is None else (X, y)

  def vectorize(
      self,
      documents: Optional[Union[str, List[str]]] = None,
      ngram_range: List[int] = (1, 1),
      max_df: float = 0.95,
      min_df: float = 2,
      max_features: int = 1000,
      sparse: str = 'csr',
      algorithm: str = 'tf',
  ) -> Union[spmatrix, CountVectorizer, TfidfVectorizer]:
    r""" Vectorizing the tokens vector

    Arguments:
      ngram_range : tuple (min_n, max_n), default=(1, 1)
          The lower and upper boundary of the range of n-values for different
          n-grams to be extracted. All values of n such that min_n <= n <= max_n
          will be used. For example an ``ngram_range`` of ``(1, 1)`` means only
          unigrams, ``(1, 2)`` means unigrams and bigrams, and ``(2, 2)`` means
          only bigrams.
          Only applies if ``analyzer is not callable``.

      max_df : float or int, default=1.0
          When building the vocabulary ignore terms that have a document
          frequency strictly higher than the given threshold (corpus-specific
          stop words).
          If float in range [0.0, 1.0], the parameter represents a proportion of
          documents, integer absolute counts.
          This parameter is ignored if vocabulary is not None.

      min_df : float or int, default=1
          When building the vocabulary ignore terms that have a document
          frequency strictly lower than the given threshold. This value is also
          called cut-off in the literature.
          If float in range of [0.0, 1.0], the parameter represents a proportion
          of documents, integer absolute counts.
          This parameter is ignored if vocabulary is not None.

      max_features : int, default=None
          If not None, build a vocabulary that only consider the top
          max_features ordered by term frequency across the corpus.

      sparse : str, {'csr', 'csc', 'coo', 'dok', 'lil'}.
        Sparse matrix format.

      algorithm: str, {'tfidf', 'tf', 'count'}
        strategy for vectorization.
    """
    algorithm = str(algorithm).lower().strip()
    assert algorithm in ('tf', 'count', 'tfidf'), \
      f"Only support 'count' (or 'tf') and 'tfidf' algorithm, given {algorithm}"
    key = f"{ngram_range}({min_df},{max_df}){max_features}"
    if algorithm in ('tf', 'count'):
      models = self._count_vectorizers
      algo = CountVectorizer
    else:
      models = self._tfidf_vectorizers
      algo = TfidfVectorizer
    if key not in models:
      vectorizer = algo(ngram_range=ngram_range,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features)
      models[key] = vectorizer
    vectorizer = models[key]
    if documents is not None:
      documents = (_prepare_doc(i)
                   if isinstance(i, string_types) else ' '.join(i)
                   for i in documents)
      try:
        check_is_fitted(vectorizer)
        x = vectorizer.transform(documents)
      except NotFittedError:
        x = vectorizer.fit_transform(documents)
      if sparse != 'csr':
        x = getattr(x, f'to{sparse.lower()}')()
      # sorted ensure right ordering for Tensorflow SparseTensor
      x.totensor = MethodType(
          lambda self: tf.SparseTensor(indices=sorted(zip(*self.nonzero())),
                                       values=x.data,
                                       dense_shape=x.shape), x)
      return x
    return vectorizer

  @property
  def labels(self) -> List[str]:
    return np.array(self._labels)

  @property
  def max_length(self) -> Union[None, int]:
    return self._max_length

  @property
  def cache_path(self) -> str:
    p = os.path.join(self._cache_path, self.__class__.__name__)
    if not os.path.exists(p):
      os.makedirs(p)
    return p

  @property
  def tokenizer_path(self) -> str:
    p = os.path.join(self.cache_path, 'tokenizer')
    if not os.path.exists(p):
      os.makedirs(p)
    return p

  @property
  def vocabulary(self) -> Dict[str, int]:
    return self.tokenizer.get_vocab()

  @property
  def vocabulary_size(self) -> int:
    return self.tokenizer.get_vocab_size()

  @property
  def tokenizer(self) -> BaseTokenizer:
    return self._tokenizer

  @tokenizer.setter
  def tokenizer(self, tk):
    # create new basics word piece tokenizer
    assert isinstance(tk, BaseTokenizer), \
      ("tokenizer must be instance of "
       "tokenizers.implementations.base_tokenizer.BaseTokenizer")
    self._tokenizer = tk

  def train_tokenizer(
      self,
      files_or_text: Union[str, List[str]],
      vocab_size: int = 10000,
      min_frequency: int = 2,
      limit_alphabet: int = 1000,
      initial_alphabet: List[str] = [],
      special_tokens: List[str] = [
          "[PAD]",
          "[UNK]",
          "[CLS]",
          "[SEP]",
          "[MASK]",
      ],
      show_progress: bool = True,
      wordpieces_prefix: str = "##",
  ):
    r""" Train the tokenizer """
    if self._tokenizer is None:
      from tokenizers import BertWordPieceTokenizer
      self._tokenizer = BertWordPieceTokenizer()
    if isinstance(files_or_text, string_types):
      files_or_text = [files_or_text]
    files = []
    path = os.path.join(self.cache_path, f"{self.__class__.__name__}_train.txt")
    for f in files_or_text:
      if not (os.path.exists(f) and os.path.isfile(f)):
        with open(path, "a") as fout:
          fout.write(f + "\n")
        files.append(path)
      else:
        files.append(f)
    files = list(set(files))
    self.tokenizer.train(files=files,
                         vocab_size=vocab_size,
                         min_frequency=min_frequency,
                         limit_alphabet=limit_alphabet,
                         initial_alphabet=initial_alphabet,
                         special_tokens=special_tokens,
                         wordpieces_prefix=wordpieces_prefix,
                         show_progress=show_progress)
    self.tokenizer.save_model(self.tokenizer_path)
    with open(os.path.join(self.tokenizer_path, 'tokenizer.pkl'), "wb") as f:
      pickle.dump(self.tokenizer, f)
    return self

  def enable_padding(self,
                     direction: Optional[str] = "right",
                     pad_to_multiple_of: Optional[int] = None,
                     pad_id: Optional[int] = 0,
                     pad_type_id: Optional[int] = 0,
                     pad_token: Optional[str] = "[PAD]",
                     max_length: Optional[int] = None,
                     stride: Optional[int] = 0,
                     strategy: Optional[str] = "longest_first"):
    r""" Change the padding and truncation strategy

    Arguments:
      stride: (`optional`) unsigned int:
          The length of the previous first sequence to be included
          in the overflowing sequence
      strategy: (`optional) str:
          Can be one of `longest_first`, `only_first` or `only_second`
    """
    if max_length is None:
      max_length = self.max_length
    self.tokenizer.enable_truncation(max_length=max_length,
                                     stride=stride,
                                     strategy=strategy)
    self.tokenizer.enable_padding(direction=direction,
                                  pad_to_multiple_of=pad_to_multiple_of,
                                  pad_id=pad_id,
                                  pad_type_id=pad_type_id,
                                  pad_token=pad_token,
                                  length=max_length)
    return self

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
    if not isinstance(ids[0], (tuple, list, np.ndarray)):
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
    factors = self._factors
    inc_labels = float(inc_labels)
    gen = tf.random.experimental.Generator.from_seed(seed=seed)

    def _process(data):
      image = tf.cast(data['image'], tf.float32)
      if inc_labels:
        label = tf.convert_to_tensor([data[i] for i in factors],
                                     dtype=tf.float32)
        if 0. < inc_labels < 1.:  # semi-supervised mask
          mask = gen.uniform(shape=(1,)) < inc_labels
          return dict(inputs=(image, label), mask=mask)
        return image, label
      return image

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
# Datasets
# ===========================================================================
class Newsgroup20(NLPDataset):

  def __init__(self,
               tokenizer: Optional[BaseTokenizer] = None,
               max_length: int = 1000,
               cache_path: str = "~/nlp_data"):
    super().__init__(tokenizer=tokenizer,
                     max_length=max_length,
                     cache_path=cache_path)
    data = fetch_20newsgroups(shuffle=True,
                              random_state=1,
                              remove=('headers', 'footers', 'quotes'),
                              return_X_y=False)
    sentences, topics = data.data, data.target
    mask = np.array([len(i.split(' ')) for i in sentences]) < int(max_length)
    sentences = np.asarray(sentences)[mask]
    topics = np.asarray(topics)[mask]
    names = data.target_names
    topics = np.array([names[i] for i in topics])
    X_train, X_test, y_train, y_test = train_test_split(sentences,
                                                        topics,
                                                        test_size=0.2,
                                                        stratify=topics,
                                                        shuffle=True,
                                                        random_state=0)
    self.train = (X_train, y_train)
    self.test = (X_test, y_test)
    self._labels = sorted(np.unique(topics))
    if self._tokenizer is None:
      self.train_tokenizer(X_train)
    self.enable_padding()


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
