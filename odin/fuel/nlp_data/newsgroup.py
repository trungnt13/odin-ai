from typing import Iterable, Union

import numpy as np
from numpy import ndarray
from odin.fuel.nlp_data._base import NLPDataset
from odin.utils import one_hot
from scipy.sparse import spmatrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split


class Newsgroup20(NLPDataset):
  r""" Categories:
    - alt.atheism
    - misc.forsale
    - soc.religion.christian
    - comp.graphics, comp.os.ms-windows.misc, comp.sys.ibm.pc.hardware,
        comp.sys.mac.hardware, comp.windows.x
    - rec.autos, rec.motorcycles, rec.sport.baseball, rec.sport.hockey
    - sci.crypt, sci.electronics, sci.med, sci.space
    - talk.politics.guns, talk.politics.mideast, talk.politics.misc,
        talk.religion.misc
  """

  def __init__(self,
               algorithm='count',
               vocab_size: int = 2000,
               min_frequency: int = 2,
               max_frequency: float = 0.95,
               max_length: int = 500,
               cache_path: str = "~/nlp_data/newsgroup20",
               **kwargs):
    categorices = kwargs.pop('categorices', None)
    super().__init__(algorithm=algorithm,
                     vocab_size=vocab_size,
                     min_frequency=min_frequency,
                     max_frequency=max_frequency,
                     max_length=max_length,
                     cache_path=cache_path,
                     **kwargs)
    kw = dict(shuffle=True,
              random_state=1,
              categories=categorices,
              remove=('headers', 'footers', 'quotes'))
    data = fetch_20newsgroups(subset='train', return_X_y=False, **kw)
    X_train, y_train = data.data, data.target
    labels_name = data.target_names
    self.X_test, y_test = fetch_20newsgroups(subset='test',
                                             return_X_y=True,
                                             **kw)
    self.X_train, self.X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, shuffle=True, random_state=0)
    self._labels = np.array(labels_name)
    self.y_train = one_hot(y_train, len(self._labels))
    self.y_valid = one_hot(y_valid, len(self._labels))
    self.y_test = one_hot(y_test, len(self._labels))

  @property
  def train_text(self) -> Iterable[str]:
    for doc in self.X_train:
      yield doc

  @property
  def valid_text(self) -> Iterable[str]:
    for doc in self.X_valid:
      yield doc

  @property
  def test_text(self) -> Iterable[str]:
    for doc in self.X_test:
      yield doc

  @property
  def train_labels(self) -> Union[ndarray, spmatrix]:
    return self.y_train

  @property
  def valid_labels(self) -> Union[ndarray, spmatrix]:
    return self.y_valid

  @property
  def test_labels(self) -> Union[ndarray, spmatrix]:
    return self.y_test


class Newsgroup5(Newsgroup20):
  r""" Subset of 5 categories:
    - 'soc.religion.christian'
    - 'comp.graphics'
    - 'rec.sport.hockey'
    - 'sci.space'
    - 'talk.politics.guns'
  """

  def __init__(self,
               algorithm='count',
               vocab_size: int = 2000,
               min_frequency: int = 2,
               max_frequency: float = 0.95,
               max_length: int = 500,
               cache_path: str = "~/nlp_data/newsgroup5",
               **kwargs):
    super().__init__(algorithm=algorithm,
                     vocab_size=vocab_size,
                     min_frequency=min_frequency,
                     max_frequency=max_frequency,
                     max_length=max_length,
                     cache_path=cache_path,
                     categorices=[
                         'soc.religion.christian', 'comp.graphics',
                         'rec.sport.hockey', 'sci.space', 'talk.politics.guns'
                     ],
                     **kwargs)
