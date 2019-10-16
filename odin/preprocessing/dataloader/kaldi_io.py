# Copyright (c) 2019 Ville Vestman
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
#
# @Author: Ville Vestman (http://cs.uef.fi/~vvestman/)
# @Modified by: Trung Ngo (https://github.com/trungnt13)
from __future__ import absolute_import, division, print_function

import inspect
import random
import sys
from collections import defaultdict
from copy import deepcopy
from functools import partial
from multiprocessing import cpu_count
from numbers import Number
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from six import string_types
from torch.utils import data
from tqdm import tqdm

from odin import backend as bk
from odin.preprocessing.base import Extractor
from odin.utils import as_tuple

__all__ = ['count_frames', 'KaldiFeaturesReader', 'KaldiDataset']


# ===========================================================================
# Helper
# ===========================================================================
def _collater(batch):
  """ In this "hack", batch is already formed in the DataSet object
  (batch consists of a single element, which is actually the batch itself).
  """
  return batch[0]


def _check_pykaldi():
  try:
    import kaldi
  except ImportError:
    raise ImportError("This module require pykaldi-cpu from "
                      "https://anaconda.org/Pykaldi/pykaldi-cpu")


def count_frames(specifiers: List[str],
                 is_matrix: bool = False,
                 is_bool_index: bool = True,
                 progressbar: bool = False,
                 num_workers: int = 3) -> List[int]:
  """
  Parameters
  ----------
  specifiers : list of `str`
    list of sepcifier `["raw_mfcc_voxceleb.1.ark:42", ...]`
  is_matrix : `bool` (default=`False`)
    input data is matrix or vector
  is_bool_index : `bool` (default=`True`)
    if `True`, the loaded data is boolean index of speech activity detection,
    the length of audio file is calculated by summing the index array.

  Return
  ------
  List of integer (i.e. the frame count)
  """
  _check_pykaldi()
  import kaldi.util.io as kio

  frame_counts = []
  read = kio.read_matrix if bool(is_matrix) else kio.read_vector
  progress = tqdm(total=len(specifiers),
                  desc="Kaldi counting frame",
                  disable=not progressbar,
                  mininterval=0.0,
                  maxinterval=10.0)

  def _count(specs):
    res = []
    for idx, s in specs:
      # both feature and VAD is provided, then get the vad only
      dat = read(s).numpy()
      if is_bool_index:
        n = np.sum(dat)
      else:
        n = dat.size
      res.append((int(idx), n))
    return res

  jobs = np.array_split([(i, s) for i, s in enumerate(specifiers)],
                        num_workers * 25)
  if num_workers == 1:
    for j in jobs:
      for r in _count(j):
        frame_counts.append(r)
      progress.update(n=len(j))
  else:
    from odin.utils.mpi import MPI
    for r in MPI(jobs=jobs, func=_count, ncpu=num_workers, batch=1):
      progress.update(n=len(r))
      frame_counts.extend(r)
  return [i[1] for i in sorted(frame_counts)]


# ===========================================================================
# Feature configurer
# ===========================================================================
class KaldiFeaturesReader(Extractor):
  """ This class is used to read features extracted by KALDI recipe.
  After loading the features are post-procesessed using delta, shifted delta
  or sliding-window cepstral mean and/or variance normalization.

  If all ultilities are enabled, they will be executed in following order:
  `delta` -> `shifted delta` -> `sliding-window cmn`

  Parameters
  ----------
  delta_order : {None, `int`}
    `pykaldi` default is `2`
  delta_window : {None, `int`}
    `pykaldi` default is `2`
  sdelta_block_shift : {None, `int`}
    `pykaldi` default is `3`
  sdelta_num_blocks : {None, `int`}
    `pykaldi` default is `7`
  sdelta_window : {None, `int`}
    `pykaldi` default is `1`
  cmn_window : {None, `int`}
    `pykaldi` default is `600`
  cmn_min_window : `int` (default=`100`)
  cmn_center : `bool` (default=`False`)
  cmn_normalize_variance : `bool` (default=`False`)
  is_matrix : `bool` (default=`True`)
    if the input is a matrix or a vector

  Example
  -------
  >>> feat_loader = kaldi_io.KaldiFeaturesReader(cmn_window=300,
  >>>                                            cmn_center=True,
  >>>                                            cmn_normalize_variance=False,
  >>>                                            cmn_min_window=100)
  >>> vad_loader = kaldi_io.KaldiFeaturesReader(is_matrix=False)
  >>> x = feat_loader.transform('raw_mfcc_voxceleb.1.ark:42')
  >>> v = vad_loader.transform('vad_voxceleb.1.ark:42')
  >>> x = x[v.astype(bool)]

  """

  def __init__(self,
               name,
               delta_order: Optional[int] = None,
               delta_window: Optional[int] = None,
               sdelta_block_shift: Optional[int] = None,
               sdelta_num_blocks: Optional[int] = None,
               sdelta_window: Optional[int] = None,
               cmn_window: Optional[int] = None,
               cmn_min_window: int = 100,
               cmn_center: bool = False,
               cmn_normalize_variance: bool = False,
               is_matrix: bool = True):
    assert isinstance(name, string_types), \
      'a short name (description) must be given for KaldiFeaturesReader'
    super(KaldiFeaturesReader, self).__init__(name=name)
    _check_pykaldi()
    import kaldi.feat.functions as featfuncs
    import kaldi.util.io as kio
    self._kio = kio
    self._featfuncs = featfuncs
    self.is_matrix = bool(is_matrix)
    # ====== prepare the features option ====== #
    self.delta_opts = None
    self.sdelta_opts = None
    self.cmn_opts = None
    if delta_order and delta_window:
      self.delta_opts = featfuncs.DeltaFeaturesOptions(order=int(delta_order),
                                                       window=int(delta_window))
    if sdelta_block_shift and sdelta_num_blocks and sdelta_window:
      self.sdelta_opts = featfuncs.ShiftedDeltaFeaturesOptions()
      self.sdelta_opts.block_shift = int(sdelta_block_shift)
      self.sdelta_opts.num_blocks = int(sdelta_num_blocks)
      self.sdelta_opts.window = int(sdelta_window)
    if cmn_window and cmn_min_window:
      self.cmn_opts = featfuncs.SlidingWindowCmnOptions()
      self.cmn_opts.cmn_window = int(cmn_window)
      self.cmn_opts.min_window = bool(cmn_min_window)
      self.cmn_opts.center = bool(cmn_center)
      self.cmn_opts.normalize_variance = bool(cmn_normalize_variance)

  def transform(self, specifier, is_matrix=None):
    """
    specifier : `str`
      file path and location joined by ':', for example:
        "/kaldi_features/voxceleb/raw_mfcc_voxceleb.1.ark:42"
        "/kaldi_features/voxceleb/vad_voxceleb.1.ark:42"
    """
    assert isinstance(specifier, string_types), "specifier must be a string"
    if is_matrix is None:
      is_matrix = self.is_matrix
    # ====== load features  ====== #
    if is_matrix:
      feats = self._kio.read_matrix(specifier)
    else:
      feats = self._kio.read_vector(specifier)
    # ====== post-processing ====== #
    if self.delta_opts is not None:
      feats = self._featfuncs.compute_deltas(self.delta_opts, feats)
    if self.sdelta_opts is not None:
      feats = self._featfuncs.compute_shift_deltas(self.sdelta_opts, feats)
    if self.cmn_opts is not None:
      self._featfuncs.sliding_window_cmn(self.cmn_opts, feats, feats)
    # ====== return results ====== #
    return feats.numpy()


# ===========================================================================
# Dataset
# ===========================================================================
def _xvector_processing(data, labels):
  return [
      torch.from_numpy(np.transpose(np.dstack(dat), (2, 0, 1)))
      for dat in data.values()
  ], labels


def _ivector_processing(data, labels):
  n_samples = [len(i) for i in list(data.values())[0]]
  # repeat the labels
  if labels is not None:
    new_labels = []
    for n, lab in zip(n_samples, labels):
      new_labels += [lab] * n
    labels = np.asarray(new_labels)
  return [torch.from_numpy(np.vstack(dat)) for dat in data.values()], labels


def _flatten_processing(data, labels):
  X = []
  for dat in data.values():
    X.extend(dat)
  return X, labels


def _select_post_processing(text):
  text = text.strip().lower()
  func_name = '_%s_processing' % text
  globs = globals()
  if func_name not in globs:
    raise RuntimeError("No support for predefined post_processing='%s'" % text)
  func = globs[func_name]
  assert inspect.isfunction(func), \
    "Object with name '%s' is not a function, invalid post_processing='%s'" % \
      (func_name, text)
  return func


class KaldiDataset(data.Dataset):
  """ Without any `post_processing` a dictionary mapping from
  `KaldiFeaturesReader.name` to list of loaded features grouped into minibatch
  will be returned (note: the labels will be returned with key 'labels'
  if `return_labels=True`)

  Check document of `post_processing` for better understand of the return of
  `KaldiDataset`.

  Parameters
  ----------
  specifier_description : dict
    mapping from `KaldiFeaturesReader` to list of Kaldi specifier in form of
    string (e.g. "/kaldi_features/voxceleb/raw_mfcc_voxceleb.1.ark:42")
  sad_name : `str` (default=`None`)
    name of the SAD `KaldiFeaturesReader`. If not given, that means the SAD
    (or VAD) is not provided.
  labels : list of `int` (default=`None`)
    integer label for each utterance, could be speaker label, language label
    or anything.
  shuffle : `bool` (default=`True`)
    if shuffle the utterance list before organizing into minibatches
    note: the order of the utterances will be reserved with `shuffle=False`
  batch_size : `int` (default=`64`)
    number of examples in single mini-batch
  post_processing : {`callable`, 'xvector', 'ivector', `None`} (default=`None`)
    post processing is a function take into account the loaded `features` and the
    `labels`
    - features : is a mapping from `KaldiFeaturesReader.name` to a mini-batch
      (i.e. a list) of matrix or vector.
    - labels : list of integer label for each example in the mini-batch. The
      labels is `None` if not provided
    a string, alias for predefined post-processor can be provided:
    - 'xvector' : minibatch return in a tensor `[batch_size, time_frames, feature_size]`
    - 'ivector' : minibatch return in a matrix `[time_frames, feature_size]`
    - 'flatten' : flatten the return dictionary of features into a list
    if `post_processing=None`, return a dictionary mapping from
    `KaldiFeaturesReader.name` to features mini-batch (a list of matrix or vector)
  clipping : tuple of two `int` (default=`None`)
    the utterance length is randomly clipped to the given range.
    If `None`, no clipping is performed
  clipping_per_batch : `bool` (default=`True`)
    if `True`, all utterances in the same minibatch are clipped to the same
    length (for 'xvector').
    Otherwise, each utterance is clipped to different length (for 'ivector').
  utt_per_label_in_epoch : {`int`, `None`} (default=`np.inf`)
    number of utterances per speaker (i.e. label) will appear in one epoch.
    if `None`, or non-positive number is provided, substitute with `np.inf`
  min_label_per_batch : `int`
    Minimum number of speakers found in single mini-batch. If any batch doesn't
    match the condition, it is ignored.
  min_frames_per_utt : {`int`, `None`} (default=`None`)
    filtering using the requirement of minimum number of frames per utterance.
    if `None, no filtering is performed
  min_utt_per_label :  {`int`, `None`} (default=`None`)
    filtering using the requirement of minimum number of utterances per speader
    (or label). Any speaker with less than required amount of utterances
    will be ignored.
    if `None, no filtering is performed
  batch_strategy : {'naive', 'stratify'}
    'utt' - each minibatch is one utterance (`batch_size` is not in used here)
    'naive' - just split the list of all utterances into minibatches
    'stratify' - focus on the label for sampling
  batch_drop_last : `bool` (default=`False`)
    set to ``True`` to drop the last incomplete batch,
    if the dataset size is not divisible by the batch size. If ``False`` and
    the size of dataset is not divisible by the batch size, then the last batch
    will be smaller. (default: ``False``)
  return_labels : `bool` (default=`True`)
    if `True`, return the labels if available in `__getitem__`
  seed : {`int`, `numpy.random.RandomState`}
    random seed
  verbose : `bool` (default=`False`)
    enable logging

  Example
  -------
  >>> specifier_description = {
  >>>     kaldi_io.KaldiFeaturesReader(cmn_window=300,
  >>>                                  cmn_center=True,
  >>>                                  cmn_normalize_variance=False,
  >>>                                  cmn_min_window=100,
  >>>                                  name='mfcc'): [
  >>>         i[0] for i in training_segment_ids
  >>>     ],
  >>>     kaldi_io.KaldiFeaturesReader(is_matrix=False, name='sad'): [
  >>>         i[1] for i in training_segment_ids
  >>>     ],
  >>> }
  >>> dataset = kaldi_io.KaldiDataset(specifier_description,
  >>>                                 labels=training_labels,
  >>>                                 sad_name='sad',
  >>>                                 post_processing='ivector',
  >>>                                 utt_per_label_in_epoch=80,
  >>>                                 min_label_per_batch=2,
  >>>                                 min_frames_per_utt=100,
  >>>                                 min_utt_per_label=100,
  >>>                                 clipping=(200, 400),
  >>>                                 clipping_per_batch=False,
  >>>                                 batch_strategy='stratify',
  >>>                                 batch_drop_last=True,
  >>>                                 verbose=True)
  >>> loader = dataset.create_dataloader()
  >>> for data, labels in tqdm(dataset):
  >>>   pass

  """

  def __init__(self,
               specifier_description: Dict[KaldiFeaturesReader, List[str]],
               sad_name: str = None,
               labels: Optional[List[int]] = None,
               shuffle=False,
               batch_size=64,
               post_processing=None,
               clipping=None,
               clipping_per_batch=True,
               utt_per_label_in_epoch=np.inf,
               min_label_per_batch=1,
               min_frames_per_utt=None,
               min_utt_per_label=None,
               batch_strategy='naive',
               batch_drop_last=False,
               return_labels=True,
               seed=8,
               verbose=False,
               **kwargs):
    _check_pykaldi()
    if not isinstance(specifier_description, dict) or \
      (not all(isinstance(loader, KaldiFeaturesReader) and
               isinstance(specs, (tuple, list)) and
               all(isinstance(s, string_types) for s in specs)
       for loader, specs in specifier_description.items())):
      raise ValueError("specifier_description is mapping from "
                       "KaldiFeaturesReader to list of string kaldi specifier")
    self.specifier_description = specifier_description
    # ====== number of utterances ====== #
    self._n_utterances = [len(spec) for spec in specifier_description.values()]
    assert len(set(self._n_utterances)) == 1, \
      "length of the specifiers list mismatch: %s" % str(self._n_utterances)
    self._n_utterances = self._n_utterances[0]
    # ====== randomization ====== #
    self.shuffle = bool(shuffle)
    self._rand = seed if isinstance(seed, np.random.RandomState) else\
      np.random.RandomState(seed=seed)
    self.seed = seed
    # ====== check the labels ====== #
    if labels is not None:
      assert len(labels) == self._n_utterances, \
        "Number of labels and number of utterances mismatch: %d vs %d" % \
          (len(labels), self._n_utterances)
    self.labels = np.asarray(labels) if isinstance(labels, (tuple, list)) \
      else labels
    self.return_labels = bool(return_labels)
    # label_id -> [utt_id, ...]
    self.lab2utt = defaultdict(list)
    self.utt2lab = {}
    if self.labels is not None:
      for utt_id, label in enumerate(self.labels):
        self.lab2utt[label].append(utt_id)
        self.utt2lab[utt_id] = label
    # ====== get the frame count ====== #
    specs = list(specifier_description.values())[0]
    is_sad_provided = False
    for key in specifier_description:
      if sad_name is not None and sad_name == key.name:
        specs = specifier_description[key]
        is_sad_provided = True
    if is_sad_provided:
      self._sad_name = sad_name
    else:
      self._sad_name = None
    if 'frame_counts' in kwargs:
      self.frame_counts = list(kwargs['frame_counts'])
    else:
      self.frame_counts = count_frames(specs,
                                       is_matrix=not is_sad_provided,
                                       is_bool_index=is_sad_provided,
                                       progressbar=verbose,
                                       num_workers=cpu_count() - 2)
    self.frame_counts = tuple(self.frame_counts)
    assert len(self.frame_counts) == self._n_utterances, 'Invalid frame_counts!'
    # ====== post_processing ====== #
    if callable(post_processing):
      self._post_processing = post_processing
    elif isinstance(post_processing, string_types):
      self._post_processing = _select_post_processing(post_processing)
    else:
      self._post_processing = None
    # ====== batch configuration ====== #
    self.batch_size = int(batch_size)
    self.batch_strategy = str(batch_strategy).strip().lower()
    self.batch_drop_last = bool(batch_drop_last)
    # ====== for filtering ====== #
    self.utt_per_label_in_epoch = float(utt_per_label_in_epoch) \
      if isinstance(utt_per_label_in_epoch, Number) else np.inf
    self.min_label_per_batch = max(1, int(min_label_per_batch))
    self.clipping = None if clipping is None else as_tuple(clipping, t=int, N=2)
    self.clipping_per_batch = bool(clipping_per_batch)
    self.min_frames_per_utt = None if min_frames_per_utt is None else \
      int(min_frames_per_utt)
    self.min_utt_per_label = None if min_utt_per_label is None else \
      int(min_utt_per_label)
    # ====== shuffle and batching ====== #
    self.verbose = bool(verbose)
    self.reset()

  def post_processing(self,
                      data: Dict[str, List[np.array]],
                      labels: Optional[np.array] = None):
    if self._post_processing is not None:
      data, labels = self._post_processing(data, labels)
    return data, labels

  # ====== shuffling ====== #
  def reset(self):
    """ This method will rebuild the meta-minibatches """
    # [(utt_id, utt_id, ...), ...]
    # each (utt_id, utt_id, ...) has length of minibatch size
    self._minibatches = []
    # ====== shuffling the data ====== #
    # everything is performed based-on the indices (utterance index or ID)
    # no need to change the original list
    if self.shuffle:
      if self.verbose:
        print("Shuffling the utterances ...")
      indices = self._rand.permutation(
          np.arange(self._n_utterances, dtype='int64'))
    else:
      indices = np.arange(self._n_utterances, dtype='int64')
    # ====== filtering ====== #
    if self.min_frames_per_utt is not None:
      n_original = len(indices)
      indices = [
          i for i in indices if self.frame_counts[i] > self.min_frames_per_utt
      ]
      n_new = len(indices)
      if self.verbose:
        print("Filtering min_frames_per_utt=%d - original:%d  new:%d" %
              (self.min_frames_per_utt, n_original, n_new))
    #
    if self.min_utt_per_label is not None and self.labels is not None:
      n_original = len(indices)
      remove_indices = {}
      for spk, utt_list in self.lab2utt.items():
        if len(utt_list) < self.min_utt_per_label:
          remove_indices.update({utt_id: True for utt_id in utt_list})
      indices = [i for i in indices if i not in remove_indices]
      n_new = len(indices)
      if self.verbose:
        print("Filtering min_utt_per_label=%d - original:%d  new:%d" %
              (self.min_utt_per_label, n_original, n_new))
    # final utterances, stored as index in the utterance list
    self._filtered_utt_id = np.array(indices, dtype='int64')
    # ====== create the batches ====== #
    attr_name = '_strategy_%s' % self.batch_strategy
    if not hasattr(self, attr_name):
      raise RuntimeError("No support for strategy with name: '%s'" % attr_name)
    getattr(self, attr_name)()
    # ====== checking if strategy return right results ====== #
    assert len(self._minibatches) > 0, \
      "Batch specifier must be a list of tuples that contains multiple " + \
        "utterance ID for minibatch"
    # ====== random clipping ====== #
    # we need to make sure all utterances in the same minibatch got the same
    # clipping length
    random.seed(self._rand.randint(0, 1e8))
    self._minibatches_clipping = defaultdict(list)

    if self.clipping is not None:
      n_original = sum(len(batch) for batch in self._minibatches)
      utt_length = self._rand.randint(low=self.clipping[0],
                                      high=self.clipping[1] + 1,
                                      size=(n_original,),
                                      dtype='int64').tolist()

      new_minibatches = []
      minibatches_clipping = []
      # random a big chunk of clip_length is faster
      for batch, clip_length in zip(
          self._minibatches,
          self._rand.randint(low=self.clipping[0],
                             high=self.clipping[1] + 1,
                             size=(len(self._minibatches),),
                             dtype='int64')):
        # iterate through batch to get different cut of utterances
        new_batch = []
        clipping_points = []
        for utt_id in batch:
          frame_count = self.frame_counts[utt_id]
          # differnt clipping length for each utterance
          if not self.clipping_per_batch:
            clip_length = utt_length.pop()
          # not enough frames
          if frame_count < clip_length:
            remove_indices[utt_id] = True
          # select random start and end point based on clip_length and frame_count
          else:
            # note: the random.randint include both end point, no need for +1
            start_point = random.randint(0, frame_count - clip_length)
            end_point = start_point + clip_length
            clipping_points.append((start_point, end_point))
            new_batch.append(utt_id)
        # end of 1-minibatch
        new_minibatches.append(new_batch)
        minibatches_clipping.append(clipping_points)
      # end of clipping and filtering
      self._minibatches = new_minibatches
      self._minibatches_clipping = minibatches_clipping
      # show log if verbose
      if self.verbose:
        n_new = sum(len(batch) for batch in self._minibatches)
        print("Filtering clipping=%s - original:%d  new:%d" %
              (self.clipping, n_original, n_new))

  # ====== batch strategy ====== #
  def _strategy_utt(self):
    for utt_id in self._filtered_utt_id:
      self._minibatches.append([utt_id])

  def _strategy_naive(self):
    """ modify `_minibatches` to a list of tuple of utterances' ID """
    for start in range(0, len(self._filtered_utt_id), self.batch_size):
      batch = self._filtered_utt_id[start:start + self.batch_size]
      if len(batch) < self.batch_size and self.batch_drop_last:
        continue
      self._minibatches.append(batch)

  def _strategy_stratify(self):
    """ modify `_minibatches` to a list of tuple of utterances' ID """
    assert self.labels is not None, \
      "Labels must be provided for 'stratify' batch strategy"
    # need to update lab2utt with the new filtered utterances' ID
    lab2utt = defaultdict(list)
    for utt_id in self._filtered_utt_id:
      label = self.utt2lab[utt_id]
      lab2utt[label].append(utt_id)

    all_lab_utt = list(lab2utt.items())
    self._rand.shuffle(all_lab_utt)

    max_utt = self.utt_per_label_in_epoch
    min_utt = self.min_label_per_batch
    utt_count = 0
    while True:
      utt_list = [
          utt_list.pop()
          for lab_id, utt_list in all_lab_utt
          if len(utt_list) > 0
      ]
      if len(utt_list) < min_utt:
        break
      # split into mini-batches
      for start in range(0, len(utt_list), self.batch_size):
        batch = utt_list[start:start + self.batch_size]
        if len(batch) < self.batch_size and self.batch_drop_last:
          continue
        self._minibatches.append(batch)

      # breaking conditions
      utt_count += 1
      if utt_count >= max_utt:
        break

  # ====== dataset methods ====== #
  def __len__(self):
    return len(self._minibatches)

  def __getitem__(self, index):
    feat_list = defaultdict(list)
    # list of utterance's index
    batch = self._minibatches[index]
    # store (start, end) tuple for each utterance in the batch
    clipping = self._minibatches_clipping[index]

    sad = None
    for loader, specs in self.specifier_description.items():
      name = loader.name.lower()
      if name != self._sad_name:
        feat_list[name] = [loader.transform(specs[utt_id]) for utt_id in batch]
      else:
        sad = [
            np.asarray(loader.transform(specs[utt_id]), dtype=bool)
            for utt_id in batch
        ]

    # applying sad
    if sad is not None:
      feat_list = {
          name: [x[s] for x, s in zip(data_list, sad)
                ] for name, data_list in feat_list.items()
      }

    # applying clipping
    if len(clipping) > 0:
      feat_list = {
          name: [x[start:end] for x, (start, end) in zip(data_list, clipping)
                ] for name, data_list in feat_list.items()
      }

    # post processing
    if self.labels is not None:
      labels = self.labels[batch]
    else:
      labels = None
    features, labels = self.post_processing(feat_list, labels)

    # appending the labels to the returns in different ways
    if labels is not None and self.return_labels:
      labels = torch.from_numpy(labels)
      if isinstance(features, dict):
        features['labels'] = labels
      elif isinstance(features, (tuple, list)):
        features = tuple(features) + (labels,)
      else:
        features = (features, labels)
    return features

  # ====== data loader ====== #
  def copy(self, seed=None) -> 'KaldiDataset':
    return KaldiDataset(
        specifier_description={
            reader: deepcopy(specs)
            for reader, specs in self.specifier_description.items()
        },
        sad_name=self._sad_name,
        labels=deepcopy(self.labels),
        shuffle=self.shuffle,
        batch_size=self.batch_size,
        post_processing=self._post_processing,
        clipping=self.clipping,
        clipping_per_batch=self.clipping_per_batch,
        utt_per_label_in_epoch=self.utt_per_label_in_epoch,
        min_label_per_batch=self.min_label_per_batch,
        min_frames_per_utt=self.min_frames_per_utt,
        min_utt_per_label=self.min_utt_per_label,
        batch_strategy=self.batch_strategy,
        batch_drop_last=self.batch_drop_last,
        return_labels=self.return_labels,
        seed=self.seed if seed is None else seed,
        verbose=self.verbose,
        # no need re-calculating frame counts
        frame_counts=self.frame_counts)

  def create_dataloader(
      self,
      reset: bool = False,
      copy: bool = False,
      num_workers: Union[str, int] = 'max',
      seed=None,
  ) -> data.DataLoader:
    """ Since the `KaldiDataset` create and return the batch, not a single
    example, this function is utilized to create `pytorch.DataLoader` that
    is compatible.

    Parameters
    ----------
    reset : `bool`
      reset the dataset before create the `DataLoader`
    copy : `bool`
      return the `DataLoader` based on the copy version of this `Dataset`
    num_workers : {`int`, 'max'}
      if 'max', use `cpu_count() - 2` workers, otherwise, number of processes
    seed : {`int`, `numpy.random.RandomState`}
      only used for new instance if `copy=True`
    """
    num_workers = cpu_count() - 2 \
      if isinstance(num_workers, string_types) and num_workers == 'max' else \
        int(num_workers)
    ds = self.copy(seed=seed) if copy else self
    if reset:
      ds.reset()
    return data.DataLoader(self,
                           batch_size=1,
                           shuffle=False,
                           num_workers=num_workers,
                           collate_fn=_collater)
