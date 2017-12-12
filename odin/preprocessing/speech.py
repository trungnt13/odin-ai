# -*- coding: utf-8 -*-
"""
Note
----
The following order is recommended for extracting spectra:
+ AudioReader:
    - Loading raw audio
    - remove DC offeset and dithering
    - preemphasis
+ SpectraExtractor (or CQTExtractor):
    - Extracting the Spectra
+ Rastafilt:
    - Rastafilt (optional for MFCC)
+ SADextractor:
    - Extracting SAD (optional)
+ Read3ColSAD:
    - Generating SAD labels from 3-cols files
+ DeltaExtractor (calculated before applying SAD)
    - Calculate Deltas (and shifted delta for MFCCs).
+ ApplyingSAD:
    - Applying SAD indexing on features
+ BNFExtractor:
    - Extracting bottleneck features
+ AcousticNorm
    - Applying CMVN and WCMVN (This is important so the SAD frames
    are not affected by the nosie).
"""
# ===========================================================================
# The waveform and spectrogram preprocess utilities is adapted from:
# [librosa](https://github.com/bmcfee/librosa)
# Copyright (c) 2016, librosa development team.
# Modified work Copyright 2016-2017 TrungNT
# ===========================================================================
from __future__ import print_function, division, absolute_import

import os
import re
import six
import math
import copy
import shutil
import inspect
import warnings
from collections import OrderedDict, Mapping, defaultdict

import numpy as np
import tensorflow as tf
from scipy.signal import lfilter

from odin import config
from odin.fuel import Dataset, MmapData, MmapDict
from odin.utils import (is_number, cache_memory, is_string, as_tuple,
                        get_all_files, is_pickleable, Progbar, mpi, ctext,
                        is_fileobj, batching)
from odin.utils.decorators import functionable
from .base import Extractor
from .signal import (smooth, pre_emphasis, spectra, vad_energy,
                     pitch_track, resample, rastafilt, mvn, wmvn,
                     shifted_deltas, stack_frames)
# import all OpenSMILE extractor
from ._opensmile import *


# ===========================================================================
# Predefined variables of speech datasets
# ===========================================================================
# ==================== Timit ==================== #
timit_61 = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay',
    'b', 'bcl', 'ch', 'd', 'dcl', 'dh', 'dx', 'eh', 'el', 'em', 'en',
    'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#', 'hh', 'hv', 'ih',
    'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow',
    'oy', 'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th',
    'uh', 'uw', 'ux', 'v', 'w', 'y', 'z', 'zh']
timit_39 = ['aa', 'ae', 'ah', 'aw', 'ay', 'b', 'ch', 'd',
    'dh', 'dx', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k',
    'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 'sil', 't',
    'th', 'uh', 'uw', 'v', 'w', 'y', 'z']
timit_map = {'ao': 'aa', 'ax': 'ah', 'ax-h': 'ah', 'axr': 'er',
    'hv': 'hh', 'ix': 'ih', 'el': 'l', 'em': 'm',
    'en': 'n', 'nx': 'n',
    'eng': 'ng', 'zh': 'sh', 'ux': 'uw',
    'pcl': 'sil', 'tcl': 'sil', 'kcl': 'sil', 'bcl': 'sil',
    'dcl': 'sil', 'gcl': 'sil', 'h#': 'sil', 'pau': 'sil', 'epi': 'sil'}


# ===========================================================================
# Basics
# ===========================================================================
def _read_pcm(path, encode):
  dtype = np.int16
  sr = None
  if encode is not None:
    if 'ulaw' in encode.lower():
      dtype = np.int8
      sr = 8000
    elif 'vast' in encode.lower():
      dtype = np.int16
      sr = 44000
  raw = np.memmap(path, dtype=dtype, mode='r')
  return raw, sr


def read(path_or_file, encode=None):
  # ====== check input ====== #
  if is_fileobj(path_or_file):
    f = path_or_file
    path = f.name
  elif os.path.isfile(path_or_file):
    f = open(path_or_file, 'rb')
    path = path_or_file
  else:
    raise ValueError("Invalid type of `path_or_file` %s" %
        str(type(path_or_file)))
  # support encode
  if encode not in (None, 'ulaw', 'vast'):
    raise ValueError("No support for encode: %s" % str(encode))
  # ====== read the audio ====== #
  if '.pcm' in path.lower():
    raw, sr = _read_pcm(f, encode=encode)
  else:
    import soundfile
    try:
      raw, sr = soundfile.read(f)
    except Exception as e:
      if '.sph' in f.name.lower():
        f.seek(0)
        raw, sr = _read_pcm(f, encode=encode)
      else:
        raise e
  # close file
  f.close()
  return raw, sr


def save(file_or_path, s, sr, subtype=None):
  '''
  Parameters
  ----------
  s : array_like
      The data to write.  Usually two-dimensional (channels x frames),
      but one-dimensional `data` can be used for mono files.
      Only the data types ``'float64'``, ``'float32'``, ``'int32'``
      and ``'int16'`` are supported.

      .. note:: The data type of `data` does **not** select the data
                type of the written file. Audio data will be
                converted to the given `subtype`. Writing int values
                to a float file will *not* scale the values to
                [-1.0, 1.0). If you write the value ``np.array([42],
                dtype='int32')``, to a ``subtype='FLOAT'`` file, the
                file will then contain ``np.array([42.],
                dtype='float32')``.
  subtype: str
      'PCM_24': 'Signed 24 bit PCM'
      'PCM_16': 'Signed 16 bit PCM'
      'PCM_S8': 'Signed 8 bit PCM'

  Return
  ------
  waveform (ndarray), sample rate (int)
  '''
  from soundfile import write
  return write(file_or_path, s, sr, subtype=subtype)


# ===========================================================================
# Helper function
# ===========================================================================
def _extract_s_sr(s_sr):
  if isinstance(s_sr, Mapping):
    s_sr = (s_sr['raw'], s_sr['sr'])
  elif not isinstance(s_sr, (tuple, list)) or \
  not isinstance(s_sr[0], np.ndarray) or \
  not is_number(s_sr[1]):
    raise ValueError("Input to SpectraExtractor must be a tuple, or list "
                     "of raw signal (ndarray) and sample rate (int).")
  s, sr = s_sr
  return s, int(sr)


def _extract_frame_step_length(sr, frame_length, step_length):
  # ====== check frame length ====== #
  if frame_length < 1.:
    frame_length = int(sr * frame_length)
  else:
    frame_length = int(frame_length)
  # ====== check step length ====== #
  if step_length is None:
    step_length = frame_length // 4
  elif step_length < 1.:
    step_length = int(sr * step_length)
  else:
    step_length = int(step_length)
  return frame_length, step_length


@cache_memory
def _num_two_factors(x):
  """return number of times x is divideable for 2"""
  if x <= 0:
    return 0
  num_twos = 0
  while x % 2 == 0:
    num_twos += 1
    x //= 2
  return num_twos


@cache_memory
def _max_fft_bins(sr, n_fft, fmax):
  return [i + 1 for i, j in enumerate(np.linspace(0, float(sr) / 2, int(1 + n_fft // 2),
                                      endpoint=True)) if j >= fmax][0]


def audio_segmenter(files, outpath, max_duration,
                    sr=None, sr_new=None, best_resample=True,
                    override=False):
  """ Segment all given files into small chunks, the new file
  name is formatted as:
   - [name_without_extension].[ID].wav

  The information for each segment is saved at a csv file:
   - [outpath]/segments.csv

  Note
  ----
  We separated the segmenter from FeatureProcessor, since you can try
  different configuration for the features on the same set of segments,
  it is efficient to do this once-for-all.
  """
  info_path = os.path.join(outpath, 'segments.csv')
  # ====== validate arguments ====== #
  max_duration = int(max_duration)
  files = [f for f in as_tuple(files, t=str)
           if os.path.isfile(f)]
  outpath = str(outpath)
  if os.path.isfile(outpath):
    raise ValueError("outpath at: %s is a file." % outpath)
  if os.path.isdir(outpath):
    if not override:
      return info_path
    else:
      shutil.rmtree(outpath)
  if not os.path.isdir(outpath):
    os.mkdir(outpath)
  reader = AudioReader(sr=sr, sr_new=sr_new, best_resample=best_resample,
                       remove_dc_n_dither=False, preemphasis=None)

  # ====== segmenting ====== #
  def segmenting(f):
    raw = reader.transform(f)
    path, sr, raw = raw['path'], raw['sr'], raw['raw']
    segs = [int(np.round(i)) for i in np.linspace(
        start=0, stop=raw.shape[0],
        num=int(np.ceil(raw.shape[0] / (sr * max_duration))) + 1,
        endpoint=True)]
    indices = list(zip(segs, segs[1:]))
    name = os.path.basename(path)
    info = []
    for idx, (s, e) in enumerate(indices):
      y = raw[s:e]
      seg_name = name.split('.')[:-1] + [str(idx), 'wav']
      seg_name = '.'.join(seg_name)
      save(os.path.join(outpath, seg_name), y, sr)
      info.append((seg_name, s / sr, e / sr))
    return path, info
  nb_files = len(files)
  prog = Progbar(target=nb_files, print_summary=True, print_report=True,
                 name='Segmenting to path: %s' % outpath)
  task = mpi.MPI(jobs=files, func=segmenting,
                 ncpu=None, batch=1, backend='python')
  # ====== running the processor ====== #
  seg_indices = []
  for f, info in task:
    prog['File'] = f
    prog['#Segs'] = len(info)
    assert all(e - s <= max_duration
               for name, s, e in info), \
        "Results contain segments > max duration, file: %s, segs: %s" %\
        (f, str(info))
    for seg, s, e in info:
      seg_indices.append((seg, os.path.basename(f), s, e))
    prog.add(1)
  # ====== save the info ====== #
  header = ' '.join(['segment', 'origin', 'start', 'end'])
  np.savetxt(info_path, seg_indices,
             fmt='%s', delimiter=' ', header=header, comments='')
  print("Segment info saved at:", ctext(info_path, 'cyan'))
  return info_path


class AudioReader(Extractor):

  """ Return a dictionary of
  {
      'raw': loaded_signal,
      'duration': in second,
      'sr': sample rate,
      'path': path_to_loaded_file
  }

  Parameters
  ----------
  sr: int or None
      provided sr for missing sr audio (i.e. pcm files),
      NOTE this value only used when cannot find `sr` information
      from audio file (example: reading raw pcm).
  sr_new: int or None
      resample sample rate for all provided audio, only
      support downsample (i.e. must be smaller than sr).


  Input
  -----
  path_or_array: string, tuple, list, mapping
      - string for path
      - tuple or list for (path-or-raw, sr)
      - mapping for provding additional information include:
      sr, encode (ulaw, vast), 'raw' or 'path'

  Note
  ----
  Dithering introduces white noise when you save the raw array into
  audio file.
  For now only support one channel
  """

  def __init__(self, sr=None, sr_new=None, best_resample=True,
               remove_dc_n_dither=True, preemphasis=0.97,
               dataset=None):
    super(AudioReader, self).__init__()
    self.sr = sr
    self.sr_new = sr_new
    self.best_resample = best_resample
    self.remove_dc_n_dither = bool(remove_dc_n_dither)
    self.preemphasis = preemphasis
    # ====== check dataset ====== #
    if is_string(dataset) and os.path.isdir(dataset):
      dataset = Dataset(dataset)
    elif dataset is not None and not isinstance(dataset, Dataset):
      raise ValueError("dataset can be instance of odin.fuel.Dataset or None")
    self.dataset = dataset

  def _transform(self, path_or_array):
    raw = None # raw
    sr = None # by default, we don't know sample rate
    name = None
    path = None
    duration = None
    encode = None
    channel = 0
    # ====== check path_or_array ====== #
    # tuple of sr and raw_array
    if isinstance(path_or_array, (tuple, list)):
      if len(path_or_array) != 2:
        raise ValueError("`path_or_array` can be a tuple or list of "
            "length 2, which contains: (string_path, sr) or "
            "(sr, string_path) or (raw_array, sr) or (sr, raw_array).")
      if is_number(path_or_array[0]):
        sr, raw = path_or_array
      else:
        raw, sr = path_or_array
    # mapping of specific data
    elif isinstance(path_or_array, Mapping):
      if 'sr' in path_or_array:
        sr = path_or_array['sr']
      if 'encode' in path_or_array:
        encode = path_or_array['encode']
      # get raw or path out of the Dictionary
      if 'raw' in path_or_array:
        raw = path_or_array['raw']
      elif 'path' in path_or_array:
        path = path_or_array
        raw, sr = read(path, encode=encode)
      else:
        raise ValueError('`path_or_array` can be a dictionary, contains '
            'following key: sr, raw, path. One of the key `raw` for '
            'raw array signal, or `path` for path to audio file must '
            'be specified.')
    # read string file pth
    elif is_string(path_or_array):
      path = path_or_array
      if os.path.isfile(path_or_array):
        raw, sr = read(path_or_array, encode=encode)
      # given a dataset
      elif self.dataset is not None:
        start, end = self.dataset['indices'][path_or_array]
        raw = self.dataset['raw'][start:end]
        sr = int(self.dataset['sr'][path_or_array])
        name = path_or_array
        if 'path' in self.dataset:
          path = self.dataset['path'][path_or_array]
    # read from file object
    elif is_fileobj(path_or_array):
      path = path_or_array.name
      raw, sr = read(path_or_array, encode=encode)
    else:
      raise ValueError("`path_or_array` can be: list, tuple, Mapping, string, file"
          ". But given: %s" % str(type(path_or_array)))
    # ====== check channel ====== #
    raw = raw.astype('float32')
    if raw.ndim == 1:
      pass
    elif raw.ndim == 2:
      if raw.shape[0] == 2:
        raw = raw[channel, :]
      elif raw.shape[1] == 2:
        raw = raw[:, channel]
    else:
      raise ValueError("No support for %d-D signal from file: %s" %
          (raw.ndim, path))
    # ====== valiate sample rate ====== #
    if sr is None and self.sr is not None:
      sr = int(self.sr)
    # resampling if necessary
    if sr is not None and self.sr_new is not None:
      raw = resample(raw, sr, self.sr_new, best_algorithm=self.best_resample)
      sr = int(self.sr_new)
    # ====== normalizing ====== #
    np.random.seed(8)  # for repeatability
    # ====== remove DC offset and diterhing ====== #
    # Approached suggested by:
    # 'Omid Sadjadi': 'omid.sadjadi@nist.gov'
    if self.remove_dc_n_dither:
      # assuming 16-bit
      if max(abs(raw)) <= 1.:
        raw = raw * 2**15
      # select alpha
      if sr == 16000:
        alpha = 0.99
      elif sr == 8000:
        alpha = 0.999
      else:
        raise ValueError('Sampling frequency %s not supported' % sr)
      slen = raw.size
      raw = lfilter([1, -1], [1, -alpha], raw)
      dither = np.random.rand(slen) + np.random.rand(slen) - 1
      s_pow = max(raw.std(), 1e-20)
      raw = raw + 1.e-6 * s_pow * dither
    else: # just remove DC offset
      raw = raw - np.mean(raw, 0)
    # ====== pre-emphasis ====== #
    if self.preemphasis is not None and 0. < self.preemphasis < 1.:
      raw = pre_emphasis(raw, coeff=float(self.preemphasis))
    # ====== get duration if possible ====== #
    if sr is not None:
      duration = max(raw.shape) / sr
    # ====== check if absolute path====== #
    if path is not None:
      if not is_string(path):
        path = str(path, 'utf-8')
      if '/' != path[0]:
        path = os.path.abspath(path)
    ret = {'raw': raw, 'sr': sr, 'duration': duration, # in second
           'path': path}
    if name is not None:
      ret['name'] = name
    return ret


class SpectraExtractor(Extractor):
  """AcousticExtractor

  Parameters
  ----------
  frame_length: int
      number of samples point for 1 frame
  step_length: int
      number of samples point for 1 step (when shifting the frames)
      If unspecified, defaults `win_length / 4`.
  nb_fft: int > 0 [scalar]
      FFT window size
      If not provided, uses the smallest power of 2 enclosing `frame_length`.
  window : string, tuple, number, function, or np.ndarray [shape=(n_fft,)]
      - a window specification (string, tuple, or number);
        see `scipy.signal.get_window`
      - a window function, such as `scipy.signal.hanning`
      - a vector or array of length `n_fft`
  power : float > 0 [scalar]
      Exponent for the magnitude spectrogram.
      e.g., 1 for energy (or magnitude), 2 for power, etc.
  log: bool
      if True, convert all power spectrogram to DB
  padding : bool
      - If `True`, the signal `y` is padded so that frame
        `D[:, t]` is centered at `y[t * hop_length]`.
      - If `False`, then `D[:, t]` begins at `y[t * hop_length]`

  """

  def __init__(self, frame_length, step_length=None, nfft=512, window='hann',
               nmels=None, nceps=None, fmin=64, fmax=None,
               power=2.0, log=True, padding=False):
    super(SpectraExtractor, self).__init__()
    # ====== STFT ====== #
    self.frame_length = frame_length
    self.step_length = step_length
    self.nfft = nfft
    self.window = window
    # ====== ceptral analysis ====== #
    self.nmels = nmels
    self.nceps = nceps
    self.fmin = fmin
    self.fmax = fmax
    # ====== power spectrum ====== #
    self.power = float(power)
    self.log = bool(log)
    # ====== others ====== #
    self.padding = bool(padding)

  def _transform(self, s_sr):
    s, sr = _extract_s_sr(s_sr)
    frame_length, step_length = _extract_frame_step_length(
        sr, self.frame_length, self.step_length)
    # ====== extract spectra ====== #
    feat = spectra(sr=sr, frame_length=frame_length, y=s, S=None,
                   step_length=step_length, nfft=self.nfft,
                   window=self.window,
                   nmels=self.nmels, nceps=self.nceps,
                   fmin=self.fmin, fmax=self.fmax,
                   top_db=80., power=self.power, log=self.log,
                   padding=self.padding)
    return feat


class CQTExtractor(Extractor):
  """ Constant-Q transform
  Using log-scale instead of linear-scale frequencies for
  signal analysis

  """

  def __init__(self, frame_length, step_length=None, nbins=96, window='hann',
               nmels=None, nceps=None, fmin=64, fmax=None, padding=False):
    super(CQTExtractor, self).__init__()
    self.frame_length = frame_length
    self.step_length = step_length
    self.nbins = int(nbins)
    self.window = window
    self.nmels = nmels
    self.nceps = nceps
    self.fmin = fmin
    self.fmax = fmax
    self.padding = padding

  def _transform(self, s_sr):
    s, sr = _extract_s_sr(s_sr)
    frame_length, step_length = _extract_frame_step_length(
        sr, self.frame_length, self.step_length)
    # ====== extract CQT ====== #
    from librosa.core import constantq
    # auto adjust bins_per_octave to get maximum range of frequency
    bins_per_octave = np.ceil(float(self.nbins - 1) / np.log2(sr / 2. / self.fmin)) + 1
    # adjust the bins_per_octave to make acceptable hop_length
    # i.e. 2_factors(hop_length) < [ceil(cqt_bins / bins_per_octave) - 1]
    if _num_two_factors(step_length) < np.ceil(self.nbins / bins_per_octave) - 1:
      bins_per_octave = np.ceil(self.nbins / (_num_two_factors(step_length) + 1))
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=DeprecationWarning)
      qtrans = constantq.cqt(s, sr=sr, hop_length=step_length, n_bins=self.nbins,
                             bins_per_octave=int(bins_per_octave),
                             fmin=self.fmin, tuning=0.0, real=False, norm=1,
                             filter_scale=1., sparsity=0.01).astype('complex64')
    # ====== ceptral analysis ====== #
    feat = spectra(sr, frame_length, y=None, S=qtrans.T,
                   step_length=step_length, nfft=None, window='hann',
                   nmels=self.nmels, nceps=self.nceps,
                   fmin=64, fmax=self.fmax,
                   top_db=80.0, power=2.0, log=True,
                   padding=self.padding)
    # ====== add 'q' prefix for CQT features ====== #
    feat = {'q' + name: X for name, X in feat.items()}
    return feat


class BNFExtractor(Extractor):
  """ Deep bottleneck feature extractor
  The following order of preprocessing features for BNF are suggested:
  * extract input features: `X`
  * extract sad: `S`
  * normalize by speech activities: `X = (X - X[S].mean()) / X[S].std()`
  * Stacking features: `X = stacking(X, context=10)`
  => X_bnf = network(X)

  Parameters
  ----------
  input_feat : str
    name of input feature
  network : odin.nnet.base.NNOp
    instance pre-trained NNOp

  Note
  ----
  It is suggested to process the input features in following order:
   - mean_var_norm based on SAD frames statistics.
   - Stacking the left and right context frames.
   - Applying SAD indices.
   - Mean-variance normalization
  """

  def __init__(self, input_feat, network,
               batch_size=2048):
    super(BNFExtractor, self).__init__()
    from odin.nnet import NNOp
    self.input_feat = str(input_feat)
    if not isinstance(network, NNOp):
      raise ValueError("`network` must be instance of odin.nnet.NNOp")
    self.network = network
    self.batch_size = int(batch_size)

  def __getstate__(self):
    from odin import nnet as N
    if not self.network.is_initialized:
      self.network()
    return (self.input_feat, self.batch_size,
            N.serialize(self.network, output_mode='bin'))

  def __setstate__(self, states):
    from odin import nnet as N
    (self.input_feat, self.batch_size, self.network) = states
    self.network = N.deserialize(self.network,
                                 force_restore_vars=False)

  def _transform(self, feat):
    if self.input_feat not in feat:
      raise RuntimeError("BNFExtractor require input feature with name: %s"
                         ", which is not found." % self.input_feat)
    X = feat[self.input_feat]
    y = []
    # make prediciton
    for s, e in batching(n=X.shape[0], batch_size=self.batch_size):
      y.append(self.network(X[s:e]))
    feat['bnf'] = np.concatenate(y, axis=0)
    return feat


class PitchExtractor(Extractor):
  """
  Parameters
  ----------
  threshold : float, optional
      Voice/unvoiced threshold. Default is 0.3 (as suggested for SWIPE)
      Threshold >= 1.0 is suggested for RAPT
  algo: 'swipe', 'rapt', 'avg'
      swipe - A Saw-tooth Waveform Inspired Pitch Estimation.
      rapt - a robust algorithm for pitch tracking.
      avg - apply swipe and rapt at the same time, then take average.
      Default is 'swipe'

  """

  def __init__(self, frame_length, step_length=None,
               threshold=0.5, fmin=20, fmax=400,
               algo='swipe', f0=False):
    super(PitchExtractor, self).__init__()
    self.threshold = threshold
    self.fmin = int(fmin)
    self.fmax = int(fmax)
    self.algo = algo
    self.f0 = f0
    self.frame_length = frame_length
    self.step_length = step_length

  def _transform(self, s_sr):
    s, sr = _extract_s_sr(s_sr)
    frame_length, step_length = _extract_frame_step_length(
        sr, self.frame_length, self.step_length)
    # ====== extract pitch ====== #
    pitch_freq = pitch_track(s, sr, step_length, fmin=self.fmin,
        fmax=self.fmax, threshold=self.threshold, otype='pitch',
        algorithm=self.algo)
    pitch_freq = np.expand_dims(pitch_freq, axis=-1)
    if self.f0:
      f0_freq = pitch_track(s, sr, step_length, fmin=self.fmin,
          fmax=self.fmax, threshold=self.threshold, otype='f0',
          algorithm=self.algo)
      f0_freq = np.expand_dims(f0_freq, axis=-1)
      return {'pitch': pitch_freq,
              'f0': f0_freq}
    return {'pitch': pitch_freq}


class SADextractor(Extractor):
  """ GMM based SAD extractor
  """

  def __init__(self, nb_mixture=3, nb_train_it=24 + 1, smooth_window=3,
               feat_name='energy'):
    super(SADextractor, self).__init__()
    self.nb_mixture = int(nb_mixture)
    self.nb_train_it = int(nb_train_it)
    self.smooth_window = int(smooth_window)
    self.feat_name = str(feat_name).lower()

  def _transform(self, feat):
    # ====== select features type ====== #
    features = feat[self.feat_name]
    if features.ndim > 1:
      features = features.sum(axis=-1)
    # ====== calculate VAD ====== #
    vad, vad_threshold = vad_energy(log_energy=features.ravel(),
        distrib_nb=self.nb_mixture, nb_train_it=self.nb_train_it)
    if self.smooth_window > 0:
      # at least 2 voice frames
      threshold = (2. / self.smooth_window)
      vad = smooth(
          vad, win=self.smooth_window, window='flat') >= threshold
    # ====== vad is only 0 and 1 so 'uint8' is enough ====== #
    vad = vad.astype('uint8')
    return {'sad': vad, 'sad_threshold': float(vad_threshold)}


class RASTAfilter(Extractor):
  """ RASTAfilter

  Specialized "Relative Spectral Transform" applying for MFCCs
  and PLP

  RASTA is a separate technique that applies a band-pass filter
  to the energy in each frequency subband in order to smooth over
  short-term noise variations and to remove any constant offset
  resulting from static spectral coloration in the speech channel
  e.g. from a telephone line

  Parameters
  ----------
  RASTA : bool
    R.A.S.T.A filter
  sdc : int
      Lag size for delta feature computation for
      "Shifted Delta Coefficients", if `sdc` > 0, the
      shifted delta features will be append to MFCCs

  References
  ----------
  [PLP and RASTA](http://www.ee.columbia.edu/ln/rosa/matlab/rastamat/)

  """

  def __init__(self, rasta=True, sdc=1, feat_name='mfcc'):
    super(RASTAfilter, self).__init__()
    self.rasta = bool(rasta)
    self.sdc = int(sdc)
    self.feat_name = str(feat_name)

  def _transform(self, feat):
    if self.feat_name not in feat:
      raise RuntimeError("Cannot find feature with name: '%s' in "
                         "processed feature list." % self.feat_name)
    mfcc = feat[self.feat_name]
    # apply RASTA
    if self.rasta:
      mfcc = rastafilt(mfcc)
    # apply SDC if required
    if self.sdc >= 1:
      mfcc = np.hstack([
          mfcc,
          shifted_deltas(mfcc, N=mfcc.shape[-1], d=self.sdc,
                         P=3, k=7)
      ])
    # store new feature
    feat[self.feat_name] = mfcc.astype("float32")
    return feat


class AcousticNorm(Extractor):
  """
  Parameters
  ----------
  sad_stats : bool
    if True, using statistics from SAD indexed frames for
    normalization
  sad_name : str
    feature name of SAD indices, only used if `sad_stats=True`

  """

  def __init__(self, mean_var_norm=True, window_mean_var_norm=True,
               win_length=301, var_norm=True,
               sad_stats=False, sad_name='sad',
               feat_name=('mspec', 'spec', 'mfcc', 'bnf',
                          'qspec', 'qmfcc', 'qmspec')):
    super(AcousticNorm, self).__init__()
    self.mean_var_norm = bool(mean_var_norm)
    self.window_mean_var_norm = bool(window_mean_var_norm)
    self.var_norm = bool(var_norm)
    # ====== check win_length ====== #
    win_length = int(win_length)
    if win_length % 2 == 0:
      raise ValueError("win_length must be odd number")
    if win_length < 3:
      raise ValueError("win_length must >= 3")
    self.win_length = win_length
    # ====== check which features will be normalized ====== #
    self.feat_name = as_tuple(feat_name, t=str)

  def _transform(self, feat):
    feat_normalized = {}
    # all `features` is [t, f] shape
    for name, features in feat.items():
      if name in self.feat_name:
        if self.mean_var_norm:
          features = mvn(features, varnorm=self.var_norm)
        if self.window_mean_var_norm:
          features = wmvn(features, w=self.win_length,
                          varnorm=False)
      feat_normalized[name] = features
    return feat_normalized


class Read3ColSAD(Extractor):
  """ Read3ColSAD simple helper for applying 3 col
  SAD (name, start-in-second, end-in-second) to extracted acoustic features

  Parameters
  ----------
  path: str
      path to folder contain all SAD files
  name_converter: call-able
      convert the 'path' element in features dictionary (provided to
      `transform`) to name for search in parsed SAD dictionary.
  file_regex: str
      regular expression for filtering the files name
  keep_unvoiced: bool
      if True, keep all the the features of utterances even though
      cannot file SAD for it. Otherwise, return None.
  feat_name: list of string
      all features type will be applied using given SAD

  Return
  ------
   - add 'sad': array of SAD indexing

  """

  def __init__(self, path_or_map, step_length, ref_feat,
               name_converter=None, ref_key='path',
               file_regex='.*'):
    super(Read3ColSAD, self).__init__()
    self.step_length = float(step_length)
    self.ref_feat = str(ref_feat)
    # ====== file regex ====== #
    file_regex = re.compile(str(file_regex))
    # ====== name_converter ====== #
    if name_converter is not None:
      if not hasattr(name_converter, '__call__'):
        raise ValueError("`name_converter` must be call-able, or Mapping.")
      if inspect.isfunction(name_converter):
        name_converter = functionable(func=name_converter)
    self.name_converter = name_converter
    self.ref_key = as_tuple(ref_key, t=str)
    # ====== parse all file ====== #
    # read the SAD from file
    if is_string(path_or_map):
      sad = defaultdict(list)
      for f in get_all_files(path_or_map):
        if file_regex.search(os.path.basename(f)) is not None:
          with open(f, 'r') as f:
            for line in f:
              name, start, end = line.strip().split(' ')
              sad[name].append((float(start), float(end)))
    elif isinstance(path_or_map, Mapping):
      sad = path_or_map
    else:
      raise ValueError("`path` must be path to folder, or dictionary.")
    self.sad = sad

  def _transform(self, feats):
    # ====== get ref name ====== #
    name = None
    for k in self.ref_key:
      name = feats.get(k, None)
      if is_string(name):
        break
    # ====== get ref feature ====== #
    if self.ref_feat not in feats:
      raise RuntimeError("Cannot find reference feature with "
          "name: '%s'" % self.ref_feat)
    nb_samples = len(feats[self.ref_feat])
    # ====== start ====== #
    if is_string(name):
      if self.name_converter is not None:
        name = (self.name_converter[name]
            if isinstance(self.name_converter, Mapping) else
            self.name_converter(name))
      # ====== convert step_length ====== #
      step_length = self.step_length
      if step_length >= 1: # step_length is number of frames
        step_length = step_length / feats['sr']
        # now step_length is in second
      # ====== found SAD ====== #
      no_sad_found = True
      if name in self.sad:
        sad = self.sad[name]
        if len(sad) > 0:
          sad_indices = np.zeros(shape=(nb_samples,), dtype=np.uint8)
          for start_sec, end_sec in sad:
            start_idx = int(start_sec / step_length)
            end_idx = int(end_sec / step_length)
            # ignore zero SAD
            if end_idx - start_idx == 0:
              continue
            sad_indices[start_idx:end_idx] = 1
          feats['sad'] = sad_indices
          no_sad_found = False
    # ====== return ====== #
    if no_sad_found:
      feats['sad'] = np.zeros(shape=(nb_samples,), dtype=np.uint8)
    return feats


class ApplyingSAD(Extractor):
  """ Applying SAD index to given features
  This extractor cutting voiced segments out, using extracted
  SAD labels previously


  Parameters
  ----------
  sad_name: str
      specific feature will be used name for the cutting
  threshold: None or float
      if `sad`, is continuous value, threshold need to be applied
  smooth_win: int (> 0)
      ammount of adjent frames will be taken into the SAD
  keep_unvoiced: bool
      if True, keep the whole audio file even though no SAD found
  stack_context : dict
      a dictionary mapping from feature name to number of
      context frames (a scalar for both left and right context)
      NOTE: the final frame length is: `context * 2 + 1`
  feat_name: str, or list of str
      all features' name will be applied.

  """

  def __init__(self, sad_name='sad', threshold=None,
               smooth_win=None, keep_unvoiced=False,
               stack_context={},
               feat_name=('spec', 'mspec', 'mfcc',
                          'qspec', 'qmspec', 'qmfcc',
                          'pitch', 'f0', 'energy')):
    super(ApplyingSAD, self).__init__()
    self.threshold = float(threshold) if is_number(threshold) else None
    self.smooth_win = int(smooth_win) if is_number(smooth_win) else None
    self.sad_name = str(sad_name)
    self.keep_unvoiced = bool(keep_unvoiced)
    self.feat_name = as_tuple(feat_name, t=str)
    self.stack_context = {str(name): int(ctx)
                          for name, ctx in stack_context.items()
                          if ctx > 0 and str(name) in feat_name}

  def _transform(self, X):
    if self.sad_name in X:
      # ====== threshold sad to index ====== #
      sad = X[self.sad_name]
      if is_number(self.threshold):
        sad = (sad >= self.threshold).astype('int32')
      if is_number(self.smooth_win) and self.smooth_win > 0:
        sad = smooth(sad, win=self.smooth_win, window='flat') > 0.
      sad = sad.astype('bool')
      # ====== keep unvoiced or not ====== #
      if np.sum(sad) == 0:
        if not self.keep_unvoiced:
          return None
        else: # take all frames
          sad[:] = True
      # ====== start ====== #
      X_new = {}
      for feat_name, X_feat in X.items():
        if feat_name in self.feat_name:
          assert len(sad) == max(X_feat.shape),\
              "Length of sad labels is: %d, but number of sample is: %s"\
              % (len(sad), max(X_feat.shape))
          X_sad = X_feat[sad]
          # applying SAD and stacking features at the same time
          if feat_name in self.stack_context:
            X_feat = ((X_feat - X_sad.mean(axis=0, keepdims=True)) /
                      (X_sad.std(axis=0, keepdims=True) + config.EPS))
            X_feat = stack_frames(X_feat,
                                  frame_length=self.stack_context[feat_name] * 2 + 1,
                                  step_length=1, keepdims=True,
                                  make_contigous=True)
            X_sad = X_feat[sad]
          # update feature
          X_new[feat_name] = X_sad
    return X_new
