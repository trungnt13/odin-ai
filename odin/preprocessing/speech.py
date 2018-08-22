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
# ===== SAD ===== #
+ SADextractor:
    - Extracting SAD (optional)
+ Read3ColSAD:
    - Generating SAD labels from 3-cols files
# ===== Normalization ===== #
+ Rastafilt:
    - Rastafilt (optional for MFCC)
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
from six import string_types
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
                     shifted_deltas, stack_frames, stft,
                     power_spectrogram, mels_spectrogram, ceps_spectrogram,
                     power2db)
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
  """
  Returns
  -------
  audio_array : [n_samples, nb_channels]
    the audio array
  sr : {int, None}
    sample rate
  """
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
  remove_dc_n_dither : bool
    dithering adds noise to the signal to remove periodic noise

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
               remove_dc=True, dataset=None):
    super(AudioReader, self).__init__(is_input_layer=True)
    self.sr = sr
    self.sr_new = sr_new
    self.best_resample = best_resample
    self.remove_dc = bool(remove_dc)
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
        sr = int(path_or_array['sr'])
      if 'encode' in path_or_array:
        encode = str(path_or_array['encode'])
      # get raw or path out of the Dictionary
      if 'raw' in path_or_array:
        raw = path_or_array['raw']
      elif 'path' in path_or_array:
        path = str(path_or_array['path'])
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
      # file not exist
      else:
        raise ValueError("Cannot locate file at path: %s" % path_or_array)
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
    if self.remove_dc: # just remove DC offset
      raw = raw - np.mean(raw, 0).astype(raw.dtype)
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

class Dithering(Extractor):
  """ Dithering """

  def __init__(self, input_name=('raw', 'sr'), output_name='raw'):
    super(Dithering, self).__init__(
        input_name=as_tuple(input_name, t=string_types),
        output_name=str(output_name))

  def _transform(self, feat):
    raw, sr = [feat[name] for name in self.input_name]
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
    return {self.output_name: raw}

class PreEmphasis(Extractor):
  """ PreEmphasis

  Parameters
  ----------
  coeff : float (0-1)
      pre-emphasis filter, if 0 or None, no filter applied
  input_name : string
      name of raw signal in the features pipeline dictionary
  """

  def __init__(self, coeff=0.97,
               input_name='raw', output_name='raw'):
    super(PreEmphasis, self).__init__(input_name=str(input_name),
                                      output_name=str(output_name))
    assert 0. < coeff < 1.
    self.coeff = float(coeff)

  def _transform(self, feat):
    raw = feat[self.input_name]
    if not 0 < raw.ndim <= 2:
      raise ValueError("Only supper 1 or 2 channel audio but given shape: %s" %
                       str(raw.shape))
    return {self.output_name: pre_emphasis(raw, coeff=self.coeff)}

# ===========================================================================
# Spectrogram
# ===========================================================================
class STFTExtractor(Extractor):
  """ Short time Fourier transform

  Parameters
  ----------
  frame_length: {int, float}
      number of samples point for 1 frame, or length of frame in millisecond
  step_length: {int, float}
      number of samples point for 1 step (when shifting the frames),
      or length of step in millisecond
      If unspecified, defaults `win_length / 4`.
  n_fft: int > 0 [scalar]
      FFT window size
      If not provided, uses the smallest power of 2 enclosing `frame_length`.

  Output
  ------
  'stft' : complex64 array [time, frequency]
  'stft_energy' : float32 array [time, 1]
  """

  def __init__(self, frame_length, step_length=None, n_fft=512,
               window='hann', padding=False,
               input_name=('raw', 'sr'), output_name='stft'):
    super(STFTExtractor, self).__init__(input_name=input_name, output_name=output_name)
    self.frame_length = frame_length
    self.step_length = step_length
    self.n_fft = n_fft
    self.window = window
    self.padding = bool(padding)

  def _transform(self, y_sr):
    y, sr = [y_sr[name] for name in self.input_name]
    frame_length, step_length = _extract_frame_step_length(
        sr, self.frame_length, self.step_length)
    results = stft(y=y, frame_length=frame_length, step_length=step_length,
                   n_fft=self.n_fft, window=self.window, padding=self.padding,
                   energy=True)
    s, e = results
    return {self.output_name: s,
            '%s_energy' % self.output_name: e}

class PowerSpecExtractor(Extractor):
  """ Extract power spectrogram from complex STFT array

  Output
  ------
  'spec' : [time, n_fft / 2 + 1]

  """

  def __init__(self, power=2.0, input_name='stft', output_name='spec'):
    super(PowerSpecExtractor, self).__init__(input_name=input_name,
                                             output_name=output_name)
    self.power = float(power)

  def _transform(self, X):
    return power_spectrogram(S=X[self.input_name], power=self.power)

class MelsSpecExtractor(Extractor):
  """
  Parameters
  ----------
  input_name : (string, string) (default: ('spec', 'sr'))
    the name of spectrogram and sample rate in the feature pipeline

  Output
  ------
  'mspec' : [time, n_mels]
  """

  def __init__(self, n_mels, fmin=64, fmax=None, top_db=80.0,
               input_name=('spec', 'sr'), output_name='mspec'):
    super(MelsSpecExtractor, self).__init__(input_name=input_name,
                                            output_name=output_name)
    self.n_mels = int(n_mels)
    self.fmin = fmin
    self.fmax = fmax
    self.top_db = top_db

  def _transform(self, X):
    return mels_spectrogram(spec=X[self.input_name[0]], sr=X[self.input_name[1]],
                            n_mels=self.n_mels,
                            fmin=self.fmin, fmax=self.fmax, top_db=self.top_db)

class MFCCsExtractor(Extractor):
  """
  """

  def __init__(self, n_ceps, remove_first_coef=True,
               input_name='mspec', output_name='mfcc'):
    super(MFCCsExtractor, self).__init__(input_name=input_name,
                                         output_name=output_name)
    self.n_ceps = int(n_ceps)
    self.remove_first_coef = bool(remove_first_coef)

  def _transform(self, X):
    return ceps_spectrogram(mspec=X[self.input_name], n_ceps=self.n_ceps,
                            remove_first_coef=self.remove_first_coef)

class Power2Db(Extractor):
  """ Convert power spectrogram to Decibel spectrogram

  """

  def __init__(self, input_name, output_name=None, top_db=80.0):
    input_name = as_tuple(input_name, t=string_types)
    super(Power2Db, self).__init__(input_name=input_name,
                                   output_name=output_name)
    self.top_db = float(top_db)

  def _transform(self, X):
    return [power2db(S=X[name], top_db=self.top_db) for name in self.input_name]

class SpectraExtractor(Extractor):
  """AcousticExtractor

  Parameters
  ----------
  frame_length: {int, float}
      number of samples point for 1 frame, or length of frame in millisecond
  step_length: {int, float}
      number of samples point for 1 step (when shifting the frames),
      or length of step in millisecond
      If unspecified, defaults `win_length / 4`.
  n_fft: int > 0 [scalar]
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

  def __init__(self, frame_length, step_length=None, n_fft=512, window='hann',
               n_mels=None, n_ceps=None, fmin=64, fmax=None,
               power=2.0, log=True, padding=False,
               input_name=('raw', 'sr')):
    super(SpectraExtractor, self).__init__(input_name=input_name)
    # ====== STFT ====== #
    self.frame_length = frame_length
    self.step_length = step_length
    self.n_fft = n_fft
    self.window = window
    # ====== ceptral analysis ====== #
    self.n_mels = n_mels
    self.n_ceps = n_ceps
    self.fmin = fmin
    self.fmax = fmax
    # ====== power spectrum ====== #
    self.power = float(power)
    self.log = bool(log)
    # ====== others ====== #
    self.padding = bool(padding)

  def _transform(self, y_sr):
    y, sr = [y_sr[i] for i in self.input_name]
    frame_length, step_length = _extract_frame_step_length(
        sr, self.frame_length, self.step_length)
    # ====== extract spectra ====== #
    feat = spectra(sr=sr, frame_length=frame_length, y=y, S=None,
                   step_length=step_length, n_fft=self.n_fft,
                   window=self.window,
                   n_mels=self.n_mels, n_ceps=self.n_ceps,
                   fmin=self.fmin, fmax=self.fmax,
                   top_db=80., power=self.power, log=self.log,
                   padding=self.padding)
    return feat

class CQTExtractor(Extractor):
  """ Constant-Q transform
  Using log-scale instead of linear-scale frequencies for
  signal analysis

  """

  def __init__(self, frame_length, step_length=None, n_bins=96, window='hann',
               n_mels=None, n_ceps=None, fmin=64, fmax=None, padding=False,
               input_name=('raw', 'sr')):
    super(CQTExtractor, self).__init__(input_name=input_name)
    self.frame_length = frame_length
    self.step_length = step_length
    self.n_bins = int(n_bins)
    self.window = window
    self.n_mels = n_mels
    self.n_ceps = n_ceps
    self.fmin = fmin
    self.fmax = fmax
    self.padding = padding

  def _transform(self, y_sr):
    y, sr = [y_sr[name] for name in self.input_name]
    frame_length, step_length = _extract_frame_step_length(
        sr, self.frame_length, self.step_length)
    # ====== extract CQT ====== #
    from librosa.core import constantq
    # auto adjust bins_per_octave to get maximum range of frequency
    bins_per_octave = np.ceil(float(self.n_bins - 1) / np.log2(sr / 2. / self.fmin)) + 1
    # adjust the bins_per_octave to make acceptable hop_length
    # i.e. 2_factors(hop_length) < [ceil(cqt_bins / bins_per_octave) - 1]
    if _num_two_factors(step_length) < np.ceil(self.n_bins / bins_per_octave) - 1:
      bins_per_octave = np.ceil(self.n_bins / (_num_two_factors(step_length) + 1))
    with warnings.catch_warnings():
      warnings.filterwarnings("ignore", category=DeprecationWarning)
      qtrans = constantq.cqt(y=y, sr=sr, hop_length=step_length, n_bins=self.n_bins,
                             bins_per_octave=int(bins_per_octave),
                             fmin=self.fmin, tuning=0.0, norm=1,
                             filter_scale=1., sparsity=0.01).astype('complex64')
    # ====== ceptral analysis ====== #
    feat = spectra(sr=sr, frame_length=frame_length, y=None, S=qtrans.T,
                   step_length=step_length, n_fft=None, window='hann',
                   n_mels=self.n_mels, n_ceps=self.n_ceps,
                   fmin=64, fmax=self.fmax,
                   top_db=80.0, power=2.0, log=True,
                   padding=self.padding)
    # ====== add 'q' prefix for CQT features ====== #
    feat = {'q' + name: X for name, X in feat.items()}
    return feat

# ===========================================================================
# More features
# ===========================================================================
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
  pre_mvn : bool (default: False)
    perform mean-variance normalization before stacking,
    then, feeding data to network.
  sad_name : {str, None}
    if None, or `sad_name` not found, don't applying SAD to
    the input feature before BNF
  dtype : (str, numpy.dtype, None)
    cast input to specific dtype before BNF, if None,
    keep original dtype
  batch_size : int
    batch size when feeding data to the network, suggest
    to have as much data as possible.

  Note
  ----
  Order of preprocessing for BNF:
   - delta and delta delta extraction (optional)
   - mean_var_norm based on SAD frames statistics.
   - Stacking the left and right context frames.
   - Applying SAD indices.
   - Mean-variance normalization
   => BNFExtractor

  """

  def __init__(self, input_name, network,
               output_name='bnf', sad_name='sad',
               stack_context=10, pre_mvn=True,
               batch_size=2048):
    assert isinstance(input_name, string_types), "`input_name` must be string"
    if isinstance(sad_name, string_types):
      input_name = (input_name, sad_name)
      self.use_sad = True
    else:
      self.use_sad = False
    super(BNFExtractor, self).__init__(input_name=input_name, output_name=output_name)
    # ====== check the network ====== #
    from odin.nnet import NNOp
    if not isinstance(network, NNOp):
      raise ValueError("`network` must be instance of odin.nnet.NNOp")
    self.network = network
    # ====== other configs ====== #
    if stack_context is None:
      stack_context = 0
    self.stack_context = int(stack_context)
    self.pre_mvn = bool(pre_mvn)
    self.batch_size = int(batch_size)

  def __getstate__(self):
    from odin import nnet as N
    if not self.network.is_initialized:
      self.network()
    return (self._input_name, self._output_name,
            self.use_sad, self.batch_size, self.stack_context, self.pre_mvn,
            N.serialize(self.network, binary_output=True))

  def __setstate__(self, states):
    from odin import nnet as N
    (self._input_name, self._output_name,
     self.use_sad, self.batch_size, self.stack_context, self.pre_mvn,
     self.network) = states
    self.network = N.deserialize(self.network,
                                 force_restore_vars=False)

  def _transform(self, feat):
    if self.use_sad:
      X, sad = feat[self.input_name[0]], feat[self.input_name[1]]
      sad = sad.astype('bool')
      assert len(sad) == len(X), \
      "Input mismatch, `sad` has shape: %s, and input feature with name: %s; shape: %s" % \
      (sad.shape, self.input_name[0], X.shape)
    else:
      X = feat[self.input_name]
      sad = None
    # ====== pre-normalization ====== #
    X_sad = X[sad] if sad is not None else X
    if self.pre_mvn:
      X = (X - X_sad.mean(0, keepdims=True)) /  \
          (X_sad.std(0, keepdims=True) + 1e-18)
    # ====== stacking context and applying SAD ====== #
    if self.stack_context > 0:
      X = stack_frames(X, frame_length=self.stack_context * 2 + 1,
                       step_length=1, keepdims=True,
                       make_contigous=True)
    if sad is not None:
      X = X[sad]
    # ====== transform ====== #
    y = []
    # make prediction
    for s, e in batching(n=X.shape[0], batch_size=self.batch_size):
      y.append(self.network(X[s:e]))
    return np.concatenate(y, axis=0)

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
               algo='swipe', f0=False, input_name=('raw', 'sr')):
    super(PitchExtractor, self).__init__(input_name=input_name)
    self.threshold = threshold
    self.fmin = int(fmin)
    self.fmax = int(fmax)
    self.algo = algo
    self.f0 = f0
    self.frame_length = frame_length
    self.step_length = step_length

  def _transform(self, y_sr):
    y, sr = [y_sr[name] for name in self.input_name]
    frame_length, step_length = _extract_frame_step_length(
        sr, self.frame_length, self.step_length)
    # ====== extract pitch ====== #
    pitch_freq = pitch_track(y=y, sr=sr, step_length=step_length, fmin=self.fmin,
        fmax=self.fmax, threshold=self.threshold, otype='pitch',
        algorithm=self.algo)
    pitch_freq = np.expand_dims(pitch_freq, axis=-1)
    if self.f0:
      f0_freq = pitch_track(y=y, sr=sr, step_length=step_length, fmin=self.fmin,
          fmax=self.fmax, threshold=self.threshold, otype='f0',
          algorithm=self.algo)
      f0_freq = np.expand_dims(f0_freq, axis=-1)
      return {'pitch': pitch_freq,
              'f0': f0_freq}
    return {'pitch': pitch_freq}


class SADextractor(Extractor):
  """ GMM-based SAD extractor
  """

  def __init__(self, nb_mixture=3, nb_train_it=24 + 1, smooth_window=3,
               input_name='energy', output_name='sad'):
    super(SADextractor, self).__init__(input_name=input_name, output_name=output_name)
    self.nb_mixture = int(nb_mixture)
    self.nb_train_it = int(nb_train_it)
    self.smooth_window = int(smooth_window)

  def _transform(self, feat):
    # ====== select features type ====== #
    features = feat[self.input_name]
    if features.ndim > 1:
      features = features.sum(axis=-1)
    # ====== calculate VAD ====== #
    sad, sad_threshold = vad_energy(log_energy=features.ravel(),
        distrib_nb=self.nb_mixture, nb_train_it=self.nb_train_it)
    if self.smooth_window > 0:
      # at least 2 voice frames
      threshold = (2. / self.smooth_window)
      sad = smooth(sad, win=self.smooth_window, window='flat') >= threshold
    # ====== vad is only 0 and 1 so 'uint8' is enough ====== #
    sad = sad.astype('uint8')
    return {self.output_name: sad,
            '%s_threshold' % self.output_name: float(sad_threshold)}

# ===========================================================================
# Normalization
# ===========================================================================
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

  def __init__(self, rasta=True, sdc=1,
               input_name='mfcc', output_name=None):
    super(RASTAfilter, self).__init__(input_name=as_tuple(input_name, t=string_types),
                                      output_name=output_name)
    self.rasta = bool(rasta)
    self.sdc = int(sdc)

  def _transform(self, feat):
    new_feat = []
    for name in self.input_name:
      mfcc = feat[name]
      # apply RASTA
      if self.rasta:
        mfcc = rastafilt(mfcc)
      # apply SDC if required
      if self.sdc >= 1:
        n_ceps = mfcc.shape[-1]
        mfcc = np.hstack([
            mfcc,
            shifted_deltas(mfcc, N=n_ceps, d=self.sdc,
                           P=3, k=n_ceps) # k = 7
        ])
      # store new feature
      new_feat.append(mfcc.astype("float32"))
    return new_feat

class AcousticNorm(Extractor):
  """
  Parameters
  ----------
  mean_var_norm : bool (default: True)
    mean-variance normalization
  windowed_mean_var_norm : bool (default: False)
    perform standardization on small windows, very computaiton
    intensive.
  sad_name : {str, None} (default: None)
    feature name of SAD indices, and only using statistics from
    SAD indexed frames for normalization
  ignore_sad_error : bool
    if True, when length of SAD and feature mismatch, still perform
    normalization, otherwise raise `RuntimeError`.

  """

  def __init__(self, input_name, output_name=None,
               mean_var_norm=True, windowed_mean_var_norm=False,
               win_length=301, var_norm=True,
               sad_name=None, ignore_sad_error=True):
    # ====== check which features will be normalized ====== #
    self.sad_name = str(sad_name) if isinstance(sad_name, string_types) else None
    self.ignore_sad_error = bool(ignore_sad_error)
    super(AcousticNorm, self).__init__(input_name=as_tuple(input_name, t=string_types),
                                       output_name=output_name)
    # ====== configs ====== #
    self.mean_var_norm = bool(mean_var_norm)
    self.windowed_mean_var_norm = bool(windowed_mean_var_norm)
    self.var_norm = bool(var_norm)
    # ====== check win_length ====== #
    win_length = int(win_length)
    if win_length % 2 == 0:
      raise ValueError("win_length must be odd number")
    if win_length < 3:
      raise ValueError("win_length must >= 3")
    self.win_length = win_length

  def _transform(self, feat):
    # ====== check SAD indices ====== #
    sad = None
    if self.sad_name is not None:
      sad = feat[self.sad_name]
      if sad.dtype != np.bool:
        sad = sad.astype(np.bool)
    # ====== normalize ====== #
    feat_normalized = []
    # all `features` is [t, f] shape
    for name in self.input_name:
      X = feat[name]
      X_sad = sad
      if sad is not None and len(sad) != len(X):
        if not self.ignore_sad_error:
          raise RuntimeError("Features with name: '%s' have length %d, but "
                             "given SAD has length %d" % (name, len(X), len(sad)))
        else:
          X_sad = None
      # mean-variance normalization
      if self.mean_var_norm:
        X = mvn(X, varnorm=self.var_norm, indices=X_sad)
      # windowed normalization
      if self.windowed_mean_var_norm:
        X = wmvn(X, w=self.win_length, varnorm=False, indices=X_sad)
      # update new features
      feat_normalized.append(X)
    return feat_normalized

class Read3ColSAD(Extractor):
  """ Read3ColSAD simple helper for applying 3 col
  SAD (name, start-in-second, end-in-second) to extracted acoustic features

  Parameters
  ----------
  path_or_map : {str, Mapping}
      path to folder contain all SAD files
      if Mapping, it is a map from `ref_key` to SAD dictionary
  step_length : float
      step length to convert second to frame index.
  ref_feat : str
      name of a reference features that must have the
      same length with the SAD.
  input_name : str (default: path)
      reference key in the pipeline can be use to get the coordinate value
      of SAD from given dictionary.
  file_regex: str
      regular expression for filtering the files name

  Return
  ------
   - add 'sad': array of SAD indexing

  """

  def __init__(self, path_or_map, step_length, ref_feat,
               input_name='path', output_name='sad',
               file_regex='.*'):
    super(Read3ColSAD, self).__init__(input_name=str(input_name))
    self.step_length = float(step_length)
    self.ref_feat = str(ref_feat)
    # ====== file regex ====== #
    file_regex = re.compile(str(file_regex))
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
    self._sad = sad

  def _transform(self, feat):
    # ====== get ref name ====== #
    name = feat[self.input_name]
    ref_feat = feat[self.ref_feat]
    n_samples = len(ref_feat)
    # ====== convert step_length ====== #
    step_length = self.step_length
    if step_length >= 1: # step_length is number of frames
      # now step_length is in second
      step_length = step_length / feat['sr']
    # ====== found SAD ====== #
    sad_indices = np.zeros(shape=(n_samples,), dtype=np.uint8)
    if name in self.sad and len(self.sad[name]) > 0:
      for start_sec, end_sec in self.sad[name]:
        start_idx = int(start_sec / step_length)
        end_idx = int(end_sec / step_length)
        # ignore zero SAD
        if end_idx - start_idx == 0:
          continue
        sad_indices[start_idx:end_idx] = 1
    # ====== return ====== #
    return {self.output_name: sad_indices}

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

  def __init__(self, input_name, output_name=None, sad_name='sad',
               threshold=None, smooth_win=None, keep_unvoiced=False):
    super(ApplyingSAD, self).__init__(input_name=as_tuple(input_name, t=string_types),
                                      output_name=output_name)
    self.sad_name = str(sad_name)
    self.threshold = float(threshold) if is_number(threshold) else None
    self.smooth_win = int(smooth_win) if is_number(smooth_win) else None
    self.keep_unvoiced = bool(keep_unvoiced)

  def _transform(self, X):
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
    X_new = []
    for name in self.input_name:
      X_feat = X[name]
      assert len(sad) == max(X_feat.shape),\
          "Feature with name: %s, length of sad labels is: %d, but number of sample is: %s" % \
          (name, len(sad), max(X_feat.shape))
      # update feature
      X_new.append(X_feat[sad])
    return X_new
