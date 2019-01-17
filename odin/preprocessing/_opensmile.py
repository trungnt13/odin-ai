# -*- coding: utf-8 -*-
"""
openSMILE LICENSE
=================
Authors: Florian Eyben, Felix Weninger, Martin Woellmer, Bjoern Schuller
Copyright (C) 2008-2013, Institute for Human-Machine Communication, TUM
Copyright (C) 2013-2014, audEERING UG (haftungsbeschrÃ¤nkt)
http://www.audeering.com/research-and-open-source/files/openSMILE-open-source-license.txt
"""
from __future__ import print_function, division, absolute_import

import os
import re
import random
import subprocess
from six import add_metaclass
from collections import Mapping
from abc import abstractproperty, ABCMeta, abstractmethod

import numpy as np

from odin.preprocessing.base import Extractor
from odin.utils import (is_string, get_script_path, ctext, is_number,
                        get_logpath, uuid)

__all__ = [
    'openSMILEloudness',
    'openSMILEf0',
    'openSMILEpitch',
    'openSMILEsad'
]

# ===========================================================================
# Helper
# ===========================================================================
def verify_dependencies():
  try:
    command = "SMILExtract -h"
    output = subprocess.check_output(command, shell=True,
                                     stderr=subprocess.STDOUT)
  except subprocess.CalledProcessError:
    raise Exception("Can't find SMILExtract executable")
  else:
    m = re.search('openSMILE version (.*)', str(output, 'utf-8'),
                  re.MULTILINE)
    if m:
      opensmile_version = m.group(1)
      print('Found openSMILE:', ctext(opensmile_version, 'magenta'))


def _get_conf_file(name):
  conf_path = os.path.join(
      os.path.dirname(__file__),
      'confs',
      name)
  if not os.path.exists(conf_path):
    raise RuntimeError("Cannot find config file at path: %s" % conf_path)
  with open(conf_path, 'r') as f:
    return f.read()

# ===========================================================================
# Base
# ===========================================================================
@add_metaclass(ABCMeta)
class _openSMILEbase(Extractor):

  def __init__(self, sr):
    verify_dependencies()
    super(_openSMILEbase, self).__init__()
    self._id = uuid(length=25)
    self.sr = sr
    self._first_config_generated = False
    self._conf = _get_conf_file('%s.cfg' % self.__class__.__name__)
    self._log_level = -1

  # ==================== abstract ==================== #
  def set_log_level(self, level):
    """ level: {int, bool}
      if `int`, log-level in integer (from 0 - 9) higher
      means more detail, -1 for turning off the log.
      if True, set the log-level to default: 2
    """
    if is_number(level):
      self._log_level = int(level)
    elif bool(level):
      self._log_level = 2
    else:
      self._log_level = -1
    return self

  @abstractproperty
  def config(self):
    pass

  @abstractmethod
  def _post_processing(self, X):
    pass

  @property
  def config_path(self):
    return os.path.join(get_logpath(), '%s%s.cfg' %
        (self.__class__.__name__, self._id))

  # ==================== utilities ==================== #
  def _update_config(self):
    if self.sr is None:
      return
    with open(self.config_path, 'w') as f:
      f.write(self._conf.format(**self.config))

  def _transform(self, X):
    # ====== file input file ====== #
    raw = None
    path = None
    if isinstance(X, Mapping):
      if 'path' in X:
        path = X['path']
      if 'sr' in X:
        if self.sr is None:
          self.sr = X['sr']
          self._update_config()
          self._first_config_generated = True
        elif self.sr != X['sr']:
          raise ValueError("Given sample rate: %d, but the audio file has "
                           "sample rate: %d" % (self.sr, X['sr']))
      if 'raw' in X:
        raw = X['raw']
    elif is_string(X):
      path = X
    elif isinstance(X, np.ndarray):
      raw = X
    else:
      raise ValueError("openSMILE extractor require path to audio file.")
    # no sample rate specified, cannot generate appropriate config
    if self.sr is None:
      raise RuntimeError("Cannot acquire sample rate for the input.")
    # ====== first time generate config ====== #
    if not self._first_config_generated:
      self._first_config_generated = True
      self._update_config()
    # ====== extract SAD ====== #
    unique_id = os.getpid() + random.randint(0, 10e8)
    inpath = os.path.join(
        get_logpath(), '%s%d.wav' % (self.__class__.__name__, unique_id))
    outpath = os.path.join(
        get_logpath(), '%s%d.csv' % (self.__class__.__name__, unique_id))
    try:
      if path is None or not os.path.exists(path):
        if raw is None:
          raise RuntimeError("openSMILE require input audio file, since "
              "we cannot find any audio file, it is required to provide "
              "raw array and sample rate, so the audio file will be cached.")
        from soundfile import write
        write(inpath, data=raw, samplerate=self.sr)
        path = inpath
      # if in debug mode or not
      command = 'SMILExtract -loglevel %d -C %s -I %s -O %s' % \
          (self._log_level, self.config_path, path, outpath)
      os.system(command)
      results = np.genfromtxt(outpath, dtype='float32',
                              delimiter=',', skip_header=0)
    except Exception as e:
      import traceback; traceback.print_exc()
      raise e
    finally:
      if os.path.exists(inpath):
        os.remove(inpath)
      if os.path.exists(outpath):
        os.remove(outpath)
    # ====== post-processing ====== #
    X_update = self._post_processing(results)
    if not isinstance(X_update, dict):
      raise ValueError("_post_processing must return a dictionary.")
    return X_update

# ===========================================================================
# Extractor
# ===========================================================================
class openSMILEf0(_openSMILEbase):
  """ HNR based on F0 harmonics and other spectral peaks """

  def __init__(self, frame_length, step_length=None,
               fmin=52, fmax=620, voicingCutoff=0.7,
               n_candidates=8, sr=None):
    super(openSMILEf0, self).__init__(sr=sr)
    self.frame_length = float(frame_length)
    if step_length is None:
      step_length = frame_length / 4
    self.step_length = float(step_length)
    self._config_file = _get_conf_file('openSMILEf0.cfg')
    self.fmin = int(fmin)
    self.fmax = int(fmax)
    self.voicingCutoff = float(voicingCutoff)
    self.n_candidates = int(n_candidates)

  @property
  def config(self):
    args = {'framesize': self.frame_length,
            'framestep': self.step_length,
            'fmin': self.fmin,
            'fmax': self.fmax,
            'ncandidates': self.n_candidates,
            'voicingCutoff': self.voicingCutoff}
    return args

  def _post_processing(self, X):
    f0 = X[:-1]
    return {'f0': f0[:, np.newaxis]}

class openSMILEloudness(_openSMILEbase):
  """ Loudness via simple auditory band model
  """

  def __init__(self, frame_length, step_length=None,
               nmel=40, fmin=20, fmax=None,
               to_intensity=False, sr=None):
    super(openSMILEloudness, self).__init__(sr=sr)
    self.frame_length = float(frame_length)
    if step_length is None:
      step_length = frame_length / 4
    self.step_length = float(step_length)
    self.nmel = int(nmel)
    self._config_file = _get_conf_file('openSMILEloudness.cfg')
    self.fmin = int(fmin)
    self.fmax = fmax
    self.to_intensity = bool(to_intensity)

  @property
  def config(self):
    if self.fmax is None:
      self.fmax = self.sr // 2
    args = {'framesize': self.frame_length,
            'framestep': self.step_length,
            'fmin': self.fmin,
            'fmax': self.fmax,
            'nmel': self.nmel}
    return args

  def _post_processing(self, X):
    name = 'loudness'
    if self.to_intensity:
      X = X * 60
      name = 'intensity'
    return {name: X[:, np.newaxis]}

class openSMILEpitch(_openSMILEbase):
  """
  Parameters
  ----------
  fmin: int
      minimum pitch frequency
  fmax: int
      maximum pitch frequency
  voicingCutoff_pitch: float [0-1]
      This sets the voicing probability threshold for pitch
      detection [0.0 - 1.0]. Frames with voicing probability
      values above this threshold will be considered as voiced.
      for 'shs', default value is `0.7`
      for 'acf', default value is `0.55`
  f0min: int
      minimum F0-frequency
  f0max: int
      maximum F0-frequency
  n_candidates: int [1-20]
      The number of F0 candidates to output [1-20] (0 disables
      ouput of candidates AND their voicing probs.).
  voicingCutoff_f0: float [0-1]
      same value applied for F0 extracting
  method: 'shs', 'acf'
      shs: subharmonic summation
      acf: autocorrelation function
  f0: bool
      if True, return F0 frequency also.
  loudness: bool
      if True, append loudness values to the output
      `L : L = (I/I0)^0.3`
      where `I` is intensity values, and I0 is constant for 60 dB and
      max. amplitude = 1.0 (I0 = 10^-6 or 0.000001)
      (for sample values normalised to the range -1..1)",0)
      intensity values is mean of squared input values
      multiplied by a Hamming window.
  voiceProb: bool
      if True, append `sap` speech activities probabilities to
      the output
  """

  def __init__(self, frame_length, step_length=None,
               window='gauss',
               fmin=52, fmax=620, voicingCutoff_pitch=0.7,
               f0min=64, f0max=400, n_candidates=8, voicingCutoff_f0=0.45,
               method='shs',
               f0=False, loudness=False, voiceProb=False,
               sr=None):
    super(openSMILEpitch, self).__init__(sr=sr)
    # ====== config ====== #
    self.frame_length = float(frame_length)
    if step_length is None:
      step_length = frame_length / 4
    self.step_length = float(step_length)
    self.window = str(window)
    # for pitch
    self.fmin = int(fmin)
    self.fmax = int(fmax)
    self.voicingCutoff_pitch = np.clip(float(voicingCutoff_pitch), 0., 1.)
    # for F0
    self.f0 = bool(f0)
    self.f0_config = _get_conf_file('smileF0.cfg')
    self.f0min = int(f0min)
    self.f0max = int(f0max)
    self.n_candidates = int(n_candidates)
    self.voicingCutoff_f0 = np.clip(float(voicingCutoff_f0), 0., 1.)
    # others
    self.loudness = bool(loudness)
    self.voiceProb = bool(voiceProb)
    # ====== method ====== #
    method = str(method).lower()
    self.method_name = method
    if method == 'shs':
      self.method = _get_conf_file('prosodyShs.cfg')
    elif method == 'acf':
      self.method = _get_conf_file('prosodyAcf.cfg')
    else:
      raise ValueError("Only two methods support: acf (autocorrelation function), "
          "and shs (subharmonic summation).")

  @property
  def config(self):
    # check method for extracting pitch
    method = self.method.format(**{
        'fmin': self.fmin, 'fmax': self.fmax,
        'voicingCutoff': self.voicingCutoff_pitch})
    method_path = self.config_path + '.method'
    with open(method_path, 'w') as f:
      f.write(method)
    # check method for extracting F0
    if self.f0:
      if self.method_name == 'acf':
        turn_on_specscale = ''
      else:
        turn_on_specscale = ';'
      f0_config = self.f0_config.format(**{
          'fmin': self.f0min, 'fmax': self.f0max,
          'nCandidates': self.n_candidates,
          'turn_on_specscale': turn_on_specscale,
          'voicingCutoff': self.voicingCutoff_f0})
      f0_path = self.config_path + '.f0'
      with open(f0_path, 'w') as f:
        f.write(f0_config)
      f0_path = '\\{%s}' % f0_path
      f0_flag = ';F0'
    else:
      f0_path = ''
      f0_flag = ''
    return {'framesize': self.frame_length, 'framestep': self.step_length,
            'window': self.window, 'method': '\\{%s}' % method_path,
            'f0': f0_path, 'f0_flag': f0_flag}

  def _post_processing(self, X):
    X = X[:, 1:] # remove timestamp
    if self.method_name == 'shs':
      X_loud = X[:, 2:3] # loud
      X_sap = X[:, 1:2] # sap
      X_pitch = X[:, 0:1] # pitch
    else:
      X_loud = X[:, 2:3] # loud
      X_sap = X[:, 0:1] # sap
      X_pitch = X[:, 1:2] # pitch
    ret = {'pitch': X_pitch}
    if self.f0:
      ret['f0'] = X[:, 3:4]
    if self.loudness:
      ret['loudness'] = X_loud
    if self.voiceProb:
      ret['sap'] = X_sap
    return ret

class openSMILEsad(_openSMILEbase):
  """ SMILEsad
  NOTE: This is only for testing, this method is really not efficience
  """

  def __init__(self, frame_length, step_length=None,
               window='ham', threshold=None, sr=None,
               output_name='sad'):
    super(openSMILEsad, self).__init__(sr=sr)
    # ====== verifying ====== #
    from odin.fuel import openSMILEsad as SADmodel
    sad_ds = SADmodel.get_dataset()
    self._lstm_path = sad_ds['lstmvad_rplp18d_12.net']
    self._init_path = sad_ds['rplp18d_norm.dat']
    # ====== config ====== #
    self.frame_length = float(frame_length)
    if step_length is None:
      step_length = frame_length / 4
    self.step_length = float(step_length)
    self.window = str(window)
    self.threshold = None if threshold is None \
        else np.clip(threshold, -1., 1.)
    self._output_name = str(output_name)

  @property
  def config(self):
    return {'framesize': self.frame_length, 'framestep': self.step_length,
            'netfile': self._lstm_path, 'initfile': self._init_path,
            'hifreq': self.sr // 2 if self.sr < 16000 else 8000,
            'window': self.window}

  def _post_processing(self, X):
    X = X[:, -1] # remove timestamp
    if is_number(self.threshold):
      X = (X >= self.threshold).astype("bool")
    return {self.output_name: X}
