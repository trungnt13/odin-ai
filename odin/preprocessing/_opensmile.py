# -*- coding: utf-8 -*-
# openSMILE LICENSE
# ===========
# Authors: Florian Eyben, Felix Weninger, Martin Woellmer, Bjoern Schuller
# Copyright (C) 2008-2013, Institute for Human-Machine Communication, TUM
# Copyright (C) 2013-2014, audEERING UG (haftungsbeschrÃ¤nkt)
# http://www.audeering.com/research-and-open-source/files/openSMILE-open-source-license.txt
from __future__ import print_function, division, absolute_import

import os
import re
import subprocess
from six import add_metaclass
from collections import Mapping
from abc import abstractproperty, ABCMeta, abstractmethod

import numpy as np

from odin.utils import (is_string, get_script_path, ctext, is_number,
                        get_logpath)

from .base import Extractor

__all__ = [
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
        m = re.search('openSMILE version (.*)', output, re.MULTILINE)
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


_UNIQUE_ID = 0


def _get_unique_id():
    global _UNIQUE_ID
    _UNIQUE_ID += 1
    return _UNIQUE_ID


# ===========================================================================
# Main extractor
# ===========================================================================
@add_metaclass(ABCMeta)
class _openSMILEbase(Extractor):

    def __init__(self, sr):
        verify_dependencies()
        super(_openSMILEbase, self).__init__()
        self._id = _get_unique_id()
        self.sr = sr
        self._first_config_generated = False
        self._conf = _get_conf_file('%s.cfg' % self.__class__.__name__)

    # ==================== abstract ==================== #
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
        if not self._first_config_generated:
            self._first_config_generated = True
            self._update_config()
        # ====== file input file ====== #
        raw = None
        if isinstance(X, Mapping):
            if 'path' in X:
                path = X['path']
            if 'sr' in X:
                if self.sr is None:
                    self.sr = X['sr']
                    self._update_config()
                elif self.sr != X['sr']:
                    raise ValueError("Given sample rate: %d, but the audio file has "
                                     "sample rate: %d" % (self.sr, X['sr']))
            if 'raw' in X:
                raw = X['raw']
        elif is_string(X):
            path = X
        else:
            raise ValueError("openSMILE extractor require path to audio file.")
        # no sample rate specified, cannot generate appropriate config
        if self.sr is None:
            raise RuntimeError("Cannot acquire sample rate for the input.")
        # ====== extract SAD ====== #
        unique_id = _get_unique_id()
        inpath = os.path.join(
            get_logpath(), '%s%d.wav' % (self.__class__.__name__, unique_id))
        outpath = os.path.join(
            get_logpath(), '%s%d.csv' % (self.__class__.__name__, unique_id))
        try:
            if not os.path.exists(path):
                if raw is None:
                    raise RuntimeError("openSMILE require input audio file, since "
                        "we cannot find any audio file, it is required to provide "
                        "raw array and sample rate, so the audio file will be cached.")
                from soundfile import write
                write(inpath, data=raw, samplerate=self.sr)
                path = inpath
            command = 'SMILExtract -loglevel -1 -C %s -I %s -O %s' % \
                (self.config_path, path, outpath)
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


class openSMILEpitch(_openSMILEbase):
    """
    Parameters
    ----------
    loudness: bool
        if True, append loudness values to the output
    voiceProb: bool
        if True, append `sap` speech activities probabilities to
        the output
    method: 'shs', 'acf'
        shs: subharmonic summation
        acf: autocorrelation function
    """

    def __init__(self, frame_length, step_length=None,
                 window='gauss', fmin=52, fmax=620,
                 f0=False, loudness=False, voiceProb=False, method='shs',
                 sr=None):
        super(openSMILEpitch, self).__init__(sr=sr)
        # ====== config ====== #
        self.frame_length = float(frame_length)
        if step_length is None:
            step_length = frame_length / 4
        self.step_length = float(step_length)
        self.window = str(window)
        self.fmin = int(fmin)
        self.fmax = int(fmax)
        self.loudness = bool(loudness)
        self.voiceProb = bool(voiceProb)
        self.f0 = bool(f0)
        self.f0_config = _get_conf_file('smileF0.cfg')
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
        method = self.method.format(**{'fmin': self.fmin, 'fmax': self.fmax})
        method_path = self.config_path + '.method'
        with open(method_path, 'w') as f:
            f.write(method)
        # check method for extracting F0
        if self.f0:
            if self.method_name == 'acf':
                turn_on_specscale = ''
            else:
                turn_on_specscale = ';'
            f0_config = self.f0_config.format(**{'fmin': self.fmin, 'fmax': self.fmax,
                                                 'turn_on_specscale': turn_on_specscale})
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
                 window='ham', threshold=None, sr=None):
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
        return {"sad": X}
