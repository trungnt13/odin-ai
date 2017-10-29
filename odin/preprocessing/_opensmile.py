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
from collections import Mapping

import numpy as np

from odin.utils import is_string, get_script_path, ctext, uuid, is_number

from .base import Extractor

__all__ = [
    'SMILEpitch',
    'SMILEsad'
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


# ===========================================================================
# Main extractor
# ===========================================================================
class oSMILEpitch(Extractor):
    """
    Parameters
    ----------
    method: 'shs', 'acf'
        shs: subharmonic summation
        acf: autocorrelation function
    """

    def __init__(self, frame_length, step_length=None,
                 window='gauss', fmin=52, fmax=620,
                 sr=8000, method='shs'):
        super(oSMILEpitch, self).__init__()


class oSMILEsad(Extractor):
    """docstring for SMILEsad"""

    def __init__(self, frame_length, step_length=None,
                 window='ham', threshold=None, sr=8000):
        super(oSMILEsad, self).__init__()
        # ====== verifying ====== #
        from odin.fuel import load_sad_model
        verify_dependencies()
        self._lstm_path, self._init_path = load_sad_model()
        self._sad_conf = _get_conf_file('sad.cfg')
        # ====== config ====== #
        self.frame_length = float(frame_length)
        if step_length is None:
            step_length = frame_length / 4
        self.step_length = float(step_length)
        self.sr = int(sr)
        self.window = str(window)
        self.threshold = None if threshold is None \
            else np.clip(threshold, -1., 1.)
        # ====== update config ====== #
        self._config_id = str(uuid(length=8))
        self._update_config()

    def get_config_path(self):
        return '/tmp/sad%s.cfg' % self._config_id

    def _update_config(self):
        kwargs = {'framesize': self.frame_length, 'framestep': self.step_length,
                  'netfile': self._lstm_path, 'initfile': self._init_path,
                  'hifreq': self.sr // 2 if self.sr < 16000 else 8000,
                  'window': self.window}
        with open(self.get_config_path(), 'w') as f:
            f.write(self._sad_conf.format(**kwargs))

    def _transform(self, X):
        # ====== file input file ====== #
        if isinstance(X, Mapping):
            if 'path' in X:
                path = X['path']
            if 'sr' in X and self.sr != X['sr']:
                raise ValueError("Given sample rate: %d, but the audio file has "
                                 "sample rate: %d" % (self.sr, X['sr']))
        elif is_string(X):
            path = X
        else:
            raise ValueError("openSMILE extractor require path to audio file.")
        # ====== extract SAD ====== #
        outpath = '/tmp/sad%d.csv' % os.getpid()
        try:
            os.system('SMILExtract -loglevel -1 -C %s -I %s -O %s' %
                      (self.get_config_path(), path, outpath))
            sad = np.genfromtxt(outpath, dtype='float32',
                                delimiter=',', skip_header=0)
            sad = sad[:, -1]
        except Exception as e:
            if os.path.exists(outpath):
                os.remove(outpath)
            import traceback; traceback.print_exc()
            raise e
        # ====== post-processing ====== #
        if is_number(self.threshold):
            sad = (sad >= self.threshold).astype("uint8")
        X['sad'] = sad
        return X
