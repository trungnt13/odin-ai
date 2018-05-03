# ======================================================================
# Author: TrungNT
# ======================================================================
from __future__ import print_function, division

import os
import unittest
from six.moves import zip, range, cPickle

import numpy as np

from odin import fuel as F
from odin.preprocessing import speech, signal


class SignalTest(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_stft_istft(self):
        try:
            import librosa
            ds = F.load_digit_wav()
            name = ds.keys()[0]
            path = ds[name]

            y, _ = speech.read(path, pcm=True)
            hop_length = int(0.01 * 8000)
            stft = signal.stft(y, n_fft=256, hop_length=hop_length, window='hann')
            stft_ = librosa.stft(y, n_fft=256, hop_length=hop_length, window='hann')
            self.assertTrue(np.allclose(stft, stft_.T))

            y1 = signal.istft(stft, hop_length=hop_length, window='hann')
            y2 = librosa.istft(stft_, hop_length=hop_length, window='hann')
            self.assertTrue(np.allclose(y1, y2))
        except ImportError:
            print("test_stft_istft require librosa.")
