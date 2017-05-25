from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn

import numpy as np
import shutil
import os
from odin import fuel as F, utils
from odin.preprocessing import speech
from odin import visual

datapath = F.load_digit_wav()
print(datapath)
files = utils.get_all_files(datapath, lambda x: '.wav' in x)
y, sr = speech.read(files[0])
print('Raw signal:', y.shape, sr)

feat = speech.speech_features(y, sr,
    win=0.02, hop=0.01, nb_melfilters=40, nb_ceps=13,
    get_spec=True, get_qspec=True, get_phase=True, get_pitch=True,
    get_vad=True, get_energy=True, get_delta=None,
    pitch_threshold=0.8, fmin=64, fmax=None,
    sr_new=None, preemphasis=0.97)

for i, j in feat.iteritems():
    print(i, j.shape)

plt.subplot(7, 1, 1)
plt.plot(y)

plt.subplot(7, 1, 2)
visual.plot_spectrogram(feat['spec'].T, vad=feat['vad'])
plt.subplot(7, 1, 3)
visual.plot_spectrogram(feat['mspec'].T, vad=feat['vad'])
plt.subplot(7, 1, 4)
visual.plot_spectrogram(feat['mfcc'].T, vad=feat['vad'])

plt.subplot(7, 1, 5)
visual.plot_spectrogram(feat['qspec'].T, vad=feat['vad'])
plt.subplot(7, 1, 6)
visual.plot_spectrogram(feat['qmspec'].T, vad=feat['vad'])
plt.subplot(7, 1, 7)
visual.plot_spectrogram(feat['qmfcc'].T, vad=feat['vad'])

visual.plot_show(block=True, tight_layout=False)
