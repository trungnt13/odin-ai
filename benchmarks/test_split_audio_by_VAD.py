from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import numpy as np
from scipy.signal import medfilt

from odin.visual import plot_save
from odin.preprocessing import signal, speech
from odin.utils import play_audio

f = '/Users/trungnt13/tmp/tmp.sph'
s, sr = speech.read(f)


def split_by_vad(s, sr, maximum_duration=30, frame_length=256,
                 nb_mixtures=3, threshold=0.4):
    """ Splitting an audio based on VAD indicator.
    * The audio is segmented into multiple with length given by `frame_length`
    * Log-energy is calculated for each frames
    * Gaussian mixtures with `nb_mixtures` is fitted, and output vad indicator
      for each frames.
    * A flat window (ones-window) of `frame_length` is convolved with the
      vad indices.
    * All frames within the percentile <= `threshold` is treated as silence.
    * The splitting process is greedy, frames is grouped until reaching the
      maximum duration

    Parameters
    ----------
    s: 1-D numpy.ndarray
        loaded audio array
    sr: int
        sample rate
    maximum_duration: float (second)
        maximum duration of each segments in seconds
    frame_length: int
        number of frames for windowing
    nb_mixtures: int
        number of Gaussian mixture for energy-based VAD (the higher
        the more overfitting).
    threshold: float (0. - 1.)
        The higher the values, the more silence points are taken into
        account for splitting the audio
        (with 1. mean that you are sure every frames are SILIENCE,
        and 0. mean that most of the frame are voiced.)

    Return
    ------
    list of audio arrays

    """
    frame_length = int(frame_length)
    maximum_duration = maximum_duration * sr
    # ====== check if audio long enough ====== #
    if len(s) < maximum_duration:
        return [s]
    # ====== start spliting ====== #
    maximum_duration /= frame_length
    frames = signal.segment_axis(s, frame_length, frame_length,
                                 axis=0, end='pad', endvalue=0.)
    energy = signal.get_energy(frames, log=True)
    vad = speech.vad_energy(energy, distrib_nb=nb_mixtures, nb_train_it=24)[0]
    vad = signal.smooth(vad, win=frame_length, window='flat')
    # ====== get all possible sliences ====== #
    indices = np.where(vad <= np.percentile(vad, q=threshold * 100))[0].tolist()
    if len(vad) - 1 not in indices:
        indices.append(len(vad) - 1)
    # ====== spliting the audio ====== #
    segments = []
    start = 0
    prev_end = 0
    # greedy adding new frames to reach desire maximum length
    for end in indices:
        # over-reach the maximum length
        if end - start > maximum_duration:
            segments.append((start, prev_end))
            start = prev_end
        # exact maximum length
        elif end - start == maximum_duration:
            segments.append((start, end))
            start = end
        prev_end = end
    # add ending index if necessary
    if indices[-1] != segments[-1][-1]:
        segments.append((start, indices[-1]))
    # ====== convert everythng to raw signal index ====== #
    segments = [[i * frame_length, j * frame_length]
                for i, j in segments]
    segments[-1][-1] = s.shape[0]
    return [s[i:j] for i, j in segments]

segs = split_by_vad(s, sr, frame_length=0.02 * sr)

plt.figure(figsize=(20, 5))
plt.plot(s)

plt.figure(figsize=(20, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.plot(segs[i])
plt.show()
