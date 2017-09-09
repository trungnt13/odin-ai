from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

import numpy as np
from scipy.signal import medfilt

from odin.visual import plot_save
from odin.preprocessing import signal, speech
from odin.utils import play_audio


def test_func(s, sr, maximum_duration=30, minimum_duration=None,
              frame_length=256, nb_mixtures=3, threshold=0.3,
              return_vad=False, return_voices=False, return_cut=False):
    """ Splitting an audio based on VAD indicator.
    * The audio is segmented into multiple with length given by `frame_length`
    * Log-energy is calculated for each frames
    * Gaussian mixtures with `nb_mixtures` is fitted, and output vad indicator
      for each frames.
    * A flat window (ones-window) of `frame_length` is convolved with the
      vad indices.
    * All frames within the percentile >= `threshold` is treated as voiced.
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
    minimum_duration: None, or float (second)
        all segments below this length will be merged into longer segments,
        if None, any segments with half of the `maximum_duration`
        are considered.
    frame_length: int
        number of frames for windowing
    nb_mixtures: int
        number of Gaussian mixture for energy-based VAD (the higher
        the more overfitting).
    threshold: float (0. to 1.)
        The higher the values, the more frames are considered as voiced,
        this value is the lower percentile of voiced frames.
    return_vad: bool
        if True, return VAD confident values
    return_voices: bool
        if True, return the voices frames indices
    return_cut: bool
        if True, return the cut points of the audio.

    Return
    ------
    segments: list of audio arrays
    vad (optional): list of 0, 1 for VAD indices
    voices (optional): list of thresholded VAD for more precise voices frames.
    cut (optional): list of indicator 0, 1 (1 for the cut point)

    Note
    ----
    this function does not guarantee the output audio length is always
    smaller than `maximum_duration`, the higher the threshold, the better
    chance you get everything smaller than `maximum_duration`
    """
    frame_length = int(frame_length)
    maximum_duration = maximum_duration * sr
    results = []
    # ====== check if audio long enough ====== #
    if len(s) < maximum_duration:
        if return_cut or return_vad or return_voices:
            raise ValueError("Cannot return `cut` points, `vad` or `voices` since"
                        "the original audio is shorter than `maximum_duration`, "
                        "hence, no need for splitting.")
        return [s]
    maximum_duration /= frame_length
    if minimum_duration is None:
        minimum_duration = maximum_duration // 2
    else:
        minimum_duration = minimum_duration * sr / frame_length
        minimum_duration = np.clip(minimum_duration, 0., 0.99 * maximum_duration)
    # ====== start spliting ====== #
    frames = signal.segment_axis(s, frame_length, frame_length,
                                 axis=0, end='pad', endvalue=0.)
    energy = signal.get_energy(frames, log=True)
    vad = signal.vad_energy(energy, distrib_nb=nb_mixtures, nb_train_it=33)[0]
    vad = signal.smooth(vad, win=frame_length, window='flat')
    # explicitly return VAD
    if return_vad:
        results.append(vad)
    # ====== get all possible sliences ====== #
    # all voice indices
    indices = np.where(vad >= np.percentile(vad, q=threshold * 100))[0].tolist()
    if len(vad) - 1 not in indices:
        indices.append(len(vad) - 1)
    # explicitly return voiced frames
    if return_voices:
        tmp = np.zeros(shape=(len(vad),))
        tmp[indices] = 1
        results.append(tmp)
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
    # if found NO segments just return original file
    if len(segments) == 0:
        return [s]
    # add ending index if necessary
    if indices[-1] != segments[-1][-1]:
        segments.append((start, indices[-1]))
    # re-fining, short segments will be merged into bigger onces
    found_under_length = True
    while found_under_length:
        new_segments = []
        found_under_length = False
        for (s1, e1), (s2, e2) in zip(segments, segments[1:]):
            # merge if length < length_threshold
            if (e1 - s1) < minimum_duration or (e2 - s2) < minimum_duration:
                new_segments.append((s1, e2))
                found_under_length = True
            # keep both of the segments
            else:
                new_segments.append((s1, e1))
                new_segments.append((s2, e2))
        segments = new_segments
    # explicitly return cut points
    if return_cut:
        tmp = np.zeros(shape=(segments[-1][-1] + 1,))
        for i, j in segments:
            tmp[i] = 1; tmp[j] = 1
        results.append(tmp)
    # ====== convert everythng to raw signal index ====== #
    segments = [[i * frame_length, j * frame_length]
                for i, j in segments]
    segments[-1][-1] = s.shape[0]
    # cut segments out of raw audio array
    segments = [s[i:j] for i, j in segments]
    results = [segments] + results
    return results[0] if len(results) == 1 else results

files = [
    '/Users/trungnt13/tmp/20051026_180611_340_fsp-b.sph',
    '/Users/trungnt13/tmp/20051118_212058_553_fsp-b.sph',
    '/Users/trungnt13/tmp/20110213_140144_892-b.sph',
    '/Users/trungnt13/tmp/en_4077-a.sph',
    '/Users/trungnt13/tmp/en_4660-a.sph',
    '/Users/trungnt13/tmp/en_6402-b.sph',
    '/Users/trungnt13/tmp/fla_0811-b.sph',
    '/Users/trungnt13/tmp/fla_0946-b.sph',
    '/Users/trungnt13/tmp/lre17_vlujcseb.sph'
]
for f in files:
    print(f)
    s, sr = speech.read(f, remove_dc_offset=True, remove_zeros=True)
    segs, vad, voices, cut = test_func(s, sr, frame_length=0.02 * sr, maximum_duration=10, minimum_duration=None,
                                       return_vad=True, return_cut=True, return_voices=True)
    for s in segs:
        print(s.shape, len(s) / sr)
    plt.figure(figsize=(10, 6))
    plt.subplot(4, 1, 1)
    plt.plot(speech.resample(s, sr, 2000, best_algorithm=False))
    plt.subplot(4, 1, 2)
    plt.plot(vad)
    plt.subplot(4, 1, 3)
    plt.plot(voices)
    plt.subplot(4, 1, 4)
    plt.plot(cut)
    plt.show()
