from __future__ import print_function, division, absolute_import

import numpy as np

# returns a list of augmented audio data, stereo or mono
def augment_audio(y, sr, n_augment = 0, allow_speedandpitch = True, allow_pitch = True,
    allow_speed = True, allow_dyn = True, allow_noise = True, allow_timeshift = True, tab="", quiet=False):

  mods = [y]                  # always returns the original as element zero
  length = y.shape[0]

  for i in range(n_augment):
    if not quiet:
      print(tab + "augment_audio: ", i + 1, "of", n_augment)
    y_mod = y
    count_changes = 0

    # change speed and pitch together
    if (allow_speedandpitch) and random_onoff():
      length_change = np.random.uniform(low=0.9, high=1.1)
      speed_fac = 1.0 / length_change
      if not quiet:
        print(tab + "    resample length_change = ", length_change)
      tmp = np.interp(np.arange(0, len(y), speed_fac), np.arange(0, len(y)), y)
      #tmp = resample(y,int(length*lengt_fac))    # signal.resample is too slow
      minlen = min(y.shape[0], tmp.shape[0])     # keep same length as original;
      y_mod *= 0                                    # pad with zeros
      y_mod[0:minlen] = tmp[0:minlen]
      count_changes += 1

    # change pitch (w/o speed)
    if (allow_pitch) and random_onoff():
      bins_per_octave = 24        # pitch increments are quarter-steps
      pitch_pm = 4                                # +/- this many quarter steps
      pitch_change = pitch_pm * 2 * (np.random.uniform() - 0.5)
      if not quiet:
        print(tab + "    pitch_change = ", pitch_change)
      y_mod = librosa.effects.pitch_shift(y, sr, n_steps=pitch_change, bins_per_octave=bins_per_octave)
      count_changes += 1

    # change speed (w/o pitch),
    if (allow_speed) and random_onoff():
      speed_change = np.random.uniform(low=0.9, high=1.1)
      if not quiet:
        print(tab + "    speed_change = ", speed_change)
      tmp = librosa.effects.time_stretch(y_mod, speed_change)
      minlen = min(y.shape[0], tmp.shape[0])        # keep same length as original;
      y_mod *= 0                                    # pad with zeros
      y_mod[0:minlen] = tmp[0:minlen]
      count_changes += 1

    # change dynamic range
    if (allow_dyn) and random_onoff():
      dyn_change = np.random.uniform(low=0.5, high=1.1)  # change amplitude
      if not quiet:
        print(tab + "    dyn_change = ", dyn_change)
      y_mod = y_mod * dyn_change
      count_changes += 1

    # add noise
    if (allow_noise) and random_onoff():
      noise_amp = 0.005 * np.random.uniform() * np.amax(y)
      if random_onoff():
        if not quiet:
          print(tab + "    gaussian noise_amp = ", noise_amp)
        y_mod += noise_amp * np.random.normal(size=length)
      else:
        if not quiet:
          print(tab + "    uniform noise_amp = ", noise_amp)
        y_mod += noise_amp * np.random.normal(size=length)
      count_changes += 1

    # shift in time forwards or backwards
    if (allow_timeshift) and random_onoff():
      timeshift_fac = 0.2 * 2 * (np.random.uniform() - 0.5)  # up to 20% of length
      if not quiet:
        print(tab + "    timeshift_fac = ", timeshift_fac)
      start = int(length * timeshift_fac)
      if (start > 0):
        y_mod = np.pad(y_mod, (start, 0), mode='constant')[0:y_mod.shape[0]]
      else:
        y_mod = np.pad(y_mod, (0, -start), mode='constant')[0:y_mod.shape[0]]
      count_changes += 1

    # last-ditch effort to make sure we made a change (recursive/sloppy, but...works)
    if (0 == count_changes):
      if not quiet:
        print("No changes made to signal, trying again")
      mods.append(augment_audio(y, sr, n_augment = 1, tab="      ", quiet=quiet)[1])
    else:
      mods.append(y_mod)

  return mods

""" scale frequency axis logarithmically """
def logscale_spec(spec, sr=44100, factor=20., alpha=1.0, f0=0.9, fmax=1):
  spec = spec[:, 0:256]
  timebins, freqbins = np.shape(spec)
  scale = np.linspace(0, 1, freqbins) #** factor

  # http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310
  scale = np.array(map(lambda x: x * alpha if x <= f0 else (fmax - alpha * f0) / (fmax - f0) * (x - f0) + alpha * f0, scale))
  scale *= (freqbins - 1) / max(scale)

  newspec = np.complex128(np.zeros([timebins, freqbins]))
  allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
  freqs = [0.0 for i in range(freqbins)]
  totw = [0.0 for i in range(freqbins)]
  for i in range(0, freqbins):
    if (i < 1 or i + 1 >= freqbins):
      newspec[:, i] += spec[:, i]
      freqs[i] += allfreqs[i]
      totw[i] += 1.0
      continue
    else:
      # scale[15] = 17.2
      w_up = scale[i] - np.floor(scale[i])
      w_down = 1 - w_up
      j = int(np.floor(scale[i]))

      newspec[:, j] += w_down * spec[:, i]
      freqs[j] += w_down * allfreqs[i]
      totw[j] += w_down

      newspec[:, j + 1] += w_up * spec[:, i]
      freqs[j + 1] += w_up * allfreqs[i]
      totw[j + 1] += w_up

  for i in range(len(freqs)):
    if (totw[i] > 1e-6):
      freqs[i] /= totw[i]

  return newspec, freqs
