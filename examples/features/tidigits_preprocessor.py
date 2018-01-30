from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = "cpu,float32"
import shutil

import numpy as np

from odin import fuel as F
from odin import preprocessing as pp
from odin.utils import (get_all_files, get_all_ext, exec_commands,
                        MPI, cpu_count, Progbar)

README = \
"""
Original sample rate: 20,000 Hz
Downsampled sample rate: 8,000 Hz

Saved WAV file format:
    * [train|test]
    * [m|w|b|g] (alias for man, women, boy, girl)
    * [age]
    * [dialectID]
    * [speakerID]
    * [production]
    * [digit_sequence]
    => "train_g_08_17_as_a_4291815"

    train material, child "g"irl, age is "08", dialect group is "17",
    speaker code "as", "a" is first production,
    digit sequence "4-2-9-1-8-1-5".
--------------------
Category       Symbol    Number    Age Range (years)
  Man            M        111           21 - 70
  Woman          W        114           17 - 59
  Boy            B         50            6 - 14
  Girl           G         51            8 - 15

Eleven digits were used:  "zero", "one", "two", ... , "nine", and "oh".

Digits for each speaker:
    22 isolated digits (two tokens of each of the eleven digits)
    11 two-digit sequences
    11 three-digit sequences
    11 four-digit sequences
    11 five-digit sequences
    11 seven-digit sequences

Example of original data:
     => "/data/adults/train/man/fd/6z97za.wav"
     training material, adult male, speaker code "fd",
     digit sequence "six zero nine seven zero", "a" is first production.

     => "/data/adults/test/woman/pf/1b.wav"
     test material, adult female, speaker code "pf",
     digit sequence "one", "b" is second production.
------------------
    City                      Dialect             M    W    B    G

01 Boston, MA            Eastern New England      5    5    0    1
02 Richmond, VA          Virginia Piedmont        5    5    2    4
03 Lubbock, TX           Southwest                5    5    0    1
04 Los Angeles, CA       Southern California      5    5    0    1
05 Knoxville, TN         South Midland            5    5    0    0
06 Rochester, NY         Central New York         6    6    0    0
07 Denver, CO            Rocky Mountains          5    5    0    0
08 Milwaukee, WS         North Central            5    5    2    0
09 Philadelphia, PA      Delaware Valley          5    6    0    1
10 Kansas City, KS       Midland                  5    5    4    1
11 Chicago, IL           North Central            5    5    1    2
12 Charleston, SC        South Carolina           5    5    1    0
13 New Orleans, LA       Gulf South               5    5    2    0
14 Dayton, OH            South Midland            5    5    0    0
15 Atlanta, GA           Gulf South               5    5    0    1
16 Miami, FL             Spanish American         5    5    1    0
17 Dallas, TX            Southwest                5    5   34   36
18 New York, NY          New York City            5    5    2    2
19 Little Rock, AR       South Midland            5    6    0    0
20 Portland, OR          Pacific Northwest        5    5    0    0
21 Pittsburgh, PA        Upper Ohio Valley        5    5    0    0
22                       Black                    5    6    1    1

                         Total Speakers         111  114   50   51    326
"""

# ===========================================================================
# CONST
# ===========================================================================
inpath = "/mnt/sdb1/TIDIGITS"
wav_path = os.path.join(inpath, "wave")
wav_ds = os.path.join(inpath, "raw")
infopath = os.path.join(inpath, 'data/children/doc/spkrinfo.txt')

print('Input path:', inpath)
print('Convert to WAV at:', wav_path)
print('WAVE dataset:', wav_ds)

exts = get_all_ext(inpath)
audio_files = get_all_files(inpath,
                filter_func=lambda f: f[-4:] == '.wav' and
                            f.split('/')[-3] in ('girl', 'boy', 'man', 'woman'))
# ID     Gender     Age     Dialect    Usage
# ID - Unique 2-character speaker identifier
# Gender - (M-man, W-woman, B-boy, G-girl)
# Age - Speaker age at time of recording
# Dialect - Dialect region identifier (see file "dialects.txt" for decode)
# Usage - (TST-test material, TRN-training material)
info = np.genfromtxt(infopath, dtype=str, skip_header=12)
info = {ID.lower(): (Gender.lower(), Age, Dialect, Usage)
        for ID, Gender, Age, Dialect, Usage in info}
gender_map = {
    "man": "m",
    "woman": "w",
    "boy": "b",
    "girl": "g"
}
usage_map = {
    "TST": "test",
    "TRN": "train"
}

def get_name(path):
  # integrate all info into the name
  usage, gender, ID, digits = path.split('/')[-4:]
  production = digits[-5]
  digits = digits[:-5]
  gender = gender_map[gender]
  gender_, age_, dialect_, usage_ = info[ID]
  usage_ = usage_map[usage_]
  assert usage == usage_ and gender == gender_, path
  name = '%s_%s_%s_%s_%s_%s_%s.wav' % \
      (usage, gender, age_, dialect_, ID, production, digits)
  return name
# ===========================================================================
# Convert all SPHERE to wav using sph2pipe
# ===========================================================================
if os.path.exists(wav_path):
  shutil.rmtree(wav_path)
# ====== convert all compress audio to .wav using sph2pipe ====== #
if not os.path.exists(wav_path):
  os.mkdir(wav_path)
  cmds = ["sph2pipe %s %s -f rif" % (path, os.path.join(wav_path, get_name(path)))
          for path in audio_files]

  def mpi_fn(cmd):
    exec_commands(cmd, print_progress=False)
    yield len(cmd)
  mpi = MPI(jobs=cmds, func=mpi_fn,
            ncpu=cpu_count() - 1, batch=12)
  prog = Progbar(target=len(cmds),
                 print_report=True, print_summary=True,
                 name='Converting .sph to .wav')
  for i in mpi:
    prog.add(i)
# ===========================================================================
# store everything in one dataset
# ===========================================================================
if os.path.exists(wav_ds):
  shutil.rmtree(wav_ds)
# ====== create new segments list ====== #
segments = get_all_files(wav_path,
                         filter_func=lambda f: f[-4:] == '.wav')
if not os.path.exists(wav_ds):
  wave = pp.FeatureProcessor(segments,
      extractor=[
          pp.speech.AudioReader(sr=None, sr_new=8000, best_resample=True,
                                remove_dc_n_dither=False, preemphasis=None),
          pp.base.NameConverter(converter=lambda x: os.path.basename(x).split('.')[0],
                                keys='path'),
          pp.base.AsType(type_map={'raw': 'float16'})
      ],
      path=wav_ds, ncache=300, ncpu=8, override=True)
  wave.run()
  pp.validate_features(wave, '/tmp/tidigits_wave', override=True,
                       nb_samples=8)
  with open(os.path.join(wav_ds, 'README'), 'w') as f:
    f.write(README)
# ====== validate the data ====== #
ds = F.Dataset(wav_ds, read_only=True)
print(ds)
ds.close()
