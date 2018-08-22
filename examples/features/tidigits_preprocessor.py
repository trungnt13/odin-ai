from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = "cpu=1,float32"
import shutil

import numpy as np

from odin import fuel as F, nnet as N
from odin import preprocessing as pp
from odin.utils import (get_all_files, get_all_ext, exec_commands,
                        MPI, cpu_count, Progbar, ArgController,
                        stdio, ctext, crypto)

args = ArgController(
).add('path', "path to TIDIGITS dataset"
).add('--wav', "re-run Converting sphere file to wave", False
).add('--ds', "re-run Group wave files into a dataset", False
).add('--compress', "re-run compression of the dataset", False
).parse()

TOTAL_FILES = 25096
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
# ====== main path ====== #
# inpath = "/mnt/sdb1/TIDIGITS"
inpath = args.path
outpath = '/home/trung/data/TIDIGITS_wav'
compress_path = '/home/trung/data/TIDIGITS.zip'
# ====== others ====== #
wav_path = os.path.join(inpath, "wave")
infopath = os.path.join(inpath, 'data/children/doc/spkrinfo.txt')
logpath = os.path.join(inpath, 'log.txt')
print("Input path:       ", ctext(inpath, 'cyan'))
print("Output path:      ", ctext(outpath, 'cyan'))
print("Convert to WAV at:", ctext(wav_path, 'cyan'))
print("Log path:         ", ctext(logpath, 'cyan'))
stdio(logpath)

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
  if args.wav:
    print("Override wave files at '%s'" % wav_path)
    shutil.rmtree(wav_path)
  elif len(os.listdir(wav_path)) != TOTAL_FILES:
    print("Found only %d files at '%s', delete old wave files" %
      (len(os.listdir(wav_path)), wav_path))
    shutil.rmtree(wav_path)
# ====== convert all compress audio to .wav using sph2pipe ====== #
if not os.path.exists(wav_path):
  os.mkdir(wav_path)
  cmds = ["sph2pipe %s %s -f rif" % (path, os.path.join(wav_path, get_name(path)))
          for path in audio_files]

  def mpi_fn(cmd):
    exec_commands(cmd, print_progress=False)
    yield len(cmd)
  prog = Progbar(target=len(cmds),
                 print_report=True, print_summary=True,
                 name='Converting .sph to .wav')
  # run the MPI tasks
  mpi = MPI(jobs=cmds, func=mpi_fn,
            ncpu=cpu_count() - 1, batch=12)
  for i in mpi:
    prog.add(i)
# ===========================================================================
# Extract Acoustic features
# ===========================================================================
jobs = get_all_files(wav_path,
                     filter_func=lambda x: '.wav' == x[-4:])
assert len(jobs) == TOTAL_FILES
# ====== configuration ====== #
if not os.path.exists(outpath) or args.ds:
  extractors = pp.make_pipeline(steps=[
      pp.speech.AudioReader(sr=None, sr_new=8000, best_resample=True,
                            remove_dc=True),
      pp.base.Converter(converter=lambda x: os.path.basename(x).split('.')[0],
                        input_name='path', output_name='name'),
      pp.base.AsType(dtype='float16', input_name='raw')
  ], debug=False)
  processor = pp.FeatureProcessor(jobs=jobs, path=outpath, extractor=extractors,
                                  n_cache=0.08, ncpu=None, override=True)
  processor.run()
  pp.validate_features(processor, path='/tmp/tidigits', nb_samples=12,
                       override=True)
  with open(os.path.join(outpath, 'README'), 'w') as f:
    f.write(README)
# ====== check the preprocessed dataset ====== #
ds = F.Dataset(outpath, read_only=True)
print(ds)
print(ctext(ds.md5, 'yellow'))
ds.close()
# ====== compress ====== #
if not os.path.exists(compress_path) or args.compress:
  if os.path.exists(compress_path):
    os.remove(compress_path)
  crypto.zip_aes(in_path=outpath, out_path=compress_path,
                 verbose=True)
print("Log at path:", ctext(logpath, 'cyan'))
