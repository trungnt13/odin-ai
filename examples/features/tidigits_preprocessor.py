from __future__ import print_function, division, absolute_import

from odin.utils import ArgController
args = ArgController(
).add('--reset', 'force re-run the wave convertion, and wave dataset preprocessing', False
).parse()

import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = "cpu,float32"
import shutil

import numpy as np

from odin.visual import plot_save, plot_audio
from odin.preprocessing import speech
from odin import nnet as N, fuel as F, backend as K
from odin.utils import get_all_files, get_all_ext, exec_commands

README = \
"""
Original sample rate: 20,000 Hz
Downsampled sample rate: 16,000 Hz

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

Saved WAV file format:
    * [train|test]
    * [m|w|b|g] (alias for man, women, boy, girl)
    * [age]
    * [dialectID]
    * [speakerID]
    * [production]
    * [digit_sequence]
    => "train_g_08_17_as_a_4291815.wav"

    train material, child "g"irl, age is "08", dialect group is "17",
    speaker code "as", "a" is first production,
    digit sequence "4-2-9-1-8-1-5".
"""


# ===========================================================================
# CONST
# ===========================================================================
inpath = "/mnt/sdb1/TIDIGITS"
outpath = '/home/trung/data/tidigits'
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
# ====== convert all compress audio to .wav using sph2pipe ====== #
if not os.path.exists(wav_path):
    os.mkdir(wav_path)
    cmds = ["sph2pipe %s %s -f rif" % (path, os.path.join(wav_path, get_name(path)))
            for path in audio_files]
    exec_commands(cmds, print_progress=True)
# ====== create new segments list ====== #
segments = get_all_files(wav_path, filter_func=lambda f: f[-4:] == '.wav')
# remove .wav extension
segments = [[os.path.basename(path).replace('.wav', ''), path, 0, -1, 0]
            for path in segments]
# ====== store everything in one dataset ====== #
if args.reset and os.path.exists(wav_ds):
    shutil.rmtree(wav_ds)
if not os.path.exists(wav_ds):
    wave = F.WaveProcessor(segments, output_path=wav_ds,
                           sr=None, sr_info={}, sr_new=16000, best_resample=True,
                           audio_ext='.wav', pcm=False, remove_dc_offset=True,
                           maxlen=None, dtype='float16', datatype='memmap',
                           ncache=0.2, ncpu=10)
    wave.run()
    wave.validate('/tmp/tidigits_report', nb_samples=25)
    with open(os.path.join(wav_ds, 'README'), 'w') as f:
        f.write(README)
# ===========================================================================
# Acoustic feature extractor
# ===========================================================================
# ====== plot sampling of files ====== #
ds = F.Dataset(wav_ds, read_only=True)
print(ds)
# ====== processing ====== #
if os.path.exists(outpath):
    print("Remove old dataset at path:", outpath)
    shutil.rmtree(outpath)
acous = F.SpeechProcessor(ds, outpath,
                sr=16000, sr_info={}, sr_new=None,
                win=0.02, hop=0.005, window='hann',
                nb_melfilters=40, nb_ceps=13,
                get_spec=True, get_qspec=False, get_phase=False,
                get_pitch=True, get_f0=True,
                get_vad=True, get_energy=True, get_delta=2,
                fmin=64, fmax=None,
                pitch_threshold=0.3, pitch_fmax=260, pitch_algo='swipe',
                vad_smooth=3, vad_minlen=0.1,
                cqt_bins=96, preemphasis=0.97,
                center=True, power=2, log=True, backend='odin',
                pca=True, pca_whiten=False,
                audio_ext=None, save_raw=True,
                save_stats=True, substitute_nan=None,
                dtype='float16', datatype='memmap',
                ncache=250, ncpu=10)
acous.run()
# copy README
with open(os.path.join(outpath, 'README'), 'w') as f:
    f.write(README)
# ====== validate saved dataset ====== #
ds = F.Dataset(outpath, read_only=True)
print(ds)
ds.close()
