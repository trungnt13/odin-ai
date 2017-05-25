from __future__ import print_function, division, absolute_import

import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = "cpu,float32,tensorflow"
import shutil
import cPickle

import numpy as np

from odin.visual import plot_save, plot_audio
from odin.preprocessing import speech
from odin import nnet as N, fuel as F, backend as K
from odin.utils import Progbar, get_all_files, get_all_ext, exec_commands

README = \
"""
Sample rate: 20,000 Hz

Category       Symbol    Number    Age Range (years)
  Man            M        111           21 - 70
  Woman          W        114           17 - 59
  Boy            B         50            6 - 14
  Girl           G         51            8 - 15

Eleven digits were used:  "zero", "one", "two", ... , "nine", and "oh".

Each speaker:
    22 isolated digits (two tokens of each of the eleven digits)
    11 two-digit sequences
    11 three-digit sequences
    11 four-digit sequences
    11 five-digit sequences
    11 seven-digit sequences

Example:
     /data/adults/train/man/fd/6z97za.wav

     training material, adult male, speaker code "fd",
     digit sequence "six zero nine seven zero", "a" is first production.

Example:
     /data/adults/test/woman/pf/1b.wav

     test material, adult female, speaker code "pf",
     digit sequence "one", "b" is second production.

Saved WAV file:
    /audio/train_g_08_17_as_a_4291815.wav

    train material, child "g"irl, age is "08", dialect group is "17",
    speaker code "as", "a" is first production,
    digit sequence "4-2-9-1-8-1-5".
"""


# ===========================================================================
# CONST
# ===========================================================================
inpath = "/mnt/sdb1/TIDIGITS"
outpath = '/home/trung/data/tidigits'
wavpath = os.path.join(inpath, "tiaudio")
wav_ds = os.path.join(inpath, "tiwave")
infopath = os.path.join(inpath, 'data/children/doc/spkrinfo.txt')
USE_DOWNSAMPLED_DATASET = True

print('Input path:', inpath)
print('Output dataset:', outpath)
print('Convert WAV to:', wavpath)
print('WAVE dataset:', wav_ds)

exts = get_all_ext(inpath)
audio_files = get_all_files(inpath, filter_func=lambda f: f[-4:] == '.wav' and
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
gender_map = {"man": "m", "woman": "w", "boy": "b", "girl": "g"}
usage_map = {"TST": "test", "TRN": "train"}


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
if True:
    if os.path.exists(wavpath):
        print("[WARNING] Remove old WAV audio at:", wavpath)
        shutil.rmtree(wavpath)
    os.mkdir(wavpath)
    cmds = ["sph2pipe %s %s -f rif" % (path, os.path.join(wavpath, get_name(path)))
            for path in audio_files]
    exec_commands(cmds)
# ====== create new segments list ====== #
segments = get_all_files(wavpath, filter_func=lambda f: f[-4:] == '.wav')
# remove .wav extension
segments = [[os.path.basename(path).replace('.wav', ''), path, 0, -1, 0]
            for path in segments]
# ====== store everything in one dataset ====== #
if True:
    wave = F.WaveProcesor(segments, output_path=wav_ds, audio_ext='.wav',
                          sr_new=8000, dtype='float16', datatype='memmap',
                          ncache=0.2, ncpu=12)
    wave.run()
    with open(os.path.join(wav_ds, 'README'), 'w') as f:
        f.write(README)
# ===========================================================================
# Acoustic feature extractor
# ===========================================================================
# ====== plot sampling of files ====== #
ds = F.Dataset(wav_ds, read_only=True)
print(ds)
print("Saving processed sample files ...")
samples_file = [i for i in ds['indices'].keys(shuffle=True)[:3]]
for f in samples_file:
    start, end = ds['indices'][f]
    s = ds['raw'][start:end]
    print(" * ", f, s.shape)
    plot_audio(s, sr=8000, win=0.02, shift=0.01, nb_melfilters=40, nb_ceps=12,
        get_qspec=False, get_vad=2, fmin=64, fmax=4000,
        sr_new=None, preemphasis=0.97,
        pitch_threshold=0.8, pitch_fmax=1200,
        vad_smooth=8, vad_minlen=0.01,
        cqt_bins=96, center=False, title=os.path.basename(f))
plot_save('/tmp/tmp.pdf', dpi=180, clear_all=True)
# ====== processing ====== #
if USE_DOWNSAMPLED_DATASET:
    segments = ds
acous = F.SpeechProcessor(segments, outpath, sr=None, sr_new=None,
                win=0.02, hop=0.01, nb_melfilters=40, nb_ceps=12,
                get_spec=True, get_qspec=False,
                get_phase=False, get_pitch=False,
                get_vad=2, get_energy=True, get_delta=2,
                fmin=64, fmax=4000, preemphasis=0.97,
                pitch_threshold=0.8, pitch_fmax=800,
                vad_smooth=12, vad_minlen=0.1,
                cqt_bins=96, pca=True, pca_whiten=False,
                center=True, audio_ext='.wav',
                save_stats=True, substitute_nan=None,
                dtype='float16', datatype='memmap', ncache=0.2, ncpu=12)
acous.run()
with open(os.path.join(outpath, 'README'), 'w') as f:
    f.write(README)
# ====== validate saved dataset ====== #
ds = F.Dataset(outpath, read_only=True)
print(ds.readme)
ds.close()
