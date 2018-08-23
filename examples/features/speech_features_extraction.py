# ===========================================================================
# Acoustic feature extraction example on TIDIGITS dataset
#
# Without PCA:
#   ncpu=1:  16s
#   ncpu=2:  9.82
#   ncpu=4:  5.9s
#   ncpu=8:  4.3
#   ncpu=12: 4.0
# ===========================================================================
from __future__ import print_function, division, absolute_import
import matplotlib
matplotlib.use('Agg')

import os
os.environ['ODIN'] = 'float32,cpu=1,thread=1,gpu=1'
import sys
import shutil
import pickle

import numpy as np

from odin import visual as V, nnet as N
from odin.utils import ctext, unique_labels, Progbar, UnitTimer
from odin import fuel as F, utils, preprocessing as pp
# ===========================================================================
log_path = utils.get_logpath('speech_features_extraction.log',
                             override=True)
utils.stdio(log_path)
# ===========================================================================
# Dataset
# Saved WAV file format:
#     * [train|test]
#     * [m|w|b|g] (alias for man, women, boy, girl)
#     * [age]
#     * [dialectID]
#     * [speakerID]
#     * [production]
#     * [digit_sequence]
#     => "train_g_08_17_as_a_4291815"
# ===========================================================================
audio = F.TIDIGITS.load()
print(audio)
all_files = sorted(list(audio['indices'].keys()))
fig_path = utils.get_figpath(name='DIGITS', override=True)
# ===========================================================================
# Configuration
# ===========================================================================
debug = True
padding = False
frame_length = 0.025
step_length = 0.005
dtype = 'float16'
bnf_network = N.models.BNF_2048_MFCC40()
bnf_sad = False
# ===========================================================================
# Extractor
# ===========================================================================
extractors = pp.make_pipeline(steps=[
    pp.speech.AudioReader(sr=8000, dataset=audio),
    pp.speech.PreEmphasis(coeff=0.97),
    # ====== STFT ====== #
    pp.speech.STFTExtractor(frame_length=frame_length, step_length=step_length,
                            n_fft=512, window='hamm'),
    pp.base.RenameFeatures(input_name='stft_energy', output_name='energy'),
    pp.speech.PowerSpecExtractor(power=2.0, output_name='spec'),
    pp.base.RemoveFeatures(input_name=['stft', 'raw']),
    # ====== SAD ====== #
    pp.speech.SADextractor(nb_mixture=3, nb_train_it=25,
                           input_name='energy', output_name='sad'),
    # # ====== spectrum ====== #
    pp.speech.MelsSpecExtractor(n_mels=40, fmin=64, fmax=4000, top_db=80.0,
                                input_name=('spec', 'sr'), output_name='mspec'),
    # # ====== sdc ====== #
    pp.speech.MFCCsExtractor(n_ceps=7, remove_first_coef=True,
                             input_name='mspec', output_name='sdc'),
    pp.speech.RASTAfilter(rasta=True, input_name='sdc', output_name='sdc'),
    # # ====== pitch ====== #
    # # pp.speech.openSMILEpitch(frame_length=0.03, step_length=step_length,
    # #                          fmin=32, fmax=620, voicingCutoff_pitch=0.7,
    # #                          f0min=64, f0max=420, voicingCutoff_f0=0.55,
    # #                          method='shs', f0=True, voiceProb=True, loudness=False),
    # # pp.speech.openSMILEloudness(frame_length=0.03, step_length=step_length,
    # #                             nmel=40, fmin=20, fmax=None, to_intensity=False),
    # ====== BNF ====== #
    pp.speech.MFCCsExtractor(n_ceps=40, remove_first_coef=False,
                             input_name='mspec', output_name='mfcc'),
    # # pp.speech.BNFExtractor(input_feat='mfcc', stack_context=10, pre_mvn=True,
    # #                        sad_name='sad' if bnf_sad else None, dtype='float32',
    # #                        network=bnf_network, batch_size=32),
    # # ====== normalization ====== #
    pp.speech.AcousticNorm(mean_var_norm=True, windowed_mean_var_norm=True,
                           sad_name=None, ignore_sad_error=True,
                           input_name=('spec', 'mspec', 'mfcc', 'sdc')),
    pp.base.RunningStatistics(),
    # ====== post processing ====== #
    pp.base.EqualizeShape0(input_name=('spec', 'mspec', 'mfcc', 'sdc',
                                       'energy', 'sad')),
    pp.base.AsType(dtype='float16'),
], debug=debug)
# If debug is ran, tensorflow session created,
# multi-processing will be stopped during execution of tensorflow BNF
if debug:
  for i, name in enumerate(all_files[:8]):
    tmp = extractors.transform(name)
    print('\n')
    V.plot_multiple_features(tmp, title=name)
  V.plot_save(os.path.join(fig_path, 'debug.pdf'))
  exit()
# ===========================================================================
# Processor
# ===========================================================================
output_path = utils.get_datasetpath(name='TIDIGITS_feats', override=True)
processor = pp.FeatureProcessor(jobs=all_files, path=output_path,
                                extractor=extractors,
                                n_cache=0.12, ncpu=None, override=True)
with utils.UnitTimer():
  processor.run()
readme_path = os.path.join(audio.path, [i for i in os.listdir(audio.path)
                                        if 'README' in i][0])
shutil.copy(readme_path,
            os.path.join(output_path, 'README.md'))
pp.calculate_pca(processor, override=True)
# ====== check the preprocessed dataset ====== #
ds = F.Dataset(output_path, read_only=True)
pp.validate_features(ds, path=os.path.join(fig_path, 'validate_features'),
                     nb_samples=8, override=True)
print(ds)
# ====== print pipeline ====== #
print(processor)
# ====== check PCA components ====== #
for n in ds.keys():
  if '_pca' in n:
    pca = ds[n]
    if pca.components_ is None:
      print(ctext(n, 'yellow'), 'components is None !')
    elif np.any(np.isnan(pca.components_)):
      print(ctext(n, 'yellow'), 'contains NaN !')
    else:
      print(ctext(n, 'yellow'),
          ':', ' '.join(['%.2f' % i + '-' + '%.2f' % j
          for i, j in zip(pca.explained_variance_ratio_[:8],
                          pca.explained_variance_[:8])]))
