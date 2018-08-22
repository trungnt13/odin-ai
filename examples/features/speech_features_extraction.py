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
audio = F.DIGITS.load()
all_files = sorted(list(audio['indices'].keys()))
# all_files = all_files[:12] # for testing
output_path = utils.get_datasetpath(name='DIGITS_feats', override=True)
fig_path = utils.get_figpath(name='DIGITS', override=True)
# ===========================================================================
# Extractor
# ===========================================================================
# ====== configuration ====== #
debug = False
padding = False
frame_length = 0.025
step_length = 0.005
dtype = 'float16'
bnf_network = N.models.BNF_2048_MFCC40()
bnf_sad = False
# ====== extractor ====== #
extractors = pp.make_pipeline(steps=[
    pp.speech.AudioReader(sr=8000, remove_dc_n_dither=False, preemphasis=0.97,
                          dataset=audio),
    pp.speech.STFTExtractor(frame_length=frame_length, step_length=step_length,
                            n_fft=512, window='hamm', energy=True),
    pp.speech.PowerSpecExtractor(power=2.0),
    pp.base.RemoveFeatures(feat_name=['stft', 'raw']),
    # ====== spectrum ====== #
    pp.speech.MelsSpecExtractor(n_mels=40, fmin=64, fmax=4000, top_db=80.0),
    pp.speech.MFCCsExtractor(n_ceps=40, output_name='mfcc', remove_first_coef=False),
    pp.speech.Power2Db(input_name='spec', top_db=80.0, output_name='pspec'),
    # ====== sdc ====== #
    pp.speech.MFCCsExtractor(n_ceps=7, output_name='sdc', remove_first_coef=True),
    pp.speech.RASTAfilter(rasta=True, input_name='sdc', output_name='sdc'),
    # ====== pitch ====== #
    # pp.speech.openSMILEpitch(frame_length=0.03, step_length=step_length,
    #                          fmin=32, fmax=620, voicingCutoff_pitch=0.7,
    #                          f0min=64, f0max=420, voicingCutoff_f0=0.55,
    #                          method='shs', f0=True, voiceProb=True, loudness=False),
    # pp.speech.openSMILEloudness(frame_length=0.03, step_length=step_length,
    #                             nmel=40, fmin=20, fmax=None, to_intensity=False),
    # ====== sad ====== #
    pp.speech.SADextractor(nb_mixture=3, nb_train_it=25,
                           feat_name='energy'),
    # ====== BNF ====== #
    # pp.speech.BNFExtractor(input_feat='mfcc', stack_context=10, pre_mvn=True,
    #                        sad_name='sad' if bnf_sad else None, dtype='float32',
    #                        network=bnf_network, batch_size=32),
    # ====== normalization ====== #
    pp.speech.AcousticNorm(mean_var_norm=True, windowed_mean_var_norm=True,
                           use_sad=False, sad_name='sad', ignore_sad_error=True,
                           feat_name=('spec', 'pspec', 'mspec', 'mfcc', 'bnf', 'sdc')),
    # pp.base.RunningStatistics(),
    # ====== post processing ====== #
    pp.base.EqualizeShape0(feat_name=('spec', 'pspec', 'mspec', 'mfcc', 'bnf', 'sdc',
                                      'pitch', 'f0', 'sad', 'energy',
                                      'sap', 'loudness')),
    pp.base.AsType(dtype),
], debug=debug)
# If debug is ran, tensorflow session created,
# multi-processing will be stopped during execution of tensorflow BNF
if debug:
  for i, name in enumerate(all_files[:8]):
    tmp = extractors.transform(name)
    V.plot_multiple_features(tmp, title=name)
  V.plot_save(os.path.join(fig_path, 'debug.pdf'))
  exit()
# ===========================================================================
# Processor
# ===========================================================================
processor = pp.FeatureProcessor(jobs=all_files, path=output_path,
                                extractor=extractors,
                                ncache=0.12, ncpu=None, override=True)
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
# ====== print all indices ====== #
print("All indices:")
for k in ds.keys():
  if 'indices' in k:
    print(' - ', ctext(k, 'yellow'))
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
# ====== check if any pitch or f0 allzeros ====== #
if 'pitch' in ds:
  indices = sorted([(name, s, e) for name, (s, e) in ds['indices']],
                   key=lambda x: x[1])
  for name, start, end in indices:
    pitch = ds['pitch'][start:end][:]
    if not np.any(pitch):
      print("Pitch and f0 of name: %s contains only zeros" % name)
# ====== Visual cluster ====== #
labels = list(set(filter(lambda x: len(x) == 1,
                         [i.split('_')[-1] for i in all_files])))
print("Labels:", ctext(labels, 'cyan'))
for feat in ('bnf', 'mspec', 'spec', 'mfcc'):
  if feat not in ds:
    continue
  from sklearn.manifold import TSNE
  X = []; y = []
  # get right feat and indices
  feat_pca = ds.find_prefix(feat, 'pca')
  indices = ds.find_prefix(feat, 'indices')
  # transform
  prog = Progbar(target=len(indices),
                 print_summary=True, print_report=True,
                 name="PCA transform: %s" % feat)
  for f, (start, end) in indices:
    if len(f.split('_')[-1]) == 1:
      X.append(np.mean(
          feat_pca.transform(ds[feat][start:end]),
          axis=0, keepdims=True))
      y.append(f.split('_')[-1])
    prog.add(1)
  X_pca = np.concatenate(X, axis=0)
  y = np.asarray(y)
  with UnitTimer(name="TSNE: feat='%s' N=%d" % (feat, X_pca.shape[0])):
    X_tsne = TSNE(n_components=2).fit_transform(X_pca)
  colors = V.generate_random_colors(len(labels), seed=12082518)
  # conver y to appropriate color
  y = [colors[labels.index(i)] for i in y]
  legend = {c: str(i) for i, c in enumerate(colors)}
  with V.figure(ncol=1, nrow=5, title='PCA: %s' % feat):
    V.plot_scatter(X_pca[:, 0], X_pca[:, 1], color=y, legend=legend)
  with V.figure(ncol=1, nrow=5, title='TSNE: %s' % feat):
    V.plot_scatter(X_tsne[:, 0], X_tsne[:, 1], color=y, legend=legend)
# ====== save all the figure ====== #
V.plot_save(os.path.join(fig_path, 'pca_tsne.pdf'),
            tight_plot=True)
# ====== print log ====== #
print('Output path:', ctext(output_path, 'cyan'))
print('Figure path:', ctext(fig_path, 'cyan'))
print('Log path:', ctext(log_path, 'cyan'))
