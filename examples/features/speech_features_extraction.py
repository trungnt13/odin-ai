# ===========================================================================
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

import numpy as np
import shutil
import os
import sys
from odin import visual
from odin import fuel as F, utils
from collections import defaultdict
from odin.ml import MiniBatchPCA

# ===========================================================================
# set LOG path
# ===========================================================================
LOG_PATH = utils.get_logpath('speech_features_extraction.log',
                             override=True)
utils.stdio(LOG_PATH)
# ===========================================================================
# Const
# ===========================================================================
backend = 'odin'
PCA = True
center = True
pitch_threshold = 0.8
pitch_algo = 'rapt'
datapath = F.load_digit_wav()
output_path = utils.get_datasetpath(name='digit_%s' % backend,
                                    override=True)
# ===========================================================================
# Processor
# ===========================================================================
feat = F.SpeechProcessor(datapath, output_path, audio_ext='wav',
                         sr=None, sr_new=None, sr_info={},
                         win=0.02, hop=0.005, window='hann',
                         nb_melfilters=40, nb_ceps=13,
                         get_delta=2, get_energy=True, get_phase=True,
                         get_spec=True, get_pitch=True, get_f0=True,
                         get_vad=2, get_qspec=True,
                         pitch_threshold=pitch_threshold,
                         pitch_fmax=280,
                         pitch_algo=pitch_algo,
                         cqt_bins=96, vad_smooth=3, vad_minlen=0.1,
                         preemphasis=None,
                         center=center, power=2, log=True,
                         backend=backend,
                         pca=PCA, pca_whiten=False,
                         save_raw=True, save_stats=True, substitute_nan=None,
                         dtype='float32', datatype='memmap',
                         ncache=251, ncpu=10)
with utils.UnitTimer():
    feat.run()
shutil.copy(os.path.join(datapath.path, 'README.md'),
            os.path.join(output_path, 'README.md'))
# ====== check the preprocessed dataset ====== #
ds = F.Dataset(output_path, read_only=True)
print('Output path:', output_path)
# F.validate_features(feat, '/tmp/tmp', override=True)
print(ds)

print("* Configurations:")
for i, j in ds['config'].iteritems():
    print(' ', i, ':', j)

for n in ds.keys():
    if '_pca' in n:
        pca = ds[n]
        if pca.components_ is None:
            print(n, 'components is None !')
        elif np.any(np.isnan(pca.components_)):
            print(n, 'contains NaN !')
        else:
            print(n, ':', ' '.join(['%.2f' % i + '-' + '%.2f' % j
                for i, j in zip(pca.explained_variance_ratio_[:8],
                                pca.explained_variance_[:8])]))
# ====== plot the processed files ====== #
figpath = '/tmp/speech_features_%s.pdf' % backend
files = np.random.choice(ds['indices'].keys(), size=8, replace=False)
for f in files:
    with visual.figure(ncol=1, nrow=5, dpi=180,
                       show=False, tight_layout=True, title=f):
        start, end = ds['indices'][f]
        vad = ds['vad'][start:end]
        pitch = ds['pitch'][start:end].astype('float32')
        energy = ds['energy'][start:end][:, 0].astype('float32')
        spec = ds['spec'][start:end].astype('float32')
        mspec = ds['mspec'][start:end][:, :40].astype('float32')
        mfcc = ds['mfcc'][start:end][:, :13].astype('float32')
        visual.subplot(5, 1, 1)
        visual.plot(energy.ravel())
        visual.subplot(5, 1, 2)
        visual.plot(pitch.ravel())
        visual.subplot(5, 1, 3)
        visual.plot_spectrogram(spec.T, vad=vad)
        visual.subplot(5, 1, 4)
        visual.plot_spectrogram(mspec.T, vad=vad)
        visual.subplot(5, 1, 5)
        visual.plot_spectrogram(mfcc.T, vad=vad)
# ====== check if any pitch or f0 allzeros ====== #
indices = sorted([(name, s, e) for name, (s, e) in ds['indices']],
                 key=lambda x: x[1])
for name, start, end in indices:
    pitch = ds['pitch'][start:end][:]
    f0 = ds['f0'][start:end][:]
    if not np.any(pitch) or not np.any(f0):
        print("Pitch and f0 of name: %s contains only zeros" % name)
# ====== Visual cluster ====== #
if PCA:
    from sklearn.manifold import TSNE
    feat = 'spec'
    X = []; y = []
    feat_pca = ds[feat + '_pca']
    for f, (start, end) in ds['indices']:
        X.append(
            np.mean(
                feat_pca.transform(ds[feat][start:end]), axis=0, keepdims=True
        ))
        y.append(int(f[0]))
    X = np.concatenate(X, axis=0)
    y = np.asarray(y)
    X_ = TSNE(n_components=2).fit_transform(X)
    colors = visual.generate_random_colors(len(set(y)), seed=12082518)
    y = [colors[i] for i in y]
    legend = {c: str(i) for i, c in enumerate(colors)}
    with visual.figure(ncol=1, nrow=5):
        visual.plot_scatter(X[:, 0], X[:, 1], color=y, legend=legend)
    with visual.figure(ncol=1, nrow=5):
        visual.plot_scatter(X_[:, 0], X_[:, 1], color=y, legend=legend)
# ====== save all the figure ====== #
visual.plot_save(figpath, tight_plot=True)
print("Figure saved to:", figpath)
ds.archive()
print("Archive at:", ds.archive_path)
