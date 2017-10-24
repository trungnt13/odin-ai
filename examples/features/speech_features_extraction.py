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
from odin.utils import ctext
from odin import fuel as F, utils, preprocessing as pp
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
PCA = True
center = True
pitch_threshold = 0.8
pitch_algo = 'rapt'
datapath = F.load_digit_wav()
print("Found %d (.wav) files" % len(datapath.keys()))
output_path = utils.get_datasetpath(name='digit')
# ===========================================================================
# Extractor
# ===========================================================================
padding = False
extractors = [
    pp.speech.AudioReader(sr_new=8000, best_resample=True,
                          remove_dc_n_dither=True,
                          preemphasis=0.97),
    pp.NameConverter(converter=lambda x:os.path.basename(x).replace('.wav', '')),
    pp.speech.SpectraExtractor(frame_length=0.025, step_length=0.005,
                               nfft=512, nmels=40, nceps=20,
                               fmin=64, fmax=4000, padding=padding),
    pp.speech.CQTExtractor(frame_length=0.025, step_length=0.005,
                           nbins=96, nmels=40, nceps=20,
                           fmin=64, fmax=4000, padding=padding),
    pp.speech.PitchExtractor(frame_length=0.025, step_length=0.005,
                             threshold=1.0, f0=True, algo='rapt'),
    pp.speech.SADextractor(nb_mixture=3, nb_train_it=25,
                           feat_type='energy'),
    pp.speech.RASTAfilter(rasta=True, sdc=1),
    pp.DeltaExtractor(width=9, order=(1, 2), axis=0,
                      feat_type=('mspec', 'qmspec')),
    pp.speech.AcousticNorm(mean_var_norm=True, window_mean_var_norm=True,
                           feat_type=('mspec', 'mfcc',
                                      'qspec', 'qmfcc', 'qmspec')),
    pp.EqualizeShape0(feat_type=('spec', 'mspec', 'mfcc',
                                 'qspec', 'qmspec', 'qmfcc',
                                 'pitch', 'f0', 'vad', 'energy')),
    pp.RunningStatistics(),
    pp.AsType({'spec': 'float16', 'mspec': 'float16', 'mfcc': 'float16',
               'qspec': 'float16', 'qmspec': 'float16', 'qmfcc': 'float16',
               'pitch': 'float16', 'f0': 'float16',
               'vad': 'float16', 'energy': 'float16',
               'raw': 'float16'})
]

# ===========================================================================
# Processor
# ===========================================================================
jobs = [path for name, path in datapath if '.wav' in path]
processor = pp.FeatureProcessor(jobs, extractors, output_path, pca=True,
                                ncache=260, ncpu=(utils.cpu_count() - 1),
                                override=True)
with utils.UnitTimer():
    processor.run()
shutil.copy(os.path.join(datapath.path, 'README.md'),
            os.path.join(output_path, 'README.md'))
# ====== check the preprocessed dataset ====== #
print('Output path:', output_path)
ds = F.Dataset(output_path, read_only=True)
pp.validate_features(ds, path='/tmp/tmp', nb_samples=6, override=True)
print(ds)
# ====== print pipeline ====== #
padding = '  '
print(ctext("* Pipeline:", 'red'))
for _, extractor in ds['pipeline'].steps:
    for line in str(extractor).split('\n'):
        print(padding, line)
# ====== print config ====== #
print(ctext("* Configurations:", 'red'))
for i, j in ds['config'].iteritems():
    print(padding, i, ':', j)
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
# ====== plot the processed files ====== #
figpath = '/tmp/speech_features.pdf'
files = np.random.choice(ds['indices'].keys(), size = 8, replace = False)
for f in files:
    with visual.figure(ncol = 1, nrow = 5, dpi = 180,
                       show = False, tight_layout = True, title = f):
        start, end = ds['indices'][f]
        vad = ds['vad'][start:end]
        pitch = ds['pitch'][start:end].astype('float32')
        energy = ds['energy'][start:end][:].astype('float32')
        spec = ds['spec'][start:end].astype('float32')
        mspec = ds['mspec'][start:end][:, :40].astype('float32')
        mfcc = ds['mfcc'][start:end][:, :20].astype('float32')
        visual.subplot(5, 1, 1)
        visual.plot(energy.ravel())
        visual.subplot(5, 1, 2)
        visual.plot(pitch.ravel())
        visual.subplot(5, 1, 3)
        visual.plot_spectrogram(spec.T, vad = vad)
        visual.subplot(5, 1, 4)
        visual.plot_spectrogram(mspec.T, vad = vad)
        visual.subplot(5, 1, 5)
        visual.plot_spectrogram(mfcc.T, vad = vad)
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
    feat = 'mspec'
    X = []; y = []
    feat_pca = ds[feat + '_pca']
    for f, (start, end) in ds['indices']:
        X.append(
            np.mean(
                feat_pca.transform(ds[feat][start:end]),
                axis=0, keepdims=True)
        )
        y.append(int(os.path.basename(f)[0]))
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
